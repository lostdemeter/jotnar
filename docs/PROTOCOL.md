# Jotnar wire protocol, v1 (Milestones 1-2: loopback + TCP transport)

Controller/node split for geometric assembly programs. The controller
is smart (builds listings, holds weights, frames requests); the node
is dumb and **model-blind** (holds programs + named blobs, executes,
returns bytes). The emitter never knows what model it lowers, and the
node never knows either: a listing is text, weights are bytes.

Out of scope: multi-node/pipeline serving, node types (later
milestones). No TLS/auth on the wire (plain TCP, stated plainly).

## Framing (loopback and TCP speak identical bytes)

Every message on the wire is one frame:

```
u32 BE length N || N bytes: {header JSON, "\n", payload bytes}
```

The header carries `{"v":1,"id":n,"type":T,...}`; the length prefix
covers header + payload together. Loopback and TCP reuse
`protocol/loopback.py:dumps/loads` byte-for-byte — no second framing.
Malformed frames (short length, bad JSON, length mismatch) are
rejected before dispatch, loudly.

## Transport (Milestone 2)

One TCP connection = a stream of frames, responses in request order.
Policy: **one client at a time**, single-threaded server; a second
connection while one is active gets `{ok:false, code:"busy"}` and a
close — loud, never queued. Every socket carries timeouts on both
ends (connect + recv, default 600s, matching serve-step budgets); a
dropped peer raises instead of hanging. Clean EOF ends the session;
the next connection starts fresh (program registry persists).

## Messages

All requests carry `id`; every response echoes it with `ok:true`
plus result fields, or `ok:false` with `error:{code,msg}`. Error
codes are stable strings (see § Errors).

### PING -> PONG

`{type:"ping"}` -> `{ok:true, pong:{impl:"jotnar-node", v:1,
backends:["lattice","served"], mnemonics:59}}`. Liveness only;
`mnemonics` lets the controller refuse early instead of failing
mid-program. (Mirrors phi_compute_protocol PING; the capability
tail is the deviation — justified: our node admits per-kind
overloads per backend, so the count alone is not actionable.)

### PROGRAM -> PROGRAM-OK

`{type:"program", name, listing, backend, config:{}, stamp?{}}`
registers a program under a client-chosen `name`. **`listing` is the
.asm text itself, utf-8 — there is no separate bytecode.** `backend`
is `"lattice"` (in-process interpreter) or `"served"` (persistent
binary the controller built out-of-band; `config` carries
`{exe, argv_files, live, outputs, err_path}`). Response:
`{ok:true, handle:name}`.

Build provenance (Milestone 2): stamp discipline used to be
convention (the server trusted the client's binary). For
`backend:"served"`, PROGRAM now carries `stamp:{graph, no_sync,
smax, sparse_bands, sparse_k, resident, fused_qkv}` and the server
compares it against the binary's `build.json` sitting next to the
exe. Missing stamp, missing build.json, or any differing key is
refused loud as `stamp-mismatch`. Lattice programs carry no stamp.

No BUILD message exists, by decision: serving a build over the
wire would mean streaming ~14GB of weights plus a 10-minute
compile — a build service, not a message. The controller builds
out-of-band (emitter next to builder); the node verifies
provenance at PROGRAM time. Deferred, with that reason.

Deviation note: phi_compute_protocol's PROGRAM is acknowledged
here as name-registration rather than upload-and-compile, because
compile is controller-side work in Jotnar (the emitter runs next
to the builder, and binaries are workdir artifacts with stamps).
The node never compiles in v1.

### STORE / DROP / LIST (blob shelf)

- `{type:"store", name, stream:{kind, dtype, shape}, payload:<bytes>}`
  puts an opaque named blob on the shelf. Weights are bytes to the
  node; kind/dtype/shape are recorded for EXEC arity checks only.
- `{type:"drop", name}` removes it. Dropping an EXEC-referenced
  blob fails loud; dropping a missing name is OK (idempotent).
- `{type:"list"}` -> `{programs:[...], stores:[...]}`.

(Mirrors phi_compute_protocol STORE/DROP/LIST one-for-one; no
deviation. Names are client-chosen; collisions overwrite loudly
— the response echoes `replaced:true` when that happens.)

### EXEC -> EXEC-OK

`{type:"exec", handle, inputs:{stream: {store:name} | {stream, off,
len}}, outputs:[stream...]}` runs the program. Inline inputs ride as
segments of the frame payload (`off`/`len` in the binding, v1, no
alignment).
`inputs` bind by **stream name** (typed streams are the addressing;
deviation from positional binding, justified: listings address
streams by name in `IN` decls, so names are the only binding that
cannot desynchronize). Response: `{ok:true,
outputs:{stream:{stream, payload}}}` with payloads in § Encodings.

The node executes through existing chain execution only
(lattice `run_text` or a `ServedExe` step). For `backend:"served"`,
`config` carries `{exe, argv_files, live:{stream:argv-idx},
outputs:{stream:{idx,dtype,shape}}, err_path}`; inputs are written
as raw file bytes, outputs read back the same way (dtypes cross the
wire exactly: f16/f32/i64). EXEC is deterministic:
same handle + same input bytes = byte-identical output payloads.

## Stream encodings (§ Encodings)

`stream:{kind, dtype, shape}` where kind ∈ {F,T,I}:

- F float: `dtype` f32 or f64, little-endian raw, row-major `shape`.
- T lattice-triple: three int64 LE arrays (s, e, z per
  phi-core `lattice.encode`), concatenated s||e||z, plus `shape`.
- I integer: int64 LE, row-major `shape` (ids, masks, counts).

The header always states kind/dtype/shape; the node checks arity
and shape against the listing signature before running and refuses
loud on mismatch (never coerces).

## Errors (fail loud, our way)

- `malformed`: frame/JSON/payload-length violations. Rejected
  before dispatch; connection state unchanged.
- `unknown-type`: type not in the catalog.
- `npattern`: listing uses an opcode (or kind) the node's backend
  refuses — the same `NoPattern` the local interpreter raises,
  across the wire. Checked at PROGRAM time (fail fast) and
  re-checked at EXEC time (fail safe).
- `unknown-handle` / `unknown-store`: EXEC references nothing
  registered. `arity`: input/output names or shapes disagree
  with the listing.
- `stamp-mismatch` (served PROGRAM only): missing stamp, missing
  `build.json` next to the exe, or any differing stamp key.
- `busy`: second TCP connection while one is active (refused + closed).
- `backend-unavailable`: PROGRAM names a backend the node does not
  offer (advertised in PING; controller routes first, node refuses
  second).
- Server bugs (exceptions) return `ok:false, error:{code:"node",
  msg}` with the message — never silent wrong bytes, never a
  bare hang. Timeouts belong to the transport (Milestone 2).

## Multi-node pipeline + backends (Milestone 3)

Each node gets its own workdir + port (no sharing: caches are sticky
per-node state). The controller holds no KV — only the H boundary
activation crosses between stages, as f32 bytes (never f16: cache
rounding would break bit-exactness vs single-node). Stronger: stage
binaries build in float mode (`use_fp16=False`), because the emitter
declares every F input `__half` in fp16 mode regardless of sample
dtype — an f32 boundary file would be misread. f16 weights cast up
exactly, so values are unchanged and pipeline-vs-single stays
bit-exact (cross-mode vs production is eps-graded). Cost: 2x weight
traffic. Honored emitter constraint: build flag, not emitter change.

### Backend advertisement + dispatch

PING reports the backends a node actually offers (configured at
serve time; default both). Dispatch rule, controller-side: served
programs route to a node advertising `"served"`, lattice programs
to one advertising `"lattice"`; no advertisement, no route (loud
`RouteError`, never misrouted). Nodes also refuse unadvertised
backends themselves (`backend-unavailable`). Every routing decision
is logged controller-side so placement is provable after the fact.

### Per-node state

KV caches are sticky files inside each stage's workdir (same relay
semantics as single-node, per layer subset). Boundaries are
stateless: H bytes in, H bytes out. Restarting a node loses nothing
but its caches (re-prefill recovers, exactly like single-node).

### Pipeline errors

A dead stage fails loud mid-EXEC: socket timeout / refused /
reset surfaces as a transport error at the controller, no partial
text is ever returned (EXEC is all-or-nothing per stage; the
controller assembles turns only from ok responses). New code:
`backend-unavailable` (above). Unchanged: `busy` (one client per
server still stands — fan-out lives in the controller).

## Remote nodes (Milestone 4: two-machine serving)

The node is the calculator, the controller is the brain. A remote
node holds zero model knowledge — no snapshot, no tokenizer, no
architecture, no GPU. It receives opaque bytes, executes opcodes on
typed streams, returns bytes. Structure lives controller-side as
programs; geometry lives node-side as blobs. Proven: a CPU-only
lattice node on a second machine serves EXECs bit-exact vs local
`run_text`, placed by the unmodified PipeController (TcpClient
already takes a host — the node runs unchanged).

### Out-of-band node setup

No BUILD message exists (M2 decision), so setup is out-of-band,
once per machine — `protocol/serve_tcp.py --backends lattice` plus
the smallest footprint that runs: `protocol/` + whatever `chain`
imports the node server needs + numpy (160KB tarball + venv; no
snapshot, no toolchain, no torch, ever):

```
python3 -m venv ~/jotnar-node && ~/jotnar-node/bin/pip install numpy
# copy protocol/ chain/ phi_core/ into ~/jotnar-m4 (no data/, no weights)
cd ~/jotnar-m4 && ~/jotnar-node/bin/python protocol/serve_tcp.py \
  --host 0.0.0.0 --port 41201 --backends lattice
```

Controller side: `JOTNAR_REMOTE_HOST=<host> python3
tests/test_remote.py` (host from env, never hardcoded; default
`gimli`). Port 41201 is convention, not protocol.

### Timeout guidance

The 600s socket defaults stand on LAN unchanged — measured, not
tuned: lattice EXEC round-trips complete in seconds; the only
failures seen were refusal/timeout-class (firewall, dead port),
which the defaults surface loudly. Change them only with a
measured reason (slower links, bigger programs).

### Transports that cross a firewall

Direct LAN TCP is preferred. If the node's firewall blocks inbound
(the proof ran into exactly this: gimli answers ping+ssh, refuses
other inbound ports, no root to change it), an SSH tunnel carries
identical frames with no code change:

```
ssh -fN -L 41201:127.0.0.1:41201 <node>
JOTNAR_REMOTE_HOST=127.0.0.1 JOTNAR_REMOTE_SSH=<alias> \
  python3 tests/test_remote.py
```

Frames are byte-identical either way (loopback codec, M2); the
tunnel changes latency only. No TLS/auth on the wire (deferred,
stated plainly since M2).

## Remote work (Milestone 5: genuine programs on CPU nodes)

M4 proved remote *execution* (microsecond chain, 20ms round-trips).
M5 proves remote *work*: the flagship image pipeline served on a
CPU-only lattice node, full-program output bytes bit-identical vs
local `run_text` every iteration, gimli-side CPU jiffies for its
server PID across the run. Measured: 10/10 identical, +555 jiffies
(5.5 CPU-s on gimli) over 5.6s wall (563ms/iter, of which 555ms is
server CPU) — the needle moves because the program is real.

### Remote-work shape

Same messages, bigger payloads: PROGRAM once, then one EXEC per
frame (304KB U8 in, 304KB U8 out at 288x352). Integer paths are
bit-exact across machines by contract (same interpreter, same
fixed-point math, no floats anywhere in the path) — grade them so:
byte-compare, never eps. The gate (`tests/test_m5work.py`) loops 10
EXECs and asserts every output identical, plus a jiffies-moved
bar (>1 CPU-s) so a silently-local run cannot pass.

### Wire note: U8 streams

Kind I carries `dtype:i64` (ids, masks, counts) and `dtype:u8`
(raw pixel/byte planes, shape-preserving). U8 decodes to uint8, not
int64 — decoding pixels as int64 is an 8x misread (caught live:
38016 elements vs 288x352x3). Same-codec rule still holds (M2).

### What stays CPU-only by hardware

Lattice nodes serve integer listings only (`served`/CUDA stays
local-only: no GPU, no snapshot, no torch on the node, ever). GPU
decode stages, prefill binaries, and anything touching weights
beyond opaque blobs are not remote-work candidates — if a workload
tempts toward either, that is a scope violation, not an extension.

## Concurrent lanes (Milestone 6: saturate the node)

M5 proved one lane. M6 proves N concurrent EXEC lanes saturating N
node cores with per-lane bit-exactness intact throughout. Shape: N
node processes on distinct ports (the busy policy is per-server, so
multi-process is the legal fan-out; no protocol change), one
threaded controller driver, one TcpClient per lane-thread (no socket
sharing). Throughput is the target; per-turn latency does not
improve and nothing is claimed about it.

### Addressing choice

N program names (`flag0`..`flag7`), each registered on exactly one
node; lane threads address nodes by program name through the
unchanged `PipeController.exec`. Justification: programs are cheap
registrations, the busy policy already scopes concurrency
per-server, and name→node mapping IS the addressing the controller
already logs in placement — a pool/round-robin extension would add
routing state for zero additional provability. Minimal, and the
placement log proves every byte went where claimed.

### Exactness-under-concurrency contract

Every lane bit-identical vs local `run_text`, every iteration —
graded so in the gate (75/75 EXECs across the 1/2/4/8 curve). No
eps anywhere: integer paths, same interpreter, byte-compare.

### Measured scaling (gimli i7-6700 4C/8T, flagship lattice, 5 iters)

```
lanes  wall   execs  serverCPU  ms/iter  eff-cores  exec/s
1      2.9s   5      2.9s       570      0.99       1.7
2      3.9s   10     7.8s       782      1.98       2.5
4      9.1s   20     36.1s      1805     3.97       2.2
8      25.3s  40     195.0s     4874     7.71       1.6
```

All 8 threads pegged (7.71 effective cores) — saturation proven.
But per-turn cost inflates 8.5x while total throughput peaks at 2
lanes and 8-lane throughput (1.6/s) ≈ 1-lane (1.7/s). Bend causes,
each measured: NOT BLAS threading (OMP=1 rerun byte-identical
numbers), NOT HT pairing (taskset 1-thread-per-physical-core rerun
identical), NOT tunnel/GIL (per-PID CPU is transport-independent;
separate processes). What remains: shared-subsystem contention —
the local many-core control shows 2.8x per-iter inflation at 8
threads with no core shortage, and 4 pinned distinct physical cores
still cost 3.2x (L3/DRAM bandwidth on this box). Contention is
workload × box, not protocol.

### Network split (2.5GbE)

608KB ferry per EXEC (304KB U8 each way) ≈ 2ms on the wire vs
~570ms+ compute — per-lane network share is noise (<1%); 8-lane
aggregate still <2% of wall. Offload is placement, not latency.

### Lane setup (out-of-band, as M4)

One lattice node per port (41211-41218), `OMP_NUM_THREADS=1`
harmless-but-irrelevant (see above), 1:1 CPU pinning optional
(measured neutral); tunnel one `-L` per port or direct LAN where
the firewall allows. Gimli stays GPU-less/snapshot-free — a lane
design needing either is a scope violation.

## Worked minimum

MATMUL+ADD+SILU chain on lattice: PROGRAM the 3-op listing, STORE
nothing (inputs inline), EXEC, compare payload bytes to local
`run_text` — bit-exact. Full-decode Paris demo goes through the
same messages with `backend:"served"` (binary built out-of-band,
referenced by config; every byte still framed).
