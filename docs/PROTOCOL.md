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
backends:["lattice","served"], mnemonics:54}}`. Liveness only;
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
- Server bugs (exceptions) return `ok:false, error:{code:"node",
  msg}` with the message — never silent wrong bytes, never a
  bare hang. Timeouts belong to the transport (Milestone 2).

## Worked minimum

MATMUL+ADD+SILU chain on lattice: PROGRAM the 3-op listing, STORE
nothing (inputs inline), EXEC, compare payload bytes to local
`run_text` — bit-exact. Full-decode Paris demo goes through the
same messages with `backend:"served"` (binary built out-of-band,
referenced by config; every byte still framed).
