"""Single-token decode server: one chained program, KV caches in files.

All 28 layers chain in one listing (x = previous layer's output, raw
embed at L0 -- no interleave break); per-layer K/V caches arrive as
fp16 files, upcast losslessly in-graph (GATHER over arange rows);
knew/vnew merge by SCATTER on the live position; the host relay
only copies files (fp32 out -> fp16
file, standard KV-cache precision, no arithmetic outside the
listings). Prefill = decode steps over prompt tokens from zero
caches. Nothing here needs new ops: composition, not extension.
Usage: python3 scripts/serve_decode.py "prompt here" [--n 8] [--smax 16]
Gates (stated): prefill-vs-recompute parity + decode-vs-recompute text.
Fresh smax-16 workdir measured ~80s (nvcc build + weight prep); re-runs are
seconds on stamp reuse. Needs the snapshot + nvcc + GPU.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID, INTER, NH, NKV, DH = 3584, 18944, 28, 4, 128
PER, KVD = NH // NKV, NKV * DH

# Audit band map (research/attn_topk + flip sizing): sharp L4-11 +
# L21-27 run top-k readout, broad L0-3/L12-20 stay dense. Imported by
# the sparse gate as the shipped configuration.
SPARSE_BANDS_DEFAULT = "4-11,21-27"


def parse_bands(spec, n_layers=N_LAYERS):
    """'4-11,21-27' -> frozenset of layer indices (fail loud on garbage)."""
    out = set()
    spec = (spec or "").strip()
    if not spec:
        return frozenset()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(part)
        if not (0 <= lo <= hi < n_layers):
            raise ValueError(f"bad band {part!r} for {n_layers} layers")
        out.update(range(lo, hi + 1))
    return frozenset(out)


def build_decode(S, sparse_bands=frozenset(), topk=8, resident=False):
    from chain.builder import Prog, qwen_dec_full_layer, \
        qwen_dec_sparse_layer
    sparse = bool(sparse_bands)
    p = Prog("qwen7b-dec-sparse" if sparse else "qwen7b-dec")
    from chain.qwen7b import QWEN_MASK_NEG
    p.config("eps_rms", "1e-6").config("beta", QWEN_MASK_NEG)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "attmask7", "allrows"]
    for L in range(N_LAYERS):
        names += [f"kP{L}", f"vP{L}"]
    for L in range(N_LAYERS):
        for k in ("wqkv", "wo", "wug", "wdown",
                  "ln1", "ln2", "bqkv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog"]
    p.inp(*names)
    x = "x0"
    kouts, vouts = [], []
    for L in range(N_LAYERS):
        if L in sparse_bands:
            x, kf, vf = qwen_dec_sparse_layer(
                p, x, f"kP{L}", f"vP{L}", f"wqkv{L}",
                f"wo{L}", f"wug{L}", f"wdown{L}", f"ln1{L}",
                f"ln2{L}", "pos", "attmask7", "allrows",
                S, pre=str(L), bqkv=f"bqkv{L}",
                topk=topk, resident=resident)
        else:
            x, kf, vf = qwen_dec_full_layer(
                p, x, f"kP{L}", f"vP{L}", f"wqkv{L}",
                f"wo{L}", f"wug{L}", f"wdown{L}", f"ln1{L}",
                f"ln2{L}", "pos", "attmask7", "allrows",
                S, pre=str(L), bqkv=f"bqkv{L}",
                resident=resident)
        kouts.append(kf)
        vouts.append(vf)
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p, kouts, vouts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The", "capital", "of"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--smax", type=int, default=16)
    ap.add_argument("--graph", dest="graph", action="store_true",
                    default=True)
    ap.add_argument("--no-graph", dest="graph", action="store_false",
                    help="opt out of CUDA-graph replay (debug)")
    ap.add_argument("--no-sync", dest="no_sync", action="store_true",
                    default=True)
    ap.add_argument("--sync", dest="no_sync", action="store_false",
                    help="opt into per-op sync (debug; costs ~2x)")
    ap.add_argument("--sparse-bands", default="",
                    help="layers with top-k readout, e.g. '4-11,21-27' "
                    "(audit map in SPARSE_BANDS_DEFAULT: sharp bands "
                    "sparse, broad middle dense; empty = fully dense)")
    ap.add_argument("--sparse-k", type=int, default=8,
                    help="top-k width for sparse bands (build-time, "
                    "graph-safe; sized by flip-rate)")
    ap.add_argument("--resident-cache", action="store_true",
                    help="keep KV caches on-device (SCATTER in place, no "
                    "host relay, no per-step cache files)")
    ap.add_argument("--prefill-dir", default=None,
                    help="skip sequential prefill: copy ckP/cvP caches from "
                    "DIR (serve_prefill.py output) and generate directly")
    ap.add_argument("--topk", type=int, default=1,
                    help="sample from top-k (1 = greedy; host boundary, "
                    "listings stay deterministic)")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature for sampling (1.0 = none)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed (replay-identical across runs)")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    S = args.smax
    from chain import mem as _mem
    from chain.qwen7b import (load7b, snapshot_ok, prep_7b_weight,
                              clear_7b_cache, input_sources, QWEN7B_VOCAB)
    from chain.emit_c import compile_program, FRef, IRef
    from chain.emit_cuda import build_cu
    from chain.serve import ServedExe
    print(_mem.report(S), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs snapshot)")
    _mem.guard(_mem.estimate_7b(S)["stream_peak_cpu"], "decode build")
    g, tok = load7b()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    if len(ids) + args.n > S:
        sys.exit(f"prompt+gen too long ({len(ids)}+{args.n} > {S})")

    keymap = dict(input_sources())
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    work = os.environ.get("QWEN_WORKDIR", "/tmp/dec7b")
    os.makedirs(work, exist_ok=True)

    def prep_one(dotted, short):
        # input_sources keys ARE listing names ("wq3" -> ("wq", dotted))
        a = prep_7b_weight(g, dotted, short, S)
        if short in ("bq", "bk", "bv"):
            a = np.ascontiguousarray(a[:1])  # decode rows are (1,*)
        return a

    def stream_bins(art, extra_frozen, live):
        # Fused layouts (same values, concatenated data-prep): wqkv =
        # [wq|wk|wv] (H,H+2KV, wq keeps its 1/sqrt(dh) fold), bqkv the
        # tiled biases, wug = [wup|wgate] (H,2I).
        def prep_fused(name):
            if name.startswith("wqkv"):
                L = name[4:]
                parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                         for s in ("wq", "wk", "wv")]
                return np.ascontiguousarray(
                    np.concatenate(parts, axis=1))
            if name.startswith("bqkv"):
                L = name[4:]
                parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                         for s in ("bq", "bk", "bv")]
                return np.ascontiguousarray(
                    np.concatenate(parts, axis=1))
            if name.startswith("wug"):
                L = name[3:]
                parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                         for s in ("wup", "wgate")]
                return np.ascontiguousarray(
                    np.concatenate(parts, axis=1))
            return None
        for n in art["inputs"]:
            if n in live:
                continue  # written per step
            fn = os.path.join(work, f"d_{n}.bin")
            if os.path.isfile(fn):
                continue
            if n in extra_frozen:
                extra_frozen[n].tofile(fn)
                continue
            fused = prep_fused(n)
            if fused is not None:
                fused.tofile(fn)
                continue
            if n in keymap:
                short, src = keymap[n]
                prep_one(src, short).tofile(fn)
            else:
                raise KeyError(f"no frozen source for {n!r}")

    from chain.emit_c import compile_program as _cp
    from chain.emit_cuda import build_cu as _bc
    from chain.serve import ServedExe
    bands = parse_bands(args.sparse_bands)
    resident = bool(args.resident_cache)
    prog, kouts, vouts = build_decode(S, sparse_bands=bands,
                                      topk=args.sparse_k, resident=resident)
    text = prog.text()
    print(f"decode program: "
          f"{sum(1 for ln in text.splitlines() if '=' in ln and not ln.strip().startswith(('IN', 'CONFIG')))} ops",
          flush=True)
    sample = {"x0": FRef((1, HID), "float16"),
              "pos": np.zeros(1, dtype=np.int64),
              "attmask7": IRef((PER, S), "int64"),
              "allrows": np.arange(S, dtype=np.int64)}
    for L in range(N_LAYERS):
        sample[f"kP{L}"] = FRef((S, KVD), "float16")
        sample[f"vP{L}"] = FRef((S, KVD), "float16")
    wshapes = {"wqkv": (HID, HID + 2 * KVD), "wo": (HID, HID),
               "wug": (HID, 2 * INTER),
               "wdown": (INTER, HID), "ln1": (HID,), "ln2": (HID,),
               "bqkv": (1, HID + 2 * KVD)}
    for L in range(N_LAYERS):
        for k, sh in wshapes.items():
            sample[f"{k}{L}"] = FRef(sh, "float16")
    sample["lnf"] = FRef((HID,), "float16")
    sample["wlog"] = FRef((HID, QWEN7B_VOCAB), "float16")
    caches = [f"kP{L}" for L in range(N_LAYERS)] + \
             [f"vP{L}" for L in range(N_LAYERS)]
    outs = ["LOGITS", "OUT"] if resident else ["LOGITS", "OUT"] + kouts + vouts
    live = ["x0", "pos", "attmask7"] if resident else (
        ["x0", "pos", "attmask7"] + caches)
    t0 = time.perf_counter()
    art = _cp(text, "cuda", sample=sample, outputs=outs,
              basedir=os.path.join(ROOT, "programs"), use_fp16=True,
              live=live, sync_each=not args.no_sync, graph=args.graph,
              time_ops=os.environ.get("QWEN_TIME_OPS") == "1")
    print(f"compile graph: {time.perf_counter() - t0:.0f}s", flush=True)
    stream_bins(art, {"allrows": np.arange(S, dtype=np.int64)},
                set(live) | set(caches))
    exe = os.path.join(work, "qdec")
    import json as _js
    stamp = {"graph": bool(args.graph), "no_sync": bool(args.no_sync),
             "smax": S, "sparse_bands": args.sparse_bands or "",
             "sparse_k": int(args.sparse_k), "resident": resident,
             "fused_qkv": True}
    if os.environ.get("QWEN_REUSE_BIN") != "1":
        t0 = time.perf_counter()
        exe = _bc(art["source"], work, name="qdec",
            units=art.get("units"))
        print(f"nvcc: {time.perf_counter() - t0:.0f}s", flush=True)
        _js.dump(stamp, open(os.path.join(work, "build.json"), "w"))
    else:
        try:
            old = _js.load(open(os.path.join(work, "build.json")))
        except OSError:
            old = {}
        bad = [k for k in stamp if old.get(k) != stamp[k]]
        if bad:
            sys.exit(f"refusing QWEN_REUSE_BIN=1: stale decode binary "
                     f"differs on {bad} (rebuild without the env key)")
        print("nvcc: reused binary (stamp verified)", flush=True)
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()

    def argv_for(art, live_map, out_files):
        argv = []
        for name in art["inputs"]:
            argv.append(live_map.get(
                name, os.path.join(work, "d_" + name + ".bin")))
        return argv + out_files

    def argv_for(art, live_map, out_files):
        argv = []
        for name in art["inputs"]:
            argv.append(live_map.get(
                name, os.path.join(work, "d_" + name + ".bin")))
        return argv + out_files

    lives = {"x0": os.path.join(work, "step_x0.bin"),
             "pos": os.path.join(work, "step_pos.bin"),
             "attmask7": os.path.join(work, "step_am.bin")}
    for L in range(N_LAYERS):
        lives[f"kP{L}"] = os.path.join(work, f"ckP{L}.bin")
        lives[f"vP{L}"] = os.path.join(work, f"cvP{L}.bin")
    flo = os.path.join(work, "step_lg.bin")
    fou = os.path.join(work, "step_ou.bin")
    kouts_files, vouts_files = [], []
    if not resident:
        for L in range(N_LAYERS):
            kouts_files.append(os.path.join(work, f"step_K{L}.bin"))
            vouts_files.append(os.path.join(work, f"step_V{L}.bin"))
    for L in range(N_LAYERS):
        for tag in ("kP", "vP"):
            fn = lives[f"{tag}{L}"]
            if not os.path.isfile(fn):
                np.zeros((S, KVD), dtype=np.float16).tofile(fn)
    np.zeros((1, HID), dtype=np.float16).tofile(lives["x0"])
    np.zeros(1, dtype=np.int64).tofile(lives["pos"])
    np.zeros((PER, S), dtype=np.int64).tofile(lives["attmask7"])
    sv = ServedExe(exe, argv_for(art, lives,
                                 [flo, fou] + kouts_files + vouts_files),
                   err_path=os.path.join(work, "dec.err"))

    def step_row(tok_id, n, need_lg=True):
        xx = np.zeros((1, HID), dtype=np.float16)
        xx[0] = E[np.array([tok_id])].astype(np.float16)
        xx.tofile(lives["x0"])
        # single-row ROTARY reads pos[0] only (k_rotary_f32: ang =
        # pos[r]); the row in flight is position n, so write [n].
        np.array([n], dtype=np.int64).tofile(lives["pos"])
        am = np.zeros((PER, S), dtype=np.int64)
        am[:, :n + 1] = 1
        am.tofile(lives["attmask7"])
        t0 = time.perf_counter()
        sv.step()
        dt = time.perf_counter() - t0
        ou = int(np.fromfile(fou, dtype=np.int64)[0])
        # Greedy decode takes the on-device ARGMAX (ou); the 152k
        # logits file is only read when sampling needs it (or for the
        # prefill parity gate). Ties-first on both sides (stated).
        lg = np.fromfile(flo, dtype=np.float32).reshape(1, -1)[0] \
            if need_lg else None
        if not resident:
            for L in range(N_LAYERS):
                # Relay: fp32 compute domain -> fp16 files (standard
                # KV-cache precision, one rounding per step, no accumulation
                # beyond the cache itself). Resident mode keeps caches
                # on-device (SCATTER in place: no relay, no step files).
                for a, b in ((kouts_files[L], lives[f"kP{L}"]),
                             (vouts_files[L], lives[f"vP{L}"])):
                    np.fromfile(a, dtype=np.float32).astype(
                        np.float16).tofile(b)
        return lg, ou, dt * 1000

    try:
        sv.start()
        rng = np.random.default_rng(args.seed)

        def pick(lg):
            # Host-boundary sampler (listings stay greedy-deterministic):
            # temperature + top-k + seeded draw. topk=1 == argmax.
            if args.topk <= 1 and args.temp == 1.0:
                return int(np.argmax(lg))
            sc = lg / max(args.temp, 1e-6)
            k = min(args.topk, sc.size)
            idx = np.argpartition(sc, -k)[-k:]
            sub = sc[idx] - sc[idx].max()
            ex = np.exp(sub.astype(np.float64))
            pr = ex / ex.sum()
            return int(idx[rng.choice(k, p=pr)])

        t0 = time.perf_counter()
        seq = list(ids)
        last = None
        if args.prefill_dir is None:
            for i, t in enumerate(seq):
                last, _, _ = step_row(t, i)
            prefill_s = time.perf_counter() - t0
            print(f"prefill {len(seq)} toks in {prefill_s:.1f}s", flush=True)
        else:
            import shutil as _sh
            for L in range(N_LAYERS):
                for tag in ("ckP", "cvP"):
                    _sh.copyfile(
                        os.path.join(args.prefill_dir, f"{tag}{L}.bin"),
                        lives[f"{'kP' if tag == 'ckP' else 'vP'}{L}"])
            prefill_s = time.perf_counter() - t0
            print(f"prefill: parallel caches from {args.prefill_dir} "
                  f"({prefill_s * 1000:.0f}ms relay)", flush=True)
            last = np.load(os.path.join(args.prefill_dir,
                                        "prefill_logits.npy"))
        np.save("/tmp/dec_prefill.npy", np.ascontiguousarray(last))
        t0 = time.perf_counter()
        ms = []
        greedy = args.topk <= 1 and args.temp == 1.0
        for _ in range(args.n):
            if len(seq) >= S:
                break
            lg, ou_, m = step_row(seq[-1], len(seq), need_lg=not greedy)
            seq.append(ou_ if greedy else pick(lg))
            ms.append(m)
        print(f"decode: {sum(ms) / max(1, len(ms)):.0f}ms/tok "
              f"(min {min(ms) if ms else 0:.0f})", flush=True)
        print("DECODED:", tok.decode(seq, skip_special_tokens=True), flush=True)
    finally:
        sv.close()


if __name__ == "__main__":
    main()
