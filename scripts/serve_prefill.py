"""Parallel prefill: one full forward that also emits KV caches.

Builds the bmmv recompute program with cache_outs (per-layer rotated-K
groups + V groups as extra outputs), runs it one-shot over the prompt,
and assembles (S,KVD) fp16 cache files for the decode server. Same
math as sequential decode-prefill in a different order (full-S vs
1-row matmuls), so agreement is eps (tight bar), not bit-exact.
Usage: python3 scripts/serve_prefill.py "prompt here" [--smax 16]
       [--workdir /tmp/gen7b] [--outdir /tmp/prefill]
Prints prefill wall time; writes OUTDIR/ckP{L}.bin + cvP{L}.bin.
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID, NH, NKV, DH = 3584, 28, 4, 128
PER = NH // NKV


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The", "capital", "of"])
    ap.add_argument("--smax", type=int, default=16)
    ap.add_argument("--workdir", default="/tmp/gen7b")
    ap.add_argument("--outdir", default="/tmp/prefill")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    S = args.smax
    from chain.builder import Prog, qwen_bmmv_layer
    from chain.qwen7b import load7b, snapshot_ok
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    from chain import mem as _mem
    print(_mem.report(S), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs snapshot)")
    g, tok = load7b()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)
    if len(ids) >= S:
        sys.exit(f"prompt too long ({len(ids)} >= {S})")
    try:
        stamp = __import__("json").load(
            open(os.path.join(args.workdir, "build.json")))
    except OSError:
        stamp = {}
    if stamp.get("smax", S) != S:
        sys.exit(f"workdir {args.workdir} built for smax {stamp.get('smax')} "
                 f"!= {S} (rebuild it first)")
    p = Prog("qwen7b-prefill")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "cmask", "cmask7"]
    for L in range(N_LAYERS):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog"]
    p.inp(*names)
    x = "x0"
    kouts, vouts = [], []
    for L in range(N_LAYERS):
        y, krgs, vgs = qwen_bmmv_layer(
            p, x, f"wq{L}", f"wk{L}", f"wv{L}", f"wo{L}", f"wup{L}",
            f"wgate{L}", f"wdown{L}", f"ln1{L}", f"ln2{L}", "pos",
            "cmask", "cmask7", S, pre=str(L), bq=f"bq{L}", bk=f"bk{L}",
            bv=f"bv{L}", cache_outs=True)
        x = y
        kouts.extend([(L, g, s) for g, s in enumerate(krgs)])
        vouts.extend([(L, g, s) for g, s in enumerate(vgs)])
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    E = g("model.embed_tokens.weight")
    x0 = np.zeros((S, HID), dtype=np.float64)
    x0[:len(ids)] = E[np.array(ids)]
    sample = {"x0": x0, "pos": np.arange(S, dtype=np.int64),
              "cmask": np.tril(np.ones((S, S), dtype=np.int64))}
    cm1 = np.tril(np.ones((S, S), dtype=np.int64))
    sample["cmask7"] = np.tile(cm1, (PER, 1))
    from chain.qwen7b import input_sources, prep_7b_weight, clear_7b_cache
    keymap = dict(input_sources())
    for L in range(N_LAYERS):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            w = prep_7b_weight(g, keymap[f"{k}{L}"][1], k, S)
            sample[f"{k}{L}"] = w
            del w
    for k in ("lnf", "wlog"):
        sample[k] = prep_7b_weight(g, keymap[k][1], k, S)
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    outs = ["LOGITS", "OUT"] + [s for _, _, s in kouts] \
        + [s for _, _, s in vouts]
    # Serve variant (warm): prefill runs as steps of a persistent
    # binary (cold one-shot pays ~19s migration; warm steps ~100ms).
    live = ["x0", "pos", "cmask", "cmask7"]
    t0 = time.perf_counter()
    art = compile_program(p.text(), "cuda", sample=sample, outputs=outs,
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=live, sync_each=True,
                          graph=True)
    print(f"compile graph: {time.perf_counter() - t0:.0f}s", flush=True)
    work = "/tmp/prefill_build"
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.perf_counter()
    exe = os.path.join(work, "qpre")
    if os.environ.get("QWEN_REUSE_BIN") != "1":
        exe = build_cu(art["source"], work, name="qpre")
        print(f"nvcc: {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        print("nvcc: reused binary", flush=True)
    argv = [exe]
    for name in art["inputs"]:
        k, _, _ = art["streams"][name]
        fn = os.path.join(work, f"p_{name}.bin")
        src = os.path.join(args.workdir, f"fr_{name}.bin")
        if name not in sample:
            raise KeyError(f"no sample for {name!r}")
        if k == "F" and os.path.isfile(src) and name not in (
                "x0", "pos", "cmask", "cmask7"):
            # identical prep as the gen build: link, don't rewrite 15GB
            if os.path.isfile(fn) or os.path.islink(fn):
                os.remove(fn)
            os.symlink(src, fn)
        elif k == "F":
            sample[name].astype(np.float16).tofile(fn)
        else:
            sample[name].astype(np.int64).tofile(fn)
        if name in live:
            argv.append(os.path.join(work, f"step_{name}.bin"))
        else:
            argv.append(fn)
    # live step files must exist for the resident load (overwritten below)
    for name in live:
        k, _, _ = art["streams"][name]
        fn = os.path.join(work, f"step_{name}.bin")
        if k == "F":
            sample[name].astype(np.float16).tofile(fn)
        else:
            sample[name].astype(np.int64).tofile(fn)
    outmap = {}
    for o in outs:
        fn = os.path.join(work, f"pout_{o}.bin")
        outmap[o] = fn
        argv.append(fn)
    from chain.serve import ServedExe
    sv = ServedExe(exe, argv[1:], err_path=os.path.join(work, "pre.err"))
    try:
        sv.start()
        # warm step (migration/JIT), then the timed prefill step
        sv.step()
        t0 = time.perf_counter()
        sv.step()
        dt = time.perf_counter() - t0
    finally:
        sv.close()
    print(f"prefill forward: {dt * 1000:.0f}ms wall (warm serve step)",
          flush=True)
    n = len(ids)
    for L in range(N_LAYERS):
        K = np.concatenate(
            [np.fromfile(outmap[s], dtype=np.float32).reshape(S, DH)
             for _, _, s in [t for t in kouts if t[0] == L]], axis=1)
        V = np.concatenate(
            [np.fromfile(outmap[s], dtype=np.float32).reshape(S, DH)
             for _, _, s in [t for t in vouts if t[0] == L]], axis=1)
        K.astype(np.float16).tofile(os.path.join(args.outdir, f"ckP{L}.bin"))
        V.astype(np.float16).tofile(os.path.join(args.outdir, f"cvP{L}.bin"))
    lg = np.fromfile(outmap["LOGITS"], dtype=np.float32).reshape(S, -1)
    np.save(os.path.join(args.outdir, "prefill_logits.npy"), lg[n - 1])
    print(f"caches assembled in {args.outdir} (prompt rows 0..{n - 1} live)",
          flush=True)


if __name__ == "__main__":
    main()
