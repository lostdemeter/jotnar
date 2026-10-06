"""Benchmark: geometric Qwen2-7B vs HF, per-token walls + per-op profile.

HF side: prefill forward + cached-decode per-token loop (past_key_values).
Geo side: full-recompute per step (no KV cache yet) + cudaEvent per-op
profile (time_ops build). Scaling across S separates O(n) from O(n^2).
Usage: python3 scripts/bench_perf.py [--s 16] [--ntok 8]
Prints a table; analysis lives in docs/PERF.md.
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

PROMPT = "The capital of France is Paris. It is the most populous city in"


def hf_bench(ntok=8):
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()  # aten reference numerics; triton JIT needs dev headers
    _, tok = load7b()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    S = ids.shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**{k: v.to("cuda") for k, v in
                        tok(PROMPT, return_tensors="pt").items()})
        out.logits.cpu()
    torch.cuda.synchronize()
    prefill = time.perf_counter() - t0
    # cached decode loop
    past, seq = None, ids
    times = []
    with torch.no_grad():
        for _ in range(ntok):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = model(input_ids=seq[:, -1:], past_key_values=past,
                      use_cache=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            past = o.past_key_values
            seq = torch.cat([seq, o.logits[:, -1:].argmax(-1)], dim=1)
    text = tok.decode(seq[0], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return {"S": S, "prefill_s": prefill, "decode_ms": [t * 1000 for t in times],
            "text": text}


def geo_build(S, profile=False):
    import importlib.util
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from chain.builder import Prog, qwen_layer
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    from chain.qwen7b import (load7b, shapes_7b, prep_7b_weight,
                             clear_7b_cache, input_sources)
    from chain import mem as _mem
    print(_mem.report(S), flush=True)
    _mem.guard(_mem.estimate_7b(S)["stream_peak_cpu"], "geo bench build")
    g, tok = load7b()
    p = Prog("qwen-bench")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "cmask"]
    for L in range(28):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog"]
    p.inp(*names)
    x = "x0"
    for L in range(28):
        x = qwen_layer(p, x, f"wq{L}", f"wk{L}", f"wv{L}", f"wo{L}",
                       f"wup{L}", f"wgate{L}", f"wdown{L}", f"ln1{L}",
                       f"ln2{L}", "pos", "cmask", pre=str(L),
                       bq=f"bq{L}", bk=f"bk{L}", bv=f"bv{L}")
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    E = np.ascontiguousarray(g("model.embed_tokens.weight"),
                              dtype=np.float16)  # 1.1GiB resident for steps
    # Exact-match lookup (never parse trailing digits). Unknown inputs
    # fail loud below.
    by_input = input_sources()
    # Zero-RAM stubs for compile; values stream straight to disk below.
    sample = shapes_7b(S)
    art = compile_program(p.text(), "cuda", sample=sample,
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, time_ops=profile)
    work = f"/tmp/bench7b_{'p' if profile else 'c'}"
    os.makedirs(work, exist_ok=True)
    t0 = time.perf_counter()
    exe = build_cu(art["source"], work, name="q7b")
    print(f"nvcc ({'profile' if profile else 'clean'}): "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"fr_{n}.bin")
        if n == "x0":
            np.zeros((S, 3584), dtype=np.float16).tofile(fn)
        elif n == "pos":
            np.arange(S, dtype=np.int64).tofile(fn)
        elif n == "cmask":
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        elif n == "lnf":
            prep_7b_weight(g, "model.norm.weight", "lnf", S).tofile(fn)
        elif n == "wlog":
            prep_7b_weight(g, "lm_head.weight", "wlog", S).tofile(fn)
        elif n in by_input:
            short, src = by_input[n]
            w16 = prep_7b_weight(g, src, short, S)
            if k == "F":
                w16.tofile(fn)
            else:
                w16.astype(np.int64).tofile(fn)
            del w16
        else:
            raise KeyError(f"no weight source for input stream {n!r}")
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    E.tofile(os.path.join(work, "fr_embed.bin"))
    del E
    _gc.collect()
    return exe, art, work


def geo_step(exe, art, work, S, tok_ids):
    import importlib.util  # noqa
    from chain.qwen7b import HIDDEN, QWEN7B_VOCAB
    n = len(tok_ids)
    # Embed table via memmap (never resident: 1.1GiB fp16 stays on disk,
    # rows paged per step; the old path loaded it float64 per call).
    E = np.memmap(os.path.join(work, "fr_embed.bin"), dtype=np.float16,
                  mode="r", shape=(QWEN7B_VOCAB, HIDDEN))
    argv = [exe]
    for name in art["inputs"]:
        k, _, _ = art["streams"][name]
        if name == "x0":
            fn = os.path.join(work, "step_x0.bin")
            xx = np.zeros((S, 3584), dtype=np.float16)
            xx[:n] = E[np.array(tok_ids)].astype(np.float16)
            xx.tofile(fn)
        elif name == "pos":
            # full arange(S): padding rows need valid positions (rotary
            # reads all S rows; short files read OOB heap).
            fn = os.path.join(work, "step_pos.bin")
            np.arange(S, dtype=np.int64).tofile(fn)
        elif name == "cmask":
            fn = os.path.join(work, "step_cm.bin")
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        else:
            fn = os.path.join(work, f"fr_{name}.bin")
        argv.append(fn)
    flo = os.path.join(work, "step_lg.bin")
    fou = os.path.join(work, "step_ou.bin")
    argv += [flo, fou]
    t0 = time.perf_counter()
    r = subprocess.run(argv, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"geo step failed rc={r.returncode}: "
                           f"{r.stderr[:300]}")
    lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
    ou = np.fromfile(fou, dtype=np.int64)
    return lg[n - 1], int(ou[n - 1]), dt, r.stderr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=16)
    ap.add_argument("--ntok", type=int, default=8)
    args = ap.parse_args()
    from chain.qwen7b import load7b
    _, tok = load7b()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy().tolist()
    print(f"prompt toks: {len(ids)}", flush=True)
    print("=== HF ===", flush=True)
    hf = hf_bench(args.ntok)
    print(f"HF prefill(S={hf['S']}): {hf['prefill_s'] * 1000:.0f}ms; "
          f"decode/token: {np.mean(hf['decode_ms']):.1f}ms "
          f"(min {min(hf['decode_ms']):.1f})", flush=True)
    print("HF:", hf["text"][:100], flush=True)
    print("=== GEO clean (wall/token) ===", flush=True)
    exe, art, work = geo_build(args.s, profile=False)
    seq = list(ids)
    times = []
    for _ in range(args.ntok):
        lg, nxt, dt, _ = geo_step(exe, art, work, args.s, seq)
        times.append(dt * 1000)
        seq.append(nxt)
        if len(seq) >= args.s:
            break
    print(f"GEO decode/token: {np.mean(times):.0f}ms (min {min(times):.0f})",
          flush=True)
    print("GEO:", tok.decode(seq, skip_special_tokens=True)[:100], flush=True)
    print("=== GEO profile (per-op cudaEvents) ===", flush=True)
    exe_p, art_p, work_p = geo_build(args.s, profile=True)
    lg, nxt, dt, err = geo_step(exe_p, art_p, work_p, args.s, list(ids))
    from collections import defaultdict
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for ln in err.strip().split("\n"):
        parts = ln.split()
        if len(parts) == 3 and parts[0] == "op":
            agg[parts[1]] += float(parts[2])
            cnt[parts[1]] += 1
    tot = sum(agg.values()) or 1
    print(f"profiled step: {tot:.0f}ms over {sum(cnt.values())} ops", flush=True)
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:14s} {v:8.1f}ms {v / tot * 100:5.1f}% x{cnt[k]}",
              flush=True)


if __name__ == "__main__":
    main()
