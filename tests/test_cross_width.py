"""Width spot-check: C backend vs CUDA at full 7B width, one layer.

The 7B chase is CUDA-only; this keeps the C backend honest at scale
without running all of 7B in float64. Same program, same weights,
both backends (CUDA in fp32, C in float64): agreement must be tight
eps (same math, different precision -- NOT bit-exact by contract).
Bar calibrated: first measurement x safety (see VELOCITY).
SKIPs without snapshot/GPU/nvcc. Slow (~5min: C scalar matmuls).
Usage: python3 tests/test_cross_width.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []
NTOK = 8
PROMPT = "The capital of France is Paris, and the capital of Germany is"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain.qwen7b import SNAP, load7b
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if shutil.which("nvcc") is None or not has_cuda:
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)
    g, tok = load7b()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][:NTOK].numpy()
    from chain.builder import Prog, qwen_layer
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    from chain.emit_c import build as _bc
    p = Prog("qwenL1-wide")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    in_names = ["x0", "pos", "cmask"]
    for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
              "ln1", "ln2", "bq", "bk", "bv"):
        in_names.append(f"{k}0")
    p.inp(*in_names)
    y = qwen_layer(p, "x0", "wq0", "wk0", "wv0", "wo0", "wup0", "wgate0",
                   "wdown0", "ln10", "ln20", "pos", "cmask", pre="0",
                   bq="bq0", bk="bk0", bv="bv0")
    n = len(ids)
    E = g("model.embed_tokens.weight")[np.array(ids)]
    keymap = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
              "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
              "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
              "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
              "ln2": "post_attention_layernorm.weight",
              "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
              "bv": "self_attn.v_proj.bias"}
    sample = {"x0": np.ascontiguousarray(E),
              "pos": np.arange(n, dtype=np.int64),
              "cmask": np.tril(np.ones((n, n), dtype=np.int64))}
    for k, src in keymap.items():
        w = g(f"model.layers.0.{src}")
        wt = np.ascontiguousarray(w.T)
        if k in ("wq", "bq"):
            wt = wt / np.sqrt(128)
        if k in ("bq", "bk", "bv"):
            wt = np.ascontiguousarray(np.tile(wt, (n, 1)))
        sample[f"{k}0"] = wt
    from chain.qwen7b import clear_7b_cache
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()

    work = "/tmp/cross_wide"
    os.makedirs(work, exist_ok=True)
    artc = compile_program(p.text(), "c", sample=sample, outputs=[y],
                           basedir=os.path.join(ROOT, "programs"))
    exec_ = _bc(artc["source"], work, name="qc")
    argv = [exec_]
    for name in artc["inputs"]:
        k, _, _ = artc["streams"][name]
        fn = os.path.join(work, f"c_{name}.bin")
        if k == "F":
            sample[name].astype(np.float64).tofile(fn)
        else:
            sample[name].astype(np.int64).tofile(fn)
        argv.append(fn)
    fo = os.path.join(work, "cout.bin")
    argv.append(fo)
    import time
    t0 = time.perf_counter()
    r = subprocess.run(argv, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    check("cross-c-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]} wall={dt:.0f}s")
    if r.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    ref = np.fromfile(fo, dtype=np.float64).reshape(n, 3584)

    artu = compile_program(p.text(), "cuda", sample=sample, outputs=[y],
                           basedir=os.path.join(ROOT, "programs"),
                           use_fp16=False)
    exeu = build_cu(artu["source"], work, name="qu")
    argv = [exeu]
    for name in artu["inputs"]:
        k, _, _ = artu["streams"][name]
        fn = os.path.join(work, f"u_{name}.bin")
        if k == "F":
            sample[name].astype(np.float32).tofile(fn)
        else:
            sample[name].astype(np.int64).tofile(fn)
        argv.append(fn)
    fo = os.path.join(work, "uout.bin")
    argv.append(fo)
    r = subprocess.run(argv, capture_output=True, text=True)
    check("cross-cu-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]}")
    if r.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    got = np.fromfile(fo, dtype=np.float32).reshape(ref.shape)
    dmax = float(np.abs(got - ref).max())
    rel = dmax / max(float(np.abs(ref).max()), 1e-12)
    print(f"cross-wide: maxabs={dmax:.3e} rel={rel:.3e}", flush=True)
    check("cross-wide-eps", np.isfinite(dmax) and rel < 1e-4,
          f"rel={rel:.3e} (C-f64 vs CUDA-f32 at full width)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
