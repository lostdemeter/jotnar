"""28-layer geometric Qwen vs HF original (comparison gate, heavy).

Same prompt through both: HF transformers (bf16, GPU) and the
builder-generated 9859-op geometric program (CUDA/FP16, VRAM). Gates
are fork-aware: greedy decoding amplifies sub-noise diffs into
different paths (measured: fork at margin 0.18 << noise 4.74, geo
picking HF's #2), so agreement is gated where it is meaningful --
parity bound (structural-break detector) + first-pick rank + fluency
(valid ids, non-empty decode) -- not string equality.
SKIPs without snapshot/GPU/nvcc. Slow (~8min: nvcc + HF load + steps).
Usage: python3 tests/test_qwen7b_gen.py
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []
PROMPT = "The capital of France is"
N_GEN = 5
SMAX = 16


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()  # aten reference numerics; triton JIT needs dev headers
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)
    g, tok = load7b()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy().tolist()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    print(f"HF load: {time.perf_counter() - t0:.0f}s", flush=True)
    with torch.no_grad():
        out = model.generate(**tok(PROMPT, return_tensors="pt").to("cuda"),
                             max_new_tokens=N_GEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        lg = model(**tok(PROMPT, return_tensors="pt").to("cuda")).logits
    hf_text = tok.decode(out[0], skip_special_tokens=True)
    hf_top = lg[0, -1].float().cpu().numpy()
    print("HF:", hf_text, flush=True)
    del model
    torch.cuda.empty_cache()

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen7b", os.path.join(ROOT, "scripts", "gen_qwen7b_geo.py"))
    # NOTE: importing would run main(); drive via subprocess instead.
    extra = []
    if os.environ.get("QWEN_GRAPH") == "1":
        extra = ["--graph", "--no-sync"]
    if os.environ.get("QWEN_BMMV") == "1":
        extra = extra + ["--bmmv"]
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "gen_qwen7b_geo.py"),
         PROMPT, "--n", str(N_GEN), "--smax", str(SMAX),
         "--out-json", "/tmp/gen7b_cmp.json"] + extra,
        capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-1500:] if r.stdout else "", flush=True)
    if r.returncode != 0:
        check("gen7b-geo-run", False, r.stderr[:300])
        print("FAILURES:", FAIL)
        sys.exit(1)
    import json
    cmp = json.load(open("/tmp/gen7b_cmp.json"))
    # Parity host-side when the geo subprocess ran geo-only (its parity
    # is None without in-process HF): compare its prompt-end logits row
    # against the HF logits living here.
    par = cmp["parity"]
    if par is None and cmp.get("logits_file"):
        glo = np.load(cmp["logits_file"])
        par = float(np.abs(glo - hf_top).max())
        print(f"host-side parity: maxabs={par:.3e}", flush=True)
    check("gen7b-parity", par is not None and par < 10.0,
          f"maxabs={par:.2f} (structural-break detector)" if par else "no logits")
    # First-pick rank is computed host-side: the geo subprocess runs
    # geo-only (combined HF+geo in one process is refused since the
    # swap-kill -- chain/mem.py), so it reports geo_first_id and the HF
    # logits living here do the ranking.
    frank = cmp.get("geo_first_rank")
    if frank is None and cmp.get("geo_first_id") is not None:
        _gf = int(cmp["geo_first_id"])
        frank = int((hf_top > hf_top[_gf]).sum()) + 1
    check("gen7b-first-pick-rank", frank is not None and frank <= 8,
          f"rank {frank} (measured 1)")
    check("gen7b-fluent", len(cmp["geo_text"].split()) >= len(ids),
          cmp["geo_text"][:80])
    print("HF :", hf_text[:120], flush=True)
    print("GEO:", cmp["geo_text"][:120], flush=True)
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
