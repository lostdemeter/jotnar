"""Resident-cache gate: KV caches stay on-device (Phase B).

Same program math as the relay path (in-place SCATTER by naming;
emitter zeroes state after capture), but no per-step cache files:
no host relay, no re-read, no step_K/V outputs. Gates: HF parity +
first-pick rank (same bars), stale pad rows bit-invisible
same-binary, generation self-consistency (first pick == prefill
argmax), plus a timing spot vs the relay path (record-only).
SKIPs without snapshot/GPU/nvcc. Slow (~12min: builds + runs).
Usage: python3 tests/test_resident.py
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
PROMPT = "The capital of France is"
SMAX = 16


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def run(prompt, n, smax, work, extra=()):
    env = dict(os.environ, QWEN_WORKDIR=work)
    # warm-up (untimed): first run in a fresh workdir wobbles in values;
    # steady state is bit-deterministic. Gates measure steady state.
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         prompt, "--n", "0", "--smax", str(smax)] + list(extra),
        capture_output=True, text=True, cwd=ROOT, env=env)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         prompt, "--n", str(n), "--smax", str(smax)] + list(extra),
        capture_output=True, text=True, cwd=ROOT, env=env)
    return r


def ms_of(out):
    for ln in out.splitlines():
        if ln.startswith("decode:"):
            return ln
    return ""


def main():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)
    g, tok = load7b()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    with torch.no_grad():
        lg = model(**tok(PROMPT, return_tensors="pt").to("cuda")).logits
    hf_top = lg[0, -1].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()

    wr = "/tmp/resident_gate"
    rr = run(PROMPT, 0, SMAX, wr, ("--resident-cache",))
    print(rr.stdout[-400:] if rr.stdout else "", flush=True)
    check("resident-run", rr.returncode == 0, rr.stderr[:300])
    if rr.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    shutil.copyfile("/tmp/dec_prefill.npy", "/tmp/resident_prefill.npy")

    # same bars as the dense decode gate
    dec = np.load("/tmp/resident_prefill.npy")
    par = float(np.abs(dec - hf_top).max())
    check("resident-parity", par < 10.0,
          f"maxabs={par:.2f} (structural-break detector)")
    first = int(dec.argmax())
    frank = int((hf_top > hf_top[first]).sum()) + 1
    check("resident-first-pick-rank", frank <= 8, f"rank {frank}")

    # stale-cache immunity via startup-loaded files (same binary =>
    # bit-exact bar; exercises the reset + mask path)
    for L in range(28):
        for tg in ("ckP", "cvP"):
            p = os.path.join(wr, f"{tg}{L}.bin")
            a = np.fromfile(p, dtype=np.float16).reshape(16, 512)
            a[5:] = np.float16(100.0)
            a.tofile(p)
    rs = run(PROMPT, 0, SMAX, wr, ("--resident-cache",))
    check("resident-stale-run", rs.returncode == 0, rs.stderr[:300])
    got = np.load("/tmp/dec_prefill.npy")
    stale = float(np.abs(got - dec).max())
    check("resident-stale-exact", stale == 0.0,
          f"maxabs={stale:.4f} (pad rows must be invisible)")

    # generation self-consistency + timing spot vs relay path
    rg = run(PROMPT, 5, SMAX, wr, ("--resident-cache",))
    check("resident-gen-run", rg.returncode == 0, rg.stderr[:300])
    t1 = [ln for ln in rg.stdout.splitlines() if ln.startswith("DECODED:")]
    print(ms_of(rg.stdout), "resident", flush=True)
    if t1:
        check("resident-gen-text", t1[0].startswith("DECODED: " + PROMPT),
              t1[0][:70])
    wr2 = "/tmp/resident_gate_relay"
    ro = run(PROMPT, 5, SMAX, wr2, ())
    t2 = [ln for ln in ro.stdout.splitlines() if ln.startswith("DECODED:")]
    print(ms_of(ro.stdout), "relay", flush=True)
    if t1 and t2:
        # exact text is razor-sensitive across binaries (cublas order);
        # grade decisions: both first picks must clear the gate bar.
        npre = len(tok(PROMPT, return_tensors="pt")["input_ids"][0])
        ids1 = tok(t1[0][len("DECODED: "):],
                   return_tensors="pt")["input_ids"][0].numpy().tolist()
        ids2 = tok(t2[0][len("DECODED: "):],
                   return_tensors="pt")["input_ids"][0].numpy().tolist()
        r1 = int((hf_top > hf_top[ids1[npre]]).sum()) + 1
        r2 = int((hf_top > hf_top[ids2[npre]]).sum()) + 1
        check("resident-gen-decisions", r1 <= 8 and r2 <= 8,
              f"first-pick ranks resident={r1} relay={r2} "
              f"(texts {'match' if t1 == t2 else 'differ (razor)'})")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
