"""Representation survey gate: the map's numbers, not its opinions.

ROWS pins each candidate's verdict integrity programmatically
(LIVE needs a falsifier + evaluated proof; DEAD needs killing
evidence; nothing unevaluated is LIVE). Measurement legs re-run the
fast rows (GEMM ubench band, flagship SPLAT share, demo ADD share,
norm-vs-pointwise ratio, LUT speed) and the slow decode rows iff
their profile workdirs exist (SKIP otherwise -- setup in
docs/REPRESENTATIONS.md). Opinions live in the spec; this file pins
the numbers underneath them.
Usage: python3 tests/test_rep_survey.py (CPU + quick CUDA ubench).
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

# name, verdict, evaluated?, falsifier, killing/supporting evidence
ROWS = [
    ("quantize-weights", "PARKED", True,
     "parity <=4.85 + traffic drop at same wall",
     "96.2% GEMM share; standing weight-change order parks it"),
    ("gemm-epilogue", "PARKED", False,
     "epilogue GEMM wall < separate at parity",
     "backend change, unmeasured (named second in FUSION.md)"),
    ("sparse-S16", "DEAD", True, "",
     "ATTN 4.7/2874ms (0.16%) at S=16 -- nothing to skip"),
    ("sparse-S128", "DEAD", True, "",
     "ATTN 13.1/3064ms (0.43%) at S=128 -- S2 growth insufficient"),
    ("rmsnorm-1pass", "DEAD", True, "",
     "0.29% S=16 in-situ; lattice 1.17ms/4k absolute"),
    ("softmax-dense", "DEAD", True, "",
     "no standalone dense softmax (inside fused ATTN 0.16%)"),
    ("scatter-resident", "DEAD", True, "",
     "shipped as resident caches (27 vs 30ms); remainder 0.3-1.6% floor"),
    ("gather-upcast", "DEAD", True, "",
     "2.6% S=128 relay; resident removes the re-read"),
    ("argmax-tail", "DEAD", True, "",
     "required every greedy step; file-skip already shipped"),
    ("splat-separable", "PARKED", False,
     "separable >=40dB vs splat + wall win",
     "191/257ms lattice flagship; no production path demands it"),
    ("lut-finer", "DEAD", True, "",
     "0.17-0.25ms/4k rows -- already LUTs, already fast"),
    ("dispatch-fusion-speed", "DEAD", True, "",
     "tied walls twice (260/255ms flagship, 2.5/2.5s unsharp)"),
    ("convert-elim", "DEAD", True, "", "0.2% in-situ"),
    ("slice-zero-copy", "DEAD", True, "",
     "views already zero-copy where possible; 9.3ms demo total"),
]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    # -- verdict integrity (no spec parsing; rules over ROWS) ------------------
    live = [r for r in ROWS if r[1] == "LIVE"]
    check("rep-no-unexamined-live",
          all(r[3] and r[2] for r in live), f"{len(live)} LIVE rows")
    check("rep-dead-have-evidence",
          all(r[4] for r in ROWS if r[1] == "DEAD"), "every DEAD names kill")
    check("rep-verdicts-closed",
          all(r[1] in ("LIVE", "DEAD", "PARKED") for r in ROWS),
          f"{len(ROWS)} rows")
    check("rep-live-empty", live == [],
          "no LIVE speed candidates (honest empty, see spec)")

    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S
    rng = np.random.default_rng(11)

    # -- GEMM throughput band (fresh ubench, skip first cold row) ---------------
    ub = os.path.join(ROOT, "bench", "matmul_decode.cu")
    ubbin = "/tmp/rep_ub"
    r = subprocess.run(["nvcc", "-O2", "-arch=native", "-o", ubbin,
                        ub, "-lcublas"], capture_output=True, text=True)
    if r.returncode:
        print(f"SKIP rep-gemm (needs nvcc: {r.stderr[-120:]})")
    else:
        import re as _re
        p = subprocess.run([ubbin, "20"], capture_output=True, text=True)
        rows = []
        for ln in p.stdout.splitlines():
            m = _re.search(r"\|\s*A\s+([\d.]+)ms\s+([\d.]+)GB/s", ln)
            if m:
                rows.append(float(m.group(2)))
        check("rep-gemm-band", len(rows) >= 9
              and all(350 <= v <= 950 for v in rows[1:]),
              f"{len(rows)} rows, min {min(rows[1:]):.0f} max "
              f"{max(rows[1:]):.0f} GB/s (cold first row skipped)")

    # -- flagship SPLAT share (>60% on 48x64 noise) ------------------------------
    ftext = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    im = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    tr = []
    ASM.run_text(ftext, REGISTRY, {"rgb": im}, sigs=SIGS, trace=tr)
    tot = sum(r["sec"] for r in tr)
    splat = sum(r["sec"] for r in tr if r["op"] == "SPLAT_BLUR")
    check("rep-splat-share", splat / tot > 0.60,
          f"{splat / tot * 100:.0f}% of lattice flagship wall")

    # -- demo_img ADD share (>35% on 48x64) ----------------------------------------
    from demo_img import build_img_prog, const_vals, tenc
    h, w = 48, 64
    dtext = build_img_prog(h, w).text()
    cv = const_vals()
    rr = (rng.random((h, w)) * 255)
    fp = {"r": tenc(rr.astype(np.int64)), "g": tenc(rr.astype(np.int64)),
          "b": tenc(rr.astype(np.int64))}
    for k, v in cv.items():
        sh = (h - 2, w - 2) if k in ("ninth", "amount") else (h, w)
        fp[k] = tenc(np.full(sh, v))
    tr = []
    ASM.run_text(dtext, REGISTRY, fp, sigs=SIGS, trace=tr)
    tot = sum(r["sec"] for r in tr)
    add = sum(r["sec"] for r in tr if r["op"] == "ADD")
    check("rep-add-share", add / tot > 0.35,
          f"{add / tot * 100:.0f}% of lattice unsharp wall")

    # -- norm vs pointwise + LUT speed (64x64 microbench) ---------------------------
    def eltime(line, pay, iters=10):
        t = "CONFIG m_acc 31344\nCONFIG m_cov 31344\n"
        t += "".join(f"IN {k}\n" for k in pay) + line + "\n"
        ASM.run_text(t, REGISTRY, dict(pay), sigs=SIGS)
        import time as _t
        t0 = _t.perf_counter()
        for _ in range(iters):
            ASM.run_text(t, REGISTRY, dict(pay), sigs=SIGS)
        return (_t.perf_counter() - t0) / iters / 4096
    enc = lambda a: S.encode(np.ascontiguousarray(a))
    e_silu = eltime("OUT = SILU(x)", {"x": enc(rng.normal(size=(64, 64)))})
    e_sig = eltime("OUT = SIGMOID(x)", {"x": enc(rng.normal(size=(64, 64)))})
    e_rms = eltime("OUT = RMSNORM(x, w)",
                   {"x": enc(rng.normal(size=(64, 64)) * .3),
                    "w": enc(np.abs(rng.normal(size=(64,))) * .1 + .5)})
    check("rep-rms-vs-pointwise", e_rms > 3 * e_silu,
          f"rms {e_rms * 1e9:.0f}ns/el vs silu {e_silu * 1e9:.0f}ns/el")
    check("rep-lut-fast", e_silu < 1e-6 and e_sig < 1e-6,
          f"silu/sigmoid sub-us/el (already LUTs)")

    # -- decode profile rows (SKIP without profile workdirs) -------------------------
    import re as _re2
    for tag, path, want, desc in [
            ("rep-decode16-matmul", "/tmp/profrep/dec.err",
             lambda s: s.get("MATMUL", (0, 0))[0] / max(s.get("TOT", 1), 1) > 0.85,
             "MATMUL share >85% at S=16"),
            ("rep-decode128-attn", "/tmp/profrep128/dec.err",
             lambda s: s.get("ATTN", (0, 0))[0] / max(s.get("TOT", 1), 1) < 0.02,
             "ATTN share <2% at S=128")]:
        if not os.path.isfile(path):
            print(f"SKIP {tag} (no {path}; setup in docs/REPRESENTATIONS.md)")
            continue
        shares, tot = {}, 0.0
        with open(path, errors="replace") as fh:
            for ln in fh:
                m = _re2.match(r"op (\S+) ([\d.]+)", ln)
                if m:
                    shares[m.group(1)] = shares.get(m.group(1), 0.0) + float(m.group(2))
                    tot += float(m.group(2))
        tab = {k: (v, 0) for k, v in shares.items()}
        tab["TOT"] = tot
        got = {k: v / tot for k, v in shares.items()}
        check(tag, want(tab), {k: f"{v * 100:.1f}%" for k, v in
                               sorted(got.items(), key=lambda kv: -kv[1])[:4]})

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
