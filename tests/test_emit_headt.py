"""Agreement gate v0.2: full transformer block (lm_headt.asm) in C.

Float path => EPSILON agreement class (lattice fixed-point vs C
float64): the LOGITS gate below is CALIBRATED (first measurement x
safety) and guards regressions -- labeled honestly, not preregistered.
OUT (argmax) is exact-or-explained: any flip must come with its margin.
Usage: python3 tests/test_emit_headt.py (needs cc; S=3 toy run).
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import build, compile_program

FAIL = []
# Calibrated 2026-10-03: measured 2.890e-02 on S=3 (matches the numpy-float
# mirror exactly -- residual is pure lattice fixed-point quantization, not
# emission error). Regression guard at ~2x; the E-materialization bug read
# 8.76 (300x), so real emission breaks cannot hide under this bar.
CAL_LOGITS_MAXABS = 6e-2


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    for f in ("lm_svd_IvoQ.npz", "bankhn.npz"):
        if not os.path.isfile(os.path.join(dd, f)):
            print(f"SKIP (missing {f})")
            sys.exit(0)
    CFG = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
           "CONFIG beta_b 0.25\n")
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ids = [12, 471, 59]
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    trip = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
            "wv": enc(d["wv"]), "wo": enc(d["wo"]), "wup": enc(d["wup"]),
            "wgate": enc(d["wgate"]), "wdown": enc(d["wdown"]),
            "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
            "wlog": enc(d["wlog"]), "ukt": enc(b["ukt"]),
            "evb": enc(b["evb"])}
    payload = {"tok": toks, "pos": pos, "cmask": cm}
    payload.update(trip)
    feeds = ASM.run_text(text, REGISTRY, payload, sigs=SIGS, basedir=sdir)
    ref_logits = dec(feeds["LOGITS"])
    ref_out = np.ascontiguousarray(feeds["OUT"]).reshape(-1)

    fpayload = {"tok": toks, "pos": pos, "cmask": cm}
    fpayload.update({k: np.ascontiguousarray(dec(v)) for k, v in trip.items()})
    art = compile_program(text, "c", sample=fpayload,
                          outputs=["LOGITS", "OUT"], basedir=sdir)
    work = "/tmp/emit_headt_v02"
    os.makedirs(work, exist_ok=True)
    exe = build(art["source"], work, name="headt")
    argv = [exe]
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        if k == "T":
            raise AssertionError("headt C run expects decoded F inputs")
        fn = os.path.join(work, f"in_{n}.bin")
        if k == "F":
            fpayload[n].astype(np.float64).tofile(fn)
        else:
            fpayload[n].astype(np.int64).tofile(fn)
        argv.append(fn)
    out_files = []
    for o in art["outputs"]:
        fn = os.path.join(work, f"out_{o}.bin")
        out_files.append(fn)
        argv.append(fn)
    r = subprocess.run(argv, capture_output=True, text=True)
    check("headtC-build-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:300]}")
    if r.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    got_logits = np.fromfile(out_files[0], dtype=np.float64
                             ).reshape(ref_logits.shape)
    got_out = np.fromfile(out_files[1], dtype=np.int64)
    dmax = float(np.abs(got_logits - ref_logits).max())
    print(f"headtC-logits: maxabs={dmax:.3e} (cal {CAL_LOGITS_MAXABS:.0e})")
    check("headtC-logits-eps", dmax < CAL_LOGITS_MAXABS,
          f"maxabs={dmax:.3e}")
    same = bool((got_out == ref_out).all())
    if not same:
        for i in range(len(ref_out)):
            if got_out[i] != ref_out[i]:
                row = ref_logits[i]
                part = np.partition(row, -2)[-2:]
                print(f"  row {i}: lattice={ref_out[i]} c={got_out[i]} "
                      f"margin={part[1] - part[0]:.3e}")
    check("headtC-out-exact", same,
          f"{int((got_out == ref_out).sum())}/{len(ref_out)} rows")
    ldd = subprocess.run(["ldd", exe], capture_output=True, text=True).stdout
    check("headtC-standalone", "phi" not in ldd and "python" not in ldd.lower(),
          "libc+libm only" if ldd else "?")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
