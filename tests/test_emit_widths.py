"""Agreement + speed gate v0.3: D32/D64 transformer blocks in C.

Same discipline as test_emit_headt (D16): LOGITS eps gate (CALIBRATED
per width, regression guards -- first measurements below), OUT exact,
binary standalone. Plus timing: lattice run_text wall vs emitted-C
wall on the same S=8 run (compile time excluded -- steady-state
inference is the claim). Asserts C is faster (weak bar) and reports
the factor.
Usage: python3 tests/test_emit_widths.py (needs cc; D64 slowest).
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import build, compile_program
from d64_lib import encode, load_bpe

FAIL = []
# Calibrated 2026-10-03 (first measurements, S=8; regression guards).
# Agreement has LEVELS (mirrors the joint rule): L1 value-eps where the
# program is conditioned, L2 decision-exact (argmax) everywhere. D64
# saturates its softmax (scores span -20..+51 over 70 points: winner-take
# ~1.0, runner-up mass e^-Delta flips on fixed-point rounding), so L1
# stops at H2 (numpy mirror: H2 5.7e-3, H3 0.51, LOGITS 3.2 -- the cliff
# is arithmetic conditioning, proven, not emission error) while L2 holds
# 8/8. D16/D32 stay conditioned through LOGITS.
CAL = {"d32": {"LOGITS": 1e-1}, "d64": {"H2": 2e-2, "LOGITS": None}}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


CASES = {
    "d32": {"fit": "lm_piece32_fit.npz", "bank": "bankp32_64.npz",
            "listing": "lm_d32_headt.asm",
            "cfg": ("CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
                    "CONFIG beta -30.0\nCONFIG beta_b 0.25\n"),
            "sents": ["alexander founded alexandria and ruled egypt well"]},
    "d64": {"fit": "lm_piece64_fit.npz", "bank": "bankpiece64_d64.npz",
            "listing": "lm_d64_headt.asm",
            "cfg": ("CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
                    "CONFIG beta -30.0\nCONFIG beta_b 0.25\n"),
            "sents": ["alexander founded alexandria and ruled egypt well"]},
}


def run_case(tag, c):
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    for f in (c["fit"], c["bank"], "bpe_vocab.json", "bpe_merges.json"):
        if not os.path.isfile(os.path.join(dd, f)):
            print(f"SKIP {tag} (missing {f})")
            return
    vocab, rank = load_bpe()
    ids = encode(c["sents"][0], vocab, rank)[:8]
    assert len(ids) >= 8, f"need S>=8, got {len(ids)}"
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    w = np.load(os.path.join(dd, c["fit"]))
    b = np.load(os.path.join(dd, c["bank"]))
    text = c["cfg"] + open(os.path.join(sdir, c["listing"])).read()
    trip = {"emb": enc(w["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
            "wv": enc(w["wv"]), "wo": enc(w["wo"]), "wup": enc(w["wup"]),
            "wgate": enc(w["wgate"]), "wdown": enc(w["wdown"]),
            "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
            "wlog": enc(w["wlog"]), "ukt": enc(b["ukt"]),
            "evb": enc(b["evb"])}
    payload = {"tok": toks, "pos": pos, "cmask": cm}
    payload.update(trip)
    t0 = time.perf_counter()
    feeds = ASM.run_text(text, REGISTRY, payload, sigs=SIGS, basedir=sdir)
    t_lat = time.perf_counter() - t0
    ref_logits = dec(feeds["LOGITS"])
    ref_out = np.ascontiguousarray(feeds["OUT"]).reshape(-1)

    fpayload = {"tok": toks, "pos": pos, "cmask": cm}
    fpayload.update({k: np.ascontiguousarray(dec(v))
                     for k, v in trip.items()})
    gate_names = list(CAL[tag])
    art = compile_program(text, "c", sample=fpayload,
                          outputs=gate_names + ["OUT"], basedir=sdir)
    work = f"/tmp/emit_{tag}_v03"
    os.makedirs(work, exist_ok=True)
    exe = build(art["source"], work, name=tag)
    argv = [exe]
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
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
    t0 = time.perf_counter()
    r = subprocess.run(argv, capture_output=True, text=True)
    t_c = time.perf_counter() - t0
    check(f"{tag}-build-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]}")
    if r.returncode != 0:
        return
    outs = {}
    for o, fn in zip(art["outputs"], out_files):
        ok = art["streams"][o][0]
        dt = np.int64 if ok == "I" else np.float64
        outs[o] = np.fromfile(fn, dtype=dt)
    got_out = outs["OUT"].reshape(ref_out.shape)
    for g in gate_names:
        ref = dec(feeds[g])
        got = outs[g].reshape(ref.shape)
        dmax = float(np.abs(got - ref).max())
        cal = CAL[tag][g]
        if cal is None:
            print(f"{tag}-{g}: maxabs={dmax:.3e} (informational, "
                  f"saturated -- no gate)")
            continue
        print(f"{tag}-{g}: maxabs={dmax:.3e} (cal {cal:.0e})")
        check(f"{tag}-{g}-eps", dmax < cal, f"maxabs={dmax:.3e}")
    same = bool((got_out == ref_out).all())
    check(f"{tag}-out-exact", same,
          f"{int((got_out == ref_out).sum())}/{len(ref_out)}")
    print(f"{tag}-time: lattice={t_lat:.2f}s c={t_c:.3f}s "
          f"speedup={t_lat / max(t_c, 1e-9):.1f}x")
    check(f"{tag}-faster", t_c < t_lat,
          f"c {t_c:.3f}s < lattice {t_lat:.2f}s")


def main():
    for tag, c in CASES.items():
        run_case(tag, c)
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
