"""Causal probes (v1.3 gate 2): zero a stream, measure the dB delta.

Interventions run IN-LISTING via ASM overrides (replacing a computed
stream before consumers read it; unknown names fail loud so probes must
bite). Each probe states its band BEFOREHAND (paper-first); confirmed
bands are barred, falsified ones become measured rows with revised
understanding (the #LIB-015 doctrine). The probe TABLE is the
representation: one row per (stream, intervention), growing into the
content map as probes accumulate.

Predictions (stated before any probe ran, from mechanism):
- D-zero removes the entire boost (AE=A -> YENH=Y): base preserved, tens
  of dB. Band [20,45]dB.
- COH-zero kills orientation (iso fallback + v5 gate carry on): moderate
  effect. Band [3,40]dB.
Results: D-zero 38.46dB CONFIRMED (mean change 1.35 LSB ~= the full
enhancement -- the boost IS the detail path, no other contributor).
COH-zero 44.23dB FALSIFIED on the upper bound (0.46 LSB -- coherence
decides WHERE boost applies more than HOW MUCH; iso/atten paths carry
the magnitude). Revised band [38,50]dB goes FORWARD onto a held-out
variant (flipped frame, never probed before): a genuine prediction.
Usage: python3 test_probe.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
BAR_LO, BAR_HI = 20.0, 45.0
ROWS = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr8(a, b, peak=255.0):
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def load_rgb():
    root = os.path.dirname(os.path.abspath(__file__))
    ext = os.path.join(root, "..", "rife_reverse", "samples", "f_012.png")
    cand = ext if os.path.isfile(ext) else os.path.join(root, "samples", "input_example.png")
    return np.asarray(Image.open(cand).convert("RGB"))


def probe(text, rgb, ov):
    base = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS)["OUT"]
    got = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS, overrides=ov)["OUT"]
    d = psnr8(got, base)
    m = float(np.abs(got.astype(np.float64) - base.astype(np.float64)).mean())
    return d, m


def main():
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "programs", "holo_flagship.asm")).read()
    rgb = load_rgb()
    Hh, Ww, _ = rgb.shape
    # sham: override COH with its own computed value (machinery adds nothing)
    f0 = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS)
    sham = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS,
                        overrides={"COH": f0["COH"]})["OUT"]
    check("probe-sham", bool((sham == f0["OUT"]).all()),
          "sham intervention bit-exact (machinery adds nothing)")
    # typo guard lives in asm.py; one probe-level refusal pins the surface
    try:
        ASM.run_text(text, REGISTRY, rgb, sigs=SIGS, overrides={"NOPE": 0})
        check("probe-refusal", False, "accepted dead override")
    except ASM.AsmError:
        check("probe-refusal", True, "dead overrides fail loud")
    cohz = S.encode(np.zeros((Hh, Ww)))
    Dz = S.encode(np.zeros(f0["D"][0].shape))
    d_coh, m_coh = probe(text, rgb, {"COH": cohz})
    ROWS.append(("COH", "zero", d_coh, m_coh, "[3,40]dB PREDICTED",
                 "FALSIFIED (upper)" if not 3.0 <= d_coh <= 40.0 else "confirmed"))
    print(f"probe-coh-measured: {d_coh:.2f}dB (mean {m_coh:.2f} LSB) -- "
          f"predicted [3,40], revised understanding: iso fallback dominates")
    d_d, m_d = probe(text, rgb, {"D": Dz})
    ok_d = BAR_LO <= d_d <= BAR_HI
    ROWS.append(("D", "zero", d_d, m_d, "[20,45]dB PREDICTED",
                 "CONFIRMED" if ok_d else "FALSIFIED"))
    check("probe-D-zero", ok_d, f"{d_d:.2f}dB in [20,45] (mean {m_d:.2f} LSB)")
    # forward prediction on HELD-OUT CONTENT (a different tracked image --
    # note: an earlier draft used fliplr, which proved nothing: the pipeline
    # is flip-equivariant, so flipped deltas match to the decimal BY
    # CONSTRUCTION. Caught by reading identical decimals. The revised
    # mechanism (iso fallback dominates magnitude) must survive genuinely
    # different buckets/coherence to mean anything.)
    var = np.asarray(Image.open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "samples",
        "output_example_phi_beta05.png")).convert("RGB"))
    vd_coh, vm_coh = probe(text, var, {"COH": cohz})
    ok_vc = 38.0 <= vd_coh <= 50.0
    ROWS.append(("COH@variant", "zero", vd_coh, vm_coh, "[38,50]dB PREDICTED",
                 "CONFIRMED" if ok_vc else "FALSIFIED"))
    check("probe-coh-forward", ok_vc, f"{vd_coh:.2f}dB in [38,50] (held-out)")
    vd_d, vm_d = probe(text, var, {"D": Dz})
    ok_vd = BAR_LO <= vd_d <= BAR_HI
    ROWS.append(("D@variant", "zero", vd_d, vm_d, "[20,45]dB PREDICTED",
                 "CONFIRMED" if ok_vd else "FALSIFIED"))
    check("probe-D-forward", ok_vd, f"{vd_d:.2f}dB in [20,45] (held-out)")
    print("--- probe table (stream | intervention | delta-dB | mean|d| | band | verdict) ---")
    print("    (mean|d| in LSB for u8 rows, activation units for xf rows)")
    for row in ROWS:
        print("    %-12s %-5s %7.2fdB %6.2f    %-18s %s" % row)
    # xf_block structure-level rows (MEASURED, not barred -- information
    # first, bands later: toy-scale weights, everything subtle here).
    # Interventions address STRUCTURE outputs (O, DOWN), not mangled
    # sub-structure wires (attn_core#1.P etc.) -- the assembly-view
    # doctrine: pipelines read as structure invocations.
    import test_xf_block as XB
    xf, posf, wf, r1f, r2f = XB.fixture()
    xtext = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "programs", "xf_block.asm")).read()
    xpay = {"x": XB.enc(xf), "pos": posf, "wq": XB.enc(wf[0]),
            "wk": XB.enc(wf[1]), "wv": XB.enc(wf[2]), "wo": XB.enc(wf[3]),
            "wup": XB.enc(wf[4]), "wgate": XB.enc(wf[5]),
            "wdown": XB.enc(wf[6]), "rms_w1": XB.enc(r1f),
            "rms_w2": XB.enc(r2f)}
    xb0 = XB.dec(ASM.run_text(xtext, REGISTRY, xpay, sigs=SIGS,
                              basedir=os.path.join(
                                  os.path.dirname(os.path.abspath(__file__)),
                                  "programs"))["OUT"])

    def xprobe(nm):
        z = S.encode(np.zeros(
            ASM.run_text(xtext, REGISTRY, xpay, sigs=SIGS,
                         basedir=os.path.join(
                             os.path.dirname(os.path.abspath(__file__)),
                             "programs"))[nm][0].shape))
        g = XB.dec(ASM.run_text(xtext, REGISTRY, xpay, sigs=SIGS,
                                basedir=os.path.join(
                                    os.path.dirname(os.path.abspath(__file__)),
                                    "programs"),
                                overrides={nm: z})["OUT"])
        mse = float(np.mean((g - xb0) ** 2))
        m = float(np.abs(g - xb0).mean())
        return (10 * np.log10(1.0 / mse) if mse > 0 else float("inf")), m

    for tag, nm in (("O (attn kill)", "O"), ("DOWN (mlp kill)", "DOWN")):
        d1, m1 = xprobe(nm)
        d2, _ = xprobe(nm)
        ROWS.append((nm + "@xf", "zero", d1, m1, "MEASURED",
                     "deterministic" if abs(d1 - d2) < 0.01 else "UNSTABLE"))
    check("probe-xf-deterministic", all(r[5] == "deterministic" for r in ROWS[-2:]),
          "rerun matches to 0.01dB")
    print("--- xf rows ---")
    for row in ROWS[-2:]:
        print("    %-12s %-5s %7.2fdB %6.2f    %-18s %s" % row)
    # FULL SWEEP: every flagship stream (except IN rgb and OUT itself).
    # Derived equalities (pipeline algebra, predicted EXACT before running):
    # BEFF-zero == BD-zero == D-zero (all force BD=0 downstream);
    # AE-zero == YENH-zero (both force YENH=0). Any mismatch means the
    # mental model of the composition is wrong somewhere.
    import phi_core.lattice as _S

    def zeros_like(v):
        if isinstance(v, tuple):
            return _S.encode(np.zeros(v[0].shape))
        return np.zeros(v.shape, dtype=np.float64)

    sweep = {}
    for nm in ("LIN", "Y", "A", "AS", "BEFF", "BD", "AE", "YENH", "G"):
        z = zeros_like(f0[nm])
        got = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS,
                           overrides={nm: z})["OUT"]
        d = psnr8(got, f0["OUT"])
        m = float(np.abs(got.astype(np.float64)
                         - f0["OUT"].astype(np.float64)).mean())
        sweep[nm] = (got, d, m)
        ROWS.append((nm, "zero", d, m, "MEASURED", "sweep"))
    check("probe-sweep ran", len(sweep) == 9, "all streams intervened")
    # BD-path triple equality: BEFF-zero, BD-zero, D-zero all force BD=0.
    dzero_out = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS,
                             overrides={"D": zeros_like(f0["D"])})["OUT"]
    check("probe-eq-bdzero",
          bool((sweep["BEFF"][0] == sweep["BD"][0]).all())
          and bool((sweep["BD"][0] == dzero_out).all()),
          "BEFF-zero == BD-zero == D-zero, bit-exact (same forced BD)")
    check("probe-eq-aeyenh",
          bool((sweep["AE"][0] == sweep["YENH"][0]).all()),
          "AE-zero == YENH-zero, bit-exact (same forced YENH)")
    # Discovered (not predicted -- MEASURED, mechanism verified after):
    # {Y,A,AE,YENH}-zero all force YENH=0 (halved image); {LIN,G}-zero both
    # force black (LIN feeds GAIN directly: 0 * clipped-gain = 0 -- the
    # census fan-out of LIN over LUMA+GAIN predicted this subtlety).
    check("probe-eq-yenh0",
          all(bool((sweep[k][0] == sweep["Y"][0]).all()) for k in ("A", "AE", "YENH")),
          "Y==A==AE==YENH-zero, bit-exact (YENH=0 class)")
    check("probe-eq-black",
          bool((sweep["LIN"][0] == sweep["G"][0]).all()),
          "LIN-zero == G-zero, bit-exact (black class)")
    print("--- full sweep ---")
    for row in ROWS[-9:]:
        print("    %-12s %-5s %7.2fdB %6.2f    %-18s %s" % row)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
