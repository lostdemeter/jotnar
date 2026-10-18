"""Diagonal seam gate (Phase 2, Step 1): flagship batch refused at the seam.

Two images stacked along rows through programs/holo_flagship.asm: the
pointwise prefix (rgb/LIN/Y/A) is split-exact, but SPLAT_BLUR bleeds
exactly one halo row each side of the seam (rows 287/288 at H=288,
max 233 int units) and everything downstream is contaminated exactly
there -- OUT is exact on 574/576 rows. This gate PINS that finding:
any change in halo width fails loud (kernel change detector). Step 2
(cross-program) is gated on full green, which this is not -- the
refusal below IS the scoping result (see docs/DIAGONALS.md).
Second image: vertical flip of samples/input_example.png
(deterministic, same shape, genuinely different rows).
Usage: python3 tests/test_diagonal2.py (CPU only, seconds).
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

POINTWISE = ["rgb", "LIN", "Y", "A"]
SEAM_OP_OUTS = ["AS", "COH"]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def rows(a):
    """Row-split view: (rows, ...) with rows along axis 0, or axis 1
    for stacked-triple (3, rows, ...) planes."""
    a = np.stack(a) if isinstance(a, tuple) else np.asarray(a)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[1] > 3:
        return a[:, :, :], 1
    return a, 0


def main():
    from PIL import Image
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    im1 = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))
    im2 = im1[::-1].copy()
    H = im1.shape[0]
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    r1 = ASM.run_text(text, REGISTRY, {"rgb": im1}, sigs=SIGS)
    r2 = ASM.run_text(text, REGISTRY, {"rgb": im2}, sigs=SIGS)
    stacked = np.concatenate([im1, im2], axis=0)
    trf = []
    rf = ASM.run_text(text, REGISTRY, {"rgb": stacked}, sigs=SIGS, trace=trf)
    _, tr1, tr2 = [], [], []
    ASM.run_text(text, REGISTRY, {"rgb": im1}, sigs=SIGS, trace=tr1)
    ASM.run_text(text, REGISTRY, {"rgb": im2}, sigs=SIGS, trace=tr2)

    # -- pointwise prefix: fully split-exact -------------------------------
    pre = True
    for k in POINTWISE:
        f, ax = rows(rf[k])
        g1, _ = rows(r1[k])
        g2, _ = rows(r2[k])
        sl = lambda a, lo, hi: a[:, lo:hi] if ax == 1 else a[lo:hi]
        pre &= np.array_equal(sl(f, 0, H), g1) and np.array_equal(
            sl(f, H, 2 * H), g2)
    check("seam-pointwise-exact", bool(pre), "rgb/LIN/Y/A split == separate")

    # -- SPLAT_BLUR halo: pinned to exactly one row each side --------------
    halo_ok = True
    for k in SEAM_OP_OUTS:
        f, _ = rows(rf[k])
        g1, _ = rows(r1[k])
        g2, _ = rows(r2[k])
        topbad = [r for r in range(H)
                  if not np.array_equal(f[:, r, :], g1[:, r, :])]
        botbad = [r for r in range(H, 2 * H)
                  if not np.array_equal(f[:, r, :], g2[:, r - H, :])]
        halo_ok &= (topbad == [H - 1] and botbad == [H])
    check("seam-blur-halo", bool(halo_ok),
          "AS/COH bad rows exactly {287,288} (radius-1 kernel detector)")

    # -- downstream: contamination confined to the halo --------------------
    OUT = np.asarray(rf["OUT"])
    O12 = np.concatenate([np.asarray(r1["OUT"]), np.asarray(r2["OUT"])],
                         axis=0)
    good = list(range(0, H - 1)) + list(range(H + 1, 2 * H))
    check("seam-out-confined",
          bool(np.array_equal(OUT[good], O12[good])
               and not np.array_equal(OUT[[H - 1, H]], O12[[H - 1, H]])),
          "OUT exact on 574/576 rows, differs exactly at the seam")

    # -- determinism + cost accounting --------------------------------------
    rf2 = ASM.run_text(text, REGISTRY, {"rgb": stacked}, sigs=SIGS)
    check("seam-determinism",
          bool(all(np.array_equal(np.stack(rf[k]), np.stack(rf2[k]))
                   for k in rf if k != "CONFIG")),
          "fused rerun byte-identical")
    check("seam-op-count", len(trf) * 2 == len(tr1) + len(tr2),
          f"fused execs={len(trf)} vs separate={len(tr1) + len(tr2)} "
          f"(dispatch halved, flops identical -- Phase-1 honesty applies)")

    # -- flagship through the contract: zero manual row-handling -----------
    from chain import batch as BT
    r = BT.required_halo(text)
    check("seam-contract-radius", r == 1, f"table says {r} (SPLAT_BLUR)")
    t, b = BT.guard_band(np.asarray(rf["OUT"]), H, r)
    O1 = np.asarray(r1["OUT"])
    O2 = np.asarray(r2["OUT"])
    check("seam-contract-exact",
          bool(np.array_equal(t, O1[:H - r]) and np.array_equal(b, O2[r:])),
          "mechanical guard band == separates outside the halo")
    # triple-valued stream through the same mechanical path
    ta, ba = BT.guard_band(rf["AS"], H, r)
    A1 = np.stack(r1["AS"])
    check("seam-contract-triple",
          bool(np.array_equal(ta, A1[:, :H - r])),
          "guard band handles stacked triples, no caller indices")
    for tag, claimed in [("seam-contract-lie0", 0), ("seam-contract-lie2", 2)]:
        try:
            BT.assert_halo(text, claimed)
            check(tag, False, "accepted")
        except BT.BatchError as e:
            check(tag, True, str(e)[:70])

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
