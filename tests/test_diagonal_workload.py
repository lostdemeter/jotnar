"""Diagonal workload gate: fused vs serial unsharp on real frames.

N=3 webcam frames (1280x480 deterministic files) through
demo_img:build_img_prog unsharp on the lattice backend. Arm A: three
serial run_texts. Arm B: one fuse() batch (build_img_prog(3H,W),
halo="drop", out_rows for the shrunk outs, plain for the
full-geometry outs). Both arms feed get_depth (DAV2, distinct cache
tags per arm) off the `us` output through a deterministic mono->RGB
adapter -- identical depth maps prove the fused bytes are real
downstream input. Bars: unsharp bytes bit-exact every frame every
output; depth identical; fused rerun deterministic; halo==1 on the
built listing (the contract-first SLICE refusal it supersedes is
pasted in the merge report).
SKIP if frames/DAV2/torch missing (gold stays green). Cleans its
wl_* cache tags in finally (tree stays clean).
Usage: python3 tests/test_diagonal_workload.py (CPU lattice + GPU
DAV2, ~1min with warm weights).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

CAPDIR = "/home/balrog/Documents/OpenCode/dav2_revisited/dav2_reverse/captures"
FRAMES = [f"geo_webcam_{i}_combined.png" for i in range(3)]
W, H = 1280, 480
OUTS = ["gray", "inv", "br", "ct", "blur", "us", "strip"]
TAGS = [f"wl_a{i}" for i in range(3)] + [f"wl_b{i}" for i in range(3)] + ["wl_warm"]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def cleanup_tags():
    import glob as _g
    import chain.depthprior as _dp
    for pat in (os.path.join(_dp.CACHE, "wl_*.npy"),
                os.path.join(_dp.CACHE, "wl_*.key")):
        for f in _g.glob(pat):
            try:
                os.remove(f)
            except OSError:
                pass


def adapter(us_triple):
    """Deterministic mono->RGB adapter: us triples -> float -> [0,1] ->
    3-channel (L1 files-at-the-seam composition, per depthprior)."""
    import phi_core.lattice as S
    s, e, z = (np.ascontiguousarray(a) for a in us_triple)
    f = S.decode(s, e) * (1 - z.astype(np.float64))
    lo, hi = float(f.min()), float(f.max())
    n = np.zeros_like(f) if hi <= lo else (f - lo) / (hi - lo)
    return np.stack([n, n, n], axis=-1)


def main():
    from PIL import Image
    for f in FRAMES:
        if not os.path.isfile(os.path.join(CAPDIR, f)):
            print(f"SKIP (missing input {f})")
            return
    import chain.depthprior as DP
    if DP.DAV2 is None:
        print("SKIP (no DAV2 checkout sibling)")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP (no torch for DAV2)")
        return
    from demo_img import build_img_prog, const_vals, tenc
    from chain import asm as ASM, batch as B
    from chain.asm_ops import REGISTRY, SIGS

    ims = [np.asarray(Image.open(os.path.join(CAPDIR, f)).convert("RGB")
                      .resize((W, H))) for f in FRAMES]
    check("wl-frames", all(im.shape == (H, W, 3) for im in ims),
          f"common shape {(H, W)} (explicit resize, identity here)")

    text = build_img_prog(H, W).text()
    try:
        halo = B.required_halo(text)
    except B.BatchError as e:
        check("wl-contract", False, f"unlisted op refused: {e}")
        return
    check("wl-contract", halo == 1, f"halo=={halo} (SLICE tables at 1)")

    cv = const_vals()

    def payload(im):
        fp = {"r": tenc(im[:, :, 0].astype(np.int64)),
              "g": tenc(im[:, :, 1].astype(np.int64)),
              "b": tenc(im[:, :, 2].astype(np.int64))}
        for k, v in cv.items():
            sh = (H - 2, W - 2) if k in ("ninth", "amount") else (H, W)
            fp[k] = tenc(np.full(sh, v))
        return fp

    # -- depth warm (weights fetch/compute OUTSIDE the compared region) ----
    try:
        t0 = time.perf_counter()
        DP.get_depth(np.asarray(ims[0]).astype(np.float64) / 255.0, "wl_warm")
        print(f"wl-warm: depth weights ready "
              f"({time.perf_counter() - t0:.1f}s, outside compared region)",
              flush=True)
    except (ImportError, OSError) as e:
        print(f"SKIP (DAV2 unavailable: {e})")
        return

    try:
        # -- Arm A: serial --------------------------------------------------
        t0 = time.perf_counter()
        trs = []
        seps = []
        for im in ims:
            tr = []
            seps.append(ASM.run_text(text, REGISTRY, payload(im),
                                     sigs=SIGS, trace=tr))
            trs.append(len(tr))
        wall_a = time.perf_counter() - t0
        print(f"arm A serial: {wall_a:.1f}s ({trs} execs)", flush=True)

        # -- Arm B: one fused batch ------------------------------------------
        ftext = build_img_prog(3 * H, W).text()
        shared = {}
        for k, v in cv.items():
            sh = (3 * H - 2, W - 2) if k in ("ninth", "amount") else (3 * H, W)
            shared[k] = tenc(np.full(sh, v))  # geometry fixtures, not data
        lanes = []
        for im in ims:
            l = {"r": tenc(im[:, :, 0].astype(np.int64)),
                 "g": tenc(im[:, :, 1].astype(np.int64)),
                 "b": tenc(im[:, :, 2].astype(np.int64))}
            l.update(shared)
            lanes.append(l)
        t0 = time.perf_counter()
        trf = []
        got = B.fuse(ftext, lanes, halo="drop",
                     out_rows={k: H - 2 for k in ("blur", "us", "strip")},
                     plain={"gray", "inv", "br", "ct"}, trace=trf)
        wall_b = time.perf_counter() - t0
        print(f"arm B fused: {wall_b:.1f}s ({len(trf)} execs)", flush=True)

        def P(v):
            return np.stack(v) if isinstance(v, tuple) else np.asarray(v)

        exact = True
        for k in OUTS:
            ok = all(np.array_equal(P(got[i][k]), P(seps[i][k]))
                     for i in range(3))
            exact &= ok
            check(f"wl-{k}-exact", bool(ok), "3/3 frames byte-identical")
        check("wl-op-count", True,
              f"fused {len(trf)} execs vs serial {sum(trs)} "
              f"(dispatch vs flops disclaimer as Phase 1)")

        # -- determinism on the fused arm --------------------------------------
        got2 = B.fuse(ftext, lanes, halo="drop",
                      out_rows={k: H - 2 for k in ("blur", "us", "strip")},
                      plain={"gray", "inv", "br", "ct"})
        check("wl-determinism",
              bool(all(np.array_equal(P(got[i][k]), P(got2[i][k]))
                       for i in range(3) for k in OUTS)),
              "fused rerun byte-identical")

        # -- depth downstream (distinct tags per arm: real double compute) -----
        t0 = time.perf_counter()
        da = [DP.get_depth(adapter(P(seps[i]["us"])), f"wl_a{i}")
              for i in range(3)]
        t1 = time.perf_counter()
        db = [DP.get_depth(adapter(P(got[i]["us"])), f"wl_b{i}")
              for i in range(3)]
        t2 = time.perf_counter()
        print(f"depth: arm A {t1 - t0:.1f}s, arm B {t2 - t1:.1f}s "
              f"(downstream, not compared)", flush=True)
        check("wl-depth-identical",
              bool(all(np.array_equal(da[i], db[i]) for i in range(3))),
              "fused bytes are real input downstream")

        print(f"wall: serial {wall_a:.1f}s vs fused {wall_b:.1f}s "
              f"(execution regions only; direction reported, never asserted)",
              flush=True)
    finally:
        cleanup_tags()
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
