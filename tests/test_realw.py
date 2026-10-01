"""Real weights (v1.3): Qwen2-0.5B layer-0 MLP, direction ablation.

First measurement on TRAINED weights + real embeddings (8 wikitext-ish
tokens): full MLP block parity vs independent torch.float64 mirror
(51.82dB), decaying spectrum (17x, vs flat toy), direction spread 33dB
with the tracking law reproduced (corr -0.84, same as toy), planted
dominant direction recovered BIT-EXACTLY (control 18.8dB).
Scope, stated: attention is boundary float (real folded scores hit 963,
1000x past the softmax contract -- T-transform backlog measured live);
K-bias omitted by cancellation proof (row-constant -> softmax-invariant);
V-bias stated-negligible (max 1e-1 vs signals O(1)); Q-bias folded into
the torch H only (listing never sees attention internals).
SKIPs without the local HF cache (needs-hardware precedent).
Usage: python3 tests/test_realw.py (slow: ~60 listing runs, ~2 min)
"""
import os
import sys

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

FAIL = []
QWEN = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-0.5B", "snapshots",
                    "91d2aff3f957f99e4c74c962f2f408dcc88a18d8")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    import numpy as np
    mse = float(np.mean((np.ascontiguousarray(a, dtype=np.float64)
                         - np.ascontiguousarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    import numpy as np
    if not os.path.isfile(os.path.join(QWEN, "model.safetensors")):
        print("SKIP (needs Qwen2-0.5B in local HF cache)")
        sys.exit(0)
    try:
        from chain.qwen_mirror import load_layer0, build_H, mlp_forward
        g, embed, tok = load_layer0()
    except ImportError as e:
        print(f"SKIP (needs torch+safetensors+transformers: {e})")
        sys.exit(0)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    H, ids, toks = build_H(
        g, embed, tok,
        "The capital of France is Paris, and the capital of Germany is")
    REF, MIDf, Wupf, Wgf, Wdf, ln2f = mlp_forward(H, g)

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    base_pay = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
                "ln": enc(ln2f)}

    def run_down(WdT):
        pay = dict(base_pay)
        pay["wdown"] = enc(WdT)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    base = run_down(Wdf.T)
    d = psnr(base, REF)
    check("realw-parity", d >= 40.0,
          f"{d:.2f}dB vs torch (peak=1.0, outmax={np.abs(REF).max():.2f})")
    WdT = Wdf.T
    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)
    check("realw-spectrum", s[0] / s[-1] > 5.0,
          f"decaying {s[0]:.2f}..{s[-1]:.2f} (ratio {s[0]/s[-1]:.0f}x, not flat)")
    idx = list(range(0, 896, 16))
    ds, sv = [], []
    for i in idx:
        Wi = WdT - np.outer(U[:, i] * s[i], Vt[i])
        ds.append(psnr(run_down(Wi), base))
        sv.append(s[i])
    ds, sv = np.array(ds), np.array(sv)
    spread = float(ds.max() - ds.min())
    corr = float(np.corrcoef(ds, sv)[0, 1])
    check("realw-spread", spread > 20.0,
          f"{spread:.1f}dB over {len(idx)} sampled directions")
    check("realw-tracks", corr < -0.5,
          f"corr(dir-dB, sval)={corr:.2f} (law reproduces on trained weights)")
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4864,))
    u /= np.linalg.norm(u)
    v = rng.normal(size=(896,))
    v /= np.linalg.norm(v)
    A = 20.0
    Wp = WdT + A * np.outer(u, v)
    d_rec = psnr(run_down(Wp - A * np.outer(u, v)), base)
    u2 = rng.normal(size=(4864,))
    u2 /= np.linalg.norm(u2)
    v2 = rng.normal(size=(896,))
    v2 /= np.linalg.norm(v2)
    d_ctl = psnr(run_down(Wp - A * np.outer(u2, v2)), base)
    check("realw-planted", np.isinf(d_rec) and d_ctl < 40.0,
          f"recovery {'inf' if np.isinf(d_rec) else f'{d_rec:.1f}'}dB "
          f"vs control {d_ctl:.1f}dB (planted direction recovered exactly)")
    # Blind prediction in-suite (the edit-with-preview use, gated standing):
    # calibrate C on the sweep above, predict three NEVER-RUN dirs from
    # statics, run them, demand max err < 2dB (measured 0.3 -- margin 7x).
    from chain import read as RD
    cal_al = np.sqrt(((MIDf @ U[:, idx]) ** 2).mean(0))
    C = RD.calibrate_C(s[idx], cal_al, ds)
    errs = []
    for i in (100, 300, 700):
        a = float(np.sqrt(((MIDf @ U[:, i]) ** 2).mean()))
        p = RD.predict_db(float(s[i]), a, C)
        Wi = WdT - np.outer(U[:, i] * s[i], Vt[i])
        got = run_down(Wi)
        mse = float(np.mean((got - base) ** 2))
        d = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        errs.append(abs(p - d))
    check("realw-predict", max(errs) < 2.0,
          f"worst blind err {max(errs):.2f}dB (preview works in-suite)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
