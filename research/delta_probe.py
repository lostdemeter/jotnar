"""Delta-ratio probe: mirror vs geo effect of the same nominal add.

The 8-16x dose gap stands unexplained by wiring (dose-responsive)
or addressing (Y2 parity identical). Candidates: L27-Jacobian
amplification in mirror (steer applies PRE-L27, transformed by
attention+MLP; ~1.5x per lin_map) + fp16-vs-fp32 response + fork
base margins. Measurement: delta-logit NORMS (and correlation)
mirror-gain-1 vs geo-gain-1 on the everest prompt, same ball data.
Ratio >>1.5x implicates fp16/fork over Jacobian; ~1.5x closes it to
Jacobian alone. Uses the Y2 workdir binary + files in place (backup
+ restore Vc around the zero run; no rebuild).
Usage: python3 research/delta_probe.py (needs snapshot + GPU)
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

WORK = "/tmp/gen7b_any_everest"
SMAX = 8
PROMPT = "The tallest mountain on Earth is"


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from siphon_battery import country_pos
    from chain.engram import yarnball_bank
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    tid = tok(" Everest", return_tensors="pt")["input_ids"][0].tolist()[0]
    _, pos = country_pos(PROMPT, "mountain", tok)
    key, _ = early_key(PROMPT, pos=pos)
    v = Wlog[tid] / np.linalg.norm(Wlog[tid])
    keys = {"everest": key}
    for c, p in (("germany", "The capital of Germany is"),
                 ("italy", "The capital of Italy is")):
        _, pp = country_pos(p, {"germany": "Germany", "italy": "Italy"}[c], tok)
        k, _ = early_key(p, pos=pp)
        keys[c] = k
    stores = [{"key": keys["everest"], "value": v, "dose": 1.0,
               "tier": "assoc", "support": "everest"}]
    for c in ("germany", "italy"):
        stores.append({"key": keys[c], "value": np.zeros(D),
                       "tier": "null", "support": f"bg-{c}"})
    Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                              stores, key_scale=8.0)
    t7, _ = fwdH(PROMPT, keep="all")
    xa = t7[2][pos]
    xa = xa / np.linalg.norm(xa)
    C = xa @ Ua
    P = np.exp(C - C.max())
    P /= P.sum()
    mag = float(np.linalg.norm(t7[27][-1]))
    y = (P @ Vc) * (1.0 * mag)
    yn = y / (np.linalg.norm(y) + 1e-12)
    lg0, _ = fwd(PROMPT)
    lg1, _ = fwd(PROMPT, steer=(27, yn, float(np.linalg.norm(y) / mag)))
    dm = lg1 - lg0
    print(f"mirror: |delta|={float(np.linalg.norm(dm)):.2f} "
          f"max|delta|={float(np.abs(dm).max()):.2f} "
          f"P={np.round(P, 3).tolist()}", flush=True)
    np.save("/tmp/delta_mirror.npy", dm)
    # geo side: restart serve per Vc (weights-load-once): gain-1 file in
    # place from the Y2 run; back it up, zero it, measure, restore.
    from chain.serve import ServedExe
    exe = os.path.join(WORK, "q7b_any")
    if not os.path.isfile(exe):
        print("SKIP (no Y2 binary; run siphon_any everest first)")
        return
    vc = os.path.join(WORK, "fr_Vc27.bin")
    bak = vc + ".bakdelta"
    shutil.copy(vc, bak)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy().tolist()
    S, HID = SMAX, 3584
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)

    def write_live():
        xx = np.zeros((S, HID), dtype=np.float16)
        xx[:len(ids)] = E[np.array(ids)].astype(np.float16)
        xx.tofile(os.path.join(WORK, "step_x0.bin"))
        np.arange(S, dtype=np.int64).tofile(os.path.join(WORK, "step_pos.bin"))
        np.tril(np.ones((S, S), dtype=np.int64)).tofile(os.path.join(WORK, "step_cm.bin"))

    def read_out():
        flo = os.path.join(WORK, "step_lg.bin")
        return np.fromfile(flo, dtype=np.float32).reshape(S, -1)[len(ids) - 1]

    # argv order = program IN declaration order (builder emission;
    # glob order would silently misassign weights -- fail-loud by
    # construction here instead).
    order = ["x0", "pos", "cmask"]
    for L in range(28):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            order.append(f"{k}{L}")
    order += ["lnf", "wlog", "Ua27", "Vc27", "ones51"]
    live = {"x0": "step_x0.bin", "pos": "step_pos.bin",
            "cmask": "step_cm.bin"}
    outs = {}
    for tag, dose in (("zero", 0.0), ("one", 1.0)):
        if dose == 0.0:
            K = Ua.shape[1]
            shp = (K, HID)
            n = int(np.prod(shp))
            cur = np.fromfile(vc, dtype=np.float16)
            assert cur.size == n, (cur.size, n)
            np.zeros(n, dtype=np.float16).tofile(vc)
        else:
            shutil.copy(bak, vc)
        write_live()
        argv = [exe] + [os.path.join(WORK, live.get(nm, f"fr_{nm}.bin"))
                        for nm in order]
        for fn in argv[1:-2]:
            assert os.path.isfile(fn), f"missing argv file {fn}"
        argv += [os.path.join(WORK, "step_lg.bin"),
                 os.path.join(WORK, "step_ou.bin")]
        sv = ServedExe(exe, argv[1:], err_path=os.path.join(WORK, "serve.err"))
        sv.start()
        try:
            sv.step()
            outs[tag] = read_out()
        finally:
            sv.close()
    shutil.copy(bak, vc)
    dg = outs["one"] - outs["zero"]
    print(f"geo: |delta|={float(np.linalg.norm(dg)):.2f} "
          f"max|delta|={float(np.abs(dg).max()):.2f}", flush=True)
    print(f"ratio mirror/geo = {float(np.linalg.norm(dm)) / (float(np.linalg.norm(dg)) + 1e-12):.2f}x "
          f"corr = {float(np.corrcoef(dm, dg)[0, 1]):.3f}", flush=True)


if __name__ == "__main__":
    main()
