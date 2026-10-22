"""Phase-B basin-supervised encoder (analysis, numpy+PIL CPU, no emission).

Same tiny mixer as train_encoder (body init bit-identical: same order,
same seed) PLUS head linear 128->K* (+bias). L = triplet(m=0.5) +
0.1*MSE + 1.0*CE(head, banked basin labels; 8 Hypersim only — fixtures
have no labels and carry no CE term). Weights OUT-OF-TREE
(~/student_runs/basin_encoder/); metrics + 11 codes banked in
samples/marigold/. Deterministic (seed 7).
Usage: python3 scripts/train_basin_encoder.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTDIR = os.path.expanduser("~/student_runs/basin_encoder")

SEED = 7
MARGIN, LAM, MU, LR, STEPS = 0.5, 0.1, 1.0, 0.01, 2000


def main():
    import train_encoder as E
    import hopfield_retrieval as R
    t0 = time.time()
    gt = json.load(open(os.path.join(FIXDIR, "basin_ground_truth.json")))
    KSTAR = int(gt["winner_k"])
    paths, scenes = E.load_images()
    thumbs = np.array([E.thumb(p) for p in paths])
    X = np.array([E.patchify(t) for t in thumbs])
    frames8 = [f"{s}_fr{fr}" for s, fr in
               __import__("teacher_lesion").eval_frames()]
    y = np.array([gt["labels"][f] for f in frames8])
    assert set(y) == set(range(KSTAR)), (set(y), KSTAR)
    print(f"K*={KSTAR} labels={dict(zip(frames8, y))}", flush=True)
    T3 = []
    for a in range(8):
        sib = [i for i in range(8) if scenes[i] == scenes[a] and i != a]
        assert len(sib) == 1, (a, sib)
        for n in [i for i in range(11) if i != a and i != sib[0]]:
            T3.append((a, sib[0], n))
    T3 = np.array(T3)
    rng = np.random.RandomState(SEED)
    W = {"proj": E.init(rng, E.PDIM, E.HDIM),
         "bproj": np.zeros(E.HDIM, np.float32)}
    for b in ("b1", "b2"):
        W[f"W1t{b}"] = E.init(rng, 24, E.NPATCH)
        W[f"W2t{b}"] = E.init(rng, E.NPATCH, 24)
        W[f"Wc1{b}"] = E.init(rng, E.HDIM, 192)
        W[f"bc1{b}"] = np.zeros(192, np.float32)
        W[f"Wc2{b}"] = E.init(rng, 192, E.HDIM)
        W[f"bc2{b}"] = np.zeros(E.HDIM, np.float32)
    W["head"] = E.init(rng, E.NPATCH * E.HDIM, E.CODE)
    W["bhead"] = np.zeros(E.CODE, np.float32)
    W["dec"] = E.init(rng, E.CODE, E.TW * E.TH * 3)
    W["bdec"] = np.zeros(E.TW * E.TH * 3, np.float32)
    W["bh"] = E.init(rng, E.CODE, KSTAR)
    W["bbh"] = np.zeros(KSTAR, np.float32)
    nT, B = len(T3), len(X)
    curve = {}
    for it in range(1, STEPS + 1):
        (ltrip, lrec, _), g = E.grads(X, T3, thumbs.reshape(B, -1), W)
        zn, _, _ = E.forward(X, W)
        logits = zn[:8] @ W["bh"] + W["bbh"]
        logits = logits - logits.max(axis=1, keepdims=True)
        pe = np.exp(logits)
        pe = pe / pe.sum(axis=1, keepdims=True)
        lce = float(-np.log(pe[np.arange(8), y] + 1e-12).mean())
        dlog = pe.copy()
        dlog[np.arange(8), y] -= 1.0
        dlog /= 8
        g["bh"] = zn[:8].T @ dlog
        g["bbh"] = dlog.sum(axis=0)
        dznh = np.zeros_like(zn)
        dznh[:8] = dlog @ W["bh"].T
        _, recon, cache = E.forward(X, W)
        drc = 2.0 * (recon - thumbs.reshape(B, -1)) / thumbs.reshape(B, -1).size
        S = zn @ zn.T
        d = MARGIN + S[T3[:, 0], T3[:, 2]] - S[T3[:, 0], T3[:, 1]]
        viol = d > 0
        dS = np.zeros_like(S)
        np.add.at(dS, (T3[viol, 0], T3[viol, 2]), 1.0 / nT)
        np.add.at(dS, (T3[viol, 0], T3[viol, 1]), -1.0 / nT)
        dzn = (dS + dS.T) @ zn + (drc @ W["dec"].T) * LAM + dznh * MU
        znn = np.linalg.norm(cache["zpre"], axis=1, keepdims=True)
        dz = (dzn - zn * (dzn * zn).sum(axis=1, keepdims=True)) / znn
        g["head"] = cache["flat"].T @ dz
        g["bhead"] = dz.sum(axis=0)
        dH = (dz @ W["head"].T).reshape(B, E.NPATCH, E.HDIM)
        for b in ("b2", "b1"):
            Apre, Cpre = cache[b]
            H_in = cache["inputs"][0 if b == "b1" else 1]
            H_mid = H_in + W[f"W2t{b}"] @ E.silu(Apre)
            dC2 = dH
            g[f"Wc2{b}"] = np.einsum("btn,btd->nd", E.silu(Cpre), dC2)
            g[f"bc2{b}"] = dC2.sum(axis=(0, 1))
            dC1 = np.einsum("btd,nd->btn", dC2, W[f"Wc2{b}"]) * E.dsilu(Cpre)
            g[f"Wc1{b}"] = np.einsum("btn,btd->nd", H_mid, dC1)
            g[f"bc1{b}"] = dC1.sum(axis=(0, 1))
            dHmid = dH + np.einsum("btd,nd->btn", dC1, W[f"Wc1{b}"])
            dB_ = dHmid
            g[f"W2t{b}"] = np.einsum("btn,bsn->ts", dB_, E.silu(Apre))
            dA = np.einsum("ts,btn->bsn", W[f"W2t{b}"], dB_) * E.dsilu(Apre)
            g[f"W1t{b}"] = np.einsum("bsn,bun->su", dA, H_in)
            dH = dHmid + np.einsum("su,bsn->bun", W[f"W1t{b}"], dA)
        g["proj"] = np.einsum("btn,btd->dn", dH, X)
        g["bproj"] = dH.sum(axis=(0, 1))
        tot = float(sum(float((v ** 2).sum()) for v in g.values()) ** 0.5)
        sc = min(1.0, 1.0 / max(tot, 1e-12))
        for k in W:
            W[k] = W[k] - LR * sc * g[k].astype(np.float32)
        if it in (1, 200, 500, 1000, 1500, 2000):
            curve[str(it)] = {"trip": ltrip, "recon": lrec, "ce": lce,
                              "gnorm": tot}
            print(f"step {it}: trip={ltrip:.4f} recon={lrec:.4f} "
                  f"ce={lce:.4f} gnorm={tot:.3f}", flush=True)
    vals = [curve[k]["trip"] + curve[k]["ce"] for k in curve]
    if any(v != v for v in vals) or max(vals) > 1e9:
        raise SystemExit("NaN/divergence (complete-empty, stated)")
    os.makedirs(OUTDIR, exist_ok=True)
    wp = os.path.join(OUTDIR, "enc.npz")
    np.savez(wp, **W)
    sha = hashlib.sha256(open(wp, "rb").read()).hexdigest()
    codes, _, _ = E.forward(X, W)
    det = float(np.abs(E.forward(X[:1], W)[0] - E.forward(X[:1], W)[0]).max())
    np.save(os.path.join(FIXDIR, "basin_codes.npy"), codes.astype(np.float32))
    pred = np.argmax(codes[:8] @ W["bh"] + W["bbh"], axis=1)
    acc = float((pred == y).mean())
    bar = 2.0 / KSTAR
    print(f"assignment: {int((pred == y).sum())}/8 = {acc:.4f} "
          f"(bar {bar:.4f} = 2/{KSTAR})", flush=True)
    H8 = codes[:8] / np.linalg.norm(codes[:8], axis=1, keepdims=True)
    G = H8 @ H8.T
    iu = np.triu_indices(8, 1)
    mx, mn = float(G[iu].max()), float(G[iu].mean())
    print(f"separability: maxcos={mx:.4f} meancos={mn:+.4f}", flush=True)
    battery = {}
    if acc >= bar:
        hit = tot = 0
        lv = {}
        for li, f in enumerate(R.LEVELS):
            h = 0
            for fi in range(8):
                for s in R.SEEDS:
                    w, _, _ = R.retrieve(H8, R.corrupt(H8[fi], f, s, li))
                    h += (fi == w)
                    tot += 1
            lv[str(f)] = {"hit": h, "tot": 40}
            hit += h
        bacc = hit / tot
        battery = {"levels": lv, "accuracy": bacc, "trials": tot,
                   "chance": R.CHANCE, "bar": R.BAR,
                   "verdict": ("alive" if bacc >= R.BAR else "dead")}
        print(f"battery: acc={bacc:.4f} verdict={battery['verdict']}",
              flush=True)
    else:
        print("assignment below bar: NO battery runs (stated)", flush=True)
    json.dump({"Kstar": KSTAR, "bar": bar, "mu": MU,
               "assignment": {"pred": pred.tolist(), "true": y.tolist(),
                              "accuracy": acc},
               "separability": {"maxcos": mx, "meancos": mn},
               "determinism_maxabs": det, "battery": battery,
               "wall_s": round(time.time() - t0, 1),
               "loss_curve": curve, "W_path": wp, "W_sha256": sha},
              open(os.path.join(FIXDIR, "basin_supervision.json"), "w"),
              indent=1)
    print(f"wall {time.time() - t0:.1f}s; weights out-of-tree; banked",
          flush=True)


if __name__ == "__main__":
    main()
