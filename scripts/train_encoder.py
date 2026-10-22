"""Tiny-mixer encoder training (analysis, numpy+PIL CPU, no emission).

96x72 thumbs -> 12x12 patches (48x432) -> linear 432->96 -> 2x mixer
(token-MLP 48->24->48 + channel-MLP 96->192->96, SILU, residuals) ->
flatten -> linear 4608->128 -> L2 code; linear 128->20736 decoder.
L = triplet-margin(m=0.5, same-scene positives) + 0.1 * MSE recon.
All math MATMUL/ADD/SUB/MUL/SILU + host-side Python (see constitution
for per-op cites). Weights OUT-OF-TREE (~/student_runs/encoder/);
metrics + configs + 11 codes banked in samples/marigold/.
Deterministic (seed 7). Usage: python3 scripts/train_encoder.py
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTDIR = os.path.expanduser("~/student_runs/encoder")

SEED = 7
TW, TH, PS = 96, 72, 12
NPATCH, PDIM, HDIM = 48, 432, 96
CODE = 128
MARGIN, LAM, LR, STEPS = 0.5, 0.1, 0.01, 2000


def load_images():
    import teacher_lesion as T
    paths, scenes = [], []
    for s, fr in T.eval_frames():
        c = sorted(glob.glob(os.path.join(T.SHARD, s, f"rgb_cam_00_fr{fr}.png")))
        assert len(c) == 1, (s, fr)
        paths.append(c[0])
        scenes.append(s)
    for n in ("church.jpg", "dogs.png", "squirrel.jpg"):
        paths.append(os.path.join(FIXDIR, "inputs", n))
        scenes.append("fixture")
    return paths, scenes


def thumb(p):
    im = Image.open(p).convert("RGB").resize((TW, TH), Image.LANCZOS)
    return (np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0)


def patchify(t):
    h, w, _ = t.shape
    P = t.reshape(h // PS, PS, w // PS, PS, 3).transpose(0, 2, 1, 3, 4)
    return P.reshape(-1, PS * PS * 3)


def silu(x):
    return x / (1.0 + np.exp(-x))


def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def init(rng, *shape):
    return (rng.randn(*shape) / np.sqrt(shape[0])).astype(np.float32)


def forward(Pb, W):
    H = Pb @ W["proj"] + W["bproj"]
    cache = {"inputs": [H]}
    for b in ("b1", "b2"):
        Apre = W[f"W1t{b}"] @ H
        A = silu(Apre)
        H = H + W[f"W2t{b}"] @ A
        Cpre = H @ W[f"Wc1{b}"] + W[f"bc1{b}"]
        C1 = silu(Cpre)
        H = H + C1 @ W[f"Wc2{b}"] + W[f"bc2{b}"]
        cache["inputs"].append(H)
        cache[b] = (Apre, Cpre)
    flat = H.reshape(len(H), -1)
    zpre = flat @ W["head"] + W["bhead"]
    zn = zpre / np.linalg.norm(zpre, axis=1, keepdims=True)
    recon = zn @ W["dec"] + W["bdec"]
    cache.update({"H": H, "flat": flat, "zpre": zpre})
    return zn, recon, cache


def grads(X, T3, thumbs, W):
    nT, B = len(T3), len(X)
    zn, recon, cache = forward(X, W)
    S = zn @ zn.T
    d = MARGIN + S[T3[:, 0], T3[:, 2]] - S[T3[:, 0], T3[:, 1]]
    viol = d > 0
    ltrip = float(d[viol].sum() / nT)
    lrec = float(((recon - thumbs) ** 2).mean())
    dS = np.zeros_like(S)
    np.add.at(dS, (T3[viol, 0], T3[viol, 2]), 1.0 / nT)
    np.add.at(dS, (T3[viol, 0], T3[viol, 1]), -1.0 / nT)
    dzn = (dS + dS.T) @ zn
    drc = 2.0 * (recon - thumbs) / thumbs.size
    dzn = dzn + (drc @ W["dec"].T) * LAM
    g = {"dec": (zn.T @ drc) * LAM, "bdec": drc.sum(axis=0) * LAM}
    znn = np.linalg.norm(cache["zpre"], axis=1, keepdims=True)
    dz = (dzn - zn * (dzn * zn).sum(axis=1, keepdims=True)) / znn
    g["head"] = cache["flat"].T @ dz
    g["bhead"] = dz.sum(axis=0)
    dH = (dz @ W["head"].T).reshape(B, NPATCH, HDIM)
    for b in ("b2", "b1"):
        Apre, Cpre = cache[b]
        H_in = cache["inputs"][0 if b == "b1" else 1]
        H_mid = H_in + W[f"W2t{b}"] @ silu(Apre)
        dC2 = dH
        g[f"Wc2{b}"] = np.einsum("btn,btd->nd", silu(Cpre), dC2)
        g[f"bc2{b}"] = dC2.sum(axis=(0, 1))
        dC1 = np.einsum("btd,nd->btn", dC2, W[f"Wc2{b}"]) * dsilu(Cpre)
        g[f"Wc1{b}"] = np.einsum("btn,btd->nd", H_mid, dC1)
        g[f"bc1{b}"] = dC1.sum(axis=(0, 1))
        dHmid = dH + np.einsum("btd,nd->btn", dC1, W[f"Wc1{b}"])
        dB_ = dHmid
        g[f"W2t{b}"] = np.einsum("btn,bsn->ts", dB_, silu(Apre))
        dA = np.einsum("ts,btn->bsn", W[f"W2t{b}"], dB_) * dsilu(Apre)
        g[f"W1t{b}"] = np.einsum("bsn,bun->su", dA, H_in)
        dH = dHmid + np.einsum("su,bsn->bun", W[f"W1t{b}"], dA)
    g["proj"] = np.einsum("btn,btd->dn", dH, X)
    g["bproj"] = dH.sum(axis=(0, 1))
    tot = float(sum(float((v ** 2).sum()) for v in g.values()) ** 0.5)
    return (ltrip, lrec, tot), g


def train(X, T3, thumbs, steps=STEPS, lr=LR):
    rng = np.random.RandomState(SEED)
    W = {"proj": init(rng, PDIM, HDIM), "bproj": np.zeros(HDIM, np.float32)}
    for b in ("b1", "b2"):
        W[f"W1t{b}"] = init(rng, 24, NPATCH)
        W[f"W2t{b}"] = init(rng, NPATCH, 24)
        W[f"Wc1{b}"] = init(rng, HDIM, 192)
        W[f"bc1{b}"] = np.zeros(192, np.float32)
        W[f"Wc2{b}"] = init(rng, 192, HDIM)
        W[f"bc2{b}"] = np.zeros(HDIM, np.float32)
    W["head"] = init(rng, NPATCH * HDIM, CODE)
    W["bhead"] = np.zeros(CODE, np.float32)
    W["dec"] = init(rng, CODE, TW * TH * 3)
    W["bdec"] = np.zeros(TW * TH * 3, np.float32)
    nT, B = len(T3), len(X)
    curve = {}
    for it in range(1, steps + 1):
        (ltrip, lrec, tot), g = grads(X, T3, thumbs, W)
        sc = min(1.0, 1.0 / max(tot, 1e-12))
        for k in W:
            W[k] = W[k] - lr * sc * g[k].astype(np.float32)
        if it in (1, 200, 500, 1000, 1500, 2000):
            curve[str(it)] = {"trip": ltrip, "recon": lrec, "gnorm": tot}
            print(f"step {it}: trip={ltrip:.4f} recon={lrec:.4f} "
                  f"gnorm={tot:.3f}", flush=True)
    return W, curve


def main():
    import hopfield_retrieval as R
    t0 = time.time()
    paths, scenes = load_images()
    thumbs = np.array([thumb(p) for p in paths])
    X = np.array([patchify(t) for t in thumbs])
    print(f"images={len(paths)} thumbs {thumbs.shape}", flush=True)
    T3 = []
    for a in range(8):
        sib = [i for i in range(8) if scenes[i] == scenes[a] and i != a]
        assert len(sib) == 1, (a, sib)
        for n in [i for i in range(11) if i != a and i != sib[0]]:
            T3.append((a, sib[0], n))
    T3 = np.array(T3)
    print(f"triplets={len(T3)} (8 anchors x 1 pos x 9 neg)", flush=True)
    W, curve = train(X, T3, thumbs.reshape(11, -1))
    trips = [curve[k]["trip"] for k in curve]
    if any(v != v for v in trips) or max(trips) > 1e9:
        raise SystemExit("NaN/divergence (complete-empty, stated)")
    os.makedirs(OUTDIR, exist_ok=True)
    wp = os.path.join(OUTDIR, "enc.npz")
    np.savez(wp, **W)
    sha = hashlib.sha256(open(wp, "rb").read()).hexdigest()
    codes, _, _ = forward(X, W)
    det = float(np.abs(forward(X[:1], W)[0] - forward(X[:1], W)[0]).max())
    np.save(os.path.join(FIXDIR, "encoder_codes.npy"),
            codes.astype(np.float32))
    H8 = codes[:8] / np.linalg.norm(codes[:8], axis=1, keepdims=True)
    G = H8 @ H8.T
    iu = np.triu_indices(8, 1)
    mx, mn = float(G[iu].max()), float(G[iu].mean())
    clears = mx < 0.9 and mn < 0.5
    print(f"8-code separability: maxcos={mx:.4f} meancos={mn:+.4f} "
          f"{'CLEARS' if clears else 'fails'}", flush=True)
    battery = {}
    if clears:
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
        acc = hit / tot
        battery = {"levels": lv, "accuracy": acc, "trials": tot,
                   "chance": R.CHANCE, "bar": R.BAR,
                   "verdict": ("alive" if acc >= R.BAR else "dead")}
        print(f"battery: acc={acc:.4f} verdict={battery['verdict']}",
              flush=True)
    else:
        print("separability fails: NO battery runs (stated)", flush=True)
    json.dump({"seed": SEED, "dims": {"thumb": [TW, TH], "patch": PS,
                                      "hidden": HDIM, "code": CODE},
               "objective": "triplet(m=0.5)+0.1*MSE", "lr": LR,
               "steps": STEPS, "images": paths, "scenes": scenes,
               "wall_s": round(time.time() - t0, 1),
               "loss_curve": curve, "W_path": wp, "W_sha256": sha,
               "separability": {"maxcos": mx, "meancos": mn,
                                "clears": bool(clears)},
               "determinism_maxabs": det, "battery": battery},
              open(os.path.join(FIXDIR, "encoder_separation.json"), "w"),
              indent=1)
    print(f"wall {time.time() - t0:.1f}s; weights out-of-tree; banked",
          flush=True)


if __name__ == "__main__":
    main()
