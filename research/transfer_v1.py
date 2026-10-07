"""Transfer v1: teacher Italy knowledge -> our D16 bankhn store.

First point off-teacher: mine teacher Italy key (L2) + direction (L27
vs-mean), random-project (JL, seed-0, no parallel data, no cheating)
into our 16-dim space, install as 129th bankhn store, score Rome-rank
on in-domain Italy prompts + containment. Arms separate address
transfer (teacher key, native value) from full transfer (both
projected) with native-path and random controls.
Success (stated upfront, research screen): Rome rank improves >=3 on
majority of Italy prompts, containment >=15/16, addrT/fullT beat
random, native calibrates the path.
Usage: python3 research/transfer_v1.py [--mine-only]
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
CACHE = "/tmp/t_xfer.npz"


def mine():
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    _, tok = load7b()
    REF = ["France", "Germany", "Spain", "China", "Japan"]
    tI, _ = fwdH("The capital of Italy is")
    tO = [fwdH(f"The capital of {c} is") for c in REF]
    d = tI[27] - np.mean([t[0][27] for t in tO], axis=0)
    d /= np.linalg.norm(d)
    tIa, ids = fwdH("The capital of Italy is", keep="all")
    toks = tok.convert_ids_to_tokens(ids)
    pos = next((i for i, x in enumerate(toks) if "taly" in x.lower()),
               len(ids) - 1)
    k = tIa[2][pos]
    k /= np.linalg.norm(k)
    np.savez(CACHE, d=d, k=k)
    print("mined teacher Italy dir+key", flush=True)


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    if not os.path.isfile(CACHE):
        mine()
    z = np.load(CACHE)
    dT, kT = z["d"], z["k"]
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    # ukt (D,K): keys are COLUMNS; evb (K,D): values are ROWS.
    assert ukt0.shape == (16, 128) and evb0.shape == (128, 16), \
        f"bank {ukt0.shape}/{evb0.shape}"
    wlog = np.ascontiguousarray(dE["wlog"])
    assert wlog.shape == (16, 513), f"wlog {wlog.shape}"
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt, evb):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                          "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                          "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                          "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                          "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                          "wlog": enc(wlog),
                          "ukt": enc(ukt), "evb": enc(evb)},
                         sigs=SIGS, basedir=sdir)
        return f

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    # Italy prompts: prefixes ending at italy(261), last 8 toks (S<=8)
    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")
    iprompts = []
    for s in lines:
        ids = ids_of(s)
        for k, v in enumerate(ids):
            if v == 261 and k >= 1:
                iprompts.append(ids[max(0, k - 7):k + 1])
    print(f"Italy prompts: {len(iprompts)}", flush=True)
    # containment: first 16 predictable positions (k>=1) from lines
    cprompts = []
    for s in lines[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break

    rng = np.random.default_rng(0)
    # residual-anchored map via the shared helper (xfer_map): teacher L2
    # rows -> our HN rows, same texts both models. Replaces the emb-
    # anchored map (dead-last retrieval: out-of-domain application).
    from xfer_map import build_resid_map
    Mr, minfo = build_resid_map()
    kf = kT @ Mr
    kf = kf / np.linalg.norm(kf) * 2.0  # x2 scale wins the key-rank gate;
    # do NOT renormalize to unit (scale is the point, not the convention)
    df = dT @ Mr
    # V2 (doll theory): normalize anchors FIRST, then contrast. For a
    # linear map this differs from V1 (normalize-after) -- and only
    # then is the comparison non-vacuous. Same Italy-vs-mean set.
    sys.path.insert(0, os.path.join(ROOT, "research"))
    from qwen_torch import fwdH as _fwdH
    REF = ["France", "Germany", "Spain", "China", "Japan"]
    tI2, _ = _fwdH("The capital of Italy is")
    tO2 = [_fwdH(f"The capital of {c} is") for c in REF]
    u = lambda v: v / (np.linalg.norm(v) + 1e-12)
    df2 = u(tI2[27] @ Mr) - np.mean([u(t[0][27] @ Mr) for t in tO2], axis=0)
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    rome_dir = np.ascontiguousarray(wlog[:, 98])
    rome_dir = rome_dir / np.linalg.norm(rome_dir) * evn
    emb261 = np.ascontiguousarray(dE["emb"][261])
    emb261 /= np.linalg.norm(emb261)
    rk = rng.normal(size=16)
    rk /= np.linalg.norm(rk)
    rv = rng.normal(size=16)
    rv = rv / np.linalg.norm(rv) * evn
    arms = {
        "base": (ukt0, evb0),
        "native": (np.concatenate([ukt0, emb261[:, None]], axis=1),
                   np.concatenate([evb0, rome_dir[None, :]], axis=0)),
        "native4x": (np.concatenate([ukt0, emb261[:, None]], axis=1),
                     np.concatenate([evb0, (4 * rome_dir)[None, :]], axis=0)),
        "native8x": (np.concatenate([ukt0, emb261[:, None]], axis=1),
                     np.concatenate([evb0, (8 * rome_dir)[None, :]], axis=0)),
        "addrR": (np.concatenate([ukt0, kf[:, None]], axis=1),
                  np.concatenate([evb0, rome_dir[None, :]], axis=0)),
        "fullR": (np.concatenate([ukt0, kf[:, None]], axis=1),
                  np.concatenate([evb0, (df / np.linalg.norm(df) * evn)[None, :]], axis=0)),
        "fullR2": (np.concatenate([ukt0, kf[:, None]], axis=1),
                   np.concatenate([evb0, (df2 / np.linalg.norm(df2) * evn)[None, :]], axis=0)),
        "random": (np.concatenate([ukt0, rk[:, None]], axis=1),
                   np.concatenate([evb0, rv[None, :]], axis=0)),
    }
    K1 = ukt0.shape[1]
    # containment = agreement with BASE tops (hold-rate), not accuracy.
    base_tops = {}
    for ids, tgt in cprompts:
        f = run(ids, ukt0, evb0)
        lg = dec(f["LOGITS"])[-1]
        base_tops[tuple(ids)] = int(lg.argmax())
    for name, (ukt, evb) in arms.items():
        rr, tops, rets = [], [], []
        for ids in iprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            rr.append(int((lg > lg[98]).sum()) + 1)
            tops.append(int(lg.argmax()))
            bp = dec(f["BP1"])[-1]
            rets.append(int(np.argmax(bp)))
        hold = 0
        for ids, tgt in cprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            hold += int(lg.argmax()) == base_tops[tuple(ids)]
        print(f"{name:7} Rome-rank {np.median(rr):.1f} "
              f"(min {min(rr)}) tops {[inv.get(t, '?') for t in tops][:4]} "
              f"retrieves-store{K1}: {sum(r == K1 for r in rets)}/{len(rets)} "
              f"hold {hold}/{len(cprompts)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-only", action="store_true")
    args = ap.parse_args()
    if args.mine_only:
        mine()
    else:
        main()
