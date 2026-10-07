"""Behavior-fit keys: teacher match patterns, native vectors, no geometry map.

For each prompt: teacher pattern t_p = cos(teacherH26(p), teacherKey).
Solve native key k: min over prompts of (HN_p . k - s*t_p)^2 + ridge.
Teacher contributes LABELS (which prompt matches how strongly), never
vectors. Install k (+ native Rome value) as store 129; compare vs
native-emb-key bank and base. Prediction: behavior-fit retrieves as
well or better (matching-patterns are what retrieval IS; coordinates
were always a proxy). Falsifier: loses to mapped-vector keys (addrR).
Usage: python3 research/behavior_keys.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from xfer_map import our_hn


def main():
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    import json
    _, tok7 = load7b()
    dd = os.path.join(ROOT, "data")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines_all = open(os.path.join(dd, "lm_test.txt")).read().split("\n")
    lines = lines_all
    # fit set: ALL Italy contexts (positives) + generic prefixes. Positives
    # are load-bearing (a fit on negatives-only learns no discrimination;
    # measured once at pattern-range [0.18,0.34] -- never again).
    fit_prompts, is_pos = [], []
    for s in lines_all:
        ids = ids_of(s)
        if len(ids) < 4:
            continue
        hit = any(v == 261 for v in ids[:8])
        if hit or len(fit_prompts) - sum(is_pos) < 20:
            fit_prompts.append((s, ids[:4]))
            is_pos.append(hit)
    HNs, TPAT = [], []
    tI, _ = fwdH("The capital of Italy is")
    tIa, gids = fwdH("The capital of Italy is", keep="all")
    gtoks = tok7.convert_ids_to_tokens(gids)
    gpos = next((i for i, x in enumerate(gtoks) if "taly" in x.lower()),
                len(gids) - 1)
    keyT = tIa[2][gpos]
    keyT /= np.linalg.norm(keyT)
    for s, pre in fit_prompts:
        hn = our_hn(pre)
        t7, tids7 = fwdH(s, keep="all")
        kt = min(int(round(3 / max(len(ids_of(s)) - 1, 1) * (len(tids7) - 1))),
                 len(tids7) - 1)
        h7 = t7[2][kt]
        TPAT.append(float(h7 @ keyT / (np.linalg.norm(h7) + 1e-12)))
        HNs.append(hn[min(3, len(hn) - 1)])
    HNs = np.stack(HNs)
    TPAT = np.array(TPAT)
    print(f"fit prompts: {len(fit_prompts)} ({sum(is_pos)} Italy-positive); "
          f"teacher pattern range [{TPAT.min():.2f},{TPAT.max():.2f}]",
          flush=True)
    s = float(np.linalg.norm(HNs, axis=1).mean())
    best = None
    for ridge in (1e-1, 1e0, 1e1):
        k, _, _, _ = np.linalg.lstsq(
            np.vstack([HNs, np.sqrt(ridge) * np.eye(HNs.shape[1])]),
            np.concatenate([s * TPAT, np.zeros(HNs.shape[1])]), rcond=None)
        pred = HNs @ k
        r = float(np.corrcoef(pred, TPAT)[0, 1])
        print(f"ridge {ridge:.0e}: pattern-corr={r:.3f}", flush=True)
        if best is None or r > best[0]:
            best = (r, k.copy())
    print(f"best pattern-corr={best[0]:.3f} (1.0 = teacher pattern "
          f"reproduced in our space)", flush=True)
    kB = best[1]
    kB = kB / np.linalg.norm(kB) * 2.0  # winner scale from key-rank gate
    # rank-gate (dots only): behavior key vs native emb key vs mapped key
    z = np.load("/tmp/t_xfer.npz")
    from xfer_map import build_resid_map
    Mr, _ = build_resid_map(verbose=False)
    kR = (z["k"] @ Mr)
    kR = kR / np.linalg.norm(kR) * 2.0
    emb261 = np.ascontiguousarray(dE["emb"][261])
    emb261 /= np.linalg.norm(emb261)
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    rome_dir = wlog[:, 98] / np.linalg.norm(wlog[:, 98]) * evn
    nq = 0
    ranks = {name: [] for name in ("behav", "native", "mapped")}
    for s in lines:
        ids = ids_of(s)
        for k, v in enumerate(ids):
            if v != 261 or k < 1 or nq >= 4:
                continue
            hn = our_hn(ids[max(0, k - 7):k + 1])[-1]
            dots = hn @ ukt0
            ranks["behav"].append(int((dots > hn @ kB).sum()) + 1)
            ranks["native"].append(int((dots > hn @ emb261).sum()) + 1)
            ranks["mapped"].append(int((dots > hn @ kR).sum()) + 1)
            nq += 1
    for name, rr in ranks.items():
        print(f"rank-gate {name}: {rr} (top1 needed)", flush=True)
    # install winner + native control, score Rome + holds
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    sdir = os.path.join(ROOT, "programs")
    text = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
            + open(os.path.join(sdir, "lm_bankhn.asm")).read())

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

    inv = {i: w for w, i in vocab.items()}
    iprompts = []
    for s in lines:
        ids = ids_of(s)
        for k, v in enumerate(ids):
            if v == 261 and k >= 1:
                iprompts.append(ids[max(0, k - 7):k + 1])
    cprompts = []
    for s in lines[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    arms = {
        "base": (ukt0, evb0),
        "native": (np.concatenate([ukt0, emb261[:, None]], axis=1),
                   np.concatenate([evb0, rome_dir[None, :]], axis=0)),
        "behav": (np.concatenate([ukt0, kB[:, None]], axis=1),
                  np.concatenate([evb0, rome_dir[None, :]], axis=0)),
    }
    K1 = ukt0.shape[1]
    base_tops = {}
    for ids, tgt in cprompts:
        f = run(ids, ukt0, evb0)
        base_tops[tuple(ids)] = int(dec(f["LOGITS"])[-1].argmax())
    for name, (ukt, evb) in arms.items():
        rr, tops, rets = [], [], []
        for ids in iprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            rr.append(int((lg > lg[98]).sum()) + 1)
            tops.append(int(lg.argmax()))
            bp = dec(f["BP1"])[-1]
            rets.append(int(np.argmax(bp)))
        hold = sum(int(dec(run(ids, ukt, evb)["LOGITS"])[-1].argmax())
                   == base_tops[tuple(ids)] for ids, tgt in cprompts)
        print(f"{name:7} Rome-rank {np.median(rr):.1f} "
              f"tops {[inv.get(t, '?') for t in tops][:4]} "
              f"retrieves-store{K1}: {sum(r == K1 for r in rets)}/{len(rets)} "
              f"hold {hold}/{len(cprompts)}", flush=True)


if __name__ == "__main__":
    main()
