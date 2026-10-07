"""Catalog turn v1: three transferred facts live in one bank, listings.

Facts (all teacher-strong, in-our-vocab, our-weak): Italy->Rome,
Caesar->Rome (shared target, different address), Alexander->
Alexandria. Keys: teacher L2 rows via shared residual map (x2);
values: native readout dirs. One K131 bank through lm_bankhn.asm.
Gates per fact: target rank improves vs base; holds 16/16 vs base
tops; retrieval of own store. Random-3 control (floor).
Success (stated): >=2/3 facts improve with holds >=15/16.
Usage: python3 research/catalog_turn.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from xfer_map import build_resid_map, our_hn

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
FACTS = [
    ("Italy", "The capital of Italy is", 261, 98),
    ("Caesar", "Julius Caesar was assassinated in the city of", 40, 98),
    ("Alexander", "Alexander the Great founded the city of", 12, 59),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kscale", type=float, default=2.0)
    args = ap.parse_args()
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    import json
    _, tok7 = load7b()
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    Mr, minfo = build_resid_map()

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    # mine teacher L2 keys per fact (UNIT keys cached; scale applied at
    # bank build so one cache serves every kscale in the sweep).
    # Values native readout dirs.
    kcache = "/tmp/cat_keys.npz"
    keys = None
    if os.path.isfile(kcache):
        try:
            z = np.load(kcache)
            keys = [z[f"k{i}"] for i in range(len(FACTS))]
            print("keys: cached (unit)", flush=True)
        except (KeyError, ValueError):
            keys = None
    if keys is None:
        keys = []
        for name, prompt, xid, tid in FACTS:
            t, gids = fwdH(prompt, keep="all")
            toks = tok7.convert_ids_to_tokens(gids)
            frag = name[1:].lower()
            pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                       len(gids) - 1)
            k = t[2][pos]
            k /= np.linalg.norm(k)
            kk = (k @ Mr)
            keys.append(kk / np.linalg.norm(kk))
            print(f"{name}: key-pos={pos}/{len(gids)} target={inv.get(tid, '?')}",
                  flush=True)
        np.savez(kcache, **{f"k{i}": k for i, k in enumerate(keys)})
    vals = []
    for (name, prompt, xid, tid) in FACTS:
        vd = np.ascontiguousarray(wlog[:, tid])
        vals.append(vd / np.linalg.norm(vd) * evn)
    rng = np.random.default_rng(1)
    rks = [r / np.linalg.norm(r) for r in rng.normal(size=(3, 16))]
    rvs = [r / np.linalg.norm(r) * evn for r in rng.normal(size=(3, 16))]
    K128 = ukt0.shape[1]
    ukt3 = np.concatenate([ukt0] + [(k * args.kscale)[:, None] for k in keys],
                          axis=1)
    # native-key catalog control (emb keys held 16/16 single-fact; if this
    # holds at catalog scale while teacher keys don't, promiscuity comes
    # from map distortion, not the bank form)
    ekeys = []
    for xid in (261, 40, 12):
        e = np.ascontiguousarray(dE["emb"][xid])
        ekeys.append(e / np.linalg.norm(e))
    uktN = np.concatenate([ukt0] + [(k * args.kscale)[:, None] for k in ekeys],
                          axis=1)
    evbN = np.concatenate([evb0] + [(np.ascontiguousarray(wlog[:, tid])
                                     / np.linalg.norm(wlog[:, tid]) * evn * 8)[None, :]
                                    for tid in (98, 98, 59)], axis=0)
    evb3 = np.concatenate([evb0] + [v[None, :] for v in vals], axis=0)
    uktR = np.concatenate([ukt0] + [k[:, None] for k in rks], axis=1)
    evbR = np.concatenate([evb0] + [v[None, :] for v in rvs], axis=0)

    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
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

    # fact prompts in OUR vocab (unk-heavy allowed; baselines measured)
    fprompts = []
    for name, prompt, xid, tid in FACTS:
        ids = ids_of(prompt)
        fprompts.append((name, ids[-8:], tid))
        cov_ids = sum(1 for i in ids[-8:] if i != 0)
        print(f"{name}: prompt coverage {cov_ids}/{len(ids[-8:])} in-vocab",
              flush=True)
    cprompts = []
    for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    arms = {"base": (ukt0, evb0), "cat3": (ukt3, evb3), "rand3": (uktR, evbR),
            "nat3": (uktN, evbN)}
    base_tops = {}
    for ids, tgt in cprompts:
        f = run(ids, ukt0, evb0)
        base_tops[tuple(ids)] = int(dec(f["LOGITS"])[-1].argmax())
    for aname, (ukt, evb) in arms.items():
        fr, rets = [], []
        for name, ids, tid in fprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            fr.append(int((lg > lg[tid]).sum()) + 1)
            bp = dec(f["BP1"])[-1]
            rets.append(int(np.argmax(bp)))
        hold = 0
        broken = []
        for ids, tgt in cprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            top = int(lg.argmax())
            if top == base_tops[tuple(ids)]:
                hold += 1
            else:
                bp = dec(f["BP1"])[-1]
                broken.append((inv.get(tgt, "?"), inv.get(top, "?"),
                               int(np.argmax(bp))))
        print(f"{aname}: target-ranks {fr} "
              f"retrieves-own [{','.join(str(int(r >= K128)) for r in rets)}] "
              f"hold {hold}/{len(cprompts)}", flush=True)
        if broken:
            print(f"  broken holds (truth->got@store): {broken[:6]}",
                  flush=True)


if __name__ == "__main__":
    main()
