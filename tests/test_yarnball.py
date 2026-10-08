"""MVYB gate: the yarn ball as a runnable structure (retrieve+hold+install).

One listing (programs/lm_yarnball.asm) through stdlib/yarnball.asm's
yarnball_apply proves the generic ball is real structure, not prose:
  parity   lm_yarnball == lm_bankhn bit-exact (CALL is macro expansion
           of the identical four ops; 0-diff, wrappers add NOTHING)
  retrieve Italy prompt retrieves its own store (argmax P == K0)
  install  Rome-rank 300-class -> catalog-class (nat3 recipe: native
           emb key x2 + Rome readout value at 8x, now via the DEF)
  hold     every stable-top battery prompt (margin>=1 bar) holds
Gates: parity exact; retrieve own-store; install improves >=4x AND
reaches <=64; holds 100% of stable (razor flips reported, not gated).
Usage: python3 tests/test_yarnball.py (slow: ~40 listing runs)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.engram import yarnball_bank

BAR_DB = 40.0
FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
MARGIN_BAR = 1.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    K0 = ukt0.shape[1]
    wlog = np.ascontiguousarray(d["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    t_bank = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()
    t_ball = CFG + open(os.path.join(sdir, "lm_yarnball.asm")).read()

    def run(text, ids, ukt, evb):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(d["emb"]), "wq": enc(d["wq"]),
                             "wk": enc(d["wk"]), "wv": enc(d["wv"]),
                             "wo": enc(d["wo"]), "wup": enc(d["wup"]),
                             "wgate": enc(d["wgate"]), "wdown": enc(d["wdown"]),
                             "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                             "wlog": enc(wlog),
                             "ukt": enc(ukt), "evb": enc(evb)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    def pstream(f):
        cands = [k for k in f if "yarnball_apply" in k and k.endswith(".P")]
        assert len(cands) == 1, f"want one yarnball P stream, got {cands}"
        return cands[0]

    # -- parity: same data, named structure vs inline ops, bit-exact --
    probes = [[12, 471, 59], [59, 38, 471, 11, 12]]
    maxd = 0.0
    for ids in probes:
        fa = run(t_bank, ids, ukt0, evb0)
        fb = run(t_ball, ids, ukt0, evb0)
        la, lb = dec(fa["LOGITS"])[-1], dec(fb["LOGITS"])[-1]
        maxd = max(maxd, float(np.abs(la - lb).max()))
    check("yarnball-parity", maxd == 0.0, f"max|dlogit|={maxd:.3g} (0-diff)")

    # -- install data: nat3 recipe (best single-fact row anywhere) --
    e261 = np.ascontiguousarray(d["emb"][261])
    e261 /= np.linalg.norm(e261)
    rome = np.ascontiguousarray(wlog[:, 98])
    rome = rome / np.linalg.norm(rome) * evn * 8.0
    Ua, Vc, ledger = yarnball_bank(ukt0, evb0, [
        {"key": e261, "value": rome / (8.0 * evn), "dose": 8.0 * evn,
         "tier": "assoc", "support": "lm_test Italy->Rome native"}],
        key_scale=8.0)
    check("yarnball-ledger", len(ledger) == K0 + 1 and ledger[-1]["tier"] == "assoc",
          f"{len(ledger)} rows, last={ledger[-1]}")

    iprompt = ids_of("The capital of Italy is")[-8:]
    fbase = run(t_ball, iprompt, ukt0, evb0)
    lbase = dec(fbase["LOGITS"])[-1]
    rbase = int((lbase > lbase[98]).sum()) + 1
    finst = run(t_ball, iprompt, Ua, Vc)
    linst = dec(finst["LOGITS"])[-1]
    rinst = int((linst > linst[98]).sum()) + 1
    ret = int(dec(finst[pstream(finst)])[-1].argmax())
    check("yarnball-retrieve", ret == K0, f"retrieved store {ret} (own={K0})")
    check("yarnball-install", rinst <= 64 and rinst * 4 <= max(rbase, 1),
          f"Rome-rank {rbase}->{rinst} top={inv.get(int(linst.argmax()), '?')}")
    # dot-reader grade alongside argmax (norm-map doctrine: argmax alone
    # understates installs; report both, gate on argmax)
    h4 = dec(finst["H4"])[-1]
    hn = h4 / np.linalg.norm(h4)
    print(f"  dot-reader: Rome-cos={float(hn @ wlog[:, 98] / np.linalg.norm(wlog[:, 98])):.3f} "
          f"vs top-cos={float(hn @ wlog[:, int(linst.argmax())] / np.linalg.norm(wlog[:, int(linst.argmax())])):.3f}",
          flush=True)

    # -- holds: stable-top battery (margin>=1), razor reported separately --
    cprompts = []
    for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    stable, razor, broken_stable, broken_razor = [], [], [], []
    for ids, tgt in cprompts:
        lg = dec(run(t_ball, ids, ukt0, evb0)["LOGITS"])[-1]
        o = np.argsort(-lg)
        (stable if float(lg[o[0]] - lg[o[1]]) >= MARGIN_BAR else razor).append((ids, int(lg.argmax())))
    for ids, top0 in stable:
        top1 = int(dec(run(t_ball, ids, Ua, Vc)["LOGITS"])[-1].argmax())
        if top1 != top0:
            broken_stable.append((inv.get(top0, "?"), inv.get(top1, "?")))
    for ids, top0 in razor:
        top1 = int(dec(run(t_ball, ids, Ua, Vc)["LOGITS"])[-1].argmax())
        if top1 != top0:
            broken_razor.append((inv.get(top0, "?"), inv.get(top1, "?")))
    check("yarnball-hold", not broken_stable,
          f"{len(stable) - len(broken_stable)}/{len(stable)} stable hold "
          f"({len(razor)} razor, {len(broken_razor)} flipped as predicted-fragile)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
