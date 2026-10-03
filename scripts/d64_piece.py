"""D64 piece refit battery driver (mission tooling, NOT a gate). Part 2: CLI."""

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d64_lib import (DD, SD, WIN, bodies, dec, encode, load_bpe,
                     load_probe, mirror, par_db, run_ids, twin_num)

CFG_P = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
CFG_H = CFG_P + "CONFIG beta_b 0.25\n"


def listing(name, headt):
    base = open(os.path.join(SD, name)).read()
    return (CFG_H if headt else CFG_P) + base


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - m) / d, 4), round((c + m) / d, 4)


def mcnemar(a, b):
    A = np.array([p == t for p, t in a])
    B = np.array([p == t for p, t in b])
    n01 = int(((A == 1) & (B == 0)).sum())
    n10 = int(((A == 0) & (B == 1)).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return n01, n10, round(chi, 2), round(math.erfc(math.sqrt(chi / 2)), 4)


def top1(E_, W_, b_, text, test, vocab, rank, tag):
    out = []
    t0 = time.time()
    for s in test:
        ids = encode(s, vocab, rank)
        for k in range(1, len(ids)):
            f = run_ids(ids[max(0, k - WIN):k], E_, W_, b_, text)
            lg = dec(f["LOGITS"])
            out.append((int(np.argmax(lg[-1])), ids[k]))
    t1 = sum(1 for p, t in out if p == t)
    print("%s top1=%d/%d=%.4f CI=%s (%.0fs)" %
          (tag, t1, len(out), t1 / len(out), wilson(t1, len(out)),
           time.time() - t0), flush=True)
    return out


def cmd_transfer():
    vocab, rank = load_bpe()
    test = load_probe()
    _, _, WF = bodies()
    E_ = dict(np.load(os.path.join(DD, "lm_piece64.npz")))
    b_ = dict(np.load(os.path.join(DD, "bankpiece64_d64.npz")))
    text = listing("lm_d64.asm", False)
    a, _ = encode("alexander founded alexandria", vocab, rank), None
    ids = a
    for key in ("H4", "LOGITS"):
        db, mx, mg = par_db(E_, WF, b_, ids, key)
        print("parity-%s %.2fdB refmax=%.2f gotmax=%.2f" % (key, db, mx, mg),
              flush=True)
    td = twin_num(E_, WF, b_, text, vocab, rank)
    print("transfer twin=%.4f" % td, flush=True)
    out = top1(E_, WF, b_, text, test, vocab, rank, "transfer")
    np.savez("/tmp/d64_transfer.npz", preds=np.array(out))
    print("BAND: twin=%.4f top1=%d/566" %
          (td, sum(1 for p, t in out if p == t)), flush=True)


def cmd_variant():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["V", "QK"])
    ap.add_argument("scale", type=float)
    ap.add_argument("--top1", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(sys.argv[2:])
    vocab, rank = load_bpe()
    _, _, WF = bodies()
    W_ = dict(WF)
    if a.kind == "V":
        W_["wv"] = np.eye(64) * a.scale
    else:
        W_["wq"] = np.eye(64) * a.scale
        W_["wk"] = np.eye(64) * a.scale
    E_ = dict(np.load(os.path.join(DD, "lm_piece64.npz")))
    b_ = dict(np.load(os.path.join(DD, "bankpiece64_d64.npz")))
    text = listing("lm_d64.asm", False)
    td = twin_num(E_, W_, b_, text, vocab, rank)
    print("%s%s twin=%.4f" % (a.kind, a.scale, td), flush=True)
    if a.top1:
        test = load_probe()
        out = top1(E_, W_, b_, text, test, vocab, rank,
                   a.tag or ("%s%s" % (a.kind, a.scale)))
        fn = "/tmp/d64_%s.npz" % (a.tag or ("%s%s" % (a.kind, a.scale)))
        np.savez(fn, preds=np.array(out))


def cmd_ladder():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="")
    ap.add_argument("--tag", default="ladder")
    a = ap.parse_args(sys.argv[2:])
    E0 = dict(np.load(os.path.join(DD, "lm_piece64.npz")))
    _, _, WF = bodies()
    W_ = dict(np.load(a.body)) if a.body else WF
    man = json.load(open(os.path.join(DD, "lm_piece64_manifest.json")))
    spec_nat = man["spec64"]
    b0 = dict(np.load(os.path.join(DD, "bankpiece64_d64.npz")))
    words = b0["words"]
    vocab, rank = load_bpe()
    text = listing("lm_d64_headt.asm", True)
    Uh, sv, _ = np.linalg.svd(E0["emb"], full_matrices=False)
    Vt = np.diag(1 / sv) @ E0["wlog"]
    sgn = np.sign((Uh * E0["emb"]).sum(axis=0))
    sgn[sgn == 0] = 1
    U = Uh * sgn[None, :]
    for tau in [1.0, 0.8, 0.6, 0.4855, 0.38, 0.25]:
        Ef = U * (sv[None, :] ** tau)
        Wf = (sv[:, None] ** tau) * Vt
        evb = Ef[words]
        ukt = (evb / np.linalg.norm(evb, axis=1, keepdims=True)).T.copy()
        b_ = {"ukt": ukt, "evb": evb}
        E_ = {"emb": Ef, "wlog": Wf}
        td = twin_num(E_, W_, b_, text, vocab, rank)
        print("%s tau=%.4f spec=%.2f twin=%.4f" %
              (a.tag, tau, spec_nat ** tau, td), flush=True)


CMDS = {"transfer": cmd_transfer, "variant": cmd_variant,
        "ladder": cmd_ladder}

if __name__ == "__main__":
    CMDS[sys.argv[1]]()
