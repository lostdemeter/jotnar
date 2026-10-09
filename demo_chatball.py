"""Yarnball chatbot demo: talk to the native ball, teach it facts live.

REPL over programs/lm_yarnball.asm (generic ball, CPU): every turn
greedily generates through the listing with the CURRENT persistent
bank. Commands:
  /teach <prompt words> -> <target word>   mine emb key + readout
      value natively, append store + ledger row (persists session)
  /bank                                    ledger rows this session
  /holds                                   16-prompt battery vs base
  /quit                                    exit
  anything else                            generate 12 tokens greedily
Example: /teach the capital of italy is -> rome
then: the capital of italy is   (answers Rome, held Rome thereafter)
Base model babbles (top1 0.34); installs land exactly (the demo is
the LOOP: read/update/generate through assembly, not fluency).
Usage: python3 demo_chatball.py [--seed-stores]
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(ROOT, "..", "phi-core")))
sys.path.insert(0, ROOT)

from chain.engram import yarnball_bank

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
N_GEN = 12


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-stores", action="store_true",
                    help="preload Italy->Rome (shows taught behavior instantly)")
    args = ap.parse_args()
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    text = CFG + open(os.path.join(sdir, "lm_yarnball.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    stores, ledger = [], []

    def bank():
        if not stores:
            return ukt0, evb0
        Ua, Vc, led = yarnball_bank(ukt0, evb0, stores, key_scale=8.0)
        return Ua, Vc

    def run(ids, ukt, evb):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog),
                             "ukt": enc(ukt), "evb": enc(evb)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    def teach(prompt, target):
        ids, tv = ids_of(prompt), vocab.get(target.lower(), 0)
        if tv == 0 or not ids:
            return f"unknown word (vocab has 'italy','rome','alexander'...)"
        # address = whole-question mean embedding (unit): retrieval
        # matches similar questions, not entities (documented choice;
        # entity keys are the precision follow-up, queued).
        e = np.ascontiguousarray(dE["emb"][np.array(ids)]).mean(axis=0)
        v = np.ascontiguousarray(wlog[:, tv])
        stores.append({"key": e / np.linalg.norm(e),
                       "value": v / np.linalg.norm(v),
                       "dose": 8.0 * evn, "tier": "assoc",
                       "support": f"{prompt} -> {target}"})
        Ua, Vc = bank()
        lg = dec(run(ids[-8:], Ua, Vc)["LOGITS"])[-1]
        return (f"taught ({len(stores)} stores): {inv.get(tv, '?')} rank "
                f"{int((lg > lg[tv]).sum()) + 1}")

    if args.seed_stores:
        print(teach("the capital of italy is", "rome"), flush=True)
    print("yarnball chat (native d16 ball). /teach a -> b | /bank | "
          "/holds | /quit", flush=True)
    base_tops = {}
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/bank":
            Ua, Vc = bank()
            print(f"{Ua.shape[1]} stores ({len(stores)} taught):", flush=True)
            for r in yarnball_bank(ukt0, evb0, stores)[2][len(stores) and -len(stores):] if stores else []:
                print(f"  {r['tier']} {r['support']} dose={r['dose']:.2f}",
                      flush=True)
            continue
        if line == "/holds":
            n = done = 0
            for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
                ids = ids_of(s)
                for k in range(1, min(len(ids), 4)):
                    if done >= 16:
                        break
                    ctx = ids[max(0, k - 7):k]
                    Ua, Vc = bank()
                    t1 = int(dec(run(ctx, Ua, Vc)["LOGITS"])[-1].argmax())
                    t0 = base_tops.get(tuple(ctx))
                    if t0 is None:
                        t0 = int(dec(run(ctx, ukt0, evb0)["LOGITS"])[-1].argmax())
                        base_tops[tuple(ctx)] = t0
                    n += t1 == t0
                    done += 1
                if done >= 16:
                    break
            print(f"holds {n}/16", flush=True)
            continue
        m = re.match(r"/teach\s+(.+?)\s*->\s*([a-z0-9']+)\s*$", line)
        if m:
            print(teach(m.group(1), m.group(2)), flush=True)
            continue
        ids = ids_of(line)[-8:]
        seq = list(ids)
        Ua, Vc = bank()
        for _ in range(N_GEN):
            if len(seq) >= 8:
                seq = seq[1:]
            top = int(dec(run(seq, Ua, Vc)["LOGITS"])[-1].argmax())
            seq.append(top)
        print(" ".join(inv.get(i, "?") for i in seq[len(ids):]), flush=True)


if __name__ == "__main__":
    main()
