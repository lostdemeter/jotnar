"""Piece-native store gate: OOV-free retrieval (content scale-up).

Frozen data/pkeys.npz (202 mined edges, piece-space keys, plex
hash-addressed, oov=0) via scripts/freeze_pkeys.py. Gates: recall
bands through assoc_mem.asm (100% @0/8, same mechanism bars),
partial-cue probe >= word-space rate (0.965 -- the point of
piece-native: no OOV drop ever), full-cue == 1.00 (deterministic
pipeline, lm-memorize precedent), determinism bit-exact.
Usage: python3 tests/test_pkey.py
"""
import hashlib
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

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    man = json.load(open(os.path.join(dd, "pkeys_manifest.json")))
    check("pkey-manifest", man["n_keys"] == 202 and man["oov"] == 0,
          f"202 keys, oov=0 (sha {man['keys_sha']})")
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in words_of(s):
            syms = [c for c in w] + ["</w>"]
            while len(syms) > 1:
                best = None
                for i in range(len(syms) - 1):
                    r = rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best[0]):
                        best = (r, i)
                if best is None:
                    break
                _, i = best
                syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
            out.extend(vocab[p] for p in syms)
        return out

    def ppat(pid):
        h = int(hashlib.sha256(f"plex:{pid}".encode()).hexdigest()[:16], 16)
        return np.random.default_rng(h).choice([-1.0, 1.0], size=DIM)

    DIM = 64
    z = np.load(os.path.join(dd, "pkeys.npz"))
    keys, values = z["keys"], z["values"]
    text = open(os.path.join(sdir, "assoc_mem.asm")).read()

    def recall(cue):
        f = ASM.run_text(text, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()), "values": enc(values)},
                         sigs=SIGS, basedir=sdir)
        return np.asarray(f["OUT"][0]).reshape(-1) > 0

    rng = np.random.default_rng(3)
    for f_, bar in ((0, 1.0), (8, 1.0)):
        ok = tot = 0
        for i in range(len(keys)):
            for _ in range(2):
                cue = keys[i].copy()
                if f_:
                    cue[rng.choice(64, size=f_, replace=False)] *= -1
                got = np.where(recall(cue), 1.0, -1.0)
                ok += bool((got == values[i]).all())
                tot += 1
        check(f"pkey-recall{f_}", ok / tot >= bar, f"{ok}/{tot}")
    edges = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]

    def probe(full):
        ok = tot = 0
        for ei, e in enumerate(edges):
            tail = words_of(e["obj"]) if full else words_of(e["obj"])[:1]
            pids = encode(e["subj"] + " " + e["pred"] + " " + " ".join(tail))
            cue = np.sign(sum(np.roll(ppat(p), k) for k, p in enumerate(pids)))
            cue[cue == 0] = 1.0
            ki = int(np.argmax(keys @ cue))
            tot += 1
            if int(z["key_edge"][ki]) == ei:
                ok += 1
        return ok, tot

    ok, tot = probe(False)
    check("pkey-partial", ok / tot >= 0.965,
          f"{ok}/{tot} (must match-or-beat word-space 0.965)")
    ok, tot = probe(True)
    check("pkey-full", ok == tot, f"{ok}/{tot} (deterministic pipeline)")
    a = recall(keys[0])
    b = recall(keys[0])
    check("pkey-deterministic", bool((a == b).all()), "same cue twice identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
