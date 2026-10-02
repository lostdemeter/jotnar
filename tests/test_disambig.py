"""Disambiguation gate: second-pass full-triple re-cue + family hunt (Gate 4 word path).

Composition (always-run, no thresholds -- each pass is a single
listing run + host arithmetic): pass 1 cues the question shape
(subj+pred+obj[0]) -> top-1 edge + ambiguity SET (top-K EDGES union
ties at cutoff -- ties are genuine collisions, dropping them is the
bug m0131 names); pass 2 re-cues the FULL triple (consumer's full
utterance / generation draft; simulated from records, stated) ->
each set edge by BEST row (order-variant max -- the fix v07/passive
names: content decides, order matches itself); family stage expands
the set to (subj,pred) FAMILIES (the tail hunt: rank>8 burials live
in giant families, 78.2 vs 2.9 mean) -> full-triple argmax over
members. K priced per bank (202: max miss rank 4; w103m5: K=8,
cost/recall; curated v07: K=4). Rejected with reason: IDF-weighted
cues (collapse the bipolar combiner to 1-term dominance, 0.70 bank /
0.01 tail); obj-answer cues (leak obj[1:], caught by contract audit);
unigram priors (wrong direction 8x, unbuilt).

Gates: first-pass numbers PINNED (195/202, 9528/10672 -- regression
tripwires); second-pass barred (202/202 exact; w103m5 >= 0.96);
family stage barred (202/202; w103m5 >= 0.995, measured 0.9960 --
losses EXACTLY family-absent (43), P2 perfect within family by
exact-self-match mechanism); v07 both sides retrieve (word path,
edge + family); listing discipline (passes execute through
assoc_mem.asm + deterministic replay). Hidden-path v07 (assoc_cos)
stays a NAMED limitation (3 re-cue designs falsified: candidate
rescore, grammar strip, subject-residual -- same-subject hidden
collapse mechanism).
Usage: python3 tests/test_disambig.py (slow: w103m5 chunked dots)
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


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    base = {w: zw[f"w{i}"] for i, w in enumerate(list(zw["lex"]))}
    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()

    def lex_for(edges):
        wpat = dict(base)
        for w in sorted({x for e in edges for f in ("subj", "pred", "obj")
                         for x in words_of(e[f])} - set(wpat)):
            h = int(hashlib.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
            wpat[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=64)
        return wpat

    def cue_of(words, wpat):
        pats = [(k, wpat[w]) for k, w in enumerate(words) if w in wpat]
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        return cue

    def edge_scores(D, row_edge, n_edges):
        E = np.full((D.shape[0], n_edges), -1e18)
        for r, e in enumerate(row_edge):
            np.maximum(E[:, e], D[:, r], out=E[:, e])
        return E

    def second_pass(edge_top, full_cue, keys, row_edge):
        sims = keys @ full_cue
        best_e, best_s = -1, -1e18
        for e in sorted(edge_top):
            s = sims[row_edge == e].max()
            if s > best_s:
                best_s, best_e = s, e
        return best_e

    def family_index(edges):
        fam, members = {}, {}
        for ei, e in enumerate(edges):
            f = (e["subj"], e["pred"])
            fam[ei] = f
            members.setdefault(f, []).append(ei)
        return fam, members

    def family_pick(eset, fam, members, full_row):
        cand = [m for f in {fam[e] for e in eset} for m in members[f]]
        best = min(cand, key=lambda m: (-full_row[m], m))
        return best, len(cand)

    # ---- bank 1: mined 202 (K=4: max miss rank 4, priced) ----
    mined = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]
    zM = np.load(os.path.join(dd, "mined_keys.npz"))
    kM, keM = zM["keys"].astype(np.float64), zM["key_edge"]
    wpatM = lex_for(mined)
    P1 = np.array([cue_of(words_of(e["subj"]) + words_of(e["pred"])
                          + words_of(e["obj"])[:1], wpatM) for e in mined])
    F = np.array([cue_of(words_of(e["subj"]) + words_of(e["pred"])
                         + words_of(e["obj"]), wpatM) for e in mined])
    D1 = P1 @ kM.T
    E1 = edge_scores(D1, keM, len(mined))
    s1 = sum(1 for ei in range(len(mined)) if int(np.argmax(E1[ei])) == ei)
    check("disambig-m202-first", s1 == 195, f"first-pass pinned {s1}/202")
    s2, sizes2 = 0, []
    for ei in range(len(mined)):
        slots = np.argpartition(-E1[ei], 4)[:4]
        eset = np.where(E1[ei] >= E1[ei, slots].min())[0]
        sizes2.append(len(eset))
        if second_pass(eset, F[ei], kM, keM) == ei:
            s2 += 1
    check("disambig-m202-second", s2 == 202,
          f"second-pass {s2}/202 (K=4 edges, mean set "
          f"{sum(sizes2) / len(sizes2):.1f})")
    D2F = F @ kM.T
    famM, memM = family_index([{"subj": e["subj"], "pred": e["pred"]} for e in mined])
    s3 = 0
    for ei in range(len(mined)):
        slots = np.argpartition(-E1[ei], 4)[:4]
        eset = np.where(E1[ei] >= E1[ei, slots].min())[0]
        best, _ = family_pick(eset, famM, memM, D2F[ei])
        if best == ei:
            s3 += 1
    check("disambig-m202-family", s3 == 202, f"family stage {s3}/202")

    # ---- bank 2: w103m5 10672 (K=8: cost/recall, mean set 11.6) ----
    w3 = [json.loads(l) for l in open(os.path.join(dd, "banks", "w103m5_edges.jsonl"))][1:]
    zW = np.load(os.path.join(dd, "banks", "w103m5_keys.npz"))
    kW = zW["keys"].astype(np.float64)
    keW = zW["key_edge"]
    N = len(w3)
    wpatW = lex_for(w3)
    s1w = s2w = 0
    sizes = []
    CH = 1000
    for a in range(0, N, CH):
        b = min(a + CH, N)
        P1c = np.array([cue_of(words_of(w3[ei]["subj"]) + words_of(w3[ei]["pred"])
                               + words_of(w3[ei]["obj"])[:1], wpatW)
                        for ei in range(a, b)])
        Fc = np.array([cue_of(words_of(w3[ei]["subj"]) + words_of(w3[ei]["pred"])
                              + words_of(w3[ei]["obj"]), wpatW)
                       for ei in range(a, b)])
        Dc = P1c @ kW.T
        Ec = edge_scores(Dc, keW, N)
        s1w += sum(1 for i in range(b - a) if int(np.argmax(Ec[i])) == a + i)
        Df = Fc @ kW.T
        for i, ei in enumerate(range(a, b)):
            slots = np.argpartition(-Ec[i], 8)[:8]
            eset = np.where(Ec[i] >= Ec[i, slots].min())[0]
            sizes.append(len(eset))
            if second_pass(eset, Fc[i], kW, keW) == ei:
                s2w += 1
    check("disambig-w103-first", s1w == 9528, f"first-pass pinned {s1w}/{N}")
    check("disambig-w103-second", s2w / N >= 0.96,
          f"second-pass {s2w}/{N}={s2w / N:.4f} (K=8, mean set "
          f"{sum(sizes) / len(sizes):.1f}; losses are rank>8 tail)")
    famW, memW = family_index([{"subj": e["subj"], "pred": e["pred"]} for e in w3])
    s3w = 0
    candsizes = []
    CH2 = 1000
    for a in range(0, N, CH2):
        b = min(a + CH2, N)
        P1c = np.array([cue_of(words_of(w3[ei]["subj"]) + words_of(w3[ei]["pred"])
                               + words_of(w3[ei]["obj"])[:1], wpatW)
                        for ei in range(a, b)])
        Fc = np.array([cue_of(words_of(w3[ei]["subj"]) + words_of(w3[ei]["pred"])
                              + words_of(w3[ei]["obj"]), wpatW)
                       for ei in range(a, b)])
        Ec = edge_scores(P1c @ kW.T, keW, N)
        Df = Fc @ kW.T
        for i, ei in enumerate(range(a, b)):
            slots = np.argpartition(-Ec[i], 8)[:8]
            eset = np.where(Ec[i] >= Ec[i, slots].min())[0]
            best, nc = family_pick(eset, famW, memW, Df[i])
            candsizes.append(nc)
            if best == ei:
                s3w += 1
    check("disambig-w103-family", s3w / N >= 0.995,
          f"family stage {s3w}/{N}={s3w / N:.4f} (mean cand "
          f"{sum(candsizes) / len(candsizes):.0f}; losses are family-absent)")

    # ---- v07 both sides through the composition (curated bank) ----
    zC = np.load(os.path.join(dd, "edge_keys.npz"))
    kC = zC["keys"].astype(np.float64)
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
    voice = json.load(open("/home/thorin/Documents/OpenCode/Echion_Revisted/data/voice_pairs.json"))
    v07 = [p for p in voice if p["id"] == "v07"][0]
    e06 = [r for r in recs if r["tag"] == "caesar/action#3"][0]
    row_edgeC = []
    for ri, r in enumerate(recs):
        row_edgeC += [ri] * r["n_keys"]
    row_edgeC = np.array(row_edgeC)
    tags = [r["tag"] for r in recs]

    vok = 0
    for side in ("active", "passive"):
        q1 = cue_of(words_of(v07[side]), base)
        e1 = edge_scores((kC @ q1).reshape(1, -1), row_edgeC, len(recs))[0]
        slots = np.argpartition(-e1, 4)[:4]
        eset = np.where(e1 >= e1[slots].min())[0]
        if tags[second_pass(eset, cue_of(e06["canon"], base), kC, row_edgeC)] == "caesar/action#3":
            vok += 1
    check("disambig-v07", vok == 2, f"v07 both sides retrieve (word path) {vok}/2")
    famC, memC = family_index([{"subj": r["subj"], "pred": r["pred"]} for r in recs])
    vfam = 0
    for side in ("active", "passive"):
        q1 = cue_of(words_of(v07[side]), base)
        e1 = edge_scores((kC @ q1).reshape(1, -1), row_edgeC, len(recs))[0]
        slots = np.argpartition(-e1, 4)[:4]
        eset = np.where(e1 >= e1[slots].min())[0]
        full = kC @ cue_of(e06["canon"], base)
        frow = np.full(len(recs), -1e18)
        for r, e in enumerate(row_edgeC):
            if full[r] > frow[e]:
                frow[e] = full[r]
        best, _ = family_pick(eset, famC, memC, frow)
        if tags[best] == "caesar/action#3":
            vfam += 1
    check("disambig-v07-family", vfam == 2, f"v07 family stage {vfam}/2")

    # ---- listing discipline: both passes execute + replay identical ----
    for tag, cue in (("pass1", P1[20]), ("pass2", F[20])):
        outs = []
        for _ in range(2):
            f = ASM.run_text(assoc, REGISTRY,
                             {"cue": enc(cue.reshape(1, -1)),
                              "keys": enc(kM.T.copy()),
                              "values": enc(zM["values"])},
                             sigs=SIGS, basedir=sdir)
            outs.append(np.asarray(f["OUT"][0]).tobytes())
        check(f"disambig-listing-{tag}", outs[0] == outs[1], "executes + replay identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
