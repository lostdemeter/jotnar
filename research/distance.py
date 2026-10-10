"""Distance battery v1: authors + units, templates x paraphrases + chatter.

Viable singles from scout_v2 (unsteered, non-top, single token):
austen/Emma r4, tesla/Tesla r6. Two templates each (paraphrase must
keep the SAME target); chatter holds ("hello", Shakespeare quote,
"how are you" -- open dialogue must route to nulls, measured not
assumed). Ball per fact: asker store (offset-mapped subject) +
sibling nulls; fixed ks=8, gains {1,2}. Gates: 4 installs + shared
paraphrases + chatter holds (tops == unsteered) + standard holds.
Live-mined every run.
Usage: python3 research/distance.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

FAMS = {
    "austen": [("Jane Austen wrote the novel", "Austen", " Emma"),
               ("Jane Austen is the author of", "Austen", " Emma")],
    "tesla": [("The SI unit of magnetic flux density is the", "flux", " Tesla"),
              ("Magnetic flux density is measured in", "flux", " Tesla")],
}
CHATTER = ["hello", "Cry havoc and let slip", "how are you doing today"]
GAINS = [1, 2]
KEY_SCALE = 8.0


def span_pos(prompt, sub, tok):
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].numpy()
    lo = prompt.find(sub)
    assert lo >= 0, (sub, prompt)
    hi = lo + len(sub)
    for i, (a, b) in enumerate(offs):
        if a < hi and lo < b:
            return i
    raise AssertionError((sub, prompt))


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    allp = [(fam, p, s, t) for fam, ms in FAMS.items() for (p, s, t) in ms]
    base = {}
    for _, p, _, _ in allp:
        lg, _ = fwd(p)
        base[p] = int(lg.argmax())
    for c in CHATTER:
        lg, _ = fwd(c)
        base[c] = int(lg.argmax())
    print("unsteered:", {p[:28]: tok.decode([v]) for p, v in base.items()},
          flush=True)
    keys = {}
    for _, p, s, _ in allp:
        pos = span_pos(p, s, tok)
        k, _ = early_key(p, pos=pos)
        keys[p] = (k, pos)
        print(f"key {p[:28]!r}: row={pos}", flush=True)
    for fam, members in FAMS.items():
        pa, aska, ta = members[0]
        tid = tok(ta, return_tensors="pt")["input_ids"][0].tolist()[0]
        v = Wlog[tid] / np.linalg.norm(Wlog[tid])
        stores = [{"key": keys[pa][0], "value": v, "dose": 1.0,
                   "tier": "assoc", "support": f"asker {aska} -> {ta}"}]
        for fam2, ms2 in FAMS.items():
            if fam2 == fam:
                continue
            for (p, s, _) in ms2:
                pos = span_pos(p, s, tok)
                k, _ = early_key(p, pos=pos)
                stores.append({"key": k, "value": np.zeros(D),
                               "tier": "null", "support": f"background {fam2}"})
                break
        Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                  stores, key_scale=KEY_SCALE)
        for gn in GAINS:
            row = []
            for p, s, t in members + [(c, None, None) for c in CHATTER]:
                t7, gids = fwdH(p, keep="all")
                if s is None:
                    # chatter: no subject; address at end row THROUGH the
                    # ball (does open dialogue trigger installs?). Hold =
                    # top unchanged vs unsteered.
                    xa = t7[2][-1]
                    xa = xa / np.linalg.norm(xa)
                    C = xa @ Ua
                    P = np.exp(C - C.max())
                    P /= P.sum()
                    mag = float(np.linalg.norm(t7[27][-1]))
                    y = (P @ Vc) * (gn * mag)
                    yn = y / (np.linalg.norm(y) + 1e-12)
                    lg, _ = fwd(p, steer=(27, yn,
                                          float(np.linalg.norm(y) / mag)))
                    top = int(lg.argmax())
                    row.append(f"chat:{p[:12]!r}:{tok.decode([top])[:10]!r}:"
                               f"{'hold' if top == base[p] else 'MOVED'}:"
                               f"maxP={float(P.max()):.2f}")
                    continue
                pos = span_pos(p, s, tok)
                xa = t7[2][pos]
                xa = xa / np.linalg.norm(xa)
                C = xa @ Ua
                P = np.exp(C - C.max())
                P /= P.sum()
                mag = float(np.linalg.norm(t7[27][-1]))
                y = (P @ Vc) * (gn * mag)
                yn = y / (np.linalg.norm(y) + 1e-12)
                lg, _ = fwd(p, steer=(27, yn, float(np.linalg.norm(y) / mag)))
                top = int(lg.argmax())
                if p == pa:
                    r = int((lg > lg[tid]).sum()) + 1
                    verdict = "INSTALL" if top == tid else f"r{r}"
                elif t == ta:
                    verdict = "SHARED" if top == tid else f"r{int((lg > lg[tid]).sum()) + 1}"
                else:
                    verdict = "hold" if top == base[p] else "MOVED"
                row.append(f"{p[:20]!r}:{tok.decode([top])[:10]!r}:{verdict}")
            print(f"{fam} gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
