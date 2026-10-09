"""Yarnball chatbot REPL, teacher side: ask questions, teach facts live.

One load (weights/tokenizer persist: fast turns after ~1min start);
persistent ball across turns (taught stores accumulate + ledger).
Single-forward point answers (top-1 + rank -- honest about scope:
generation trajectories are research/gen_survive.py's job).
Commands:
  /teach <prompt> | <subject> | <target>   mine early key @ subject
      (offset-mapped) + target readout-row value, gain 1 fixed
      (operating point, no laddering in the loop)
  /bank        ledger rows this session
  /holds       battery tops vs unsteered (re-runs base live)
  /quit        exit
  anything else: ask -> top token + target-rank readout
Example: /teach The capital of Germany is | Germany |  Paris
then: The capital of Germany is   (answers Paris, holds thereafter)
Usage: python3 demo_chat_qwen.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(ROOT, "..", "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

KEY_SCALE = 8.0
# Demo world: capitals battery (sibling prompts known). Holds need
# SIBLING nulls (same-template rows match the target key at 0.627:
# anti-key nulls never compete). Novel facts beyond capitals need
# explicit nulls (documented limit, not a silent gap).
SIBLINGS = {"Germany": "The capital of Germany is",
            "Italy": "The capital of Italy is",
            "Japan": "The capital of Japan is"}
# Default dose 0.25 (ladder-measured: Paris-row installs @0.25,
# saturates beyond (r151389 @1.0) -- inverted-U, per-fact doctrine.
# Tokyo-class values keep their own rung (queued); dose is data.
GAIN = 0.25


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
        sys.exit("SKIP (needs Qwen2-7B-Instruct snapshot)")
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    stores = []

    def ball():
        if not stores:
            return None, None
        return yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                             stores, key_scale=KEY_SCALE)[:2]

    def ask(prompt, verbose=False):
        Ua, Vc = ball()
        t7, gids = fwdH(prompt, keep="all")
        toks = tok.convert_ids_to_tokens(gids)
        # address: best-matching taught subject row, else end row
        if Ua is not None:
            best, bi = -1e9, len(gids) - 1
            nq = Ua.shape[1]
            for i in range(len(gids)):
                xa = t7[2][i]
                xa = xa / np.linalg.norm(xa)
                s = float(np.max(xa @ Ua[:, :nq]))
                if s > best:
                    best, bi = s, i
            pos = bi
        else:
            pos = len(gids) - 1
        xa = t7[2][pos]
        xa = xa / np.linalg.norm(xa)
        if Ua is None:
            lg, _ = fwd(prompt)
            return lg, None
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        mag = float(np.linalg.norm(t7[27][-1]))
        y = (P @ Vc) * (GAIN * mag)
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(prompt, steer=(27, yn, float(np.linalg.norm(y) / mag)))
        return lg, (pos, toks[pos] if pos < len(toks) else "?", P)

    print("yarnball chat (Qwen2-7B + persistent ball). /teach prompt | "
          "subject | target | /bank | /holds | /quit", flush=True)
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
            print(f"{len(stores)} taught stores:", flush=True)
            for st in stores:
                print(f"  {st['tier']} {st['support']}", flush=True)
            continue
        if line == "/holds":
            for p in ("The capital of Germany is", "The capital of Italy is",
                      "The capital of Japan is"):
                lg, extra = ask(p)
                top = int(lg.argmax())
                print(f"  {p!r} -> {tok.decode([top])!r}", flush=True)
            continue
        if line.startswith("/teach "):
            try:
                prompt, subj, target = [x.strip() for x in
                                        line[len("/teach "):].split("|")]
            except ValueError:
                print("usage: /teach prompt | subject | target", flush=True)
                continue
            pos = span_pos(prompt, subj, tok)
            key, _ = early_key(prompt, pos=pos)
            cc = next((_c for _c, _p in SIBLINGS.items() if _p == prompt),
                      None)
            # reteach = update: drop prior stores for this country first
            # (stale null-vs-assoc twins would split retrieval)
            if cc is not None:
                stores[:] = [s for s in stores if s.get("cc") != cc]
            for _c, _p in SIBLINGS.items():
                if _p == prompt:
                    continue
                _t = tok(_p, return_tensors="pt")["input_ids"][0].numpy()
                _toks = tok.convert_ids_to_tokens(_t)
                _frag = _c[1:].lower()
                _pp = next((i for i, _t2 in enumerate(_toks)
                            if _frag in _t2.lower()), len(_t) - 1)
                _k, _ = early_key(_p, pos=_pp)
                stores.append({"key": _k, "value": np.zeros(D),
                               "tier": "null", "cc": _c,
                               "support": f"sibling {_c}"})
            tid = tok(" " + target.strip(), return_tensors="pt")["input_ids"][0].tolist()[0]
            v = Wlog[tid] / np.linalg.norm(Wlog[tid])
            stores.append({"key": key, "value": v, "dose": 1.0,
                           "tier": "assoc", "cc": cc,
                           "support": f"{prompt} -> {target.strip()}"})
            stores = [s for s in stores if s["tier"] != "null" or
                      s["support"].startswith("sibling")]
            # NOTE: anti-key null replaced by sibling nulls above (anti
            # never competes: same-template rows match target at 0.627)
            lg, _ = ask(prompt)
            print(f"taught ({len(stores)} stores): {target.strip()} rank "
                  f"{int((lg > lg[tid]).sum()) + 1}", flush=True)
            continue
        lg, extra = ask(line)
        top = int(lg.argmax())
        o = np.argsort(-lg)[:3]
        print(f"{tok.decode([top])!r} "
              f"(top3={[(tok.decode([int(i)]), round(float(lg[int(i)]), 1)) for i in o]})",
              flush=True)
        if extra is not None:
            pos, t, P = extra
            print(f"  via row {pos} ({t}), retrieve={int(P.argmax())}",
                  flush=True)


if __name__ == "__main__":
    main()
