"""Siphon install gate, teacher side (generic ball, mirror).

Ball [Germany: L27 France-minus-Germany contrast | Italy/Japan:
nulls] via yarnball_bank + ledger; apply is yarnball_apply math
(softmax routes; non-targets land on nulls). Live-mined every run.
Gates at gain 1: Germany top=Paris rank 1; Italy=Rome + Japan=blank
hold; every prompt retrieves its own store.
SKIPs without 7B snapshot/GPU (convention: test_qwen7b_gen).
Usage: python3 tests/test_siphon.py (slow: ~10 mirror calls)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "research"))

FAIL = []
COUNTRIES = ["Germany", "Italy", "Japan"]
TARGET = "Germany"
KEY_SCALE = 8.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import torch
    if not torch.cuda.is_available():
        print("SKIP (needs GPU)")
        sys.exit(0)
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    from chain.engram import yarnball_bank
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    _, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]

    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(promp[c])
        base[c] = int(lg.argmax())
    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    d = tfr[27] - tde[27]
    d /= np.linalg.norm(d)
    keys = {c: early_key(promp[c])[0] for c in COUNTRIES}
    D = 3584
    stores = [{"key": keys[TARGET], "value": d, "dose": 1.0,
               "tier": "assoc", "support": "L27 France-minus-Germany"}]
    for c in COUNTRIES:
        if c != TARGET:
            stores.append({"key": keys[c], "value": np.zeros(D),
                           "tier": "null", "support": "background"})
    Ua, Vc, ledger = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                   stores, key_scale=KEY_SCALE)
    check("siphon-ledger", [r["tier"] for r in ledger] == ["assoc", "null", "null"],
          str([r["tier"] for r in ledger]))
    for c in COUNTRIES:
        t7, gids = fwdH(promp[c], keep="all")
        gtoks = tok.convert_ids_to_tokens(gids)
        frag = {"Germany": "ermany", "Italy": "taly", "Japan": "apan"}[c]
        pos = next(i for i, t in enumerate(gtoks) if frag in t.lower())
        xa = t7[2][pos]
        xa = xa / np.linalg.norm(xa)
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        mag = float(np.linalg.norm(t7[27][-1]))
        y = (P @ Vc) * mag
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(promp[c], steer=(27, yn, float(np.linalg.norm(y) / mag)))
        top = int(lg.argmax())
        check(f"siphon-retrieve-{c}", int(P.argmax()) == COUNTRIES.index(c),
              f"store {int(P.argmax())}")
        if c == TARGET:
            check("siphon-install", top == paris,
                  f"top={tok.decode([top])!r} paris-rank={int((lg > lg[paris]).sum()) + 1}")
        else:
            check(f"siphon-hold-{c}", top == base[c],
                  f"top={tok.decode([top])!r}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
