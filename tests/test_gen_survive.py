"""Generation-survival gate, teacher side (methodology pinned).

Criterion v1 (set from gensurvive2 data, not assumed): SURVIVE =
target ids appear as an ordered subsequence within the first 4
generated tokens AND at most 1 degenerate token total (replacement
char U+FFFD or 3+ consecutive repeats). Separates cleanly: hamlet
0.5 (Ham,let at steps 1-2 + 1 toll char -> SURVIVE) vs orwell 1.0
(comma collapse -> FAIL) vs unsteered (no target -> FAIL).
Install: hamlet ball (first-token readout row + subject address +
null, gain 0.5, every step), 8 tokens greedy. Gates: SURVIVE +
unsteered does not (control: install adds the behavior).
SKIPs without snapshot/GPU.
Usage: python3 tests/test_gen_survive.py (slow: ~20 mirror calls)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "research"))

FAIL = []
PROMPT = "Shakespeare wrote the play"
SUBJ = "Shakespeare"
TARGET = " Hamlet"
GAIN = 0.5
N_GEN = 8
KEY_SCALE = 8.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def degenerate(seq, tok):
    txt = tok.decode(seq)
    nfffd = txt.count("�")
    rep = 0
    for i in range(2, len(seq)):
        if seq[i] == seq[i - 1] == seq[i - 2]:
            rep += 1
    return nfffd + rep


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
    from siphon_battery import country_pos
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    tids = tok(TARGET, return_tensors="pt")["input_ids"][0].tolist()
    _, pos = country_pos(PROMPT, SUBJ, tok)
    key, _ = early_key(PROMPT, pos=pos)
    v = Wlog[tids[0]] / np.linalg.norm(Wlog[tids[0]])
    D = 3584
    Ua, Vc, _ = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": key, "value": v, "dose": 1.0, "tier": "assoc",
          "support": f"first-token readout {TARGET}"},
         {"key": -key, "value": np.zeros(D), "tier": "null",
          "support": "background (anti-address)"}],
        key_scale=KEY_SCALE)

    def gen(dose):
        seq = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
        for step in range(N_GEN):
            cur = tok.decode(seq)
            t7, _ = fwdH(cur, keep="all")
            n = len(seq)
            prow = pos if pos < n else n - 1
            xa = t7[2][prow]
            xa = xa / np.linalg.norm(xa)
            C = xa @ Ua
            P = np.exp(C - C.max())
            P /= P.sum()
            mag = float(np.linalg.norm(t7[27][-1]))
            y = (P @ Vc) * (dose * mag)
            yn = y / (np.linalg.norm(y) + 1e-12)
            lg, _ = fwd(cur, steer=(27, yn, float(np.linalg.norm(y) / mag))
                        if dose else None)
            seq = seq + [int(lg.argmax())]
        return seq[len(tok(PROMPT, return_tensors="pt")["input_ids"][0]):]

    got = gen(GAIN)
    sub = got[:4]
    ordered = all(t in sub[len(sub) and 0:] for t in tids) and \
        [t for t in sub if t in set(tids)] == [t for t in tids if t in sub]
    check("gen-target-ordered", ordered,
          f"first4={[tok.decode([t]) for t in sub]} want{[tok.decode([t]) for t in tids]}")
    check("gen-clean", degenerate(got, tok) <= 1,
          f"degenerate={degenerate(got, tok)} seq={tok.decode(got)[:60]!r}")
    base = gen(0.0)
    check("gen-control", not ([t for t in base[:4] if t in set(tids)]
                              == [t for t in tids if t in base[:4]]),
          "unsteered must not already complete it")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
