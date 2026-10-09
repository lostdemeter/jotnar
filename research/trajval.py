"""Trajectory-aware values: does point-optimal generate worse?

Candidates on Hamlet (installed territory): Ham readout row (status
quo: installs + 1 toll), maximin-for-Ham (better point margins),
contrast control (installs nothing), random control. Per candidate:
point margin (Ham logit gap), displacement (steered vs unsteered
prompt-end state, relative), 8-step generation (surface + toll).
Prediction under test: margin and displacement ANTI-correlate across
candidates (point-optimal is most off-manifold) -- i.e., trajectory
quality needs its own objective, not maximin leftovers. If rankings
agree instead, trajectory-awareness adds nothing (also an answer).
Usage: python3 research/trajval.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

PROMPT = "Shakespeare wrote the play"
SUBJ = "Shakespeare"
N_GEN = 8
KEY_SCALE = 8.0
GAIN = 0.5


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from siphon_battery import country_pos
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    ham = tok(" Hamlet", return_tensors="pt")["input_ids"][0].tolist()
    _, pos = country_pos(PROMPT, SUBJ, tok)
    key, _ = early_key(PROMPT, pos=pos)
    t7, _ = fwdH(PROMPT, keep="all")
    xa = t7[2][pos]
    xa = xa / np.linalg.norm(xa)
    x0 = t7[27][-1]
    mag0 = float(np.linalg.norm(x0))

    vham = Wlog[ham[0]] / np.linalg.norm(Wlog[ham[0]])
    # maximin for Ham-top1 over flat-ish head top blockers (point-optimal)
    lg0, _ = fwd(PROMPT)
    o = np.argsort(-lg0)[:8]
    comps = [int(i) for i in o if int(i) != ham[0]]
    A = np.stack([Wlog[ham[0]] - Wlog[c] for c in comps], axis=0)
    rng = np.random.default_rng(0)
    bv, bt = None, -1e18
    for _ in range(6):
        v = rng.normal(size=D)
        v /= np.linalg.norm(v)
        st = 0.2
        for _it in range(3000):
            s = A.dot(v)
            j = int(np.argmin(s))
            gg = A[j] - (A[j].dot(v)) * v
            n = np.linalg.norm(gg)
            if n > 1e-12:
                v = v + st * gg / n
                v /= np.linalg.norm(v)
            st *= 0.9995
        t = float(np.min(A.dot(v)))
        if t > bt:
            bt, bv = t, v.copy()
    vrng = rng.normal(size=D)
    vrng /= np.linalg.norm(vrng)
    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    vcon = tfr[27] - tde[27]
    vcon /= np.linalg.norm(vcon)
    cands = {"hamrow": vham, "maximin": bv, "contrast": vcon, "random": vrng}
    Ua, _, _ = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": key, "value": vham, "dose": 1.0, "tier": "assoc",
          "support": "address (shared)"},
         {"key": -key, "value": np.zeros(D), "tier": "null",
          "support": "bg"}],
        key_scale=KEY_SCALE)

    import torch  # noqa (torch ops via _torch below)
    print(f"{'cand':9} {'margin':>7} {'displ':>7} trajectory", flush=True)
    for name, v in cands.items():
        Vc = np.concatenate([(GAIN * mag0 * v)[None, :], np.zeros((1, D))], axis=0)
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        # point margin under this value (single forward)
        y = (P @ Vc) * 1.0
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(PROMPT, steer=(27, yn, float(np.linalg.norm(y) / mag0)))
        mg = float(lg[ham[0]] - np.sort(lg)[-2] if int(lg.argmax()) == ham[0]
                   else lg[ham[0]] - np.sort(lg)[-1])
        # displacement: steered vs unsteered prompt-end state
        from qwen_torch import _get, _layer
        _torch, _gg, _tt = _get()
        dt = _torch.float32
        ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy()
        n = len(ids)
        E = _torch.tensor(_gg("model.embed_tokens.weight")[ids], dtype=dt,
                          device="cuda")
        x = E
        for L in range(28):
            st = None
            if L == 27:
                st = (27, yn, float(np.linalg.norm(y) / mag0))
            x = _layer(x, _gg, _torch, L, n, steer=st)
        disp = float(_torch.norm(x[-1] - _torch.tensor(x0, dtype=dt,
                                                       device="cuda"))
                     / (mag0 + 1e-12))
        # generation
        seq = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
        for step in range(N_GEN):
            cur = tok.decode(seq)
            lg2, _ = fwd(cur) if step else (lg, None)
            seq = seq + [int(lg2.argmax())]
        txt = tok.decode(seq)
        deg = txt.count("�")
        print(f"{name:9} {mg:+7.2f} {disp:7.3f} {txt[:75]!r} deg={deg}",
              flush=True)


if __name__ == "__main__":
    main()
