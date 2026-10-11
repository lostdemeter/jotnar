"""Early-exit battery: where would each prompt have stopped? (mirror)

Per-layer residuals through final-norm + unembed (lens pattern) on
the battery; per prompt, FIRST layer whose top-1 equals the final
top-1, plus margin there. Exit layer << 27 with margin to spare =
early exit viable with a margin bar (skip the tail per prompt).
Exit ~= 27 everywhere = no savings (decisions form late, priced).
Reports exit-layer histogram + mean saved layers + margin at exit.
Usage: python3 research/early_exit.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

PROMPTS = ["The capital of Germany is", "The capital of Italy is",
           "The capital of Japan is", "Shakespeare wrote the play",
           "The tallest mountain on Earth is", "George Orwell wrote the novel"]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwdH, _rms
    import torch
    g, tok = load7b()
    dt = torch.float32
    Wlog = torch.tensor(g("lm_head.weight"), dtype=dt, device="cuda")
    lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
    exits, saved = [], []
    for pr in PROMPTS:
        traj, ids = fwdH(pr)
        n = len(ids)
        final = None
        tops = []
        for L in range(28):
            x = torch.tensor(traj[L], dtype=dt, device="cuda")
            hn = _rms(x, lnf)
            lg = (hn @ Wlog.T)[n - 1].detach().cpu().numpy()
            o = np.argsort(-lg)
            tops.append(int(o[0]))
            if L == 27:
                final = int(o[0])
        ex = next((L for L, t in enumerate(tops) if t == final), 27)
        # margin at exit (top1 - top2 abs logits)
        x = torch.tensor(traj[ex], dtype=dt, device="cuda")
        hn = _rms(x, lnf)
        lg = (hn @ Wlog.T)[n - 1].detach().cpu().numpy()
        o = np.argsort(-lg)
        mg = float(lg[o[0]] - lg[o[1]])
        exits.append(ex)
        saved.append(27 - ex)
        print(f"{pr[:30]!r}: exit@{ex} margin={mg:.2f} final={tok.decode([final])!r} "
              f"stable-after={all(t == final for t in tops[ex:])}", flush=True)
    print(f"exit histogram: {sorted(exits)} mean-saved={float(np.mean(saved)):.1f}/27 "
          f"({'VIABLE' if float(np.mean(saved)) >= 2 else 'thin'})", flush=True)


if __name__ == "__main__":
    main()
