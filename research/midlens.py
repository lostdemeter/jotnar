"""Mid-depth content lens: WHEN does target content enter the trajectory?

Lens_7b tracked the final pick's rank (smooth 114k->1, no boundary).
This tracks CONTENT: per-layer residual through final-norm + unembed
(fp64 host, LENS pattern), reporting top-5 + target-content ranks per
layer on Germany/Italy/Japan prompts. Questions: at which layer does
Paris/Rome/content enter top-k? Does content emerge gradually (like
rank) or at a boundary (unlike rank)? Same prompt, three contents =
which layers are content-specific vs shared substrate. Take-late/
skip-early (LENS) says late; this says exactly where per fact.
First probe of the no-black-boxes program: read the middle, don't
just place at endpoints.
Usage: python3 research/midlens.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

COUNTRIES = ["Germany", "Italy", "Japan"]
TARGETS = {"Germany": " Paris", "Italy": " Rome", "Japan": " Tokyo"}


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwdH, _rms
    import torch
    g, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    tids = {c: tok(t, return_tensors="pt")["input_ids"][0].tolist()[0]
            for c, t in TARGETS.items()}
    dt = torch.float32
    Wlog = torch.tensor(g("lm_head.weight"), dtype=dt, device="cuda")
    lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
    for c in COUNTRIES:
        traj, ids = fwdH(promp[c])
        n = len(ids)
        print(f"== {c} (target={TARGETS[c]!r})", flush=True)
        entered = {}
        for L in range(28):
            x = torch.tensor(traj[L], dtype=dt, device="cuda")
            hn = _rms(x, lnf)
            lg = (hn @ Wlog.T).detach().cpu().numpy()
            o = np.argsort(-lg)
            top5 = [(tok.decode([int(i)]), round(float(lg[int(i)]), 1))
                    for i in o[:5]]
            r = {name: int((lg > lg[tids[name]]).sum()) + 1
                 for name in COUNTRIES}
            mark = ""
            for name in COUNTRIES:
                if r[name] <= 10 and name not in entered:
                    entered[name] = L
                    mark += f" [{TARGETS[name]}-content enters top10 @L{L}]"
            print(f"  L{L:02}: top5={top5} ranks=" + str(r) + mark, flush=True)
        print(f"  entry layers: {entered}", flush=True)


if __name__ == "__main__":
    main()
