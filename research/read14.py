"""L14 content levers: predictor + verify (no 2000-run sweep).

Full direction_readout over 18944 dirs is unaffordable here; the
MODEL_READ factorization makes it unnecessary: ablation dB =
-20log(s) - 10log(alignment^2) + C, computable from STATICS (one SVD
+ one MID) with zero runs. This predicts L14 down_proj's top movers
of the Germany prompt-end output, then VERIFIES the top-3 by
ablation (measured dB + which token moves + selectivity spread).
A verified mover at L14 is a content lever inside the transport --
the first representation-content (not mass, not rank) datum
mid-depth. Selectivity decides broadcasters vs specialists.
Usage: python3 research/read14.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

LAYER = 14
PROMPT = "The capital of Germany is"
TOPK = 3


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwd_mid
    _, tok = load7b()
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    lg0, ids0 = fwd(PROMPT)
    base_top = int(lg0.argmax())
    print(f"unsteered top={tok.decode([base_top])!r}", flush=True)

    mid, Wd = fwd_mid(PROMPT, LAYER)
    print(f"MID norm={float(np.linalg.norm(mid)):.1f} Wd{Wd.shape}", flush=True)
    U, s, Vt = np.linalg.svd(np.ascontiguousarray(Wd, dtype=np.float64),
                              full_matrices=False)
    align = np.abs(mid @ U) / (np.linalg.norm(mid) + 1e-12)
    # predicted dB up to constant C: rank by s*align (energy x match)
    score = s * align.ravel()
    order = np.argsort(-score)[:50]
    # calibrate C on 3 spread dirs, then predict top candidates
    import torch
    from qwen_torch import _get, _layer
    _torch, g, _tok = _get()

    def run_ablated(di):
        dt = torch.float32
        tids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy()
        n = len(tids)
        E = torch.tensor(g("model.embed_tokens.weight")[tids], dtype=dt,
                         device="cuda")
        x = E
        for L in range(28):
            edits = None
            if L == LAYER:
                W = torch.tensor(g(f"model.layers.{L}.mlp.down_proj.weight"),
                                 dtype=dt, device="cuda")
                Wi = W - torch.outer(torch.tensor(U[:, di] * s[di], dtype=dt,
                                                  device="cuda),
                                     torch.tensor(Vt[di], dtype=dt, device="cuda"))
                edits = {L: {"wdown": Wi}}
            x = _layer(x, g, torch, L, n, edits=edits)
        from qwen_torch import _rms
        lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
        hn = _rms(x, lnf)
        out = (hn @ torch.tensor(g("lm_head.weight"), dtype=dt,
                                 device="cuda").T)[n - 1]
        return out.detach().cpu().numpy()

    cal = []
    for di in [0, len(s) // 2, len(s) - 1]:
        got = run_ablated(int(di))
        mse = float(((got - lg0) ** 2).mean())
        gdb = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        cal.append(gdb + 20 * np.log10(s[di])
                   + 10 * np.log10(align.ravel()[di] ** 2 + 1e-30))
    C = float(np.mean(cal))
    print(f"calibration C={C:.2f} (spread dirs 0/mid/tail)", flush=True)
    pred = {int(i): float(-20 * np.log10(s[i])
                          - 10 * np.log10(align.ravel()[i] ** 2 + 1e-30) + C)
            for i in order[:12]}
    for di, pd in sorted(pred.items(), key=lambda kv: kv[1])[:TOPK + 2]:
        print(f"candidate dir{di}: pred {pd:.1f}dB sval={s[di]:.2f} "
              f"align={align.ravel()[di]:.3f}", flush=True)
    for di in sorted(pred, key=lambda k: pred[k])[:TOPK]:
        got = run_ablated(int(di))
        mse = float(((got - lg0) ** 2).mean())
        gdb = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        top = int(got.argmax())
        print(f"VERIFY dir{di}: meas {gdb:.1f}dB (pred {pred[di]:.1f}) "
              f"top {tok.decode([base_top])!r}->{tok.decode([top])!r} "
              f"paris-rank={int((got > got[paris]).sum()) + 1}", flush=True)


if __name__ == "__main__":
    main()
