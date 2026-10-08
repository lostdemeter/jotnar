"""L14 Paris-specialists: fingerprints, not volume (36-dir sweep).

Energy screen (read14.py): top-12 movers all broadcast (tops frozen,
Paris r50-65). Selectivity lives elsewhere: for every ~100th SVD dir
of L14 down_proj (+ dedup), one ablation run records global dB AND
Paris-token dB AND top mover. Specialist score = mean(tokdb) -
paris-tokdb (large positive = moves Paris, little else). The runs
already compute full logit vectors -- fingerprints are free once the
forward runs. Top-5 selective reported with verify numbers.
A Paris-specialist at L14 is a content lever inside the transport
(addressable mid-depth content, not mass, not rank).
Usage: python3 research/select14.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

LAYER = 14
PROMPT = "The capital of Germany is"
STEP = 100


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwd_mid, _get, _layer, _rms
    import torch
    torch, g, tok = _get()
    _, tok = load7b()
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    lg0, _ = fwd(PROMPT)
    base_top = int(lg0.argmax())
    mid, Wd = fwd_mid(PROMPT, LAYER)
    U, s, Vt = np.linalg.svd(np.ascontiguousarray(Wd, dtype=np.float64),
                              full_matrices=False)
    dt = torch.float32

    def ablate(di):
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
                                                  device="cuda"),
                                     torch.tensor(Vt[di], dtype=dt, device="cuda"))
                edits = {L: {"wdown": Wi}}
            x = _layer(x, g, torch, L, n, edits=edits)
        lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
        hn = _rms(x, lnf)
        out = (hn @ torch.tensor(g("lm_head.weight"), dtype=dt,
                                 device="cuda").T)[n - 1]
        return out.detach().cpu().numpy()

    seen = set()
    dirs = []
    for di in range(0, len(s), STEP):
        dirs.append(di)
        seen.add(di)
    rows = []
    for di in dirs:
        got = ablate(int(di))
        d2 = (got - lg0) ** 2
        with np.errstate(divide="ignore"):
            tdb = np.array([float("inf") if v == 0 else 10 * np.log10(1.0 / v)
                            for v in d2])
        finite = tdb[np.isfinite(tdb)]
        gdb = float(10 * np.log10(1.0 / float(((got - lg0) ** 2).mean())))
        spec = float(np.mean(finite) - tdb[paris]) if np.isfinite(tdb[paris]) else -1e9
        top = int(got.argmax())
        rows.append((spec, int(di), gdb, float(tdb[paris]),
                     tok.decode([base_top]), tok.decode([top])))
        print(f"dir{di}: spec={spec:+.1f} global={gdb:.1f}dB "
              f"paris-tok={float(tdb[paris]):.1f}dB "
              f"{tok.decode([base_top])!r}->{tok.decode([top])!r}", flush=True)
    rows.sort(reverse=True)
    print("TOP-5 Paris-specialists (spec, dir, global, paris-tok, move):",
          flush=True)
    for r in rows[:5]:
        print(f"  spec={r[0]:+.1f} dir{r[1]} global={r[2]:.1f} "
              f"paris={r[3]:.1f} {r[4]!r}->{r[5]!r}", flush=True)


if __name__ == "__main__":
    main()
