"""Per-layer lens on geometric Qwen2-7B: what each layer decides.

Same 28-layer program, but every layer residual Y{L} is an output;
host reads the trajectory through final-norm + unembed and reports,
per layer at the prompt-end row: top-1, entropy, rank of the final
answer, agreement with the final pick. Tests the funnel/filter shape
directly (broad early? sharpening late? where does the pick lock in?).
Usage: python3 scripts/lens_7b.py "prompt here" [--smax 16]
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID = 3584


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The", "capital", "of"])
    ap.add_argument("--smax", type=int, default=16)
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    S = args.smax
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from gen_qwen7b_geo import build_program
    from chain.qwen7b import shapes_7b, load7b, snapshot_ok, clear_7b_cache
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    if not snapshot_ok():
        sys.exit("SKIP (needs snapshot)")
    g, tok = load7b()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    if len(ids) >= S:
        sys.exit("prompt too long")
    prog = build_program(S, bmmv=True)
    outs = [f"Y{L}" for L in range(N_LAYERS)]
    # LOGITS/OUT ride along: the emitter declares ARGMAX outputs only
    # as program outputs (intermediate I streams get no buffer).
    art = compile_program(prog.text(), "cuda", sample=shapes_7b(S),
                          outputs=outs + ["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True)
    work = "/tmp/lens7b"
    os.makedirs(work, exist_ok=True)
    exe = build_cu(art["source"], work, name="lens")
    E = g("model.embed_tokens.weight")
    argv = [exe]
    for name in art["inputs"]:
        k, _, _ = art["streams"][name]
        fn = os.path.join(work, f"fr_{name}.bin")
        if name == "x0":
            xx = np.zeros((S, HID), dtype=np.float16)
            xx[:len(ids)] = E[np.array(ids)].astype(np.float16)
            xx.tofile(fn)
        elif name == "pos":
            np.arange(S, dtype=np.int64).tofile(fn)
        elif name == "cmask":
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        elif name == "cmask7":
            cm1 = np.tril(np.ones((S, S), dtype=np.int64))
            np.tile(cm1, (7, 1)).tofile(fn)
        else:
            from chain.qwen7b import input_sources, prep_7b_weight
            keymap = dict(input_sources())
            short, src = keymap[name]
            prep_7b_weight(g, src, short, S).tofile(fn)
        argv.append(fn)
    for o in art["outputs"]:
        argv.append(os.path.join(work, f"out_{o}.bin"))
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"lens failed rc={r.returncode}: {r.stderr[:300]}")
    import torch
    lnf = torch.tensor(g("model.norm.weight"), dtype=torch.float64)
    wlog = torch.tensor(g("lm_head.weight").T, dtype=torch.float64)
    eps = 1e-6
    n = len(ids)
    print(f"{'L':>3} {'top1':>8} {'ent':>6} {'rankFinal':>9} {'agree':>5}",
          flush=True)
    finals = []
    rows = []
    for L in range(N_LAYERS):
        Y = np.fromfile(os.path.join(work, f"out_Y{L}.bin"),
                        dtype=np.float32).reshape(S, HID)[n - 1]
        h = torch.tensor(Y, dtype=torch.float64)
        hn = h / torch.sqrt((h ** 2).mean() + eps) * lnf
        lg = (hn @ wlog).numpy()
        p = np.exp(lg - lg.max())
        p /= p.sum()
        ent = float(-(p * np.log(p + 1e-300)).sum())
        top = int(lg.argmax())
        rows.append(lg)
        finals.append(top)
        print(f"{L:3d} {tok.decode([top])[:8]!r:>8} {ent:6.2f}", flush=True)
    final = finals[-1]
    agree = [t == final for t in finals]
    for L in range(N_LAYERS):
        rk = int((rows[L] > rows[L][final]).sum()) + 1
        print(f"L{L:02d} rankFinal={rk} agree={agree[L]}", flush=True)
    lock = next((L for L, a in enumerate(agree) if a and
                 all(agree[L:])), None)
    print(f"final pick {tok.decode([final])!r} locks in at L={lock} "
          f"({N_LAYERS - (lock or 0)} layers of holding)", flush=True)


if __name__ == "__main__":
    main()
