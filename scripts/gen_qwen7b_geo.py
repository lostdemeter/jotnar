"""Geometric Qwen2-7B vs HF original: same prompt, side by side.

Builder emits all 28 decoder layers (qwen_layer) + final norm + unembed
+ greedy sampler as ONE geometric program; CUDA/FP16 backend runs it
with weights in VRAM. HF transformers runs the original. Compares a
single forward (logits parity) then generates N tokens each.
Usage: python3 scripts/gen_qwen7b_geo.py "prompt here" [--n 20] [--smax 32]
KEYS (env): QWEN_SKIP_HF=1 (geo only), QWEN_SKIP_GEO=1 (HF only).
Slow: compile ~10min (10k ops), ~2-5s/token geo, HF load ~1min.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID, INTER, NH, NKV, DH = 3584, 18944, 28, 4, 128


def load_all():
    from chain.qwen7b import SNAP, load7b
    g, tok = load7b()
    return g, tok


def build_program(S):
    from chain.builder import Prog, qwen_layer
    p = Prog("qwen7b-28")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "cmask"]
    for L in range(N_LAYERS):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog"]
    p.inp(*names)
    x = "x0"
    for L in range(N_LAYERS):
        x = qwen_layer(p, x, f"wq{L}", f"wk{L}", f"wv{L}", f"wo{L}",
                       f"wup{L}", f"wgate{L}", f"wdown{L}", f"ln1{L}",
                       f"ln2{L}", "pos", "cmask", pre=str(L),
                       bq=f"bq{L}", bk=f"bk{L}", bv=f"bv{L}")
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The", "capital", "of"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--smax", type=int, default=32)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    g, tok = load_all()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)
    if len(ids) >= args.smax:
        sys.exit(f"prompt too long ({len(ids)} >= smax {args.smax})")

    # ---- HF original -------------------------------------------------
    hf_text, hf_logits = None, None
    if os.environ.get("QWEN_SKIP_HF") != "1":
        import torch
        from transformers import AutoModelForCausalLM
        from chain.qwen7b import SNAP
        os.environ["HF_HUB_OFFLINE"] = "1"
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda")
        print(f"HF load: {time.perf_counter() - t0:.0f}s", flush=True)
        model.eval()
        with torch.no_grad():
            inp = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=args.n, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        hf_text = tok.decode(out[0], skip_special_tokens=True)
        print("HF ORIGINAL:", hf_text, flush=True)
        with torch.no_grad():
            logits = model(**tok(prompt, return_tensors="pt").to("cuda")).logits
        hf_logits = logits[0, -1].float().cpu().numpy()
        del model
        torch.cuda.empty_cache()

    # ---- geometric build ----------------------------------------------
    if os.environ.get("QWEN_SKIP_GEO") == "1":
        return
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    S = args.smax
    pos = np.arange(S, dtype=np.int64)
    cmask = np.tril(np.ones((S, S), dtype=np.int64))
    prog = build_program(S)
    text = prog.text()
    print(f"program: {sum(1 for ln in text.splitlines() if '=' in ln and not ln.strip().startswith(('IN', 'CONFIG')))} ops",
          flush=True)
    # payload: embeddings for S slots (prompt filled, rest 0) + weights
    E = g("model.embed_tokens.weight")
    pos = np.arange(S, dtype=np.int64)
    cmask = np.tril(np.ones((S, S), dtype=np.int64))
    x0 = np.zeros((S, HID), dtype=np.float64)
    x0[:len(ids)] = E[np.array(ids)]
    sample, dump = {"pos": pos, "cmask": cmask, "x0": x0}, {}
    dump["x0"] = x0
    dump["pos"] = pos
    dump["cmask"] = cmask
    keymap = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
              "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
              "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
              "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
              "ln2": "post_attention_layernorm.weight",
              "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
              "bv": "self_attn.v_proj.bias"}
    for L in range(N_LAYERS):
        for k, src in keymap.items():
            w = g(f"model.layers.{L}.{src}")
            if k in ("bq", "bk", "bv"):
                wt = np.ascontiguousarray(np.tile(w, (S, 1)))
            elif k in ("ln1", "ln2"):
                wt = np.ascontiguousarray(w)
            else:
                wt = np.ascontiguousarray(w.T)
            if k == "wq":
                wt = wt / np.sqrt(128)  # temperature fold (exact)
            sample[f"{k}{L}"] = wt
            dump[f"{k}{L}"] = wt
    lnf = g("model.norm.weight")
    wlog = g("lm_head.weight")
    sample["lnf"] = np.ascontiguousarray(lnf)
    sample["wlog"] = np.ascontiguousarray(wlog.T)
    dump["lnf"] = sample["lnf"]
    dump["wlog"] = sample["wlog"]
    # temperature fold needs the bias scaled too (measured 0.5 rel if not)
    for L in range(N_LAYERS):
        sample[f"bq{L}"] = sample[f"bq{L}"] / np.sqrt(128)
        dump[f"bq{L}"] = sample[f"bq{L}"]
    t0 = time.perf_counter()
    art = compile_program(text, "cuda", sample=sample,
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True)
    print(f"compile graph: {time.perf_counter() - t0:.0f}s", flush=True)
    work = "/tmp/gen7b"
    os.makedirs(work, exist_ok=True)
    exe = os.path.join(work, "q7b")
    if os.environ.get("QWEN_REUSE_BIN") != "1":
        t0 = time.perf_counter()
        exe = build_cu(art["source"], work, name="q7b")
        print(f"nvcc: {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        print("nvcc: reused binary", flush=True)
    # dump frozen inputs once (F16)
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"fr_{n}.bin")
        if k == "F":
            dump[n].astype(np.float16).tofile(fn)
        else:
            dump[n].astype(np.int64).tofile(fn)

    def step(tok_ids, poss):
        n = len(tok_ids)
        argv = [exe]
        for name in art["inputs"]:
            k, _, _ = art["streams"][name]
            if name == "x0":
                fn = os.path.join(work, "step_x0.bin")
                xx = np.zeros((S, HID), dtype=np.float16)
                xx[:n] = E[np.array(tok_ids)].astype(np.float16)
                xx.tofile(fn)
            elif name == "pos":
                fn = os.path.join(work, "step_pos.bin")
                np.array(poss, dtype=np.int64).tofile(fn)
            elif name == "cmask":
                fn = os.path.join(work, "step_cm.bin")
                np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
            elif k == "T":
                continue  # no T streams in this program
            else:
                fn = os.path.join(work, f"fr_{name}.bin")
            argv.append(fn)
        flo = os.path.join(work, "step_lg.bin")
        fou = os.path.join(work, "step_ou.bin")
        argv += [flo, fou]
        # NOTE: x0/pos/cmask are S-padded; model reads all S rows every
        # step (no KV cache in v0.1 -- full recompute, stated).
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo step failed rc={r.returncode}: "
                               f"{r.stderr[:300]}")
        lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
        ou = np.fromfile(fou, dtype=np.int64)
        return lg, ou

    # ---- single-forward parity -----------------------------------------
    lg, ou = step(ids, list(range(len(ids))) + [0] * (S - len(ids)))
    # careful: pos passed full-S with padding zeros; restrict below.
    print(f"geo forward: LOGITS {lg.shape} OUT {ou.shape}", flush=True)
    if hf_logits is not None:
        # compare at the last REAL position (n-1); padding rows diverge
        # legitimately (untrained positions -- stated, not gated).
        n = len(ids)
        dmax = float(np.abs(lg[n - 1] - hf_logits).max())
        print(f"parity @last-prompt-pos: maxabs={dmax:.3e}", flush=True)
    else:
        dmax = None

    # ---- generation loop -------------------------------------------------
    seq = list(ids)
    abspos = list(range(len(ids)))
    t0 = time.perf_counter()
    for _ in range(args.n):
        if len(seq) >= S:
            seq, abspos = seq[1:], abspos[1:]
            # NOTE: sliding window with ABSOLUTE positions would need
            # re-baked RoPE tables; v0.1 keeps window < smax so no slide
            # fires for the default prompt+n (stated limit).
            break
        lg, ou = step(seq, abspos)
        seq.append(int(ou[len(seq) - 1]))
        abspos.append(abspos[-1] + 1)
    dt = time.perf_counter() - t0
    print(f"geo generated {len(seq) - len(ids)} toks in {dt:.0f}s", flush=True)
    geo_text = tok.decode(seq, skip_special_tokens=True)
    print("GEOMETRIC:", geo_text, flush=True)
    if args.out_json:
        import json as _json
        rec = {"geo_text": geo_text, "parity": dmax, "geo_first_rank": None}
        # geo first pick rank inside HF distribution at prompt end
        if hf_logits is not None and len(seq) > len(ids):
            _gfirst = seq[len(ids)]
            rec["geo_first_rank"] = int((hf_logits > hf_logits[_gfirst]).sum()) + 1
        _json.dump(rec, open(args.out_json, "w"))


if __name__ == "__main__":
    main()
