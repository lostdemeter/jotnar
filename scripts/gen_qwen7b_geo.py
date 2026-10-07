"""Geometric Qwen2-7B vs HF original: same prompt, side by side.

Builder emits all 28 decoder layers (qwen_layer) + final norm + unembed
+ greedy sampler as ONE geometric program; CUDA/FP16 backend runs it
with weights in VRAM. HF transformers runs the original. Compares a
single forward (logits parity) then generates N tokens each.
Usage: python3 scripts/gen_qwen7b_geo.py "prompt here" [--n 20] [--smax 32]
MODES (default: geo only): --hf-only (HF only), --geo-only (geo only);
both in one process needs --allow-combined (refused by default: HF +
geo weights together exceeded 61GiB physical and killed an SSD --
see chain/mem.py). Env keys QWEN_SKIP_HF/QWEN_SKIP_GEO still honored.
Memory: prints chain/mem.py budget first, refuses over-cap plans;
weights stream to disk one fp16 weight at a time (peak ~2.5GiB CPU).
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


def build_program(S, bmmv=False):
    from chain.builder import Prog, qwen_layer, qwen_bmmv_layer
    p = Prog("qwen7b-28")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "cmask"]
    if bmmv:
        names.append("cmask7")
    for L in range(N_LAYERS):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog"]
    p.inp(*names)
    layer = qwen_bmmv_layer if bmmv else qwen_layer
    x = "x0"
    for L in range(N_LAYERS):
        kw = {} if not bmmv else {"cmask7": "cmask7", "S": S}
        x = layer(p, x, f"wq{L}", f"wk{L}", f"wv{L}", f"wo{L}",
                  f"wup{L}", f"wgate{L}", f"wdown{L}", f"ln1{L}",
                  f"ln2{L}", "pos", "cmask", pre=str(L),
                  bq=f"bq{L}", bk=f"bk{L}", bv=f"bv{L}", **kw)
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
    ap.add_argument("--hf-only", action="store_true",
                    help="HF original only (no geo build)")
    ap.add_argument("--geo-only", action="store_true",
                    help="geometric only (default; no HF load)")
    ap.add_argument("--allow-combined", action="store_true",
                    help="permit HF+geo in one process (refused by default)")
    ap.add_argument("--oneshot", action="store_true",
                    help="fresh process per step (retired reload path; "
                    "default is persistent serve, weights-load-once)")
    ap.add_argument("--no-sync", action="store_true",
                    help="drop per-op device-sync (same-stream ordering; "
                    "stores still sync before host reads)")
    ap.add_argument("--graph", action="store_true",
                    help="capture the step into a CUDA graph (single replay "
                    "launch; needs serve mode, refuses time_ops)")
    ap.add_argument("--bmmv", action="store_true",
                    help="head-batched attention (BMMV, 48th op): same math, "
                    "bit-exact, fewer cublas calls")
    ap.add_argument("--mem-cap-gb", type=float, default=None,
                    help="physical-RAM cap override (default 60%% of phys)")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    # Default is geo-only. Combined HF+geo in one process is opt-in
    # (both weight sets at once exceeded physical RAM and killed an SSD).
    # Env keys preserved: QWEN_SKIP_HF=1 -> geo, QWEN_SKIP_GEO=1 -> HF.
    skip_hf_env = os.environ.get("QWEN_SKIP_HF") == "1"
    skip_geo_env = os.environ.get("QWEN_SKIP_GEO") == "1"
    want_hf = args.hf_only or (args.allow_combined and not skip_hf_env)
    want_geo = not args.hf_only and not skip_geo_env
    if not want_hf and not want_geo:
        sys.exit("nothing to do (both sides skipped)")
    if want_hf and want_geo and not args.allow_combined:
        sys.exit("refusing HF+geo in one process without --allow-combined "
                 "(combined weights exceeded physical RAM and killed an SSD; "
                 "run --hf-only and --geo-only separately)")

    from chain import mem as _mem
    from chain.qwen7b import snapshot_ok
    cap = int(args.mem_cap_gb * 2 ** 30) if args.mem_cap_gb else None
    print(_mem.report(args.smax, cap_bytes=cap), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs Qwen2-7B-Instruct snapshot; "
                 "no weights re-downloaded since the SSD loss)")
    if want_geo:
        _mem.guard(_mem.estimate_7b(args.smax)["stream_peak_cpu"],
                   "geo streaming build", cap_bytes=cap)
    g, tok = load_all()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)
    if len(ids) >= args.smax:
        sys.exit(f"prompt too long ({len(ids)} >= smax {args.smax})")

    # ---- HF original -------------------------------------------------
    hf_text, hf_logits = None, None
    if want_hf:
        import torch
        from transformers import AutoModelForCausalLM
        from chain.qwen7b import SNAP, hf_no_triton
        hf_no_triton()  # aten reference numerics; triton JIT needs dev headers
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
    if not want_geo:
        return
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    from chain.qwen7b import (shapes_7b, prep_7b_weight, clear_7b_cache,
                             input_sources)
    S = args.smax
    prog = build_program(S, bmmv=args.bmmv)
    text = prog.text()
    print(f"program: {sum(1 for ln in text.splitlines() if '=' in ln and not ln.strip().startswith(('IN', 'CONFIG')))} ops",
          flush=True)
    # payload: shapes are stubs (zero RAM); values stream straight to
    # disk one fp16 weight at a time (the retired bulk-float64 sample
    # held ~61GiB and killed an SSD -- chain/mem.py guard above).
    sample = shapes_7b(S)
    live = None if args.oneshot else ["x0", "pos", "cmask"]
    t0 = time.perf_counter()
    art = compile_program(prog.text(), "cuda", sample=sample,
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=live,
                          sync_each=not args.no_sync, graph=args.graph)
    print(f"compile graph: {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"mode: {'one-shot (reload per step)' if args.oneshot else 'serve (weights-load-once)'}"
          f" / {'per-op sync' if not args.no_sync else 'stream sync only'}",
          flush=True)
    work = os.environ.get("QWEN_WORKDIR", "/tmp/gen7b")
    os.makedirs(work, exist_ok=True)
    E = np.ascontiguousarray(g("model.embed_tokens.weight"),
                             dtype=np.float16)  # 1.1GiB resident for steps
    # Exact-match lookup (never parse trailing digits: ln1@L0 is "ln10",
    # ln1@L10 is "ln110"). Unknown inputs fail loud below.
    by_input = input_sources()
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"fr_{n}.bin")
        if n == "x0":
            np.zeros((S, HID), dtype=np.float16).tofile(fn)
        elif n == "pos":
            np.arange(S, dtype=np.int64).tofile(fn)
        elif n == "cmask":
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        elif n == "cmask7":
            # frozen tiled mask: per=7 copies of tril(S,S) stacked (7S,S)
            cm1 = np.tril(np.ones((S, S), dtype=np.int64))
            np.tile(cm1, (7, 1)).tofile(fn)
        elif n == "lnf":
            prep_7b_weight(g, by_input["lnf"][1], "lnf", S).tofile(fn)
        elif n == "wlog":
            prep_7b_weight(g, by_input["wlog"][1], "wlog", S).tofile(fn)
        elif n in by_input:
            short, src = by_input[n]
            w16 = prep_7b_weight(g, src, short, S)
            if k == "F":
                w16.tofile(fn)
            else:
                w16.astype(np.int64).tofile(fn)
            del w16
        else:
            raise KeyError(f"no weight source for input stream {n!r}")
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    import json as _js
    import hashlib as _hl
    # Stamp covers flags AND program text (input-count changes from new
    # streams like cmask7 must invalidate reuse: argc mismatches are rc=9,
    # silent-wrong-shape is worse -- fail loud here instead).
    stamp = {"bmmv": bool(args.bmmv), "graph": bool(args.graph),
             "no_sync": bool(args.no_sync), "oneshot": bool(args.oneshot),
             "smax": S, "live": live,
             "prog": _hl.sha256(prog.text().encode()).hexdigest()[:16]}
    exe = os.path.join(work, "q7b")
    if os.environ.get("QWEN_REUSE_BIN") != "1":
        t0 = time.perf_counter()
        exe = build_cu(art["source"], work, name="q7b")
        print(f"nvcc: {time.perf_counter() - t0:.0f}s", flush=True)
        _js.dump(stamp, open(os.path.join(work, "build.json"), "w"))
    else:
        try:
            old = _js.load(open(os.path.join(work, "build.json")))
        except OSError:
            old = {}
        bad = [k for k in stamp if old.get(k) != stamp[k]]
        if bad:
            sys.exit(f"refusing QWEN_REUSE_BIN=1: stale binary differs on "
                     f"{bad} (rebuild without the env key)")
        print("nvcc: reused binary (stamp verified)", flush=True)

    flo = os.path.join(work, "step_lg.bin")
    fou = os.path.join(work, "step_ou.bin")

    def write_live(tok_ids, poss):
        # Per-step files, byte-identical either mode (x0 rows live first,
        # the rest zeros; pos FULL arange -- padding rows need valid
        # positions (rotary reads all S rows; short files read OOB heap,
        # benign at S<=128 by luck, biting at 512); cmask full).
        n = len(tok_ids)
        xx = np.zeros((S, HID), dtype=np.float16)
        xx[:n] = E[np.array(tok_ids)].astype(np.float16)
        xx.tofile(os.path.join(work, "step_x0.bin"))
        np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
        np.tril(np.ones((S, S), dtype=np.int64)).tofile(
            os.path.join(work, "step_cm.bin"))

    def read_out():
        lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
        ou = np.fromfile(fou, dtype=np.int64)
        return lg, ou

    def live_argv():
        argv = [exe]
        for name in art["inputs"]:
            k, _, _ = art["streams"][name]
            if name in ("x0", "pos", "cmask"):
                fn = os.path.join(work, {"x0": "step_x0.bin",
                                         "pos": "step_pos.bin",
                                         "cmask": "step_cm.bin"}[name])
            elif k == "T":
                continue  # no T streams in this program
            else:
                fn = os.path.join(work, f"fr_{name}.bin")
            argv.append(fn)
        return argv + [flo, fou]

    sv = None
    if args.oneshot:
        def step(tok_ids, poss):
            write_live(tok_ids, poss)
            # NOTE: x0/pos/cmask are S-padded; model reads all S rows
            # every step (no KV cache in v0.1 -- full recompute, stated).
            r = subprocess.run(live_argv(), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"geo step failed rc={r.returncode}: "
                                   f"{r.stderr[:300]}")
            return read_out()
    else:
        from chain.serve import ServedExe
        write_live(ids, list(range(len(ids))))
        argv_files = live_argv()[1:]
        sv = ServedExe(exe, argv_files,
                       err_path=os.path.join(work, "serve.err"))
        t0 = time.perf_counter()
        sv.start()
        print(f"serve ready: weights resident after {time.perf_counter() - t0:.0f}s",
              flush=True)

        def step(tok_ids, poss):
            write_live(tok_ids, poss)
            t0 = time.perf_counter()
            sv.step()
            dt = time.perf_counter() - t0
            print(f"serve step: {dt * 1000:.0f}ms (warm, no reload)",
                  flush=True)
            return read_out()

    # ---- single-forward parity -----------------------------------------
    lg, ou = step(ids, list(range(len(ids))) + [0] * (S - len(ids)))
    # careful: pos passed full-S with padding zeros; restrict below.
    print(f"geo forward: LOGITS {lg.shape} OUT {ou.shape}", flush=True)
    n0 = len(ids)
    lg0 = np.ascontiguousarray(lg[n0 - 1])  # prompt-end row (parity sidecar)
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
        rec = {"geo_text": geo_text, "parity": dmax, "geo_first_rank": None,
               "geo_first_id": seq[len(ids)] if len(seq) > len(ids) else None,
               "logits_file": None}
        # geo first pick rank inside HF distribution at prompt end (only
        # when HF ran in-process, i.e. --allow-combined; the gate test
        # ranks host-side from geo_first_id instead).
        if hf_logits is not None and len(seq) > len(ids):
            _gfirst = seq[len(ids)]
            rec["geo_first_rank"] = int((hf_logits > hf_logits[_gfirst]).sum()) + 1
        if hf_logits is None:
            # geo-only: leave the prompt-end logits row beside the JSON
            # so the host (which owns HF) can check parity itself.
            rec["logits_file"] = args.out_json + ".promptlogits.npy"
            np.save(rec["logits_file"], lg0)
        _json.dump(rec, open(args.out_json, "w"))
    if sv is not None:
        sv.close()


if __name__ == "__main__":
    main()
