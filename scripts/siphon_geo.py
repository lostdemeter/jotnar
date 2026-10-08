"""Siphon install as a geometric program (teacher side, CUDA/FP16).

Same ball as research/siphon_ball.py (L27 pair-contrast content x L2
lexical-early address, null stores for non-targets, ledger), but the
apply runs IN-LISTING: builder emits all 28 Qwen2-7B layers plus
SLICE(H2,row3) -> CALL yarnball_apply -> tile -> ADD at L27 output
(explicit routing: addr_sep.py proved the country row separates 1.0
vs <=0.64 while the prompt-end row ties -- per-row self-gating would
misroute). Dose mapping: mirror gain g ==> Vc row = d*g*|x27| (the
mirror multiplies by residual mag; the listing adds directly).
Gates: Germany top=Paris; Italy/Japan == mirror unsteered; own-store
retrieval is read from the geo P stream dump (outputs += YBP).
Usage: python3 scripts/siphon_geo.py [--gain 1] [--smax 8]
Slow: mirror mine (~10min, weight reloads) + compile ~10min + steps.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

N_LAYERS = 28
HID, INTER, NH, NKV, DH = 3584, 18944, 28, 4, 128
COUNTRIES = ["Germany", "Italy", "Japan"]
TARGET = "Germany"


def build_program(S, bmmv=False):
    from chain.builder import Prog, qwen_layer
    p = Prog("qwen7b-28-siphon")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    # NOTE: yarnball_apply body inlined (not CALLed): emit_cuda cannot
    # lower CALL-namespaced streams (yarnball_apply#1.P is not a valid C
    # identifier -- toolchain gap, filed). Ops below ARE the DEF body
    # (stdlib/yarnball.asm), same math, explicit names.
    names = ["x0", "pos", "cmask"]
    for L in range(N_LAYERS):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            names.append(f"{k}{L}")
    names += ["lnf", "wlog", "Ua27", "Vc27", "ones51"]
    p.inp(*names)
    x = "x0"
    for L in range(N_LAYERS):
        x = qwen_layer(p, x, f"wq{L}", f"wk{L}", f"wv{L}", f"wo{L}",
                       f"wup{L}", f"wgate{L}", f"wdown{L}", f"ln1{L}",
                       f"ln2{L}", "pos", "cmask", pre=str(L),
                       bq=f"bq{L}", bk=f"bk{L}", bv=f"bv{L}")
        if L == 27:
            # Explicit routing (addr_sep.py): the country row (3) separates
            # (own 1.0 vs <=0.64); the prompt-end row (4) does NOT (tied
            # ~0.42, Japan slightly ahead). So address = SLICE of row 3,
            # value lands tiled (only row 4 is read downstream: post-L27
            # ops are all row-wise, rows 0-3/5+ unread -- stated, airtight).
            # Geometry (len 5, country@3, end@4) uniform across the battery:
            # one binary serves all three prompts.
            xa = p.op("SLICE", "H2", 0, 3, 4, out="XA27")
            ybc = p.op("MATMUL", xa, "Ua27", out="YBC27")
            ybs = p.op("TSHIFT", ybc, out="YBS27")
            ybp = p.op("SOFTMAX_WIDE", ybs, out="YBP27")
            yb = p.op("MATMUL", ybp, "Vc27", out="YB27")
            ybt = p.op("MATMUL", "ones51", yb, out="YBT27")
            x = p.op("ADD", x, ybt, out="X27s")
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--smax", type=int, default=8)
    ap.add_argument("--key-scale", type=float, default=8.0)
    ap.add_argument("--out-json", type=str, default="/tmp/siphon_geo.json")
    ap.add_argument("--base-json", type=str, default=None,
                    help="geo zero-dose arms for hold grading (fork doctrine:"
                    " geo FP16 base differs from mirror on near-ties, so"
                    " holds grade vs geo-base, not mirror; make it with"
                    " --gain 0)")
    args = ap.parse_args()

    from chain import mem as _mem
    from chain.qwen7b import snapshot_ok
    print(_mem.report(args.smax), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs Qwen2-7B-Instruct snapshot)")
    _mem.guard(_mem.estimate_7b(args.smax)["stream_peak_cpu"], "geo siphon build")

    # ---- mirror mine (live; no caches) --------------------------------
    from chain.qwen7b import load7b, hf_no_triton
    hf_no_triton()
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from chain.engram import yarnball_bank
    g, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(promp[c])
        base[c] = int(lg.argmax())
    print("mirror unsteered:", {c: tok.decode([v]) for c, v in base.items()},
          flush=True)
    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    d = tfr[27] - tde[27]
    d /= np.linalg.norm(d)
    mag27 = float(np.linalg.norm(tde[27]))
    print(f"|x27| Germany prompt-end = {mag27:.1f}", flush=True)
    keys = {c: early_key(promp[c])[0] for c in COUNTRIES}
    D = HID
    Ua, Vc, ledger = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": keys["Germany"], "value": d, "dose": args.gain * mag27,
          "tier": "assoc", "support": "L27 France-minus-Germany contrast"},
         {"key": keys["Italy"], "value": np.zeros(D),
          "tier": "null", "support": "background"},
         {"key": keys["Japan"], "value": np.zeros(D),
          "tier": "null", "support": "background"}],
        key_scale=args.key_scale)
    json.dump([{"tier": r["tier"], "support": r["support"],
                "dose": r["dose"], "sha": r["sha"]} for r in ledger],
              open("/tmp/siphon_geo_ledger.json", "w"), indent=2)
    print(f"ball: {Ua.shape[1]} stores dose={args.gain}*mag27="
          f"{args.gain * mag27:.1f}", flush=True)
    del g
    import gc as _gc
    _gc.collect()

    # ---- geometric build ----------------------------------------------
    from chain.emit_c import compile_program, FRef
    from chain.emit_cuda import build_cu
    from chain.qwen7b import shapes_7b, prep_7b_weight, clear_7b_cache, input_sources
    g, tok = load7b()
    S = args.smax
    prog = build_program(S)
    text = prog.text()
    print(f"program: {sum(1 for ln in text.splitlines() if '=' in ln and not ln.strip().startswith(('IN', 'CONFIG', 'IMPORT')))} ops",
          flush=True)
    sample = shapes_7b(S)
    sample["Ua27"] = FRef((HID, 3), "float16")
    sample["Vc27"] = FRef((3, HID), "float16")
    sample["ones51"] = FRef((S, 1), "float16")
    live = ["x0", "pos", "cmask"]
    t0 = time.perf_counter()
    art = compile_program(text, "cuda", sample=sample,
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=live, sync_each=False)
    print(f"compile graph: {time.perf_counter() - t0:.0f}s", flush=True)
    work = os.environ.get("QWEN_WORKDIR", "/tmp/gen7b_siphon")
    os.makedirs(work, exist_ok=True)
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    by_input = input_sources()
    _alias = art.get("cname", {})
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"fr_{n}.bin")
        if n == "x0":
            np.zeros((S, HID), dtype=np.float16).tofile(fn)
        elif n == "pos":
            np.arange(S, dtype=np.int64).tofile(fn)
        elif n == "cmask":
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        elif n == "Ua27":
            np.ascontiguousarray(Ua, dtype=np.float16).tofile(fn)
        elif n == "Vc27":
            np.ascontiguousarray(Vc, dtype=np.float16).tofile(fn)
        elif n == "ones51":
            np.ones((S, 1), dtype=np.float16).tofile(fn)
        elif n == "lnf":
            prep_7b_weight(g, by_input["lnf"][1], "lnf", S).tofile(fn)
        elif n == "wlog":
            prep_7b_weight(g, by_input["wlog"][1], "wlog", S).tofile(fn)
        elif _alias.get(n, n) in by_input:
            short, src = by_input[_alias.get(n, n)]
            w16 = prep_7b_weight(g, src, short, S)
            (w16 if k == "F" else w16.astype(np.int64)).tofile(fn)
            del w16
        else:
            raise KeyError(f"no weight source for input stream {n!r}")
    clear_7b_cache(g)
    _gc.collect()
    exe = build_cu(art["source"], work, name="q7b_siphon")
    print("nvcc done", flush=True)

    from chain.serve import ServedExe
    flo = os.path.join(work, "step_lg.bin")
    fou = os.path.join(work, "step_ou.bin")

    def write_live(tok_ids):
        n = len(tok_ids)
        xx = np.zeros((S, HID), dtype=np.float16)
        xx[:n] = E[np.array(tok_ids)].astype(np.float16)
        xx.tofile(os.path.join(work, "step_x0.bin"))
        np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
        np.tril(np.ones((S, S), dtype=np.int64)).tofile(os.path.join(work, "step_cm.bin"))

    def live_argv():
        argv = [exe]
        for name in art["inputs"]:
            k, _, _ = art["streams"][name]
            if name in ("x0", "pos", "cmask"):
                fn = os.path.join(work, {"x0": "step_x0.bin",
                                         "pos": "step_pos.bin",
                                         "cmask": "step_cm.bin"}[name])
            elif k == "T":
                continue
            else:
                fn = os.path.join(work, f"fr_{name}.bin")
            argv.append(fn)
        return argv + [flo, fou]

    rec = {"base": {c: tok.decode([v]) for c, v in base.items()}, "arms": {}}
    pids = {c: tok(promp[c], return_tensors="pt")["input_ids"][0].numpy().tolist()
            for c in COUNTRIES}
    print("prompt lens:", {c: len(v) for c, v in pids.items()}, flush=True)
    write_live(pids["Germany"])
    sv = ServedExe(exe, live_argv()[1:], err_path=os.path.join(work, "serve.err"))
    sv.start()
    try:
        for c in COUNTRIES:
            write_live(pids[c])
            sv.step()
            lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
            ou = np.fromfile(fou, dtype=np.int64)
            n = len(pids[c])
            top = int(ou[n - 1])
            rec["arms"][c] = {"top": tok.decode([top]),
                              "paris_rank": int((lg[n - 1] > lg[n - 1][paris]).sum()) + 1}
            print(f"geo {c}: top={rec['arms'][c]['top']!r} "
                  f"paris-rank={rec['arms'][c]['paris_rank']}", flush=True)
    finally:
        sv.close()
    json.dump(rec, open(args.out_json, "w"), indent=2)
    hold_base = rec["base"]
    if args.base_json:
        hb = json.load(open(args.base_json))
        hold_base = {c: hb["arms"][c]["top"] for c in COUNTRIES}
    ok = (rec["arms"]["Germany"]["paris_rank"] == 1
          and rec["arms"]["Italy"]["top"] == hold_base["Italy"]
          and rec["arms"]["Japan"]["top"] == hold_base["Japan"])
    print("SIPHON-GEO:", "INSTALL+HOLD" if ok else "see arms", flush=True)


if __name__ == "__main__":
    main()
