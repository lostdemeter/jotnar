"""Arbitrary-fact siphon install as a geometric program (CUDA/FP16).

siphon_geo.py generalized beyond capitals: FACTS table (prompt,
subject, value-token) + shared controls; subject row computed at
build (frozen per prompt); keys = subject + control-subject nulls;
readout-row values; gains {0, 1} built in ONE run (same binary?
no -- Vc data differs: two builds, base first for fork-correct
grading). Ledger + stamp per run. New facts = new table rows (data,
never structure).
Usage: python3 scripts/siphon_any.py [--fact everest] [--smax 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

N_LAYERS = 28
HID, INTER, NH, NKV, DH = 3584, 18944, 28, 4, 128
FACTS = {
    "everest": ("The tallest mountain on Earth is", "mountain", " Everest"),
    "nile": ("The longest river in Africa is", "river", " Nile"),
}
CONTROLS = ["The capital of Germany is", "The capital of Italy is",
            "The capital of Japan is"]
GAINS = [float(g) for g in os.environ.get("GEO_GAINS", "0.0,1.0").split(",")]


def build_program(S, spos, bmmv=False):
    from chain.builder import Prog, qwen_layer
    from chain.qwen7b import QWEN_MASK_NEG
    p = Prog("qwen7b-28-any")
    p.config("eps_rms", "1e-6").config("beta", QWEN_MASK_NEG)
    p.config("rope_base", 1000000.0)
    p.imp("yarnball.asm")
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
            # explicit routing (addr_sep doctrine): subject row slices
            # per-fact at build (spos frozen like cmask); value tiles
            # (only prompt-end read: post-final-attention doctrine).
            # ADDRESS STREAM IS POST-MLP Y2 (mirror keys are post-MLP
            # L2 rows; pre-MLP H2 mismatches -> 8-16x dose gap. parity
            # with the mine, not convenience).
            xa = p.op("SLICE", "Y2", 0, spos, spos + 1, out="XA27")
            yb = p.call("yarnball_apply", xa, "Ua27", "Vc27", out="YB27")
            ybt = p.op("MATMUL", "ones51", yb, out="YBT27")
            x = p.op("ADD", x, ybt, out="X27s")
    hn = p.op("RMSNORM", x, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", type=str, default="everest", choices=list(FACTS))
    ap.add_argument("--smax", type=int, default=8)
    ap.add_argument("--key-scale", type=float, default=8.0)
    ap.add_argument("--out-json", type=str, default="/tmp/siphon_any.json")
    args = ap.parse_args()

    from chain import mem as _mem
    from chain.qwen7b import snapshot_ok
    print(_mem.report(args.smax), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs Qwen2-7B-Instruct snapshot)")
    _mem.guard(_mem.estimate_7b(args.smax)["stream_peak_cpu"], "geo any build")

    from chain.qwen7b import load7b, hf_no_triton
    hf_no_triton()
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from chain.engram import yarnball_bank
    g, tok = load7b()
    prompt, subj, vtok = FACTS[args.fact]
    tids = tok(" " + vtok.strip(), return_tensors="pt")["input_ids"][0].tolist()
    assert len(tids) == 1, f"value token must be single (got {tids})"
    tid = tids[0]
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].numpy()
    lo = prompt.find(subj)
    assert lo >= 0, (subj, prompt)
    spos = next(i for i, (a, b) in enumerate(offs) if a < lo + len(subj) and lo < b)
    nlen = len(enc["input_ids"][0])
    print(f"fact {args.fact}: len={nlen} subject-row={spos} value={vtok!r}",
          flush=True)
    assert nlen < args.smax, "prompt too long for smax"
    base = {}
    for p in [prompt] + CONTROLS:
        lg, _ = fwd(p)
        base[p] = int(lg.argmax())
    print("mirror unsteered:", {p[:32]: tok.decode([v]) for p, v in base.items()},
          flush=True)
    key, _ = early_key(prompt, pos=spos)
    D = HID
    nulls = []
    for p in CONTROLS:
        ids = tok(p, return_tensors="pt")["input_ids"][0].numpy()
        toks = tok.convert_ids_to_tokens(ids)
        frag = p.split()[3].lower() if len(p.split()) > 3 else p.split()[1].lower()
        pp = next((i for i, t in enumerate(toks) if frag in t.lower()),
                  len(ids) - 1)
        k, _ = early_key(p, pos=pp)
        nulls.append(k)
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    d = Wlog[tid] / np.linalg.norm(Wlog[tid])
    stores = [{"key": key, "value": d, "dose": 1.0, "tier": "assoc",
               "support": f"{args.fact} readout {vtok}"}]
    for i, k in enumerate(nulls):
        stores.append({"key": k, "value": np.zeros(D), "tier": "null",
                       "support": f"control-null-{i}"})
    Ua, Vc, ledger = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                   stores, key_scale=args.key_scale)
    json.dump([{"tier": r["tier"], "support": r["support"]}
               for r in ledger],
              open("/tmp/siphon_any_ledger.json", "w"), indent=2)
    del g
    import gc as _gc
    _gc.collect()

    from chain.emit_c import compile_program, FRef
    from chain.emit_cuda import build_cu
    from chain.qwen7b import shapes_7b, prep_7b_weight, clear_7b_cache, input_sources
    g, tok = load7b()
    S = args.smax
    prog = build_program(S, spos)
    text = prog.text()
    print(f"program: {sum(1 for ln in text.splitlines() if '=' in ln and not ln.strip().startswith(('IN', 'CONFIG', 'IMPORT')))} ops",
          flush=True)
    sample = shapes_7b(S)
    sample["Ua27"] = FRef((HID, Ua.shape[1]), "float16")
    sample["Vc27"] = FRef((Ua.shape[1], HID), "float16")
    sample["ones51"] = FRef((S, 1), "float16")
    live = ["x0", "pos", "cmask"]
    art = compile_program(text, "cuda", sample=sample,
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=live, sync_each=False)
    work = os.environ.get("QWEN_WORKDIR", f"/tmp/gen7b_any_{args.fact}")
    os.makedirs(work, exist_ok=True)
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    by_input = input_sources()
    alias = art.get("cname", {})
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
            pass  # per-gain below (dose lives in data)
        elif n == "ones51":
            np.ones((S, 1), dtype=np.float16).tofile(fn)
        elif n == "lnf":
            prep_7b_weight(g, by_input["lnf"][1], "lnf", S).tofile(fn)
        elif n == "wlog":
            prep_7b_weight(g, by_input["wlog"][1], "wlog", S).tofile(fn)
        elif alias.get(n, n) in by_input:
            short, src = by_input[alias.get(n, n)]
            w16 = prep_7b_weight(g, src, short, S)
            (w16 if k == "F" else w16.astype(np.int64)).tofile(fn)
            del w16
        else:
            raise KeyError(f"no weight source for input stream {n!r}")
    clear_7b_cache(g)
    _gc.collect()
    exe = build_cu(art["source"], work, name="q7b_any",
             units=art.get("units"))
    print("nvcc done (shared binary, Vc per gain)", flush=True)
    g, tok = load7b()
    rec = {"fact": args.fact, "gains": {}}
    for gn in GAINS:
        Vd = Vc.copy()
        Vd[0] = Vd[0] * gn  # dose the install row (nulls stay zero)
        np.ascontiguousarray(Vd, dtype=np.float16).tofile(
            os.path.join(work, "fr_Vc27.bin"))
        print(f"Vc dumped (gain {gn}; binary shared)", flush=True)
        from chain.serve import ServedExe
        flo = os.path.join(work, "step_lg.bin")
        fou = os.path.join(work, "step_ou.bin")

        def write_live(tok_ids):
            nn = len(tok_ids)
            xx = np.zeros((S, HID), dtype=np.float16)
            xx[:nn] = E[np.array(tok_ids)].astype(np.float16)
            xx.tofile(os.path.join(work, "step_x0.bin"))
            np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(os.path.join(work, "step_cm.bin"))

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
        argv += [flo, fou]
        sv = ServedExe(exe, argv[1:], err_path=os.path.join(work, "serve.err"))
        p0 = [prompt] + CONTROLS
        write_live(tok(p0[0], return_tensors="pt")["input_ids"][0].numpy().tolist())
        sv.start()
        arm = {}
        try:
            for p in [prompt] + CONTROLS:
                ids = tok(p, return_tensors="pt")["input_ids"][0].numpy().tolist()
                write_live(ids)
                sv.step()
                lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
                ou = np.fromfile(fou, dtype=np.int64)
                nn = len(ids)
                top = int(ou[nn - 1])
                arm[p] = {"top": tok.decode([top]),
                          "tgt_rank": int((lg[nn - 1] > lg[nn - 1][tid]).sum()) + 1}
                print(f"geo gain={gn} {p[:32]!r}: top={arm[p]['top']!r} "
                      f"tgt-rank={arm[p]['tgt_rank']}", flush=True)
        finally:
            sv.close()
        rec["gains"][str(gn)] = arm
    json.dump(rec, open(args.out_json, "w"), indent=2)
    b0, b1 = rec["gains"]["0.0"], rec["gains"]["1.0"]
    ok = (b1[prompt]["tgt_rank"] == 1
          and all(b1[p]["top"] == b0[p]["top"] for p in CONTROLS))
    print("SIPHON-ANY:", "INSTALL+HOLD" if ok else "see arms", flush=True)


if __name__ == "__main__":
    main()
