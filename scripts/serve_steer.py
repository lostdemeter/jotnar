"""Steered generation as listings: gated install is a program.

Prefix = full bmmv recompute program with Y2/Y26 taps (all 28 layers
chained; the taps are extra outputs, zero math change). Host computes
the lexical gate scalar from Y2 (cosine to frozen early key, ^6) and
a dose file. Suffix = one bmmv layer (L27) with steering injected at
its input (ADD of MUL(d_frozen, dose_live)) + head. All existing ops;
the siphon product is a listing, not a torch trick.
Usage: python3 scripts/serve_steer.py "The capital of Germany is" [--n 5]
Prints gate, dose, generation; asserts nothing (gates live in tests/).
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID, NH, NKV, DH = 3584, 28, 4, 128
PER = NH // NKV
SMAX = 16


def build_prefix(S):
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, "scripts"))
    from gen_qwen7b_geo import build_program
    return build_program(S, bmmv=True)


def build_suffix(S):
    from chain.builder import Prog, qwen_bmmv_layer
    p = Prog("qwen7b-steer27")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x", "dose", "dsteer", "pos", "cmask", "cmask7",
             "wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
             "ln1", "ln2", "bq", "bk", "bv", "lnf", "wlog"]
    p.inp(*names)
    xs = p.op("ADD", "x", p.op("MUL", "dsteer", "dose"), out="XS")
    y = qwen_bmmv_layer(p, xs, "wq", "wk", "wv", "wo", "wup", "wgate",
                        "wdown", "ln1", "ln2", "pos", "cmask", "cmask7",
                        S, pre="27", bq="bq", bk="bk", bv="bv")
    hn = p.op("RMSNORM", y, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p


def build_suffix(S):
    from chain.builder import Prog, qwen_bmmv_layer
    p = Prog("qwen7b-steer27")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    names = ["x", "dose", "dsteer", "pos", "cmask", "cmask7",
             "wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
             "ln1", "ln2", "bq", "bk", "bv", "lnf", "wlog"]
    p.inp(*names)
    xs = p.op("ADD", "x", p.op("MUL", "dsteer", "dose"), out="XS")
    y = qwen_bmmv_layer(p, xs, "wq", "wk", "wv", "wo", "wup", "wgate",
                        "wdown", "ln1", "ln2", "pos", "cmask", "cmask7",
                        S, pre="s", bq="bq", bk="bk", bv="bv")
    hn = p.op("RMSNORM", y, "lnf", out="HNf")
    lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
    p.op("ARGMAX", lg, 1, out="OUT")
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The", "capital", "of"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--smax", type=int, default=SMAX)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--workdir", default="/tmp/steer7b")
    ap.add_argument("--country", default="Germany",
                    help="fact to install: capital of COUNTRY")
    ap.add_argument("--capital", default=" Paris",
                    help="target token text (must be single-token)")
    ap.add_argument("--contrast", default="France",
                    help="pair-contrast country for direction mining")
    ap.add_argument("--autocal", action="store_true",
                    help="bisect gain for rank-1 + margin>1 + controls hold")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    S = args.smax
    from chain.qwen7b import (load7b, snapshot_ok, prep_7b_weight,
                              clear_7b_cache, input_sources)
    from chain.emit_c import compile_program, FRef, IRef
    from chain.emit_cuda import build_cu
    from chain.serve import ServedExe
    from chain import mem as _mem
    print(_mem.report(S), flush=True)
    if not snapshot_ok():
        sys.exit("SKIP (needs snapshot)")
    g, tok = load7b()
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy().tolist()
    if len(ids) + args.n > S:
        sys.exit("prompt too long")
    work = args.workdir
    os.makedirs(work, exist_ok=True)
    keymap = dict(input_sources())
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)

    # -- mine direction + key on host (torch mirror, pinned fixtures) --
    # Direction: L27 country-vs-mean-others contrast (late content).
    # Key: L2 state at the country token (lexical address). Both via
    # the shared mirror (research/qwen_torch) -- no inline copies.
    # The default fact is Germany/Paris (history); pass --country /
    # --capital for fresh facts (demo: Portugal/Lisbon).
    sys.path.insert(0, os.path.join(ROOT, "research"))
    from qwen_torch import fwdH, fwd
    # Direction recipe (measured): PAIR contrast installs rank-1;
    # vs-mean-others dilutes to rank-250 (see SIPHON_ADDRESSING #6).
    # --contrast picks the contrast country (default France, proven
    # for the Germany/Paris demo fact; other facts need their own
    # contrast calibration -- backlog, stated).
    dfn = os.path.join(work, "steer_d.bin")
    kfn = os.path.join(work, "steer_key.bin")
    if os.path.isfile(dfn) and os.path.isfile(kfn) and not os.environ.get("QWEN_REMINE") \
            and not os.environ.get("QWEN_RANDOM_DIR"):
        # fixtures are fact-fixed: mine once per workdir, reuse across
        # gains/prompts (mining is ~1min of torch; the sweep is the point)
        d = np.fromfile(dfn, dtype=np.float16).astype(np.float64)
        key = np.fromfile(kfn, dtype=np.float16).astype(np.float64)
        print("fixtures reused (QWEN_REMINE=1 to re-mine)", flush=True)
        cap_ids = tok(args.capital, return_tensors="pt")["input_ids"][0].tolist()
        if len(cap_ids) != 1:
            sys.exit(f"capital {args.capital!r} is not single-token {cap_ids}")
        clear_7b_cache(g)
        import gc as _gc
        _gc.collect()
    else:
        tC, _ = fwdH(f"The capital of {args.country} is")
        tO, _ = fwdH(f"The capital of {args.contrast} is")
        # Orientation MATTERS (measured): the installing direction points
        # contrast-ward (France-minus-Germany installs Paris on Germany;
        # the negation anti-installs to rank 19912 -- cosine -1.0 found
        # live). d = tO - tC, never the reverse.
        d = tO[27] - tC[27]
        d /= np.linalg.norm(d)
        if os.environ.get("QWEN_RANDOM_DIR") == "1":
            # Control: random unit direction, same dose -- separates content
            # (only the true direction should install) from margin fragility
            # (any perturbation flips razor picks). NOTE: random poisons
            # shared workdirs -- use a scratch --workdir for controls.
            d = np.random.default_rng(0).normal(size=d.shape)
            d /= np.linalg.norm(d)
            print("RANDOM direction control", flush=True)
        cap_ids = tok(args.capital, return_tensors="pt")["input_ids"][0].tolist()
        if len(cap_ids) != 1:
            sys.exit(f"capital {args.capital!r} is not single-token {cap_ids}")
        tG, gids = fwdH(f"The capital of {args.country} is", keep="all")
        gtoks = tok.convert_ids_to_tokens(gids)
        frag = args.country[1:].lower()
        gpos = next((i for i, t in enumerate(gtoks) if frag in t.lower()),
                    len(gids) - 1)
        key = tG[2][gpos]
        key /= np.linalg.norm(key)
        d.astype(np.float16).tofile(dfn)
        key.astype(np.float16).tofile(kfn)
        print("fixtures mined (direction + early key)", flush=True)
        clear_7b_cache(g)
        import gc as _gc
        _gc.collect()

    # -- prefix program: full bmmv + Y2/Y26 taps (zero math change) ----
    import sys as _s2
    _s2.path.insert(0, os.path.join(ROOT, "scripts"))
    from gen_qwen7b_geo import build_program
    from chain.emit_c import compile_program, FRef, IRef
    from chain.emit_cuda import build_cu
    from chain.serve import ServedExe
    from chain.qwen7b import shapes_7b, input_sources, prep_7b_weight
    prog = build_program(S, bmmv=True)
    text = prog.text()
    # tap Y2 (gate address space) + Y26 (steer point): extra outputs
    # (any stream, even intermediates, can be an output -- no new ops).
    sample = shapes_7b(S)
    keymap = dict(input_sources())
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    pwork = os.path.join(work, "prefix")
    os.makedirs(pwork, exist_ok=True)
    art = compile_program(text, "cuda", sample=sample,
                          outputs=["LOGITS", "OUT", "Y2", "Y26"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=["x0", "pos", "cmask", "cmask7"])
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(pwork, f"f_{n}.bin")
        if n in ("x0", "pos", "cmask", "cmask7"):
            pass  # live per prompt: always (re)write below, never cache
        elif os.path.isfile(fn):
            continue
        elif k == "F":
            short, src = keymap[n]
            prep_7b_weight(g, src, short, S).tofile(fn)
        else:
            raise KeyError(n)
    xx = np.zeros((S, HID), dtype=np.float16)
    xx[:len(ids)] = E[np.array(ids)].astype(np.float16)
    xx.tofile(os.path.join(pwork, "f_x0.bin"))
    np.arange(S, dtype=np.int64).tofile(os.path.join(pwork, "f_pos.bin"))
    np.tril(np.ones((S, S), dtype=np.int64)).tofile(
        os.path.join(pwork, "f_cmask.bin"))
    cm1 = np.tril(np.ones((S, S), dtype=np.int64))
    np.tile(cm1, (7, 1)).tofile(os.path.join(pwork, "f_cmask7.bin"))
    clear_7b_cache(g)
    _gc.collect()
    import hashlib as _hl
    # source-hash stamps: refuse binaries built from other sources
    # (stale-binary reuse silently runs superseded code -- the STATIC
    # tap era bit us here; now it fails loud).
    def _stamp_ok(d, source, name):
        import json as _js
        fn = os.path.join(d, "build.json")
        h = _hl.sha256(source.encode()).hexdigest()[:16]
        try:
            old = _js.load(open(fn))
        except OSError:
            old = {}
        if os.environ.get("QWEN_REUSE_BIN") == "1":
            if old.get(name) != h:
                sys.exit(f"refusing reuse: {name} source changed "
                         f"(rebuild without QWEN_REUSE_BIN=1)")
            print(f"nvcc {name}: reused binary (stamp verified)", flush=True)
            return os.path.join(d, name)
        _js.dump({**old, name: h}, open(fn, "w"))
        return None

    pexe = os.path.join(pwork, "qpre2")
    hit = _stamp_ok(pwork, art["source"], "qpre2")
    pexe = hit or build_cu(art["source"], pwork, name="qpre2")
    pargv = [os.path.join(pwork, f"f_{n}.bin") for n in art["inputs"]]
    pargv += [os.path.join(pwork, f"o_{o}.bin") for o in art["outputs"]]
    svp = ServedExe(pexe, pargv, err_path=os.path.join(pwork, "pre.err"))
    svp.start()
    svp.close()
    y2 = np.fromfile(os.path.join(pwork, "o_Y2.bin"),
                     dtype=np.float32).reshape(S, HID)
    y26 = np.fromfile(os.path.join(pwork, "o_Y26.bin"),
                      dtype=np.float32).reshape(S, HID)
    # -- gate on host (scalar control plane): match early key, sharpen
    cids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    ctoks = tok.convert_ids_to_tokens(cids)
    cfrag = args.country[1:].lower()
    cpos = next((i for i, t in enumerate(ctoks) if cfrag in t.lower()),
                len(cids) - 1)
    m = float(y2[cpos] @ key / (np.linalg.norm(y2[cpos]) + 1e-12))
    s = m ** 6 if m > 0 else 0.0
    mag = float(np.linalg.norm(y26[len(ids) - 1]))
    dose = np.full((S, HID), args.gain * s * mag, dtype=np.float16)
    print(f"gate match={m:.3f} s={s:.3f} dose={args.gain * s * mag:.2f}",
          flush=True)
    # -- suffix program: one bmmv layer + steering + head -------------
    ps = build_suffix(S)
    qmap = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
            "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
            "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
            "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
            "ln2": "post_attention_layernorm.weight",
            "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
            "bv": "self_attn.v_proj.bias"}
    ssample = {"x": FRef((S, HID), "float16"),
               "dose": FRef((S, HID), "float16"),
               "dsteer": FRef((S, HID), "float16"),
               "pos": np.arange(S, dtype=np.int64),
               "cmask": np.tril(np.ones((S, S), dtype=np.int64))}
    cm1 = np.tril(np.ones((S, S), dtype=np.int64))
    ssample["cmask7"] = np.tile(cm1, (7, 1))
    for k, src in qmap.items():
        ssample[k] = FRef((HID, HID) if k in ("wq", "wo") else
                          (HID, 512) if k in ("wk", "wv") else
                          (HID, 18944) if k in ("wup", "wgate") else
                          (18944, HID) if k == "wdown" else
                          (HID,) if k in ("ln1", "ln2") else
                          (S, HID) if k == "bq" else (S, 512),
                          "float16")
    ssample["lnf"] = FRef((HID,), "float16")
    ssample["wlog"] = FRef((HID, 152064), "float16")
    sart = compile_program(ps.text(), "cuda", sample=ssample,
                           outputs=["LOGITS", "OUT"],
                           basedir=os.path.join(ROOT, "programs"),
                           use_fp16=True, graph=True,
                           live=["x", "dose"])
    swork = os.path.join(work, "suffix")
    os.makedirs(swork, exist_ok=True)
    from chain.qwen7b import input_sources as _isrc, prep_7b_weight as _prep
    for n in sart["inputs"]:
        k, _, _ = sart["streams"][n]
        fn = os.path.join(swork, f"s_{n}.bin")
        if n in ("x", "dose", "dsteer"):
            pass  # live per prompt/fixture: always (re)write below
        elif os.path.isfile(fn):
            continue
        elif n == "pos":
            np.arange(S, dtype=np.int64).tofile(fn)
        elif n == "cmask":
            np.tril(np.ones((S, S), dtype=np.int64)).tofile(fn)
        elif n == "cmask7":
            np.tile(np.tril(np.ones((S, S), dtype=np.int64)), (7, 1)).tofile(fn)
        elif k == "F":
            short = {"wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo",
                     "wup": "wup", "wgate": "wgate", "wdown": "wdown",
                     "ln1": "ln1", "ln2": "ln2", "bq": "bq", "bk": "bk",
                     "bv": "bv", "lnf": "lnf", "wlog": "wlog"}[n]
            src = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
                   "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
                   "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
                   "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
                   "ln2": "post_attention_layernorm.weight",
                   "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
                   "bv": "self_attn.v_proj.bias",
                   "lnf": "model.norm.weight",
                   "wlog": "lm_head.weight"}[n if n in (
                       "lnf", "wlog") else short]
            dotted = src if n in ("lnf", "wlog") else f"model.layers.27.{src}"
            _prep(g, dotted, short, S).tofile(fn)
        else:
            raise KeyError(n)
    clear_7b_cache(g)
    _gc.collect()
    # live per prompt/fixture (never cached): Y26 as x, gate dose,
    # and the direction tile (fixture-fixed but tiny: rewrite keeps
    # workdirs honest across re-mines and random controls)
    y26.astype(np.float16).tofile(os.path.join(swork, "s_x.bin"))
    dose.astype(np.float16).tofile(os.path.join(swork, "s_dose.bin"))
    np.tile(d.astype(np.float16), (S, 1)).tofile(
        os.path.join(swork, "s_dsteer.bin"))
    sexe = os.path.join(swork, "qsuf")
    hit = _stamp_ok(swork, sart["source"], "qsuf")
    sexe = hit or build_cu(sart["source"], swork, name="qsuf")
    sargv = [sexe] + [os.path.join(swork, f"s_{n}.bin")
                      for n in sart["inputs"]]
    sargv += [os.path.join(swork, "s_lg.bin"),
              os.path.join(swork, "s_ou.bin")]
    svs = ServedExe(sexe, sargv[1:], err_path=os.path.join(swork, "suf.err"))
    svs.start()
    svs.close()
    slg = np.fromfile(os.path.join(swork, "s_lg.bin"),
                      dtype=np.float32).reshape(S, -1)
    ou = np.fromfile(os.path.join(swork, "s_ou.bin"), dtype=np.int64)
    row = slg[len(ids) - 1]
    tgt = cap_ids[0]
    print(f"steered pick: {tok.decode([int(ou[len(ids) - 1])])!r} "
          f"target-rank={int((row > row[tgt]).sum()) + 1}", flush=True)


if __name__ == "__main__":
    main()
