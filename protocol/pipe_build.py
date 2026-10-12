"""Pipeline stage builds (protocol Milestone 3, controller side).

Builds per-stage decode binaries from layer shards using the builder
and emitter exactly as scripts/serve_decode.py does for the full
program -- no emitter/opcode changes. Each stage is an independent
listing over its layer range:

- first stage:  IN x0 (f16 embed) ... -> boundary Y{hi} (f32, exact)
- middle stage: IN x0 (f32 boundary, same values, no rounding) ...
- last stage:   ... -> RMSNORM/MATMUL/ARGMAX tail (LOGITS/OUT)

Boundary activations cross stages as f32 bytes (never f16: the
standard cache rounding would break bit-exactness vs single-node).
KV caches stay per-stage sticky files; the controller holds no KV,
only the H boundary. Workdirs are independent (frozen weights
prepped fresh per stage; nothing shared with /tmp/dec7b512).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

NL, HID, KVD, PER, INTER = 28, 3584, 512, 7, 18944


def stage_program(S, lo, hi, last):
    """(prog text, kouts, vouts, boundary) for layers [lo..hi]."""
    from chain.builder import Prog, qwen_dec_full_layer
    from chain.qwen7b import QWEN_MASK_NEG
    p = Prog(f"qdec-S{lo}-{hi}")
    p.config("eps_rms", "1e-6").config("beta", QWEN_MASK_NEG)
    p.config("rope_base", 1000000.0)
    names = ["x0", "pos", "attmask7", "allrows"]
    for L in range(lo, hi + 1):
        names += [f"kP{L}", f"vP{L}"]
    for L in range(lo, hi + 1):
        for k in ("wqkv", "wo", "wug", "wdown", "ln1", "ln2", "bqkv"):
            names.append(f"{k}{L}")
    if last:
        names += ["lnf", "wlog"]
    p.inp(*names)
    x = "x0"
    kouts, vouts = [], []
    for L in range(lo, hi + 1):
        x, kf, vf = qwen_dec_full_layer(
            p, x, f"kP{L}", f"vP{L}", f"wqkv{L}",
            f"wo{L}", f"wug{L}", f"wdown{L}", f"ln1{L}",
            f"ln2{L}", "pos", "attmask7", "allrows",
            S, pre=str(L), bqkv=f"bqkv{L}", resident=False)
        kouts.append(kf)
        vouts.append(vf)
    if last:
        hn = p.op("RMSNORM", x, "lnf", out="HNf")
        lg = p.op("MATMUL", hn, "wlog", out="LOGITS")
        p.op("ARGMAX", lg, 1, out="OUT")
        return p.text(), kouts, vouts, None
    return p.text(), kouts, vouts, x


def build_stage(g, keymap, S, lo, hi, last, work, first=True,
                graph=True, no_sync=True):
    """Compile one stage binary into work/. Returns stage config dict."""
    from chain.qwen7b import prep_7b_weight, QWEN7B_VOCAB
    from chain.emit_c import compile_program, FRef
    from chain.emit_cuda import build_cu as _bc
    os.makedirs(work, exist_ok=True)
    stamp = {"graph": bool(graph), "no_sync": bool(no_sync), "smax": S,
             "sparse_bands": "", "sparse_k": 8, "resident": False,
             "fused_qkv": True, "stage": [lo, hi]}
    exe_name = f"qdecS{lo}-{hi}"
    exe = os.path.join(work, exe_name)
    text, kouts, vouts, boundary = stage_program(S, lo, hi, last)

    def prep_one(dotted, short):
        a = prep_7b_weight(g, dotted, short, S)
        if short in ("bq", "bk", "bv"):
            a = np.ascontiguousarray(a[:1])
        return a

    def prep_fused(name):
        if name.startswith("wqkv"):
            L = name[4:]
            parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                     for s in ("wq", "wk", "wv")]
            return np.ascontiguousarray(np.concatenate(parts, axis=1))
        if name.startswith("bqkv"):
            L = name[4:]
            parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                     for s in ("bq", "bk", "bv")]
            return np.ascontiguousarray(np.concatenate(parts, axis=1))
        if name.startswith("wug"):
            L = name[3:]
            parts = [prep_one(*keymap[f"{s}{L}"][::-1])
                     for s in ("wup", "wgate")]
            return np.ascontiguousarray(np.concatenate(parts, axis=1))
        return None

    sample = {"x0": FRef((1, HID), "float16" if first else "float32"),
              "pos": np.zeros(1, dtype=np.int64),
              "attmask7": np.zeros((PER, S), dtype=np.int64),
              "allrows": np.arange(S, dtype=np.int64)}
    for L in range(lo, hi + 1):
        sample[f"kP{L}"] = FRef((S, KVD), "float16")
        sample[f"vP{L}"] = FRef((S, KVD), "float16")
    wshapes = {"wqkv": (HID, HID + 2 * KVD), "wo": (HID, HID),
               "wug": (HID, 2 * INTER), "wdown": (INTER, HID),
               "ln1": (HID,), "ln2": (HID,), "bqkv": (1, HID + 2 * KVD)}
    for L in range(lo, hi + 1):
        for k, sh in wshapes.items():
            sample[f"{k}{L}"] = FRef(sh, "float16")
    if last:
        sample["lnf"] = FRef((HID,), "float16")
        sample["wlog"] = FRef((HID, QWEN7B_VOCAB), "float16")
    caches = [f"kP{L}" for L in range(lo, hi + 1)] + \
             [f"vP{L}" for L in range(lo, hi + 1)]
    if last:
        outs = ["LOGITS", "OUT"] + kouts + vouts
    else:
        outs = [boundary] + kouts + vouts
    live = ["x0", "pos", "attmask7"] + caches
    try:
        old = json.load(open(os.path.join(work, "build.json")))
    except OSError:
        old = {}
    # inputs order defines argv; persist it so reuse skips compile too.
    if old.get("stamp") == stamp and old.get("inputs") and \
            os.path.isfile(exe):
        print(f"stage {lo}-{hi}: reused binary (stamp verified)", flush=True)
        in_order = list(old["inputs"])
    else:
        art = compile_program(text, "cuda", sample=sample, outputs=outs,
                              basedir=os.path.join(ROOT, "programs"),
                              use_fp16=True, live=live,
                              sync_each=not no_sync, graph=graph)
        in_order = list(art["inputs"])
    # frozen inputs prepped fresh per stage (nothing shared)
    for n in in_order:
        if n in live:
            continue
        fn = os.path.join(work, f"d_{n}.bin")
        if os.path.isfile(fn):
            continue
        if n == "allrows":
            np.arange(S, dtype=np.int64).tofile(fn)
            continue
        fused = prep_fused(n)
        if fused is not None:
            fused.tofile(fn)
            continue
        if n in keymap:
            short, src = keymap[n]
            prep_one(src, short).tofile(fn)
        else:
            raise KeyError(f"no frozen source for {n!r}")
    if not (old.get("stamp") == stamp and os.path.isfile(exe)):
        exe = _bc(art["source"], work, name=exe_name,
                  units=art.get("units"))
        json.dump({"stamp": stamp, "inputs": in_order},
                  open(os.path.join(work, "build.json"), "w"))
    stamp = {k: v for k, v in stamp.items() if k != "stage"}

    def argv_for(lives, out_files):
        argv = []
        for name in in_order:
            argv.append(lives.get(name, os.path.join(work, f"d_{name}.bin")))
        return argv + out_files

    lives = {"x0": os.path.join(work, "step_x0.bin"),
             "pos": os.path.join(work, "step_pos.bin"),
             "attmask7": os.path.join(work, "step_am.bin")}
    for L in range(lo, hi + 1):
        lives[f"kP{L}"] = os.path.join(work, f"ckP{L}.bin")
        lives[f"vP{L}"] = os.path.join(work, f"cvP{L}.bin")
        for tag in ("ckP", "cvP"):
            fn = os.path.join(work, f"{tag}{L}.bin")
            if not os.path.isfile(fn):
                np.zeros((S, KVD), dtype=np.float16).tofile(fn)
    np.zeros((1, HID), dtype=np.float16 if first else np.float32).tofile(lives["x0"])
    np.zeros(1, dtype=np.int64).tofile(lives["pos"])
    np.zeros((PER, S), dtype=np.int64).tofile(lives["attmask7"])
    if last:
        out_files = [os.path.join(work, "step_lg.bin"),
                     os.path.join(work, "step_ou.bin")]
    else:
        out_files = [os.path.join(work, "step_H.bin")]
    for L in range(lo, hi + 1):
        out_files += [os.path.join(work, f"step_K{L}.bin"),
                      os.path.join(work, f"step_V{L}.bin")]
    return {"text": text, "argv": argv_for(lives, out_files),
            "lives": lives, "kouts": kouts, "vouts": vouts,
            "boundary": boundary, "last": last, "lo": lo, "hi": hi,
            "exe": exe, "stamp": stamp, "work": work}


def build_pipeline(g, keymap, S, splits, workbase):
    """Build all stage binaries. splits=[(lo,hi),...]. Returns configs."""
    cfgs = []
    for i, (lo, hi) in enumerate(splits):
        last = i == len(splits) - 1
        work = os.path.join(workbase, f"stage{i}_{lo}-{hi}")
        print(f"stage {lo}-{hi} -> {work}", flush=True)
        cfgs.append(build_stage(g, keymap, S, lo, hi, last, work,
                                first=(i == 0)))
    return cfgs
