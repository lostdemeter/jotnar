"""Builder gate v0.1: Python frontend emits assembly the assembler trusts.

Strong gates: builder-rebuilt bigram runs identical to the hand-written
listing; builder DEF/CALL matches stdlib swiglu_block; builder output
compiles to C and the binary agrees. Composite attn_head: smoke
(runs, deterministic, finite) -- its full gate arrives with the
builder-built block.
Usage: python3 tests/test_builder.py (fast).
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.builder import Prog, attn_head
from chain.emit_c import build

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    root = ROOT
    sdir = os.path.join(root, "programs")
    dd = os.path.join(root, "data")
    rng = np.random.default_rng(11)

    # -- 1. bigram rebuilt by builder == hand-written listing --------
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    bs = np.zeros_like(counts, np.int8)
    be = np.zeros_like(counts, np.int32)
    bz = np.zeros_like(counts, np.uint8)
    for v in np.unique(counts):
        if v == 0:
            continue
        ws, we, wz = S.encode(np.array([float(v)]))
        m = counts == v
        bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
    bz[counts == 0] = 1
    bank = (np.ascontiguousarray(bs), np.ascontiguousarray(be),
            np.ascontiguousarray(bz))
    toks = np.array([12, 471, 59, 38], np.int64)
    p = Prog("bigram")
    p.inp("tok", "bank")
    p.op("GATHER", "bank", "tok", out="ROW")
    p.op("ARGMAX", "ROW", 1, out="OUT")
    f_hand = ASM.run_text(open(os.path.join(sdir, "bigram_lm.asm")).read(),
                          REGISTRY, {"tok": toks, "bank": bank},
                          sigs=SIGS, basedir=sdir)
    f_built = ASM.run_text(p.text(), REGISTRY, {"tok": toks, "bank": bank},
                           sigs=SIGS, basedir=sdir)
    check("builder-bigram-agrees",
          bool((np.ascontiguousarray(f_hand["OUT"]) ==
                np.ascontiguousarray(f_built["OUT"])).all()),
          f"hand={np.ascontiguousarray(f_hand['OUT']).tolist()} "
          f"built={np.ascontiguousarray(f_built['OUT']).tolist()}")

    # -- 2. DEF/CALL matches stdlib swiglu_block ----------------------
    def swiglu_body(sub):
        up = sub.op("MATMUL", "hn", "wup")
        gate = sub.op("MATMUL", "hn", "wgate")
        gs = sub.op("SILU", gate)
        mid = sub.op("MUL", gs, up)
        return sub.op("MATMUL", mid, "wdown", out="down")

    q = Prog("swiglu-test")
    q.inp("hn", "wup", "wgate", "wdown")
    q.defn("swiglu_block", ["hn", "wup", "wgate", "wdown"], swiglu_body)
    q.call("swiglu_block", "hn", "wup", "wgate", "wdown", out="DOWN")
    hn = S.encode(rng.normal(size=(3, 16)))
    wup = S.encode(rng.normal(size=(16, 32)))
    wgate = S.encode(rng.normal(size=(16, 32)))
    wdown = S.encode(rng.normal(size=(32, 16)))
    pay = {"hn": hn, "wup": wup, "wgate": wgate, "wdown": wdown}
    f_lib = ASM.run_text('IMPORT "mlp.asm"\nIN hn\nIN wup\nIN wgate\nIN wdown\n'
                         'DOWN = CALL swiglu_block(hn, wup, wgate, wdown)\n',
                         REGISTRY, pay, sigs=SIGS, basedir=sdir)
    f_bld = ASM.run_text(q.text(), REGISTRY, pay, sigs=SIGS, basedir=sdir)

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    d = float(np.abs(dec(f_lib["DOWN"]) - dec(f_bld["DOWN"])).max())
    check("builder-defcall-agrees", d == 0.0, f"maxabs={d:.1e} (macro-identical)")

    # -- 3. builder output compiles to C and agrees -------------------
    from chain.emit_c import compile_program
    art = compile_program(p.text(), "c",
                          sample={"tok": toks, "bank": bank},
                          outputs=["OUT"], basedir=sdir)
    work = "/tmp/emit_builder_v01"
    os.makedirs(work, exist_ok=True)
    exe = build(art["source"], work, name="bigram_b")
    fbs, fbe, fbz, ftok, fout = (os.path.join(work, f"x_{c}.bin")
                                 for c in ("s", "e", "z", "tok", "out"))
    bs.tofile(fbs)
    be.tofile(fbe)
    bz.tofile(fbz)
    toks.tofile(ftok)
    r = subprocess.run([exe, ftok, fbs, fbe, fbz, fout],
                       capture_output=True, text=True)
    check("builder-c-run", r.returncode == 0, f"rc={r.returncode}")
    got = np.fromfile(fout, dtype=np.int64)
    ref = np.ascontiguousarray(f_hand["OUT"]).reshape(-1)
    check("builder-c-agrees", bool((got == ref).all()),
          f"{int((got == ref).sum())}/{len(ref)}")

    # -- 4. attn_head composite smoke ----------------------------------
    # head-narrow random inputs (S=3, Dh=8)
    hx = S.encode(rng.normal(size=(3, 8)))
    hw = S.encode(rng.normal(size=(8, 8)))
    hpos = np.arange(3, dtype=np.int64)
    hcm = np.tril(np.ones((3, 3), dtype=np.int64))
    h2 = Prog("head-smoke")
    h2.config("eps_rms", "1e-6").config("beta", -30.0)
    h2.inp("x", "wq", "wk", "wv", "pos", "cmask")
    ctx = attn_head(h2, "x", "wq", "wk", "wv", "pos", "cmask", out="CTX")
    pay2 = {"x": hx, "wq": hw, "wk": hw, "wv": hw, "pos": hpos, "cmask": hcm}
    g1 = ASM.run_text(h2.text(), REGISTRY, pay2, sigs=SIGS, basedir=sdir)
    g2 = ASM.run_text(h2.text(), REGISTRY, pay2, sigs=SIGS, basedir=sdir)
    v = dec(g1[ctx])
    check("builder-head-runs", np.isfinite(v).all(), f"shape={v.shape}")
    check("builder-head-deterministic",
          float(np.abs(v - dec(g2[ctx])).max()) == 0.0, "replay-identical")

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
