"""Demo: the LLM generating text through compiled backends.

D16 headt word model (lm_svd_IvoQ + bankhn, V=513 words): greedy
sliding-window generation, one subprocess step per token. Backend
selects the executor; the program and weights are identical.
  python3 demo_gen.py [seed words...] [--n 12] [--backend c|cuda|nonfpu|lattice]
Weights dump once; per step only tok/pos/cmask change. Deterministic
(greedy); --topk N samples host-side from emitted LOGITS.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

WIN = 8
CFG = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
       "CONFIG beta_b 0.25\n")


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


class LatticeRunner:
    name = "lattice"

    def __init__(self, text, trip, sdir):
        self.text, self.trip, self.sdir = text, trip, sdir

    def step(self, ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(self.text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos,
                          "cmask": cm, **self.trip}, sigs=SIGS,
                         basedir=self.sdir)
        return (dec(f["LOGITS"])[-1],
                int(np.ascontiguousarray(f["OUT"]).reshape(-1)[-1]))


class ExeRunner:
    """One compiled binary, stepped via files. kind=float|triples."""

    def __init__(self, text, trip, sdir, work, name, target, libs):
        from chain.emit_c import compile_program, build
        # sample shapes from a WIN-length dummy (dynamic batch covers <=WIN)
        dummy = dict(trip)
        dummy.update({"tok": np.zeros(WIN, np.int64),
                      # pos spans the window so baked RoPE tables cover it
                      "pos": np.arange(WIN, dtype=np.int64),
                      "cmask": np.zeros((WIN, WIN), np.int64)})
        if target in ("c", "cuda"):
            # float backends run decoded weights (lattice quantum stays
            # in the eps-agreement class, as gated)
            dummy = {k: (np.ascontiguousarray(dec(v)) if isinstance(v, tuple)
                         else v) for k, v in dummy.items()}
        art = compile_program(text, target, sample=dummy,
                              outputs=["LOGITS", "OUT"], basedir=sdir)
        self.name = target
        self.art = art
        kw = {} if libs == "dflt" else {"libs": libs}
        if target == "cuda":
            from chain.emit_cuda import build_cu
            self.exe = build_cu(art["source"], work, name=name)
            self.fdt = np.float32
        else:
            self.exe = build(art["source"], work, name=name, **kw)
            self.fdt = np.float64
        self.work = work
        self.trip = trip
        self.floats = target in ("c", "cuda")
        # dump frozen streams once
        for n in art["inputs"]:
            k, _, _ = art["streams"][n]
            if n in ("tok", "pos", "cmask"):
                continue
            v = trip[n]
            if k == "T":
                for comp, arr in zip("sez", v):
                    np.ascontiguousarray(arr).tofile(
                        os.path.join(work, f"fr_{n}_{comp}.bin"))
            elif self.floats:
                np.ascontiguousarray(dec(v)).astype(
                    np.float32 if target == "cuda" else np.float64).tofile(
                    os.path.join(work, f"fr_{n}.bin"))
            else:
                np.ascontiguousarray(v, dtype=np.int64).tofile(
                    os.path.join(work, f"fr_{n}.bin"))

    def step(self, ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        argv = [self.exe]
        for n in self.art["inputs"]:
            k, _, _ = self.art["streams"][n]
            if n == "tok":
                fn = os.path.join(self.work, "step_tok.bin")
                np.array(ctx, np.int64).tofile(fn)
                argv.append(fn)
            elif n == "pos":
                fn = os.path.join(self.work, "step_pos.bin")
                pos.astype(np.int64).tofile(fn)
                argv.append(fn)
            elif n == "cmask":
                fn = os.path.join(self.work, "step_cm.bin")
                cm.astype(np.int64).tofile(fn)
                argv.append(fn)
            elif k == "T":
                for comp in "sez":
                    argv.append(os.path.join(self.work, f"fr_{n}_{comp}.bin"))
            else:
                argv.append(os.path.join(self.work, f"fr_{n}.bin"))
        f_lo = os.path.join(self.work, "step_logits.bin")
        f_ou = os.path.join(self.work, "step_out.bin")
        ok = self.art["streams"]["LOGITS"][0]
        if ok == "T":
            fl, fe, fz = (os.path.join(self.work, f"step_lg_{c}.bin")
                          for c in "sez")
            argv += [fl, fe, fz, f_ou]
        else:
            argv += [f_lo, f_ou]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{self.name} step failed rc={r.returncode}: "
                               f"{r.stderr[:200]}")
        if ok == "T":
            d = {c: np.fromfile(fn, dtype=dt) for c, fn, dt in
                 (("s", fl, np.int8), ("e", fe, np.int32), ("z", fz, np.uint8))}
            logits = dec((d["s"], d["e"], d["z"])).reshape(len(ctx), -1)
        else:
            logits = np.fromfile(f_lo, dtype=self.fdt).reshape(len(ctx), -1)
        gout = np.fromfile(f_ou, dtype=np.int64)
        return logits[-1], int(gout.reshape(-1)[-1])


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    words, n, i = [], 12, 1
    backend, topk, seedn = "c", 0, 7
    no_unk = True
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--allow-unk":
            no_unk = False
            i += 1
        elif a == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            i += 2
        elif a == "--topk" and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()
    trip = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
            "wv": enc(d["wv"]), "wo": enc(d["wo"]), "wup": enc(d["wup"]),
            "wgate": enc(d["wgate"]), "wdown": enc(d["wdown"]),
            "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
            "wlog": enc(d["wlog"]), "ukt": enc(b["ukt"]),
            "evb": enc(b["evb"])}
    work = f"/tmp/gen_{backend}"
    os.makedirs(work, exist_ok=True)
    if backend == "lattice":
        runner = LatticeRunner(text, trip, sdir)
    elif backend == "c":
        runner = ExeRunner(text, trip, sdir, work, "gen", "c", "dflt")
    elif backend == "cuda":
        runner = ExeRunner(text, trip, sdir, work, "gen", "cuda", "dflt")
    elif backend == "nonfpu":
        runner = ExeRunner(text, trip, sdir, work, "gen", "nonfpu", [])
    else:
        sys.exit(f"unknown backend {backend}")
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    rng = np.random.default_rng(seedn)
    out = list(ids)
    for _ in range(n):
        logits, greedy = runner.step(out)
        if no_unk:
            logits = np.array(logits, dtype=np.float64)
            logits[0] = -np.inf
            greedy = int(np.argmax(logits))
        if topk > 0:
            keep = np.argsort(-logits)[:topk]
            w = np.zeros_like(logits, dtype=np.float64)
            w[keep] = np.exp(logits[keep] - logits[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        else:
            out.append(greedy)
    print(f"backend: {backend}")
    print("seed:", seed)
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))


if __name__ == "__main__":
    main()
