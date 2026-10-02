"""Generation through teacher-coprocessor stacks (teacher-scaffold branch).

Same seed + same guards (seeded topk-12, trie mask, no-repeat-4)
through three listings: baseline (lm_d32_headt), Qwen-L0 adapter
(lm_d32_adapt, seed-11 projections), SmolLM2-L0 adapter
(lm_d32_adapt_smol, seed-12 projections). Read-only (weights never
change); honesty boundary: coprocessor mode (one teacher block
beside our loop), NOT full-model port. Compares voice, not top1
(qwen-adapt top1 identical 33/566; smol twin 0.494 vs base 0.665).
Usage: python3 research/gen_adapt.py [seed words...] [--n 20]
"""
import glob
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
SDIR = os.path.join(ROOT, "programs")
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16
NREP = 4


class _T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.p = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        self.skip = tag in ("script", "style", "nav", "header", "footer",
                            "aside")

    def handle_endtag(self, tag):
        self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.p.append(d)


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    vocab = json.load(open(os.path.join(DD, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(DD, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in words_of(s):
            syms = [c for c in w] + ["</w>"]
            while len(syms) > 1:
                best = None
                for i in range(len(syms) - 1):
                    r = rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best[0]):
                        best = (r, i)
                if best is None:
                    break
                _, i = best
                syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
            out.extend(vocab[p] for p in syms)
        return out

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    # trie (groki + w103 words speakable; validity rule)
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    try:
        _w103 = json.load(open(os.path.join(ROOT, "data_ingest", "wikitext103_vocab.json")))
        _extra = {w for w in _w103 if w != "<unk>"}
    except Exception:
        _extra = set()

    def encw(w):
        syms = [c for c in w] + ["</w>"]
        while len(syms) > 1:
            best = None
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
        return [vocab[p] for p in syms]

    trie = {}
    for w in {w for s in sents for w in words_of(s)} | _extra:
        try:
            node = trie
            for pid in encw(w):
                node = node.setdefault(pid, {})
        except KeyError:
            continue

    w = np.load(os.path.join(DD, "lm_piece32_fit.npz"))
    Ep = np.load(os.path.join(DD, "lm_piece32.npz"))
    b = np.load(os.path.join(DD, "bankp32_64.npz"))
    P0 = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
          "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
          "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
          "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    sys.path.insert(0, os.path.join(ROOT, "chain"))
    from qwen_mirror import load_layer0
    g, _, _ = load_layer0()
    Q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(896, 32)))
    PQ = dict(P0)
    PQ.update({"adUup": enc(Q.T.copy()), "adUdn": enc(Q.copy()),
               "adwup": enc(g("model.layers.0.mlp.up_proj.weight").T),
               "adwgate": enc(g("model.layers.0.mlp.gate_proj.weight").T),
               "adwdown": enc(g("model.layers.0.mlp.down_proj.weight").T),
               "adln": enc(g("model.layers.0.post_attention_layernorm.weight"))})
    import torch
    from transformers import AutoModelForCausalLM
    net = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", trust_remote_code=True,
        torch_dtype=torch.float32).eval()
    L0 = net.model.layers[0]
    Q2, _ = np.linalg.qr(np.random.default_rng(12).normal(size=(576, 32)))
    PS = dict(P0)
    PS.update({"adUup": enc(Q2.T.copy()), "adUdn": enc(Q2.copy()),
               "adwup": enc(L0.mlp.up_proj.weight.detach().numpy().T),
               "adwgate": enc(L0.mlp.gate_proj.weight.detach().numpy().T),
               "adwdown": enc(L0.mlp.down_proj.weight.detach().numpy().T),
               "adln": enc(L0.post_attention_layernorm.weight.detach().numpy())})
    stacks = [("base", CFG + open(os.path.join(SDIR, "lm_d32_headt.asm")).read(), P0),
              ("qwen-L0", CFG + open(os.path.join(SDIR, "lm_d32_adapt.asm")).read(), PQ),
              ("smol-L0", CFG + open(os.path.join(SDIR, "lm_d32_adapt_smol.asm")).read(), PS)]

    def generate(text, P, seed_ids, n, seedn, topk=12):
        out = list(seed_ids)
        rng = np.random.default_rng(seedn)
        frag = []
        for pid in out:
            frag.append(pid)
            if inv[pid].endswith("</w>"):
                frag = []
        for _ in range(n):
            ctx = out[-WIN:]
            pos = np.arange(len(ctx), dtype=np.int64)
            cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": np.array(ctx, np.int64), "pos": pos,
                              "cmask": cm, **P}, sigs=SIGS, basedir=SDIR)
            t = f["LOGITS"]
            lg = (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                  * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1].copy()
            node = trie
            for pid in frag:
                node = node.get(pid)
                if node is None:
                    break
            allowed = set(node) if node else set(trie)
            if allowed:
                mask = np.ones_like(lg, dtype=bool)
                mask[list(allowed)] = False
                lg[mask] = -1e9
            if len(out) >= NREP - 1:
                seen = {tuple(out[k:k + NREP]) for k in range(len(out) - NREP + 1)}
                prefix = tuple(out[-(NREP - 1):]) if NREP > 1 else ()
                for j in range(len(lg)):
                    if prefix + (j,) in seen:
                        lg[j] = -1e9
            keep = np.argsort(-lg)[:topk]
            wt = np.zeros_like(lg)
            wt[keep] = np.exp(lg[keep] - lg[keep].max())
            wt = wt / wt.sum()
            nxt = int(rng.choice(len(wt), p=wt))
            out.append(nxt)
            frag.append(nxt)
            if inv[nxt].endswith("</w>"):
                frag = []
        for _ in range(4):
            if inv[out[-1]].endswith("</w>"):
                break
            ctx = out[-WIN:]
            pos = np.arange(len(ctx), dtype=np.int64)
            cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": np.array(ctx, np.int64), "pos": pos,
                              "cmask": cm, **P}, sigs=SIGS, basedir=SDIR)
            t = f["LOGITS"]
            lg = (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                  * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1].copy()
            out.append(int(np.argmax(lg)))
        t = "".join(inv[i] for i in out).replace("</w>", " ")
        return " ".join(t.split())

    words, n, i, seedn = [], 20, 1, 7
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    print(f"seed: {seed}")
    for tag, text, P in stacks:
        print(f"[{tag}] {generate(text, P, encode(seed), n, seedn)}", flush=True)


if __name__ == "__main__":
    main()
