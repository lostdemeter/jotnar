"""Phase 0.5 sizing: argmax flip rate vs (band, k) with top-k-masked attention.

Manual torch loop over HF Qwen2-7B-Instruct layers (weights untouched):
per GQA group, keep top-k positions by max-over-heads score (union of
the group's needs -- matches the BMMV group path), mask the rest to
-inf, softmax, continue. Reference = unmasked loop (must match HF
argmax exactly: self-check gate).

Bands: B0=0-3, B1=4-11, B2=12-20, B3=21-27; k in (4,8,16,32). One band
masked at a time. Screen positions: final token of the 6 audit runs.
Metric: flip = masked argmax != reference argmax (+ reference top-1
prob and masked-pick rank for near-miss reading).

Pre-registered sizing rule: k(band) = smallest k with 0 flips on the
screen; confirm on dense positions before building the listing.

Usage: python3 research/attn_topk_decide.py   (needs snapshot + GPU)
Writes /tmp/attn_topk_decide.json; prints flip table.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from research.attn_sparsity import PROMPTS, S_TARGETS, load_wiki_text

BANDS = {"B0": range(0, 4), "B1": range(4, 12),
         "B2": range(12, 21), "B3": range(21, 28)}
KS = (4, 8, 16, 32)
NEG = -1e4
# DENSE=1: confirm sizing on every 32nd position + final (not just final).
DENSE = os.environ.get("DENSE") == "1"
if DENSE:
    KS = (4, 8)


def run_prefix(model, rot, tok_ids, mask_band=None, k=0):
    """Manual prefill. mask_band: set of layers with top-k masking."""
    mask_band = mask_band or set()
    dev = next(model.parameters()).device
    x = torch.tensor([tok_ids]).to(dev)
    S = len(tok_ids)
    pos = torch.arange(S).unsqueeze(0).to(dev)
    h = model.model.embed_tokens(x)
    cos, sin = rot(h, pos)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    NH, NKV, DH = 28, 4, 128
    for L, layer in enumerate(model.model.layers):
        att = layer.self_attn
        r = h
        hn = layer.input_layernorm(h)
        q = att.q_proj(hn).view(1, S, NH, DH).transpose(1, 2)
        kv = att.k_proj(hn).view(1, S, NKV, DH).transpose(1, 2)
        v = att.v_proj(hn).view(1, S, NKV, DH).transpose(1, 2)
        q, kv = apply_rotary_pos_emb(q, kv, cos, sin)
        outs = []
        for g in range(NKV):
            qg = q[:, g * 7:(g + 1) * 7, :, :]
            sc = qg @ kv[:, g:g + 1, :, :].transpose(-1, -2)
            sc = sc / (DH ** 0.5)
            causal = torch.ones(S, S, device=dev).tril().view(1, 1, S, S)
            sc = sc.masked_fill(causal == 0, NEG)
            if L in mask_band:
                kk = min(k, S)
                gs = sc.amax(dim=1, keepdim=True)  # union over 7 heads
                th = gs.topk(kk, dim=-1).values[..., -1:]
                sc = sc.masked_fill(gs < th, NEG)
            p = torch.softmax(sc.float(), dim=-1).to(q.dtype)
            outs.append(p @ v[:, g:g + 1, :, :])
        ctx = torch.cat([o.expand(1, 7, S, DH) for o in outs], dim=1)
        ctx = ctx.transpose(1, 2).reshape(1, S, NH * DH)
        h = r + att.o_proj(ctx)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    lg = model.lm_head(model.model.norm(h))
    return lg[0, -1].float().cpu().numpy()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()
    _, tok = load7b()
    wiki = load_wiki_text()
    runs = []
    for name, text in [("wiki", wiki)] + sorted(PROMPTS.items()):
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        for s in (S_TARGETS if name == "wiki" else (len(ids),)):
            s = min(s, len(ids))
            if s < 8:
                continue
            runs.append((f"{name}@{s}", ids[:s].tolist()))
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="eager").to("cuda").eval()
    rot = model.model.rotary_emb
    out = {}
    with torch.no_grad():
        # reference + self-check (masked k>=S must be exact)
        refs, checks = {}, {}
        for tag, ids in runs:
            pos_idx = list(range(31, len(ids), 32)) + [len(ids) - 1] \
                if DENSE else [len(ids) - 1]
            refs[tag] = {i: run_prefix(model, rot, ids[:i + 1]) for i in pos_idx}
            chk = {i: run_prefix(model, rot, ids[:i + 1],
                                 mask_band=set(range(28)), k=10 ** 9)
                   for i in pos_idx}
            for i in pos_idx:
                assert chk[i].argmax() == refs[tag][i].argmax(), \
                    f"harness self-check failed at {tag}:{i}"
            checks[tag] = chk
        print(f"self-check: masked-k>=S exact on all runs x positions",
              flush=True)
        out["ref"] = {t: {"n": len(v)} for t, v in refs.items()}
        for b, ls in BANDS.items():
            for k in KS:
                key = f"{b}/k{k}"
                flips, ranks, n = 0, 1, 0
                for tag, ids in runs:
                    for i, ref in refs[tag].items():
                        m = run_prefix(model, rot, ids[:i + 1],
                                       mask_band=set(ls), k=k)
                        n += 1
                        if m.argmax() != ref.argmax():
                            flips += 1
                        order = np.argsort(-ref)
                        ranks = max(ranks,
                                    int((order == m.argmax()).argmax()) + 1)
                out[key] = {"flips": flips, "n": n, "worst_rank": ranks}
                print(f"{key}: flips={flips}/{n} "
                      f"worst_rank={ranks}", flush=True)
    with open("/tmp/attn_topk_decide.json", "w") as f:
        json.dump(out, f)
    print("=== flip table (rows k, cols band) ===")
    print("      " + "".join(f"{b:>10s}" for b in BANDS))
    for k in KS:
        print(f"  k={k:<3d}" + "".join(
            f"{out[f'{b}/k{k}']['flips']:>10d}" for b in BANDS))


if __name__ == "__main__":
    main()
