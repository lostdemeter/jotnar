"""Transport map: where does the country token ride to the end position?

Per-layer, per-head attention mass from the prompt-end query onto the
country-token key, over the prompt battery. Qwen2-7B GQA: 28 Q heads
sharing 4 KV groups (7 Q per group). Reports per-layer totals, hottest
head, and per-KV-group sums -- a concentrated hotspot is a native
mid-stream address (read where it lives, no host slicing) and an
install point with 28-L layers of model amplification behind it.
Local mirror loop (shared qwen_torch.py untouched); P captured live.
Usage: python3 research/transport_map.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

COUNTRIES = ["Germany", "France", "Italy", "Spain", "Japan", "China"]


def country_pos(prompt, country, tok):
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].numpy()
    lo = prompt.find(country)
    hi = lo + len(country)
    for i, (a, b) in enumerate(offs):
        if a < hi and lo < b:
            return i
    raise AssertionError(f"no token covers {country} in {prompt!r}")


def traj_with_attn(prompt):
    """Full 28-layer forward; returns (Pmass, n, cpos) with Pmass[L] =
    (28, n, n) attention probs (Q heads, GQA-expanded) on CPU."""
    from qwen_torch import _get, _rope
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt, device="cuda")
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    x = E
    Pm = []
    for L in range(28):
        p = f"model.layers.{L}."
        w = {k: torch.tensor(g(p + k), dtype=dt, device="cuda")
             for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.up_proj.weight", "mlp.gate_proj.weight",
                       "mlp.down_proj.weight", "input_layernorm.weight",
                       "post_attention_layernorm.weight")}
        bl = f"model.layers.{L}.self_attn."
        bq = torch.tensor(g(bl + "q_proj.bias"), dtype=dt, device="cuda")
        bk = torch.tensor(g(bl + "k_proj.bias"), dtype=dt, device="cuda")
        bv = torch.tensor(g(bl + "v_proj.bias"), dtype=dt, device="cuda")
        xn = rms(x, w["input_layernorm.weight"])
        Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(n, 28, 128)
        K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(n, 4, 128)
        V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(n, 4, 128)
        K = K.repeat_interleave(7, dim=1)
        V = V.repeat_interleave(7, dim=1)
        _QR, _KR, _V = (t.permute(1, 0, 2) for t in (_rope(Q), _rope(K), V))
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((n, n), float("-inf"),
                                                  device="cuda"), 1), dim=-1)
        Pm.append(P.detach().cpu().numpy())
        x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
        hn = rms(x, w["post_attention_layernorm.weight"])
        down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                * (hn @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
        x = x + down
    return Pm, n


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    _, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    acc = None
    for c in COUNTRIES:
        Pm, n = traj_with_attn(promp[c])
        cpos = country_pos(promp[c], c, tok)
        end = n - 1
        print(f"== {c} (len={n} country@{cpos} end@{end} uniform={1.0 / (end + 1):.3f})",
              flush=True)
        tot = np.zeros(28)
        for L in range(28):
            m = Pm[L][:, end, cpos]  # 28 Q heads
            tot[L] = m.sum()
            g4 = [round(float(m[g * 7:(g + 1) * 7].sum()), 3) for g in range(4)]
            hot = int(m.argmax())
            flag = " <== HOT" if tot[L] > 2.0 else ""
            print(f"  L{L:02}: total={tot[L]:6.3f} hot=h{hot}({m[hot]:.3f}) "
                  f"kv-groups={g4}{flag}", flush=True)
        acc = tot if acc is None else acc + tot
    print("MEAN over battery (per-layer total end->country mass, 28 heads):",
          flush=True)
    print("  " + " ".join(f"L{L}:{acc[L] / len(COUNTRIES):.2f}"
                           for L in range(28)), flush=True)


if __name__ == "__main__":
    main()
