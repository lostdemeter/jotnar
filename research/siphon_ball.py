"""Siphon install through the yarn ball (teacher side, torch mirror).

Same proven recipe as lex_gate.py (L27 pair-contrast content x L2
lexical-early address), but the address x content x dose product lives
in yarnball_bank data + ledger, and the apply is yarnball_apply math
(softmax over the bank routes; non-targets land on null stores = hold
by construction). Bank: [Germany: Paris-value | Italy: null |
Japan: null]. Live-mined every run (no caches: stale-direction
species, siphon section 6).
Gates: Germany top=Paris rank 1; Italy/Japan tops == unsteered;
every prompt retrieves its own store.
Usage: python3 research/siphon_ball.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

COUNTRIES = ["Germany", "Italy", "Japan"]
TARGET = "Germany"
GAINS = [1, 2, 4, 8, 16]
KEY_SCALE = 8.0


def early_key(prompt, layer=2, pos=None):
    from qwen_torch import _layer, _get, _rms
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    toks = tok.convert_ids_to_tokens(ids)
    if pos is None:
        pos = next(i for i, t in enumerate(toks)
                   if "ermany" in t or "taly" in t or "apan" in t)
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt, device="cuda")
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    x = E
    for L in range(layer + 1):
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
        from qwen_torch import _rope
        _QR, _KR, _V = (t.permute(1, 0, 2) for t in (_rope(Q), _rope(K), V))
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((n, n), float("-inf"),
                                                  device="cuda"), 1), dim=-1)
        x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
        hn = rms(x, w["post_attention_layernorm.weight"])
        down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                * (hn @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
        x = x + down
    v = x[pos].detach().cpu().numpy()
    return v / np.linalg.norm(v), pos


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    _, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]

    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(promp[c])
        base[c] = int(lg.argmax())
    print("unsteered:", {c: tok.decode([v]) for c, v in base.items()}, flush=True)

    # content: pair-contrast L27 direction (France-minus-Germany installs)
    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    d = tfr[27] - tde[27]
    d /= np.linalg.norm(d)

    # address: L2 lexical keys per country (live-mined)
    keys = {}
    for c in COUNTRIES:
        k, pos = early_key(promp[c])
        keys[c] = k
        print(f"key {c}: country-tok-pos={pos}", flush=True)

    D = 3584
    Ua, Vc, ledger = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": keys["Germany"], "value": d, "dose": 1.0,
          "tier": "assoc", "support": "L27 France-minus-Germany contrast"},
         {"key": keys["Italy"], "value": np.zeros(D),
          "tier": "null", "support": "background (hold by construction)"},
         {"key": keys["Japan"], "value": np.zeros(D),
          "tier": "null", "support": "background (hold by construction)"}],
        key_scale=KEY_SCALE)
    print(f"ball: {Ua.shape[1]} stores, tiers={[r['tier'] for r in ledger]}",
          flush=True)

    for g in GAINS:
        row = []
        for c in COUNTRIES:
            t7, gids = fwdH(promp[c], keep="all")
            gtoks = tok.convert_ids_to_tokens(gids)
            frag = {"Germany": "ermany", "Italy": "taly",
                    "Japan": "apan"}[c]
            pos = next(i for i, t in enumerate(gtoks) if frag in t.lower())
            xa = t7[2][pos]
            xa = xa / np.linalg.norm(xa)
            C = xa @ Ua
            P = np.exp(C - C.max())
            P /= P.sum()
            x27 = t7[27][-1]
            mag = float(np.linalg.norm(x27))
            y = (P @ Vc) * (g * mag)  # gain in residual-magnitude units
            alpha = float(np.linalg.norm(y) / (mag + 1e-12))
            yn = y / (np.linalg.norm(y) + 1e-12)
            lg, _ = fwd(promp[c], steer=(27, yn, alpha))
            top = int(lg.argmax())
            pr = int((lg > lg[paris]).sum()) + 1
            ret = int(P.argmax())
            want = 0 if c == TARGET else COUNTRIES.index(c)
            mark = "RETR" if ret == want else f"ret{ret}"
            if c == TARGET:
                verdict = "INSTALL" if top == paris else ""
            else:
                verdict = "hold" if top == base[c] else "MOVED"
            row.append(f"{c[:4]}:{tok.decode([top])[:8]}^paris{pr}:{mark}:{verdict}")
        print(f"gain={g}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
