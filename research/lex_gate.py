"""Lexical-address gate: country-token early state as the address.

Full-state cosine can't separate same-template contexts (0.61/0.54:
near-parallel). This gates on the country token's EARLY residual
(layer 2, still mostly lexical) instead: address (early/lexical) and
content (late direction) as separate channels -- the positional
analogy made concrete. Same L27 contrast implant, scaled by the
sharpened lexical match. Specificity = Italy/Japan hold.
Usage: python3 research/lex_gate.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH, _layer, _get, _rms


def main():
    torch, g, tok = _get()
    dt = torch.float32
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]

    def early_key(prompt, layer=2):
        ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
        toks = tok.convert_ids_to_tokens(ids)
        pos = next(i for i, t in enumerate(toks)
                   if "ermany" in t or "taly" in t or "apan" in t)
        E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                         device="cuda")
        n = len(ids)
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
            _QR, _KR, _V = None, None, None
            from qwen_torch import _rope
            _QR, _KR, _V = (t.permute(1, 0, 2)
                            for t in (_rope(Q), _rope(K), V))
            P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                              + torch.triu(torch.full((n, n), float("-inf"),
                                                      device="cuda"), 1), dim=-1)
            x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
            hn = rms(x, w["post_attention_layernorm.weight"])
            down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                    * (hn @ w["mlp.up_proj.weight"].T)
                    ) @ w["mlp.down_proj.weight"].T
            x = x + down
        v = x[pos].detach().cpu().numpy()
        return v / np.linalg.norm(v)

    tfr = [fwdH("The capital of France is"), fwdH("The capital of Germany is")]
    d = tfr[0][0][27] - tfr[1][0][27]
    d /= np.linalg.norm(d)
    key = early_key("The capital of Germany is")
    base = {}
    for name, pr in (("Germany", "The capital of Germany is"),
                     ("Italy", "The capital of Italy is"),
                     ("Japan", "The capital of Japan is")):
        lg, _ = fwd(pr)
        base[name] = int(lg.argmax())
    print("unsteered:", {k: tok.decode([v]) for k, v in base.items()},
          flush=True)
    for tgt, tname in (("The capital of Germany is", "Germany"),
                       ("The capital of Italy is", "Italy"),
                       ("The capital of Japan is", "Japan")):
        mkey = early_key(tgt)
        m = float(mkey @ key)
        s = m ** 6 if m > 0 else 0.0
        lg, _ = fwd(tgt, steer=(27, d, 1.2 * 6 * s))
        top = int(lg.argmax())
        print(f"lex-gated {tname}: top={tok.decode([top])!r} "
              f"paris-rank={int((lg > lg[paris]).sum()) + 1} "
              f"match={m:.3f} gate={s:.3f} "
              f"ctl={'hold' if top == base[tname] or tname == 'Germany' else 'MOVED'}",
              flush=True)


if __name__ == "__main__":
    main()
