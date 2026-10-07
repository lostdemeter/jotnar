"""Cleanliness screen: steered installs of Paris, scored with containment.

Grid: gains x layers x direction-forms on France/Germany (target Paris),
plus same-steering controls on Italy (must stay Rome). Score per cell:
target rank 1 + margin + control top unchanged. Host-side torch mirror
(fast); winners port to listings after.
Usage: python3 research/clean_screen.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

PAIRS = [("France", "Paris"), ("Germany", "Berlin"), (" Italy", "Rome")]
TPLS = ["The capital of {c} is", "{c}'s capital is the city of"]
LAYERS = [24, 25, 26, 27]
GAINS = [0.25, 0.5, 1.0, 1.5]


def main():
    import torch
    from chain.qwen7b import load7b
    g, tok = load7b()
    dt = torch.float32
    H = {}

    def fwd(prompt, steer=None):
        ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
        n = len(ids)
        E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                         device="cuda")
        eps = 1e-6

        def rms(x, w):
            return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

        def rope(x, base=1000000.0):
            Sq, Hh, D = x.shape
            i = torch.arange(D // 2, dtype=dt, device="cuda")
            th = base ** (-2.0 * i / D)
            ang = torch.arange(Sq, dtype=dt, device="cuda")[:, None] * th[None, :]
            c, s = torch.cos(ang), torch.sin(ang)
            c, s = c[:, None, :], s[:, None, :]
            y = torch.empty_like(x)
            y[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
            y[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
            return y

        x = E
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
            if steer is not None and L == steer[0]:
                d, alpha = steer[1], steer[2]
                mag = float(x[n - 1].norm())
                x = x + torch.tensor(d * alpha * mag, dtype=dt, device="cuda")
            xn = rms(x, w["input_layernorm.weight"])
            Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(n, 28, 128)
            K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(n, 4, 128)
            V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(n, 4, 128)
            K = K.repeat_interleave(7, dim=1)
            V = V.repeat_interleave(7, dim=1)
            _QR, _KR, _V = (t.permute(1, 0, 2) for t in (rope(Q), rope(K), V))
            P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                              + torch.triu(torch.full((n, n), float("-inf"),
                                                      device="cuda"), 1), dim=-1)
            x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
            hn = rms(x, w["post_attention_layernorm.weight"])
            down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                    * (hn @ w["mlp.up_proj.weight"].T)
                    ) @ w["mlp.down_proj.weight"].T
            x = x + down
        lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
        hn = rms(x, lnf)
        lg = (hn @ torch.tensor(g("lm_head.weight"), dtype=dt,
                               device="cuda").T)[n - 1].detach().cpu().numpy()
        return lg, ids

    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    rome = tok(" Rome", return_tensors="pt")["input_ids"][0].tolist()[0]
    base_de = fwd("The capital of Germany is")[0]
    base_it = fwd("The capital of Italy is")[0]
    ctl_top = int(base_it.argmax())
    print(f"controls: Germany top={tok.decode([int(base_de.argmax())])!r} "
          f"Italy top={tok.decode([ctl_top])!r}", flush=True)

    # directions need trajectories: wrap fwd to also dump H
    Hs = {}

    def fwdH(prompt):
        import torch as _t  # noqa
        ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
        n = len(ids)
        E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                         device="cuda")
        eps = 1e-6

        def rms(x, w):
            return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

        def rope(x, base=1000000.0):
            Sq, Hh, D = x.shape
            i = torch.arange(D // 2, dtype=dt, device="cuda")
            th = base ** (-2.0 * i / D)
            ang = torch.arange(Sq, dtype=dt, device="cuda")[:, None] * th[None, :]
            c, s = torch.cos(ang), torch.sin(ang)
            c, s = c[:, None, :], s[:, None, :]
            y = torch.empty_like(x)
            y[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
            y[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
            return y

        x = E
        traj = {}
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
            _QR, _KR, _V = (t.permute(1, 0, 2) for t in (rope(Q), rope(K), V))
            P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                              + torch.triu(torch.full((n, n), float("-inf"),
                                                      device="cuda"), 1), dim=-1)
            x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
            hn = rms(x, w["post_attention_layernorm.weight"])
            down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                    * (hn @ w["mlp.up_proj.weight"].T)
                    ) @ w["mlp.down_proj.weight"].T
            x = x + down
            traj[L] = x[n - 1].detach().cpu().numpy()
        return traj

    for t in TPLS:
        Hs[t] = (fwdH(t.format(c="France")), fwdH(t.format(c="Germany")))
    W = g("lm_head.weight").astype(np.float64)
    wparis = W[paris] / np.linalg.norm(W[paris])
    print(f"{'gain':>5} {'L':>3} {'form':>9} {'rank':>5} {'margin':>7} "
          f"{'ctl':>4}", flush=True)
    wins = []
    for L in LAYERS:
        d_con = Hs[TPLS[0]][0][L] - Hs[TPLS[0]][1][L]
        d_con /= np.linalg.norm(d_con)
        d_avg = np.mean([Hs[t][0][L] - Hs[t][1][L] for t in TPLS], axis=0)
        d_avg /= np.linalg.norm(d_avg)
        for fname, d in (("contrast", d_con), ("tpl-avg", d_avg),
                         ("decoder", wparis)):
            for a in GAINS:
                lg, _ = fwd("The capital of Germany is", steer=(L, d, a))
                o = np.argsort(-lg)
                rank = int((lg > lg[paris]).sum()) + 1
                mg = float(lg[o[0]] - lg[o[1]])
                lgc, _ = fwd("The capital of Italy is", steer=(L, d, a))
                ctl = "hold" if int(lgc.argmax()) == ctl_top else "MOVED"
                tag = ""
                if rank == 1 and mg > 1.0 and ctl == "hold":
                    tag = " <-- CLEAN"
                    wins.append((a, L, fname, mg))
                print(f"{a:5.2f} {L:3d} {fname:>9} {rank:5d} {mg:7.2f} "
                      f"{ctl:>4}{tag}", flush=True)
    print(f"clean cells: {wins if wins else 'none'}", flush=True)


if __name__ == "__main__":
    main()
