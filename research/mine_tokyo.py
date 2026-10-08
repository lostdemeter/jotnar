"""Mine a Tokyo-carrying direction (bounded screen).

France-minus-Japan carries Paris, not Tokyo (multi-fact log). Screen
candidates: China-minus-Japan, Spain-minus-Japan, largest-city-minus-
capital (both Tokyo-evocative, differential), lm_head Tokyo row
(predicted fail control: decoder-form overshoots, siphon section 6).
Gains 1/2/4 on the Japan prompt; Tokyo rank + top; Germany/Italy hold
check per cell. Winner (Tokyo r1 + holds) graduates to a bank.
Live-mined every run.
Usage: python3 research/mine_tokyo.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

GAINS = [1, 2, 4]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    g, tok = load7b()
    JP = "The capital of Japan is"
    tok_ids = {t: tok(" " + t, return_tensors="pt")["input_ids"][0].tolist()[0]
               for t in ("Tokyo", "Paris")}
    lg0, _ = fwd(JP)
    print(f"Japan base: top={tok.decode([int(lg0.argmax())])!r} "
          f"Tokyo-rank={int((lg0 > lg0[tok_ids['Tokyo']]).sum()) + 1}",
          flush=True)
    base_de, _ = fwd("The capital of Germany is")
    base_it, _ = fwd("The capital of Italy is")
    print(f"controls: Germany={tok.decode([int(base_de.argmax())])!r} "
          f"Italy={tok.decode([int(base_it.argmax())])!r}", flush=True)

    T = {}
    for name, pr in (("fr", "The capital of France is"),
                     ("de", "The capital of Germany is"),
                     ("jp", JP),
                     ("cn", "The capital of China is"),
                     ("es", "The capital of Spain is"),
                     ("big", "The largest city in Japan is")):
        t, _ = fwdH(pr)
        T[name] = t[27]
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-12)

    cands = {
        "cn-jp": unit(T["cn"] - T["jp"]),
        "es-jp": unit(T["es"] - T["jp"]),
        "big-jp": unit(T["big"] - T["jp"]),
        "wrow": unit(Wlog[tok_ids["Tokyo"]]),
    }
    print(f"candidate readouts: " +
          " ".join(f"{k}={float(unit(v) @ unit(Wlog[tok_ids['Tokyo']])):.3f}"
                   for k, v in cands.items()), flush=True)

    tjp, _ = fwdH(JP, keep="all")
    mag = float(np.linalg.norm(tjp[27][-1]))
    for name, d in cands.items():
        cells = []
        for gn in GAINS:
            lg, _ = fwd(JP, steer=(27, d, gn))
            top = int(lg.argmax())
            r = int((lg > lg[tok_ids["Tokyo"]]).sum()) + 1
            lde, _ = fwd("The capital of Germany is", steer=(27, d, gn))
            lit, _ = fwd("The capital of Italy is", steer=(27, d, gn))
            hd = "hold" if int(lde.argmax()) == int(base_de.argmax()) else "MOVED"
            hi = "hold" if int(lit.argmax()) == int(base_it.argmax()) else "MOVED"
            cells.append(f"g{gn}:{tok.decode([top])[:8]}r{r}/{hd}/{hi}")
        print(f"{name:6} " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
