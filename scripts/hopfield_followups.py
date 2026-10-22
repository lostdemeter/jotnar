"""Hopfield follow-ups: beta-sweep + joint-24 memory (analysis, no training).

(1) Frozen 600-trial battery per beta in {0.1, 0.5, 2, 5, 10} on the
three per-point 8-pattern memories. (2) All 24 banked patterns in ONE
store, battery rerun with per-origin queries (correct = winner's frame
== query frame, any copy). Pure functions (gate recomputes from them);
main() banks samples/marigold/hopfield_followups.json. CPU-only,
numpy, deterministic. Token-level excluded (see constitution).
Usage: python3 scripts/hopfield_followups.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")

BETAS = (0.1, 0.5, 2.0, 5.0, 10.0)


def main():
    import hopfield_retrieval as R
    ids, X = R.load_prototypes()
    Xn = {p: X[p] / np.linalg.norm(X[p], axis=1, keepdims=True)
          for p in R.POINTS}

    # -- (1) beta sweep ---------------------------------------------------
    sweep = {}
    for beta in BETAS:
        for p in R.POINTS:
            hit, tot, convs, trials, maxit = 0, 0, 0, 0, 0
            for li, f in enumerate(R.LEVELS):
                for fi in range(len(ids)):
                    for seed in R.SEEDS:
                        w, it, conv = R.retrieve(
                            Xn[p], R.corrupt(Xn[p][fi], f, seed, li),
                            beta=beta)
                        hit += ids[w] == ids[fi]
                        tot += 1
                        convs += conv
                        trials += 1
                        maxit = max(maxit, it)
            acc = hit / tot
            sweep[f"{p}:b={beta}"] = {
                "accuracy": acc, "bar": R.BAR, "chance": R.CHANCE,
                "converged_frac": convs / trials, "max_iters": maxit,
                "clears": bool(acc >= R.BAR)}
            print(f"beta={beta} {p}: acc={acc:.4f} "
                  f"conv={convs / trials:.3f} maxit={maxit} "
                  f"{'CLEARS' if acc >= R.BAR else 'below'}", flush=True)

    # -- (2) joint-24 ------------------------------------------------------
    J = np.array([Xn[p] for p in R.POINTS]).reshape(-1, 3072)
    jhit, jtot, jconv, jtrials, jmaxit = 0, 0, 0, 0, 0
    for oi in range(3):
        for li, f in enumerate(R.LEVELS):
            for fi in range(len(ids)):
                for seed in R.SEEDS:
                    w, it, conv = R.retrieve(
                        J, R.corrupt(J[oi * 8 + fi], f, seed, li))
                    jhit += (w % 8) == fi
                    jtot += 1
                    jconv += conv
                    jtrials += 1
                    jmaxit = max(jmaxit, it)
    loo_h = 0
    for oi in range(3):
        for fi in range(len(ids)):
            qi = oi * 8 + fi
            others = [i for i in range(24) if i != qi]
            sims = J[others] @ J[qi]
            w = others[int(np.argmax(sims))]
            loo_h += R.scene_of(ids[w % 8]) == R.scene_of(ids[fi])
    jacc = jhit / jtot
    joint = {"patterns": 24, "trials": jtot, "accuracy": jacc,
             "bar": R.BAR, "chance": 3.0 / 24.0,
             "loo_accuracy": loo_h / 24, "loo_chance": 1.0 / 23.0,
             "converged_frac": jconv / jtrials, "max_iters": jmaxit,
             "clears": bool(jacc >= R.BAR)}
    print(f"joint-24: acc={jacc:.4f} (bar {R.BAR}, chance {3 / 24:.4f}) "
          f"loo={loo_h / 24:.3f} (chance {1 / 23:.3f}) "
          f"conv={jconv / jtrials:.3f} maxit={jmaxit} "
          f"{'CLEARS' if jacc >= R.BAR else 'below'}", flush=True)

    json.dump({"betas": list(BETAS), "sweep": sweep, "joint": joint},
              open(os.path.join(FIXDIR, "hopfield_followups.json"), "w"),
              indent=1)
    print("wrote samples/marigold/hopfield_followups.json", flush=True)


if __name__ == "__main__":
    main()
