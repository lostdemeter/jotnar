"""Siphon demo: install a fact through listings, calibrated.

Default fact Germany/Paris (cross-install with strong anchor -- proven
rank-1 with controls holding). Fact selection matters: weak-anchor
facts (Portugal rank 2-by-luck, Japan/top-junk) resist installation
or worse (pair directions become noise); the demo documents its fact
as installable, and the uninstallable class is a finding, not a
failure (minimum-information lower bound).
Takes the fact, mines direction + key, sweeps gains with Germany/Italy
controls (gain 0.0 = unsteered baseline through the SAME path),
calibrates the first gain with target rank-1 + controls holding, and
reports before/after. One command: the siphon as a product.
Usage: python3 demo_siphon.py [--country Germany] [--capital " Paris"]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

GAINS = [0.0, 0.5, 0.7, 1.0]
CONTROL_POOL = ["The capital of Germany is", "The capital of Italy is",
                "The capital of Japan is", "The capital of Spain is"]


def run_steer(prompt, country, capital, gain, workdir, reuse):
    env = dict(os.environ)
    if reuse:
        env["QWEN_REUSE_BIN"] = "1"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_steer.py"),
         prompt, "--n", "0", "--gain", str(gain), "--country", country,
         "--capital", capital, "--workdir", workdir],
        capture_output=True, text=True, cwd=ROOT, env=env)
    rec = {"rc": r.returncode}
    for ln in r.stdout.splitlines():
        m = re.match(r"gate match=([\d.]+) s=([\d.]+) dose=([\d.]+)", ln)
        if m:
            rec["gate"] = float(m.group(1))
        m = re.match(r"steered pick: (.*) target-rank=(\d+)", ln)
        if m:
            rec["pick"] = m.group(1)
            rec["rank"] = int(m.group(2))
    if r.returncode != 0:
        rec["err"] = (r.stdout[-500:] + r.stderr[:200])
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="Germany")
    ap.add_argument("--capital", default=" Paris")
    ap.add_argument("--workdir", default="/tmp/siphon_demo")
    args = ap.parse_args()
    prompt = f"The capital of {args.country} is"
    controls = [p for p in CONTROL_POOL if args.country not in p][:2]
    print(f"fact: {prompt} ->{args.capital}", flush=True)
    print(f"controls: {controls}", flush=True)
    prompts = [prompt] + controls
    first = True
    table = {}
    for g in GAINS:
        for p in prompts:
            # first call builds (no reuse); rest reuse binaries
            t = run_steer(p, args.country, args.capital, g, args.workdir,
                          reuse=not first)
            first = False
            table[(g, p)] = t
            status = (f"rank={t.get('rank')} pick={t.get('pick')}"
                      if t.get("rc") == 0 else f"RC={t.get('rc')}")
            print(f"gain={g} {p[14:22]:>8}: {status}", flush=True)
            if t.get("rc") != 0:
                print(t.get("err", "")[-300:], flush=True)
                sys.exit(1)
    base = {p: table[(0.0, p)]["pick"] for p in prompts}
    print(f"baselines: {base}", flush=True)
    for g in GAINS[1:]:
        t = table[(g, prompt)]["rank"]
        holds = all(table[(g, p)]["pick"] == base[p] for p in controls)
        if t == 1 and holds:
            print(f"CALIBRATED gain={g}: {prompt} -> "
                  f"{table[(g, prompt)]['pick']} "
                  f"(controls {[table[(g, p)]['pick'] for p in controls]})",
                  flush=True)
            return
    print("no gain installed rank-1 with controls holding", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
