"""Generation-quality battery: HF vs geo over N prompts, JSONL records.

Modes (memory discipline: never both sides in one process):
  --hf-only    HF transformers generations -> OUT.jsonl
  --geo-only   geometric serve generations -> OUT.jsonl (reuses /tmp/gen7b
               bins+binary when the build stamp matches; else builds)
  --compare A.jsonl B.jsonl
Compares: fork position (first differing token), first-pick rank,
top-5 overlap at prompt end, horizons. Parity (one prompt, one row)
can't see distribution drift; this can.
Usage: python3 scripts/eval_battery.py --hf-only --out /tmp/b_hf.jsonl
       python3 scripts/eval_battery.py --geo-only --out /tmp/b_geo.jsonl
       python3 scripts/eval_battery.py --compare /tmp/b_hf.jsonl /tmp/b_geo.jsonl
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "Alexander the Great founded",
    "The Roman Empire fell in",
    "Water boils at",
    "The quick brown fox",
    "Once upon a time",
    "def fibonacci(n):",
    "The president of the United States is",
    "In the beginning",
    "Machine learning is",
    "The Eiffel Tower is located in",
]
N_GEN = 8
SMAX = 16
WORKDIR = "/tmp/gen7b"


def tok_of():
    from chain.qwen7b import load7b
    _, tok = load7b()
    return tok


def hf_run(out, n_gen=N_GEN, smax=SMAX):
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, hf_no_triton
    hf_no_triton()
    _, tok = __import__("chain.qwen7b", fromlist=["load7b"]).load7b()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    recs = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok(pr, return_tensors="pt")["input_ids"][0].tolist()
            assert len(ids) + n_gen <= smax, f"prompt too long: {pr}"
            out_ids = model.generate(
                **tok(pr, return_tensors="pt").to("cuda"),
                max_new_tokens=n_gen, do_sample=False,
                pad_token_id=tok.eos_token_id)[0].tolist()
            lg = model(**tok(pr, return_tensors="pt").to("cuda")).logits
            top = lg[0, -1].float().cpu().numpy()
            recs.append({"prompt": pr, "ids": ids, "gen": out_ids[len(ids):],
                         "top5": sorted(range(len(top)),
                                        key=lambda i: -top[i])[:5]})
            print(f"hf: {pr!r:45.45} -> "
                  f"{tok.decode(out_ids[len(ids):])[:40]!r}", flush=True)
    del model
    torch.cuda.empty_cache()
    json.dump(recs, open(out, "w"))
    print(f"wrote {out} ({len(recs)} prompts)", flush=True)


def geo_run(out, n_gen=N_GEN, smax=SMAX, workdir=WORKDIR):
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from gen_qwen7b_geo import build_program
    from chain.qwen7b import shapes_7b, load7b, clear_7b_cache
    from chain.emit_c import compile_program
    from chain.serve import ServedExe
    S, HID = smax, 3584
    want = {"bmmv": True, "graph": True, "no_sync": True,
            "oneshot": False, "smax": S}
    work = workdir
    try:
        stamp = json.load(open(os.path.join(work, "build.json")))
    except OSError:
        stamp = {}
    if any(stamp.get(k) != v for k, v in want.items()):
        sys.exit(f"{work} stamp {stamp} != {want}: run gen_qwen7b_geo.py "
                 f"--bmmv --graph --no-sync once first (builds bins+binary)")
    prog = build_program(S, bmmv=True)
    art = compile_program(prog.text(), "cuda", sample=shapes_7b(S),
                          outputs=["LOGITS", "OUT"],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True, live=["x0", "pos", "cmask"],
                          sync_each=False, graph=True)
    g, tok = load7b()
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    flo = os.path.join(work, "step_lg.bin")
    fou = os.path.join(work, "step_ou.bin")

    def live_argv():
        argv = []
        for name in art["inputs"]:
            if name == "x0":
                argv.append(os.path.join(work, "step_x0.bin"))
            elif name == "pos":
                argv.append(os.path.join(work, "step_pos.bin"))
            elif name == "cmask":
                argv.append(os.path.join(work, "step_cm.bin"))
            else:
                argv.append(os.path.join(work, f"fr_{name}.bin"))
        return argv + [flo, fou]

    sv = ServedExe(os.path.join(work, "q7b"), live_argv(),
                    err_path=os.path.join(work, "batt.err"))
    recs = []
    try:
        # prime live files so the resident load has valid shapes
        ids0 = tok(PROMPTS[0], return_tensors="pt")["input_ids"][0].tolist()
        xx = np.zeros((S, HID), dtype=np.float16)
        xx[:len(ids0)] = E[np.array(ids0)]
        xx.tofile(os.path.join(work, "step_x0.bin"))
        np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
        np.tril(np.ones((S, S), dtype=np.int64)).tofile(
            os.path.join(work, "step_cm.bin"))
        sv.start()
        for pr in PROMPTS:
            ids = tok(pr, return_tensors="pt")["input_ids"][0].tolist()
            assert len(ids) + n_gen <= S, f"prompt too long: {pr}"
            seq, abspos = list(ids), list(range(len(ids)))
            for _ in range(n_gen):
                if len(seq) >= S:
                    break
                xx = np.zeros((S, HID), dtype=np.float16)
                xx[:len(seq)] = E[np.array(seq)].astype(np.float16)
                xx.tofile(os.path.join(work, "step_x0.bin"))
                np.arange(S, dtype=np.int64).tofile(
                    os.path.join(work, "step_pos.bin"))
                np.tril(np.ones((S, S), dtype=np.int64)).tofile(
                    os.path.join(work, "step_cm.bin"))
                sv.step()
                lg = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
                ou = np.fromfile(fou, dtype=np.int64)
                seq.append(int(ou[len(seq) - 1]))
                abspos.append(abspos[-1] + 1)
            # prompt-end top5 needs a dedicated forward (cache the row)
            xx = np.zeros((S, HID), dtype=np.float16)
            xx[:len(ids)] = E[np.array(ids)].astype(np.float16)
            xx.tofile(os.path.join(work, "step_x0.bin"))
            np.arange(S, dtype=np.int64).tofile(
                os.path.join(work, "step_pos.bin"))
            sv.step()
            lgp = np.fromfile(flo, dtype=np.float32).reshape(S, -1)
            row = lgp[len(ids) - 1]
            recs.append({"prompt": pr, "ids": ids, "gen": seq[len(ids):],
                         "top5": sorted(range(len(row)),
                                        key=lambda i: -row[i])[:5]})
            print(f"geo: {pr!r:45.45} -> "
                  f"{tok.decode(seq[len(ids):])[:40]!r}", flush=True)
    finally:
        sv.close()
    json.dump(recs, open(out, "w"))
    print(f"wrote {out} ({len(recs)} prompts)", flush=True)


def compare(fa, fb):
    A = {r["prompt"]: r for r in json.load(open(fa))}
    B = {r["prompt"]: r for r in json.load(open(fb))}
    forks, ovs = [], []
    for pr in PROMPTS:
        a, b = A[pr]["gen"], B[pr]["gen"]
        f = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if f is None:
            f = min(len(a), len(b)) if len(a) != len(b) else None
        forks.append(f)
        ovs.append(len(set(A[pr]["top5"]) & set(B[pr]["top5"])))
        tag = "IDENTICAL" if f is None else f"fork@{f}"
        print(f"{pr!r:40.40} {tag:12} top5∩={ovs[-1]}/5", flush=True)
    ident = sum(1 for f in forks if f is None)
    print(f"identical: {ident}/{len(PROMPTS)}; "
          f"mean top5 overlap: {sum(ovs) / len(ovs):.2f}/5", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-only", action="store_true")
    ap.add_argument("--geo-only", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("HF", "GEO"))
    ap.add_argument("--out", default="/tmp/batt.jsonl")
    ap.add_argument("--smax", type=int, default=SMAX)
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--n", type=int, default=N_GEN)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.hf_only:
        hf_run(args.out, n_gen=args.n, smax=args.smax)
    elif args.geo_only:
        geo_run(args.out, n_gen=args.n, smax=args.smax,
                workdir=args.workdir)
    else:
        sys.exit("need --hf-only, --geo-only, or --compare A B")


if __name__ == "__main__":
    main()
