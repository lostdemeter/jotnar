"""Chat loop over the geometric decode server (multi-turn REPL).

One persistent decode binary; each turn re-prefills the full
conversation (Qwen chat template) then generates with host-boundary
sampling. Session state is messages + seeded RNG (replay a session by
re-entering lines with the same seed). Commands: /quit /reset /seed N
/topk K /temp T.
Usage: python3 demo_chat.py [--smax 512] [--n 40] [--workdir /tmp/dec7b512]
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

N_LAYERS = 28
HID, KVD, PER = 3584, 512, 7


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smax", type=int, default=512)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workdir", default="/tmp/dec7b512")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    S = args.smax
    work = args.workdir
    from chain.qwen7b import load7b, snapshot_ok
    if not snapshot_ok():
        sys.exit("SKIP (needs snapshot)")
    try:
        want = {"graph": False, "no_sync": False, "smax": S}
        import json as _js
        stamp = _js.load(open(os.path.join(work, "build.json")))
    except OSError:
        stamp = {}
    if any(stamp.get(k) != v for k, v in want.items()):
        sys.exit(f"{work} stamp {stamp} != {want}: build the decode "
                 f"binary there first (serve_decode.py --smax {S})")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from chain.serve import ServedExe
    g, tok = load7b()
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    from chain.qwen7b import clear_7b_cache
    clear_7b_cache(g)
    import gc as _gc
    _gc.collect()
    flo = os.path.join(work, "chat_lg.bin")
    fou = os.path.join(work, "chat_ou.bin")
    ko_f = [os.path.join(work, f"chat_K{L}.bin") for L in range(N_LAYERS)]
    vo_f = [os.path.join(work, f"chat_V{L}.bin") for L in range(N_LAYERS)]
    # argv follows build_decode's p.inp order exactly (6 + 56 live/caches
    # + 336 weights + lnf/wlog); the binary argc-checks (rc=9), so drift
    # fails loud, never silent.
    wnames = ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
              "ln1", "ln2", "bq", "bk", "bv")
    argv = [os.path.join(work, f) for f in
            ("step_x0.bin", "step_pos.bin", "step_am.bin", "step_oh.bin",
             "d_ones.bin", "d_allrows.bin")]
    # kP/vP INTERLEAVED per layer (build_decode order: kP0,vP0,kP1,vP1..;
    # all-k-then-all-v shifts every layer past 0 into garbage -- gated
    # by argc only in count, never in order, so this comment is the pin)
    for L in range(N_LAYERS):
        argv.append(os.path.join(work, f"ckP{L}.bin"))
        argv.append(os.path.join(work, f"cvP{L}.bin"))
    for L in range(N_LAYERS):
        for k in wnames:
            argv.append(os.path.join(work, f"d_{k}{L}.bin"))
    argv += [os.path.join(work, "d_lnf.bin"),
             os.path.join(work, "d_wlog.bin")]
    assert len(argv) == 6 + 56 + 336 + 2, len(argv)
    sv = ServedExe(os.path.join(work, "qdec"),
                    argv + [flo, fou] + ko_f + vo_f,
                    err_path=os.path.join(work, "chat.err"))
    for L in range(N_LAYERS):
        for tag in ("ckP", "cvP"):
            fn = os.path.join(work, f"{tag}{L}.bin")
            if not os.path.isfile(fn):
                np.zeros((S, KVD), dtype=np.float16).tofile(fn)
    np.zeros((1, HID), dtype=np.float16).tofile(
        os.path.join(work, "step_x0.bin"))
    np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
    np.zeros((PER, S), dtype=np.int64).tofile(
        os.path.join(work, "step_am.bin"))
    np.zeros((S, KVD), dtype=np.int64).tofile(
        os.path.join(work, "step_oh.bin"))
    print(f"chat: {S} slots, /quit /reset /seed N /topk K /temp T",
          flush=True)
    messages, seed, topk, temp = [], args.seed, args.topk, args.temp
    rng = np.random.default_rng(seed)
    try:
        sv.start()
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/reset":
                messages = []
                print("(context cleared)", flush=True)
                continue
            if line.startswith("/seed "):
                seed = int(line.split()[1])
                rng = np.random.default_rng(seed)
                print(f"(seed {seed})", flush=True)
                continue
            if line.startswith("/topk "):
                topk = int(line.split()[1])
                print(f"(topk {topk})", flush=True)
                continue
            if line.startswith("/temp "):
                temp = float(line.split()[1])
                print(f"(temp {temp})", flush=True)
                continue
            messages.append({"role": "user", "content": line})
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
            if len(ids) + args.n > S:
                print(f"(history too long: {len(ids)} toks, /reset)",
                      flush=True)
                messages.pop()
                continue
            for i, t in enumerate(ids):
                step_row(sv, work, E, t, i, S)
            seq = list(ids)
            out = []
            for _ in range(args.n):
                if len(seq) >= S:
                    break
                lg, _ = step_row(sv, work, E, seq[-1], len(seq), S,
                                 ret=True)
                nxt = pick(lg, rng, topk, temp)
                seq.append(nxt)
                out.append(nxt)
                if nxt == tok.eos_token_id:
                    break
            text = tok.decode(out, skip_special_tokens=True)
            print(f"jotnar> {text}", flush=True)
            messages.append({"role": "assistant", "content": text})
    finally:
        sv.close()


def step_row(sv, work, E, tok_id, n, S, ret=False):
    import numpy as np
    HID, KVD, PER, NL = 3584, 512, 7, 28
    xx = np.zeros((1, HID), dtype=np.float16)
    xx[0] = E[np.array([tok_id])].astype(np.float16)
    xx.tofile(os.path.join(work, "step_x0.bin"))
    np.arange(S, dtype=np.int64).tofile(os.path.join(work, "step_pos.bin"))
    am = np.zeros((PER, S), dtype=np.int64)
    am[:, :n + 1] = 1
    am.tofile(os.path.join(work, "step_am.bin"))
    oh = np.zeros((S, KVD), dtype=np.int64)
    oh[n, :] = 1
    oh.tofile(os.path.join(work, "step_oh.bin"))
    sv.step()
    lg = np.fromfile(os.path.join(work, "chat_lg.bin"),
                     dtype=np.float32).reshape(1, -1)
    ou = np.fromfile(os.path.join(work, "chat_ou.bin"), dtype=np.int64)
    for L in range(NL):
        for a, b, tag in (
                (os.path.join(work, f"chat_K{L}.bin"),
                 os.path.join(work, f"ckP{L}.bin"), "kP"),
                (os.path.join(work, f"chat_V{L}.bin"),
                 os.path.join(work, f"cvP{L}.bin"), "vP")):
            np.fromfile(a, dtype=np.float32).astype(
                np.float16).tofile(b)
    if ret:
        return lg[0], int(ou[0])
    return None, None


def pick(lg, rng, topk, temp):
    import numpy as np
    if topk <= 1 and temp == 1.0:
        return int(np.argmax(lg))
    sc = lg / max(temp, 1e-6)
    k = min(topk, sc.size)
    idx = np.argpartition(sc, -k)[-k:]
    sub = sc[idx] - sc[idx].max()
    ex = np.exp(sub.astype(np.float64))
    pr = ex / ex.sum()
    return int(idx[rng.choice(k, p=pr)])


if __name__ == "__main__":
    main()
