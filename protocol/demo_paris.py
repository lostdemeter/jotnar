"""Paris through messages (protocol Milestone 1 demo).

"The capital of France is" -> Paris, executed as client->messages->
server->messages->client through protocol/loopback.py. Client side is
smart (tokenizer, weights, drive loop — may import Qwen); the server
half stays model-blind (only paths + bytes + steps).

Needs: HF snapshot + the REPL workdir binary (/tmp/dec7b512, built by
README Demo 2; reuse is instant). GPU minutes only on first build.
KV relay (chat_K/V -> ckP/cvP copies) is client-side file work in the
shared workdir; every compute step crosses the wire as EXEC/EXEC-OK.
Usage: python3 protocol/demo_paris.py [--n 6]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

NL, HID, KVD, PER = 28, 3584, 512, 7
WANT = {"graph": True, "no_sync": True, "smax": 512, "sparse_bands": "",
        "sparse_k": 8, "resident": False, "fused_qkv": True}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workdir", default="/tmp/dec7b512")
    args = ap.parse_args()
    S, work = 512, args.workdir
    from chain.qwen7b import load7b, snapshot_ok
    if not snapshot_ok():
        print("SKIP (needs snapshot)")
        return
    try:
        stamp = json.load(open(os.path.join(work, "build.json")))
    except OSError:
        stamp = {}
    if any(stamp.get(k) != v for k, v in WANT.items()):
        sys.exit(f"{work} stamp {stamp} != {WANT}: build it first "
                 f"(README Demo 2: serve_decode.py --smax 512)")
    g, tok = load7b()
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    from chain.qwen7b import clear_7b_cache
    clear_7b_cache(g)
    import gc
    gc.collect()

    # -- program text is builder output (client side, model-blind) ------
    from serve_decode import build_decode
    prog, _, _ = build_decode(S)
    text = prog.text()

    # -- argv order mirrors demo_chat (fixed contract, argc-checked) ----
    wnames = ("wqkv", "wo", "wug", "wdown", "ln1", "ln2", "bqkv")
    argv = [os.path.join(work, f) for f in
            ("step_x0.bin", "step_pos.bin", "step_am.bin", "d_allrows.bin")]
    for L in range(NL):
        argv += [os.path.join(work, f"ckP{L}.bin"),
                 os.path.join(work, f"cvP{L}.bin")]
    for L in range(NL):
        for k in wnames:
            argv.append(os.path.join(work, f"d_{k}{L}.bin"))
    argv += [os.path.join(work, "d_lnf.bin"), os.path.join(work, "d_wlog.bin")]
    flo = os.path.join(work, "chat_lg.bin")
    fou = os.path.join(work, "chat_ou.bin")
    ko_f = [os.path.join(work, f"chat_K{L}.bin") for L in range(NL)]
    vo_f = [os.path.join(work, f"chat_V{L}.bin") for L in range(NL)]
    argv += [flo, fou] + ko_f + vo_f
    i_flo, i_fou = 4 + 56 + 196 + 2, 4 + 56 + 196 + 3
    n_kv = os.path.getsize(ko_f[0]) // 4
    outputs = {"logits": {"idx": i_flo, "dtype": "f32", "shape": [1, 152064]},
               "choice": {"idx": i_fou, "dtype": "i64", "shape": [1]}}
    # Output files append BLOCKED: all-K then all-V ([flo, fou] + ko + vo),
    # unlike the interleaved INPUT caches above. Index accordingly.
    i_k0, i_v0 = i_fou + 1, i_fou + 1 + NL
    for L in range(NL):
        outputs[f"knew{L}"] = {"idx": i_k0 + L, "dtype": "f32",
                               "shape": [n_kv]}
        outputs[f"vnew{L}"] = {"idx": i_v0 + L, "dtype": "f32",
                               "shape": [n_kv]}
    for L in range(NL):
        for tag in ("ckP", "cvP"):
            fn = os.path.join(work, f"{tag}{L}.bin")
            if not os.path.isfile(fn):
                np.zeros((S, KVD), dtype=np.float16).tofile(fn)
    live = {"step_x0": 0, "step_pos": 1, "step_am": 2}

    from protocol.loopback import Server, Client
    sv, cl = Server(), None
    cl = Client(Server())
    h, _ = cl.ping()
    assert h.get("ok"), h
    h, _ = cl.program("paris", text, backend="served",
                      config={"exe": os.path.join(work, "qdec"),
                              "argv_files": argv, "live": live,
                              "outputs": outputs,
                              "err_path": os.path.join(work, "paris.err")})
    assert h.get("ok"), h

    ids = tok("The capital of France is", return_tensors="pt")["input_ids"][0].tolist()
    out_names = (["logits", "choice"] +
                 [f"knew{L}" for L in range(NL)] + [f"vnew{L}" for L in range(NL)])

    def step(tok_id, n):
        xx = np.zeros((1, HID), dtype=np.float16)
        xx[0] = E[np.array([tok_id])].astype(np.float16)
        am = np.zeros((PER, S), dtype=np.int64)
        am[:, :n + 1] = 1
        h, out = cl.exec("paris", {"step_x0": xx,
                                   "step_pos": np.array([n], dtype=np.int64),
                                   "step_am": am}, out_names)
        assert h.get("ok"), h.get("error")
        for L in range(NL):
            np.asarray(out[f"knew{L}"], dtype=np.float32).astype(
                np.float16).tofile(os.path.join(work, f"ckP{L}.bin"))
            np.asarray(out[f"vnew{L}"], dtype=np.float32).astype(
                np.float16).tofile(os.path.join(work, f"cvP{L}.bin"))
        return np.asarray(out["logits"]).reshape(-1), int(np.asarray(out["choice"]).reshape(-1)[0])

    for i, t in enumerate(ids):
        step(t, i)
    seq, gen = list(ids), []
    for _ in range(args.n):
        lg, nxt = step(seq[-1], len(seq))
        seq.append(nxt)
        gen.append(nxt)
        if nxt == tok.eos_token_id:
            break
    text = tok.decode(gen, skip_special_tokens=True)
    print(f"paris-demo> {text}", flush=True)


if __name__ == "__main__":
    main()
