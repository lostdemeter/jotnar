"""Paris through messages (protocol Milestone 1 demo).

"The capital of France is" -> Paris, executed as client->messages->
server->messages->client through protocol/loopback.py. Client side is
smart (tokenizer, weights, drive loop — may import Qwen); the server
half stays model-blind (only paths + bytes + steps).

Needs: HF snapshot + the REPL workdir binary (/tmp/dec7b512, built by
README Demo 2; reuse is instant). GPU minutes only on first build.
KV relay (chat_K/V -> ckP/cvP copies) is client-side file work in the
shared workdir; every compute step crosses the wire as EXEC/EXEC-OK.
--tcp serves the same demo over real TCP sockets (default loopback).
Usage: python3 protocol/demo_paris.py [--n 6]
       python3 protocol/demo_paris.py --tcp --workdir /tmp/protodec512 --n 6
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
_WANT_BASE = {"graph": True, "no_sync": True, "sparse_bands": "",
              "sparse_k": 8, "resident": False, "fused_qkv": True}
WANT = dict(_WANT_BASE, smax=512)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workdir", default="/tmp/dec7b512")
    ap.add_argument("--tcp", action="store_true",
                    help="serve over real TCP sockets (in-process server "
                    "thread, loopback interface); default is in-process "
                    "loopback transport")
    ap.add_argument("--pipeline", default="",
                    help="layer splits like 0-13,14-27: build one binary "
                    "per stage, ferry the f32 boundary, caches sticky "
                    "per stage (needs --smax, distinct workdirs)")
    ap.add_argument("--smax", type=int, default=512)
    args = ap.parse_args()
    S, work = args.smax, args.workdir
    if args.pipeline and not args.tcp:
        sys.exit("--pipeline needs --tcp (stages are socket nodes)")
    if args.pipeline:
        return pipeline_demo(args)
    S, work = args.smax, args.workdir
    WANT = dict(_WANT_BASE, smax=S)
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
    tsv = None
    if args.tcp:
        import threading
        from protocol.tcp import TcpServer, TcpClient
        tsv = TcpServer(Server(), port=0)
        th = threading.Thread(target=tsv.serve_forever, daemon=True)
        th.start()
        cl = TcpClient("127.0.0.1", tsv.port)
    else:
        cl = Client(Server())
    h, _ = cl.ping()
    assert h.get("ok"), h
    h, _ = cl.program("paris", text, backend="served",
                      config={"exe": os.path.join(work, "qdec"),
                              "argv_files": argv, "live": live,
                              "outputs": outputs,
                              "err_path": os.path.join(work, "paris.err")},
                      stamp=dict(WANT))
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
    if tsv is not None:
        tsv.stop()
        tsv.close()


def pipeline_demo(args):
    """Layer-sharded Paris: one binary per split, f32 H ferried over
    sockets, caches sticky per stage. Plus a hetero kicker (integer
    chain on the lattice node through the same controller)."""
    import threading
    S = args.smax
    splits = []
    for part in args.pipeline.split(","):
        lo, hi = part.split("-")
        splits.append((int(lo), int(hi)))
    workbase = args.workdir
    if args.workdir == "/tmp/dec7b512":
        workbase = f"/tmp/pipe{S}"  # never build stages into dec7b512
    from chain.qwen7b import load7b, snapshot_ok, input_sources
    from chain.qwen7b import clear_7b_cache
    if not snapshot_ok():
        print("SKIP (needs snapshot)")
        return
    g, tok = load7b()
    E = np.ascontiguousarray(g("model.embed_tokens.weight"), dtype=np.float16)
    clear_7b_cache(g)
    import gc
    gc.collect()
    from protocol.pipe_build import build_pipeline
    from protocol.tcp import TcpServer, TcpClient
    from protocol.loopback import Server
    from protocol.pipeline import PipeController
    cfgs = build_pipeline(g, dict(input_sources()), S, splits, workbase)
    srvs = []
    for _ in cfgs:
        s = TcpServer(Server(), port=0)
        t = threading.Thread(target=s.serve_forever, daemon=True)
        t.start()
        srvs.append(s)
    try:
        ctl = PipeController()
        # lattice node first: route_for picks the first advertiser,
        # so the hetero chain lands on the dedicated integer node
        # (stages are explicitly addressed, order-independent).
        lat = TcpServer(Server(allowed=["lattice"]), port=0)
        lt = threading.Thread(target=lat.serve_forever, daemon=True)
        lt.start()
        ctl.add_node("lattice", TcpClient("127.0.0.1", lat.port))
        for i in range(len(cfgs)):
            ctl.add_node(f"s{i}", TcpClient("127.0.0.1", srvs[i].port))
        try:
            for i, cfg in enumerate(cfgs):
                argv, nlay = cfg["argv"], cfg["hi"] - cfg["lo"] + 1
                n_in = len(argv) - (2 + 2 * nlay if cfg["last"]
                                    else 1 + 2 * nlay)
                if cfg["last"]:
                    outs = {"logits": {"idx": n_in, "dtype": "f32",
                                       "shape": [1, 152064]},
                            "choice": {"idx": n_in + 1, "dtype": "i64",
                                       "shape": [1]}}
                else:
                    outs = {"H": {"idx": n_in, "dtype": "f32",
                                  "shape": [1, HID]}}
                for L in range(cfg["lo"], cfg["hi"] + 1):
                    nlay = cfg["hi"] - cfg["lo"] + 1
                    b = n_in + (2 if cfg["last"] else 1)
                    # BLOCKED outputs (all-K then all-V), not interleaved
                    outs[f"knew{L}"] = {"idx": b + (L - cfg["lo"]),
                                        "dtype": "f32",
                                        "shape": [S * KVD]}
                    outs[f"vnew{L}"] = {"idx": b + nlay + (L - cfg["lo"]),
                                        "dtype": "f32",
                                        "shape": [S * KVD]}
                ctl.program(f"s{i}", f"st{i}", cfg["text"],
                            backend="served",
                            config={"exe": cfg["exe"], "argv_files": argv,
                                    "live": {"step_x0": 0, "step_pos": 1,
                                             "step_am": 2},
                                    "outputs": outs,
                                    "err_path": os.path.join(cfg["work"], "pipe.err")},
                            stamp=dict(cfg["stamp"]))
            first, mid, last = cfgs[0], cfgs[1:-1], cfgs[-1]

            def stage_exec(tag, cfg, x_arr, n, am, want):
                h, o = ctl.exec(tag, {"step_x0": x_arr,
                                      "step_pos": np.array([n], dtype=np.int64),
                                      "step_am": am}, want)
                assert h.get("ok"), h.get("error")
                for L in range(cfg["lo"], cfg["hi"] + 1):
                    w = os.path.join(cfg["work"], f"ckP{L}.bin")
                    a = np.asarray(o[f"knew{L}"], dtype=np.float32)
                    assert a.size == S * KVD, (L, a.shape)
                    a.tofile(w)
                    w = os.path.join(cfg["work"], f"cvP{L}.bin")
                    a = np.asarray(o[f"vnew{L}"], dtype=np.float32)
                    assert a.size == S * KVD, (L, a.shape)
                    a.tofile(w)
                return o

            def step(tok_id, n):
                xx = np.zeros((1, HID), dtype=np.float32)
                xx[0] = E[np.array([tok_id])].astype(np.float32)
                am = np.zeros((PER, S), dtype=np.int64)
                am[:, :n + 1] = 1
                names = (["H"] + [f"knew{L}" for L in range(first["lo"], first["hi"] + 1)] +
                         [f"vnew{L}" for L in range(first["lo"], first["hi"] + 1)])
                o = stage_exec("st0", first, xx, n, am, names)
                H = np.asarray(o["H"], dtype=np.float32)
                for cfg, tag in [(c, f"st{j + 1}") for j, c in enumerate(mid)]:
                    names = (["H"] +
                             [f"knew{L}" for L in range(cfg["lo"], cfg["hi"] + 1)] +
                             [f"vnew{L}" for L in range(cfg["lo"], cfg["hi"] + 1)])
                    o = stage_exec(tag, cfg, H, n, am, names)
                    H = np.asarray(o["H"], dtype=np.float32)
                names = (["logits", "choice"] +
                         [f"knew{L}" for L in range(last["lo"], last["hi"] + 1)] +
                         [f"vnew{L}" for L in range(last["lo"], last["hi"] + 1)])
                o = stage_exec(f"st{len(cfgs) - 1}", last, H, n, am, names)
                return (np.asarray(o["logits"]).reshape(-1),
                        int(np.asarray(o["choice"]).reshape(-1)[0]))

            ids = tok("The capital of France is",
                      return_tensors="pt")["input_ids"][0].tolist()
            for i, t in enumerate(ids):
                step(t, i)
            seq, gen = list(ids), []
            for _ in range(args.n):
                _, nxt = step(seq[-1], len(seq))
                seq.append(nxt)
                gen.append(nxt)
            print(f"paris-demo> {tok.decode(gen, skip_special_tokens=True)}",
                  flush=True)
            import phi_core.lattice as _S
            rng = np.random.default_rng(3)
            A = _S.encode(np.ascontiguousarray(rng.normal(size=(2, 4)) * .5))
            B = _S.encode(np.ascontiguousarray(rng.normal(size=(4, 3)) * .5))
            C = _S.encode(np.ascontiguousarray(rng.normal(size=(2, 3)) * .5))
            ctl.program_auto("ichain", _ICHAIN, backend="lattice")
            h, o = ctl.exec("ichain", {"a": A, "b": B, "c": C}, ["OUT"])
            print(f"hetero: lattice chain ok={h.get('ok')} "
                  f"placed={ctl.placement('ichain')}", flush=True)
            print(f"placement log: {ctl.log}", flush=True)
        finally:
            lat.stop()
            lat.close()
    finally:
        for s in srvs:
            s.stop()
            s.close()


_ICHAIN = """CONFIG m_acc 31344
CONFIG m_cov 31344
IN a AS T:SEQ
IN b AS T:SEQ
IN c AS T:SEQ
MID = MATMUL(a, b)
SUM = ADD(MID, c)
OUT = SILU(SUM)
"""


if __name__ == "__main__":
    main()
