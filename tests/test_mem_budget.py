"""Memory-budget gate: the 7B path must fit physical RAM (no swap).

Retired bulk-float64 plans (~61GiB sample + ~15GiB cache + HF on top)
exceeded a 61GiB box and killed an SSD; the streaming path peaks at
~2.5GiB CPU. This gate pins the estimator math and the refusal itself.

Stdlib-only by design (chain.mem has no third-party imports), so it
runs even when numpy/torch are absent. Weight-touching checks
(shapes_7b stubs, prep equivalence) are gated behind numpy + snapshot
and SKIP otherwise.
Usage: python3 tests/test_mem_budget.py
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import mem as M

    # -- estimator pins (pure ints, machine-independent) -----------------
    pc = M.param_counts(16)
    check("mem-total-params", pc["total"] == 7617551872, f"{pc['total']}")
    check("mem-per-layer", pc["layer_sum"] == 233053184, f"{pc['layer_sum']}")
    e = M.estimate_7b(16)
    check("mem-old-f64", e["old_cpu_f64"] == 7617551872 * 8 + 544997376 * 8,
          f"{e['old_cpu_f64'] / 2 ** 30:.1f}GiB")
    check("mem-stream-peak", e["stream_peak_cpu"] < 3 * 2 ** 30,
          f"{e['stream_peak_cpu'] / 2 ** 30:.2f}GiB")
    check("mem-vram", e["vram_fp16"] == 7617551872 * 2,
          f"{e['vram_fp16'] / 2 ** 30:.1f}GiB (24GiB card)")
    check("mem-ratio", e["old_cpu_f64"] > 20 * e["stream_peak_cpu"],
          f"{e['old_cpu_f64'] / e['stream_peak_cpu']:.0f}x smaller")

    # -- refusal itself (explicit tiny cap: no host dependence) ----------
    try:
        M.guard(e["old_cpu_f64"], "retired bulk path", cap_bytes=40 * 2 ** 30)
        check("mem-refuses-old", False, "bulk path was NOT refused")
    except M.BudgetExceeded as ex:
        check("mem-refuses-old", "Refusing to swap" in str(ex), "loud refusal")
    try:
        M.guard(e["stream_peak_cpu"], "streaming build", cap_bytes=40 * 2 ** 30)
        check("mem-allows-stream", True, "streaming fits with room")
    except M.BudgetExceeded:
        check("mem-allows-stream", False, "streaming wrongly refused")

    # -- default cap sanity (host-relative, informational but gated) -----
    try:
        cap = M.default_cap()
        check("mem-cap-floor", cap >= 1 * 2 ** 30, f"{cap / 2 ** 30:.1f}GiB")
        check("mem-old-exceeds-cap", e["old_cpu_f64"] > cap,
              "old path must exceed any sane cap")
    except M.BudgetExceeded:
        check("mem-cap-floor", False, "no meminfo and no override")

    # -- shape stubs (needs numpy; SKIP without) -------------------------
    try:
        import numpy as np  # noqa
    except ImportError:
        print("SKIP stub checks (needs numpy)", flush=True)
        print("FAILURES:", FAIL if FAIL else "none")
        sys.exit(1 if FAIL else 0)
    try:
        from chain.qwen7b import shapes_7b, snapshot_ok
        from chain.emit_c import FRef
    except ImportError as ex:
        print(f"SKIP stub checks ({ex})", flush=True)
        print("FAILURES:", FAIL if FAIL else "none")
        sys.exit(1 if FAIL else 0)
    s = shapes_7b(16)
    check("mem-stub-count", len(s) == 3 + 28 * 12 + 3, f"{len(s)} streams")
    check("mem-stub-zero-ram",
          sum(getattr(v, "nbytes", 0) for v in s.values()
              if isinstance(v, FRef)) == 0, "stubs carry no storage")
    check("mem-stub-wq", s["wq0"].shape == (3584, 3584), f"{s['wq0']}")
    check("mem-stub-bias", s["bq0"].shape == (16, 3584), f"{s['bq0']}")
    check("mem-stub-wlog", s["wlog"].shape == (3584, 152064), f"{s['wlog']}")
    check("mem-stub-pos-real", isinstance(np.asanyarray(s["pos"]), np.ndarray)
          and s["pos"].shape == (16,), "pos stays real (rope needs values)")
    # stub/real source equivalence on the C target too (shared
    # infer_shapes; the CUDA leg proved it, C was ungated until now)
    from chain.builder import Prog
    from chain.emit_c import compile_program, FRef
    pc = Prog("stub-c")
    pc.config("eps_rms", "1e-6")
    pc.inp("x0", "w", "ln")
    xnc = pc.op("RMSNORM", "x0", "ln", out="XN")
    yc = pc.op("MATMUL", xnc, "w", out="Y")
    rng = np.random.default_rng(0)
    real = {"x0": rng.normal(size=(4, 8)).astype(np.float64),
            "w": rng.normal(size=(8, 8)).astype(np.float64),
            "ln": np.abs(rng.normal(size=(8,))).astype(np.float64)}
    stub = {"x0": FRef((4, 8), "float16"), "w": FRef((8, 8), "float16"),
            "ln": FRef((8,), "float16")}
    ac = compile_program(pc.text(), "c", sample=stub, outputs=[yc],
                         basedir=os.path.join(ROOT, "programs"))
    bc = compile_program(pc.text(), "c", sample=real, outputs=[yc],
                         basedir=os.path.join(ROOT, "programs"))
    check("mem-stub-c-equiv", ac["source"] == bc["source"],
          "identical C source from stubs")
    if not snapshot_ok():
        print("SKIP weight-prep checks (no snapshot since SSD loss)", flush=True)
    else:
        from chain.qwen7b import load7b, prep_7b_weight, clear_7b_cache
        g, _ = load7b()
        wq = prep_7b_weight(g, "model.layers.0.self_attn.q_proj.weight",
                            "wq", 16)
        check("mem-prep-shape", wq.shape == (3584, 3584) and wq.dtype == np.float16,
              f"{wq.shape} {wq.dtype}")
        raw = g("model.layers.0.self_attn.q_proj.weight")
        ref = (raw.T / np.sqrt(128)).astype(np.float16)
        check("mem-prep-equiv",
              np.abs(wq.astype(np.float32) - ref.astype(np.float32)).max() < 1e-3,
              "fp32-scale matches retired fp64-then-cast within quantum")
        del wq, ref, raw
        clear_7b_cache(g)

    # -- full 28-layer graph from stubs (no weights, no nvcc) -------------
    # This is the gate that caught a transposed wup stub: the whole
    # 9859-op program must shape-check from zero-RAM refs alone.
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from gen_qwen7b_geo import build_program
        from chain.emit_c import compile_program
        p28 = build_program(16)
        ops = sum(1 for ln in p28.text().splitlines()
                  if "=" in ln and not ln.strip().startswith(("IN", "CONFIG")))
        check("mem-28layer-ops", ops == 9859, f"{ops} ops")
        art28 = compile_program(p28.text(), "cuda", sample=shapes_7b(16),
                                outputs=["LOGITS", "OUT"],
                                basedir=os.path.join(ROOT, "programs"),
                                use_fp16=True)
        check("mem-28layer-inputs", len(art28["inputs"]) == 341,
              f"{len(art28['inputs'])} inputs")
        # bmmv variant: one more frozen input (cmask7), fewer ops.
        p28b = build_program(16, bmmv=True)
        opsb = sum(1 for ln in p28b.text().splitlines()
                   if "=" in ln and not ln.strip().startswith(("IN", "CONFIG")))
        art28b = compile_program(p28b.text(), "cuda", sample=shapes_7b(16),
                                 outputs=["LOGITS", "OUT"],
                                 basedir=os.path.join(ROOT, "programs"),
                                 use_fp16=True)
        check("mem-bmmv-ops", opsb == 5351, f"{opsb} ops (was 9859)")
        check("mem-bmmv-inputs", len(art28b["inputs"]) == 342,
              f"{len(art28b['inputs'])} inputs (+cmask7)")
        # Every compiled input must resolve to a weight source by
        # exact-match lookup (the ln1@L10 = "ln110" trap lives here).
        from chain.qwen7b import input_sources
        srcs = input_sources()
        fixed = {"x0", "pos", "cmask", "lnf", "wlog"}
        missing = [n for n in art28["inputs"]
                   if n not in fixed and n not in srcs]
        check("mem-inputs-resolve", not missing, f"unmapped: {missing[:3]}")
        check("mem-ln-lookup", srcs["ln10"] == ("ln1", "model.layers.0.input_layernorm.weight")
              and srcs["ln110"] == ("ln1", "model.layers.10.input_layernorm.weight")
              and srcs["ln227"] == ("ln2", "model.layers.27.post_attention_layernorm.weight"),
              "ln1@L0/L10 + ln2@L27 exact")
    except ImportError as ex:
        print(f"SKIP 28-layer graph check ({ex})", flush=True)

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
