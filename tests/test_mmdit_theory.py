"""MMDiT-theory gate (docs/MMDIT_THEORY.md): CPU-only, seconds.

Legs (all read-only, no imports of surveyed code, no weights):
- source-integrity: sibling paths exist at pinned rev with every cited
  symbol present (string-verified, not trusted from notes); README rows
  cited carry the stated status tokens; banked-round files exist with
  key strings.
- mapping-completeness: M1-M12 each has a table status + cite in spec.
- no-implementation tripwire: named MMDiT/register/modulation component
  files absent from our tree (survey adds zero patterns).
Usage: python3 tests/test_mmdit_theory.py
"""
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIB = os.path.normpath(os.path.join(ROOT, "..", "mmdit-ro"))
SPEC = os.path.join(ROOT, "docs", "MMDIT_THEORY.md")

FAIL = []

T = os.path.join(SIB, "mmdit", "mmdit_pytorch.py")
G = os.path.join(SIB, "mmdit", "mmdit_generalized_pytorch.py")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def has(path, needle):
    try:
        return needle in open(path, errors="ignore").read()
    except OSError:
        return False


def main():
    spec = open(SPEC).read()

    rev = ""
    try:
        rev = subprocess.run(["git", "-C", SIB, "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception:
        rev = ""
    check("mmdit-rev",
          rev.startswith("76a5846") and "76a5846" in spec,
          f"sibling @ {rev[:7]} (pinned, read-only)")

    syms = [(T, "class JointAttention"), (T, "self.to_qkv = ModuleList"),
            (T, "self.to_out = ModuleList"), (T, "qk_rmsnorm = False"),
            (T, "class MultiHeadRMSNorm"), (T, "class MMDiTBlock"),
            (T, "nn.init.zeros_(to_cond_linear.weight)"),
            (T, "elementwise_affine = not has_cond"),
            (T, "skip_feedforward_text_tokens"),
            (T, "num_register_tokens = 0"), (T, "class MMDiT"),
            (T, "num_residual_streams = 4"),
            (G, "class AdaptiveLayerNorm"), (G, "class MMDiTBlock"),
            (G, "dim_modalities")]
    missing = [s for p, s in syms if not has(p, s)]
    check("mmdit-symbols", not missing,
          f"{len(syms) - len(missing)}/{len(syms)} symbols; missing={missing[:3]}")

    rows = ["MATMUL", "BMMV", "SOFTMAX_WIDE", "CONCAT", "RMSNORM",
            "LAYERNORM", "SILU", "MUL", "ADD", "SUB", "DIV", "CLIP",
            "SELECT", "BETA"]
    readme = open(os.path.join(ROOT, "README.md")).read().splitlines()
    lines = [ln for ln in readme if ln.startswith("| ")]
    bad = [op for op in rows if not any(f"| {op} |" in ln for ln in lines)]
    check("mmdit-rows", not bad, f"{len(rows) - len(bad)}/{len(rows)} rows; bad={bad[:3]}")
    check("mmdit-parked",
          "parked: drilled in test_asm" in open(os.path.join(ROOT, "README.md")).read()
          and "parked" in spec,
          "CLIP parked status cross-checked README<->spec")
    check("mmdit-notanh",
          not any("TANH" in ln for ln in lines) and "no TANH row" in spec,
          "softclamp gap stated (no TANH row)")

    mechs = [f"M{i}" for i in range(1, 13)]
    check("mmdit-mapping",
          all(m in spec for m in mechs)
          and spec.lower().count("tabled") >= 8 and "parked" in spec
          and "absent-with-price" in spec,
          f"{sum(m in spec for m in mechs)}/12 mechanisms mapped")
    check("mmdit-steals",
          all(f"**{t}**" in spec or t in spec for t in
              ["Register tokens", "Logit-normal", "Modulation budgets",
               "QK-RMSNorm", "softclamp", "Two-stream", "HyperConnections"])
          and "Falsifier" in spec,
          "ranked steal-list with falsifiers")

    banked = [("docs/STUDENT.md", "0/156"), ("docs/STUDENT.md", "0/36"),
              ("docs/STUDENT.md", "1.27"), ("docs/PATTERN_GAPS.md", "0.31x"),
              ("docs/HOPFIELD.md", "PR_24")]
    missing_b = [f for f, s in banked
                 if not has(os.path.join(ROOT, f), s)]
    check("mmdit-banked", not missing_b,
          f"{len(banked) - len(missing_b)}/{len(banked)} failure cites; missing={missing_b[:2]}")

    forbidden = ["chain/mmdit.py", "chain/joint_attn.py", "chain/register.py",
                 "chain/modulation.py", "chain/softclamp.py",
                 "chain/hyperconn.py", "programs/mmdit.asm",
                 "programs/register.asm", "scripts/build_mmdit.py",
                 "docs/MMDIT_BUILD.md"]
    present = [f for f in forbidden if os.path.exists(os.path.join(ROOT, f))]
    check("mmdit-tripwire", not present, f"forbidden present={present}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
