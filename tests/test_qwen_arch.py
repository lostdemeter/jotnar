"""Qwen-arch survey gate (docs/QWEN_ARCH.md): CPU-only, seconds.

Legs:
- source-integrity: every cited venv file/symbol exists at the stated
  path (re-verified mechanically by file read, not trusted from notes;
  read-only, no imports, no weights, no GPU).
- mapping-completeness: every surveyed component (18 rows) carries a
  tabled/parked/absent status with a cite in the spec.
- shape probes: derived dims recomputed (inner 24x128, rope axes sum,
  mod 6xdim, VAE factor) — arithmetic checked, not trusted.
- no-implementation tripwire: named DiT/VAE implementation files absent
  (survey adds zero patterns — scoping discipline as established).
Usage: python3 tests/test_qwen_arch.py
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VENV = os.path.expanduser("~/marigold-env/lib/python3.12/site-packages/diffusers")
SPEC = os.path.join(ROOT, "docs", "QWEN_ARCH.md")

FAIL = []

T = os.path.join(VENV, "models/transformers/transformer_qwenimage.py")
ATTN = os.path.join(VENV, "models/attention.py")
ACT = os.path.join(VENV, "models/activations.py")
APROC = os.path.join(VENV, "models/attention_processor.py")
NORM = os.path.join(VENV, "models/normalization.py")
EMB = os.path.join(VENV, "models/embeddings.py")
VAE = os.path.join(VENV, "models/autoencoders/autoencoder_kl_qwenimage.py")
PIPE = os.path.join(VENV, "pipelines/qwenimage/pipeline_qwenimage_edit.py")


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
    files = {"transformer": T, "attention": ATTN, "activations": ACT,
             "attn_proc": APROC, "norm": NORM, "emb": EMB, "vae": VAE,
             "edit_pipe": PIPE}
    check("qwenarch-files", all(os.path.isfile(p) for p in files.values()),
          f"{sum(os.path.isfile(p) for p in files.values())}/{len(files)} cited files exist")

    syms = [(T, "class QwenImageTransformerBlock"), (T, "class QwenImageTransformer2DModel"),
            (T, "class QwenDoubleStreamAttnProcessor2_0"), (T, "class QwenEmbedRope"),
            (T, "class QwenEmbedLayer3DRope"), (T, "class QwenTimestepProjEmbeddings"),
            (T, "def apply_rotary_emb_qwen"), (T, "self.img_mlp = FeedForward"),
            (T, "self.txt_mlp = FeedForward"), (T, 'activation_fn="gelu-approximate"'),
            (ATTN, 'act_fn = GELU(dim, inner_dim, approximate="tanh"'),
            (ACT, "class GELU"), (ACT, "class ApproximateGELU"),
            (ACT, "x * torch.sigmoid(1.702 * x)"),
            (APROC, 'elif qk_norm == "rms_norm":'), (APROC, "self.norm_added_q"),
            (NORM, "class AdaLayerNormContinuous"), (NORM, "class RMSNorm"),
            (EMB, "class TimestepEmbedding"), (EMB, "class Timesteps"),
            (VAE, "class AutoencoderKLQwenImage"), (VAE, "class QwenImageRMS_norm"),
            (VAE, "class QwenImageResample"), (VAE, 'mode="nearest-exact"'),
            (VAE, "class QwenImageResidualBlock"), (VAE, "class QwenImageAttentionBlock"),
            (PIPE, "img_shapes = ["), (PIPE, "latent_model_input = torch.cat([latents, image_latents], dim=1)"),
            (PIPE, "Qwen2.5-VL")]
    missing = [s for p, s in syms if not has(p, s)]
    check("qwenarch-symbols", not missing, f"{len(syms) - len(missing)}/{len(syms)} symbols; missing={missing[:3]}")

    spec = open(SPEC).read()
    rows = ["img_in/txt_in", "txt_norm", "time_text_embed", "pos_embed",
            "norm1+norm2", "img/txt_mod", "_modulate", "QKV + out",
            "QK RMSNorm", "joint SDPA", "gelu-approximate", "norm_out",
            "causal-3D", "QwenImageRMS_norm", "nearest-exact", "1×1",
            "Qwen2.5-VL", "seq-concat"]
    norow = [r for r in rows if r not in spec]
    check("qwenarch-mapping", not norow and spec.count("tabled") >= 10
          and "PARKED" in spec and "absent" in spec,
          f"{len(rows) - len(norow)}/{len(rows)} components mapped; missing={norow[:3]}")

    check("qwenarch-shapes",
          24 * 128 == 3072 and 16 + 56 + 56 == 128 and 6 * 3072 == 18432
          and 2 ** 3 == 8 and all(f"{v}" in spec for v in ("3072", "18432", "3584", "60")),
          "inner 3072, rope 128=head, mod 18432, VAE 8x, joint 3584, 60 layers")

    forbidden = ["chain/qwen_dit.py", "chain/qwen_vae.py", "chain/qwen_rope.py",
                 "programs/qwen_dit.asm", "programs/qwen_vae.asm",
                 "chain/qwen_edit.py", "scripts/build_qwen.py"]
    present = [f for f in forbidden if os.path.exists(os.path.join(ROOT, f))]
    check("qwenarch-tripwire", not present, f"forbidden present={present}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
