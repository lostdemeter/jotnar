"""Teacher anatomy gate (student Phase 7): banked LoRA-delta numbers are true.

SKIP-clean without the sibling checkout or checkpoints (gold stays
green machineless): fast path only -- safetensors header reads for
shapes/dtypes, factor-side delta recompute (B@A from the 1.85GB
trainables + tuned-minus-base over the 254MB VAE base; no 54GB
backbone load in routine runs), banked text-embedding integrity
(shapes/mask/sha), blueprint config legs (transformer/vae
config.json, no weight load). Backbone-relative norms were computed
once on the final tree, CPU-only, and banked in
samples/marigold/teacher_anatomy.json; the full recompute is a
documented manual step, stated as such:
  python3 /tmp/opencode/teacher_anatomy_compute.py
which streams the 40GB backbone read-only and rewrites
/tmp/opencode/anatomy_raw.json (never the banked file, never weights).
Usage: python3 tests/test_teacher_anatomy.py (~1 min with assets).
"""
import hashlib
import json
import os
import struct
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []

SIBLING = os.path.normpath(os.path.join(ROOT, "..", "marigold-v2"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def sha256(path, nbytes=0):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if nbytes and f.tell() >= nbytes:
                break
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    if not os.path.isdir(SIBLING):
        print("SKIP (no marigold-v2 sibling checkout)")
        return
    ck = os.path.join(SIBLING, "assets", "checkpoints")
    train = os.path.join(ck, "Marigold-V2", "depth", "Log-stage2",
                         "trainables.safetensors")
    trdir = os.path.join(ck, "Qwen-Image-Edit-2509", "transformer")
    vaeb = os.path.join(ck, "Qwen-Image-Edit-2509", "vae",
                        "diffusion_pytorch_model.safetensors")
    embp = os.path.join(ck, "Marigold-V2", "qwen_text_embeddings",
                        "qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt")
    mskp = embp.replace("_prompt_embeds.pt", "_prompt_mask.pt")
    for tag, p in (("anatomy-trainables-present", train),
                   ("anatomy-backbone-present", trdir),
                   ("anatomy-vaebase-present", vaeb),
                   ("anatomy-embeds-present", embp),
                   ("anatomy-mask-present", mskp)):
        check(tag, os.path.exists(p), p)
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    bank_path = os.path.join(FIXDIR, "teacher_anatomy.json")
    check("anatomy-banked-present", os.path.isfile(bank_path), bank_path)
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)
    bank = json.load(open(bank_path))
    check("anatomy-schema",
          all(k in bank for k in ("provenance", "lora", "census", "delta_norms",
                                  "pairs", "vae_pairs", "embeds", "blueprint",
                                  "frozen_map", "recompute")),
          f"{len(bank['pairs'])} pairs, {len(bank['vae_pairs'])} vae_pairs")

    # -- header integrity (no weight load) --------------------------------
    head = read_header(train)
    ht = {k: v for k, v in head.items() if k != "__metadata__"}
    check("anatomy-census", len(ht) == 1556, f"{len(ht)} tensors")
    exp = {}
    for pr in bank["pairs"]:
        base = pr["path"]  # full suffix after "Diffuser." (stem or transformer_blocks.N.*)
        exp[f"Diffuser.{base}.lora_A.default.weight"] = ("BF16", pr["ashape"])
        exp[f"Diffuser.{base}.lora_B.default.weight"] = ("BF16", pr["bshape"])
    for pr in bank["vae_pairs"]:
        exp[f"VAE.{pr['path']}"] = ("BF16", pr["shape"])
    # iREPA keys carry dots in the module name; match by suffix instead
    irepa_ok = True
    for it in bank["census"]["irepa"]:
        cand = [k for k in ht if k.endswith(it["path"].split("iREPAStudentProjector.")[1])]
        irepa_ok &= (len(cand) == 1 and ht[cand[0]]["dtype"] == "F32"
                     and ht[cand[0]]["shape"] == it["shape"])
    hdr_ok = (set(ht) == set(exp) | {k for k in ht if k.startswith("iREPAStudentProjector.")})
    hdr_ok &= all(ht[k]["dtype"] == dt and ht[k]["shape"] == sh
                  for k, (dt, sh) in exp.items())
    check("anatomy-header-match", bool(hdr_ok and irepa_ok),
          "1556 shapes/dtypes vs banked")

    # -- factor-side delta integrity (trainables + 254MB VAE base only) ---
    import torch
    from safetensors import safe_open
    tr = safe_open(train, framework="pt", device="cpu")
    worst = 0.0
    for pr in bank["pairs"]:
        base = pr["path"]  # full suffix after "Diffuser."
        A = tr.get_tensor(f"Diffuser.{base}.lora_A.default.weight").float()
        B = tr.get_tensor(f"Diffuser.{base}.lora_B.default.weight").float()
        got = float(((B @ A) * 1.0).norm())
        want = pr["delta_fro"]
        rel = abs(got - want) / max(want, 1e-300)
        worst = max(worst, rel)
    vae = safe_open(vaeb, framework="pt", device="cpu")
    for pr in bank["vae_pairs"]:
        T = tr.get_tensor("VAE." + pr["path"]).float()
        W = vae.get_tensor(pr["path"]).float()
        got = float((T - W).norm())
        rel = abs(got - pr["delta_fro"]) / max(pr["delta_fro"], 1e-300)
        worst = max(worst, rel)
    check("anatomy-delta-integrity", worst < 1e-6, f"worst rel err {worst:.2e}")

    # -- text-embedding integrity (19MB load, CPU) -------------------------
    e = torch.load(embp, map_location="cpu", weights_only=False)
    m = torch.load(mskp, map_location="cpu", weights_only=False)
    eb = bank["embeds"]
    check("anatomy-embeds",
          tuple(e.shape) == tuple(eb["embeds_shape"]) and str(e.dtype) == "torch.bfloat16"
          and tuple(m.shape) == tuple(eb["mask_shape"])
          and bool(torch.isfinite(e).all())
          and m.view(-1).sum().item() == m.numel()
          and sha256(embp) == eb["sha256"]["prompt_embeds.pt"]
          and sha256(mskp) == eb["sha256"]["prompt_mask.pt"],
          f"{tuple(e.shape)} {e.dtype}, mask all-true {m.numel()}/{m.numel()}")

    # -- blueprint legs (config.json only, no weight load) -----------------
    tcfg = json.load(open(os.path.join(trdir, "config.json")))
    bp = bank["blueprint"]
    check("anatomy-blueprint",
          tcfg["num_layers"] == bp["num_layers"] == 60
          and tcfg["num_attention_heads"] == bp["heads"] == 24
          and tcfg["attention_head_dim"] == bp["head_dim"] == 128
          and tcfg["joint_attention_dim"] == 3584
          and tcfg["patch_size"] == 2
          and tcfg["in_channels"] == 64 and tcfg["out_channels"] == 16,
          "60x3072/24h DiT + 3584 joint dim")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
