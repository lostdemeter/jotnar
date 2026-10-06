"""Re-download Qwen2-7B-Instruct at the pinned revision (post-SSD recovery).

Revision f2826a00... reproduces the exact snapshot dir chain/qwen7b.py
points at (SNAP), so no code changes are needed. Streaming download,
low RAM; ~15GB disk. Run: python3 scripts/download_qwen7b.py
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

REPO = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"


def main():
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    from chain.qwen7b import SNAP
    print(f"target: {SNAP}", flush=True)
    d = snapshot_download(repo_id=REPO, revision=REVISION,
                          allow_patterns=["*.json", "*.safetensors",
                                          "tokenizer*", "config.json",
                                          "generation_config.json"])
    print(f"downloaded: {d}", flush=True)
    idx = os.path.join(SNAP, "model.safetensors.index.json")
    if os.path.normpath(d) != os.path.normpath(SNAP):
        print(f"NOTE: hub returned {d}, SNAP wants {SNAP}", flush=True)
    assert os.path.isfile(idx), f"missing index: {idx}"
    tot = 0
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            tot += os.path.getsize(fp)
            print(f"  {f} {os.path.getsize(fp) / 2 ** 30:.2f}GiB", flush=True)
    print(f"total {tot / 2 ** 30:.2f}GiB", flush=True)
    assert tot > 10 * 2 ** 30, "snapshot suspiciously small"


if __name__ == "__main__":
    main()
