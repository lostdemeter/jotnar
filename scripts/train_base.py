"""Shared training-harness base (student rounds): the loop shape all seven
hand-rolled trainers share, extracted once after the hyper-mscale profile
(docs/TRAINING.md §Profile — feature cache is the cut, everything else is
plumbing). Trainers keep their science (models, losses, eval bodies); the
base owns: deterministic seeding, device pick, stride sampling, LR
schedules, eval cadence, phase timers (every future run prints its own
forward/backward/eval/dataloader/aux shares), the feature cache, the
weights-out-of-tree discipline, and the metrics-banking schema (fail loud
on missing keys — the batteries read these JSONs). torch/numpy import
lazily inside functions (house style: heavy imports live in main, never
module top). No science here: no losses, no models, no metrics math.
"""
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTBASE = os.path.expanduser("~/student_runs")

# metrics-banking schema: every banked run JSON carries these (the
# batteries + harness readers key on tag/frames; weights must point
# out-of-tree).
REQUIRED_KEYS = ("tag", "torch", "device", "weights", "frames")

PHASES = ("dataloader", "forward", "backward", "eval", "aux")


def seed_all(seed):
    """Deterministic seeding, fixed order (torch first — matches every
    hand-rolled trainer). No RNG in-loop anywhere downstream."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def pick_device():
    """cuda iff available (all trainers' rule), with the device print."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    return device


def log_gpu_mem():
    """nvidia-smi one-liner (hyper's format); best-effort, never fails."""
    try:
        import subprocess as _sp
        nvidia = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30).stdout.strip()
    except Exception:
        nvidia = ""
    print(f"gpu: {nvidia}", flush=True)


def stride_index(n, stride=7, device="cpu"):
    """Deterministic stride sample (fixed order, no RNG in-loop)."""
    import torch
    return torch.arange(0, n, stride, device=device)


def apply_schedule(opt, kind, lr, it, steps, warmup=400):
    """LR plumbing covering all trainers' schedules (const keeps --lr;
    cosine anneals to 0; linear decays to 0; warmcos linear-warms then
    cosine-anneals). Returns the applied lr."""
    import math as _math
    if kind == "const":
        lr_t = lr
    elif kind == "cosine":
        lr_t = lr * 0.5 * (1.0 + _math.cos(_math.pi * it / steps))
    elif kind == "linear":
        lr_t = lr * max(0.0, 1.0 - it / steps)
    elif kind == "warmcos":
        if it < warmup:
            lr_t = lr * (it + 1) / warmup
        else:
            jj = (it - warmup) / max(1, steps - warmup)
            lr_t = lr * 0.5 * (1.0 + _math.cos(_math.pi * jj))
    else:
        raise ValueError(f"unknown schedule '{kind}'")
    for pg in opt.param_groups:
        pg["lr"] = lr_t
    return lr_t


def cadence(it, steps, every):
    """True on print/eval cadence steps and always on the last step."""
    return it % every == 0 or it == steps - 1


class PhaseTimer:
    """Named wall accumulators; every harness run prints its own phase
    shares (profile-first is a standing habit, not a one-off)."""

    def __init__(self):
        self.t = {p: 0.0 for p in PHASES}

    def add(self, name, seconds):
        assert name in self.t, f"unknown phase '{name}' (want {PHASES})"
        self.t[name] += seconds

    def time(self, name):
        """`with timer.time('forward'):` — accumulates the block's wall."""
        assert name in self.t, f"unknown phase '{name}' (want {PHASES})"
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                return self

            def __exit__(self, *a):
                timer.t[name] += time.time() - self.t0
                return False

        return _Ctx()

    def shares(self):
        tot = sum(self.t.values())
        return {k: (v, (v / tot if tot > 0 else 0.0))
                for k, v in self.t.items()}

    def summary(self):
        """Printed lines + returned dict (shares sum to 1 — gated)."""
        out = {}
        for k, (v, s) in self.shares().items():
            print(f"phase {k}: {v:.1f}s ({100 * s:.1f}%)", flush=True)
            out[k] = {"seconds": v, "share": s}
        return out


class FeatureCache:
    """Disk cache for deterministic featurization (the profiled cut:
    multiscale_features recomputed 1200x/train-run + 180x/eval at ~1s a
    call). Key = frame id + featurizer name/version/args fingerprint;
    versioned keys refuse stale entries (no silent reuse of yesterday's
    features under today's math). Same array bytes in or out — the
    hoist changes timing, never values."""

    def __init__(self, root, featurizer, version, args_fp=""):
        self.dir = os.path.join(os.path.expanduser(root), "featcache",
                                f"{featurizer}-v{version}"
                                + (f"-{args_fp}" if args_fp else ""))
        os.makedirs(self.dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path(self, frame_id):
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_"
                       for c in str(frame_id))
        return os.path.join(self.dir, safe + ".npy")

    def get(self, frame_id, compute):
        """Cached array (compute() runs only on miss). Returns array."""
        import numpy as np
        p = self.path(frame_id)
        if os.path.isfile(p):
            self.hits += 1
            return np.load(p)
        arr = np.ascontiguousarray(compute())
        np.save(p, arr)
        self.misses += 1
        return arr

    def stats(self):
        tot = self.hits + self.misses
        print(f"featcache {self.dir}: {self.hits} hits, {self.misses} "
              f"misses ({tot} lookups)", flush=True)
        return {"dir": self.dir, "hits": self.hits, "misses": self.misses}


def save_weights(state_dict, tag, outbase=OUTBASE, fname="weights.pt"):
    """Weights out-of-tree discipline (never in-repo). Returns path."""
    import torch
    outdir = os.path.join(os.path.expanduser(outbase), tag)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    torch.save(state_dict, path)
    return path


def bank_metrics(rep, path):
    """Metrics-banking schema (fail loud on missing keys) + write."""
    missing = [k for k in REQUIRED_KEYS if k not in rep]
    if missing:
        raise ValueError(f"metrics missing schema keys {missing} "
                         f"(want {REQUIRED_KEYS})")
    if not rep["weights"].startswith(os.path.expanduser("~/")):
        raise ValueError(f"weights path breaks out-of-tree discipline: "
                         f"{rep['weights']}")
    json.dump(rep, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)
    return path
