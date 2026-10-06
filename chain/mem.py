"""Physical-memory budget for the 7B geometric path (stdlib only).

Why this exists: the first 28-layer Qwen2-7B attempt held every weight
as float64 in one process (sample + safetensor cache + HF model at
once, ~76GB+ on a 61GiB box) and thrashed swap until the SSD died.
This module prices the footprint BEFORE anything loads, and refuses
loudly when the plan exceeds physical RAM -- never swap.

Deliberately stdlib-only: the guard must run even when the ML stack
is absent. Geometry constants mirror chain/qwen7b.py (owner); the
math here is pure ints so it stays importable anywhere.
"""
import os

# Canonical Qwen2-7B-Instruct geometry (owner: chain/qwen7b.py:13).
HID, INTER, N_LAYERS, NH, NKV, DH = 3584, 18944, 28, 28, 4, 128
KV_DIM = NKV * DH  # 512
VOCAB = 152064  # Qwen2 vocab; asserted at runtime when weights load


class BudgetExceeded(RuntimeError):
    """Planned bytes exceed the physical-RAM cap (fail loud, not swap)."""


def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 2 and p[0].endswith(":"):
                    out[p[0][:-1]] = int(p[1]) * 1024  # kB -> B
    except OSError:
        pass
    return out


def phys_bytes():
    """Total physical RAM, or 0 when unknowable (non-Linux)."""
    return _meminfo().get("MemTotal", 0)


def avail_bytes():
    """Currently available RAM, or 0 when unknowable."""
    return _meminfo().get("MemAvailable", 0)


def default_cap():
    """Refusal threshold: 60% of physical RAM.

    Headroom is deliberate: the OS, GPU driver pins, and compile
    scratch all live outside our estimate. Override with
    JOTNAR_MEM_CAP_GB or an explicit cap_bytes.
    """
    env = os.environ.get("JOTNAR_MEM_CAP_GB")
    if env:
        return int(float(env) * 2 ** 30)
    phys = phys_bytes()
    if phys <= 0:
        raise BudgetExceeded("cannot read /proc/meminfo: pass cap_bytes "
                             "explicitly (refusing to guess on swap-backed "
                             "machines)")
    return int(phys * 0.6)


def param_counts(smax):
    """Exact parameter counts for the 28-layer listing (no weights read)."""
    per = {"wq": HID * HID, "wk": HID * KV_DIM, "wv": HID * KV_DIM,
           "wo": HID * HID, "wup": INTER * HID, "wgate": INTER * HID,
           "wdown": HID * INTER, "ln1": HID, "ln2": HID,
           "bq": smax * HID, "bk": smax * KV_DIM, "bv": smax * KV_DIM}
    layer = sum(v for k, v in per.items() if not k.startswith("b"))
    # Bias planes are tiled host-side to (S, dim): count stored form.
    layer_stored = sum(per.values())
    total = N_LAYERS * layer_stored + VOCAB * HID + VOCAB * HID + HID
    return {"per_layer": per, "layer_sum": layer,
            "layer_stored": layer_stored, "total": total}


def estimate_7b(smax):
    """Byte estimates for both paths (pure math, no loading).

    old_cpu: the retired bulk-float64 sample dict (sample + embed copy).
    cache: safetensor torch-file cache retained by chain/qwen7b.load7b.
    stream_peak: new path -- one fp64 weight transient + one fp16 copy
      + fp16 embed resident for decode steps.
    vram_fp16: managed-memory weights on the GPU (fp16 storage).
    """
    pc = param_counts(smax)
    total = pc["total"]
    embed = VOCAB * HID
    largest = max(HID * INTER, INTER * HID)  # up/gate/down, 67.9M
    return {
        "params": total,
        "old_cpu_f64": total * 8 + embed * 8,  # sample + separate E copy
        "safetensor_cache_bf16": total * 2,
        "stream_peak_cpu": largest * 8 + largest * 2 + embed * 2,
        "vram_fp16": total * 2,
        "smax": smax,
    }


def guard(need_bytes, label, cap_bytes=None):
    """Raise BudgetExceeded when need_bytes exceeds the cap. Returns need."""
    cap = cap_bytes if cap_bytes is not None else default_cap()
    if need_bytes > cap:
        raise BudgetExceeded(
            f"{label}: needs {need_bytes / 2 ** 30:.1f}GiB > cap "
            f"{cap / 2 ** 30:.1f}GiB physical-RAM budget "
            f"(phys {phys_bytes() / 2 ** 30:.1f}GiB, "
            f"avail {avail_bytes() / 2 ** 30:.1f}GiB). "
            f"Refusing to swap -- split the job or stream it.")
    return need_bytes


def report(smax=16, cap_bytes=None):
    """One-block budget print for script headers (pure math)."""
    e = estimate_7b(smax)
    try:
        cap = cap_bytes if cap_bytes is not None else default_cap()
        cap_s = f"{cap / 2 ** 30:.1f}GiB"
    except BudgetExceeded:
        cap_s = "unknown (pass cap)"
    L = [f"[membudget] Qwen2-7B smax={smax} params={e['params']} "
         f"({e['params'] * 2 / 2 ** 30:.1f}GiB fp16 / "
         f"{e['old_cpu_f64'] / 2 ** 30:.1f}GiB old-f64 path)"]
    L.append(f"[membudget] phys={phys_bytes() / 2 ** 30:.1f}GiB "
             f"avail={avail_bytes() / 2 ** 30:.1f}GiB cap={cap_s}")
    L.append(f"[membudget] old bulk-f64 CPU ~{e['old_cpu_f64'] / 2 ** 30:.1f}GiB "
             f"+ cache ~{e['safetensor_cache_bf16'] / 2 ** 30:.1f}GiB "
             f"(RETIRED path -- guard refuses this)")
    L.append(f"[membudget] streaming peak CPU ~{e['stream_peak_cpu'] / 2 ** 30:.1f}GiB, "
             f"VRAM fp16 ~{e['vram_fp16'] / 2 ** 30:.1f}GiB")
    return "\n".join(L)
