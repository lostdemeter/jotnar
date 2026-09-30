"""Range estimator (static verifier, scales half): interval analysis over listings.

Propagates [lo,hi] intervals through per-mnemonic transfer functions (all
conservative hulls, float, offline) and checks triple-stream intervals
against frozen scale coverage from M.json:
  floor(m) = PHI^((m-FRAC_CAP-BIAS)/512)  (below: bridged to zero)
  cap(m)   = PHI^((m-BIAS)/512)           (above: saturates)
Findings are WARNINGS (a small interval below floor is often fine -- flat
fields legitimately hold ~0; the discriminant case was a bug because the
quantity MATTERED downstream). Ranges come from RANGE declarations
(RANGE name lo hi, parsed like IN AS); undeclared streams are UNKNOWN
(gradual, like layouts). Per-op m override is NOT modeled (checks run
against m_cov; accumulation-scale subtleties stay with unit gates --
stated limit).
"""
import math
import os

import numpy as np

import sys
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
import phi_core.lattice as S


def coverage(m):
    """(floor, cap) of lattice scale m (see module docstring)."""
    K, BIAS, FRAC_CAP = S.K, S.BIAS, S.FRAC_CAP
    PHI = (1 + math.sqrt(5)) / 2
    return (PHI ** ((m - FRAC_CAP - BIAS) / K),
            PHI ** ((m - BIAS) / K))


def _rules():
    import json
    from chain.control import load_ctrl
    P = load_ctrl()
    beta_lv = [P.get("iso_atten", 0.5), P.get("mid_atten", 0.6),
               P.get("strong_atten", 1.0)]
    return {
        "SRGB_DECODE": lambda ins, cfg: (0.0, 1.0),
        "LUMA": lambda ins, cfg: (0.0, 1.0),
        "SQRT": lambda ins, cfg: (0.0, math.sqrt(max(ins[0][1], 0.0))),
        "SPLAT_BLUR": lambda ins, cfg: ins[0],
        "ISO_BLUR": lambda ins, cfg: ins[0],
        "GAUSS": lambda ins, cfg: ins[0],
        "SUB": lambda ins, cfg: (ins[0][0] - ins[1][1], ins[0][1] - ins[1][0]),
        "ADD": lambda ins, cfg: (ins[0][0] + ins[1][0], ins[0][1] + ins[1][1]),
        "MUL": lambda ins, cfg: _hull_mul(ins[0], ins[1]),
        "BETA": lambda ins, cfg: (float(cfg.get("beta", 0.5)),
                                  float(cfg.get("beta", 0.5))),
        "BETA_V5": lambda ins, cfg: (
            float(cfg.get("beta", 0.5)) * min(beta_lv),
            float(cfg.get("beta", 0.5)) * max(beta_lv)),
        "SQUARE": lambda ins, cfg: (0.0, 1.0),  # op clips to [0,1] always
        "GAIN": lambda ins, cfg: (0.0, 1.0),
        "SRGB_ENCODE": lambda ins, cfg: (0.0, 255.0),
        "WARP": lambda ins, cfg: ins[0],
        "STATIC": lambda ins, cfg: (0.0, 1.0),
        "MIXDYAD": lambda ins, cfg: ins[0],
        "SELECT": lambda ins, cfg: ins[1],
        "MATMUL": lambda ins, cfg: None,
        "BATCH_MATMUL": lambda ins, cfg: None,
        "SOFTMAX": lambda ins, cfg: (0.0, 1.0),
        "RMSNORM": lambda ins, cfg: ins[0],
        "SILU": lambda ins, cfg: ins[0],
        "ROTARY": lambda ins, cfg: ins[0],
        "TRANSPOSE": lambda ins, cfg: ins[0],
        "RESHAPE2": lambda ins, cfg: ins[0],
        "RESHAPE3": lambda ins, cfg: ins[0],
        "PERMUTE3": lambda ins, cfg: ins[0],
    }


def _hull_mul(a, b):
    ps = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(ps), max(ps))


def estimate(text, registry, sigs=None, m_cov=None):
    """Run interval analysis. Returns (findings, report): findings = list of
    warning strings (stream below floor or above cap at m_cov); report has
    per-stream intervals. Undeclared/unmodeled streams are UNKNOWN (silent).
    RANGE declarations seed IN streams (parse here; run() ignores them)."""
    from chain import asm as ASM
    from chain.holo_phi import _load_scales
    if m_cov is None:
        _, m_cov = _load_scales()
    floor, cap = coverage(m_cov)
    config, inp, bound, _ = ASM.assemble(text, registry, sigs)
    ranges = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.upper().startswith("RANGE"):
            parts = line.split()
            if len(parts) == 4:
                _, nm, lo, hi = parts
                ranges[nm] = (float(lo), float(hi))
    rules = _rules()
    # layouts for coverage scoping: coverage is a BRIDGE property of lattice
    # values -- U8/F/I streams never bridge, so m_cov cap/floor don't apply
    # (checking them produced false positives on SRGB bytes; caught, fixed).
    from chain import asm as ASM
    try:
        _, vrep = ASM.verify(text, registry, sigs)
        layouts = vrep["layouts"]
    except Exception:
        layouts = {}
    findings = []
    for outs, mn, fn, args, ln, sig in bound:
        vals = []
        skip = False
        for a in args:
            if a in ranges:
                vals.append(ranges[a])
            elif ASM._is_literal(a):
                f = float(a)
                vals.append((f, f))
            else:
                skip = True
                break
        if skip or mn not in rules:
            for o in outs:
                ranges[o] = ranges.get(o)
            continue
        try:
            r = rules[mn](vals, config)
        except Exception:
            r = None
        if r is None:
            for o in outs:
                ranges[o] = ranges.get(o)
            continue
        for o in outs:
            ranges[o] = r
            lay = layouts.get(o, "T:HW")
            if lay != "UNKNOWN" and not lay.startswith("T:"):
                continue  # coverage binds lattice values only (stated);
            # UNKNOWN layouts still check (conservative estimator) -- only
            # positively-non-triple layouts skip.
            lo, hi = r
            # ASYMMETRIC doctrine (stated): underflow flags only whole-range-
            # below (tiny values are often legitimate ~0, e.g. detail on
            # flats -- flagging those would false-positive constantly), but
            # saturation flags on ANY exceedance (values above cap clip:
            # real damage on part of the range, never legitimate).
            if hi < floor:
                findings.append(
                    f"line {ln} ({mn}): {o} range [{lo:.2e},{hi:.2e}] "
                    f"underflows m_cov floor {floor:.2e} (bridged to zero)")
            elif hi > cap:
                findings.append(
                    f"line {ln} ({mn}): {o} range [{lo:.2e},{hi:.2e}] "
                    f"exceeds m_cov cap {cap:.2e} (saturates)")
    return findings, {"floor": floor, "cap": cap, "m_cov": m_cov,
                      "ranges": {k: v for k, v in ranges.items() if v is not None}}
