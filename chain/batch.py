"""Batching contract (diagonals demo, scoped): which ops fuse, with what halo.

Two classes, nothing else:
- halo-0: pointwise ops batch freely (Phase-1 shape: taller matrices,
  same listing, split by halves). Membership is MEASURED, not assumed:
  every op here passed a stacked-shape trial (tests/test_diagonal.py
  for the chain ops, tests/test_diagonal2.py for the flagship prefix).
- halo-radius: spatial ops declare their bleed radius
  (SPLAT_BLUR: 1, per the pinned seam detector). The mechanical rule
  drops `radius` rows per seam side automatically (guard_band) --
  callers never touch row indices.

Anything in neither class REFUSES LOUD naming the op (house doctrine:
missing = NoPattern-style refusal, never silent). Adding a spatial op
means adding its measured row here; claiming anything other than the
tabled radius (0 for the blur, or a widened 2) fails loud via
assert_halo. No emitter changes, no new opcodes, no protocol changes.
"""
import numpy as np

HALO_ZERO = frozenset({
    # Phase-1 chain (tests/test_diagonal.py, 12/12 split-exact)
    "MATMUL", "ADD", "SILU",
    # flagship pointwise prefix + tail (tests/test_diagonal2.py:
    # rgb/LIN/Y/A fully exact; tail exact outside the halo)
    "SRGB_DECODE", "LUMA", "SQRT", "SUB", "BETA_V5", "MUL",
    "SQUARE", "GAIN", "SRGB_ENCODE",
})

HALO_RADIUS = {
    # spatial ops: measured bleed radius in rows (SPLAT_BLUR: 1,
    # pinned by seam-blur-halo -- kernel change trips the detector)
    "SPLAT_BLUR": 1,
}


class BatchError(Exception):
    """Batching contract refusal (unlisted op, or radius lie)."""


def ops_in(text):
    """Mnemonics used by a listing (assembler only, never executes)."""
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    bound = ASM.assemble(text, REGISTRY, SIGS)[2]
    return [mn for _, mn, _, _, _, _ in bound]


def required_halo(text):
    """Max halo radius over the listing's ops. Raises BatchError naming
    any op in neither class (unmeasured = refused, never defaulted)."""
    need = 0
    for mn in ops_in(text):
        if mn in HALO_RADIUS:
            need = max(need, HALO_RADIUS[mn])
        elif mn not in HALO_ZERO:
            raise BatchError(
                f"no batching class for {mn} (measure it, then table it)")
    return need


def assert_halo(text, claimed):
    """The table is the authority: anything but the required radius --
    0 for the blur, a widened 2, any mismatch -- fails loud."""
    need = required_halo(text)
    if claimed != need:
        raise BatchError(
            f"claimed halo {claimed} != required {need} "
            f"(SPLAT_BLUR tables at {HALO_RADIUS.get('SPLAT_BLUR')})")
    return need


def guard_band(value, H, radius):
    """Drop `radius` rows per seam side automatically. value is an array
    (rows on axis 0) or a stacked triple (3 planes: rows on axis 1).
    Returns (top, bot) with all row math inside -- callers pass shapes
    (H), never indices."""
    triple = isinstance(value, tuple)
    a = np.stack(value) if triple else np.asarray(value)
    if triple:  # (planes, rows, ...): seam rows on axis 1
        top = a[:, 0:H - radius] if radius else a[:, 0:H]
        bot = a[:, H + radius:2 * H] if radius else a[:, H:2 * H]
    else:
        top = a[0:H - radius] if radius else a[0:H]
        bot = a[H + radius:2 * H] if radius else a[H:2 * H]
    return top, bot
