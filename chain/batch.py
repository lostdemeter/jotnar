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
    # unsharp CONCAT (axis-1 glue, rows preserved -- structural
    # halo-0, measured on the workload listing: strip blocks exact)
    "CONCAT",
})

HALO_RADIUS = {
    # spatial ops: measured bleed radius in rows (SPLAT_BLUR: 1,
    # pinned by seam-blur-halo -- kernel change trips the detector;
    # SLICE: 1, measured on the unsharp 3x3-blur listing -- covers
    # offset<=1 neighborhoods; wider offsets need re-measurement and
    # the halo-width assertion in the workload gate is the detector)
    "SPLAT_BLUR": 1,
    "SLICE": 1,
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


def _rows_of(value):
    """Row count of a stackable value (arrays: axis 0; triples: axis 1
    of plane 0). Scalars and ragged-sharing inputs are not stackable."""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise BatchError(f"stackable tuple needs 3 planes, got {len(value)}")
        return np.shape(value[0])[0] if np.ndim(value[0]) >= 2 else None
    a = np.asanyarray(value)
    if a.ndim < 1:
        return None
    return a.shape[0]


def _same(a, b):
    a = np.stack(a) if isinstance(a, tuple) else np.asanyarray(a)
    b = np.stack(b) if isinstance(b, tuple) else np.asanyarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _stack_vals(vals):
    """Concatenate one stream across lanes (arrays axis 0, triples per
    plane axis 0). All-or-shared decided by the caller."""
    if isinstance(vals[0], (tuple, list)):
        return tuple(np.concatenate(
            [np.asanyarray(v[i]) for v in vals], axis=0) for i in range(3))
    return np.concatenate([np.asanyarray(v) for v in vals], axis=0)


def _split_val(value, sizes, radius):
    """Split a fused feed back into lanes.

    Seam geometry (measured, not assumed): lane blocks tile the fused
    stream with `extra = total - sum(sizes)` surplus rows shared over
    the N-1 seams (extra=0: same geometry, contamination in place;
    extra=2/seam: valid-trimmed streams like the unsharp blur, surplus
    rows ARE the seam artifacts). Lane i block starts at
    B_i (B_0=0, B_{i+1}=B_i+s_i+extra/(N-1)) and keeps the block minus
    t=max(0, radius-extra/2/(N-1)) rows per seam side. Anything else
    (negative/odd/uneven surplus) refuses loud -- unknown seam
    geometry, never guessed. Single lane (no seams) returns whole.
    sizes = per-lane row counts."""
    triple = isinstance(value, tuple)
    a = np.stack(value) if triple else np.asanyarray(value)
    ax = 1 if triple else 0
    total = a.shape[ax]
    n = len(sizes)
    if n == 1:
        return [tuple(p.copy() for p in a) if triple else a.copy()]
    extra = total - sum(sizes)
    if extra < 0 or extra % (n - 1) != 0:
        raise BatchError(
            f"split mismatch: {total} rows over lanes {sizes} "
            f"(surplus {extra} not shareable over {n - 1} seams)")
    e = extra // (n - 1)
    if e % 2 != 0:
        raise BatchError(
            f"split mismatch: odd seam surplus {e} (lanes {sizes}, "
            f"total {total})")
    t = max(0, radius - e // 2)
    out, off = [], 0
    for i, s in enumerate(sizes):
        lo = off + (t if i > 0 else 0)
        hi = off + s - (t if i < n - 1 else 0)
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(lo, hi)
        piece = a[tuple(sl)]
        out.append(tuple(p.copy() for p in piece) if triple
                   else piece.copy())
        off += s + e
    return out


def fuse(text, feeds_list, halo="refuse", sigs=None, trace=None,
         out_rows=None, plain=None):
    """Two-or-more inferences, one execution (hypotenuse, not width+height).

    Contract check up front via required_halo -- refuse loud before any
    execution. Inputs identical across ALL lanes pass through shared
    (readonly weights read once, no locks: the fused run is single
    execution); differing inputs stack along rows. Single run_text,
    outputs split per lane (guard band under halo="drop").
    Policy: "refuse" (default; spatial present -> loud naming the
    radius), "drop" (mechanical guard band), "exchange" (NOT built --
    needs a seam protocol; refused loud with that reason). Unlisted op
    -> BatchError as today. Shape errors name the lane and stream.
    out_rows maps output-stream name -> per-lane row count (int, or
    list per lane) for streams whose geometry differs from the inputs
    (e.g. valid-trimmed blur: H-2 per lane): those split to the given
    sizes (mismatch refuses loud -- caller-specified, must hold).
    plain names halo-0 streams split with trim 0 always; every other
    stream takes the listing radius as its trim (conservative default:
    over-trimming fails loud at byte-compare gates, under-trimming
    would risk silent contamination). Mixed listings must map every
    halo-0 stream class in plain.
    Non-IN feeds that tile neither way (offset-window intermediates)
    are shared WHOLE across lanes (true fused values, not
    lane-partitioned -- do not mutate; exactness is graded per
    requested output, never assumed for intermediates).
    Returns [per-lane output dicts] (feeds minus IN streams minus
    CONFIG); input row counts may differ per lane.
    """
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    if sigs is None:
        sigs = SIGS
    if len(feeds_list) < 1:
        raise BatchError("fuse needs at least one lane")
    if halo not in ("refuse", "drop", "exchange"):
        raise BatchError(f"unknown halo policy {halo!r}")
    need = required_halo(text)  # unlisted op refuses here, first
    if need > 0 and halo == "refuse":
        raise BatchError(
            f"halo policy is refuse and the listing needs radius {need} "
            f"(use halo=\"drop\", or await the seam protocol)")
    if halo == "exchange":
        raise BatchError(
            "halo=\"exchange\" needs a seam protocol (halo ferry between "
            "batch slots) -- not built; use halo=\"drop\" (guard band) or "
            "halo=\"refuse\"")
    radius = need if halo == "drop" else 0

    keys = list(feeds_list[0].keys())
    for i, feeds in enumerate(feeds_list):
        if list(feeds.keys()) != keys:
            raise BatchError(
                f"lane {i} streams {list(feeds.keys())} != lane 0 {keys}")
    config, inp, _, _ = ASM.assemble(text, REGISTRY, sigs)
    in_names = {n for n, _ in inp}
    payload, sizes = {}, None
    for k in keys:
        vals = [feeds[k] for feeds in feeds_list]
        if all(_same(vals[0], v) for v in vals[1:]):
            payload[k] = vals[0]  # shared readonly: read once, no locks
            continue
        rows = []
        for i, v in enumerate(vals):
            r = _rows_of(v)
            if r is None:
                raise BatchError(
                    f"lane {i} stream {k!r} is not row-stackable")
            rows.append(r)
        ref0 = (np.stack(vals[0]) if isinstance(vals[0], tuple)
                else np.asanyarray(vals[0]))
        ax0 = 1 if isinstance(vals[0], tuple) else 0
        for i, v in enumerate(vals):
            a = np.stack(v) if isinstance(v, tuple) else np.asanyarray(v)
            ax = 1 if isinstance(v, tuple) else 0
            if ax != ax0 or (a.shape[:ax] + a.shape[ax + 1:] !=
                             ref0.shape[:ax0] + ref0.shape[ax0 + 1:]):
                raise BatchError(
                    f"lane {i} stream {k!r} shape {a.shape} incompatible "
                    f"with lane 0 for row stacking")
        payload[k] = _stack_vals(vals)
        if sizes is None:
            sizes = rows
        elif sizes != rows:
            raise BatchError(
                f"lane row counts disagree across streams: {sizes} vs {rows} "
                f"(stream {k!r})")
    if sizes is None:
        sizes = [1] * len(feeds_list)  # all inputs shared: degenerate batch

    feeds = ASM.run_text(text, REGISTRY, payload, sigs=sigs, trace=trace)
    lanes = [{} for _ in feeds_list]
    orows = {}
    for k, n in (out_rows or {}).items():
        orows[k] = list(n) if isinstance(n, list) else [n] * len(feeds_list)
    for k, v in feeds.items():
        if k in in_names or k == "CONFIG":
            continue
        try:
            rk = 0 if k in (plain or {}) else radius
            parts = _split_val(v, orows.get(k, sizes), rk)
        except BatchError as e:
            if k in orows:
                raise BatchError(f"output {k!r}: {e}")
            # non-tileable intermediate (offset windows): share the true
            # fused values whole -- exactness graded per requested output
            parts = [v] * len(feeds_list)
        for lane, piece in zip(lanes, parts):
            lane[k] = piece
    return lanes
