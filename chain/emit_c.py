"""Pattern-table codegen core + C backend (v0.2).

Shader model: one bound program (chain.asm.assemble IR) -> any backend.
A backend is an opcode->pattern table plus prologue/epilogue finishing
touches and a value model (SIGS layouts -> target types). Adding an
opcode = one registry row + one shape rule + one pattern per backend;
anything missing fails loud (NoPattern) -- never silently wrong.

Agreement classes (stated, per-path):
  - int paths (I/T streams, GATHER/ARGMAX/SELECT-masks): BIT-EXACT.
  - float paths (F streams, decoded weights): EPSILON. The lattice
    computes in fixed-point (m_acc/m_cov); the C backend computes the
    same real-valued functions in float64. The gate is CALIBRATED
    (first measurement x safety, regression guard) -- not preregistered,
    labeled honestly in the test.
Value model (C): I -> int64_t; T triples -> parallel int arrays;
F -> double. Data-prep (Python side) decodes T inputs to F64 doubles;
the emitted binary links libc+libm only.

v0.2 patterns: GATHER (T->T exact + F->F rows), GATHER3 (T/F
two-index rows), ARGMAX (T rows + F), TOPK (F scores -> F values + I
indices, ties-first), IADD/ISUB/IMUL/IMAD (I exact, 64-bit wrap),
RMSNORM, MATMUL, BATCH_MATMUL (2D/3D), SLICE (2D ax0/1), ROTARY,
TRANSPOSE (2D), TBETA/BETA, MUL/ADD (same-shape F), SELECT (F),
TSHIFT, SOFTMAX_WIDE, CONCAT (2D). Speed hooks (backend owns them):
ikj matmul + restrict + calloc-zeroed outputs, zero-copy gathers,
baked frozen dims, -O2 -march=native default; arena/BLAS are listed
finishing touches, not this cut.

Resident C mode: compile_program(..., target="c", live=[...]) emits
the same READY/step loop as CUDA (frozen inputs load once, live inputs
re-read in place, fixed geometry). pack_outputs=True writes all output
streams into one file per step; the host slices by output order.

Codec mode: compile_program(..., target="c", mode="codec", live=[...])
emits the small create/bind/step/destroy ABI for fully static F/I
programs.  Frozen inputs are read-only bindings, live inputs are
read-write bindings, outputs are write-only bindings, and intermediates
are scratch allocated once at create.  This is the boring in-process
shape a host codec can drive; build it with build_shared.
"""
import math
import os
import re
import subprocess

import numpy as np

from chain.asm import AsmError, assemble


class NoPattern(AsmError):
    """Opcode has no pattern for this backend (missing, not wrong)."""


def _short_site(ln):
    """Condense an origin string for emitted comments: basename the file
    part, keep :line and via-suffixes. Errors keep full sites."""
    import os as _os
    parts = str(ln).split(" via ")
    head = parts[0]
    if ":" in head:
        f, rest = head.rsplit(":", 1)
        head = f"{_os.path.basename(f) or f}:{rest}"
    return " via ".join([head] + parts[1:])


class Backend:
    """Target interface. Subclass per backend; the driver is shared."""

    name = "base"
    emits = False  # set True when pattern() covers a compilable set
    f_repr = "double"  # backend F storage spelling (non-FPU uses raw uint64)

    def value_repr(self, layout):
        raise NotImplementedError

    def prologue(self, ctx):
        raise NotImplementedError

    def pattern(self, mn, outs, args, sig, ctx):
        raise NoPattern(f"{self.name}: no pattern for {mn}")

    def epilogue(self, ctx):
        raise NotImplementedError


_C_INT = {
    "int8": "int8_t", "int16": "int16_t", "int32": "int32_t",
    "int64": "int64_t", "uint8": "uint8_t", "uint16": "uint16_t",
    "uint32": "uint32_t", "uint64": "uint64_t",
}


def _c_type(dt):
    dt = np.dtype(dt).name
    if dt in _C_INT:
        return _C_INT[dt]
    if dt == "float32":
        return "float"
    if dt == "float64":
        return "double"
    raise AsmError(f"C backend: no C type for dtype {dt}")


class FRef:
    """Zero-RAM float stream reference (shape stub for codegen).

    Carries (shape, dtype) with no storage so compile_program can infer
    shapes for multi-GB weights without loading them. Values must never
    be read through a stub: every value-reading site fails loud
    (find_consts skips, rope-pos refuses). Small streams (pos/cmask/x0)
    stay real arrays; stubs are for the large weights only.
    """
    def __init__(self, shape, dtype="float16"):
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)

    def __repr__(self):
        return f"FRef({self.shape}, {self.dtype.name})"


class IRef:
    """Zero-RAM integer stream reference (shape stub, same contract)."""
    def __init__(self, shape, dtype="int64"):
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.integer):
            raise AsmError(f"IRef needs integer dtype, got {self.dtype}")

    def __repr__(self):
        return f"IRef({self.shape}, {self.dtype.name})"


class TRef:
    """Zero-RAM triple stream reference (shape stub for large banks).

    Carries the three plane dtypes + shape, no storage. Codegen only
    needs shapes; values are still supplied at run time as s/e/z files.
    C/non-FPU compile paths accept it; CUDA rightly refuses T inputs at
    data-prep. This is the shape-only declaration for multi-GB trigram
    banks: TRef((V1,V2,C)) without allocating V1*V2*C triples.
    """
    def __init__(self, shape, dtypes=("int8", "int32", "uint8")):
        self.shape = tuple(int(d) for d in shape)
        if len(dtypes) != 3:
            raise AsmError(f"TRef needs three plane dtypes, got {dtypes}")
        self.dtypes = tuple(np.dtype(d) for d in dtypes)

    def __repr__(self):
        return f"TRef({self.shape}, {tuple(d.name for d in self.dtypes)})"


def _kind_of_sample(v):
    if isinstance(v, FRef):
        return "F"
    if isinstance(v, IRef):
        return "I"
    if isinstance(v, TRef):
        return "T"
    if isinstance(v, tuple) and len(v) == 3:
        return "T"
    v = np.asanyarray(v)
    return "I" if np.issubdtype(v.dtype, np.integer) else "F"


def _int_lit(a, what):
    try:
        f = float(a)
    except (TypeError, ValueError):
        raise AsmError(f"{what}: need integral literal, got {a!r}")
    if not f.is_integer():
        raise AsmError(f"{what}: need integral literal, got {a!r}")
    return int(f)


def _c_i64_lit(v):
    """C99 int64 literal, including the one value whose decimal spelling
    would otherwise be parsed as an unsigned constant."""
    v = int(v)
    if v == -(1 << 63):
        return "(-9223372036854775807LL - 1LL)"
    return f"{v}LL"


def _dyadic(f):
    """Dyadic scale check via frexp (exact, mirrors asm _is_dyadic)."""
    import math
    f = float(f)
    if f <= 0:
        return False
    return math.frexp(f)[0] == 0.5


def infer_shapes(bound, sample):
    """Static shape/dtype pass over bound IR using sample payload.

    streams[name] = (kind, sub, shape): kind I -> sub=C type; T -> sub=
    (s,e,z C types); F -> sub="double". Shape dims are ints (frozen) or
    None (batch-dynamic, from a GATHER ids stream). Rules exist only for
    v0.2 opcodes; anything else -> NoPattern.
    """
    _, inp, prog = bound
    in_names = [n for n, _ in inp]
    streams = {}
    for n in in_names:
        if n not in sample:
            raise AsmError(f"C backend: sample missing IN stream '{n}'")
        k = _kind_of_sample(sample[n])
        if k == "T":
            if isinstance(sample[n], TRef):
                r = sample[n]
                streams[n] = ("T", tuple(_c_type(d) for d in r.dtypes),
                              tuple(r.shape))
                continue
            s, e, z = sample[n]
            streams[n] = ("T", (_c_type(s.dtype), _c_type(e.dtype),
                                _c_type(z.dtype)),
                          tuple(np.shape(s)))
        elif k == "I":
            if isinstance(sample[n], IRef):
                r = sample[n]
                streams[n] = ("I", _c_type(r.dtype), tuple(r.shape))
                continue
            a = np.ascontiguousarray(sample[n])
            streams[n] = ("I", _c_type(a.dtype), tuple(a.shape))
        elif k == "F":
            if isinstance(sample[n], FRef):
                streams[n] = ("F", "double", tuple(sample[n].shape))
                continue
            a = np.ascontiguousarray(sample[n], dtype=np.float64)
            streams[n] = ("F", "double", tuple(a.shape))

    def S(a):
        return streams.get(a)

    def dyn1(ish):
        if len(ish) == 1:
            return None
        if len(ish) == 0:
            return 1
        return "BAD"

    for outs, mn, _fn, args, ln, _sig in prog:
        o = outs[0] if outs else None
        where = f"line {ln} ({mn})"
        if mn == "GATHER":
            tk, tsub, tsh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            # Static 1D ids (arange files, baked length) give static rows:
            # dynamic-None rows sized downstream by the global ids_n var
            # (first 1D-I input) OVERFLOW when a second 1D ids stream is
            # wider (allrows S=16 vs pos N=1: 16x heap overwrite into
            # neighbors with input-dependent payload). Truly dynamic ids
            # keep the None path (unchanged behavior).
            if len(ish) == 1 and ish[0] is not None:
                n = ish[0]
            else:
                n = dyn1(ish)
            if ik != "I" or len(tsh) != 2:
                raise AsmError(f"{where}: v0.2 needs table (V,C) + int ids")
            if len(ish) == 2:
                # Batched row-gather: static-2D ids (R,k) -> (R,k,C).
                # Same bytes as R*k flat fibers; the 2D shape keeps
                # downstream per-head BMMV strides expressible (TOPK
                # idx is always static; dynamic batch dims refused).
                if any(d is None for d in ish):
                    raise AsmError(f"{where}: GATHER 2D ids must be "
                                   f"static, got {ish}")
                if tk == "T":
                    streams[o] = ("T", tsub, (ish[0], ish[1], tsh[1]))
                elif tk == "F":
                    streams[o] = ("F", "double",
                                  (ish[0], ish[1], tsh[1]))
                else:
                    raise AsmError(f"{where}: GATHER table must be T/F")
            else:
                if n == "BAD":
                    raise AsmError(f"{where}: v0.2 needs (N,)/scalar ids")
                if tk == "T":
                    streams[o] = ("T", tsub, (n, tsh[1]))
                elif tk == "F":
                    streams[o] = ("F", "double", (n, tsh[1]))
                else:
                    raise AsmError(f"{where}: GATHER table must be T/F")
        elif mn == "GATHER3":
            tk, tsub, tsh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            jk, _, jsh = streams[args[2]]
            if len(tsh) != 3 or any(d is None for d in tsh):
                raise AsmError(f"{where}: GATHER3 v0.1 needs baked table "
                               f"(V1,V2,C), got {tsh}")
            if tk not in ("T", "F"):
                raise AsmError(f"{where}: GATHER3 table must be T/F")
            if ik != "I" or jk != "I" or len(ish) != 1 or len(jsh) != 1:
                raise AsmError(f"{where}: GATHER3 needs 1D I i/j")
            if ish[0] is None or jsh[0] is None:
                raise AsmError(f"{where}: GATHER3 v0.1 needs baked i/j "
                               f"length, got {ish}/{jsh}")
            if ish[0] != jsh[0]:
                raise AsmError(f"{where}: GATHER3 i/j length mismatch "
                               f"{ish[0]} vs {jsh[0]}")
            if tk == "T":
                streams[o] = ("T", tsub, (ish[0], tsh[2]))
            else:
                streams[o] = ("F", "double", (ish[0], tsh[2]))
        elif mn == "SCATTER":
            bk, _, bsh = streams[args[0]]
            rk, _, rsh = streams[args[1]]
            ik, _, ish = streams[args[2]]
            if bk != "F" or rk != "F" or ik != "I":
                raise AsmError(f"{where}: SCATTER needs F base + F row "
                               f"+ I idx")
            if len(bsh) != 2 or tuple(rsh) != (1, bsh[1]):
                raise AsmError(f"{where}: SCATTER base {bsh} + row {rsh} "
                               f"mismatch (row must be (1,C))")
            if len(ish) > 1 or (len(ish) == 1 and ish[0] not in (None, 1)):
                raise AsmError(f"{where}: SCATTER idx must be scalar/1-elem")
            streams[o] = ("F", "double", (bsh[0], bsh[1]))
        elif mn in ("IADD", "ISUB", "IMUL", "IMAD", "IDIV", "IMOD"):
            ak, _, ash = streams[args[0]]
            if ak != "I":
                raise AsmError(f"{where}: {mn} needs I-kind first operand")
            for _i, _a in enumerate(args[1:], 1):
                if _a in streams:
                    _k, _, _sh = streams[_a]
                    if _k != "I" or tuple(_sh) != tuple(ash):
                        raise AsmError(
                            f"{where}: {mn} operand {_i} must be I with "
                            f"shape {ash}, got {_k} {_sh}")
                else:
                    _lit = _int_lit(_a, f"{where} {mn} operand {_i}")
                    if not (-(1 << 63) <= _lit < (1 << 63)):
                        raise AsmError(f"{where}: {mn} literal outside "
                                       f"int64: {_lit}")
            streams[o] = ("I", "int64_t", ash)
        elif mn in ("IXOR", "IAND", "IOR"):
            ak, _, ash = streams[args[0]]
            if ak != "I":
                raise AsmError(f"{where}: {mn} needs I-kind first operand")
            for _i, _a in enumerate(args[1:], 1):
                if _a in streams:
                    _k, _, _sh = streams[_a]
                    if _k != "I" or tuple(_sh) != tuple(ash):
                        raise AsmError(
                            f"{where}: {mn} operand {_i} must be I with "
                            f"shape {ash}, got {_k} {_sh}")
                else:
                    _lit = _int_lit(_a, f"{where} {mn} operand {_i}")
                    if not (-(1 << 63) <= _lit < (1 << 63)):
                        raise AsmError(f"{where}: {mn} literal outside "
                                       f"int64: {_lit}")
            streams[o] = ("I", "int64_t", ash)
        elif mn in ("ISHL", "ISHR"):
            ak, _, ash = streams[args[0]]
            if ak != "I":
                raise AsmError(f"{where}: {mn} needs I-kind first operand")
            s = _int_lit(args[1], f"{where} {mn} shift")
            if not (0 <= s <= 63):
                raise AsmError(f"{where}: {mn} shift must be 0..63, got {s}")
            streams[o] = ("I", "int64_t", ash)
        elif mn == "F2I":
            sk, _, ssh = streams[args[0]]
            if sk != "F":
                raise AsmError(f"{where}: F2I needs F input")
            streams[o] = ("I", "int64_t", ssh)
        elif mn == "I2F":
            sk, _, ssh = streams[args[0]]
            if sk != "I":
                raise AsmError(f"{where}: I2F needs I input")
            streams[o] = ("F", "double", ssh)
        elif mn == "ARGMAX":
            sk, _, ssh = streams[args[0]]
            ax = _int_lit(args[1], f"{where} axis") if len(args) > 1 \
                and args[1] not in streams else -1
            if sk not in ("T", "F", "I"):
                raise AsmError(f"{where}: ARGMAX needs T/F/I input")
            if len(ssh) == 1:
                out_sh = ()
            elif len(ssh) == 2 and ax % 2 == 1:
                out_sh = (ssh[0],)
            else:
                raise AsmError(f"{where}: v0.2 ARGMAX handles 1D or "
                               f"2D-axis-1, got {ssh} ax {ax}")
            streams[o] = ("I", "int64_t", out_sh)
        elif mn == "TOPK":
            xk, _, xsh = streams[args[0]]
            k = _int_lit(args[1], f"{where} k")
            if xk != "F" or len(xsh) != 2:
                raise AsmError(f"{where}: TOPK needs 2D F scores")
            if xsh[1] is None or not (1 <= k <= xsh[1]):
                raise AsmError(f"{where}: TOPK k={k} out of "
                               f"(1,S={xsh[1]})")
            streams[outs[0]] = ("F", "double", (xsh[0], k))
            streams[outs[1]] = ("I", "int64_t", (xsh[0], k))
        elif mn == "RMSNORM":
            xk, xsub, xsh = streams[args[0]]
            wk, _, wsh = streams[args[1]]
            if xk not in ("F", "T") or xk != wk or len(xsh) < 1 \
                    or tuple(wsh) != (xsh[-1],):
                raise NoPattern(f"{where}: RMSNORM needs same-kind "
                                "x(...,D)+w(D,)")
            streams[o] = (xk, xsub, xsh)
        elif mn == "LAYERNORM":
            xk, xsub, xsh = streams[args[0]]
            wk, _, wsh = streams[args[1]]
            bk, _, bsh = streams[args[2]]
            if xk not in ("F", "T") or xk != wk or xk != bk or len(xsh) < 1 \
                    or tuple(wsh) != (xsh[-1],) or tuple(bsh) != (xsh[-1],):
                raise NoPattern(f"{where}: LAYERNORM needs same-kind "
                                "x(...,D)+w(D,)+b(D,)")
            streams[o] = (xk, xsub, xsh)
        elif mn == "INTERP":
            # v0.1: 2D (H,W) or 3D (H,W,C), dyadic scales only (house
            # doctrine: non-dyadic is a LOWERING ERROR, never approx).
            xk, xsub, xsh = streams[args[0]]
            try:
                sy, sx = float(args[1]), float(args[2])
            except (TypeError, ValueError):
                raise NoPattern(f"{where}: INTERP needs literal scales "
                                "(v0.1)")
            if xk != "F" or len(xsh) not in (2, 3):
                raise NoPattern(f"{where}: INTERP needs F x(H,W[,C])")
            if not (_dyadic(sy) and _dyadic(sx)):
                raise AsmError(f"{where}: INTERP non-dyadic scale "
                               f"({sy},{sx}) is a LOWERING ERROR")
            H, W = xsh[0], xsh[1]
            if H is None or W is None:
                raise AsmError(f"{where}: INTERP needs static HW (v0.1)")
            Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)
            streams[o] = (xk, xsub, (Ho, Wo) + tuple(xsh[2:]))
        elif mn == "CONV":
            # v0.1: single-channel 2D, same-size output, stride 1,
            # dilation 1, edge-replicate boundary (audited). No bias.
            xk, xsub, xsh = streams[args[0]]
            kk, _, ksh = streams[args[1]]
            if xk != "F" or kk != "F" or len(xsh) != 2 or len(ksh) != 2:
                raise NoPattern(f"{where}: CONV needs F x(H,W)+K(kH,kW) "
                                "(v0.1)")
            H, W = xsh
            kH, kW = ksh
            if H is None or W is None or kH is None or kW is None:
                raise AsmError(f"{where}: CONV needs static shapes "
                               f"(v0.1), got x({H},{W}) k({kH},{kW})")
            if kH % 2 == 0 or kW % 2 == 0 or kH > H or kW > W:
                raise AsmError(f"{where}: CONV needs odd kernel within "
                               f"image, got x({H},{W}) k({kH},{kW})")
            streams[o] = (xk, xsub, (H, W))
        elif mn in ("MATMUL", "BATCH_MATMUL"):
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak not in ("F", "T") or ak != bk:
                raise NoPattern(f"{where}: {mn} needs same-kind inputs")
            tsub = asub if ak == "T" else "double"
            if len(ash) == 2 and len(bsh) == 2 and ash[1] == bsh[0]:
                streams[o] = (ak, tsub, (ash[0], bsh[1]))
            elif len(ash) == 3 and len(bsh) == 3 \
                    and (ash[0] == bsh[0] or bsh[0] == 1) \
                    and ash[2] == bsh[1]:
                streams[o] = (ak, tsub, (ash[0], ash[1], bsh[2]))
            else:
                raise AsmError(f"{where}: {mn} shape mismatch {ash}/{bsh}")
        elif mn == "BMMV":
            ak, _, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak != "F" or bk != "F" or len(ash) < 2 or len(bsh) < 2:
                raise NoPattern(f"{where}: BMMV needs F bases rank>=2 "
                                f"(v0.1)")
            (BB, M, N_, K, LAA, SA, LAB, SB, _LAC, _SC, TB) = tuple(
                _int_lit(args[i], f"{where} BMMV") for i in range(2, 13))
            if TB not in (0, 1):
                raise AsmError(f"{where}: BMMV TRANSB must be 0/1")
            # View-overrun check on flat element counts (rank-free:
            # strided views index flat storage; a 3D gathered base
            # works like its 2D flattening).
            na = 1
            for d in ash:
                if d is None:
                    na = None
                    break
                na *= d
            if na is not None and \
                    (BB - 1) * SA + (M - 1) * LAA + K > na:
                raise AsmError(f"{where}: BMMV A view overruns base")
            rb = N_ if TB else K
            cb = K if TB else N_
            nb = 1
            for d in bsh:
                if d is None:
                    nb = None
                    break
                nb *= d
            if nb is not None and \
                    (BB - 1) * SB + (rb - 1) * LAB + cb > nb:
                raise AsmError(f"{where}: BMMV B view overruns base")
            streams[o] = ("F", "double", (BB * M, N_))
        elif mn == "ATTN":
            # Fused decode GQA attention: qr (R,H) + Kf/Vf (S,KVD) +
            # mask (PER,S) + nh/nkv/dh/S literals -> ctx (R,H).
            # Single-row decode (R==1); R>1 stays with the BMMV path.
            qk, _, qsh = streams[args[0]]
            fk, _, fsh = streams[args[1]]
            vk, _, vsh = streams[args[2]]
            mk, _, msh = streams[args[3]]
            if qk != "F" or fk != "F" or vk != "F" or mk != "I":
                raise AsmError(f"{where}: ATTN needs F qr/Kf/Vf + I mask")
            (nh, nkv, dh, s) = tuple(
                _int_lit(args[i], f"{where} ATTN") for i in range(4, 8))
            if nh % nkv != 0 or len(qsh) != 2 or len(fsh) != 2 \
                    or len(vsh) != 2 or len(msh) != 2:
                raise AsmError(f"{where}: ATTN needs 2D streams + "
                               f"nh%nkv==0, got {qsh} {fsh} {msh}")
            per = nh // nkv
            if tuple(qsh) != (1, nh * dh) or tuple(fsh) != (s, nkv * dh) \
                    or tuple(vsh) != (s, nkv * dh) \
                    or tuple(msh) != (per, s):
                raise AsmError(f"{where}: ATTN geometry mismatch: qr{qsh} "
                               f"Kf{fsh} Vf{vsh} mask{msh} vs "
                               f"nh={nh} nkv={nkv} dh={dh} S={s}")
            streams[o] = ("F", "double", (1, nh * dh))
        elif mn == "SLICE":
            sk, ssub, ssh = streams[args[0]]
            ax = _int_lit(args[1], f"{where} axis")
            lo = _int_lit(args[2], f"{where} lo")
            hi = _int_lit(args[3], f"{where} hi")
            if sk not in ("F", "T", "I") or len(ssh) != 2 or ax not in (0, 1):
                raise NoPattern(f"{where}: v0.2 SLICE handles 2D ax0/1 "
                                f"F/T/I; use IWINDOW for 1D I")
            dd = list(ssh)
            if dd[ax] is not None and not (0 <= lo <= hi <= dd[ax]):
                raise AsmError(f"{where}: SLICE [{lo},{hi}) of dim {dd[ax]}")
            dd[ax] = hi - lo
            streams[o] = (sk, ssub, tuple(dd))
        elif mn == "IWINDOW":
            xk, xsub, xsh = streams[args[0]]
            lo = _int_lit(args[1], f"{where} start")
            ln = _int_lit(args[2], f"{where} len")
            if xk != "I" or len(xsh) != 1:
                raise NoPattern(f"{where}: IWINDOW handles 1D I; "
                                f"use SLICE for 2D+ F/T")
            if xsh[0] is not None and not (0 <= lo and 0 <= ln
                                           and lo + ln <= xsh[0]):
                raise AsmError(f"{where}: IWINDOW [{lo},{lo + ln}) of "
                               f"dim {xsh[0]}")
            streams[o] = ("I", xsub, (ln,))
        elif mn == "IGATHER1":
            tk, tsub, tsh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            if tk != "I" or len(tsh) != 1:
                raise NoPattern(f"{where}: IGATHER1 needs a 1D I table")
            if ik != "I" or len(ish) != 1:
                raise NoPattern(f"{where}: IGATHER1 needs 1D I ids")
            streams[o] = ("I", tsub, (ish[0],))
        elif mn == "IASSEMBLE":
            xk, xsub, xsh = streams[args[0]]
            lo = _int_lit(args[1], f"{where} start")
            ln = _int_lit(args[2], f"{where} len")
            if xk != "I" or len(xsh) != 1:
                raise NoPattern(f"{where}: IASSEMBLE handles 1D I")
            if not (1 <= ln <= 8):
                raise AsmError(f"{where}: IASSEMBLE len must be 1..8, got {ln}")
            if xsh[0] is not None and not (0 <= lo and lo + ln <= xsh[0]):
                raise AsmError(f"{where}: IASSEMBLE [{lo},{lo + ln}) of "
                               f"dim {xsh[0]}")
            streams[o] = ("I", xsub, (1,))
        elif mn == "ISUM":
            xk, xsub, xsh = streams[args[0]]
            lo = _int_lit(args[1], f"{where} start")
            ln = _int_lit(args[2], f"{where} len")
            if xk != "I" or len(xsh) != 1:
                raise NoPattern(f"{where}: ISUM handles 1D I")
            if xsh[0] is not None and not (0 <= lo and 0 <= ln
                                           and lo + ln <= xsh[0]):
                raise AsmError(f"{where}: ISUM [{lo},{lo + ln}) of "
                               f"dim {xsh[0]}")
            streams[o] = ("I", xsub, (1,))
        elif mn == "ISCATTER1":
            ok, osub, osh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            vk, _, vsh = streams[args[2]]
            if ok != "I" or len(osh) != 1:
                raise NoPattern(f"{where}: ISCATTER1 needs a 1D I OUT")
            if ik != "I" or len(ish) != 1 or vk != "I" or len(vsh) != 1:
                raise NoPattern(f"{where}: ISCATTER1 needs 1D I IDS/VALS")
            if ish[0] is not None and vsh[0] is not None \
                    and ish[0] != vsh[0]:
                raise AsmError(f"{where}: ISCATTER1 IDS {ish[0]} vs "
                               f"VALS {vsh[0]}")
            streams[o] = ("I", osub, osh)
        elif mn == "ROTARY":
            xk, xsub, xsh = streams[args[0]]
            pk, _, psh = streams[args[1]]
            if xk not in ("F", "T") or len(xsh) != 2 or xsh[1] is None \
                    or xsh[1] % 2 or pk != "I" or len(psh) != 1:
                raise NoPattern(f"{where}: ROTARY needs F/T (S,Deven) + "
                                "I pos (S,)")
            if xsh[0] is not None and tuple(psh) != (xsh[0],):
                raise AsmError(f"{where}: ROTARY pos len {psh} vs rows "
                               f"{xsh[0]}")
            streams[o] = (xk, xsub, xsh)
        elif mn == "ROPE":
            # Strided per-head rotary over flat (R,nh*dh): same shape out.
            # Rows R + pos len must agree (static where baked).
            xk, xsub, xsh = streams[args[0]]
            pk, _, psh = streams[args[1]]
            nh = _int_lit(args[2], f"{where} nh")
            dh = _int_lit(args[3], f"{where} dh")
            if xk != "F" or len(xsh) != 2 or pk != "I" or len(psh) != 1:
                raise AsmError(f"{where}: ROPE needs 2D F + 1D I pos")
            if xsh[1] is not None and xsh[1] != nh * dh:
                raise AsmError(f"{where}: ROPE width {xsh[1]} != nh*dh "
                               f"({nh}*{dh})")
            if dh % 2:
                raise AsmError(f"{where}: ROPE needs even dh, got {dh}")
            if xsh[0] is not None and tuple(psh) != (xsh[0],):
                raise AsmError(f"{where}: ROPE pos len {psh} vs rows "
                               f"{xsh[0]}")
            streams[o] = (xk, xsub, (xsh[0], nh * dh))
        elif mn == "SWIGLU":
            # SwiGLU gate: ug (R,2I) + inter literal -> mid (R,I).
            xk, _, xsh = streams[args[0]]
            inter = _int_lit(args[1], f"{where} inter")
            if xk != "F" or len(xsh) != 2:
                raise AsmError(f"{where}: SWIGLU needs 2D F")
            if xsh[1] is not None and xsh[1] != 2 * inter:
                raise AsmError(f"{where}: SWIGLU width {xsh[1]} != 2*I "
                               f"({inter})")
            streams[o] = ("F", "double", (xsh[0], inter))
        elif mn == "TRANSPOSE":
            sk, _, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: v0.2 TRANSPOSE handles 2D F/T")
            streams[o] = (sk, streams[args[0]][1], (ssh[1], ssh[0]))
        elif mn in ("TBETA", "BETA"):
            # ref is PURELY geometric (constant fill takes its shape;
            # values never read -- op_beta doctrine). F/T refs preserve
            # kind; I refs (masks) yield F fills (nonfpu callers pass T).
            rk, rsub, rsh = streams[args[0]]
            if rk == "I":
                streams[o] = ("F", "double", rsh)
            else:
                streams[o] = (rk, rsub, rsh)
        elif mn in ("MUL", "ADD", "SUB"):
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak not in ("F", "T") or ak != bk \
                    or tuple(ash) != tuple(bsh):
                raise NoPattern(f"{where}: {mn} needs same-kind "
                                "same-shape (no broadcast in v0.2)")
            streams[o] = (ak, asub, ash)
        elif mn == "GT":
            ak, _, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak != "T" or bk != "T" \
                    or tuple(ash) != tuple(bsh):
                raise NoPattern(f"{where}: GT needs same-shape T "
                                "triples (no broadcast in v0.2)")
            streams[o] = ("I", "int64_t", tuple(ash))
        elif mn == "SQUARE":
            sk, ssub, ssh = streams[args[0]]
            if sk != "T":
                raise NoPattern(f"{where}: SQUARE needs T (integer "
                                "exactness lives in triples)")
            streams[o] = ("T", ssub, ssh)
        elif mn == "SELECT":
            mk, _, msh = streams[args[0]]
            ak, asub, ash = streams[args[1]]
            bk, _, bsh = streams[args[2]]
            if mk != "I" or ak not in ("F", "T") or ak != bk \
                    or tuple(ash) != tuple(bsh):
                raise NoPattern(f"{where}: SELECT needs I mask + same-kind "
                                "same-shape branches (v0.2)")
            # mask may be baked while branches are batch-dynamic: allow
            # axis-wise (equal OR branch-dynamic), guarded at runtime.
            for dm, da in zip(msh, ash):
                if dm != da and da is not None:
                    raise AsmError(f"{where}: SELECT mask {msh} vs {ash}")
            if len(msh) != len(ash):
                raise AsmError(f"{where}: SELECT rank mismatch")
            streams[o] = (ak, streams[args[1]][1], ash)
        elif mn in ("TSHIFT", "SOFTMAX_WIDE"):
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: {mn} needs 2D F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "SILU":
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T"):
                raise NoPattern(f"{where}: SILU needs F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "SOFTMAX":
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: SOFTMAX needs 2D F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "CONCAT":
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            ax = _int_lit(args[2], f"{where} axis") if args[2] not in \
                streams else None
            if ak not in ("F", "T") or ak != bk or len(ash) != 2 or ax != 1 \
                    or ash[0] != bsh[0] or ash[1] is None or bsh[1] is None:
                raise NoPattern(f"{where}: v0.2 CONCAT handles 2D ax1 "
                                "same-kind, same rows, baked widths")
            streams[o] = (ak, asub, (ash[0], ash[1] + bsh[1]))
        else:
            raise NoPattern(f"C backend: no pattern for {mn} ({where})")
    return streams


class CBackend(Backend):
    """C99 backend. Optimized patterns (backend owns its speed):

    - GATHER zero-copy (row pointers); MATMUL ikj + restrict over
      calloc-zeroed outputs; frozen dims bake in; batch dims dynamic.
    - ARGMAX-T replicates lattice class/exponent order bit-for-bit;
      ARGMAX-F is plain max-first (float path: eps-agreement class).
    - Tune via cflags (default -O2 -march=native) and BLK (matmul
      blocking hook, default off at these sizes -- finishing touch).
    """

    name = "c"
    emits = True
    use_blas = False
    DEFAULT_CFLAGS = ("-O2", "-march=native", "-std=c99", "-Wall")
    BLAS_LIBS = ("-lgslcblas",)

    def value_repr(self, layout):
        kind = (layout or "").split(":")[0]
        return {"I": "int64_t", "F": "double"}.get(kind, "void/*T*/")

    def prologue(self, ctx):
        L = ["#include <stdint.h>", "#include <stdio.h>",
             "#include <stdlib.h>", "#include <string.h>",
             "#include <math.h>", "#include <time.h>", ""]
        if getattr(self, "use_blas", False):
            L.append("#include <gsl/gsl_cblas.h>")
        for k, v in ctx["dims"].items():
            L.append(f"#define DIM_{k} {v}L")
        th = ctx.get("theta", [])
        if th:
            L.append("static const double THETA[] = {%s};" %
                     ", ".join(repr(t) for t in th))
        return "\n".join(L) + "\n"

    # -- helpers -----------------------------------------------------
    def _we(self, S, x, ctx, where):
        """Row-width expr of stream x: baked int, or the batch var when
        the last dim is dynamic (all dynamic dims derive from ids)."""
        w = S[x][2][-1]
        return str(int(w)) if w is not None else ctx["ids_n"]

    @staticmethod
    def _w(S, x, where):
        """Baked row-width of stream x (last dim); dynamic fails loud."""
        w = S[x][2][-1]
        if w is None:
            raise NoPattern(f"C backend: dynamic width in {where} "
                            f"(finishing touch, not this cut)")
        return int(w)

    # -- patterns ----------------------------------------------------
    def pattern(self, mn, outs, args, sig, ctx):
        S = ctx["streams"]
        cfg = ctx["config"]
        ids_n = ctx["ids_n"]
        if mn == "GATHER":
            t, ids = args
            tk = S[t][0]
            o = outs[0]
            if tk == "T":
                st, et, zt = S[t][1]
                w = self._we(S, t, ctx, f"GATHER {o}")
                pref = f"gr_{o}"
                return (
                    f"/* {o} = GATHER({t}, {ids}) materialized + row views */\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {{\n"
                    f"  memcpy(&{o}_s[g_i * {w}], &{t}_s[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_s[0]));\n"
                    f"  memcpy(&{o}_e[g_i * {w}], &{t}_e[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_e[0]));\n"
                    f"  memcpy(&{o}_z[g_i * {w}], &{t}_z[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_z[0]));\n"
                    f"}}\n"
                    f"const {st} *{pref}_rows_s[{ids}_N];\n"
                    f"const {et} *{pref}_rows_e[{ids}_N];\n"
                    f"const {zt} *{pref}_rows_z[{ids}_N];\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {{\n"
                    f"  {pref}_rows_s[g_i] = &{o}_s[g_i * {w}];\n"
                    f"  {pref}_rows_e[g_i] = &{o}_e[g_i * {w}];\n"
                    f"  {pref}_rows_z[g_i] = &{o}_z[g_i * {w}];\n"
                    f"}}",
                    (o, (w, pref)))
            if tk == "F":
                w = self._we(S, t, ctx, f"GATHER {o}")
                return (
                    f"/* {o} = GATHER({t}, {ids}) materialized rows */\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i)\n"
                    f"  memcpy(&{o}[g_i * {w}], &{t}[{ids}[g_i] * {w}], "
                    f"{w} * sizeof(double));",
                    (o, w))
            raise NoPattern("GATHER table must be T/F (v0.2)")
        if mn == "GATHER3":
            t, ii, jj = args
            tk = S[t][0]
            o = outs[0]
            tsh = S[t][2]
            if len(tsh) != 3 or any(d is None for d in tsh):
                raise NoPattern(f"GATHER3 needs baked (V1,V2,C), got {tsh}")
            v1, v2, w = (int(d) for d in tsh)
            guard = (f"if ({ii}_N != {jj}_N) return 20;\n"
                     f"for (int64_t g_i = 0; g_i < {o}_N; ++g_i) {{\n"
                     f"  int64_t _a = {ii}[g_i], _b = {jj}[g_i];\n"
                     f"  if (_a < 0 || _a >= {v1} || _b < 0 || _b >= {v2}) "
                     f"return 4;\n")
            if tk == "T":
                st, et, zt = S[t][1]
                pref = f"g3_{o}"
                L = [f"/* {o} = GATHER3({t}, {ii}, {jj}) materialized + "
                     f"row views */", guard]
                for comp in ("s", "e", "z"):
                    L.append(
                        f"  memcpy(&{o}_{comp}[g_i * {w}], "
                        f"&{t}_{comp}[(_a * {v2} + _b) * {w}], "
                        f"{w} * sizeof({o}_{comp}[0]));")
                L.append("}")
                L.append(f"const {st} *{pref}_rows_s[{o}_N];")
                L.append(f"const {et} *{pref}_rows_e[{o}_N];")
                L.append(f"const {zt} *{pref}_rows_z[{o}_N];")
                L.append(f"for (int64_t g_i = 0; g_i < {o}_N; ++g_i) {{")
                for comp in ("s", "e", "z"):
                    L.append(f"  {pref}_rows_{comp}[g_i] = "
                             f"&{o}_{comp}[g_i * {w}];")
                L.append("}")
                return ("\n".join(L), (o, (str(w), pref)))
            if tk == "F":
                return (
                    f"/* {o} = GATHER3({t}, {ii}, {jj}) materialized rows */\n"
                    + guard +
                    f"  memcpy(&{o}[g_i * {w}], "
                    f"&{t}[(_a * {v2} + _b) * {w}], "
                    f"{w} * sizeof(double));\n}}",
                    (o, str(w)))
            raise NoPattern("GATHER3 table must be T/F")
        if mn == "IWINDOW":
            x = args[0]
            o = outs[0]
            lo = _int_lit(args[1], "IWINDOW start")
            ln = _int_lit(args[2], "IWINDOW len")
            if S[x][0] != "I" or len(S[x][2]) != 1:
                raise NoPattern("C backend IWINDOW needs 1D I")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend IWINDOW: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]
            return (
                f"/* {o} = IWINDOW({x}, {lo}, {ln}) exact */\n"
                f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                f"  {o}[e_i] = {x}[{lo} + e_i];",
                None)
        if mn == "IGATHER1":
            t, ids = args
            o = outs[0]
            if S[t][0] != "I" or len(S[t][2]) != 1:
                raise NoPattern("C backend IGATHER1 needs a 1D I table")
            if S[ids][0] != "I" or len(S[ids][2]) != 1:
                raise NoPattern("C backend IGATHER1 needs 1D I ids")
            vN = S[t][2][0]
            if vN is None:
                raise NoPattern("C backend IGATHER1: table length must be "
                                "static (bounds check)")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend IGATHER1: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]
            return (
                f"/* {o} = IGATHER1({t}, {ids}) exact, bounds-checked */\n"
                f"for (int64_t g_i = 0; g_i < {co}; ++g_i) {{\n"
                f"  int64_t _b = {ids}[g_i];\n"
                f"  if (_b < 0 || _b >= {vN}) return 4;\n"
                f"  {o}[g_i] = {t}[_b];\n"
                f"}}",
                None)
        if mn == "IASSEMBLE":
            x = args[0]
            o = outs[0]
            lo = _int_lit(args[1], "IASSEMBLE start")
            ln = _int_lit(args[2], "IASSEMBLE len")
            if S[x][0] != "I" or len(S[x][2]) != 1:
                raise NoPattern("C backend IASSEMBLE needs 1D I")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend IASSEMBLE: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            return (
                f"/* {o} = IASSEMBLE({x}, {lo}, {ln}) little-endian, exact */\n"
                f"{{ uint64_t _v = 0;\n"
                f"  for (int64_t e_i = 0; e_i < {ln}; ++e_i)\n"
                f"    _v |= ((uint64_t)({x}[{lo} + e_i] & 0xFF)) << (8 * e_i);\n"
                f"  {o}[0] = (int64_t)_v; }}",
                None)
        if mn == "ISUM":
            x = args[0]
            o = outs[0]
            lo = _int_lit(args[1], "ISUM start")
            ln = _int_lit(args[2], "ISUM len")
            if S[x][0] != "I" or len(S[x][2]) != 1:
                raise NoPattern("C backend ISUM needs 1D I")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend ISUM: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            return (
                f"/* {o} = ISUM({x}, {lo}, {ln}) exact */\n"
                f"{{ int64_t _v = 0;\n"
                f"  for (int64_t e_i = 0; e_i < {ln}; ++e_i)\n"
                f"    _v += {x}[{lo} + e_i];\n"
                f"  {o}[0] = _v; }}",
                None)
        if mn == "ISCATTER1":
            base, ids, vals = args
            o = outs[0]
            if S[base][0] != "I" or len(S[base][2]) != 1:
                raise NoPattern("C backend ISCATTER1 needs a 1D I OUT")
            if S[ids][0] != "I" or len(S[ids][2]) != 1:
                raise NoPattern("C backend ISCATTER1 needs 1D I IDS")
            if S[vals][0] != "I" or len(S[vals][2]) != 1:
                raise NoPattern("C backend ISCATTER1 needs 1D I VALS")
            vn = S[base][2][0]
            if vn is None:
                raise NoPattern("C backend ISCATTER1: OUT length must be "
                                "static (bounds check)")
            nids = S[ids][2][0]
            if nids is None:
                raise NoPattern("C backend ISCATTER1: IDS length must be "
                                "static (loop bound)")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend ISCATTER1: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]
            return (
                f"/* {o} = ISCATTER1({base}, {ids}, {vals}) exact, "
                f"bounds-checked */\n"
                f"for (int64_t s_i = 0; s_i < {co}; ++s_i) "
                f"{o}[s_i] = {base}[s_i];\n"
                f"for (int64_t s_i = 0; s_i < {nids}; ++s_i) {{\n"
                f"  int64_t _b = {ids}[s_i];\n"
                f"  if (_b < 0 || _b >= {vn}) return 4;\n"
                f"  {o}[_b] = {vals}[s_i];\n"
                f"}}",
                None)
        if mn in ("IADD", "ISUB", "IMUL", "IMAD", "IDIV", "IMOD"):
            a = args[0]
            o = outs[0]
            if S[a][0] != "I":
                raise NoPattern(f"C backend {mn} needs I first operand")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend {mn}: dynamic intermediate '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]

            def _term(x, pos):
                if x in S:
                    if S[x][0] != "I" or tuple(S[x][2]) != tuple(S[a][2]):
                        raise NoPattern(
                            f"C backend {mn} operand {pos} must be I with "
                            f"shape {S[a][2]}, got {S[x][0]} {S[x][2]}")
                    return f"{x}[e_i]"
                return _c_i64_lit(_int_lit(x, f"{mn} operand {pos}"))

            ta = f"{a}[e_i]"
            tb = _term(args[1], 1)
            if mn in ("IDIV", "IMOD"):
                if args[1] not in S and _int_lit(
                        args[1], f"{mn} operand 1") == 0:
                    raise AsmError(f"{mn}: division by zero (literal)")
                op = "/" if mn == "IDIV" else "%"
                return (
                    f"/* {o} = {mn}({', '.join(args)}) truncating, "
                    f"checked */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i) {{\n"
                    f"  int64_t _a = {a}[e_i];\n"
                    f"  int64_t _b = {tb};\n"
                    f"  if (_b == 0 || "
                    f"(_a == INT64_MIN && _b == -1)) return 30;\n"
                    f"  {o}[e_i] = _a {op} _b;\n"
                    f"}}",
                    None)
            if mn == "IADD":
                expr = f"((uint64_t){ta} + (uint64_t)({tb}))"
            elif mn == "ISUB":
                expr = f"((uint64_t){ta} - (uint64_t)({tb}))"
            elif mn == "IMUL":
                expr = f"((uint64_t){ta} * (uint64_t)({tb}))"
            else:
                tc = _term(args[2], 2)
                expr = (f"((uint64_t){ta} * (uint64_t)({tb}) + "
                        f"(uint64_t)({tc}))")
            return (
                f"/* {o} = {mn}({', '.join(args)}) 64-bit wrap */\n"
                f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                f"  {o}[e_i] = (int64_t){expr};",
                None)
        if mn in ("IXOR", "IAND", "IOR"):
            a = args[0]
            o = outs[0]
            if S[a][0] != "I":
                raise NoPattern(f"C backend {mn} needs I first operand")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend {mn}: dynamic intermediate '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]

            def _term(x, pos):
                if x in S:
                    if S[x][0] != "I" or tuple(S[x][2]) != tuple(S[a][2]):
                        raise NoPattern(
                            f"C backend {mn} operand {pos} must be I with "
                            f"shape {S[a][2]}, got {S[x][0]} {S[x][2]}")
                    return f"{x}[e_i]"
                return _c_i64_lit(_int_lit(x, f"{mn} operand {pos}"))

            ta = f"{a}[e_i]"
            tb = _term(args[1], 1)
            op = {"IXOR": "^", "IAND": "&", "IOR": "|"}[mn]
            return (
                f"/* {o} = {mn}({', '.join(args)}) 64-bit wrap */\n"
                f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                f"  {o}[e_i] = (int64_t)(((uint64_t){ta}) {op} "
                f"((uint64_t)({tb})));",
                None)
        if mn in ("ISHL", "ISHR"):
            a = args[0]
            o = outs[0]
            if S[a][0] != "I":
                raise NoPattern(f"C backend {mn} needs I first operand")
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend {mn}: dynamic intermediate '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]
            s = _int_lit(args[1], f"{mn} shift")
            if not (0 <= s <= 63):
                raise AsmError(f"{mn}: shift must be 0..63, got {s}")
            op = "<<" if mn == "ISHL" else ">>"
            return (
                f"/* {o} = {mn}({a}, {s}) logical */\n"
                f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                f"  {o}[e_i] = (int64_t)(((uint64_t){a}[e_i]) {op} {s});",
                None)
        if mn in ("F2I", "I2F"):
            x = args[0]
            o = outs[0]
            if o not in ctx["count"]:
                raise NoPattern(
                    f"C backend {mn}: dynamic output '{o}' "
                    f"(v0.1 needs a static shape)")
            co = ctx["count"][o]
            if mn == "F2I":
                if S[x][0] != "F":
                    raise NoPattern(f"C backend F2I needs F input, "
                                    f"got {S[x][0]}")
                return (
                    f"/* {o} = F2I({x}) trunc toward zero, range-checked */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i) {{\n"
                    f"  double _v = {x}[e_i];\n"
                    f"  if (_v != _v || "
                    f"_v >= 9223372036854775808.0 || "
                    f"_v < -9223372036854775808.0) return 30;\n"
                    f"  {o}[e_i] = (int64_t)_v;\n"
                    f"}}",
                    None)
            if S[x][0] != "I":
                raise NoPattern(f"C backend I2F needs I input, "
                                f"got {S[x][0]}")
            return (
                f"/* {o} = I2F({x}) round-to-nearest-even */\n"
                f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                f"  {o}[e_i] = (double){x}[e_i];",
                None)
        if mn == "SCATTER":
            base, row, idx = args
            o = outs[0]
            bk, _, bsh = S[base]
            rk, _, rsh = S[row]
            if bk != "F" or rk != "F":
                raise NoPattern("C SCATTER needs F base + F row "
                                "(T path unbuilt: caches are F)")
            if len(bsh) != 2 or tuple(rsh) != (1, bsh[1]):
                raise AsmError(f"C SCATTER base {bsh} + row {rsh} "
                               f"mismatch (row must be (1,C))")
            if bsh[0] is None or bsh[1] is None:
                raise NoPattern("C SCATTER needs static shapes (v0.2)")
            s, c = bsh
            return (
                f"/* {o} = SCATTER({base},{row},{idx}) */\n"
                f"{{ int64_t _n = {idx}[0];\n"
                f"  if (_n < 0 || _n >= {s}) return 4;\n"
                f"  memcpy({o}, {base}, ({s} * {c}) * sizeof(double));\n"
                f"  memcpy(&{o}[_n * {c}], {row}, {c} * sizeof(double)); }}",
                None)
        if mn == "ARGMAX":
            src = args[0]
            sk = S[src][0]
            o = outs[0]
            if sk == "T":
                if src not in ctx["gathered"]:
                    raise AsmError(f"ARGMAX C pattern v0.2 needs a "
                                   f"gathered T row-stream, got '{src}'")
                st, et, zt = S[src][1]
                w, pref = ctx["gathered"][src]
                return (
                    f"/* {o} = ARGMAX({src}, 1) -- class rank + exponent, "
                    f"ties first */\n"
                    f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                    f"  int64_t best = 0; int bcls = -1; int64_t bkey = 0;\n"
                    f"  for (int64_t a_j = 0; a_j < {w}; ++a_j) {{\n"
                    f"    {st} s = {pref}_rows_s[a_i][a_j];\n"
                    f"    {et} e = {pref}_rows_e[a_i][a_j];\n"
                    f"    {zt} z = {pref}_rows_z[a_i][a_j];\n"
                    f"    int cls = (!z && s > 0) ? 2 : (z ? 1 : 0);\n"
                    f"    int64_t key = (!z && s > 0) ? (int64_t)e : "
                    f"(z ? 0 : -(int64_t)e);\n"
                    f"    if (bcls < 0 || cls > bcls || "
                    f"(cls == bcls && key > bkey)) {{\n"
                    f"      best = a_j; bcls = cls; bkey = key;\n"
                    f"    }}\n"
                    f"  }}\n"
                    f"  {o}[a_i] = best;\n"
                    f"}}",
                    None)
            if sk == "F":
                if len(S[src][2]) != 2:
                    raise NoPattern("ARGMAX-F v0.2 needs 2D input")
                w = self._we(S, src, ctx, f"ARGMAX {o}")
                return (
                    f"/* {o} = ARGMAX({src}) float max-first */\n"
                    f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                    f"  int64_t best = 0; double bv = {src}[a_i * {w}];\n"
                    f"  for (int64_t a_j = 1; a_j < {w}; ++a_j)\n"
                    f"    if ({src}[a_i * {w} + a_j] > bv) {{\n"
                    f"      bv = {src}[a_i * {w} + a_j]; best = a_j;\n"
                    f"    }}\n"
                    f"  {o}[a_i] = best;\n"
                    f"}}",
                    None)
            raise NoPattern(f"ARGMAX needs T/F input, got {sk} (v0.2)")
        if mn == "TOPK":
            x = args[0]
            vo, io = outs[0], outs[1]
            if S[x][0] != "F" or len(S[x][2]) != 2:
                raise NoPattern("C TOPK needs 2D F scores")
            w = S[x][2][1]
            k = _int_lit(args[1], "TOPK k")
            if w is None or not (1 <= k <= w):
                raise NoPattern(
                    f"C TOPK needs static width S>=k (k={k}, S={w})")
            # Stable descending selection, ties first: scan the row for
            # the best not-yet-taken index each slot. O(rows*k*S*k) but
            # scores are tiny (256) and this is the exact lattice rule.
            return (
                f"/* {vo},{io} = TOPK({x},{k}) stable ties-first */\n"
                f"for (int64_t t_i = 0; t_i < {vo}_N; ++t_i) {{\n"
                f"  for (int64_t t_j = 0; t_j < {k}; ++t_j) {{\n"
                f"    int64_t best = -1;\n"
                f"    for (int64_t t_c = 0; t_c < {w}; ++t_c) {{\n"
                f"      int _used = 0;\n"
                f"      for (int64_t t_p = 0; t_p < t_j; ++t_p)\n"
                f"        if ({io}[t_i * {k} + t_p] == t_c) {{"
                f" _used = 1; break; }}\n"
                f"      if (_used) continue;\n"
                f"      if (best < 0 || {x}[t_i * {w} + t_c] > "
                f"{x}[t_i * {w} + best]) best = t_c;\n"
                f"    }}\n"
                f"    {vo}[t_i * {k} + t_j] = {x}[t_i * {w} + best];\n"
                f"    {io}[t_i * {k} + t_j] = best;\n"
                f"  }}\n"
                f"}}",
                None)
        # -- F64 patterns (double *X streams, malloc'd per op) -------
        if mn == "RMSNORM":
            x, w = args
            o = outs[0]
            if S[x][0] != "F" or S[w][0] != "F":
                raise NoPattern("C backend RMSNORM needs F (T lives "
                                "on nonfpu)")
            eps = float(cfg.get("eps_rms", 0.0)) if "eps_rms" in cfg \
                else (_ for _ in ()).throw(
                    AsmError("RMSNORM C pattern needs CONFIG eps_rms "
                             "(true float; counts convention is lattice-only)"))
            d = self._we(S, x, ctx, f"RMSNORM {o}")
            return (
                f"/* {o} = RMSNORM({x}) eps={eps} */\n"
                f"for (int64_t r_i = 0; r_i < {o}_N; ++r_i) {{\n"
                f"  double ss = 0;\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    ss += {x}[r_i * {d} + r_j] * {x}[r_i * {d} + r_j];\n"
                f"  double rs = 1.0 / sqrt(ss / {d} + {eps!r});\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    {o}[r_i * {d} + r_j] = {x}[r_i * {d} + r_j] * rs "
                f"* {w}[r_j];\n"
                f"}}",
                None)
        if mn == "LAYERNORM":
            x, w, b = args
            o = outs[0]
            if S[x][0] != "F" or S[w][0] != "F" or S[b][0] != "F":
                raise NoPattern("C backend LAYERNORM needs F (T lives "
                                "on nonfpu)")
            eps = float(cfg.get("eps_ln", 0.0)) if "eps_ln" in cfg \
                else (_ for _ in ()).throw(
                    AsmError("LAYERNORM C pattern needs CONFIG eps_ln "
                             "(true float; counts convention is lattice-only)"))
            d = self._we(S, x, ctx, f"LAYERNORM {o}")
            return (
                f"/* {o} = LAYERNORM({x}) eps={eps} */\n"
                f"for (int64_t r_i = 0; r_i < {o}_N; ++r_i) {{\n"
                f"  double mu = 0;\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    mu += {x}[r_i * {d} + r_j];\n"
                f"  mu /= {d};\n"
                f"  double ss = 0;\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j) {{\n"
                f"    double dv = {x}[r_i * {d} + r_j] - mu;\n"
                f"    ss += dv * dv;\n"
                f"  }}\n"
                f"  double rs = 1.0 / sqrt(ss / {d} + {eps!r});\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    {o}[r_i * {d} + r_j] = ({x}[r_i * {d} + r_j] - mu) * rs "
                f"* {w}[r_j] + {b}[r_j];\n"
                f"}}",
                None)
        if mn == "CONV":
            # v0.1 float single-channel 2D, same output, stride 1,
            # dilation 1, edge-replicate boundary (audited vs torch;
            # replicate_pad EXACT, zero-pad diverges at borders).
            x, k = args
            o = outs[0]
            if S[x][0] != "F" or S[k][0] != "F":
                raise NoPattern("C backend CONV needs F (T lives "
                                "on nonfpu)")
            H, W = S[x][2]
            kH, kW = S[k][2]
            if (None in (H, W, kH, kW) or kH % 2 == 0 or kW % 2 == 0
                    or kH > H or kW > W):
                raise AsmError("C backend CONV needs static 2D odd "
                               f"kernel within image, got ({H},{W}) ({kH},{kW})")
            rH, rW = kH // 2, kW // 2
            return (
                f"/* {o} = CONV({x}) edge-replicate {kH}x{kW} */\n"
                f"for (int64_t oy = 0; oy < {H}; ++oy) {{\n"
                f"  for (int64_t ox = 0; ox < {W}; ++ox) {{\n"
                f"    double acc = 0;\n"
                f"    for (int64_t ky = 0; ky < {kH}; ++ky) {{\n"
                f"      int64_t iy = oy + ky - {rH};\n"
                f"      iy = iy < 0 ? 0 : (iy >= {H} ? {H} - 1 : iy);\n"
                f"      for (int64_t kx = 0; kx < {kW}; ++kx) {{\n"
                f"        int64_t ix = ox + kx - {rW};\n"
                f"        ix = ix < 0 ? 0 : (ix >= {W} ? {W} - 1 : ix);\n"
                f"        acc += {x}[iy * {W} + ix] * {k}[ky * {kW} + kx];\n"
                f"      }}\n"
                f"    }}\n"
                f"    {o}[oy * {W} + ox] = acc;\n"
                f"  }}\n"
                f"}}",
                None)
        if mn == "INTERP":
            # Bilinear, half-pixel centers (align_corners=False —
            # audited: lattice matches torch align_corners=False to
            # fixed-point noise, True diverges). Border: clamp
            # (torch default). 2D or 3D(HWC); channels outermost.
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend INTERP needs F (T lives "
                                "on nonfpu)")
            try:
                sy, sx = float(args[1]), float(args[2])
            except (TypeError, ValueError):
                raise NoPattern("C backend INTERP needs literal scales")
            if not (_dyadic(sy) and _dyadic(sx)):
                raise AsmError("C backend INTERP non-dyadic scale "
                               f"({sy},{sx}) is a LOWERING ERROR")
            sh = S[x][2]
            if len(sh) not in (2, 3) or None in sh:
                raise AsmError("C backend INTERP needs static (H,W[,C])")
            H, W = sh[0], sh[1]
            C = sh[2] if len(sh) == 3 else 1
            Ho = max(int(round(H * sy)), 1)
            Wo = max(int(round(W * sx)), 1)
            return (
                f"/* {o} = INTERP({x}) bilinear {H}x{W}->{Ho}x{Wo} */\n"
                f"for (int64_t oy = 0; oy < {Ho}; ++oy) {{\n"
                f"  double fsy = ((double)oy + 0.5) * {(H / Ho)!r} - 0.5;\n"
                f"  fsy = fsy < 0 ? 0 : (fsy > {H} - 1 ? {H} - 1 : fsy);\n"
                f"  int64_t y0 = (int64_t)floor(fsy); double wy = fsy - y0;\n"
                f"  int64_t y1 = y0 + 1 >= {H} ? {H} - 1 : y0 + 1;\n"
                f"  for (int64_t ox = 0; ox < {Wo}; ++ox) {{\n"
                f"    double fsx = ((double)ox + 0.5) * {(W / Wo)!r} - 0.5;\n"
                f"    fsx = fsx < 0 ? 0 : (fsx > {W} - 1 ? {W} - 1 : fsx);\n"
                f"    int64_t x0 = (int64_t)floor(fsx); double wx = fsx - x0;\n"
                f"    int64_t x1 = x0 + 1 >= {W} ? {W} - 1 : x0 + 1;\n"
                f"    for (int64_t cc = 0; cc < {C}; ++cc)\n"
                f"      {o}[(oy * {Wo} + ox) * {C} + cc] =\n"
                f"        (1 - wy) * (1 - wx) * {x}[(y0 * {W} + x0) * {C} + cc]\n"
                f"        + (1 - wy) * wx * {x}[(y0 * {W} + x1) * {C} + cc]\n"
                f"        + wy * (1 - wx) * {x}[(y1 * {W} + x0) * {C} + cc]\n"
                f"        + wy * wx * {x}[(y1 * {W} + x1) * {C} + cc];\n"
                f"  }}\n"
                f"}}",
                None)
        if mn in ("MATMUL", "BATCH_MATMUL"):
            a, b = args
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"C backend {mn} needs F (T matmul "
                                "lives on nonfpu)")
            co = ctx["count"][o]
            ash, bsh = S[a][2], S[b][2]
            zero = f"memset({o}, 0, {co} * sizeof(double));"
            if len(ash) == 2:
                m, k, n = ash[0], ash[1], bsh[1]
                # Any dynamic dim is batch-derived (only GATHER makes
                # None); baked otherwise. Contraction over batch is legal.
                me = str(m) if m is not None else ids_n
                ke = str(k) if k is not None else ids_n
                ne = str(n) if n is not None else ids_n
                return (
                    f"/* {o} = {mn} ikj, restrict */\n{zero}\n"
                    f"for (int64_t m_i = 0; m_i < {me}; ++m_i)\n"
                    f"  for (int64_t m_k = 0; m_k < {ke}; ++m_k) {{\n"
                    f"    double aik = {a}[m_i * {ke} + m_k];\n"
                    f"    double *restrict crow = &{o}[m_i * {ne}];\n"
                    f"    const double *restrict brow = &{b}[m_k * {ne}];\n"
                    f"    for (int64_t m_j = 0; m_j < {ne}; ++m_j)\n"
                    f"      crow[m_j] += aik * brow[m_j];\n"
                    f"  }}",
                    None)
            B, m, k, n = ash[0], ash[1], ash[2], bsh[2]
            if B is None:
                raise NoPattern(f"C backend: {mn} needs baked batch "
                                f"dim, got {ash}/{bsh}")
            me = str(m) if m is not None else ids_n
            ke = str(k) if k is not None else ids_n
            ne = str(n) if n is not None else ids_n
            if getattr(self, "use_blas", False):
                return (
                    f"/* {o} = {mn} batched cblas_dgemm x{B} */\n"
                    f"for (int64_t m_b = 0; m_b < {B}; ++m_b)\n"
                    f"  cblas_dgemm(CblasRowMajor, CblasNoTrans, "
                    f"CblasNoTrans, {me}, {ne}, {ke}, 1.0,\n"
                    f"    &{a}[m_b * ({me}) * ({ke})], {ke},\n"
                    f"    &{b}[m_b * ({ke}) * ({ne})], {ne}, 0.0,\n"
                    f"    &{o}[m_b * ({me}) * ({ne})], {ne});",
                    None)
            return (
                f"/* {o} = {mn} batched ikj */\n{zero}\n"
                f"for (int64_t m_b = 0; m_b < {B}; ++m_b)\n"
                f"  for (int64_t m_i = 0; m_i < {me}; ++m_i)\n"
                f"    for (int64_t m_k = 0; m_k < {ke}; ++m_k) {{\n"
                f"      double aik = {a}[(m_b * {me} + m_i) * {ke} + m_k];\n"
                f"      double *restrict crow = &{o}[(m_b * {me} + m_i)"
                f" * {ne}];\n"
                f"      const double *restrict brow = &{b}[(m_b * {ke} + m_k)"
                f" * {ne}];\n"
                f"      for (int64_t m_j = 0; m_j < {ne}; ++m_j)\n"
                f"        crow[m_j] += aik * brow[m_j];\n"
                f"    }}",
                None)
        if mn == "BMMV":
            a, b = args[0], args[1]
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"C backend BMMV needs F bases")
            (BB, M, N_, K, LAA, SA, LAB, SB, _LAC, _SC, TB) = tuple(
                _int_lit(args[i], "BMMV") for i in range(2, 13))
            if TB not in (0, 1):
                raise NoPattern(f"C backend BMMV TRANSB must be 0/1")
            co = ctx["count"][o]
            zero = f"memset({o}, 0, {co} * sizeof(double));"
            if TB:
                inner = (f"      const double *restrict bcol = &{b}[m_b * {SB} + m_k];\n"
                         f"      for (int64_t m_j = 0; m_j < {N_}; ++m_j)\n"
                         f"        crow[m_j] += aik * bcol[m_j * {LAB}];\n")
            else:
                inner = (f"      const double *restrict brow = &{b}[m_b * {SB} + m_k * {LAB}];\n"
                         f"      for (int64_t m_j = 0; m_j < {N_}; ++m_j)\n"
                         f"        crow[m_j] += aik * brow[m_j];\n")
            return (
                f"/* {o} = BMMV strided ikj, restrict */\n{zero}\n"
                f"for (int64_t m_b = 0; m_b < {BB}; ++m_b)\n"
                f"  for (int64_t m_i = 0; m_i < {M}; ++m_i)\n"
                f"    for (int64_t m_k = 0; m_k < {K}; ++m_k) {{\n"
                f"      double aik = {a}[m_b * {SA} + m_i * {LAA} + m_k];\n"
                f"      double *restrict crow = &{o}[(m_b * {M} + m_i) * {N_}];\n"
                f"{inner}"
                f"  }}",
                None)
        if mn == "SLICE":
            x, ax, lo, hi = args[0], _int_lit(args[1], "SLICE ax"), \
                _int_lit(args[2], "SLICE lo"), _int_lit(args[3], "SLICE hi")
            o = outs[0]
            co = ctx["count"][o]
            in_d = S[x][2][ax]
            if in_d is None:
                raise NoPattern("C backend: SLICE with dynamic stride")
            if S[x][0] == "T":
                st, et, zt = S[x][1]
                if ax == 1:
                    L = [f"/* {o} = SLICE({x},1,{lo},{hi}) triples, exact */"]
                    for comp in ("s", "e", "z"):
                        L.append(
                            f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                            f"  memcpy(&{o}_{comp}[s_i * {hi - lo}], "
                            f"&{x}_{comp}[s_i * {in_d} + {lo}], "
                            f"{hi - lo} * sizeof({o}_{comp}[0]));")
                    return ("\n".join(L), None)
                w1 = S[x][2][1]
                if w1 is None:
                    raise NoPattern("C backend: SLICE-T ax0 dynamic width")
                co = ctx["count"][o]
                L = [f"/* {o} = SLICE({x},0,{lo},{hi}) triples, exact */"]
                for comp in ("s", "e", "z"):
                    L.append(
                        f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                        f"  memcpy(&{o}_{comp}[s_i * {w1}], "
                        f"&{x}_{comp}[(s_i + {lo}) * {w1}], "
                        f"{w1} * sizeof({o}_{comp}[0]));")
                return ("\n".join(L), None)
            if S[x][0] == "I":
                osh = S[o][2]
                if ax == 1:
                    rows = osh[0]
                    if rows is None:
                        raise NoPattern(
                            "C backend: SLICE-I ax1 dynamic rows")
                    return (
                        f"/* {o} = SLICE({x},1,{lo},{hi}) int64 */\n"
                        f"for (int64_t s_i = 0; s_i < {rows}; ++s_i)\n"
                        f"  memcpy(&{o}[s_i * {hi - lo}], "
                        f"&{x}[s_i * {in_d} + {lo}], "
                        f"{hi - lo} * sizeof(int64_t));",
                        None)
                w1 = S[x][2][1]
                if w1 is None:
                    raise NoPattern(
                        "C backend: SLICE-I ax0 dynamic width")
                return (
                    f"/* {o} = SLICE({x},0,{lo},{hi}) int64 rows */\n"
                    f"memcpy({o}, &{x}[{lo} * {w1}], "
                    f"{co} * sizeof(int64_t));",
                    None)
            if ax == 1:
                return (
                    f"/* {o} = SLICE({x},1,{lo},{hi}) */\n"
                    f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                    f"  memcpy(&{o}[s_i * {hi - lo}], "
                    f"&{x}[s_i * {in_d} + {lo}], "
                    f"{hi - lo} * sizeof(double));",
                    None)
            w1 = self._we(S, x, ctx, f"SLICE {o} stride")
            return (
                f"/* {o} = SLICE({x},0,{lo},{hi}) rows */\n"
                f"memcpy({o}, &{x}[{lo} * {w1}], "
                f"{co} * sizeof(double));",
                None)
        if mn == "ROTARY":
            x, pos = args
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend ROTARY needs F (T lives "
                                "on nonfpu)")
            d = self._w(S, x, f"ROTARY {o}")
            nh = d // 2
            return (
                f"/* {o} = ROTARY({x}) runtime sincos */\n"
                f"for (int64_t t_i = 0; t_i < {o}_N; ++t_i) {{\n"
                f"  int64_t pp = {pos}[t_i];\n"
                f"  for (int64_t t_h = 0; t_h < {nh}; ++t_h) {{\n"
                f"    double ang = (double)pp * THETA[t_h];\n"
                f"    double c = cos(ang), s = sin(ang);\n"
                f"    double x0 = {x}[(t_i * {d}) + 2 * t_h];\n"
                f"    double x1 = {x}[(t_i * {d}) + 2 * t_h + 1];\n"
                f"    {o}[(t_i * {d}) + 2 * t_h] = x0 * c - x1 * s;\n"
                f"    {o}[(t_i * {d}) + 2 * t_h + 1] = x0 * s + x1 * c;\n"
                f"  }}\n"
                f"}}",
                None)
        if mn == "TRANSPOSE":
            x = args[0]
            o = outs[0]
            if S[x][0] == "T":
                st, et, zt = S[x][1]
                c = S[x][2][1]
                if c is None:
                    raise NoPattern("C backend: TRANSPOSE-T dynamic width")
                r = S[x][2][0]
                re = str(int(r)) if r is not None else f"{x}_N"
                L = [f"/* {o} = TRANSPOSE({x}) triples, exact move */"]
                for comp, ct in (("s", st), ("e", et), ("z", zt)):
                    L.append(
                        f"for (int64_t t_i = 0; t_i < {re}; ++t_i)\n"
                        f"  for (int64_t t_j = 0; t_j < {c}; ++t_j)\n"
                        f"    {o}_{comp}[t_j * {re} + t_i] = "
                        f"{x}_{comp}[t_i * {c} + t_j];")
                return ("\n".join(L), None)
            c = self._we(S, x, ctx, f"TRANSPOSE {o}")
            r = S[x][2][0]
            if r is None and x in ctx["inputs"]:
                raise NoPattern("C backend: TRANSPOSE of dynamic-row input")
            re = str(int(r)) if r is not None else f"{x}_N"
            return (
                f"/* {o} = TRANSPOSE({x}) */\n"
                f"for (int64_t t_i = 0; t_i < {re}; ++t_i)\n"
                f"  for (int64_t t_j = 0; t_j < {c}; ++t_j)\n"
                f"    {o}[t_j * {re} + t_i] = {x}[t_i * {c} + t_j];",
                None)
        if mn in ("TBETA", "BETA"):
            ref = args[0]
            o = outs[0]
            # ref is geometric only (constant fill); any kind accepted.
            co = ctx["count"][o]
            val = float(cfg.get("beta_b", float(cfg.get("beta", 0.5)))) \
                if mn == "TBETA" else float(cfg.get("beta", 0.5))
            return (f"/* {o} = {mn} ({val}) over {ref} shape */\n"
                    f"for (int64_t k_i = 0; k_i < {co}; ++k_i)\n"
                    f"  {o}[k_i] = {val!r};",
                    None)
        if mn in ("MUL", "ADD", "SUB"):
            a, b = args
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"C backend {mn} needs F (T arithmetic "
                                "lives on nonfpu)")
            co = ctx["count"][o]
            op = {"MUL": "*", "ADD": "+", "SUB": "-"}[mn]
            consts = ctx.get("consts", {})
            la = repr(float(consts[a])) if a in consts else f"{a}[e_i]"
            lb = repr(float(consts[b])) if b in consts else f"{b}[e_i]"
            tag = (" (const-folded %s%s)" % (a if a in consts else "",
                                             b if b in consts else ""))
            return (f"/* {o} = {a} {op} {b}{tag} */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                    f"  {o}[e_i] = {la} {op} {lb};",
                    None)
        if mn == "SILU":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend SILU needs F (T lives "
                                "on nonfpu)")
            co = ctx["count"][o]
            return (f"/* {o} = SILU({x}) */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                    f"  {o}[e_i] = {x}[e_i] / (1.0 + exp(-{x}[e_i]));",
                    None)
        if mn == "SELECT":
            m, a, b = args
            o = outs[0]
            co = ctx["count"][o]
            # Runtime shape contract (mask file elements vs branch count):
            # sample-S may differ from runtime-S; static baked counts lie.
            guard = f"if ({m}_N != {co}) return 21;\n  "
            if S[a][0] == "T":
                L = [f"/* {o} = SELECT({m}) triples, exact */\n  {guard}"
                     f"for (int64_t v_i = 0; v_i < {co}; ++v_i) {{"]
                for comp in ("s", "e", "z"):
                    L.append(f"  {o}_{comp}[v_i] = {m}[v_i] ? "
                             f"{a}_{comp}[v_i] : {b}_{comp}[v_i];")
                L.append("}")
                return ("\n".join(L), None)
            return (f"/* {o} = SELECT({m}) nonzero picks A */\n  {guard}"
                    f"for (int64_t v_i = 0; v_i < {co}; ++v_i)\n"
                    f"  {o}[v_i] = {m}[v_i] ? {a}[v_i] : {b}[v_i];",
                    None)
        if mn == "TSHIFT":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend TSHIFT needs F (T lives "
                                "on nonfpu)")
            d = self._we(S, x, ctx, f"TSHIFT {o}")
            return (
                f"/* {o} = TSHIFT({x}) row-max shift */\n"
                f"for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {{\n"
                f"  double mx = {x}[f_i * {d}];\n"
                f"  for (int64_t f_j = 1; f_j < {d}; ++f_j)\n"
                f"    if ({x}[f_i * {d} + f_j] > mx)\n"
                f"      mx = {x}[f_i * {d} + f_j];\n"
                f"  for (int64_t f_j = 0; f_j < {d}; ++f_j)\n"
                f"    {o}[f_i * {d} + f_j] = {x}[f_i * {d} + f_j] - mx;\n"
                f"}}",
                None)
        if mn == "SOFTMAX_WIDE":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend SOFTMAX_WIDE needs F (T lives "
                                "on nonfpu)")
            d = self._we(S, x, ctx, f"SOFTMAX_WIDE {o}")
            return (
                f"/* {o} = SOFTMAX_WIDE({x}) stable rows */\n"
                f"for (int64_t w_i = 0; w_i < {o}_N; ++w_i) {{\n"
                f"  double mx = {x}[w_i * {d}];\n"
                f"  for (int64_t w_j = 1; w_j < {d}; ++w_j)\n"
                f"    if ({x}[w_i * {d} + w_j] > mx)\n"
                f"      mx = {x}[w_i * {d} + w_j];\n"
                f"  double se = 0;\n"
                f"  for (int64_t w_j = 0; w_j < {d}; ++w_j) {{\n"
                f"    {o}[w_i * {d} + w_j] = exp({x}[w_i * {d} + w_j] - mx);\n"
                f"    se += {o}[w_i * {d} + w_j];\n"
                f"  }}\n"
                f"  for (int64_t w_j = 0; w_j < {d}; ++w_j)\n"
                f"    {o}[w_i * {d} + w_j] /= se;\n"
                f"}}",
                None)
        if mn == "CONCAT":
            a, b = args[0], args[1]
            o = outs[0]
            da, db = S[a][2][1], S[b][2][1]
            if S[a][0] == "T":
                L = [f"/* {o} = CONCAT({a},{b},1) triples, exact */"]
                for comp in ("s", "e", "z"):
                    L.append(
                        f"for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {{\n"
                        f"  memcpy(&{o}_{comp}[n_i * {da + db}], "
                        f"&{a}_{comp}[n_i * {da}], "
                        f"{da} * sizeof({o}_{comp}[0]));\n"
                        f"  memcpy(&{o}_{comp}[n_i * {da + db} + {da}], "
                        f"&{b}_{comp}[n_i * {db}], "
                        f"{db} * sizeof({o}_{comp}[0]));\n}}")
                return ("\n".join(L), None)
            return (
                f"/* {o} = CONCAT({a},{b},1) */\n"
                f"for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {{\n"
                f"  memcpy(&{o}[n_i * {da + db}], &{a}[n_i * {da}], "
                f"{da} * sizeof(double));\n"
                f"  memcpy(&{o}[n_i * {da + db} + {da}], &{b}[n_i * {db}], "
                f"{db} * sizeof(double));\n"
                f"}}",
                None)
        return super().pattern(mn, outs, args, sig, ctx)

    def epilogue(self, ctx):
        return ""


def _rows_expr(sh, ids_n):
    """Row-count value: first dim baked, batch source when dynamic, 1."""
    if not sh:
        return "1"
    if sh[0] is None:
        return "(%s)" % ids_n
    return str(int(sh[0]))


def _dim_expr(sh, ids_n):
    if not sh:
        raise AsmError("C backend: scalar streams need explicit shape")
    return " * ".join(ids_n if d is None else str(int(d)) for d in sh)



def find_consts(bound, inp, streams, sample):
    """Const-folding: single-valued F/T IN streams consumed SOLELY by
    elementwise MUL/ADD/SUB become compile-time literals (no argv file,
    no malloc). I streams never fold (ids/masks stay addressable).
    Shared by the C and CUDA drivers (same IR, same deal)."""
    _uses = {}
    _, _, _prog0 = bound
    for _outs, _mn, _fn, _args, _ln, _sg in _prog0:
        for _a in _args:
            _uses.setdefault(_a, set()).add(_mn)
    consts = {}
    for _n, _ in inp:
        if _n not in sample or _n not in streams:
            continue
        if isinstance(sample[_n], (FRef, IRef, TRef)):
            continue  # shape stub: no values to fold (weights never fold;
            # they feed MATMUL/RMSNORM, excluded by the uses-check anyway)
        _k, _sub, _sh = streams[_n]
        if _k not in ("F", "T"):
            continue
        if not all(_m in ("MUL", "ADD", "SUB") for _m in _uses.get(_n, ())):
            continue
        if _k == "F":
            _u = np.unique(np.ascontiguousarray(sample[_n], dtype=np.float64))
            if len(_u) == 1:
                consts[_n] = float(_u[0])
        else:
            _ss, _ee, _zz = (np.ascontiguousarray(p) for p in sample[_n])
            _sig = (_ss.astype(np.int64) * 1000000000000
                    + _ee.astype(np.int64) * 10 + _zz.astype(np.int64))
            if np.unique(_sig).size == 1:
                consts[_n] = (int(_ss.flat[0]), int(_ee.flat[0]),
                              int(_zz.flat[0]))
    return consts

def sanitize_cnames(inp, bound, sample, outputs, live=None):
    """Rewrite stream names to valid C identifiers at the emit seam.

    CALL expansion names streams `name#k.stream` (assembler convention,
    CPU path green); verbatim lowering emits `name#k.P_h16`, which is
    not C (nvcc dies -- siphon_geo found it; no CALL program ever
    reached a backend before). Mapping: non-word chars -> '_', leading
    digit guarded; collisions fail loud naming both streams. Only known
    stream names are rewritten (numeric-literal args like "8" pass
    through untouched). Clean-name programs map identically (zero diff
    by construction). Returns (inp, bound, sample, outputs, live,
    alias) with alias {sanitized: original} for changed names only
    (host weight-source lookup).
    """
    known = {n for n, _ in inp}
    for _outs, _mn, _fn, _args, _ln, _sg in bound:
        known.update(_outs)
    mp, seen, alias = {}, {}, {}
    for n in known:
        c = re.sub(r"\W", "_", n)
        if c and c[0].isdigit():
            c = "_" + c
        if c in seen and seen[c] != n:
            raise AsmError(f"emit: stream names {seen[c]!r} and {n!r} "
                           f"collide as C identifier {c!r} (rename one)")
        seen[c] = n
        mp[n] = c
        if c != n:
            alias[c] = n

    def _m(x):
        return mp.get(x, x) if isinstance(x, str) else x

    inp2 = [(mp[n], rest) for n, rest in inp]
    bound2 = [([mp[o] for o in _outs], _mn, _fn, [_m(a) for a in _args],
               _ln, _sg)
              for _outs, _mn, _fn, _args, _ln, _sg in bound]
    sample2 = {(mp[k] if isinstance(k, str) and k in mp else k): v
               for k, v in sample.items()} if sample is not None else None
    outputs2 = [_m(o) for o in outputs] if outputs is not None else None
    live2 = [_m(v) for v in live] if live is not None else None
    return inp2, bound2, sample2, outputs2, live2, alias


def _emit_codec_c(be, ctx, bound, streams, in_names, consts, outputs,
                  live, ids_n, alias):
    """Small, explicit codec ABI for the C backend.

    The generated translation unit exposes one opaque handle and four
    calls: create / bind / step / destroy.  Frozen inputs are read-only
    bindings, live inputs are read-write bindings, outputs are
    write-only bindings; intermediate tensors are allocated once in
    create.  Everything must be fully static, F/I only (T and dynamic
    shapes fail loud).  This is deliberately not a general streaming
    framework -- it is the boring step function that a host codec can
    drive.
    """
    prog = bound
    frepr = getattr(be, "f_repr", "double")

    def kind(n):
        return streams[n][0]

    def shape(n):
        return streams[n][2]

    def static_shape(n, where):
        sh = shape(n)
        if not sh or any(d is None for d in sh):
            raise NoPattern(
                f"C codec: {where} '{n}' needs a fully static shape, "
                f"got {sh}")
        return sh

    # -- validate and collect the three storage classes ----------------
    inter_f: set[str] = set()
    inter_i: set[str] = set()
    for outs, mn, _fn, _args, _ln, _sig in prog:
        for x in outs:
            if x in in_names or x in outputs:
                continue
            k = kind(x)
            static_shape(x, "intermediate")
            if k == "F":
                inter_f.add(x)
            elif k == "I":
                inter_i.add(x)
            else:
                raise NoPattern(
                    f"C codec: intermediate '{x}' is T; codec mode is F/I")
    for n in in_names:
        if n in consts:
            continue
        static_shape(n, "input")
        if kind(n) not in ("F", "I"):
            raise NoPattern(f"C codec: input '{n}' is T; codec mode is F/I")
    for o in outputs:
        static_shape(o, "output")
        if kind(o) not in ("F", "I"):
            raise NoPattern(f"C codec: output '{o}' is T; codec mode is F/I")

    live_set = set(live or ())
    for n in live_set:
        if n not in in_names:
            raise NoPattern(f"C codec: live input '{n}' is not an IN stream")
        if n in consts:
            raise NoPattern(
                f"C codec: live input '{n}' is const-folded")
        if n in outputs:
            raise NoPattern(f"C codec: live input '{n}' is also an output")

    bindings: list[tuple[str, int]] = []
    for n in in_names:
        if n in consts:
            continue
        bindings.append((n, 1 if n in live_set else 0))
    for o in outputs:
        bindings.append((o, 2))

    scratch = sorted(inter_f) + sorted(inter_i)

    def elem_c(n):
        return "8" if kind(n) == "I" else f"sizeof({frepr})"

    def elem_count(n):
        return _dim_expr(static_shape(n, "stream"), ids_n)

    def count_expr(n):
        return elem_count(n)

    def n_expr(n):
        # Mirrors the one-shot setter: F streams carry row counts, I
        # streams carry element counts (masks/indices).
        if kind(n) == "F":
            return _rows_expr(static_shape(n, "stream"), ids_n)
        return _dim_expr(static_shape(n, "stream"), ids_n)

    def alias_decl(n, where):
        return (f"  {('int64_t' if kind(n) == 'I' else frepr)} *{n} = "
                f"({('int64_t' if kind(n) == 'I' else frepr)} *){where};")

    parts = [be.prologue(ctx)]
    parts.append("/* small codec ABI: create / bind / step / destroy */")
    parts.append("typedef struct JotnarCodec JotnarCodec;")
    parts.append("enum { JOTNAR_BIND_FROZEN = 0, JOTNAR_BIND_LIVE = 1,"
                 " JOTNAR_BIND_OUTPUT = 2 };")
    nb = len(bindings) or 1
    ns = len(scratch) or 1
    parts.append(f"struct JotnarCodec {{ void *slots[{nb}];"
                 f" void *scratch[{ns}]; }};")
    if bindings:
        parts.append("static const char *const _codec_names[] = {%s};"
                     % ", ".join(f'"{n}"' for n, _ in bindings))
        parts.append("static const int _codec_modes[] = {%s};"
                     % ", ".join(str(m) for _, m in bindings))
        parts.append("static const size_t _codec_counts[] = {%s};"
                     % ", ".join(count_expr(n) for n, _ in bindings))
        parts.append("static const size_t _codec_sizes[] = {%s};"
                     % ", ".join(elem_c(n) for n, _ in bindings))
    else:
        parts.append("static const char *const _codec_names[1] = {0};")
        parts.append("static const int _codec_modes[1] = {0};")
        parts.append("static const size_t _codec_counts[1] = {0};")
        parts.append("static const size_t _codec_sizes[1] = {0};")
    parts.append(f"#define CODEC_NBIND {len(bindings)}")
    parts.append(f"#define CODEC_NSCRATCH {len(scratch)}")

    # create / destroy
    parts.append("void jotnar_codec_destroy(JotnarCodec *c);")
    create = ["JotnarCodec *jotnar_codec_create(void) {",
              "  JotnarCodec *c = (JotnarCodec *)calloc(1, sizeof(JotnarCodec));",
              "  if (!c) return NULL;"]
    for j, n in enumerate(scratch):
        create.append(
            f"  c->scratch[{j}] = malloc(({count_expr(n)}) * {elem_c(n)});"
        )
        create.append(
            f"  if (!c->scratch[{j}]) {{ jotnar_codec_destroy(c);"
            f" return NULL; }}")
    create.append("  return c;")
    create.append("}")
    parts.append("\n".join(create))

    destroy = ["void jotnar_codec_destroy(JotnarCodec *c) {",
               "  if (!c) return;"]
    for j in range(len(scratch)):
        destroy.append(f"  free(c->scratch[{j}]);")
    destroy.append("  free(c);")
    destroy.append("}")
    parts.append("\n".join(destroy))

    # bind
    bind = [
        "int jotnar_codec_bind(JotnarCodec *c, const char *name,"
        " void *data, size_t bytes, int wmode) {",
        "  if (!c || !name) return -1;",
        "  for (int i = 0; i < CODEC_NBIND; ++i) {",
        "    if (strcmp(_codec_names[i], name) != 0) continue;",
        "    if (wmode != _codec_modes[i]) return -2;",
        "    if (bytes != _codec_counts[i] * _codec_sizes[i]) return -3;",
        "    c->slots[i] = data;",
        "    return i;",
        "  }",
        "  return -4;",
        "}",
    ]
    parts.append("\n".join(bind))

    # step
    step = ["int jotnar_codec_step(JotnarCodec *c) {",
            "  if (!c) return -1;",
            "  for (int i = 0; i < CODEC_NBIND; ++i)",
            "    if (!c->slots[i]) return 10 + i;"]
    for i, (n, _m) in enumerate(bindings):
        step.append(alias_decl(n, f"c->slots[{i}]"))
    for j, n in enumerate(scratch):
        step.append(alias_decl(n, f"c->scratch[{j}]"))
    for n, _m in bindings:
        step.append(f"  int64_t {n}_N = ({n_expr(n)}); (void){n}_N;")
    for n in scratch:
        step.append(f"  int64_t {n}_N = ({n_expr(n)}); (void){n}_N;")
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        step.append(f"  /* L{_short_site(ln)}: {mn} */")
        step.append("  " + pat.replace("\n", "\n  "))
    step.append("  return 0;")
    step.append("}")
    parts.append("\n".join(step))
    parts.append(be.epilogue(ctx))
    source = "\n".join(parts)

    meta = {
        "symbol": "jotnar_codec",
        "bindings": [
            {
                "name": n,
                "mode": m,
                "kind": kind(n),
                "shape": tuple(int(d) for d in static_shape(n, "stream")),
                "bytes": int(np.prod(static_shape(n, "stream"),
                                     dtype=np.int64))
                * (8 if kind(n) == "I" else 8),
            }
            for n, m in bindings
        ],
        "inputs": list(in_names),
        "outputs": list(outputs),
        "live": sorted(live_set),
        "scratch": scratch,
    }
    return source, meta


def compile_program(text, target="c", sample=None, outputs=None,
                    registry=None, sigs=None, basedir=".", origin=None,
                    use_blas=False, time_ops=False, use_fp16=False,
                    live=None, sync_each=True, graph=False,
                    pack_outputs=False, mode=None):
    """Frontend entry: asm text -> target source. Returns dict with
    source/backend/streams/dims/config. Sample payload required (shapes).
    live: persistent-loop live inputs (C/CUDA; None = one-shot). The C
    loop mirrors chain/serve.ServedExe: frozen inputs load once, live
    inputs re-read in place (fixed shape), outputs are rewritten, then
    READY is printed for the host.
    sync_each: per-op device-sync (CUDA only; False trusts same-stream
    ordering).
    graph: capture the step into a CUDA graph (CUDA only; needs live,
    refuses time_ops, forces sync_each=False).
    pack_outputs: C serve only. Write all output streams back-to-back
    into one argv file per step instead of one file per output. The
    host slices by output order; this removes the per-position file
    open/read tax from chunked resident models.
    mode: execution contract. None is the default one-shot/serve path;
    "codec" emits the small C codec ABI (create/bind/step/destroy) for
    fully static F/I programs, with live=[...] naming read-write
    bindings and everything else read-only or output. C only for now.
    """
    if mode not in (None, "codec"):
        raise AsmError(f"unknown mode {mode!r}")
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    config, inp, bound, _ = assemble(text, registry or _R, sigs or _S,
                                     basedir=basedir, origin=origin)
    inp, bound, sample, outputs, live, _alias = sanitize_cnames(
        inp, bound, sample, outputs, live)
    from chain.backends import get_backend as _gb
    _be = _gb(target)  # unknown names fail loud here
    _be.use_blas = use_blas
    _be.use_fp16 = use_fp16
    if graph and _be.name != "cuda":
        raise NoPattern(f"target '{target}': graph is CUDA-only")
    if not sync_each and _be.name != "cuda":
        raise NoPattern(f"target '{target}': sync_each=False is CUDA-only")
    if mode == "codec":
        if _be.name != "c":
            raise NoPattern(
                f"target '{target}': codec mode is C-only for now")
        if graph or time_ops or pack_outputs:
            raise NoPattern(
                "codec mode does not combine with graph/time_ops/"
                "pack_outputs")
    if live is not None and _be.name not in ("c", "cuda"):
        # Serve mode is implemented by the resident C and CUDA loops.
        # Other backends stay one-shot: refuse loudly rather than
        # silently dropping live= (a dropped request would run once and
        # look served).
        raise NoPattern(
            f"target '{target}': live= serve mode is C/CUDA only "
            f"(got live={live})")
    if pack_outputs and (_be.name != "c" or live is None):
        raise NoPattern(
            f"target '{target}': pack_outputs is C serve-only "
            f"(target='c', live=[...])")
    if _be.name == "cuda":
        from chain.emit_cuda import compile_cuda as _cc
        return _cc(text, sample=sample, outputs=outputs,
                   registry=registry, sigs=sigs, basedir=basedir,
                   origin=origin,
                   use_fp16=getattr(_be, "use_fp16", False),
                   time_ops=time_ops, live=live, sync_each=sync_each,
                   graph=graph)
    if not _be.emits:
        # stub backend: surface its extension recipe, don't compile air.
        raise NoPattern(getattr(_be, "stub_note",
                                f"no emitting '{target}' backend yet"))
    if sample is None:
        raise AsmError("compile_program needs sample payload (shapes)")
    if not outputs:
        raise AsmError("compile_program needs outputs=[...] (OUT streams)")
    be = _be
    _frepr = getattr(be, "f_repr", "double")
    streams = infer_shapes((config, inp, bound), sample)
    consts = find_consts((config, inp, bound), inp, streams, sample)
    in_names = [n for n, _ in inp]
    if live is not None:
        live = list(live)
        for n in live:
            if n not in in_names:
                raise NoPattern(f"C serve: live input '{n}' is not an IN stream")
            if n in consts:
                raise NoPattern(
                    f"C serve: live input '{n}' const-folded "
                    f"(no argv file to re-read)")
            if n in outputs:
                raise NoPattern(f"C serve: live input '{n}' is an output")
            k, _, sh = streams[n]
            if k == "T":
                raise NoPattern(
                    f"C serve: live triple input '{n}' is not supported "
                    f"(v0.1; live inputs are F/I)")
            if sh is None or any(d is None for d in sh):
                raise NoPattern(
                    f"C serve: live input '{n}' needs fully static shape "
                    f"(fixed geometry), got {sh}")
    if pack_outputs:
        for o in outputs:
            if streams[o][0] == "T":
                raise NoPattern(
                    f"C serve pack_outputs: triple output '{o}' is not "
                    f"supported (v0.1; packed outputs are F/I)")
    dims = {}
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "T" and sh and all(d is not None for d in sh):
            for ax, d in enumerate(sh):
                dims[f"D{n}_{ax}"] = d
            if len(sh) == 2:
                dims[f"V_{n}"], dims[f"C_{n}"] = sh
    # batch-ids stream: first (N,) I input (drives dynamic dims);
    # else a dynamic-row F input (rows from file bytes / baked width).
    ids_n = None
    for n in in_names:
        k, _, sh = streams[n]
        if k == "I" and len(sh) == 1:
            ids_n = f"{n}_N"
            break
    if ids_n is None:
        for n in in_names:
            k, _, sh = streams[n]
            if k == "F" and sh and sh[0] is None and sh[-1] is not None:
                ids_n = f"{n}_N"
                break
    if ids_n is None:
        ids_n = "1"
    # rotary theta constants from head dim (baked by construction)
    theta = []
    for _on, _mn, _fn, _aa, _ln, _sg in bound:
        if _mn == "ROTARY":
            from chain.asm_ops import rope_tables  # noqa
            d = streams[_aa[0]][2][1]
            base = float(config.get("rope_base", 10000.0))
            th = np.power(float(base),
                          -2.0 * np.arange(d // 2, dtype=np.float64) / d)
            theta = [float(t) for t in th]
            break
    rope = None
    for _on, _mn, _fn, _aa, _ln, _sg in bound:
        if _mn == "ROTARY" and len(_aa) > 1 and _aa[1] in sample:
            _d = streams[_aa[0]][2][-1]
            if _d is None:
                continue
            if isinstance(sample[_aa[1]], (FRef, IRef, TRef)):
                raise AsmError(
                    f"ROTARY pos '{_aa[1]}' is a shape stub: rope tables "
                    f"need real (small) position values, not a stub")
            _pos = np.ascontiguousarray(sample[_aa[1]], dtype=np.int64)
            _P = int(_pos.max(initial=0)) + 1
            from chain.asm_ops import rope_tables as _rt
            _base = float(config.get("rope_base", 10000.0))
            _ct, _st = _rt(_P, _d, _base)
            rope = {"P": _P, "D": _d, "cos": np.ascontiguousarray(_ct),
                    "sin": np.ascontiguousarray(_st)}
            break
    ctx = {"streams": streams, "dims": dims, "dyn": {}, "ids_n": ids_n,
           "inputs": in_names, "outputs": outputs, "gathered": {},
           "config": dict(config), "theta": theta, "rope": rope,
           "count": {}, "mns": [mn for _, mn, _, _, _, _ in bound],
           "consts": consts}
    for _xn, (_xk, _xs, _xsh) in streams.items():
        if _xk in ("F", "T") and _xsh:
            ctx["count"][_xn] = "(%s)" % _dim_expr(_xsh, ids_n)
        elif _xk == "I" and _xsh and all(d is not None for d in _xsh):
            # integer intermediates (GT masks etc.): same COUNT treatment
            # as F/T so patterns share one loop-bound source (outputs keep
            # their legacy branch below unchanged).
            ctx["count"][_xn] = "(%s)" % int(
                np.prod(list(_xsh), dtype=np.int64))
        elif _xk == "I" and _xn in outputs:
            _osh = _xsh
            if _osh and all(d is not None for d in _osh):
                ctx["count"][_xn] = "(%s)" % int(
                    np.prod(list(_osh), dtype=np.int64))
            else:
                ctx["count"][_xn] = "(%s)" % ids_n
    if mode == "codec":
        source, codec_meta = _emit_codec_c(
            be, ctx, bound, streams, in_names, consts, outputs,
            live, ids_n, _alias)
        return {"source": source, "backend": be, "streams": streams,
                "dims": dims, "config": dict(config), "inputs": in_names,
                "outputs": outputs, "n_argv": 0, "libs": ["-lm"],
                "consts": consts, "cname": _alias,
                "live": list(live) if live is not None else None,
                "graph": False, "pack_outputs": False,
                "codec": codec_meta}
    parts = [be.prologue(ctx)]
    parts.append("/* inputs: argv files, order below */")
    decls = []
    for n in in_names:
        if n in consts:
            continue  # compile-time literal: no storage, no argv file
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            decls.append(f"static {st} *{n}_s; static {et} *{n}_e;"
                         f" static {zt} *{n}_z; static int64_t {n}_N;")
        elif k == "F":
            decls.append(f"static {_frepr} *{n}; static int64_t {n}_N;")
        else:
            decls.append(f"static int64_t *{n}; static int64_t {n}_N;")
    for o in outputs:
        ok, osub, osh = streams[o]
        if ok == "F":
            decls.append(f"static {_frepr} *{o};")
        elif ok == "T":
            st, et, zt = osub
            decls.append(f"static {st} *{o}_s; static {et} *{o}_e;"
                         f" static {zt} *{o}_z; static int64_t {o}_N;")
        else:
            decls.append(f"static int64_t *{o}; static int64_t {o}_N;")
    parts.append("\n".join(decls))
    # intermediate declarations (F/T streams malloc'd per op in-body)
    inter = set()
    interT = {}
    interI = set()
    _, _, prog = (config, inp, bound)
    for outs, _mn2, _fn2, _aa2, _ln2, _sg2 in prog:
        for x in outs:
            if x not in in_names and x not in outputs:
                k, ksub, _ = streams[x]
                if k == "F":
                    inter.add(x)
                elif k == "T":
                    interT[x] = ksub
                elif k == "I":
                    interI.add(x)
    if inter:
        parts.append("/* intermediates */")
        parts.append("\n".join(f"{_frepr} *{x};" for x in sorted(inter)))
    if interT:
        parts.append("/* triple intermediates */")
        for x in sorted(interT):
            st, et, zt = interT[x]
            parts.append(f"{st} *{x}_s; {et} *{x}_e; {zt} *{x}_z;")
    if interI:
        parts.append("/* integer intermediates */")
        parts.append("\n".join(f"int64_t *{x};" for x in sorted(interI)))
    if inter:
        parts.append("/* intermediates */")
        parts.append("\n".join(f"{_frepr} *{x};" for x in sorted(inter)))
    if interT:
        parts.append("/* triple intermediates */")
        for x in sorted(interT):
            st, et, zt = interT[x]
            parts.append(f"{st} *{x}_s; {et} *{x}_e; {zt} *{x}_z;")
    parts.append("int main(int argc, char **argv) {")
    live_ins = [n for n in in_names if n not in consts]
    n_argv = sum(3 if streams[n][0] == "T" else 1 for n in live_ins)
    n_argv += (1 if pack_outputs else
               sum(3 if streams[o][0] == "T" else 1 for o in outputs))
    parts.append(f"  if (argc != {n_argv + 1}) return 9;")
    idx = 1
    argidx = {}
    loads = []
    for n in in_names:
        if n in consts:
            continue
        argidx[n] = idx
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            if not sh or any(d is None for d in sh):
                raise NoPattern(f"C backend: T input '{n}' must be fully "
                                f"baked {sh}")
            cnt = " * ".join(f"DIM_D{n}_{ax}" for ax in range(len(sh)))
            loads.append(f"  {n}_N = {f'DIM_D{n}_0' if len(sh) else '1'}; "
                         f"/* rows */")
            loads.append(f"  /* load {n} triples ({' x '.join(str(d) for d in sh)}) */")
            for comp, ct in (("s", st), ("e", et), ("z", zt)):
                loads.append(
                    f"  {{ FILE *f = fopen(argv[{idx}], \"rb\");"
                    f" if (!f) return 1;\n"
                    f"    {n}_{comp} = malloc(({cnt})"
                    f" * sizeof({ct}));"
                    f" if (fread({n}_{comp}, sizeof({ct}), ({cnt}), f)"
                    f" != (size_t)({cnt})) return 2;"
                    f" fclose(f); }}")
                idx += 1
        elif k == "F":
            _k2, _s2, _sh2 = streams[n]
            if _sh2 and _sh2[-1] is None:
                raise NoPattern(f"C backend: fully-dynamic F input '{n}'")
            _wr = f"({_sh2[-1]} * 8)" if _sh2 else "8"
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); long {n}_B = ftell(f);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n}_N = {_sh2[0] if _sh2 and _sh2[0] is not None else f'{n}_B / {_wr}'};"
                f"\n    {n} = malloc({n}_B);"
                f" if (fread({n}, 1, {n}_B, f) != (size_t){n}_B) return 2;"
                f" fclose(f); }}")
            idx += 1
        else:
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); {n}_N = ftell(f)/sizeof(int64_t);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n} = malloc({n}_N * sizeof(int64_t));"
                f" if (fread({n}, 8, {n}_N, f) != (size_t){n}_N) return 2;"
                f" fclose(f); }}")
            idx += 1
    # allocate intermediates + outputs (COUNT exprs; _N vars per stream)
    allocs = []
    for x in sorted(interI):
        _k, _s, xsh = streams[x]
        cnt = _dim_expr(xsh, ids_n)
        allocs.append(f"  {x} = malloc(({cnt}) * sizeof(int64_t));")
        # NOTE: _N is ELEMENT count here (not rows): file-backed I masks
        # define _N as elements, and SELECT's guard compares mask _N
        # against branch count -- same units or fail loud (rc 21).
        allocs.append(f"  int64_t {x}_N = ({cnt});")
    for x in sorted(inter) + [o for o in outputs if streams[o][0] == "F"]:
        _k, _s, xsh = streams[x]
        cnt = _dim_expr(xsh, ids_n)
        allocs.append(f"  {x} = malloc(({cnt}) * sizeof({_frepr}));")
        # per-stream ROW-count var for patterns (<stream>_N)
        allocs.append(f"  int64_t {x}_N = {_rows_expr(xsh, ids_n)};")
    for o in outputs:
        if streams[o][0] == "I":
            _osh = streams[o][2]
            # ELEMENT count (not rows): matches file-load _N semantics
            # and SELECT's guard; rows==elements only held for 1D (ARGMAX
            # luck). 2D masks need full count (else heap overflow).
            _cnt = _dim_expr(_osh, ids_n)
            allocs.append(f"  {o}_N = ({_cnt}); "
                          f"{o} = malloc(({_cnt}) * 8);")
    for x in sorted(interT) + [o for o in outputs if streams[o][0] == "T"]:
        _k, _tsub, xsh = streams[x]
        _st, _et, _zt = _tsub
        cnt = _dim_expr(xsh, ids_n)
        for _comp, _ct in (("s", _st), ("e", _et), ("z", _zt)):
            allocs.append(f"  {x}_{_comp} = malloc(({cnt}) * sizeof({_ct}));")
        allocs.append(f"  int64_t {x}_N = {_rows_expr(xsh, ids_n)};")

    # -- resident (serve) helpers ------------------------------------
    # Live inputs are fixed geometry: compile-time element count, so a
    # short/long rewrite fails loud before touching the buffer. Frozen
    # inputs load once; live inputs are re-read in place; shadowed
    # inputs (an IN also used as an OUT) are restored from their startup
    # files before READY, matching the CUDA warmup contract.
    def _live_elems(n):
        return int(np.prod(list(streams[n][2]), dtype=np.int64))

    def _live_elem_size(n):
        return f"sizeof({_frepr})" if streams[n][0] == "F" else "sizeof(int64_t)"

    def _size_check(n, ind):
        i = argidx[n]
        cnt = _live_elems(n)
        el = _live_elem_size(n)
        return "\n".join([
            f"{ind}{{ FILE *f = fopen(argv[{i}], \"rb\"); if (!f) return 1;",
            f"{ind}  fseek(f, 0, SEEK_END); long _B = ftell(f);"
            f" fseek(f, 0, SEEK_SET);",
            f"{ind}  if (_B != (long)({cnt} * {el})) return 10;",
            f"{ind}  fclose(f); }}",
        ])

    def _reload(n, ind):
        i = argidx[n]
        cnt = _live_elems(n)
        el = _live_elem_size(n)
        return "\n".join([
            f"{ind}{{ FILE *f = fopen(argv[{i}], \"rb\"); if (!f) return 1;",
            f"{ind}  fseek(f, 0, SEEK_END); long _B = ftell(f);"
            f" fseek(f, 0, SEEK_SET);",
            f"{ind}  if (_B != (long)({cnt} * {el})) return 10;",
            f"{ind}  if (fread({n}, 1, _B, f) != (size_t)_B) return 2;"
            f" fclose(f); }}",
        ])

    shadowed = []
    if live is not None:
        _seen_out = set()
        for _outs, _mn, _fn, _args, _ln, _sig in prog:
            _seen_out.update(_outs)
        for n in in_names:
            if n in consts or n not in _seen_out:
                continue
            k, _, sh = streams[n]
            if k == "T":
                raise NoPattern(
                    f"C serve: shadowed triple input '{n}' is not "
                    f"supported (v0.1)")
            if sh is None or any(d is None for d in sh):
                raise NoPattern(
                    f"C serve: shadowed input '{n}' needs fully static "
                    f"shape (fixed geometry), got {sh}")
            shadowed.append(n)

    compute = []
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        body = pat.replace("\n", "\n  ")
        if time_ops:
            body = ("{ clock_t _t0 = clock();\n  " + body +
                    f"\n  fprintf(stderr, \"op {mn} %f\\n\", "
                    f"(double)(clock() - _t0) / CLOCKS_PER_SEC); }}")
        compute.append(f"  /* L{_short_site(ln)}: {mn} */\n  " + body)

    stores = []
    if pack_outputs:
        pack = [f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                f" return 3;"]
        for o in outputs:
            ok = streams[o][0]
            el = "8" if ok == "I" else f"sizeof({_frepr})"
            co = ctx["count"][o]
            pack.append(f"    fwrite({o}, {el}, {co}, f);")
        pack.append("    fclose(f); }")
        stores.append("\n".join(pack))
        idx += 1
    else:
        for o in outputs:
            ok, osub, _osh2 = streams[o]
            if ok == "T":
                st, et, zt = osub
                co = ctx["count"][o]
                for comp, ct in (("s", st), ("e", et), ("z", zt)):
                    stores.append(
                        f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                        f" return 3;\n"
                        f"    fwrite({o}_{comp}, sizeof({ct}), {co}, f);"
                        f" fclose(f); }}")
                    idx += 1
                continue
            el = "8" if ok == "I" else f"sizeof({_frepr})"
            co = ctx["count"][o]
            stores.append(f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                          f" return 3;\n"
                          f"    fwrite({o}, {el}, {co}, f); fclose(f); }}")
            idx += 1

    if live is None:
        # One-shot output is kept byte-identical to the pre-serve
        # emitter: same loads, allocs, compute, stores, return.
        parts.append("\n".join(loads))
        parts.append("\n".join(allocs))
        parts.extend(compute)
        parts.extend(stores)
        parts.append("  return 0;\n}")
    else:
        parts.append("\n".join(loads))
        for n in live:
            parts.append(_size_check(n, "  "))
        parts.append("\n".join(allocs))
        parts.extend(compute)
        parts.extend(stores)
        for n in shadowed:
            parts.append(_reload(n, "  "))
        parts.append('  printf("READY\\n"); fflush(stdout);')
        parts.append("  { char line[16];")
        parts.append("    while (fgets(line, sizeof(line), stdin)) {")
        parts.append("      if (line[0] == 'q') break;")
        for n in live:
            parts.append(_reload(n, "    "))
        for c in compute:
            parts.append("  " + c)
        for s in stores:
            parts.append("  " + s)
        parts.append('      printf("READY\\n"); fflush(stdout);')
        parts.append("    } }")
        parts.append("  return 0;\n}")
    parts.append(be.epilogue(ctx))
    src = "\n".join(parts)
    _libs = ["-lm"]
    if getattr(be, "use_blas", False) and be.name == "c":
        _libs = _libs + list(CBackend.BLAS_LIBS)
    return {"source": src, "backend": be, "streams": streams,
            "dims": dims, "config": dict(config), "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1, "libs": _libs,
            "consts": consts, "cname": _alias,
            "live": list(live) if live is not None else None,
            "graph": False, "pack_outputs": bool(pack_outputs)}


def build(source, workdir, name="prog", cc="cc", cflags=None, libs=None):
    """Compile emitted source -> exe path. Backend owns its speed flags."""
    cflags = list(cflags or CBackend.DEFAULT_CFLAGS)
    libs = list(libs or ["-lm"])
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".c")
    exe = os.path.join(workdir, name)
    with open(src, "w") as f:
        f.write(source)
    r = subprocess.run([cc] + cflags + [src, "-o", exe] + libs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"C build failed:\n{r.stderr}")
    return exe


def build_shared(source, workdir, name="codec", cc="cc", cflags=None,
                 libs=None):
    """Compile emitted codec source -> shared library path."""
    cflags = list(cflags or CBackend.DEFAULT_CFLAGS)
    libs = list(libs or ["-lm"])
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".c")
    so = os.path.join(workdir, name + ".so")
    with open(src, "w") as f:
        f.write(source)
    r = subprocess.run(
        [cc] + cflags + ["-fPIC", "-shared", src, "-o", so] + libs,
        capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"C shared build failed:\n{r.stderr}")
    return so
