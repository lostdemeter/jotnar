"""DPT-tiny builder-constructed listing (fiber doctrine, all tabled ops).

Every stream is a fiber (rows x cols); NO reshape anywhere (views
carry no opcode). 3x3 convs run GATHER-tap form (Phase-5 precedent:
9 GATHER + 9 MATMUL + 8 ADD per conv, edge-replicate baked ids) —
mathematically identical to P1 CONV (equivalence probed in the
gate). Upsampling runs nearest-GATHER broadcast (P2 degraded
default, as designed). LayerNorm (A1), GELU-form via BETA/TBETA
consts (A2 note: SILU(MUL(x,1.702))/1.702), unfused per-head
attention (A4: BMMV math covered; explicit MATMUL form here).
Attention scale is baked into Wq (documented standard practice —
no third const needed; BETA/TBETA are both spent on GELU-form).
All Linears bias-free (v0.1 scope, stated).

Grid convention (single, everywhere): token/row idx = iy*C + ix
with iy row 0..R-1, ix col 0..C-1. Tokens (42,64)=2688;
s32 (21,32)=672; s8 (84,128)=10752; s4 (168,256)=43008;
full (672,1024)=688128.
"""
import numpy as np

H, W = 672, 1024
P = 16
GR, GC = 42, 64
T = GR * GC
WD, NH, DH, NB = 128, 4, 32, 6
SCALES = {"s32": (21, 32), "s16": (42, 64), "s8": (84, 128), "s4": (168, 256)}
GELU_C1, GELU_C2 = 1.702, 1.0 / 1.702


def _ids_patch():
    """256 taps x (2688,) int64: tap(dy,dx)[iy*64+ix] = flat RGB index."""
    taps = []
    for dy in range(P):
        for dx in range(P):
            ids = np.empty(T, np.int64)
            k = 0
            for iy in range(GR):
                for ix in range(GC):
                    ids[k] = (iy * P + dy) * W + (ix * P + dx)
                    k += 1
            taps.append(ids)
    return taps


def _ids_stride(fine, coarse):
    """fine(r,c) <- coarse(r//f, c//f); shapes (fr,fc),(cr,cc)."""
    (fr, fc), (cr, cc) = fine, coarse
    f = fr // cr
    assert fr == cr * f and fc == cc * f, (fine, coarse)
    ids = np.empty(fr * fc, np.int64)
    k = 0
    for iy in range(fr):
        for ix in range(fc):
            ids[k] = (iy // f) * cc + (ix // f)
            k += 1
    return ids


def _ids_down(fine, coarse):
    """fine(r,c) <- coarse(2r, 2c); strided subselect, shapes (fr,fc),(cr,cc)."""
    (fr, fc), (cr, cc) = fine, coarse
    assert cr == fr * 2 and cc == fc * 2, (fine, coarse)
    ids = np.empty(fr * fc, np.int64)
    k = 0
    for iy in range(fr):
        for ix in range(fc):
            ids[k] = (2 * iy) * cc + (2 * ix)
            k += 1
    return ids


def _ids_conv_taps(gh, gw):
    """9 taps x (gh*gw,) int64, edge-replicate baked ids."""
    taps = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ids = np.empty(gh * gw, np.int64)
            k = 0
            for iy in range(gh):
                for ix in range(gw):
                    jy = min(max(iy + dy, 0), gh - 1)
                    jx = min(max(ix + dx, 0), gw - 1)
                    ids[k] = jy * gw + jx
                    k += 1
            taps.append(ids)
    return taps


def gconv(p, z, tap_ids, Wtaps, tag):
    """Full 3x3 (edge-replicate baked taps): 9 GATHER + 9 MATMUL + ADD tree."""
    acc = None
    for ti, (ids, wnm) in enumerate(zip(tap_ids, Wtaps)):
        g = p.op("GATHER", z, ids, out=f"{tag}g{ti}")
        m = p.op("MATMUL", g, wnm, out=f"{tag}m{ti}")
        acc = m if acc is None else p.op("ADD", acc, m, out=f"{tag}a{ti}")
    return acc


def rcu(p, z, tap_ids, W0taps, W1taps, tag):
    a = gconv(p, z, tap_ids, W0taps, tag + "c0")
    s = p.op("SILU", a, out=tag + "s")
    b = gconv(p, s, tap_ids, W1taps, tag + "c1")
    return p.op("ADD", z, b, out=tag)


def block(p, x, Wq, Wk, Wv, Wo, W1, W2, tag):
    n = p.op("LAYERNORM", x, "lnw", "lnb", out=tag + "n1")
    q = p.op("MATMUL", n, Wq, out=tag + "q")
    k = p.op("MATMUL", n, Wk, out=tag + "k")
    v = p.op("MATMUL", n, Wv, out=tag + "v")
    outs = []
    for h in range(NH):
        qh = p.op("SLICE", q, 1, h * DH, (h + 1) * DH, out=f"{tag}q{h}")
        kh = p.op("SLICE", k, 1, h * DH, (h + 1) * DH, out=f"{tag}k{h}")
        vh = p.op("SLICE", v, 1, h * DH, (h + 1) * DH, out=f"{tag}v{h}")
        kt = p.op("TRANSPOSE", kh, out=f"{tag}t{h}")
        sc = p.op("MATMUL", qh, kt, out=f"{tag}s{h}")
        pr = p.op("SOFTMAX_WIDE", sc, out=f"{tag}p{h}")
        outs.append(p.op("MATMUL", pr, vh, out=f"{tag}o{h}"))
    ctx = outs[0]
    for hi, o in enumerate(outs[1:]):
        ctx = p.op("CONCAT", ctx, o, 1, out=tag + f"cc{hi}")
    co = p.op("MATMUL", ctx, Wo, out=tag + "co")
    x = p.op("ADD", x, co, out=tag + "r1")
    h = p.op("LAYERNORM", x, "lnw", "lnb", out=tag + "n2")
    m = p.op("MATMUL", h, W1, out=tag + "m1")
    g1 = p.op("MUL", m, p.op("BETA", m, out=tag + "b1"), out=tag + "g1")
    gs = p.op("SILU", g1, out=tag + "gs")
    gm = p.op("MUL", gs, p.op("TBETA", gs, out=tag + "b2"), out=tag + "g2")
    m2 = p.op("MATMUL", gm, W2, out=tag + "m2")
    return p.op("ADD", x, m2, out=tag + "r2")


def build_dpttiny():
    """Returns (Prog, id_dict) — ids computed here, fed by the gate."""
    from chain.builder import Prog
    p = Prog("dpt-tiny")
    # fixed-point scales: builder_block convention (m_acc/m_cov override
    # the coarse file defaults; file scales give 12dB on O(1) matmuls,
    # these give 63dB — measured, documented, same values as headt).
    p.config("m_acc", 36118).config("m_cov", 35048)
    p.config("eps_ln", "1e-5").config("beta", GELU_C1).config("beta_b", GELU_C2)
    ids = {"patch": _ids_patch(),
           "dn16_32": _ids_down((21, 32), (42, 64)),
           "up16_8": _ids_stride((84, 128), (42, 64)),
           "up16_4": _ids_stride((168, 256), (42, 64)),
           "up8_4": _ids_stride((168, 256), (84, 128)),
           "up4_full": _ids_stride((672, 1024), (168, 256)),
           "up32_16": _ids_stride((42, 64), (21, 32)),
           "conv32": _ids_conv_taps(21, 32),
           "conv16": _ids_conv_taps(42, 64),
           "conv8": _ids_conv_taps(84, 128),
           "conv4": _ids_conv_taps(168, 256)}
    p.inp("RGB")
    for ti in range(256):
        p.inp(f"ptap{ti}")
    for nm in ("pos", "lnw", "lnb", "pemb"):
        p.inp(nm)
    for bi in range(NB):
        for nm in ("wq", "wk", "wv", "wo", "w1", "w2"):
            p.inp(f"b{bi}{nm}")
    for nm in ("r32", "r16", "r8", "r4"):
        p.inp(nm)
    for li in range(4):
        for ri in range(2):
            for ci in range(2):
                for ti in range(9):
                    p.inp(f"f{li}r{ri}c{ci}t{ti}")
    p.inp("hp", "hl")
    for nm in ("up16_8", "up16_4", "dn16_32", "up8_4", "up4_full", "up32_16",
               "conv32", "conv16", "conv8", "conv4"):
        if nm.startswith("conv"):
            for ti in range(9):
                p.inp(f"{nm}t{ti}")
        else:
            p.inp(nm)
    # -- patchify: 256 taps + concat + proj + pos -------------------------
    tp = [p.op("GATHER", "RGB", f"ptap{ti}", out=f"pt{ti}") for ti in range(256)]
    acc = tp[0]
    for t in tp[1:]:
        acc = p.op("CONCAT", acc, t, 1)
    e = p.op("MATMUL", acc, "pemb", out="E")
    x = p.op("ADD", e, "pos", out="X0")
    # -- encoder blocks ----------------------------------------------------
    tap_out = {}
    for bi in range(NB):
        x = block(p, x, f"b{bi}wq", f"b{bi}wk", f"b{bi}wv", f"b{bi}wo",
                  f"b{bi}w1", f"b{bi}w2", tag=f"B{bi}")
        if bi in (1, 3, 4, 5):
            tap_out[bi] = x
    # -- reassemble (strided/broadcast GATHER + 1x1 proj) -------------------
    r32 = p.op("GATHER", tap_out[1], "dn16_32", out="g32")
    m32 = p.op("MATMUL", r32, "r32", out="m32")
    m16 = p.op("MATMUL", tap_out[3], "r16", out="m16")
    g8 = p.op("GATHER", tap_out[4], "up16_8", out="g8")
    m8 = p.op("MATMUL", g8, "r8", out="m8")
    g4 = p.op("GATHER", tap_out[5], "up16_4", out="g4")
    m4 = p.op("MATMUL", g4, "r4", out="m4")
    # -- fusion coarse-to-fine ----------------------------------------------
    f32 = rcu(p, m32, [f"conv32t{ti}" for ti in range(9)],
              [f"f0r0c0t{ti}" for ti in range(9)],
              [f"f0r0c1t{ti}" for ti in range(9)], tag="F0a")
    f32 = rcu(p, f32, [f"conv32t{ti}" for ti in range(9)],
              [f"f0r1c0t{ti}" for ti in range(9)],
              [f"f0r1c1t{ti}" for ti in range(9)], tag="F0b")
    up = p.op("GATHER", f32, "up32_16", out="u16")
    f16 = rcu(p, m16, [f"conv16t{ti}" for ti in range(9)],
              [f"f1r0c0t{ti}" for ti in range(9)],
              [f"f1r0c1t{ti}" for ti in range(9)], tag="F1a")
    f16 = p.op("ADD", f16, up, out="F1f")
    f16 = rcu(p, f16, [f"conv16t{ti}" for ti in range(9)],
              [f"f1r1c0t{ti}" for ti in range(9)],
              [f"f1r1c1t{ti}" for ti in range(9)], tag="F1b")
    up = p.op("GATHER", f16, "up16_8", out="u8")
    f8 = rcu(p, m8, [f"conv8t{ti}" for ti in range(9)],
             [f"f2r0c0t{ti}" for ti in range(9)],
             [f"f2r0c1t{ti}" for ti in range(9)], tag="F2a")
    f8 = p.op("ADD", f8, up, out="F2f")
    f8 = rcu(p, f8, [f"conv8t{ti}" for ti in range(9)],
             [f"f2r1c0t{ti}" for ti in range(9)],
             [f"f2r1c1t{ti}" for ti in range(9)], tag="F2b")
    up = p.op("GATHER", f8, "up8_4", out="u4")
    f4 = rcu(p, m4, [f"conv4t{ti}" for ti in range(9)],
             [f"f3r0c0t{ti}" for ti in range(9)],
             [f"f3r0c1t{ti}" for ti in range(9)], tag="F3a")
    f4 = p.op("ADD", f4, up, out="F3f")
    f4 = rcu(p, f4, [f"conv4t{ti}" for ti in range(9)],
             [f"f3r1c0t{ti}" for ti in range(9)],
             [f"f3r1c1t{ti}" for ti in range(9)], tag="F3b")
    # -- head: proj at 1/4, nearest-up full, linear out ---------------------
    hp = p.op("MATMUL", f4, "hp", out="HP")
    big = p.op("GATHER", hp, "up4_full", out="BIG")
    p.op("MATMUL", big, "hl", out="OUT")
    return p, ids
