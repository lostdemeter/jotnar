"""Python frontend builder v0.1: geometric assembly without hand-writing it.

Prog accumulates IN/CONFIG/IMPORT/DEF/op/CALL lines and emits asm text
(or compiles straight through). Stream names are explicit (like the
listings) or auto (t1, t2, ...) when out=None. Literals pass through
as int/float args. The assembler remains the authority: builder output
is ordinary text, gated by assemble/run agreement, never trusted raw.

Composite helpers (the stdlib seed) belong here as plain functions
over Prog -- see attn_head() below. More arrive by writing them, not
by extending the class.
"""
from chain.asm import AsmError


class Prog:
    def __init__(self, name="<?>"):
        self.name = name
        self._config = []
        self._imports = []
        self._ins = []
        self._lines = []
        self._auto = 0

    # -- declarations ------------------------------------------------
    def config(self, k, v):
        self._config.append((k, v))
        return self

    def imp(self, path):
        self._imports.append(path)
        return self

    def inp(self, *names):
        for n in names:
            if n in self._ins:
                raise AsmError(f"builder: duplicate IN '{n}'")
            self._ins.append(n)
        return self

    # -- code ----------------------------------------------------------
    def _fresh(self, hint="t"):
        self._auto += 1
        return f"{hint}{self._auto}"

    @staticmethod
    def _render(a):
        if isinstance(a, bool):
            raise AsmError("builder: bool args refused (int/float/str only)")
        if isinstance(a, int):
            return str(a)
        if isinstance(a, float):
            return repr(a)
        if isinstance(a, str):
            return a
        raise AsmError(f"builder: bad arg {a!r} (want stream/int/float)")

    def op(self, mn, *args, out=None):
        o = out or self._fresh()
        self._lines.append(("op", o, mn, [self._render(a) for a in args]))
        return o

    def defn(self, name, formals, body):
        """DEF block: body is fn(sub: Prog) building DEF-local lines and
        returning the output stream name(s) (str or list). Formals become
        the DEF's INs."""
        sub = Prog(name=f"{self.name}:{name}")
        sub.inp(*formals)
        outs = body(sub)
        if isinstance(outs, str):
            outs = [outs]
        self._lines.append(("def", name, list(formals), sub._lines,
                            list(outs)))
        return self

    def call(self, name, *args, out=None, nout=1):
        o = out or (self._fresh() if nout == 1 else
                    [self._fresh() for _ in range(nout)])
        self._lines.append(("call", o, name, [self._render(a) for a in args]))
        return o

    # -- emission ------------------------------------------------------
    def text(self):
        L = []
        for k, v in self._config:
            L.append(f"CONFIG {k} {v}")
        for p in self._imports:
            L.append(f'IMPORT "{p}"')

        def emit(lines, ind, out):
            for ln in lines:
                if ln[0] == "op":
                    _, o, mn, args = ln
                    out.append(f"{'  ' * ind}{o} = {mn}({', '.join(args)})")
                elif ln[0] == "call":
                    _, o, name, args = ln
                    o_txt = o if isinstance(o, str) else \
                        f"({', '.join(o)})"
                    out.append(f"{'  ' * ind}{o_txt} = CALL {name}"
                               f"({', '.join(args)})")
                elif ln[0] == "def":
                    _, name, formals, body, outs = ln
                    out.append(f"{'  ' * ind}DEF {name}"
                               f"({', '.join(formals)}) -> "
                               f"({', '.join(outs)})")
                    emit(body, ind + 1, out)
                    out.append(f"{'  ' * ind}END")
        for n in self._ins:
            L.append(f"IN {n}")
        emit(self._lines, 0, L)
        return "\n".join(L) + "\n"

    def compile(self, target, sample=None, outputs=None, registry=None,
                sigs=None, basedir="."):
        """Straight to a backend (no text round-trip needed, but text()
        always shows what compiled). Origin marks generated code."""
        from chain.emit_c import compile_program
        return compile_program(self.text(), target, sample=sample,
                               outputs=outputs, registry=registry,
                               sigs=sigs, basedir=basedir,
                               origin=f"builder:{self.name}")


# -- stdlib seed: composites as plain functions ------------------------

def attn_scores(p, qh, kh, pos, scale_head2=False):
    """Head scores phase: RoPE + K-transpose + QK^T + optional head-temp
    scaling. Returns the score stream (caller owns masking/select)."""
    qr = p.op("ROTARY", qh, pos)
    kr = p.op("ROTARY", kh, pos)
    kt = p.op("TRANSPOSE", kr)
    sc = p.op("BATCH_MATMUL", qr, kt)
    if scale_head2:
        ht = p.op("TBETA", sc)
        sc = p.op("MUL", sc, ht)
    return sc


def attn_select(p, sc, vh, cmask, neg, out="CTX"):
    """Head select phase: causal mask + shift/wide-softmax + context."""
    ms = p.op("SELECT", cmask, sc, neg)
    s = p.op("TSHIFT", ms)
    pr = p.op("SOFTMAX_WIDE", s)
    return p.op("BATCH_MATMUL", pr, vh, out=out)


def attn_head(p, x, wq, wk, wv, pos, cmask, out="CTX",
              scale_head2=False):
    """Full single-head attention (QKV matmuls + narrow head). Kept for
    the builder smoke test; new code prefers attn_head_narrow."""
    q = p.op("MATMUL", x, wq)
    k = p.op("MATMUL", x, wk)
    v = p.op("MATMUL", x, wv)
    return attn_head_narrow(p, q, k, v, pos, cmask, out,
                            scale_head2=scale_head2)


def attn_head_narrow(p, qh, kh, vh, pos, cmask, out="CTX",
                     scale_head2=False, neg=None):
    """Head-narrow attention: scores + mask-fill + select. Matches
    lm_headt head structure. neg: shared mask-fill stream (BETA emitted
    once per layer by the caller); None emits it here."""
    sc = attn_scores(p, qh, kh, pos, scale_head2)
    neg = p.op("BETA", sc) if neg is None else neg
    return attn_select(p, sc, vh, cmask, neg, out)


def bank_retrieve(p, h, ukt, evb, rms_w, out="H2"):
    """Bank retrieval path: RMSNorm + key-match + shift/softmax +
    value-down + residual. Matches lm_headt bank structure."""
    hn = p.op("RMSNORM", h, rms_w)
    bc = p.op("MATMUL", hn, ukt)
    bs = p.op("TSHIFT", bc)
    bp = p.op("SOFTMAX_WIDE", bs)
    bdown = p.op("MATMUL", bp, evb)
    return p.op("ADD", h, bdown, out=out)


def qwen_layer(p, x, wq, wk, wv, wo, wup, wgate, wdown, ln1, ln2, pos,
               cmask, nh=28, nkv=4, dh=128, pre="", bq=None, bk=None,
               bv=None):
    """Full Qwen2 decoder layer (GQA attention + SwiGLU MLP) for the
    builder: per-head slices (KV groups shared), RoPE, causal select,
    shift/wide-softmax, concat-merge, residuals. Weight layouts are
    listing convention (K-major: wq (H,H), wk (H,KV*Dh), ...). NEG fill
    takes cmask geometry (S,S). bq/bk/bv: optional QKV bias PLANES
    (S,H)-tiled host-side (broadcast-as-input doctrine); None skips.
    Returns the layer output stream."""
    xn = p.op("RMSNORM", x, ln1, out=f"XN{pre}")
    q = p.op("MATMUL", xn, wq, out=f"Q{pre}")
    k = p.op("MATMUL", xn, wk, out=f"K{pre}")
    v = p.op("MATMUL", xn, wv, out=f"V{pre}")
    if bq is not None:
        q = p.op("ADD", q, bq, out=f"Q{pre}b")
    if bk is not None:
        k = p.op("ADD", k, bk, out=f"K{pre}b")
    if bv is not None:
        v = p.op("ADD", v, bv, out=f"V{pre}b")
    neg = p.op("BETA", cmask, out=f"NEG{pre}")
    per = nh // nkv
    # NOTE: no temperature scale here by design -- fold 1/sqrt(dh) into
    # the Q weights offline (exact: RoPE and matmul are linear in Q).
    # A scale op would need scalar broadcast (not in v0.1 ISA).
    ctx = None
    for h in range(nh):
        gh = h // per
        qh = p.op("SLICE", q, 1, h * dh, (h + 1) * dh)
        kh = p.op("SLICE", k, 1, gh * dh, (gh + 1) * dh)
        vh = p.op("SLICE", v, 1, gh * dh, (gh + 1) * dh)
        qr = p.op("ROTARY", qh, pos)
        kr = p.op("ROTARY", kh, pos)
        sc = p.op("BATCH_MATMUL", qr, p.op("TRANSPOSE", kr))
        ms = p.op("SELECT", cmask, sc, neg)
        pr = p.op("SOFTMAX_WIDE", p.op("TSHIFT", ms))
        ch = p.op("BATCH_MATMUL", pr, vh)
        ctx = ch if ctx is None else p.op("CONCAT", ctx, ch, 1)
    o = p.op("MATMUL", ctx, wo, out=f"O{pre}")
    h1 = p.op("ADD", x, o, out=f"H{pre}")
    hn = p.op("RMSNORM", h1, ln2, out=f"HN{pre}")
    up = p.op("MATMUL", hn, wup, out=f"UP{pre}")
    gate = p.op("MATMUL", hn, wgate, out=f"GATE{pre}")
    gs = p.op("SILU", gate, out=f"GS{pre}")
    mid = p.op("MUL", gs, up, out=f"MID{pre}")
    down = p.op("MATMUL", mid, wdown, out=f"DOWN{pre}")
    hn2 = p.op("ADD", h1, down, out=f"Y{pre}")
    return hn2


def qwen_dec_full_layer(p, x, Kp, Vp, wq, wk, wv, wo, wup, wgate,
                        wdown, ln1, ln2, pos, attmask7, onehot, ones,
                        allrows, S, nh=28, nkv=4, dh=128, pre="",
                        bq=None, bk=None, bv=None):
    """One full decoder layer for per-layer decode binaries (no new ops).

    x is the previous layer's output (1,H) (raw embed at L0); Kp/Vp are
    this layer's fp16 cache files, upcast losslessly in-graph (GATHER
    over arange rows); knew/vnew broadcast via MATMUL(ones, k/v) and
    merged by SELECT(onehot, new, prev); the host relay only copies
    files (fp32 out -> fp16 file, standard KV-cache precision).
    Attention runs the bmmv group path; mask is live attmask7.
    K rows are rotated once at insert (never re-rotated); Q rotary
    stays per-head (theta period Dh). Returns (y, Kf, Vf).
    """
    xn = p.op("RMSNORM", x, ln1, out=f"XN{pre}")
    q = p.op("MATMUL", xn, wq, out=f"Q{pre}")
    k = p.op("MATMUL", xn, wk, out=f"K{pre}")
    v = p.op("MATMUL", xn, wv, out=f"V{pre}")
    if bq is not None:
        q = p.op("ADD", q, bq, out=f"Q{pre}b")
    if bk is not None:
        k = p.op("ADD", k, bk, out=f"K{pre}b")
    if bv is not None:
        v = p.op("ADD", v, bv, out=f"V{pre}b")
    per = nh // nkv
    kp32 = p.op("GATHER", Kp, allrows, out=f"KP32{pre}")
    vp32 = p.op("GATHER", Vp, allrows, out=f"VP32{pre}")
    # Pin dynamic gather rows to baked S (static assertion as an op;
    # the ids file is frozen arange(S), so rows == S every step).
    kps = p.op("SLICE", kp32, 0, 0, S, out=f"KPS{pre}")
    vps = p.op("SLICE", vp32, 0, 0, S, out=f"VPS{pre}")
    vn = p.op("MATMUL", ones, v, out=f"VN{pre}")
    # NOTE: k is rotated per group below into kr; the knew broadcast
    # must use the ROTATED row: rotate first, then broadcast.
    qr = None
    for h in range(nh):
        qh = p.op("SLICE", q, 1, h * dh, (h + 1) * dh)
        qrh = p.op("ROTARY", qh, pos)
        qr = qrh if qr is None else p.op("CONCAT", qr, qrh, 1)
    kr = None
    for g in range(nkv):
        kh = p.op("SLICE", k, 1, g * dh, (g + 1) * dh)
        krh = p.op("ROTARY", kh, pos)
        kr = krh if kr is None else p.op("CONCAT", kr, krh, 1)
    knr = p.op("MATMUL", ones, kr, out=f"KNR{pre}")
    Kfull = p.op("SELECT", onehot, knr, kps, out=f"KF{pre}")
    Vfull = p.op("SELECT", onehot, vn, vps, out=f"VF{pre}")
    neg = p.op("BETA", attmask7, out=f"NEG{pre}")
    ctx = None
    for g in range(nkv):
        qg = p.op("SLICE", qr, 1, g * per * dh, (g + 1) * per * dh)
        kg = p.op("SLICE", Kfull, 1, g * dh, (g + 1) * dh)
        vg = p.op("SLICE", Vfull, 1, g * dh, (g + 1) * dh)
        sc = p.op("BMMV", qg, kg, per, 1, S, dh,
                  per * dh, dh, dh, 0, S, S, 1)
        ms = p.op("SELECT", attmask7, sc, neg)
        pr = p.op("SOFTMAX_WIDE", p.op("TSHIFT", ms))
        ch = p.op("BMMV", pr, vg, per, 1, dh, S,
                  S, S, dh, 0, dh, dh, 0)
        for j in range(per):
            chh = p.op("SLICE", ch, 0, j, j + 1)
            ctx = chh if ctx is None else p.op("CONCAT", ctx, chh, 1)
    o = p.op("MATMUL", ctx, wo, out=f"O{pre}")
    h1 = p.op("ADD", x, o, out=f"H{pre}")
    hn = p.op("RMSNORM", h1, ln2, out=f"HN{pre}")
    up = p.op("MATMUL", hn, wup, out=f"UP{pre}")
    gate = p.op("MATMUL", hn, wgate, out=f"GATE{pre}")
    gs = p.op("SILU", gate, out=f"GS{pre}")
    mid = p.op("MUL", gs, up, out=f"MID{pre}")
    down = p.op("MATMUL", mid, wdown, out=f"DOWN{pre}")
    return p.op("ADD", h1, down, out=f"Y{pre}"), Kfull, Vfull


def qwen_attn_layer(p, x, qr, K, V, wo, wup, wgate, wdown, ln2, attmask7,
                    S, nh=28, nkv=4, dh=128, pre=""):
    """Single-token attention+MLP (decode phase 2): K/V caches arrive as
    F16 files and upcast losslessly in-graph (GATHER over arange rows);
    the host relay does exact row-assign (no arithmetic outside the
    listings). BMMV group path over the caches; mask is the live
    host-tiled attmask7. Returns y (1,H).
    """
    per = nh // nkv
    kf = p.op("GATHER", K, "allrows", out=f"KF{pre}")
    vf = p.op("GATHER", V, "allrows", out=f"VF{pre}")
    neg = p.op("BETA", attmask7, out=f"NEG{pre}")
    ctx = None
    for g in range(nkv):
        qg = p.op("SLICE", qr, 1, g * per * dh, (g + 1) * per * dh)
        kg = p.op("SLICE", kf, 1, g * dh, (g + 1) * dh)
        vg = p.op("SLICE", vf, 1, g * dh, (g + 1) * dh)
        sc = p.op("BMMV", qg, kg, per, 1, S, dh,
                  per * dh, dh, dh, 0, S, S, 1)
        ms = p.op("SELECT", attmask7, sc, neg)
        pr = p.op("SOFTMAX_WIDE", p.op("TSHIFT", ms))
        ch = p.op("BMMV", pr, vg, per, 1, dh, S,
                  S, S, dh, 0, dh, dh, 0)
        for j in range(per):
            chh = p.op("SLICE", ch, 0, j, j + 1)
            ctx = chh if ctx is None else p.op("CONCAT", ctx, chh, 1)
    o = p.op("MATMUL", ctx, wo, out=f"O{pre}")
    h1 = p.op("ADD", x, o, out=f"H{pre}")
    hn = p.op("RMSNORM", h1, ln2, out=f"HN{pre}")
    up = p.op("MATMUL", hn, wup, out=f"UP{pre}")
    gate = p.op("MATMUL", hn, wgate, out=f"GATE{pre}")
    gs = p.op("SILU", gate, out=f"GS{pre}")
    mid = p.op("MUL", gs, up, out=f"MID{pre}")
    down = p.op("MATMUL", mid, wdown, out=f"DOWN{pre}")
    return p.op("ADD", h1, down, out=f"Y{pre}")


def qwen_bmmv_layer(p, x, wq, wk, wv, wo, wup, wgate, wdown, ln1, ln2, pos,
                    cmask, cmask7, S, nh=28, nkv=4, dh=128, pre="",
                    bq=None, bk=None, bv=None, cache_outs=False):
    """Qwen2 decoder layer with head-batched attention (BMMV, 48th op).

    Same math as qwen_layer, bit-exact (same products, same order):
    per-head RoPE is kept (theta repeats every Dh -- a wide ROTARY
    would read the wrong table), then rotated heads re-concatenate to
    QR and each GQA group runs ONE strided BMMV for scores and ONE for
    contexts (no per-head transpose/BMM, no 27-merge re-slicing from
    scattered heads). cmask7 is the host-tiled (per*S, S) causal mask
    (frozen); neg fills from it. S is the baked sequence slots.
    Returns the layer output stream, or (y, [krg...], [vg...]) with
    cache_outs=True (parallel-prefill cache outputs: rotated K groups
    + V groups, assembled host-side into (S,KV) caches)."""
    xn = p.op("RMSNORM", x, ln1, out=f"XN{pre}")
    q = p.op("MATMUL", xn, wq, out=f"Q{pre}")
    k = p.op("MATMUL", xn, wk, out=f"K{pre}")
    v = p.op("MATMUL", xn, wv, out=f"V{pre}")
    if bq is not None:
        q = p.op("ADD", q, bq, out=f"Q{pre}b")
    if bk is not None:
        k = p.op("ADD", k, bk, out=f"K{pre}b")
    if bv is not None:
        v = p.op("ADD", v, bv, out=f"V{pre}b")
    neg = p.op("BETA", cmask7, out=f"NEG{pre}")
    per = nh // nkv
    # NOTE: temperature folded offline (see qwen_layer); RoPE stays
    # per-head (theta period is Dh, not H).
    qr = None
    for h in range(nh):
        qh = p.op("SLICE", q, 1, h * dh, (h + 1) * dh)
        qrh = p.op("ROTARY", qh, pos)
        qr = qrh if qr is None else p.op("CONCAT", qr, qrh, 1)
    ctx = None
    krgs, vgs = [], []
    for g in range(nkv):
        qg = p.op("SLICE", qr, 1, g * per * dh, (g + 1) * per * dh)
        kg = p.op("SLICE", k, 1, g * dh, (g + 1) * dh)
        vg = p.op("SLICE", v, 1, g * dh, (g + 1) * dh)
        # One K head per group (GQA sharing): single rotary, shared by
        # all `per` Q heads via the stride-0 B view (no concat needed).
        krg = p.op("ROTARY", kg, pos)
        krgs.append(krg)
        vgs.append(vg)
        sc = p.op("BMMV", qg, krg, per, S, S, dh,
                  per * dh, dh, dh, 0, S, S * S, 1)
        ms = p.op("SELECT", cmask7, sc, neg)
        pr = p.op("SOFTMAX_WIDE", p.op("TSHIFT", ms))
        ch = p.op("BMMV", pr, vg, per, S, dh, S,
                  S, S * S, dh, 0, dh, S * dh, 0)
        for j in range(per):
            chh = p.op("SLICE", ch, 0, j * S, (j + 1) * S)
            ctx = chh if ctx is None else p.op("CONCAT", ctx, chh, 1)
    o = p.op("MATMUL", ctx, wo, out=f"O{pre}")
    h1 = p.op("ADD", x, o, out=f"H{pre}")
    hn = p.op("RMSNORM", h1, ln2, out=f"HN{pre}")
    up = p.op("MATMUL", hn, wup, out=f"UP{pre}")
    gate = p.op("MATMUL", hn, wgate, out=f"GATE{pre}")
    gs = p.op("SILU", gate, out=f"GS{pre}")
    mid = p.op("MUL", gs, up, out=f"MID{pre}")
    down = p.op("MATMUL", mid, wdown, out=f"DOWN{pre}")
    y = p.op("ADD", h1, down, out=f"Y{pre}")
    if cache_outs:
        return y, krgs, vgs
    return y
    """Single attention head over (S,D)->(S,Dh): QKV, RoPE, scores,
    head-temp, causal mask, shift+wide-softmax, context. Returns the
    context stream name. Matches lm_headt head structure (head2 gets
    TBETA scaling when scale_head2)."""
    q = p.op("MATMUL", x, wq)
    k = p.op("MATMUL", x, wk)
    v = p.op("MATMUL", x, wv)
    # NOTE: Dh slicing is the caller's job (SLICE before this call);
    # this composite takes head-narrow streams.
    qr = p.op("ROTARY", q, pos)
    kr = p.op("ROTARY", k, pos)
    kt = p.op("TRANSPOSE", kr)
    sc = p.op("BATCH_MATMUL", qr, kt)
    if scale_head2:
        ht = p.op("TBETA", sc)
        sc = p.op("MUL", sc, ht)
    neg = p.op("BETA", sc)
    ms = p.op("SELECT", cmask, sc, neg)
    s = p.op("TSHIFT", ms)
    pr = p.op("SOFTMAX_WIDE", s)
    return p.op("BATCH_MATMUL", pr, v, out=out)
