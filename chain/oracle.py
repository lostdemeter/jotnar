"""Float oracle: same TRUE-AMPLITUDE math as holo_phi, in float64.

Not shipped. Exists only to state the parity basis (units, peak, rounding)
for test_parity.py: int chain must match THIS, not the old ratio hack.
Math (default, Debt 2 verdict): A=sqrt(Y); As=gauss_blur(A, replicate);
D=A-As; Aenh=A+beta*D; Ienh=clip(Aenh^2,0,1); g=clip(Ienh/(Y+eps),.5,2).
use_alpha=True recovers the deprecated parabola ablation only.
"""
import numpy as np


def gauss_kernel(radius=2, sigma=1.0):
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (ax / sigma) ** 2)
    k /= k.sum()
    return np.outer(k, k)


def blur_replicate(x, kernel):
    r = kernel.shape[0] // 2
    xp = np.pad(x, ((r, r), (r, r)), mode="edge")
    H, W = x.shape
    out = np.zeros_like(x)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            out += kernel[dy, dx] * xp[dy:dy + H, dx:dx + W]
    return out


def enhance_luminance_float(y, beta=0.5, kernel=None, use_alpha=False):
    if kernel is None:
        kernel = gauss_kernel()
    a = np.sqrt(np.maximum(y, 0))
    as_ = blur_replicate(a, kernel)
    d = a - as_
    if use_alpha:
        alpha = np.clip(4 * y * (1 - y) + 0.3, 0.3, 1.0)
    else:
        alpha = 1.0
    aenh = a + beta * alpha * d
    return np.clip(aenh ** 2, 0, 1)


def enhance_image_float(rgb, beta=0.5, sigma=1.0, radius=2, use_alpha=False):
    kernel = gauss_kernel(radius=radius, sigma=sigma)
    y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    yenh = enhance_luminance_float(y, beta=beta, kernel=kernel, use_alpha=use_alpha)
    g = np.clip(yenh / np.maximum(y, 1e-12), 0.5, 2.0)
    out = np.clip(rgb * g[:, :, None], 0, 1)
    return out.astype(np.float32), yenh


# ---- splat_blur float mirror (parity basis for chain/splat.py) ----
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 8.0
SOBEL_Y = SOBEL_X.T.copy()
COH_THR = 0.25


def _corr_replicate(x, k):
    """Correlation with edge replicate (mirrors conv_trip, NOT flipped)."""
    r = k.shape[0] // 2
    xp = np.pad(x, ((r, r), (r, r)), mode="edge")
    H, W = x.shape
    out = np.zeros_like(x)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * xp[dy:dy + H, dx:dx + W]
    return out


def _aniso(sx, sy, radius=2):
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    gx = np.exp(-0.5 * (ax / sx) ** 2)
    gy = np.exp(-0.5 * (ax / sy) ** 2)
    k = np.outer(gy, gx)
    return k / k.sum()


def _rotated(theta_deg, s_long=1.8, s_short=0.5, radius=2):
    th = np.deg2rad(theta_deg)
    ux, uy = np.cos(th), np.sin(th)
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    u = xx * ux + yy * uy
    v = -xx * uy + yy * ux
    k = np.exp(-0.5 * ((u / s_long) ** 2 + (v / s_short) ** 2))
    return k / k.sum()


def splat_bank():
    return [_rotated(90), _rotated(0), _rotated(45), _rotated(-45),
            _aniso(1.0, 1.0)]


def splat_blur_float(a, coh_thr=None, soft=False):
    thr = COH_THR if coh_thr is None else float(coh_thr)
    gx = _corr_replicate(a, SOBEL_X)
    gy = _corr_replicate(a, SOBEL_Y)
    sk = gauss_kernel(radius=1, sigma=0.8)
    jxx = _corr_replicate(gx * gx, sk)
    jyy = _corr_replicate(gy * gy, sk)
    jxy = _corr_replicate(gx * gy, sk)
    disc = (jxx - jyy) ** 2 + 4 * jxy ** 2
    tr = jxx + jyy
    coh = np.clip(np.sqrt(np.maximum(disc, 0)) / np.maximum(tr, 1e-12), 0, 1)
    coh = np.where(tr <= 1e-12, 0.0, coh)
    gate = coh >= thr
    # orientation from smoothed tensor (mirrors splat.py v1.1): gate shut ->
    # 4; diagonal (|2Jxy|>|diff|) by sign -> 2 (sq<0, \\) / 3 (sq>=0, /);
    # axis by sign(diff) -> 0 (V) / 1 (H).
    diff = jxx - jyy
    diag_dom = 2 * np.abs(jxy) > np.abs(diff)
    vert = diff > 0
    diag_pos = jxy >= 0
    bucket = np.where(~gate, 4,
             np.where(~diag_dom, np.where(vert, 0, 1),
                      np.where(diag_pos, 3, 2))).astype(np.int8)
    bank = splat_bank()
    outs = [_corr_replicate(a, k) for k in bank]
    if soft:
        rV = np.maximum(diff, 0)
        rH = np.maximum(-diff, 0)
        rD1 = np.maximum(2 * jxy, 0)
        rD2 = np.maximum(-2 * jxy, 0)
        den = rV + rH + rD1 + rD2
        with np.errstate(divide="ignore", invalid="ignore"):
            w = [np.where(den > 0, r / np.maximum(den, 1e-300), 0)
                 for r in (rV, rH, rD1, rD2)]
        # mapping mirrors _soft_blend (rD1 = s2>0 = slash = outs[3]);
        # a swap here once mirrored the numpy bug -- parity stayed green
        # while rotation failed. Cross-check vs coherence_bucket, not parity.
        out = w[0] * outs[0] + w[1] * outs[1] + w[3] * outs[2] + w[2] * outs[3]
        out = np.where(den > 0, out, outs[4])
        return out, bucket, coh
    out = outs[4]
    for b in (3, 2, 1, 0):
        out = np.where(bucket == b, outs[b], out)
    return out, bucket, coh


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def enhance_luminance_float_splat(y, beta=0.5, iso_atten=1.0, coh_thr=None,
                                  mid_atten=1.0, coh_hi=0.5, depth=None,
                                  far_atten=0.5, strong_atten=1.0, soft=False,
                                  soft_k=30.0, soft_blur=False, v5=False,
                                  v5_w0=0.0, v5_w1=0.0, v5_w2=0.0):
    a = np.sqrt(np.maximum(y, 0))
    as_, bucket, coh = splat_blur_float(a, coh_thr=coh_thr, soft=soft_blur)
    if soft:
        w = (iso_atten + (mid_atten - iso_atten) * _sigmoid(soft_k * (coh - (coh_thr if coh_thr is not None else 0.25)))
             + (strong_atten - mid_atten) * _sigmoid(soft_k * (coh - coh_hi)))
        beff = beta * w
    else:
        beff = np.where(bucket == 4, beta * iso_atten,
                        np.where(coh >= coh_hi, beta * strong_atten,
                                 beta * mid_atten))
    if v5:
        d = a - as_
        dhat = np.clip(np.abs(d) * 4.0, 0, 1)
        scale = 0.5 + _sigmoid(v5_w0 + v5_w1 * coh + v5_w2 * dhat)
        beff = beff * scale
    if depth is not None:
        d = np.ascontiguousarray(depth, dtype=np.float64)
        n = np.zeros_like(d) if d.max() <= d.min() else (d - d.min()) / (d.max() - d.min())
        beff = beff * np.where(n <= float(np.median(n)), 1.0, far_atten)
    d = a - as_
    return np.clip((a + beff * d) ** 2, 0, 1)


def enhance_image_float_splat(rgb, beta=0.5, iso_atten=1.0, coh_thr=None,
                              mid_atten=1.0, coh_hi=0.5, depth=None,
                              far_atten=0.5, strong_atten=1.0, soft=False,
                              soft_blur=False, v5=False,
                              v5_w0=0.0, v5_w1=0.0, v5_w2=0.0):
    y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    yenh = enhance_luminance_float_splat(y, beta=beta, iso_atten=iso_atten,
                                         coh_thr=coh_thr, mid_atten=mid_atten,
                                         coh_hi=coh_hi, depth=depth,
                                         far_atten=far_atten,
                                         strong_atten=strong_atten, soft=soft,
                                         soft_blur=soft_blur, v5=v5,
                                         v5_w0=v5_w0, v5_w1=v5_w1,
                                         v5_w2=v5_w2)
    g = np.clip(yenh / np.maximum(y, 1e-12), 0.5, 2.0)
    return np.clip(rgb * g[:, :, None], 0, 1).astype(np.float32), yenh
