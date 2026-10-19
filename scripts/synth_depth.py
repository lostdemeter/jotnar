"""Procedural analytic scenes (student Phase 12a): generator, not data.

Pinhole renderer over spheres / axis-aligned boxes / floor / back wall,
closed-form intersections, exact float64 z-depth. Flat shading (fixed
per-face shades, no lights) + 3D-checker textured variants. Scenes
generate on the fly from fixed seeds (12000+idx) — zero bytes banked,
perfectly reproducible; the generator is the fixture. 320x256
(diagnostic scale; dims divide by 32 for student grids).
Strata by construction in meta: object count, occlusion, depth range,
texture on/off. No torch here (numpy + PIL only).
"""
import numpy as np

W, H = 320, 256
FOCAL = 300.0
N_SCENES = 24
SEED_BASE = 12000

FLOOR_Y = -1.2
BACK_Z = 14.0

PALETTE = np.array([
    [0.80, 0.25, 0.25],
    [0.25, 0.55, 0.85],
    [0.30, 0.70, 0.35],
    [0.85, 0.70, 0.25],
    [0.65, 0.35, 0.75],
    [0.30, 0.75, 0.75],
    [0.85, 0.50, 0.30],
    [0.55, 0.55, 0.60],
])
WALL_ALBEDO = np.array([0.55, 0.55, 0.58])
FLOOR_ALBEDO = np.array([0.45, 0.45, 0.48])
# fixed per-face shades (axis, sign) — constants, not lighting
FACE_SHADE = {(0, 1): 0.85, (0, -1): 0.70, (1, 1): 1.00,
              (1, -1): 0.55, (2, 1): 0.90, (2, -1): 0.65}


def catalog(idx):
    """Construction axes for scene idx (each of 24 combos exactly once)."""
    return {"count": (1, 2, 4)[idx % 3],
            "occlusion": ("single", "stacked")[(idx // 3) % 2],
            "range": ("shallow", "deep")[(idx // 6) % 2],
            "textured": bool((idx // 12) % 2)}


def _place(rng, count, stacked, zband):
    """Primitive list; approximate non-intersection except intended pair."""
    objs = []
    tries = 0
    while len(objs) < count and tries < 500:
        tries += 1
        kind = "sphere" if rng.rand() < 0.5 else "box"
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-0.4, 2.4)
        z = rng.uniform(*zband)
        s = rng.uniform(0.4, 1.1)
        if kind == "sphere":
            o = ("sphere", np.array([x, y, z]), s)
        else:
            h = np.array([s, s * rng.uniform(0.6, 1.4), s])
            o = ("box", np.array([x, y, z]) - h / 2, np.array([x, y, z]) + h / 2)
        if stacked and len(objs) == 1:
            # intended occlusion pair: near-duplicate screen pos, delta-z
            px = objs[0][1] if objs[0][0] == "sphere" else \
                (objs[0][1] + objs[0][2]) / 2
            nz = px[2] - rng.uniform(2.5, 4.0)
            if nz < 1.5:
                continue
            if kind == "sphere":
                o = ("sphere",
                     np.array([px[0] + rng.uniform(-0.4, 0.4),
                               px[1] + rng.uniform(-0.4, 0.4), nz]),
                     s * 0.8)
            else:
                c = np.array([px[0] + rng.uniform(-0.4, 0.4),
                              px[1] + rng.uniform(-0.4, 0.4), nz])
                h = np.array([s, s, s]) * 0.8
                o = ("box", c - h / 2, c + h / 2)
            objs.append(o)
            continue
        if stacked and len(objs) == 0:
            objs.append(o)
            continue
        if not stacked:
            ok = True
            for q in objs:
                c1 = o[1] if o[0] == "sphere" else (o[1] + o[2]) / 2
                r1 = o[2] if o[0] == "sphere" else np.linalg.norm(o[2] - o[1]) / 2
                c2 = q[1] if q[0] == "sphere" else (q[1] + q[2]) / 2
                r2 = q[2] if q[0] == "sphere" else np.linalg.norm(q[2] - q[1]) / 2
                if np.linalg.norm(c1 - c2) < r1 + r2 + 0.6:
                    ok = False
                    break
            if ok:
                objs.append(o)
        else:
            objs.append(o)
    return objs


def render_scene(idx):
    """-> (rgb uint8 (H,W,3), depth float64 (H,W), meta dict)."""
    meta = catalog(idx)
    rng = np.random.RandomState(SEED_BASE + idx)
    zband = (2.5, 5.0) if meta["range"] == "shallow" else (2.5, 11.0)
    objs = _place(rng, meta["count"], meta["occlusion"] == "stacked", zband)

    xs = (np.arange(W) - W / 2) / FOCAL
    ys = (np.arange(H) - H / 2) / FOCAL
    dx, dy = np.meshgrid(xs, ys)
    dz = np.ones_like(dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(dx) < 1e-12, np.inf, 1.0 / dx)
        invy = np.where(np.abs(dy) < 1e-12, np.inf, 1.0 / dy)

    best_t = np.full((H, W), BACK_Z)
    albedo = np.tile(WALL_ALBEDO, (H, W, 1))
    hitx = np.zeros((H, W))
    hity = np.zeros((H, W))

    # floor y=FLOOR_Y (normal +y)
    tf = np.where(dy < -1e-12, (FLOOR_Y - 0.0) / np.where(dy == 0, -1e-12, dy),
                  np.inf)
    m = tf < best_t
    best_t[m] = tf[m]
    albedo[m] = FLOOR_ALBEDO
    hitx[m] = tf[m] * dx[m]
    hity[m] = tf[m] * dy[m]

    for oi, o in enumerate(objs):
        if o[0] == "sphere":
            c, r = o[1], o[2]
            b = dx * c[0] + dy * c[1] + c[2]
            cc = float(c @ c - r * r)
            disc = b * b - (dx * dx + dy * dy + 1.0) * cc
            valid = disc > 0
            sq = np.sqrt(np.maximum(disc, 0.0))
            aa = dx * dx + dy * dy + 1.0
            t0 = (b - sq) / aa
            hit = valid & (t0 > 1e-6) & (t0 < best_t)
            best_t[hit] = t0[hit]
            albedo[hit] = PALETTE[oi % len(PALETTE)]
        else:
            mn, mx = o[1], o[2]
            t1 = np.where(dx > 0, (mn[0]) * inv, (mx[0]) * inv)
            t2 = np.where(dx > 0, (mx[0]) * inv, (mn[0]) * inv)
            t3 = np.where(dy > 0, (mn[1]) * invy, (mx[1]) * invy)
            t4 = np.where(dy > 0, (mx[1]) * invy, (mn[1]) * invy)
            t5 = np.where(mn[2] > 0, mn[2], mx[2])
            t6 = np.where(mn[2] > 0, mx[2], mn[2])
            tn = np.maximum(np.maximum(t1, t3), t5)
            tf_ = np.minimum(np.minimum(t2, t4), t6)
            hit = (tn < tf_) & (tn > 1e-6) & (tn < best_t)
            best_t[hit] = tn[hit]
            face = np.zeros((H, W, 3))
            use_x = (t1 >= t3) & (t1 >= t5)
            use_y = (~use_x) & (t3 >= t5)
            face[hit & use_x] = [1, 0, 0]
            face[hit & use_y] = [0, 1, 0]
            face[hit & ~(use_x | use_y)] = [0, 0, 1]
            with np.errstate(invalid="ignore"):
                sgn = np.sign(tn * dx * face[:, :, 0]
                              + tn * dy * face[:, :, 1]
                              + tn * face[:, :, 2])
            shade = np.ones((H, W))
            for (ax, sg), sh in FACE_SHADE.items():
                sel = hit & (face[:, :, ax] == 1) & (
                    (sgn > 0) if sg > 0 else (sgn <= 0))
                shade[sel] = sh
            alb = (PALETTE[oi % len(PALETTE)] * shade[:, :, None])[hit]
            albedo[hit] = alb
        hitx[hit] = best_t[hit] * dx[hit]
        hity[hit] = best_t[hit] * dy[hit]

    if meta["textured"]:
        par = (np.floor(hitx) + np.floor(hity)
               + np.floor(best_t)).astype(np.int64) % 2
        albedo = albedo * np.where(par[:, :, None] == 0, 0.8, 0.2)

    rgb = (np.clip(albedo, 0, 1) * 255 + 0.5).astype(np.uint8)
    meta = dict(meta, seed=SEED_BASE + idx, res=(W, H),
                nprims=len(objs))
    return rgb, best_t.copy(), meta


def check_self_consistency(idx, stride=47):
    """Independent scalar-loop depth recompute over a pixel subset.

    Returns max abs deviation vs render_scene depth (expect ~0)."""
    rgb, depth, meta = render_scene(idx)
    rng = np.random.RandomState(SEED_BASE + idx)
    zband = (2.5, 5.0) if meta["range"] == "shallow" else (2.5, 11.0)
    objs = _place(rng, meta["count"], meta["occlusion"] == "stacked", zband)
    worst = 0.0
    for yy in range(0, H, stride):
        for xx in range(0, W, stride):
            dx = (xx - W / 2) / FOCAL
            dy = (yy - H / 2) / FOCAL
            t = BACK_Z
            if dy < -1e-12:
                t = min(t, FLOOR_Y / dy)
            for o in objs:
                if o[0] == "sphere":
                    c, r = o[1], o[2]
                    b = dx * c[0] + dy * c[1] + c[2]
                    cc = float(c @ c - r * r)
                    aa = dx * dx + dy * dy + 1.0
                    disc = b * b - aa * cc
                    if disc > 0:
                        t0 = (b - disc ** 0.5) / aa
                        if 1e-6 < t0 < t:
                            t = t0
                else:
                    mn, mx = o[1], o[2]
                    lo, hi = -1e300, 1e300
                    for d, l, u in ((dx, mn[0], mx[0]),
                                    (dy, mn[1], mx[1]), (1.0, mn[2], mx[2])):
                        if abs(d) < 1e-12:
                            if not (l <= 0 <= u):
                                lo, hi = 1, 0
                                break
                        else:
                            a, b_ = (l / d, u / d) if d > 0 else (u / d, l / d)
                            lo, hi = max(lo, a), min(hi, b_)
                    if lo < hi and 1e-6 < lo < t:
                        t = lo
            worst = max(worst, abs(t - depth[yy, xx]))
    return worst
