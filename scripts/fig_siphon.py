"""Figure: where the siphon breaks (architecture + margin problem).

Left: geometric 7B flow with the siphon path located (extract late,
gate lexical, install at L27) and the two obscurers marked (norm-map
reader, razor margins). Right: measured install quality vs dose at
L27 (micro-sweep + random control): margin grows, controls never
hold past installation -- the empty clean region IS the finding.
Usage: python3 scripts/fig_siphon.py (writes gallery/siphon.png)
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, label, sub="", fc="#dbeafe", ec="#1e40af"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center",
            fontsize=9, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center", va="center",
                fontsize=7, style="italic")


def arrow(ax, x1, y1, x2, y2, label="", ls="-", c="#111827"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.5, ls=ls, color=c))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label, ha="center",
                fontsize=7, bbox=dict(fc="white", ec="none", pad=1))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Siphon cleanliness: where it breaks (measured, Qwen2-7B geo)",
                 fontsize=12, weight="bold")

    # ---- left: architecture -----------------------------------------
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Flow with the problem located", fontsize=10)
    box(ax1, 0.3, 3.9, 1.6, 2.2, "prompt", "tokens in")
    box(ax1, 2.6, 3.9, 2.0, 2.2, "L0-L25", "funnel: moves mass\n(rank 100k->10)")
    box(ax1, 5.3, 3.9, 2.0, 2.2, "L26-L27", "lock: decides\n(rank 10->1)",
        fc="#fef3c7", ec="#b45309")
    box(ax1, 8.0, 3.9, 1.7, 2.2, "argmax", "reads NORM MAP\n(picks giants)",
        fc="#fee2e2", ec="#b91c1c")
    box(ax1, 8.0, 0.9, 1.7, 1.8, "dot reader", "reads CONTENT\n(norm-free)",
        fc="#dcfce7", ec="#15803d")
    arrow(ax1, 1.9, 5.0, 2.6, 5.0)
    arrow(ax1, 4.6, 5.0, 5.3, 5.0)
    arrow(ax1, 7.3, 5.2, 8.0, 5.2, "logits")
    arrow(ax1, 7.3, 4.7, 8.0, 1.9, "hidden", ls="--")
    # siphon path (green, below)
    box(ax1, 5.3, 7.7, 2.0, 1.4, "extract", "late direction d", fc="#dcfce7",
        ec="#15803d")
    box(ax1, 2.6, 7.7, 2.0, 1.4, "key", "lexical-early match", fc="#dcfce7",
        ec="#15803d")
    box(ax1, 5.3, 1.7, 2.0, 1.4, "INSTALL", "x + d.dose @ L27", fc="#fef3c7",
        ec="#b45309")
    arrow(ax1, 6.3, 7.7, 6.3, 6.7, ls="--", c="#15803d")
    arrow(ax1, 4.6, 8.2, 5.3, 8.2, ls="--", c="#15803d")
    ax1.text(6.3, 0.4, "dose must beat local noise\n(random flips too!)",
             ha="center", fontsize=7, style="italic", color="#b91c1c",
             bbox=dict(fc="#fee2e2", ec="none", pad=3))

    # ---- right: margin problem (measured) -----------------------------
    ax2.set_title("Install quality vs dose at L27 (measured)", fontsize=10)
    ax2.set_xlabel("dose (residual magnitudes)")
    ax2.set_ylabel("Paris margin (top1 - top2, abs logits)")
    # micro-sweep L27 (gain -> dose scale not needed: plot gain, note scales)
    gains = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    margins = [0.24, 0.25, 0.65, 1.00, 1.31, 1.60]
    ranks = [2, 1, 1, 1, 1, 1]
    holds = [True, False, False, False, False, False]
    for gx, mg, rk, hd in zip(gains, margins, ranks, holds):
        ax2.scatter([gx], [mg], s=90, c="#15803d" if hd else "#b91c1c",
                    marker="o" if rk == 1 else "s", zorder=3,
                    label="" if True else "")
    ax2.scatter([0.7], [0.0], s=140, c="#b91c1c", marker="x", lw=2.5,
                zorder=4)
    ax2.text(0.7, 0.12, "RANDOM dir, same dose\nflips Italy too", ha="center",
             fontsize=7, color="#b91c1c")
    ax2.axhspan(1.0, 3.0, color="#dcfce7", alpha=0.35)
    ax2.text(0.95, 2.2, "margin bar", fontsize=8, style="italic")
    ax2.text(0.55, 2.55, "CLEAN REGION:\nEMPTY (no cell holds\npast install)",
             ha="center", fontsize=8, weight="bold", color="#b91c1c",
             bbox=dict(fc="white", ec="#b91c1c", pad=4))
    ax2.set_xlim(0.4, 1.1)
    ax2.set_ylim(-0.2, 3.0)
    from matplotlib.lines import Line2D
    ax2.legend([Line2D([0], [0], marker="o", c="w", mfc="#15803d", ms=8),
                Line2D([0], [0], marker="o", c="w", mfc="#b91c1c", ms=8),
                Line2D([0], [0], marker="s", c="w", mfc="#15803d", ms=8)],
               ["rank 1 + control holds (never)",
                "rank 1, control moved", "rank 2-3 (weak)"],
               fontsize=7, loc="lower right")
    fig.tight_layout()
    out = os.path.join(ROOT, "gallery", "siphon.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
