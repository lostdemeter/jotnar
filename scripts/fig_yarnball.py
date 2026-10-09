"""Figure: how the yarn ball works (structure + behavior, measured).

Left: the object -- stores (key/value/gain x tiers incl. nulls),
ledger, and the three applies (yarnball soft-gate, dualbank log-space
AND, entity route). Right: the behavior -- address x content x dose
through the bank into routed dual heads (skewed prior + flat
install), annotated with measured numbers (Italy 299->1, holds
bit-identical, 4/4 installs, superposition 19/19).
Usage: python3 scripts/fig_yarnball.py (writes gallery/yarnball.png)
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, label, sub="", fc="#dbeafe", ec="#1e40af", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2 + (0.28 if sub else 0), label, ha="center",
            va="center", fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.42, sub, ha="center", va="center",
                fontsize=7, style="italic")


def arrow(ax, x1, y1, x2, y2, label="", ls="-", c="#111827"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.5, ls=ls, color=c))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, label, ha="center",
                fontsize=7, bbox=dict(fc="white", ec="none", pad=1))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle("YarnBall: the generic weight object (measured, native + Qwen2-7B)",
                 fontsize=12, weight="bold")

    # ---- left: the object -------------------------------------------
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Structure: stores + tiers + three applies", fontsize=10)
    box(ax1, 0.2, 6.8, 3.1, 2.4, "stores", "key (WHERE) x value (WHAT)\nx gain (HOW MUCH)")
    box(ax1, 0.2, 4.2, 3.1, 1.8, "tiers", "exact | assoc | opt | null\nnull = hold by construction",
        fc="#dcfce7", ec="#15803d")
    box(ax1, 0.2, 1.6, 3.1, 1.8, "ledger", "tier, support, norms,\ndose, sha per store",
        fc="#f3f4f6", ec="#4b5563")
    box(ax1, 4.2, 6.8, 2.4, 2.4, "yarnball_apply", "C=xa@U -> softmax\n-> y=P@V")
    box(ax1, 4.2, 4.2, 2.4, 1.8, "dualbank_apply", "C=HN@Uh + E@Ue\n(log-space AND)", fc="#dcfce7",
        ec="#15803d")
    box(ax1, 4.2, 1.6, 2.4, 1.8, "entity_route", "find row -> read AT\nrow -> tile", fc="#fef3c7",
        ec="#b45309")
    box(ax1, 7.3, 4.2, 2.5, 2.4, "loops", "negmine (learns bg)\ndose/key ladders\nsynthesis QP (6.8x)",
        fc="#f3f4f6", ec="#4b5563")
    arrow(ax1, 3.3, 7.6, 4.2, 7.6)
    arrow(ax1, 3.3, 5.1, 4.2, 5.1)
    arrow(ax1, 3.3, 2.5, 4.2, 2.5)
    ax1.text(8.55, 2.4, "zero new\nmnemonics", ha="center", fontsize=7,
             style="italic", color="#4b5563")

    # ---- right: the behavior ----------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Behavior: address x content x dose -> routed decision", fontsize=10)
    box(ax2, 0.2, 6.8, 2.2, 2.4, "address", "lexical-early keys\n(entity, asker)", fc="#dcfce7",
        ec="#15803d")
    box(ax2, 2.9, 6.8, 2.2, 2.4, "content", "readout rows +\ncontrasts (values)", fc="#dbeafe",
        ec="#1e40af")
    box(ax2, 5.6, 6.8, 2.2, 2.4, "dose", "gain in value rows\nwindows 1-2x", fc="#fef3c7",
        ec="#b45309")
    box(ax2, 1.4, 3.8, 5.0, 1.8, "bank + routing", "softmax routes; nulls hold; blend mass\nroutes heads (self-routing, no mask)",
        fc="#f3f4f6", ec="#4b5563")
    box(ax2, 0.2, 1.0, 3.7, 1.8, "skewed head", "prior intact\n(0.337, glue, UNK)", fc="#dbeafe",
        ec="#1e40af")
    box(ax2, 4.6, 1.0, 3.7, 1.8, "flat head", "installs flip\n(Rome r299->1)", fc="#dcfce7",
        ec="#15803d")
    box(ax2, 8.7, 3.8, 1.1, 4.4, "top-1", "4/4 native\nParis r1\n19/19 pred", fc="#dcfce7",
        ec="#15803d", fs=8)
    for xa, ya, xb, yb in ((1.3, 6.8, 2.2, 6.0), (4.0, 6.8, 3.9, 6.0), (6.7, 6.8, 5.6, 6.0)):
        arrow(ax2, xa, ya, xb, yb)
    arrow(ax2, 3.9, 3.8, 2.0, 2.8)
    arrow(ax2, 3.9, 3.8, 6.5, 2.8)
    arrow(ax2, 8.35, 5.0, 8.7, 5.0)
    ax2.text(5.0, 0.25, "split, don't tune  |  predict before running  |  grade on stable",
             ha="center", fontsize=7, style="italic", color="#4b5563")

    fig.tight_layout()
    out = os.path.join(ROOT, "gallery", "yarnball.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
