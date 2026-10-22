#!/usr/bin/env python3
"""
fig_localization_region.py -- Fig. "localization_parameter_region" (Chapter 4, Sec. 4.2.1)

Localized (x_* > 0) and free-exploration (x_* = 0) regions of the single-peak holey landscape in the
(q, F) plane. The boundary is r_q = F p0 / s_q = 1, i.e.

    F_c(q) = s_q / p0 = [1 - q (1 - p0)] / p0,     p0 = exp(-lambda),

with F_c(0) = e^lambda. No simulation data are needed.
The thesis figure was drawn with lambda = 0.2 (the default here).

    python fig_localization_region.py --lam 0.2 --out Figures
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lam", type=float, default=0.20, help="mean number of mutations per replication")
    ap.add_argument("--out", default="Figures")
    a = ap.parse_args()

    p0 = np.exp(-a.lam)
    q = np.linspace(0.0, 1.0, 1000)
    F_c = (1.0 - q * (1.0 - p0)) / p0
    F_min, F_max = 1.0, 1.15 * np.exp(a.lam)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.fill_between(q, F_c, F_max, color="0.82", label=r"Localized ($x_*>0$)")
    ax.fill_between(q, F_min, F_c, color="white", label=r"Free exploration ($x_*=0$)")
    ax.plot(q, F_c, color="black", linewidth=1.8)
    ax.scatter([0], [1.0 / p0], color="black", s=28, zorder=5)
    ax.annotate(r"$F_c=e^{\lambda}$", xy=(0, 1.0 / p0), xytext=(0.08, 1.0 / p0 + 0.025), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8))
    ax.text(0.70, F_c[int(0.70 * (len(q) - 1))] + 0.055, "Localized", fontsize=12, ha="center", va="center")
    ax.text(0.28, 1.025, "Free exploration", fontsize=12, ha="center", va="center")
    q_label = 0.43
    ax.text(q_label, (1.0 - q_label * (1.0 - p0)) / p0 + 0.012, r"$r_q=1$", fontsize=11, rotation=-8,
            ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(F_min, F_max)
    ax.set_xlabel(r"$q$", fontsize=13)
    ax.set_ylabel(r"$F$", fontsize=13)
    ax.tick_params(axis="both", labelsize=11, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"localization_parameter_region.{ext}"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
