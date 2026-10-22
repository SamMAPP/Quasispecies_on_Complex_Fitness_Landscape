#!/usr/bin/env python3
"""
fig_stationary_Qk.py -- Fig. "model-fq_Qk_analytical_exponential_clean_t_100" (Chapter 4, Sec. 4.2.2)

Radial distribution Q(k) = fraction of individuals at Hamming distance k from the selected genotype,
pooled over all iterations, at t = 100, for the localization-transition runs (N=1e3, L=1e4, F=2,
lambda=1). Localized cases (q > q_c) are compared with the parameter-free exponential tail of the
stationary generating function G_*(z) = (r_q - 1) K_q(z) / (r_q - K_q(z)):

    K_q(z)  = [q p0 + (1-q) e^{lambda(z-1)}] / s_q,     s_q = 1 - q(1-p0),   r_q = F p0 / s_q,
    z_*     = (1/lambda) ln[(F - q)/(1 - q)]            (first pole, K_q(z_*) = r_q),
    Q_k    ~ C_q z_*^{-k} = C_q e^{-gamma_q k},          gamma_q = ln z_*,
    C_q     = (r_q - 1) r_q / [z_* K_q'(z_*)]            (residue at the pole).

Input: <data>/Distribs/*_q_<q>_*_t_100.txt written by generate_single_peak.py.

    python fig_stationary_Qk.py --data Data/single_peak --out Figures
"""
import argparse
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.linewidth": 1.1, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.major.size": 4, "ytick.major.size": 4, "pdf.fonttype": 42, "ps.fonttype": 42,
})
COLORS = {0.60: "#31688e", 0.75: "#35b779", 0.90: "#fde725"}
MARKERS = {0.60: "s", 0.75: "o", 0.90: "^"}


def tail_parameters(q, F, lam):
    """gamma_q, C_q and z_* of the stationary exponential tail (NaN if not localized)."""
    p0 = np.exp(-lam)
    s_q = 1.0 - q * (1.0 - p0)
    r_q = F * p0 / s_q
    z_star = np.log((F - q) / (1.0 - q)) / lam
    if r_q <= 1.0 or z_star <= 1.0:
        return np.nan, np.nan, np.nan
    dK = (1.0 - q) * lam * np.exp(lam * (z_star - 1.0)) / s_q
    return np.log(z_star), (r_q - 1.0) * r_q / (z_star * dK), z_star


def pooled_Qk(distribs, q, t):
    """Q(k) pooled over all Distribs files with this q and t."""
    counts, n_files = Counter(), 0
    for path in sorted(Path(distribs).glob("dataOut_*.txt")):
        mq = re.search(r"_q_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", path.stem)
        mt = re.search(r"_t_(\d+)$", path.stem)
        if mq and mt and np.isclose(float(mq.group(1)), q) and int(mt.group(1)) == t:
            counts.update(int(x) for x in path.read_text().split())
            n_files += 1
    if not counts:
        return None
    total = sum(counts.values())
    k = np.array(sorted(counts))
    print(f"q={q:g}: {n_files} files, {total} individuals")
    return k, np.array([counts[x] / total for x in k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="Data/single_peak", help="directory containing Distribs/")
    ap.add_argument("--out", default="Figures")
    ap.add_argument("--t", type=int, default=100)
    ap.add_argument("--F", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--q-localized", type=float, nargs="+", default=[0.60, 0.75, 0.90])
    ap.add_argument("--q-delocalized", type=float, default=0.30)
    ap.add_argument("--k-max", type=int, default=120)
    a = ap.parse_args()

    p0 = np.exp(-a.lam)
    print(f"q_c = {(1.0 - a.F * p0) / (1.0 - p0):.6f}")
    distribs = os.path.join(a.data, "Distribs")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    handles = []
    for q in a.q_localized:
        res = pooled_Qk(distribs, q, a.t)
        if res is None:
            continue
        k, Q = res
        keep = (Q > 0) & (k <= a.k_max)
        color, marker = COLORS.get(q, "C0"), MARKERS.get(q, "o")
        ax.plot(k[keep], Q[keep], ls="none", marker=marker, ms=5.0, mfc=color, mec=color, mew=0.0, zorder=4)
        gamma, C, z_star = tail_parameters(q, a.F, a.lam)
        if np.isfinite(gamma):
            kt = np.arange(0, int(k[keep].max()) + 1)
            ax.plot(kt, C * np.exp(-gamma * kt), ls="--", lw=1.8, color=color, zorder=2)
            print(f"q={q:.2f}: gamma={gamma:.6f}, z*={z_star:.6f}")
        handles.append(Line2D([0], [0], marker=marker, ls="none", color=color, ms=6, label=rf"$q={q:g}$"))
    res = pooled_Qk(distribs, a.q_delocalized, a.t)
    if res is not None:
        k, Q = res
        keep = (Q > 0) & (k <= a.k_max)
        ax.plot(k[keep], Q[keep], ls="-", lw=1.2, marker="o", ms=3.8, color="0.35", zorder=3)
        handles.append(Line2D([0], [0], marker="o", ls="-", color="0.35", ms=4.5, lw=1.2,
                              label=rf"$q={a.q_delocalized:g}<q_c$"))
    handles.append(Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="Analytical exponential tail"))

    ax.set_yscale("log")
    ax.set_xlim(0, a.k_max)
    ax.set_ylim(1e-6, 1.3)
    ax.set_xlabel(r"Hamming distance from the selected genotype, $k$")
    ax.set_ylabel(r"Relative frequency, $Q(k)$")
    ax.grid(True, which="major", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right", columnspacing=1.2, handlelength=2.3)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"model-fq_Qk_analytical_exponential_clean_t_{a.t}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":
    main()
