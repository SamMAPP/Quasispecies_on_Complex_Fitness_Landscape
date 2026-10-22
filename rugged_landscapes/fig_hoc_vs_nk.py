#!/usr/bin/env python3
"""
fig_hoc_vs_nk.py -- Fig. "hoc_nk_compact_comparison_t200" (Chapter 4, Sec. 4.4.4)

House of Cards (top row) vs correlated NK, K=4 (bottom row) at t=200 (N=1e5, L=250, lambda=0.275,
linear-threshold fitness): degree CCDF P(K>=k), nearest-neighbour degree k_nn(k) and rank abundance
p(r) of the LCC, for three independent quenched realizations (replicates 0, 1, 2). For NK, each
realization is fitted separately over the shaded finite ranges (3<=k<=300, 10<=k<=100,
10<=r<=1000) and the dashed guide uses the mean slope (vertically displaced); the mean and standard
deviation over the three realizations of alpha_k, beta_nn and gamma are printed.

Input: snapshots from generate_rugged.py --condition local --dynamics explicit (or the thesis suite).
Port of the thesis notebook 17-Simulations-NK/suite_detector.ipynb (cell 6).

    python fig_hoc_vs_nk.py --data Data/hoc_vs_nk --out Figures
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import os
import sys
_ap = argparse.ArgumentParser(description="HoC vs correlated NK at t=200")
_ap.add_argument("--data", default="Data/hoc_vs_nk", help="output directory of generate_rugged.py (or a thesis suite)")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--t", type=int, default=200)
_ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
ARGS = _ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_analysis import analyze_snapshot, find_snapshot

# =============================================================================
# FIGURE 1 — HoC vs correlated NK
#
# Thesis version:
#
#   2 x 3 panels
#
#       Degree CCDF      k_nn(k)       Rank abundance
#   HoC
#   NK
#
# IMPORTANT:
#   - t = 200
#   - Model C only
#   - show exactly 3 independent realizations
#   - SCATTER ONLY: no pooled mean curve
#   - no pooled IQR band
#   - no fits for HoC
#   - for NK:
#         * fit each of the 3 displayed realizations independently
#         * average their fitted exponents
#         * draw a dashed visual guide using mean log-log slope/intercept
#         * use an observable-specific vertical offset for the guide
#   - panel labels outside plotting area
#   - no global title
#
# =============================================================================

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


MODEL = "model_c_linear"
LAMBDA = 0.275

TIME = ARGS.t


ARCHITECTURES = [
    "house_of_cards",
    "single_nk",
]

ARCH_LABELS = {
    "house_of_cards":
        "House of Cards",

    "single_nk":
        r"Correlated NK ($K=4$)",
}


# -----------------------------------------------------------------------------
# Exactly three independent realizations to display
# -----------------------------------------------------------------------------

DISPLAY_REPLICATES = ARGS.replicates


# -----------------------------------------------------------------------------
# Operational fitting windows used in the chapter
# -----------------------------------------------------------------------------

FIT_WINDOWS = {

    "ccdf":
        (
            3,
            300,
        ),

    "knn":
        (
            10,
            100,
        ),

    "rank":
        (
            10,
            1000,
        ),
}


# -----------------------------------------------------------------------------
# Observable-specific vertical displacement of dashed guide lines
#
# These factors affect ONLY the vertical visual placement of the dashed guides.
# They do NOT modify the fitted slopes or fitted exponents.
#
# A smaller value places the line closer to the data.
# -----------------------------------------------------------------------------

GUIDE_VERTICAL_FACTOR = {

    "ccdf":
        5.0,

    "knn":
        1.8,

    "rank":
        5.0,
}


# -----------------------------------------------------------------------------
# Plot appearance
# -----------------------------------------------------------------------------

SCATTER_SIZE = 11
SCATTER_ALPHA = 0.55

GUIDE_LINEWIDTH = 1.25
GUIDE_ALPHA = 0.95
GUIDE_LINESTYLE = "--"

GRID_ALPHA = 0.15
GRID_LINEWIDTH = 0.45

FIT_REGION_ALPHA = 0.045

DPI = 400


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path(ARGS.out)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PDF = (
    OUTPUT_DIR
    / "hoc_nk_compact_comparison_t200.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "hoc_nk_compact_comparison_t200.png"
)

# =============================================================================
# 2. ANALYZE THE DISPLAYED REALIZATIONS
# =============================================================================

analyses = {arch: [] for arch in ARCHITECTURES}
for arch in ARCHITECTURES:
    for rep in DISPLAY_REPLICATES:
        snapshot = find_snapshot(ARGS.data, arch, rep, TIME)
        print(f"{ARCH_LABELS[arch]} replicate {rep}: {snapshot}")
        analyses[arch].append({"replicate": int(rep), "result": analyze_snapshot(snapshot, replicate=rep)})


for arch in ARCHITECTURES:

    print(
        f"{ARCH_LABELS[arch]} analyzed: "
        f"{len(analyses[arch])} realization(s)"
    )


# =============================================================================
# 7. HELPER — EXTRACT (x,y) FROM INDIVIDUAL ANALYSIS
# =============================================================================

def extract_xy(
    result,
    observable,
):

    if observable not in result:

        return (
            np.array([]),
            np.array([]),
        )


    obj = result[
        observable
    ]


    if not isinstance(
        obj,
        (tuple, list),
    ):

        return (
            np.array([]),
            np.array([]),
        )


    if len(obj) < 2:

        return (
            np.array([]),
            np.array([]),
        )


    x = np.asarray(
        obj[0],
        dtype=float,
    )

    y = np.asarray(
        obj[1],
        dtype=float,
    )


    n = min(
        len(x),
        len(y),
    )


    x = x[:n]
    y = y[:n]


    good = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        (x > 0)
        &
        (y > 0)
    )


    return (
        x[
            good
        ],
        y[
            good
        ],
    )


# =============================================================================
# 8. FIT ONE REALIZATION
#
# CCDF:
#   P(K >= k) ~ k^(1-alpha_k)
#   alpha_k = 1 - slope
#
# k_nn:
#   k_nn(k) ~ k^(-beta_nn)
#   beta_nn = -slope
#
# Rank:
#   p(r) ~ r^(-gamma)
#   gamma = -slope
# =============================================================================

def fit_loglog_realization(
    result,
    observable,
):

    x, y = extract_xy(
        result,
        observable,
    )


    if len(x) < 3:

        return None


    xmin, xmax = FIT_WINDOWS[
        observable
    ]


    use = (
        (x >= xmin)
        &
        (x <= xmax)
        &
        (y > 0)
    )


    xf = x[
        use
    ]

    yf = y[
        use
    ]


    if len(xf) < 3:

        return None


    lx = np.log10(
        xf
    )

    ly = np.log10(
        yf
    )


    slope, intercept = np.polyfit(
        lx,
        ly,
        1,
    )


    # -------------------------------------------------------------------------
    # R^2 in log-log space
    # -------------------------------------------------------------------------

    predicted = (
        intercept
        +
        slope
        *
        lx
    )

    residual = (
        ly
        -
        predicted
    )


    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (
            ly
            -
            np.mean(
                ly
            )
        ) ** 2
    )


    r2 = (
        1.0
        -
        ss_res
        /
        ss_tot

        if ss_tot > 0

        else np.nan
    )


    # -------------------------------------------------------------------------
    # Convert raw slope to chapter coefficient
    # -------------------------------------------------------------------------

    if observable == "ccdf":

        coefficient = (
            1.0
            -
            slope
        )


    elif observable == "knn":

        coefficient = (
            -slope
        )


    elif observable == "rank":

        coefficient = (
            -slope
        )


    else:

        coefficient = np.nan


    return {

        "slope":
            float(
                slope
            ),

        "intercept":
            float(
                intercept
            ),

        "coefficient":
            float(
                coefficient
            ),

        "r2":
            float(
                r2
            ),

        "n":
            int(
                len(xf)
            ),
    }


# =============================================================================
# 9. FIT THE THREE NK REALIZATIONS INDEPENDENTLY
# =============================================================================

nk_fit_results = {

    observable: []

    for observable in (
        "ccdf",
        "knn",
        "rank",
    )
}


for item in analyses[
    "single_nk"
]:

    rep = item[
        "replicate"
    ]

    result = item[
        "result"
    ]


    for observable in nk_fit_results:

        fit = fit_loglog_realization(
            result,
            observable,
        )


        if fit is not None:

            fit[
                "replicate"
            ] = rep

            nk_fit_results[
                observable
            ].append(
                fit
            )


# =============================================================================
# 10. SUMMARIZE FITS
# =============================================================================

fit_summary = {}


for observable, fits in nk_fit_results.items():

    if not fits:

        fit_summary[
            observable
        ] = None

        continue


    slopes = np.asarray(
        [
            f["slope"]
            for f in fits
        ],
        dtype=float,
    )


    intercepts = np.asarray(
        [
            f["intercept"]
            for f in fits
        ],
        dtype=float,
    )


    coefficients = np.asarray(
        [
            f["coefficient"]
            for f in fits
        ],
        dtype=float,
    )


    r2_values = np.asarray(
        [
            f["r2"]
            for f in fits
        ],
        dtype=float,
    )


    fit_summary[
        observable
    ] = {

        "mean_slope":
            float(
                np.mean(
                    slopes
                )
            ),

        "mean_intercept":
            float(
                np.mean(
                    intercepts
                )
            ),

        "mean_coefficient":
            float(
                np.mean(
                    coefficients
                )
            ),

        "std_coefficient":
            float(
                np.std(
                    coefficients,
                    ddof=1,
                )
                if len(coefficients) > 1
                else 0.0
            ),

        "mean_r2":
            float(
                np.nanmean(
                    r2_values
                )
            ),
    }


# =============================================================================
# 11. HELPER — ADD NK REFERENCE GUIDE
# =============================================================================

def add_nk_reference_line(
    ax,
    observable,
):

    summary = fit_summary.get(
        observable
    )


    if summary is None:

        return


    xmin, xmax = FIT_WINDOWS[
        observable
    ]


    xguide = np.logspace(
        np.log10(
            xmin
        ),
        np.log10(
            xmax
        ),
        200,
    )


    # -------------------------------------------------------------------------
    # Guide uses average slope and average log-space intercept
    # from the three independently fitted realizations.
    #
    # The vertical position is then adjusted by an observable-specific factor.
    # -------------------------------------------------------------------------

    yguide = (

        10 ** (

            summary[
                "mean_intercept"
            ]

            +

            summary[
                "mean_slope"
            ]

            *

            np.log10(
                xguide
            )
        )

        *

        GUIDE_VERTICAL_FACTOR[
            observable
        ]
    )


    line, = ax.plot(

        xguide,
        yguide,

        linestyle=GUIDE_LINESTYLE,

        linewidth=GUIDE_LINEWIDTH,

        alpha=GUIDE_ALPHA,

        zorder=5,
    )


    # -------------------------------------------------------------------------
    # Label
    # -------------------------------------------------------------------------

    coefficient = summary[
        "mean_coefficient"
    ]


    if observable == "ccdf":

        symbol = (
            r"\overline{\alpha}_k"
        )


    elif observable == "knn":

        symbol = (
            r"\overline{\beta}_{\rm nn}"
        )


    else:

        symbol = (
            r"\overline{\gamma}"
        )


    label = (
        rf"${symbol}"
        rf"={coefficient:.2f}$"
    )


    label_index = int(
        0.62
        *
        (
            len(xguide)
            -
            1
        )
    )


    ax.text(

        xguide[
            label_index
        ],

        yguide[
            label_index
        ]
        *
        1.06,

        label,

        fontsize=8,

        ha="left",
        va="bottom",

        color=line.get_color(),
    )


# =============================================================================
# 12. FIXED COLORS FOR THREE REALIZATIONS
# =============================================================================

color_cycle = plt.rcParams[
    "axes.prop_cycle"
].by_key()[
    "color"
]


REPLICATE_COLORS = {

    DISPLAY_REPLICATES[i]:
        color_cycle[
            i
            %
            len(
                color_cycle
            )
        ]

    for i in range(
        len(
            DISPLAY_REPLICATES
        )
    )
}


# =============================================================================
# 13. HELPER — PLOT THE THREE REALIZATIONS AS SCATTER
# =============================================================================

def scatter_realizations(
    ax,
    architecture,
    observable,
):

    for item in analyses[
        architecture
    ]:

        rep = item[
            "replicate"
        ]

        result = item[
            "result"
        ]


        x, y = extract_xy(
            result,
            observable,
        )


        if len(x) == 0:

            continue


        ax.scatter(

            x,
            y,

            s=SCATTER_SIZE,

            alpha=SCATTER_ALPHA,

            marker="o",

            linewidths=0,

            color=REPLICATE_COLORS[
                rep
            ],

            label=(
                f"Replicate {rep}"
            ),

            zorder=2,
        )


# =============================================================================
# 14. CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(
        11.2,
        6.8,
    ),
)


# =============================================================================
# 15. DRAW ALL PANELS
# =============================================================================

OBSERVABLES = [
    "ccdf",
    "knn",
    "rank",
]


COLUMN_TITLES = {

    "ccdf":
        "Degree CCDF",

    "knn":
        "Nearest-neighbor degree",

    "rank":
        "Rank abundance",
}


X_LABELS = {

    "ccdf":
        r"Degree $k$",

    "knn":
        r"Degree $k$",

    "rank":
        r"Rank $r$",
}


Y_LABELS = {

    "ccdf":
        r"$P(K\geq k)$",

    "knn":
        r"$k_{\rm nn}(k)$",

    "rank":
        r"Relative abundance $p(r)$",
}


for row, arch in enumerate(
    ARCHITECTURES
):

    for col, observable in enumerate(
        OBSERVABLES
    ):

        ax = axes[
            row,
            col
        ]


        # ---------------------------------------------------------------------
        # Three independent realizations
        # ---------------------------------------------------------------------

        scatter_realizations(
            ax,
            arch,
            observable,
        )


        # ---------------------------------------------------------------------
        # Highlight operational fit region
        # ---------------------------------------------------------------------

        xmin, xmax = FIT_WINDOWS[
            observable
        ]


        ax.axvspan(

            xmin,
            xmax,

            alpha=FIT_REGION_ALPHA,

            zorder=0,
        )


        # ---------------------------------------------------------------------
        # Fit guide ONLY for correlated NK
        # ---------------------------------------------------------------------

        if arch == "single_nk":

            add_nk_reference_line(
                ax,
                observable,
            )


        # ---------------------------------------------------------------------
        # Axes
        # ---------------------------------------------------------------------

        ax.set_xscale(
            "log"
        )

        ax.set_yscale(
            "log"
        )


        ax.set_xlabel(
            X_LABELS[
                observable
            ]
        )

        ax.set_ylabel(
            Y_LABELS[
                observable
            ]
        )


        # ---------------------------------------------------------------------
        # Column titles only on upper row
        # ---------------------------------------------------------------------

        if row == 0:

            ax.set_title(
                COLUMN_TITLES[
                    observable
                ],
                pad=8,
            )


        ax.grid(

            alpha=GRID_ALPHA,

            linewidth=GRID_LINEWIDTH,
        )


# =============================================================================
# 16. ROW LABELS
# =============================================================================

for row, arch in enumerate(
    ARCHITECTURES
):

    axes[
        row,
        0
    ].text(

        -0.30,
        0.50,

        ARCH_LABELS[
            arch
        ],

        transform=axes[
            row,
            0
        ].transAxes,

        rotation=90,

        va="center",
        ha="center",

        fontsize=11,

        fontweight="semibold",
    )


# =============================================================================
# 17. PANEL LABELS OUTSIDE AXES
# =============================================================================

panel_labels = [

    "(a)",
    "(b)",
    "(c)",

    "(d)",
    "(e)",
    "(f)",
]


for ax, label in zip(

    axes.ravel(),

    panel_labels,
):

    ax.annotate(

        label,

        xy=(
            0,
            1,
        ),

        xycoords="axes fraction",

        xytext=(
            -4,
            10,
        ),

        textcoords="offset points",

        ha="left",
        va="bottom",

        fontsize=10,

        fontweight="semibold",

        annotation_clip=False,
    )


# =============================================================================
# 18. STYLE
# =============================================================================

for ax in axes.ravel():

    ax.tick_params(
        labelsize=8
    )

    ax.xaxis.label.set_size(
        9
    )

    ax.yaxis.label.set_size(
        9
    )


# =============================================================================
# 19. NO GLOBAL TITLE
# =============================================================================

fig.subplots_adjust(

    left=0.12,

    right=0.985,

    bottom=0.09,

    top=0.93,

    wspace=0.31,

    hspace=0.34,
)


# =============================================================================
# 20. SAVE
# =============================================================================

fig.savefig(

    OUTPUT_PDF,

    bbox_inches="tight",
)


fig.savefig(

    OUTPUT_PNG,

    dpi=DPI,

    bbox_inches="tight",
)


print(
    "\nSaved:"
)

print(
    f"  {OUTPUT_PDF}"
)

print(
    f"  {OUTPUT_PNG}"
)


# =============================================================================
# 21. PRINT THE THREE INDIVIDUAL FITS + THEIR MEAN
# =============================================================================

print(
    "\n"
    + "=" * 78
)

print(
    "CORRELATED NK — FITS OF THE THREE DISPLAYED REALIZATIONS"
)

print(
    "=" * 78
)


for observable in OBSERVABLES:

    print(
        f"\n{observable.upper()}"
    )


    fits = nk_fit_results[
        observable
    ]


    for fit in fits:

        print(

            f"  replicate "
            f"{fit['replicate']:>2d}: "

            f"coefficient = "
            f"{fit['coefficient']:.4f}, "

            f"raw slope = "
            f"{fit['slope']:+.4f}, "

            f"R^2 = "
            f"{fit['r2']:.4f}"
        )


    summary = fit_summary[
        observable
    ]


    if summary is not None:

        print(
            "  ---------------------------------------------"
        )

        print(

            f"  mean coefficient = "
            f"{summary['mean_coefficient']:.4f} "
            f"± "
            f"{summary['std_coefficient']:.4f}"
        )

        print(

            f"  mean R^2 = "
            f"{summary['mean_r2']:.4f}"
        )


plt.show()