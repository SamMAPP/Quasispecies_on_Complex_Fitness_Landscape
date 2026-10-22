#!/usr/bin/env python3
"""
fig_evolutionary_search.py -- Fig. "NK_evolutionary_search_examples" (Chapter 4, Sec. 4.4.6)

Two long Wright-Fisher searches (5000 generations, N=1e6) on correlated NK landscapes compared with
the exact global optimum of the same landscape, against logarithmic generation:
  (a,e) best score so far Z_best(t), maximum occupied score, population mean score and Z_global;
  (b,f) remaining gap Delta Z(t) = Z_global - Z_best(t);
  (c,g) Hamming distance between each new record genotype and the global sequence;
  (d,h) distances from the search root: population mean, best record, dominant genotype.
A summary table (final Z_best, gap, distances) is written next to the figure.

Inputs: --evolved, output of generate_rugged.py --condition evolved (search_progress.csv,
search_records.csv), and --global, output of --condition global (peak.json) for the SAME base seed
and replicates; or the corresponding thesis suites.
Port of the thesis notebook 17-Simulations-NK/suite_detector.ipynb (cell 21).

    python fig_evolutionary_search.py --evolved Data/nk_evolved --global Data/nk_global --out Figures
"""
import matplotlib
matplotlib.use("Agg")
import argparse
_ap = argparse.ArgumentParser(description="Evolutionary search vs exact NK optimum")
_ap.add_argument("--evolved", default="Data/nk_evolved")
_ap.add_argument("--global", dest="global_", default="Data/nk_global")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--n-show", type=int, default=2, help="number of replicates shown")
ARGS = _ap.parse_args()

# =============================================================================
# FIGURE 3 — LONG EVOLUTIONARY SEARCH ON A CORRELATED NK LANDSCAPE
#
# PhD-thesis version — 2 x 4
#
# Columns:
#
#   1. Landscape score during evolutionary search
#        Z_best(t)
#        maximum Z currently occupied
#        population mean <Z>
#        exact global score Z_global
#
#   2. Remaining score gap
#        Delta Z(t) = Z_global - Z_best(t)
#
#   3. Distance of record genotypes to the exact global optimum
#        d_H(g_record, g_global)
#
#   4. Displacement from the initial evolutionary-search root
#        <m_0(t)>
#        d_H(g_best, g_0)
#        d_H(g_dom, g_0)
#
#
# AXES
# ----
# Generation is shown logarithmically in all columns.
#
# Z itself remains on a linear axis because it is a bounded raw landscape
# score occupying a narrow range; a logarithmic Z axis would not improve
# interpretation.
#
# Generation 0 cannot be represented on a logarithmic axis and is omitted
# from plotting only. The underlying data are unchanged.
#
# =============================================================================


from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

EVOLUTION_SUITE = Path(ARGS.evolved)
EXACT_SUITE = Path(ARGS.global_)


SELECT_REPLICATES = None
N_REPLICATES_TO_SHOW = ARGS.n_show


# -----------------------------------------------------------------------------
# Time axis
# -----------------------------------------------------------------------------

LOG_TIME = True
MIN_PLOTTED_GENERATION = 1.0


# -----------------------------------------------------------------------------
# Figure styling
# -----------------------------------------------------------------------------

FIGSIZE = (
    16.2,
    7.4,
)

DPI = 450

GRID_ALPHA = 0.14
GRID_LINEWIDTH = 0.45

MAIN_LINEWIDTH = 1.8
SECONDARY_LINEWIDTH = 1.20

RECORD_MARKER_SIZE = 20

PANEL_LABEL_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10.5
TICK_FONTSIZE = 9
TITLE_FONTSIZE = 11
ROW_LABEL_FONTSIZE = 10.5
LEGEND_FONTSIZE = 8


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path(ARGS.out)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PDF = (
    OUTPUT_DIR
    /
    "NK_evolutionary_search_examples.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    /
    "NK_evolutionary_search_examples.png"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    /
    "NK_evolutionary_search_examples_summary.csv"
)


# =============================================================================
# 2. CHECK INPUT SUITES
# =============================================================================

if not EVOLUTION_SUITE.exists():

    raise FileNotFoundError(
        f"Evolutionary-search suite not found:\n"
        f"{EVOLUTION_SUITE.resolve()}"
    )


if not EXACT_SUITE.exists():

    raise FileNotFoundError(
        f"Exact-global suite not found:\n"
        f"{EXACT_SUITE.resolve()}"
    )


# =============================================================================
# 3. SAFE FILE READERS
# =============================================================================

def safe_csv(path):

    path = Path(
        path
    )

    if not path.exists():

        return None


    for _ in range(2):

        try:

            df = pd.read_csv(
                path
            )

            return (
                df
                if len(df)
                else None
            )

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError,
        ):

            time.sleep(
                0.15
            )


    return None


def safe_json(path):

    path = Path(
        path
    )

    if not path.exists():

        return None


    for _ in range(2):

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as fh:

                return json.load(
                    fh
                )

        except (
            json.JSONDecodeError,
            OSError,
        ):

            time.sleep(
                0.15
            )


    return None


# =============================================================================
# 4. SEQUENCE / HAMMING HELPERS
# =============================================================================

def sequence_to_array(seq):

    if seq is None:

        return None


    if isinstance(
        seq,
        str,
    ):

        seq = (
            seq
            .strip()
            .upper()
        )


        mapping = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
        }


        try:

            return np.asarray(
                [
                    mapping[x]
                    for x in seq
                ],
                dtype=np.uint8,
            )

        except KeyError:

            return None


    arr = np.asarray(
        seq
    )


    if arr.ndim == 1:

        try:

            return arr.astype(
                np.uint8
            )

        except Exception:

            return None


    return None


def hamming(
    seq_a,
    seq_b,
):

    a = sequence_to_array(
        seq_a
    )

    b = sequence_to_array(
        seq_b
    )


    if (
        a is None
        or
        b is None
        or
        len(a) != len(b)
    ):

        return np.nan


    return int(
        np.count_nonzero(
            a != b
        )
    )


# =============================================================================
# 5. REPLICATE NUMBER FROM DIRECTORY
# =============================================================================

def infer_rep_from_dir(
    rep_dir,
):

    digits = "".join(
        c
        for c in rep_dir.name
        if c.isdigit()
    )


    return (
        int(digits)
        if digits
        else None
    )


# =============================================================================
# 6-7. DISCOVER RUNS (thesis suites or generate_rugged.py output)
# =============================================================================

def evolutionary_runs(suite):
    """Search trajectories: {replicate: {progress, records}}."""
    runs = {}
    for name, rec_name in (("evolutionary_peak_search_progress.csv", "evolutionary_peak_records.csv"),
                           ("search_progress.csv", "search_records.csv")):
        for progress_file in sorted(suite.rglob(name)):
            rep = infer_rep_from_dir(progress_file.parent)
            progress = safe_csv(progress_file)
            # the thesis Global suite also writes a one-row placeholder progress file: skip it
            if rep is None or progress is None or len(progress) < 2:
                continue
            runs[rep] = {"dir": progress_file.parent, "progress": progress,
                         "records": safe_csv(progress_file.parent / rec_name)}
    return runs


def exact_runs(suite):
    """Exact global optima: {replicate: {global_z, global_sequence}}."""
    runs = {}
    for path in sorted(suite.rglob("exact_global_peak_optimizer.json")):
        js = safe_json(path)
        rep = int(js.get("replicate", infer_rep_from_dir(path.parent)))
        runs[rep] = {"dir": path.parent, "global_z": float(js["global_peak_z"]),
                     "global_sequence": js["global_peak_sequence"]}
    for path in sorted(suite.rglob("peak.json")):
        js = safe_json(path)
        if js.get("condition") == "global":
            runs[int(js["replicate"])] = {"dir": path.parent, "global_z": float(js["z_peak"]),
                                          "global_sequence": js["peak_sequence"]}
    return runs


# =============================================================================
# 8. LOAD RUNS
# =============================================================================

evolutionary = evolutionary_runs(
    EVOLUTION_SUITE
)

exact = exact_runs(
    EXACT_SUITE
)


print(
    "Evolutionary-search replicates found:",
    sorted(
        evolutionary
    ),
)


print(
    "Exact-global replicates found:",
    sorted(
        exact
    ),
)


common_replicates = sorted(
    set(
        evolutionary
    )
    &
    set(
        exact
    )
)


common_replicates = [

    rep

    for rep
    in common_replicates

    if (
        evolutionary[
            rep
        ][
            "progress"
        ] is not None

        and

        len(
            evolutionary[
                rep
            ][
                "progress"
            ]
        ) > 0

        and

        np.isfinite(
            exact[
                rep
            ][
                "global_z"
            ]
        )
    )
]


if SELECT_REPLICATES is None:

    selected_replicates = (
        common_replicates[
            :
            N_REPLICATES_TO_SHOW
        ]
    )

else:

    selected_replicates = list(
        SELECT_REPLICATES
    )


missing = [

    rep

    for rep
    in selected_replicates

    if rep not in common_replicates
]


if missing:

    raise RuntimeError(
        f"Requested replicate(s) are not present "
        f"in both suites: {missing}"
    )


if len(
    selected_replicates
) != 2:

    raise RuntimeError(

        "Figure 3 is designed as a 2x4 figure. "
        f"Found/selected {len(selected_replicates)} usable replicates: "
        f"{selected_replicates}"
    )


print(
    "\nSelected replicates:",
    selected_replicates,
)


# =============================================================================
# 9. DATAFRAME COLUMN HELPERS
# =============================================================================

def first_existing_column(
    df,
    candidates,
):

    if df is None:

        return None


    for col in candidates:

        if col in df.columns:

            return col


    return None


def numeric_column(
    df,
    candidates,
):

    col = first_existing_column(
        df,
        candidates,
    )


    if col is None:

        return None, None


    values = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    ).to_numpy(
        float
    )


    return col, values


# =============================================================================
# 10. RECORD DISTANCE TO EXACT GLOBAL OPTIMUM
# =============================================================================

def get_record_distance_table(
    records,
    global_sequence,
):

    if (
        records is None
        or
        global_sequence is None
        or
        len(records) == 0
    ):

        return None


    sequence_col = first_existing_column(

        records,

        [
            "peak_sequence",
            "best_sequence",
            "sequence",
        ],
    )


    generation_col = first_existing_column(

        records,

        [
            "search_generation",
            "generation",
        ],
    )


    if (
        sequence_col is None
        or
        generation_col is None
    ):

        return None


    out = records.copy()


    out[
        "distance_to_exact_global"
    ] = [

        hamming(
            seq,
            global_sequence,
        )

        for seq
        in out[
            sequence_col
        ]
    ]


    out[
        "_generation"
    ] = pd.to_numeric(
        out[
            generation_col
        ],
        errors="coerce",
    )


    out = out[

        np.isfinite(
            out[
                "_generation"
            ]
        )

        &

        np.isfinite(
            pd.to_numeric(
                out[
                    "distance_to_exact_global"
                ],
                errors="coerce",
            )
        )

    ].copy()


    return out


# =============================================================================
# 11. POSITIVE-TIME MASK FOR LOG X
# =============================================================================

def positive_time_mask(
    generation,
):

    generation = np.asarray(
        generation,
        dtype=float,
    )


    return (
        np.isfinite(
            generation
        )
        &
        (
            generation
            >=
            MIN_PLOTTED_GENERATION
        )
    )


# =============================================================================
# 12. PREPARE DATA
# =============================================================================

plot_data = {}
summary_rows = []


for rep in selected_replicates:

    evo = evolutionary[
        rep
    ]

    ex = exact[
        rep
    ]


    progress = evo[
        "progress"
    ].copy()


    records = evo[
        "records"
    ]


    generation_col = first_existing_column(

        progress,

        [
            "search_generation",
            "generation",
        ],
    )


    if generation_col is None:

        raise KeyError(
            f"Could not find generation column "
            f"for replicate {rep}."
        )


    generation = pd.to_numeric(
        progress[
            generation_col
        ],
        errors="coerce",
    ).to_numpy(
        float
    )


    _, best_z = numeric_column(

        progress,

        [
            "best_z_ever",
            "best_z",
        ],
    )


    if best_z is None:

        raise KeyError(
            f"Could not find best-Z column "
            f"for replicate {rep}."
        )


    _, mean_z = numeric_column(

        progress,

        [
            "population_mean_z",
            "mean_z",
        ],
    )


    _, max_occupied_z = numeric_column(

        progress,

        [
            "max_z_occupied",
            "max_occupied_z",
        ],
    )


    _, dominant_z = numeric_column(

        progress,

        [
            "dominant_z",
        ],
    )


    _, mean_hamming_root = numeric_column(

        progress,

        [
            "mean_hamming_from_search_root",
            "mean_distance_from_search_root",
            "mean_hamming_from_initial_root",
        ],
    )


    _, best_distance_root = numeric_column(

        progress,

        [
            "best_distance_from_search_root",
            "best_hamming_from_search_root",
        ],
    )


    _, dominant_distance_root = numeric_column(

        progress,

        [
            "dominant_distance_from_search_root",
            "dominant_hamming_from_search_root",
        ],
    )


    global_z = float(
        ex[
            "global_z"
        ]
    )


    global_sequence = ex[
        "global_sequence"
    ]


    gap = (
        global_z
        -
        best_z
    )


    record_table = get_record_distance_table(
        records,
        global_sequence,
    )


    finite_generation = generation[
        np.isfinite(
            generation
        )
    ]


    finite_best = best_z[
        np.isfinite(
            best_z
        )
    ]


    final_generation = (

        float(
            finite_generation[
                -1
            ]
        )

        if len(
            finite_generation
        )

        else np.nan
    )


    final_best = (

        float(
            finite_best[
                -1
            ]
        )

        if len(
            finite_best
        )

        else np.nan
    )


    final_gap = (
        global_z
        -
        final_best
    )


    final_record_distance = np.nan


    if (
        record_table is not None
        and
        len(
            record_table
        )
    ):

        final_record_distance = float(

            record_table[
                "distance_to_exact_global"
            ].iloc[
                -1
            ]
        )


    def final_finite_value(
        arr,
    ):

        if arr is None:

            return np.nan


        finite = arr[
            np.isfinite(
                arr
            )
        ]


        if not len(
            finite
        ):

            return np.nan


        return float(
            finite[
                -1
            ]
        )


    final_mean_root_distance = final_finite_value(
        mean_hamming_root
    )

    final_best_root_distance = final_finite_value(
        best_distance_root
    )

    final_dominant_root_distance = final_finite_value(
        dominant_distance_root
    )


    summary_rows.append(

        {

            "replicate":
                rep,

            "final_generation":
                final_generation,

            "best_Z_final":
                final_best,

            "exact_global_Z":
                global_z,

            "remaining_gap":
                final_gap,

            "final_record_distance_to_global":
                final_record_distance,

            "final_mean_distance_from_search_root":
                final_mean_root_distance,

            "final_best_distance_from_search_root":
                final_best_root_distance,

            "final_dominant_distance_from_search_root":
                final_dominant_root_distance,

            "n_record_genotypes":
                (
                    len(
                        record_table
                    )
                    if record_table is not None
                    else 0
                ),
        }
    )


    plot_data[
        rep
    ] = {

        "generation":
            generation,

        "best_z":
            best_z,

        "mean_z":
            mean_z,

        "max_occupied_z":
            max_occupied_z,

        "dominant_z":
            dominant_z,

        "global_z":
            global_z,

        "gap":
            gap,

        "records":
            record_table,

        "mean_hamming_root":
            mean_hamming_root,

        "best_distance_root":
            best_distance_root,

        "dominant_distance_root":
            dominant_distance_root,
    }


summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# 13. COMMON AXIS LIMITS
# =============================================================================

all_generations = np.concatenate(

    [

        d[
            "generation"
        ][
            positive_time_mask(
                d[
                    "generation"
                ]
            )
        ]

        for d
        in plot_data.values()
    ]
)


COMMON_XLIM = (

    max(
        MIN_PLOTTED_GENERATION,
        float(
            np.min(
                all_generations
            )
        ),
    ),

    float(
        np.max(
            all_generations
        )
    )
    *
    1.03,
)


# -----------------------------------------------------------------------------
# Raw landscape-score limits
# -----------------------------------------------------------------------------

z_values = []


for d in plot_data.values():

    z_values.extend(
        d[
            "best_z"
        ][
            np.isfinite(
                d[
                    "best_z"
                ]
            )
        ].tolist()
    )


    if d[
        "mean_z"
    ] is not None:

        z_values.extend(

            d[
                "mean_z"
            ][
                np.isfinite(
                    d[
                        "mean_z"
                    ]
                )
            ].tolist()
        )


    if d[
        "max_occupied_z"
    ] is not None:

        z_values.extend(

            d[
                "max_occupied_z"
            ][
                np.isfinite(
                    d[
                        "max_occupied_z"
                    ]
                )
            ].tolist()
        )


    z_values.append(
        d[
            "global_z"
        ]
    )


z_min = min(
    z_values
)

z_max = max(
    z_values
)


z_pad = max(
    0.005,
    0.07
    *
    (
        z_max
        -
        z_min
    ),
)


COMMON_ZLIM = (

    z_min
    -
    z_pad,

    z_max
    +
    z_pad,
)


# -----------------------------------------------------------------------------
# Gap limits
# -----------------------------------------------------------------------------

gap_values = np.concatenate(

    [

        d[
            "gap"
        ][
            np.isfinite(
                d[
                    "gap"
                ]
            )
        ]

        for d
        in plot_data.values()
    ]
)


COMMON_GAP_YLIM = (

    max(
        0.0,
        float(
            np.min(
                gap_values
            )
        )
        *
        0.90,
    ),

    float(
        np.max(
            gap_values
        )
    )
    *
    1.08,
)


# -----------------------------------------------------------------------------
# Distance-to-global limits
# -----------------------------------------------------------------------------

record_distances = []


for d in plot_data.values():

    rec = d[
        "records"
    ]


    if rec is not None:

        record_distances.extend(

            pd.to_numeric(
                rec[
                    "distance_to_exact_global"
                ],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )


if record_distances:

    dmin = min(
        record_distances
    )

    dmax = max(
        record_distances
    )


    COMMON_GLOBAL_DISTANCE_YLIM = (

        max(
            0,
            dmin
            -
            5,
        ),

        dmax
        +
        5,
    )

else:

    COMMON_GLOBAL_DISTANCE_YLIM = None


# -----------------------------------------------------------------------------
# Displacement-from-root limits
# -----------------------------------------------------------------------------

root_distance_values = []


for d in plot_data.values():

    for key in [

        "mean_hamming_root",
        "best_distance_root",
        "dominant_distance_root",

    ]:

        values = d[
            key
        ]


        if values is not None:

            root_distance_values.extend(

                values[
                    np.isfinite(
                        values
                    )
                ].tolist()
            )


if root_distance_values:

    root_max = max(
        root_distance_values
    )


    COMMON_ROOT_DISTANCE_YLIM = (

        0.0,

        root_max
        *
        1.08,
    )

else:

    COMMON_ROOT_DISTANCE_YLIM = None


# =============================================================================
# 14. TIME-AXIS HELPER
# =============================================================================

def configure_time_axis(
    ax,
    row,
):

    if LOG_TIME:

        ax.set_xscale(
            "log"
        )


    ax.set_xlim(
        *COMMON_XLIM
    )


    # Only bottom row receives x-axis title
    if row == 1:

        ax.set_xlabel(
            "Generation",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    else:

        ax.set_xlabel(
            ""
        )


# =============================================================================
# 15. CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    4,

    figsize=FIGSIZE,
)


# -----------------------------------------------------------------------------
# Two-line column titles for better spacing
# -----------------------------------------------------------------------------

COLUMN_TITLES = [

    "Landscape score during\n"
    "evolutionary search",

    "Gap to exact\n"
    "global score",

    "Distance of record genotype\n"
    "to global optimum",

    "Displacement from initial\n"
    "search root",
]


for col, title in enumerate(
    COLUMN_TITLES
):

    axes[
        0,
        col
    ].set_title(

        title,

        fontsize=TITLE_FONTSIZE,

        pad=14,
    )


# =============================================================================
# 16. DRAW BOTH REALIZATIONS
# =============================================================================

fitness_legend_handles = None
fitness_legend_labels = None


for row, rep in enumerate(
    selected_replicates
):

    d = plot_data[
        rep
    ]


    generation = d[
        "generation"
    ]


    time_mask = positive_time_mask(
        generation
    )


    t_plot = generation[
        time_mask
    ]


    # =========================================================================
    # COLUMN 1 — RAW LANDSCAPE SCORE
    # =========================================================================

    ax = axes[
        row,
        0
    ]


    ax.plot(

        t_plot,

        d[
            "best_z"
        ][
            time_mask
        ],

        linewidth=MAIN_LINEWIDTH,

        label=r"$Z_{\rm best}(t)$",
    )


    if d[
        "max_occupied_z"
    ] is not None:

        ax.plot(

            t_plot,

            d[
                "max_occupied_z"
            ][
                time_mask
            ],

            linewidth=SECONDARY_LINEWIDTH,

            alpha=0.82,

            label=r"$\max Z_{\rm occ}(t)$",
        )


    if d[
        "mean_z"
    ] is not None:

        ax.plot(

            t_plot,

            d[
                "mean_z"
            ][
                time_mask
            ],

            linewidth=SECONDARY_LINEWIDTH,

            alpha=0.82,

            label=r"$\langle Z\rangle(t)$",
        )


    ax.axhline(

        d[
            "global_z"
        ],

        linestyle="--",

        linewidth=1.15,

        label=r"$Z_{\rm global}$",
    )


    configure_time_axis(
        ax,
        row,
    )


    ax.set_ylim(
        *COMMON_ZLIM
    )


    ax.set_ylabel(
        r"Raw landscape score $Z$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    ax.grid(
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )


    # Save legend handles from upper row
    if row == 0:

        fitness_legend_handles, fitness_legend_labels = (
            ax.get_legend_handles_labels()
        )


    # Show legend only in panel (e), slightly lower than before
    if (
        row == 1
        and
        fitness_legend_handles is not None
    ):

        ax.legend(

            fitness_legend_handles,
            fitness_legend_labels,

            frameon=True,

            framealpha=0.90,

            facecolor="white",

            edgecolor="none",

            fontsize=LEGEND_FONTSIZE,

            loc="upper left",

            bbox_to_anchor=(
                0.02,
                0.80,
            ),
        )


    # =========================================================================
    # COLUMN 2 — GAP TO EXACT GLOBAL SCORE
    # =========================================================================

    ax = axes[
        row,
        1
    ]


    ax.plot(

        t_plot,

        d[
            "gap"
        ][
            time_mask
        ],

        linewidth=MAIN_LINEWIDTH,
    )


    configure_time_axis(
        ax,
        row,
    )


    ax.set_ylim(
        *COMMON_GAP_YLIM
    )


    ax.set_ylabel(
        r"$\Delta Z(t)=Z_{\rm global}-Z_{\rm best}(t)$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    ax.grid(
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )


    # =========================================================================
    # COLUMN 3 — RECORD DISTANCE TO GLOBAL OPTIMUM
    # =========================================================================

    ax = axes[
        row,
        2
    ]


    records = d[
        "records"
    ]


    if (
        records is not None
        and
        len(
            records
        )
    ):

        xrec = records[
            "_generation"
        ].to_numpy(
            float
        )


        yrec = pd.to_numeric(
            records[
                "distance_to_exact_global"
            ],
            errors="coerce",
        ).to_numpy(
            float
        )


        good = (
            np.isfinite(
                xrec
            )
            &
            np.isfinite(
                yrec
            )
            &
            (
                xrec
                >=
                MIN_PLOTTED_GENERATION
            )
        )


        ax.plot(

            xrec[
                good
            ],

            yrec[
                good
            ],

            linewidth=1.0,

            alpha=0.72,
        )


        ax.scatter(

            xrec[
                good
            ],

            yrec[
                good
            ],

            s=RECORD_MARKER_SIZE,

            alpha=0.84,

            linewidths=0,

            zorder=3,
        )


    configure_time_axis(
        ax,
        row,
    )


    if COMMON_GLOBAL_DISTANCE_YLIM is not None:

        ax.set_ylim(
            *COMMON_GLOBAL_DISTANCE_YLIM
        )


    ax.set_ylabel(
        r"$d_H(g_{\rm record},g_{\rm global})$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    ax.yaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )


    ax.grid(
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )


    # =========================================================================
    # COLUMN 4 — DISPLACEMENT FROM INITIAL SEARCH ROOT
    # =========================================================================

    ax = axes[
        row,
        3
    ]


    if d[
        "mean_hamming_root"
    ] is not None:

        ax.plot(

            t_plot,

            d[
                "mean_hamming_root"
            ][
                time_mask
            ],

            linewidth=MAIN_LINEWIDTH,

            label=r"$\langle m_0(t)\rangle$",
        )


    if d[
        "best_distance_root"
    ] is not None:

        ax.plot(

            t_plot,

            d[
                "best_distance_root"
            ][
                time_mask
            ],

            linewidth=SECONDARY_LINEWIDTH,

            alpha=0.82,

            label=r"$d_H(g_{\rm best},g_0)$",
        )


    if d[
        "dominant_distance_root"
    ] is not None:

        ax.plot(

            t_plot,

            d[
                "dominant_distance_root"
            ][
                time_mask
            ],

            linewidth=SECONDARY_LINEWIDTH,

            alpha=0.82,

            label=r"$d_H(g_{\rm dom},g_0)$",
        )


    configure_time_axis(
        ax,
        row,
    )


    if COMMON_ROOT_DISTANCE_YLIM is not None:

        ax.set_ylim(
            *COMMON_ROOT_DISTANCE_YLIM
        )


    ax.set_ylabel(
        r"Hamming distance from $g_0$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    ax.yaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )


    ax.grid(
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )


    if row == 0:

        ax.legend(

            frameon=True,

            framealpha=0.90,

            facecolor="white",

            edgecolor="none",

            fontsize=LEGEND_FONTSIZE,

            loc="upper left",
        )


# =============================================================================
# 17. COMMON STYLE
# =============================================================================

for ax in axes.ravel():

    ax.tick_params(
        labelsize=TICK_FONTSIZE,
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            0.75
        )


# =============================================================================
# 18. ROW LABELS
# =============================================================================

ROW_LABELS = [

    rf"Evolutionary search 1" "\n"
    rf"(replicate {selected_replicates[0]})",

    rf"Evolutionary search 2" "\n"
    rf"(replicate {selected_replicates[1]})",
]


ROW_Y = [
    0.715,
    0.285,
]


for y, text in zip(
    ROW_Y,
    ROW_LABELS,
):

    fig.text(

        0.013,
        y,

        text,

        rotation=90,

        ha="center",
        va="center",

        fontsize=ROW_LABEL_FONTSIZE,

        fontweight="semibold",
    )


# =============================================================================
# 19. PANEL LABELS OUTSIDE AXES
# =============================================================================

PANEL_LABELS = [

    "(a)",
    "(b)",
    "(c)",
    "(d)",

    "(e)",
    "(f)",
    "(g)",
    "(h)",
]


for ax, label in zip(
    axes.ravel(),
    PANEL_LABELS,
):

    ax.annotate(

        label,

        xy=(
            0,
            1,
        ),

        xycoords="axes fraction",

        xytext=(
            -8,
            13,
        ),

        textcoords="offset points",

        ha="left",
        va="bottom",

        fontsize=PANEL_LABEL_FONTSIZE,

        fontweight="semibold",

        annotation_clip=False,
    )


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.subplots_adjust(

    left=0.073,

    right=0.992,

    bottom=0.105,

    top=0.90,

    wspace=0.36,

    hspace=0.33,
)


# =============================================================================
# 21. SAVE
# =============================================================================

fig.savefig(

    OUTPUT_PDF,

    bbox_inches="tight",

    facecolor="white",
)


fig.savefig(

    OUTPUT_PNG,

    dpi=DPI,

    bbox_inches="tight",

    facecolor="white",
)


summary.to_csv(
    OUTPUT_CSV,
    index=False,
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

print(
    f"  {OUTPUT_CSV}"
)


# =============================================================================
# 22. PRINT SUMMARY
# =============================================================================

print(
    "\n"
    +
    "=" * 95
)

print(
    "LONG EVOLUTIONARY SEARCH — LANDSCAPE SCORE AND SEQUENCE-SPACE MOTION"
)

print(
    "=" * 95
)


print(
    summary
)


for _, row in summary.iterrows():

    print(
        f"\nReplicate {int(row['replicate'])}"
    )


    print(
        f"  final generation                  = "
        f"{row['final_generation']:.0f}"
    )


    print(
        f"  best raw landscape score          = "
        f"{row['best_Z_final']:.9f}"
    )


    print(
        f"  exact global landscape score      = "
        f"{row['exact_global_Z']:.9f}"
    )


    print(
        f"  remaining score gap               = "
        f"{row['remaining_gap']:.9f}"
    )


    print(
        f"  final d_H(record, global)         = "
        f"{row['final_record_distance_to_global']:.0f}"
    )


    print(
        f"  final <m_0> from search root      = "
        f"{row['final_mean_distance_from_search_root']:.3f}"
    )


    print(
        f"  final d_H(best, g_0)              = "
        f"{row['final_best_distance_from_search_root']:.3f}"
    )


    print(
        f"  final d_H(dominant, g_0)          = "
        f"{row['final_dominant_distance_from_search_root']:.3f}"
    )


    print(
        f"  number of record genotypes        = "
        f"{int(row['n_record_genotypes'])}"
    )


plt.show()