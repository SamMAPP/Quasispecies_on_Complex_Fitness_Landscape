#!/usr/bin/env python3
"""
fig_variance_dispersion.py -- Fig. "model4a30_variance_dispersion_origin_root_panel" (Chapter 4, Sec. 4.3.2)

Variance of the Hamming-distance distribution from (a) the ancestral genotype and (b) the instantaneous
root, and the corresponding indices of dispersion D = Var(m)/<m> in (c) and (d), averaged over iterations
(band: one standard deviation), with the Poisson reference D = 1 and the peak-introduction times.
Input: <data>/Means/*_ref_origin.csv, *_ref_root.csv from compute_distances.py.
Port of the thesis notebook 16-Simulations/old/plots-multipeak.ipynb (cell 2).

    python fig_variance_dispersion.py --data Data/multi_peak_delta15 --out Figures
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import argparse
_ap = argparse.ArgumentParser(description="Variance and index of dispersion from origin and root")
_ap.add_argument("--data", default="Data/multi_peak", help="model directory with Abundances/ (Means/, Distribs/ are written there)")
_ap.add_argument("--results", default=None, help="derived tables (default: <data>/results)")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--filter", default=None, help="substring selecting one parameter set, e.g. nrw_1000_L_10000_q_0.0")
ARGS = _ap.parse_args()
_RESULTS = ARGS.results or str(Path(ARGS.data) / "results")

"""
Create a PhD-quality 2 x 2 panel for model4a30:

    (a) Variance of Hamming distance from the ancestral origin
    (b) Variance of Hamming distance from the instantaneous root
    (c) Index of dispersion from the ancestral origin
    (d) Index of dispersion from the instantaneous root

For each independent simulation,

    D_i(t) = variance_hamming_i(t) / mean_hamming_i(t)

is calculated first. The plotted dispersion curve is then the mean of D_i(t)
across independent simulations, and the shaded region is plus/minus one
between-simulation standard deviation.

This is intentionally different from

    mean(variance_hamming) / mean(mean_hamming),

because the requested quantity is the index of dispersion of each realization.

Expected input directory
------------------------
    ./Data/model4a30/Means/

Expected filenames
------------------
    dataOut_<parameters>_iteration_<i>_<parameters>_ref_origin.csv
    dataOut_<parameters>_iteration_<i>_<parameters>_ref_root.csv

Each input CSV must contain:
    time, mean_hamming, variance_hamming

Designed to be pasted into one Jupyter cell or executed with:

    %run model4a30_plot_variance_dispersion_panel.py
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# Configuration
# =============================================================================

MEANS_DIR = Path(ARGS.data) / "Means"
FIGURE_DIR = Path(ARGS.out)
SUMMARY_DIR = Path(_RESULTS)

# Select exactly one parameter set when MEANS_DIR contains several.
# The filter is matched against the full filename.
#
# Example:
# FILE_FILTER = "nrw_1000_L_10000_q_0.0_fpeak_3"
FILE_FILTER = ARGS.filter

OUTPUT_STEM = "model4a30_variance_dispersion_origin_root_panel"

# -------------------------------------------------------------------------
# Peak-event markers
# -------------------------------------------------------------------------

PLOT_PEAK_EVENTS = True
ANNOTATE_PEAK_EVENTS = False

# The number includes the initial peak.
# Leave as None for all scheduled events found from tau and the time range.
MAX_TOTAL_PEAKS = None

# -------------------------------------------------------------------------
# Curves and uncertainty
# -------------------------------------------------------------------------

# Individual simulations are usually unnecessary in a thesis panel.
PLOT_INDIVIDUAL_ITERATIONS = False
INDIVIDUAL_ALPHA = 0.10
INDIVIDUAL_LINEWIDTH = 0.60

PLOT_STANDARD_DEVIATION = True
STD_MULTIPLIER = 1.0

# Draw the Poisson reference D = 1 in both dispersion panels.
PLOT_POISSON_REFERENCE = True

# The dispersion is undefined when mean_hamming <= this threshold.
# This mainly removes t = 0, where mean = variance = 0.
MEAN_EPSILON = 1e-14

# -------------------------------------------------------------------------
# Axes
# -------------------------------------------------------------------------

X_LIMITS = None

Y_LIMITS_VARIANCE_ORIGIN = None
Y_LIMITS_VARIANCE_ROOT = None
Y_LIMITS_DISPERSION_ORIGIN = None
Y_LIMITS_DISPERSION_ROOT = None

# Optional logarithmic axes. Normally leave False for these data.
LOG_Y_VARIANCE = False
LOG_Y_DISPERSION = False

# -------------------------------------------------------------------------
# Export
# -------------------------------------------------------------------------

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 600
SHOW_FIGURE = False


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    "figure.figsize": (10.8, 8.0),
    "figure.dpi": 140,
    "savefig.dpi": PNG_DPI,
    "font.family": "serif",
    "font.serif": [
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 11.5,
    "axes.labelsize": 12.5,
    "axes.titlesize": 12.0,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# Filename parsing
# =============================================================================

ITERATION_PATTERN = re.compile(
    r"_iteration_(?P<iteration>\d+)"
)

REFERENCE_PATTERN = re.compile(
    r"_ref_(?P<reference>origin|root)\.csv$"
)

TAU_PATTERN = re.compile(
    r"_tau_(?P<tau>\d+)"
)

DELTA_PATTERN = re.compile(
    r"_delta_(?P<delta>\d+)"
)

Q_PATTERN = re.compile(
    r"_q_(?P<q>-?\d+(?:\.\d+)?)"
)

FPEAK_PATTERN = re.compile(
    r"_fpeak_(?P<fpeak>-?\d+(?:\.\d+)?)"
)

FBG_PATTERN = re.compile(
    r"_fbg_(?P<fbg>-?\d+(?:\.\d+)?)"
)

FSTEP_PATTERN = re.compile(
    r"_fstep_(?P<fstep>-?\d+(?:\.\d+)?)"
)

MU_PATTERN = re.compile(
    r"_mu_(?P<mu>-?\d+(?:\.\d+)?)"
)

N_PATTERN = re.compile(
    r"_nrw_(?P<N>\d+)"
)

L_PATTERN = re.compile(
    r"_L_(?P<L>\d+)"
)


def parse_mean_filename(filepath):
    """Extract iteration and reference type from one mean CSV filename."""
    filename = filepath.name

    iteration_match = ITERATION_PATTERN.search(filename)
    reference_match = REFERENCE_PATTERN.search(filename)

    missing = []

    if iteration_match is None:
        missing.append("iteration")

    if reference_match is None:
        missing.append("reference")

    if missing:
        raise ValueError(
            f"Could not extract {', '.join(missing)} from:\n"
            f"{filename}"
        )

    return {
        "iteration": int(
            iteration_match.group("iteration")
        ),
        "reference": reference_match.group("reference"),
    }


def simulation_signature(filepath):
    """
    Remove iteration and reference from a filename.

    Both reference types and all independent simulations belonging to the
    same parameter set then share one signature.
    """
    signature = ITERATION_PATTERN.sub(
        "_iteration_*",
        filepath.name,
    )

    signature = REFERENCE_PATTERN.sub(
        "_ref_*.csv",
        signature,
    )

    return signature


def compact_signature_hash(signature):
    """Stable compact ID used to avoid overwriting different parameter sets."""
    return hashlib.blake2b(
        signature.encode("utf-8"),
        digest_size=6,
        person=b"m4a30VD",
    ).hexdigest()


def extract_parameter(
    pattern,
    signature,
    group_name,
    cast=None,
):
    """Extract one optional parameter from the selected filename signature."""
    match = pattern.search(signature)

    if match is None:
        return None

    value = match.group(group_name)

    if cast is None:
        return value

    return cast(value)


# =============================================================================
# Input loading and validation
# =============================================================================

REQUIRED_COLUMNS = {
    "time",
    "mean_hamming",
    "variance_hamming",
}


def read_one_mean_file(filepath):
    """
    Read one per-simulation mean CSV and calculate its dispersion index.
    """
    metadata = parse_mean_filename(filepath)

    dataframe = pd.read_csv(filepath)

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns {sorted(missing_columns)} in:\n"
            f"{filepath}"
        )

    dataframe = dataframe[
        [
            "time",
            "mean_hamming",
            "variance_hamming",
        ]
    ].copy()

    for column in [
        "time",
        "mean_hamming",
        "variance_hamming",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe["time"].duplicated().any():
        duplicated_times = sorted(
            dataframe.loc[
                dataframe["time"].duplicated(
                    keep=False
                ),
                "time",
            ].unique()
        )

        raise ValueError(
            f"Duplicated time values "
            f"{duplicated_times} in:\n{filepath}"
        )

    if (
        dataframe["variance_hamming"]
        < -1e-12
    ).any():
        raise ValueError(
            "Negative population variance detected in:\n"
            f"{filepath}"
        )

    # Protect against tiny negative values caused by floating-point rounding.
    dataframe["variance_hamming"] = (
        dataframe["variance_hamming"]
        .clip(lower=0.0)
    )

    dataframe["dispersion_index"] = np.where(
        dataframe["mean_hamming"] > MEAN_EPSILON,
        (
            dataframe["variance_hamming"]
            / dataframe["mean_hamming"]
        ),
        np.nan,
    )

    dataframe["iteration"] = metadata["iteration"]
    dataframe["reference"] = metadata["reference"]
    dataframe["filename"] = filepath.name

    return dataframe


def validate_reference_pairs(dataframe):
    """Require origin and root files for exactly the same iterations."""
    origin_iterations = set(
        dataframe.loc[
            dataframe["reference"] == "origin",
            "iteration",
        ].unique()
    )

    root_iterations = set(
        dataframe.loc[
            dataframe["reference"] == "root",
            "iteration",
        ].unique()
    )

    if origin_iterations != root_iterations:
        only_origin = sorted(
            origin_iterations - root_iterations
        )

        only_root = sorted(
            root_iterations - origin_iterations
        )

        raise ValueError(
            "Origin/root iteration sets do not match.\n"
            f"Only origin: {only_origin}\n"
            f"Only root: {only_root}"
        )


def warn_about_incomplete_time_points(dataframe):
    """
    Warn if simulations do not have identical time grids.
    """
    for reference in ["origin", "root"]:
        subset = dataframe[
            dataframe["reference"] == reference
        ]

        time_sets = {
            iteration: tuple(
                sorted(group["time"].unique())
            )
            for iteration, group
            in subset.groupby("iteration")
        }

        if not time_sets:
            continue

        reference_iteration = next(iter(time_sets))
        expected_times = time_sets[
            reference_iteration
        ]

        incomplete = {
            iteration: times
            for iteration, times in time_sets.items()
            if times != expected_times
        }

        if incomplete:
            print(
                "Warning: incomplete time grids for "
                f"reference '{reference}':"
            )

            for iteration, times in incomplete.items():
                print(
                    f"  iteration {iteration}: "
                    f"{len(times)} time points"
                )


def validate_origin_root_time_pairs(dataframe):
    """
    Require the origin and root CSV of each iteration to contain the same times.
    """
    mismatches = []

    for iteration, iteration_data in dataframe.groupby(
        "iteration"
    ):
        origin_times = set(
            iteration_data.loc[
                iteration_data["reference"] == "origin",
                "time",
            ]
        )

        root_times = set(
            iteration_data.loc[
                iteration_data["reference"] == "root",
                "time",
            ]
        )

        if origin_times != root_times:
            mismatches.append(
                {
                    "iteration": iteration,
                    "only_origin": sorted(
                        origin_times - root_times
                    ),
                    "only_root": sorted(
                        root_times - origin_times
                    ),
                }
            )

    if mismatches:
        lines = [
            "Origin/root time grids differ within iterations:"
        ]

        for mismatch in mismatches:
            lines.append(
                f"  iteration {mismatch['iteration']}: "
                f"only origin={mismatch['only_origin']}, "
                f"only root={mismatch['only_root']}"
            )

        raise ValueError("\n".join(lines))


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_reference(dataframe, reference):
    """
    Aggregate variance and per-simulation dispersion across simulations.
    """
    subset = dataframe[
        dataframe["reference"] == reference
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No files with reference='{reference}' were found."
        )

    summary = (
        subset
        .groupby("time", as_index=False)
        .agg(
            mean_variance=(
                "variance_hamming",
                "mean",
            ),
            std_variance=(
                "variance_hamming",
                "std",
            ),
            min_variance=(
                "variance_hamming",
                "min",
            ),
            max_variance=(
                "variance_hamming",
                "max",
            ),
            n_variance=(
                "variance_hamming",
                "count",
            ),

            mean_dispersion=(
                "dispersion_index",
                "mean",
            ),
            std_dispersion=(
                "dispersion_index",
                "std",
            ),
            min_dispersion=(
                "dispersion_index",
                "min",
            ),
            max_dispersion=(
                "dispersion_index",
                "max",
            ),
            n_dispersion=(
                "dispersion_index",
                "count",
            ),

            mean_hamming_across_simulations=(
                "mean_hamming",
                "mean",
            ),
            mean_variance_across_simulations=(
                "variance_hamming",
                "mean",
            ),
            n_iterations=(
                "iteration",
                "nunique",
            ),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    summary["std_variance"] = (
        summary["std_variance"]
        .fillna(0.0)
    )

    # A single finite D value has undefined sample SD; use zero shading.
    summary["std_dispersion"] = (
        summary["std_dispersion"]
        .fillna(0.0)
    )

    # Diagnostic quantity. It is saved but not plotted.
    summary["ratio_of_ensemble_means"] = np.where(
        summary[
            "mean_hamming_across_simulations"
        ] > MEAN_EPSILON,
        (
            summary[
                "mean_variance_across_simulations"
            ]
            / summary[
                "mean_hamming_across_simulations"
            ]
        ),
        np.nan,
    )

    summary["reference"] = reference

    return summary


# =============================================================================
# Locate one parameter set
# =============================================================================

if not MEANS_DIR.exists():
    raise FileNotFoundError(
        f"Means directory not found:\n"
        f"{MEANS_DIR.resolve()}"
    )

mean_files = sorted(
    MEANS_DIR.glob("dataOut_*_ref_*.csv")
)

if FILE_FILTER is not None:
    mean_files = [
        filepath
        for filepath in mean_files
        if FILE_FILTER in filepath.name
    ]

if not mean_files:
    raise FileNotFoundError(
        "No mean CSV files matched MEANS_DIR "
        "and FILE_FILTER."
    )

signatures = {
    simulation_signature(filepath)
    for filepath in mean_files
}

if len(signatures) > 1:
    print("Multiple parameter sets were detected:\n")

    for detected_signature in sorted(signatures):
        print(f"  {detected_signature}")

    raise ValueError(
        "\nSet FILE_FILTER so that exactly one "
        "parameter set is selected."
    )

signature = next(iter(signatures))
analysis_id = compact_signature_hash(signature)


# =============================================================================
# Read selected files
# =============================================================================

frames = [
    read_one_mean_file(filepath)
    for filepath in mean_files
]

all_data = (
    pd.concat(
        frames,
        ignore_index=True,
    )
    .sort_values(
        [
            "reference",
            "iteration",
            "time",
        ]
    )
    .reset_index(drop=True)
)

validate_reference_pairs(all_data)
validate_origin_root_time_pairs(all_data)
warn_about_incomplete_time_points(all_data)

origin_summary = aggregate_reference(
    all_data,
    reference="origin",
)

root_summary = aggregate_reference(
    all_data,
    reference="root",
)


# =============================================================================
# Parameter information and peak-event times
# =============================================================================

tau = extract_parameter(
    TAU_PATTERN,
    signature,
    "tau",
    int,
)

delta = extract_parameter(
    DELTA_PATTERN,
    signature,
    "delta",
    int,
)

q_value = extract_parameter(
    Q_PATTERN,
    signature,
    "q",
    float,
)

f_peak = extract_parameter(
    FPEAK_PATTERN,
    signature,
    "fpeak",
    float,
)

f_background = extract_parameter(
    FBG_PATTERN,
    signature,
    "fbg",
    float,
)

f_step = extract_parameter(
    FSTEP_PATTERN,
    signature,
    "fstep",
    float,
)

mu_value = extract_parameter(
    MU_PATTERN,
    signature,
    "mu",
    float,
)

population_size = extract_parameter(
    N_PATTERN,
    signature,
    "N",
    int,
)

genome_length = extract_parameter(
    L_PATTERN,
    signature,
    "L",
    int,
)

maximum_time = int(
    max(
        origin_summary["time"].max(),
        root_summary["time"].max(),
    )
)

if tau is not None and tau > 0:
    peak_event_times = np.arange(
        tau,
        maximum_time + 1,
        tau,
        dtype=int,
    )
else:
    peak_event_times = np.array(
        [],
        dtype=int,
    )

if MAX_TOTAL_PEAKS is not None:
    if MAX_TOTAL_PEAKS < 1:
        raise ValueError(
            "MAX_TOTAL_PEAKS must be at least "
            "1 or None."
        )

    number_of_new_peaks = max(
        MAX_TOTAL_PEAKS - 1,
        0,
    )

    peak_event_times = peak_event_times[
        :number_of_new_peaks
    ]


# =============================================================================
# Save exact source data used in the panel
# =============================================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def rename_summary_columns(summary, suffix):
    """Rename all non-time columns before combining both references."""
    rename_map = {
        column: f"{column}_{suffix}"
        for column in summary.columns
        if column not in {"time", "reference"}
    }

    return (
        summary
        .drop(columns=["reference"])
        .rename(columns=rename_map)
    )


summary_output = (
    rename_summary_columns(
        origin_summary,
        "origin",
    )
    .merge(
        rename_summary_columns(
            root_summary,
            "root",
        ),
        on="time",
        how="outer",
        validate="one_to_one",
    )
    .sort_values("time")
    .reset_index(drop=True)
)

summary_csv_path = (
    SUMMARY_DIR
    / f"{OUTPUT_STEM}_{analysis_id}_source_data.csv"
)

summary_output.to_csv(
    summary_csv_path,
    index=False,
)


# =============================================================================
# Plotting helpers
# =============================================================================

def finite_xy(dataframe, x_column, y_column):
    """Return finite x and y arrays for robust plotting."""
    finite_mask = np.isfinite(
        dataframe[y_column].to_numpy(
            dtype=float
        )
    )

    return (
        dataframe.loc[
            finite_mask,
            x_column,
        ].to_numpy(dtype=float),
        dataframe.loc[
            finite_mask,
            y_column,
        ].to_numpy(dtype=float),
        finite_mask,
    )


def draw_panel(
    axis,
    raw_data,
    summary,
    reference,
    quantity,
    panel_label,
    panel_title,
    ylabel,
):
    """
    Draw variance or dispersion for one reference type.
    """
    if quantity == "variance":
        raw_column = "variance_hamming"
        mean_column = "mean_variance"
        std_column = "std_variance"

    elif quantity == "dispersion":
        raw_column = "dispersion_index"
        mean_column = "mean_dispersion"
        std_column = "std_dispersion"

    else:
        raise ValueError(
            f"Unknown quantity: {quantity}"
        )

    reference_data = raw_data[
        raw_data["reference"] == reference
    ]

    if PLOT_INDIVIDUAL_ITERATIONS:
        for _, iteration_data in reference_data.groupby(
            "iteration"
        ):
            iteration_data = iteration_data.sort_values(
                "time"
            )

            x_values, y_values, _ = finite_xy(
                iteration_data,
                "time",
                raw_column,
            )

            axis.plot(
                x_values,
                y_values,
                linewidth=INDIVIDUAL_LINEWIDTH,
                alpha=INDIVIDUAL_ALPHA,
                zorder=1,
            )

    x_values, mean_values, finite_mask = finite_xy(
        summary,
        "time",
        mean_column,
    )

    std_values = (
        summary.loc[
            finite_mask,
            std_column,
        ]
        .to_numpy(dtype=float)
    )

    mean_line, = axis.plot(
        x_values,
        mean_values,
        linewidth=2.15,
        zorder=3,
    )

    if PLOT_STANDARD_DEVIATION:
        lower = np.maximum(
            mean_values
            - STD_MULTIPLIER * std_values,
            0.0,
        )

        upper = (
            mean_values
            + STD_MULTIPLIER * std_values
        )

        axis.fill_between(
            x_values,
            lower,
            upper,
            color=mean_line.get_color(),
            alpha=0.20,
            linewidth=0,
            zorder=2,
        )

    if (
        quantity == "dispersion"
        and PLOT_POISSON_REFERENCE
    ):
        axis.axhline(
            1.0,
            linestyle=":",
            linewidth=1.15,
            alpha=0.80,
            zorder=0,
        )

    if PLOT_PEAK_EVENTS:
        for event_time in peak_event_times:
            axis.axvline(
                event_time,
                linestyle="--",
                linewidth=0.95,
                alpha=0.55,
                zorder=0,
            )

    axis.set_ylabel(ylabel)
    axis.set_title(
        panel_title,
        pad=6.0,
    )

    axis.text(
        0.025,
        0.955,
        panel_label,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )

    axis.grid(
        alpha=0.20,
        linewidth=0.6,
    )

    axis.minorticks_on()


# =============================================================================
# Build the 2 x 2 panel
# =============================================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10.8, 8.0),
    sharex=True,
)

draw_panel(
    axis=axes[0, 0],
    raw_data=all_data,
    summary=origin_summary,
    reference="origin",
    quantity="variance",
    panel_label="(a)",
    panel_title="Reference: ancestral origin",
    ylabel=r"Variance, $\mathrm{Var}(m)$",
)

draw_panel(
    axis=axes[0, 1],
    raw_data=all_data,
    summary=root_summary,
    reference="root",
    quantity="variance",
    panel_label="(b)",
    panel_title="Reference: instantaneous root",
    ylabel=r"Variance, $\mathrm{Var}(m)$",
)

draw_panel(
    axis=axes[1, 0],
    raw_data=all_data,
    summary=origin_summary,
    reference="origin",
    quantity="dispersion",
    panel_label="(c)",
    panel_title="Reference: ancestral origin",
    ylabel=(
        r"Index of dispersion, "
        r"$D=\mathrm{Var}(m)/\langle m\rangle$"
    ),
)

draw_panel(
    axis=axes[1, 1],
    raw_data=all_data,
    summary=root_summary,
    reference="root",
    quantity="dispersion",
    panel_label="(d)",
    panel_title="Reference: instantaneous root",
    ylabel=(
        r"Index of dispersion, "
        r"$D=\mathrm{Var}(m)/\langle m\rangle$"
    ),
)

for axis in axes[1, :]:
    axis.set_xlabel(r"Time, $t$")

if X_LIMITS is not None:
    for axis in axes.flat:
        axis.set_xlim(X_LIMITS)

if Y_LIMITS_VARIANCE_ORIGIN is not None:
    axes[0, 0].set_ylim(
        Y_LIMITS_VARIANCE_ORIGIN
    )

if Y_LIMITS_VARIANCE_ROOT is not None:
    axes[0, 1].set_ylim(
        Y_LIMITS_VARIANCE_ROOT
    )

if Y_LIMITS_DISPERSION_ORIGIN is not None:
    axes[1, 0].set_ylim(
        Y_LIMITS_DISPERSION_ORIGIN
    )

if Y_LIMITS_DISPERSION_ROOT is not None:
    axes[1, 1].set_ylim(
        Y_LIMITS_DISPERSION_ROOT
    )

if LOG_Y_VARIANCE:
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")

if LOG_Y_DISPERSION:
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")


# =============================================================================
# Optional event annotations
# =============================================================================

if (
    PLOT_PEAK_EVENTS
    and ANNOTATE_PEAK_EVENTS
):
    for axis in axes.flat:
        lower_y, upper_y = axis.get_ylim()

        if axis.get_yscale() == "log":
            annotation_y = (
                upper_y
                / (
                    upper_y / lower_y
                ) ** 0.04
            )
        else:
            annotation_y = (
                upper_y
                - 0.035 * (upper_y - lower_y)
            )

        for event_index, event_time in enumerate(
            peak_event_times,
            start=1,
        ):
            axis.text(
                event_time,
                annotation_y,
                f"Peak {event_index}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=7.8,
                alpha=0.70,
            )


# =============================================================================
# Shared legend
# =============================================================================

default_color = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"][0]

legend_handles = [
    Line2D(
        [0],
        [0],
        linewidth=2.15,
        color=default_color,
        label="Mean across simulations",
    ),
]

if PLOT_STANDARD_DEVIATION:
    legend_handles.append(
        Patch(
            facecolor=default_color,
            alpha=0.20,
            edgecolor="none",
            label=(
                rf"$\pm {STD_MULTIPLIER:g}\,"
                rf"\mathrm{{SD}}$ across simulations"
            ),
        )
    )

if PLOT_PEAK_EVENTS and len(peak_event_times) > 0:
    legend_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=0.95,
            alpha=0.65,
            label=r"New peak introduced, $t=k\tau$",
        )
    )

if PLOT_POISSON_REFERENCE:
    legend_handles.append(
        Line2D(
            [0],
            [0],
            linestyle=":",
            linewidth=1.15,
            alpha=0.85,
            label=r"Poisson reference, $D=1$",
        )
    )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=min(len(legend_handles), 4),
    frameon=False,
)


# =============================================================================
# Figure title
# =============================================================================

parameter_fragments = []

if q_value is not None:
    parameter_fragments.append(
        rf"$q={q_value:g}$"
    )

if tau is not None:
    parameter_fragments.append(
        rf"$\tau={tau}$"
    )

if delta is not None:
    parameter_fragments.append(
        rf"$\delta={delta}$"
    )

if f_peak is not None:
    parameter_fragments.append(
        rf"$f_0={f_peak:g}$"
    )

parameter_text = ", ".join(
    parameter_fragments
)

if parameter_text:
    fig.suptitle(
        parameter_text,
        y=0.955,
        fontsize=11.5,
    )


# =============================================================================
# Layout and export
# =============================================================================

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.085,
    top=0.895,
    wspace=0.26,
    hspace=0.25,
)

output_base = (
    FIGURE_DIR
    / f"{OUTPUT_STEM}_{analysis_id}"
)

pdf_path = output_base.with_suffix(".pdf")
png_path = output_base.with_suffix(".png")

if SAVE_PDF:
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

if SAVE_PNG:
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

print("Analysis complete")
print("-----------------")
print(f"Selected signature:\n  {signature}")
print(
    f"Independent simulations: "
    f"{all_data['iteration'].nunique()}"
)
print(
    f"Time range: "
    f"{all_data['time'].min()}–"
    f"{all_data['time'].max()}"
)
print(
    "Dispersion definition: "
    "D_i(t) = variance_i(t) / mean_i(t), "
    "then averaged across simulations."
)
print(
    "Undefined dispersion points "
    f"(mean <= {MEAN_EPSILON:g}) are omitted."
)
print(f"Source-data CSV:\n  {summary_csv_path.resolve()}")

if SAVE_PDF:
    print(f"PDF figure:\n  {pdf_path.resolve()}")

if SAVE_PNG:
    print(f"PNG figure:\n  {png_path.resolve()}")