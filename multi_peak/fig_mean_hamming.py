#!/usr/bin/env python3
"""
fig_mean_hamming.py -- Fig. "model4a30_mean_hamming_origin_root_panel" (Chapter 4, Sec. 4.3.1)

Mean Hamming distance (a) from the ancestral genotype and (b) from the instantaneous root, averaged over
iterations (band: one standard deviation), with the peak-introduction times (t = tau, 2 tau, ...).
Thesis run: N=1e3, L=1e4, lambda=1, q=0, tau=40, delta=15, F0=3, Delta_f=1, t_max=200, 100 iterations.
Input: <data>/Means/*_ref_origin.csv, *_ref_root.csv from compute_distances.py.
Port of the thesis notebook 16-Simulations/old/plots-multipeak.ipynb (cell 1).

    python fig_mean_hamming.py --data Data/multi_peak_delta15 --out Figures
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import argparse
_ap = argparse.ArgumentParser(description="Mean Hamming distance from origin and root")
_ap.add_argument("--data", default="Data/multi_peak", help="model directory with Abundances/ (Means/, Distribs/ are written there)")
_ap.add_argument("--results", default=None, help="derived tables (default: <data>/results)")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--filter", default=None, help="substring selecting one parameter set, e.g. nrw_1000_L_10000_q_0.0")
ARGS = _ap.parse_args()
_RESULTS = ARGS.results or str(Path(ARGS.data) / "results")

"""
Create a PhD-quality two-panel figure for model4a30:

(a) mean Hamming distance from the ancestral origin versus time;
(b) mean Hamming distance from the instantaneous root versus time.

The instantaneous root is the most abundant genotype at each generation, as
calculated by model4a30_hamming_analysis.py.

Expected input directory
------------------------
    ./Data/model4a30/Means/

Expected filenames
------------------
    dataOut_<parameters>_iteration_<i>_<parameters>_ref_origin.csv
    dataOut_<parameters>_iteration_<i>_<parameters>_ref_root.csv

Each CSV must contain:
    time, mean_hamming, variance_hamming

The solid curve in each panel is the mean of ``mean_hamming`` across
independent iterations. The shaded region is plus/minus one standard deviation
of those iteration means. It is therefore the between-simulation variability,
not the within-population variance stored in ``variance_hamming``.

Designed to be pasted into one Jupyter cell or executed with:

    %run model4a30_plot_mean_hamming_panel.py
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

MEANS_DIR = Path(ARGS.data) / "Means"
FIGURE_DIR = Path(ARGS.out)
SUMMARY_DIR = Path(_RESULTS)

# Select exactly one parameter set when MEANS_DIR contains several sets.
# The filter is matched against the filename.
#
# Example:
# FILE_FILTER = "nrw_1000_L_10000_q_0.0"
FILE_FILTER = ARGS.filter

# Output stem. A compact signature hash is appended automatically so different
# parameter sets do not overwrite one another.
OUTPUT_STEM = "model4a30_mean_hamming_origin_root_panel"

# Plot options
PLOT_PEAK_EVENTS = True
ANNOTATE_PEAK_EVENTS = False
MAX_TOTAL_PEAKS = None

# Optional faint individual-iteration curves behind the mean.
# For a thesis figure, False is usually cleaner.
PLOT_INDIVIDUAL_ITERATIONS = False
INDIVIDUAL_ALPHA = 0.12
INDIVIDUAL_LINEWIDTH = 0.65

# Standard-deviation region across independent simulations.
PLOT_STANDARD_DEVIATION = True
STD_MULTIPLIER = 1.0

# Figure export
SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 600
SHOW_FIGURE = False

# A shared y-axis is normally inappropriate because the origin- and
# root-referenced distances can differ greatly.
SHARE_Y_AXIS = False

# Optional common x limits. Leave as None to infer them from the data.
X_LIMITS = None

# Optional manually specified y limits for each panel.
Y_LIMITS_ORIGIN = None
Y_LIMITS_ROOT = None


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    "figure.figsize": (10.6, 4.25),
    "figure.dpi": 140,
    "savefig.dpi": PNG_DPI,
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
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

ITERATION_PATTERN = re.compile(r"_iteration_(?P<iteration>\d+)")
REFERENCE_PATTERN = re.compile(r"_ref_(?P<reference>origin|root)\.csv$")
TAU_PATTERN = re.compile(r"_tau_(?P<tau>\d+)")
DELTA_PATTERN = re.compile(r"_delta_(?P<delta>\d+)")
Q_PATTERN = re.compile(r"_q_(?P<q>-?\d+(?:\.\d+)?)")
FPEAK_PATTERN = re.compile(r"_fpeak_(?P<fpeak>-?\d+(?:\.\d+)?)")
FBG_PATTERN = re.compile(r"_fbg_(?P<fbg>-?\d+(?:\.\d+)?)")
FSTEP_PATTERN = re.compile(r"_fstep_(?P<fstep>-?\d+(?:\.\d+)?)")
MU_PATTERN = re.compile(r"_mu_(?P<mu>-?\d+(?:\.\d+)?)")
N_PATTERN = re.compile(r"_nrw_(?P<N>\d+)")
L_PATTERN = re.compile(r"_L_(?P<L>\d+)")


def parse_mean_filename(filepath):
    """Extract the iteration and reference type from one mean CSV filename."""
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
            f"Could not extract {', '.join(missing)} from:\n{filename}"
        )

    return {
        "iteration": int(iteration_match.group("iteration")),
        "reference": reference_match.group("reference"),
    }


def simulation_signature(filepath):
    """
    Return the parameter-dependent filename signature.

    The iteration number and reference suffix are removed, so all independent
    iterations and both references belonging to the same parameter set have
    the same signature.
    """
    signature = ITERATION_PATTERN.sub("_iteration_*", filepath.name)
    signature = REFERENCE_PATTERN.sub("_ref_*.csv", signature)
    return signature


def compact_signature_hash(signature):
    """Return a stable short ID used in output filenames."""
    return hashlib.blake2b(
        signature.encode("utf-8"),
        digest_size=6,
        person=b"m4a30plot",
    ).hexdigest()


def extract_parameter(pattern, signature, group_name, cast=None):
    """Extract an optional parameter value from the selected signature."""
    match = pattern.search(signature)

    if match is None:
        return None

    value = match.group(group_name)
    return cast(value) if cast is not None else value


# =============================================================================
# Data loading and validation
# =============================================================================

REQUIRED_COLUMNS = {
    "time",
    "mean_hamming",
    "variance_hamming",
}


def read_one_mean_file(filepath):
    """Read and validate one origin- or root-reference mean CSV file."""
    metadata = parse_mean_filename(filepath)
    dataframe = pd.read_csv(filepath)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns {sorted(missing_columns)} in:\n{filepath}"
        )

    dataframe = dataframe[
        ["time", "mean_hamming", "variance_hamming"]
    ].copy()

    for column in ["time", "mean_hamming", "variance_hamming"]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe["time"].duplicated().any():
        duplicated_times = sorted(
            dataframe.loc[
                dataframe["time"].duplicated(keep=False),
                "time",
            ].unique()
        )

        raise ValueError(
            f"Duplicated time values {duplicated_times} in:\n{filepath}"
        )

    dataframe["iteration"] = metadata["iteration"]
    dataframe["reference"] = metadata["reference"]
    dataframe["filename"] = filepath.name

    return dataframe


def aggregate_across_iterations(dataframe, reference):
    """
    Compute the mean and standard deviation of the simulation means.

    The stored population variance is also averaged and retained in the
    summary CSV, but it is not used for the shaded region.
    """
    subset = dataframe[dataframe["reference"] == reference].copy()

    if subset.empty:
        raise ValueError(
            f"No files with reference='{reference}' were found."
        )

    summary = (
        subset
        .groupby("time", as_index=False)
        .agg(
            mean_hamming=("mean_hamming", "mean"),
            std_between_iterations=("mean_hamming", "std"),
            mean_population_variance=("variance_hamming", "mean"),
            std_population_variance=("variance_hamming", "std"),
            minimum_mean_hamming=("mean_hamming", "min"),
            maximum_mean_hamming=("mean_hamming", "max"),
            n_iterations=("iteration", "nunique"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    summary["std_between_iterations"] = (
        summary["std_between_iterations"].fillna(0.0)
    )
    summary["std_population_variance"] = (
        summary["std_population_variance"].fillna(0.0)
    )

    summary["reference"] = reference

    return summary


def warn_about_incomplete_time_points(dataframe):
    """Warn when independent simulations do not contain identical time grids."""
    for reference in ["origin", "root"]:
        subset = dataframe[dataframe["reference"] == reference]

        time_sets = {
            iteration: tuple(sorted(group["time"].unique()))
            for iteration, group in subset.groupby("iteration")
        }

        if not time_sets:
            continue

        reference_iteration = next(iter(time_sets))
        expected_times = time_sets[reference_iteration]

        incomplete = {
            iteration: times
            for iteration, times in time_sets.items()
            if times != expected_times
        }

        if incomplete:
            print(
                f"Warning: incomplete time grids for reference '{reference}':"
            )

            for iteration, times in incomplete.items():
                print(
                    f"  iteration {iteration}: {len(times)} time points"
                )


def validate_reference_pairs(dataframe):
    """Ensure origin and root files exist for the same iterations."""
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
        only_origin = sorted(origin_iterations - root_iterations)
        only_root = sorted(root_iterations - origin_iterations)

        raise ValueError(
            "Origin/root iteration sets do not match.\n"
            f"Only origin: {only_origin}\n"
            f"Only root: {only_root}"
        )


# =============================================================================
# Locate one parameter set
# =============================================================================

if not MEANS_DIR.exists():
    raise FileNotFoundError(
        f"Means directory not found:\n{MEANS_DIR.resolve()}"
    )

mean_files = sorted(MEANS_DIR.glob("dataOut_*_ref_*.csv"))

if FILE_FILTER is not None:
    mean_files = [
        filepath
        for filepath in mean_files
        if FILE_FILTER in filepath.name
    ]

if not mean_files:
    raise FileNotFoundError(
        "No mean CSV files matched MEANS_DIR and FILE_FILTER."
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
        "\nSet FILE_FILTER so that only one parameter set is selected."
    )

signature = next(iter(signatures))
analysis_id = compact_signature_hash(signature)


# =============================================================================
# Read all selected mean files
# =============================================================================

frames = [
    read_one_mean_file(filepath)
    for filepath in mean_files
]

all_data = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["reference", "iteration", "time"])
    .reset_index(drop=True)
)

validate_reference_pairs(all_data)
warn_about_incomplete_time_points(all_data)

origin_summary = aggregate_across_iterations(
    all_data,
    reference="origin",
)

root_summary = aggregate_across_iterations(
    all_data,
    reference="root",
)


# =============================================================================
# Parameter information and peak-event times
# =============================================================================

tau = extract_parameter(TAU_PATTERN, signature, "tau", int)
delta = extract_parameter(DELTA_PATTERN, signature, "delta", int)
q_value = extract_parameter(Q_PATTERN, signature, "q", float)
f_peak = extract_parameter(FPEAK_PATTERN, signature, "fpeak", float)
f_background = extract_parameter(FBG_PATTERN, signature, "fbg", float)
f_step = extract_parameter(FSTEP_PATTERN, signature, "fstep", float)
mu_value = extract_parameter(MU_PATTERN, signature, "mu", float)
population_size = extract_parameter(N_PATTERN, signature, "N", int)
genome_length = extract_parameter(L_PATTERN, signature, "L", int)

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
    peak_event_times = np.array([], dtype=int)

if MAX_TOTAL_PEAKS is not None:
    if MAX_TOTAL_PEAKS < 1:
        raise ValueError("MAX_TOTAL_PEAKS must be at least 1 or None.")

    number_of_new_peaks = max(MAX_TOTAL_PEAKS - 1, 0)
    peak_event_times = peak_event_times[:number_of_new_peaks]


# =============================================================================
# Save the exact aggregated source data used in the figure
# =============================================================================

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

summary_output = (
    origin_summary[
        [
            "time",
            "mean_hamming",
            "std_between_iterations",
            "mean_population_variance",
            "std_population_variance",
            "n_iterations",
        ]
    ]
    .rename(
        columns={
            "mean_hamming": "mean_hamming_origin",
            "std_between_iterations": "std_between_iterations_origin",
            "mean_population_variance": "mean_population_variance_origin",
            "std_population_variance": "std_population_variance_origin",
            "n_iterations": "n_iterations_origin",
        }
    )
    .merge(
        root_summary[
            [
                "time",
                "mean_hamming",
                "std_between_iterations",
                "mean_population_variance",
                "std_population_variance",
                "n_iterations",
            ]
        ].rename(
            columns={
                "mean_hamming": "mean_hamming_root",
                "std_between_iterations": "std_between_iterations_root",
                "mean_population_variance": "mean_population_variance_root",
                "std_population_variance": "std_population_variance_root",
                "n_iterations": "n_iterations_root",
            }
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
summary_output.to_csv(summary_csv_path, index=False)


# =============================================================================
# Plotting helpers
# =============================================================================

def draw_panel(
    axis,
    raw_data,
    summary,
    reference,
    panel_label,
    panel_title,
    ylabel,
):
    """Draw one panel with optional individual curves and mean plus/minus SD."""
    if PLOT_INDIVIDUAL_ITERATIONS:
        reference_data = raw_data[
            raw_data["reference"] == reference
        ]

        for _, iteration_data in reference_data.groupby("iteration"):
            iteration_data = iteration_data.sort_values("time")

            axis.plot(
                iteration_data["time"],
                iteration_data["mean_hamming"],
                linewidth=INDIVIDUAL_LINEWIDTH,
                alpha=INDIVIDUAL_ALPHA,
                zorder=1,
            )

    mean_line, = axis.plot(
        summary["time"],
        summary["mean_hamming"],
        linewidth=2.15,
        label=r"Mean across simulations",
        zorder=3,
    )

    if PLOT_STANDARD_DEVIATION:
        lower = np.maximum(
            summary["mean_hamming"]
            - STD_MULTIPLIER * summary["std_between_iterations"],
            0.0,
        )

        upper = (
            summary["mean_hamming"]
            + STD_MULTIPLIER * summary["std_between_iterations"]
        )

        axis.fill_between(
            summary["time"],
            lower,
            upper,
            color=mean_line.get_color(),
            alpha=0.20,
            linewidth=0,
            label=rf"$\pm {STD_MULTIPLIER:g}\,\mathrm{{SD}}$",
            zorder=2,
        )

    if PLOT_PEAK_EVENTS:
        for event_index, event_time in enumerate(
            peak_event_times,
            start=1,
        ):
            axis.axvline(
                event_time,
                linestyle="--",
                linewidth=0.95,
                alpha=0.55,
                zorder=0,
                label=(
                    r"New peak introduced"
                    if event_index == 1
                    else None
                ),
            )

    axis.set_xlabel(r"Time, $t$")
    axis.set_ylabel(ylabel)
    axis.set_title(panel_title, pad=6.0)

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

    axis.set_ylim(bottom=0.0)
    axis.minorticks_on()
    axis.grid(alpha=0.18, linewidth=0.55)

    if X_LIMITS is not None:
        axis.set_xlim(*X_LIMITS)

    if ANNOTATE_PEAK_EVENTS and PLOT_PEAK_EVENTS:
        lower_y, upper_y = axis.get_ylim()
        annotation_y = upper_y - 0.025 * (upper_y - lower_y)

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
# Create the two-panel figure
# =============================================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10.6, 4.25),
    sharex=True,
    sharey=SHARE_Y_AXIS,
    constrained_layout=True,
)

draw_panel(
    axis=axes[0],
    raw_data=all_data,
    summary=origin_summary,
    reference="origin",
    panel_label="(a)",
    panel_title="Distance from the ancestral genotype",
    ylabel=r"Mean Hamming distance, $\langle m_{0}(t)\rangle$",
)

draw_panel(
    axis=axes[1],
    raw_data=all_data,
    summary=root_summary,
    reference="root",
    panel_label="(b)",
    panel_title="Distance from the instantaneous root",
    ylabel=r"Mean Hamming distance, $\langle m_{\mathrm{root}}(t)\rangle$",
)

if Y_LIMITS_ORIGIN is not None:
    axes[0].set_ylim(*Y_LIMITS_ORIGIN)

if Y_LIMITS_ROOT is not None:
    axes[1].set_ylim(*Y_LIMITS_ROOT)

# One compact legend is sufficient because both panels use the same encoding.
handles, labels = axes[0].get_legend_handles_labels()

# Remove repeated labels while preserving order.
unique_legend = dict(zip(labels, handles))
axes[0].legend(
    unique_legend.values(),
    unique_legend.keys(),
    loc="best",
)


# =============================================================================
# Export
# =============================================================================

output_base = FIGURE_DIR / f"{OUTPUT_STEM}_{analysis_id}"

if SAVE_PDF:
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

if SAVE_PNG:
    fig.savefig(
        output_base.with_suffix(".png"),
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

print("Selected simulation signature:")
print(f"  {signature}")
print()
print(f"Independent simulations: {all_data['iteration'].nunique()}")
print(
    f"Time range: {all_data['time'].min():g}–{all_data['time'].max():g}"
)

if tau is not None:
    print(f"tau = {tau}")

if delta is not None:
    print(f"delta = {delta}")

if q_value is not None:
    print(f"q = {q_value:g}")

if f_peak is not None:
    print(f"initial peak fitness = {f_peak:g}")

if f_background is not None:
    print(f"background fitness = {f_background:g}")

if f_step is not None:
    print(f"peak fitness increment = {f_step:g}")

if mu_value is not None:
    print(f"lambda = {mu_value:g}")

if population_size is not None:
    print(f"N = {population_size}")

if genome_length is not None:
    print(f"L = {genome_length}")

print()
print("Figure files:")

if SAVE_PDF:
    print(f"  {output_base.with_suffix('.pdf').resolve()}")

if SAVE_PNG:
    print(f"  {output_base.with_suffix('.png').resolve()}")

print("Figure source data:")
print(f"  {summary_csv_path.resolve()}")