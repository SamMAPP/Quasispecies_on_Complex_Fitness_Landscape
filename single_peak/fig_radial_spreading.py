#!/usr/bin/env python3
"""
fig_radial_spreading.py -- Fig. "model-fq_mean_hamming_and_beta_panel_last_15_points" (Chapter 4, Sec. 4.2.2)

(a) Mean Hamming distance from the selected genotype, <m_t>, averaged over iterations (band: one
    standard deviation), for every q of the localization-transition runs (N=1e3, L=1e4, F=2, lambda=1).
(b) Distribution over iterations of the late-time spreading rate beta, the slope of a linear fit to
    m_t over the final 15 generations of each iteration, compared with the free-exploration
    prediction of Model q, beta_q = lambda (1-q) / [1 - q (1 - p0)], and with q_c.

Input: <data>/Means/*.csv written by generate_single_peak.py (one file per q and iteration).
Only files with the requested N and L are used. The thesis figure also loaded 5 extra runs at
q=0.7 with N=1e5, L=250 that sat in the same folder; pass --N 0 --L 0 to reproduce that exactly.
Tables of the trajectories and fitted slopes are written next to the figure.

    python fig_radial_spreading.py --data Data/single_peak --out Figures
"""
import matplotlib
matplotlib.use("Agg")
import argparse
_ap = argparse.ArgumentParser(description="Radial spreading across the single-peak localization transition")
_ap.add_argument("--data", default="Data/single_peak", help="directory containing Means/")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--N", type=int, default=1000, help="population size to keep (0 = all)")
_ap.add_argument("--L", type=int, default=10000, help="sequence length to keep (0 = all)")
ARGS = _ap.parse_args()

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MODEL = "model-fq"

# Directory containing CSV files with the columns:
#
# t
# mean_hamming
# variance_hamming
MEANS_DIR = Path(ARGS.data) / "Means"

FIGURE_DIR = Path(ARGS.out)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Number of decimal places used to normalize q values extracted
# from filenames.
Q_ROUND_DECIMALS = 2

# Standard deviation convention across independent iterations.
STD_DDOF = 1

# Set to True to show individual simulation trajectories faintly
# behind the averaged trajectories.
SHOW_INDIVIDUAL_ITERATIONS = False

# Number of final time points used to estimate beta.
N_FIT_POINTS = 15

# Minimum number of time points required for one linear fit.
MIN_FIT_POINTS = 3

# Theoretical model parameters.
LAMBDA = 1.0
P0 = np.exp(-LAMBDA)

# Critical hole probability.
Q_CRITICAL = 0.42

# Accepted input extensions.
FILE_EXTENSIONS = {".csv"}

# Output basename.
PANEL_OUTPUT_BASENAME = (
    f"{MODEL}_mean_hamming_and_beta_panel_"
    f"last_{N_FIT_POINTS}_points"
)

# Width of each violin in q-axis units.
VIOLIN_WIDTH = 0.055

# Alpha of standard-deviation bands in panel a.
TRAJECTORY_BAND_ALPHA = 0.13

# Alpha of violins in panel b.
VIOLIN_ALPHA = 0.38

# q values to label directly in panel a.
#
# This avoids severe label overlap for trajectories that remain
# close to zero. Values absent from the data are ignored.
LABEL_Q_VALUES = {
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    1.0,
}

# x displacement of direct labels relative to the full time range.
LABEL_X_OFFSET_FRACTION = 0.018

# Length of the small horizontal leader segment that visually connects
# each direct label to the exact endpoint of its trajectory.
LABEL_LEADER_FRACTION = 0.55


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 600,

    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Metadata helpers
# ============================================================

def normalize_q(q: float | None) -> float | None:
    """
    Normalize q values to remove floating-point filename artifacts.

    For example:

        0.7000000001 -> 0.70
    """
    if q is None:
        return None

    return round(float(q), Q_ROUND_DECIMALS)


def parse_metadata(path: Path) -> dict:
    """
    Extract simulation metadata from a filename.

    Expected filename fragments include, for example:

        _q_0.3_
        _iteration_12_

    A possible complete filename is:

        dataOut_mu_1_nrw_1000_L_10000_q_0.3_
        iteration_12_fpeak_2_fbg_1.csv

    Only q is strictly required for this analysis.
    """
    name = path.stem

    patterns = {
        "mu": (
            r"(?:^|_)mu_"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "nrw": r"(?:^|_)nrw_(\d+)",
        "L": r"(?:^|_)L_(\d+)",
        "q": (
            r"(?:^|_)q_"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "iteration": r"(?:^|_)iteration_(\d+)",
        "fpeak": (
            r"(?:^|_)fpeak_"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "fbg": (
            r"(?:^|_)fbg_?"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
    }

    metadata = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, name)

        if match is None:
            metadata[key] = None
            continue

        value = match.group(1)

        if key in {"nrw", "L", "iteration"}:
            metadata[key] = int(value)
        else:
            metadata[key] = float(value)

    metadata["q"] = normalize_q(metadata["q"])

    return metadata


# ============================================================
# File reading
# ============================================================

def read_means_file(path: Path) -> pd.DataFrame:
    """
    Read one Means CSV file.

    Required columns
    ----------------
    t
    mean_hamming
    variance_hamming

    The function first tries a standard comma-separated CSV. If the
    required columns are not found, it retries with a more flexible
    separator.
    """
    required_columns = {
        "t",
        "mean_hamming",
        "variance_hamming",
    }

    dataframe = pd.read_csv(path)

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not required_columns.issubset(dataframe.columns):
        dataframe = pd.read_csv(
            path,
            sep=r"[,;\t\s]+",
            engine="python",
        )

        dataframe.columns = (
            dataframe.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"File {path} is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Columns found: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        [
            "t",
            "mean_hamming",
            "variance_hamming",
        ]
    ].copy()

    for column in [
        "t",
        "mean_hamming",
        "variance_hamming",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=["t", "mean_hamming"]
    )

    dataframe = (
        dataframe
        .sort_values("t")
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# Data collection
# ============================================================

def collect_means_data() -> pd.DataFrame:
    """
    Read all valid Means files.

    Returns
    -------
    pandas.DataFrame
        One row per time point and independent simulation.
    """
    if not MEANS_DIR.exists():
        raise FileNotFoundError(
            "The Means directory does not exist:\n"
            f"{MEANS_DIR.resolve()}"
        )

    paths = sorted(
        path
        for path in MEANS_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in FILE_EXTENSIONS
        )
    )

    if not paths:
        raise RuntimeError(
            "No CSV files were found in:\n"
            f"{MEANS_DIR.resolve()}"
        )

    dataframes = []
    skipped_files = []

    for file_index, path in enumerate(paths):
        metadata = parse_metadata(path)
        if (ARGS.N and metadata["nrw"] != ARGS.N) or (ARGS.L and metadata["L"] != ARGS.L):
            continue

        q = metadata["q"]
        iteration = metadata["iteration"]

        if q is None:
            skipped_files.append(
                (
                    path.name,
                    "q could not be extracted from filename",
                )
            )
            continue

        try:
            dataframe = read_means_file(path)

        except (
            ValueError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            skipped_files.append(
                (path.name, str(error))
            )
            continue

        if dataframe.empty:
            skipped_files.append(
                (
                    path.name,
                    "file contains no valid data rows",
                )
            )
            continue

        dataframe["file"] = path.name
        dataframe["q"] = q

        # When the iteration number is absent, assign one unique
        # identifier so that the file remains an independent run.
        dataframe["iteration"] = (
            iteration
            if iteration is not None
            else file_index
        )

        dataframe["mu"] = metadata["mu"]
        dataframe["nrw"] = metadata["nrw"]
        dataframe["L"] = metadata["L"]
        dataframe["fpeak"] = metadata["fpeak"]
        dataframe["fbg"] = metadata["fbg"]

        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError(
            "No valid Means files could be loaded. "
            "Check the filenames, q metadata, and CSV columns."
        )

    raw_data = pd.concat(
        dataframes,
        ignore_index=True,
    )

    raw_data = (
        raw_data
        .sort_values(["q", "iteration", "t"])
        .reset_index(drop=True)
    )

    q_values = sorted(
        raw_data["q"].dropna().unique()
    )

    print(f"\nLoaded {raw_data['file'].nunique()} files.")

    print(f"Found {len(q_values)} q values:")
    print(", ".join(f"{q:g}" for q in q_values))

    print("\nNumber of iterations per q:")

    iteration_counts = (
        raw_data[["q", "iteration"]]
        .drop_duplicates()
        .groupby("q")
        .size()
    )

    for q, number_iterations in iteration_counts.items():
        print(
            f"  q = {q:g}: "
            f"{number_iterations} iterations"
        )

    if skipped_files:
        print("\nSkipped files:")

        for filename, reason in skipped_files:
            print(f"  {filename}: {reason}")

    return raw_data


# ============================================================
# Aggregation across iterations
# ============================================================

def aggregate_over_iterations(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean-Hamming statistics across independent runs.

    For each q and t, the function calculates:

    - mean of mean_hamming;
    - standard deviation across iterations;
    - standard error of the mean;
    - number of contributing iterations.
    """
    aggregated = (
        raw_data
        .groupby(["q", "t"], as_index=False)
        .agg(
            average_mean_hamming=(
                "mean_hamming",
                "mean",
            ),
            std_mean_hamming=(
                "mean_hamming",
                lambda values: values.std(
                    ddof=STD_DDOF
                ),
            ),
            n_iterations=(
                "mean_hamming",
                "count",
            ),
            average_variance_hamming=(
                "variance_hamming",
                "mean",
            ),
        )
    )

    aggregated["std_mean_hamming"] = (
        aggregated["std_mean_hamming"]
        .fillna(0.0)
    )

    aggregated["sem_mean_hamming"] = (
        aggregated["std_mean_hamming"]
        / np.sqrt(aggregated["n_iterations"])
    )

    aggregated["lower_std"] = (
        aggregated["average_mean_hamming"]
        - aggregated["std_mean_hamming"]
    ).clip(lower=0.0)

    aggregated["upper_std"] = (
        aggregated["average_mean_hamming"]
        + aggregated["std_mean_hamming"]
    )

    aggregated = (
        aggregated
        .sort_values(["q", "t"])
        .reset_index(drop=True)
    )

    return aggregated


# ============================================================
# Linear fitting
# ============================================================

def linear_fit_with_error(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Fit the linear model

        y = intercept + beta * x

    and estimate the standard error of beta from the residuals.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    n_points = len(x)

    if n_points < MIN_FIT_POINTS:
        raise ValueError(
            f"At least {MIN_FIT_POINTS} valid points are "
            f"required, but only {n_points} were found."
        )

    if np.allclose(x, x[0]):
        raise ValueError(
            "All fitting times are identical, so the slope "
            "cannot be estimated."
        )

    design_matrix = np.column_stack([
        np.ones(n_points),
        x,
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        y,
        rcond=None,
    )

    intercept = float(coefficients[0])
    beta = float(coefficients[1])

    predicted = intercept + beta * x
    residuals = y - predicted

    residual_sum_squares = float(
        np.sum(residuals**2)
    )

    total_sum_squares = float(
        np.sum((y - np.mean(y)) ** 2)
    )

    degrees_of_freedom = n_points - 2

    if degrees_of_freedom > 0:
        residual_variance = (
            residual_sum_squares
            / degrees_of_freedom
        )

        covariance_matrix = (
            residual_variance
            * np.linalg.inv(
                design_matrix.T @ design_matrix
            )
        )

        beta_fit_error = float(
            np.sqrt(covariance_matrix[1, 1])
        )

    else:
        beta_fit_error = np.nan

    if total_sum_squares > 0:
        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )
    else:
        r_squared = np.nan

    return {
        "beta": beta,
        "intercept": intercept,
        "beta_fit_error": beta_fit_error,
        "r_squared": r_squared,
        "n_points": n_points,
    }


def fit_beta_per_iteration(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit beta independently for every q and iteration.

    The slope is estimated using the final N_FIT_POINTS time
    points of each trajectory.
    """
    fit_rows = []
    skipped_fits = []

    grouped = raw_data.groupby(
        ["q", "iteration"],
        sort=True,
    )

    for (q, iteration), iteration_data in grouped:
        iteration_data = (
            iteration_data
            .dropna(subset=["t", "mean_hamming"])
            .sort_values("t")
        )

        # Average duplicated time points within the same run.
        iteration_data = (
            iteration_data
            .groupby("t", as_index=False)
            .agg(
                mean_hamming=(
                    "mean_hamming",
                    "mean",
                ),
            )
            .sort_values("t")
        )

        fit_data = iteration_data.tail(
            N_FIT_POINTS
        )

        if len(fit_data) < MIN_FIT_POINTS:
            skipped_fits.append({
                "q": q,
                "iteration": iteration,
                "reason": (
                    f"only {len(fit_data)} valid time points"
                ),
            })
            continue

        try:
            fit_result = linear_fit_with_error(
                x=fit_data["t"].to_numpy(
                    dtype=float
                ),
                y=fit_data[
                    "mean_hamming"
                ].to_numpy(dtype=float),
            )

        except ValueError as error:
            skipped_fits.append({
                "q": q,
                "iteration": iteration,
                "reason": str(error),
            })
            continue

        fit_rows.append({
            "q": float(q),
            "iteration": iteration,
            "beta": fit_result["beta"],
            "intercept": fit_result["intercept"],
            "beta_fit_error": (
                fit_result["beta_fit_error"]
            ),
            "r_squared": fit_result["r_squared"],
            "n_fit_points": fit_result["n_points"],
            "t_fit_min": float(
                fit_data["t"].min()
            ),
            "t_fit_max": float(
                fit_data["t"].max()
            ),
        })

    if not fit_rows:
        raise RuntimeError(
            "No valid linear fits could be calculated."
        )

    beta_per_iteration = pd.DataFrame(
        fit_rows
    )

    beta_per_iteration = (
        beta_per_iteration
        .sort_values(["q", "iteration"])
        .reset_index(drop=True)
    )

    print(
        "\nLinear fits calculated using the final "
        f"{N_FIT_POINTS} points:"
    )

    fit_counts = (
        beta_per_iteration
        .groupby("q")
        .size()
    )

    for q, count in fit_counts.items():
        print(
            f"  q = {q:g}: "
            f"{count} fitted iterations"
        )

    if skipped_fits:
        print("\nSkipped linear fits:")

        for skipped in skipped_fits:
            print(
                f"  q={skipped['q']:g}, "
                f"iteration={skipped['iteration']}: "
                f"{skipped['reason']}"
            )

    return beta_per_iteration


def aggregate_beta_over_iterations(
    beta_per_iteration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize fitted beta values for every q.
    """
    beta_summary = (
        beta_per_iteration
        .groupby("q", as_index=False)
        .agg(
            average_beta=(
                "beta",
                "mean",
            ),
            median_beta=(
                "beta",
                "median",
            ),
            std_beta=(
                "beta",
                lambda values: values.std(
                    ddof=STD_DDOF
                ),
            ),
            n_iterations=(
                "beta",
                "count",
            ),
            mean_beta_fit_error=(
                "beta_fit_error",
                "mean",
            ),
            rms_fit_error=(
                "beta_fit_error",
                lambda values: np.sqrt(
                    np.nanmean(
                        np.asarray(
                            values,
                            dtype=float,
                        ) ** 2
                    )
                ),
            ),
            mean_r_squared=(
                "r_squared",
                "mean",
            ),
            min_r_squared=(
                "r_squared",
                "min",
            ),
        )
    )

    beta_summary["std_beta"] = (
        beta_summary["std_beta"]
        .fillna(0.0)
    )

    beta_summary["sem_beta"] = (
        beta_summary["std_beta"]
        / np.sqrt(beta_summary["n_iterations"])
    )

    beta_summary["combined_beta_error"] = np.sqrt(
        beta_summary["std_beta"] ** 2
        + beta_summary[
            "rms_fit_error"
        ].fillna(0.0) ** 2
    )

    return (
        beta_summary
        .sort_values("q")
        .reset_index(drop=True)
    )


# ============================================================
# Theoretical beta
# ============================================================

def theoretical_beta(
    q: np.ndarray | float,
    lambda_value: float = LAMBDA,
) -> np.ndarray:
    """
    Calculate the theoretical exploration slope

                      lambda (1 - q)
        beta(q) = -------------------------
                   1 - q (1 - p0)

    with

        p0 = exp(-lambda).

    The curve is evaluated over the complete q interval, including
    both sides of q_c, as requested.
    """
    q = np.asarray(q, dtype=float)

    p0 = np.exp(-lambda_value)

    denominator = (
        1.0
        - q * (1.0 - p0)
    )

    return (
        lambda_value
        * (1.0 - q)
        / denominator
    )


# ============================================================
# Utility functions for plotting
# ============================================================

def q_is_selected_for_label(q: float) -> bool:
    """
    Return True when q is included in LABEL_Q_VALUES.
    """
    return any(
        np.isclose(q, selected_q)
        for selected_q in LABEL_Q_VALUES
    )


def get_fit_window(
    raw_data: pd.DataFrame,
) -> tuple[float, float]:
    """
    Estimate the common fitting interval displayed in panel a.

    The interval is based on the final N_FIT_POINTS unique time
    values of the complete dataset.
    """
    unique_times = np.sort(
        raw_data["t"]
        .dropna()
        .unique()
        .astype(float)
    )

    if len(unique_times) == 0:
        raise RuntimeError(
            "No valid time values were found."
        )

    fit_times = unique_times[-N_FIT_POINTS:]

    return (
        float(np.min(fit_times)),
        float(np.max(fit_times)),
    )


def prepare_violin_distribution(
    values: np.ndarray,
) -> np.ndarray:
    """
    Prepare one beta distribution for matplotlib.violinplot.

    Matplotlib cannot estimate a KDE from one value or from several
    exactly identical values. In those cases, a negligible numerical
    perturbation is added only for visualization.

    The original values are still used to calculate means and medians.
    """
    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(
            "Cannot construct a violin from an empty distribution."
        )

    if len(values) == 1:
        epsilon = max(
            abs(values[0]) * 1e-6,
            1e-8,
        )

        return np.array([
            values[0] - epsilon,
            values[0],
            values[0] + epsilon,
        ])

    if np.allclose(values, values[0]):
        epsilon = max(
            abs(values[0]) * 1e-6,
            1e-8,
        )

        plotting_values = values.copy()

        plotting_values[0] -= epsilon
        plotting_values[-1] += epsilon

        return plotting_values

    return values


# ============================================================
# Panel a: mean-Hamming trajectories
# ============================================================

def plot_mean_hamming_panel(
    ax: plt.Axes,
    raw_data: pd.DataFrame,
    aggregated_data: pd.DataFrame,
    cmap,
    norm: Normalize,
) -> None:
    """
    Draw mean Hamming distance as a function of time.
    """
    q_values = np.sort(
        aggregated_data["q"]
        .dropna()
        .unique()
    )

    global_t_min = float(
        aggregated_data["t"].min()
    )

    global_t_max = float(
        aggregated_data["t"].max()
    )

    global_t_range = (
        global_t_max - global_t_min
    )

    if np.isclose(global_t_range, 0.0):
        global_t_range = 1.0

    label_x_offset = (
        LABEL_X_OFFSET_FRACTION
        * global_t_range
    )

    # --------------------------------------------------------
    # Show the interval used for the beta fits
    # --------------------------------------------------------

    fit_t_min, fit_t_max = get_fit_window(
        raw_data
    )

    ax.axvspan(
        fit_t_min,
        fit_t_max,
        facecolor="0.5",
        alpha=0.075,
        linewidth=0,
        zorder=0,
    )

    

    final_points = []

    # --------------------------------------------------------
    # Plot trajectories
    # --------------------------------------------------------

    for q in q_values:
        color = cmap(norm(q))

        q_aggregated = (
            aggregated_data[
                np.isclose(
                    aggregated_data["q"],
                    q,
                )
            ]
            .sort_values("t")
        )

        if q_aggregated.empty:
            continue

        t = q_aggregated[
            "t"
        ].to_numpy(dtype=float)

        mean_hamming = q_aggregated[
            "average_mean_hamming"
        ].to_numpy(dtype=float)

        lower = q_aggregated[
            "lower_std"
        ].to_numpy(dtype=float)

        upper = q_aggregated[
            "upper_std"
        ].to_numpy(dtype=float)

        if SHOW_INDIVIDUAL_ITERATIONS:
            q_raw = raw_data[
                np.isclose(
                    raw_data["q"],
                    q,
                )
            ]

            for _, iteration_data in q_raw.groupby(
                "iteration"
            ):
                iteration_data = (
                    iteration_data
                    .sort_values("t")
                )

                ax.plot(
                    iteration_data["t"],
                    iteration_data[
                        "mean_hamming"
                    ],
                    color=color,
                    linewidth=0.6,
                    alpha=0.09,
                    zorder=1,
                )

        ax.fill_between(
            t,
            lower,
            upper,
            color=color,
            alpha=TRAJECTORY_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )

        ax.plot(
            t,
            mean_hamming,
            color=color,
            linewidth=1.9,
            solid_capstyle="round",
            zorder=3,
        )

        if q_is_selected_for_label(q):
            final_points.append({
                "q": float(q),
                "x": float(t[-1]),
                "y": float(mean_hamming[-1]),
                "color": color,
            })

    # --------------------------------------------------------
    # Direct labels
    # --------------------------------------------------------
    #
    # Labels are placed at the exact final y value of each selected
    # trajectory. This avoids the misleading manual vertical offsets
    # used previously. A short horizontal leader segment makes the
    # correspondence between each curve and its label explicit.
    # --------------------------------------------------------

    for point in final_points:
        q_value = point["q"]
        label_y = point["y"]

        leader_end_x = (
            point["x"]
            + LABEL_LEADER_FRACTION * label_x_offset
        )

        ax.plot(
            [
                point["x"],
                leader_end_x,
            ],
            [
                label_y,
                label_y,
            ],
            color=point["color"],
            linewidth=0.8,
            alpha=0.9,
            clip_on=False,
            zorder=4,
        )

        ax.text(
            point["x"] + label_x_offset,
            label_y,
            rf"$q={q_value:g}$",
            color=point["color"],
            fontsize=8.8,
            fontweight="medium",
            ha="left",
            va="center",
            clip_on=False,
            zorder=5,
        )

    # Provide horizontal space for the labels.
    right_margin = (
        global_t_max
        + 7.0 * label_x_offset
    )

    ax.set_xlim(
        global_t_min,
        right_margin,
    )

    ax.set_xlabel(r"Time, $t$")

    ax.set_ylabel(
        r"Mean Hamming distance, $\langle m\rangle$"
    )

    ax.grid(
        True,
        which="major",
        linewidth=0.7,
        alpha=0.22,
        zorder=0,
    )

    # Panel label.
    ax.text(
        0.025,
        0.975,
        r"$\mathbf{a}$",
        transform=ax.transAxes,
        fontsize=15,
        ha="left",
        va="top",
    )


# ============================================================
# Panel b: beta distributions
# ============================================================

def plot_beta_panel(
    ax: plt.Axes,
    beta_summary: pd.DataFrame,
    beta_per_iteration: pd.DataFrame,
    cmap,
    norm: Normalize,
) -> None:
    """
    Draw the distribution of fitted beta values for every q.

    Violin:
        Distribution across independent iterations.

    Circle:
        Mean beta.

    Short horizontal segment:
        Median beta.
    """
    q_values = (
        beta_summary["q"]
        .to_numpy(dtype=float)
    )

    violin_distributions = []
    original_distributions = []
    valid_q_values = []

    for q_value in q_values:
        original_values = (
            beta_per_iteration.loc[
                np.isclose(
                    beta_per_iteration["q"],
                    q_value,
                ),
                "beta",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(original_values) == 0:
            continue

        plotting_values = prepare_violin_distribution(
            original_values
        )

        original_distributions.append(
            original_values
        )

        violin_distributions.append(
            plotting_values
        )

        valid_q_values.append(
            float(q_value)
        )

    if not violin_distributions:
        raise RuntimeError(
            "No beta distributions are available "
            "for the violin plot."
        )

    valid_q_values = np.asarray(
        valid_q_values,
        dtype=float,
    )

    # --------------------------------------------------------
    # Violin distributions
    # --------------------------------------------------------

    violin_parts = ax.violinplot(
        dataset=violin_distributions,
        positions=valid_q_values,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )

    for body, q_value in zip(
        violin_parts["bodies"],
        valid_q_values,
    ):
        color = cmap(norm(q_value))

        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(VIOLIN_ALPHA)
        body.set_linewidth(1.0)

    # --------------------------------------------------------
    # Mean and median
    # --------------------------------------------------------

    for q_value, values in zip(
        valid_q_values,
        original_distributions,
    ):
        color = cmap(norm(q_value))

        mean_beta = float(
            np.mean(values)
        )        

        # Mean shown as a colored circle.
        ax.scatter(
            q_value,
            mean_beta,
            s=45,
            marker="o",
            facecolor=color,
            edgecolor="black",
            linewidth=0.75,
            zorder=7,
        )

    # --------------------------------------------------------
    # Theoretical curve across the complete q interval
    # --------------------------------------------------------

    q_theory = np.linspace(
        0.0,
        1.0,
        600,
    )

    beta_theory = theoretical_beta(
        q=q_theory,
        lambda_value=LAMBDA,
    )

    ax.plot(
        q_theory,
        beta_theory,
        color="black",
        linestyle="--",
        linewidth=1.65,
        dash_capstyle="butt",
        label=r"Theoretical $\beta(q)$",
        zorder=4,
    )

    # --------------------------------------------------------
    # Critical transition
    # --------------------------------------------------------

    ax.axvline(
        Q_CRITICAL,
        color="0.38",
        linestyle="--",
        linewidth=1.25,
        alpha=0.95,
        zorder=3,
    )

    # q_c annotation placed directly beside the transition line.
    ax.annotate(
        rf"$q_c\simeq {Q_CRITICAL:.2f}$",
        xy=(Q_CRITICAL, 0.965),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9.5,
        color="0.25",
        ha="left",
        va="top",
    )

    # --------------------------------------------------------
    # Regime labels
    # --------------------------------------------------------

    ax.text(
        0.20,
        1.025,
        "free exploration",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
        color="0.35",
        ha="center",
        va="bottom",
        clip_on=False,
    )

    ax.text(
        0.72,
        1.025,
        "peak trapping",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
        color="0.35",
        ha="center",
        va="bottom",
        clip_on=False,
    )

    # --------------------------------------------------------
    # Zero reference
    # --------------------------------------------------------

    ax.axhline(
        0.0,
        color="0.50",
        linewidth=0.9,
        linestyle=":",
        alpha=0.85,
        zorder=1,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.65,
            label=r"Theoretical $\beta(q)$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            markerfacecolor="0.65",
            markeredgecolor="black",
            linestyle="none",
            label="Mean",
        ),
        
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        handlelength=2.5,
        borderaxespad=0.5,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(
        -0.035,
        1.035,
    )

    ax.set_xticks(
        np.linspace(0.0, 1.0, 6)
    )

    # Determine suitable vertical limits from both theoretical and
    # empirical values.
    all_empirical_beta = (
        beta_per_iteration["beta"]
        .dropna()
        .to_numpy(dtype=float)
    )

    empirical_min = float(
        np.min(all_empirical_beta)
    )

    empirical_max = float(
        np.max(all_empirical_beta)
    )

    theory_min = float(
        np.min(beta_theory)
    )

    theory_max = float(
        np.max(beta_theory)
    )

    y_min = min(
        empirical_min,
        theory_min,
        0.0,
    )

    y_max = max(
        empirical_max,
        theory_max,
    )

    y_range = y_max - y_min

    if np.isclose(y_range, 0.0):
        y_range = 1.0

    ax.set_ylim(
        y_min - 0.08 * y_range,
        y_max + 0.13 * y_range,
    )

    ax.set_xlabel(
        r"Holeyness, $q$"
    )

    ax.set_ylabel(
        r"Late-time slope, $\beta$"
    )

    ax.grid(
        True,
        which="major",
        linewidth=0.7,
        alpha=0.22,
        zorder=0,
    )

    # Panel label.
    ax.text(
        0.025,
        0.975,
        r"$\mathbf{b}$",
        transform=ax.transAxes,
        fontsize=15,
        ha="left",
        va="top",
    )


# ============================================================
# Combined panel
# ============================================================

def plot_combined_panel(
    raw_data: pd.DataFrame,
    aggregated_data: pd.DataFrame,
    beta_summary: pd.DataFrame,
    beta_per_iteration: pd.DataFrame,
) -> None:
    """
    Create and save the complete two-panel figure.
    """
    all_q_values = np.sort(
        raw_data["q"]
        .dropna()
        .unique()
    )

    if len(all_q_values) == 0:
        raise RuntimeError(
            "No q values are available for plotting."
        )

    q_min = float(
        np.min(all_q_values)
    )

    q_max = float(
        np.max(all_q_values)
    )

    # Since q is a probability, normalize the colors over [0, 1].
    norm = Normalize(
        vmin=min(0.0, q_min),
        vmax=max(1.0, q_max),
    )

    cmap = plt.colormaps["viridis"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.2, 5.25),
        gridspec_kw={
            "width_ratios": [1.15, 1.0],
            "wspace": 0.25,
        },
    )

    plot_mean_hamming_panel(
        ax=axes[0],
        raw_data=raw_data,
        aggregated_data=aggregated_data,
        cmap=cmap,
        norm=norm,
    )

    plot_beta_panel(
        ax=axes[1],
        beta_summary=beta_summary,
        beta_per_iteration=beta_per_iteration,
        cmap=cmap,
        norm=norm,
    )

    # Leave room above panel b for the regime labels.
    fig.subplots_adjust(
        top=0.91,
        bottom=0.14,
        left=0.075,
        right=0.985,
    )

    output_base = (
        FIGURE_DIR
        / PANEL_OUTPUT_BASENAME
    )

    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nCombined panel figure saved as:")

    print(
        f"  {output_base.with_suffix('.png').resolve()}"
    )

    print(
        f"  {output_base.with_suffix('.pdf').resolve()}"
    )

    print(
        f"  {output_base.with_suffix('.svg').resolve()}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # Read and aggregate mean-Hamming trajectories
    # --------------------------------------------------------

    raw_data = collect_means_data()

    aggregated_data = aggregate_over_iterations(
        raw_data
    )

    raw_output_path = (
        FIGURE_DIR
        / f"{MODEL}_mean_hamming_raw_data.csv"
    )

    aggregated_output_path = (
        FIGURE_DIR
        / (
            f"{MODEL}_mean_hamming_"
            f"averaged_over_iterations.csv"
        )
    )

    raw_data.to_csv(
        raw_output_path,
        index=False,
    )

    aggregated_data.to_csv(
        aggregated_output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Fit beta separately for every independent iteration
    # --------------------------------------------------------

    beta_per_iteration = fit_beta_per_iteration(
        raw_data
    )

    beta_summary = aggregate_beta_over_iterations(
        beta_per_iteration
    )

    beta_per_iteration_output = (
        FIGURE_DIR
        / (
            f"{MODEL}_beta_per_iteration_"
            f"last_{N_FIT_POINTS}_points.csv"
        )
    )

    beta_summary_output = (
        FIGURE_DIR
        / (
            f"{MODEL}_beta_summary_"
            f"last_{N_FIT_POINTS}_points.csv"
        )
    )

    beta_per_iteration.to_csv(
        beta_per_iteration_output,
        index=False,
    )

    beta_summary.to_csv(
        beta_summary_output,
        index=False,
    )

    # --------------------------------------------------------
    # Produce the combined figure
    # --------------------------------------------------------

    plot_combined_panel(
        raw_data=raw_data,
        aggregated_data=aggregated_data,
        beta_summary=beta_summary,
        beta_per_iteration=beta_per_iteration,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    fit_t_min, fit_t_max = get_fit_window(
        raw_data
    )

    print("\nParameters used:")

    print(
        f"  lambda = {LAMBDA:g}"
    )

    print(
        f"  p0 = exp(-lambda) = {P0:.8f}"
    )

    print(
        f"  q_c = {Q_CRITICAL:.2f}"
    )

    print(
        f"  final fitting points = {N_FIT_POINTS}"
    )

    print(
        f"  displayed fit interval = "
        f"[{fit_t_min:g}, {fit_t_max:g}]"
    )

    print("\nMean-Hamming tables saved as:")

    print(
        f"  {raw_output_path.resolve()}"
    )

    print(
        f"  {aggregated_output_path.resolve()}"
    )

    print("\nBeta tables saved as:")

    print(
        f"  {beta_per_iteration_output.resolve()}"
    )

    print(
        f"  {beta_summary_output.resolve()}"
    )

    print("\nBeta summary:")

    print(
        beta_summary[
            [
                "q",
                "average_beta",
                "median_beta",
                "std_beta",
                "sem_beta",
                "n_iterations",
                "mean_r_squared",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()