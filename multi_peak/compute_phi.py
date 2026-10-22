#!/usr/bin/env python3
"""
compute_phi.py -- overlap angle between consecutive populations (multi-peak)

For every iteration and every pair of consecutive snapshots computes
    cos phi(t) = <n_t, n_{t+1}> / (|n_t| |n_{t+1}|),
the angle between the genotype-abundance vectors of generations t and t+1 (Chapter 3 definition), and
writes <results>/<stem>_phi_consecutive.csv plus pooled tables. Needed by fig_phi.py.
Port of the thesis notebook 16-Simulations/old/plots-multipeak.ipynb (cell 5).

    python compute_phi.py --data Data/multi_peak_delta15
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import argparse
_ap = argparse.ArgumentParser(description="Overlap angle between consecutive populations")
_ap.add_argument("--data", default="Data/multi_peak", help="model directory with Abundances/ (Means/, Distribs/ are written there)")
_ap.add_argument("--results", default=None, help="derived tables (default: <data>/results)")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--filter", default=None, help="substring selecting one parameter set, e.g. nrw_1000_L_10000_q_0.0")
ARGS = _ap.parse_args()
_RESULTS = ARGS.results or str(Path(ARGS.data) / "results")

"""
Compute the overlap angle phi between consecutive model4a30 populations.

For every simulation and pair of consecutive generations, the population is
represented by an abundance vector

    v(t) = [n_g1(t), n_g2(t), ..., n_gM(t)],

where each coordinate corresponds to one genotype and n_g(t) is its abundance.

The two snapshots are aligned on the union of their genotype identities, so
the same coordinate always represents the same genotype:

    G(t-1,t) = G(t-1) union G(t).

The cosine overlap and angle are

    cos(phi_t) =
        v(t-1) dot v(t)
        --------------------------------
        ||v(t-1)|| ||v(t)||

    phi_t = arccos[cos(phi_t)].

The pair is labeled by the later generation t, so row t compares t-1 and t.

Efficiency
----------
Each abundance snapshot is read only once. Within every simulation, the code
keeps the previous Counter in memory, reads the next snapshot, aligns the two
abundance vectors, computes the overlap, and then discards the older snapshot.

Outputs
-------
For every independent simulation:

    ./Computational_results/model4a30/
        dataOut_<parameters>_phi_consecutive.csv

A parameter-set summary across simulations:

    ./Computational_results/model4a30/
        model4a30_phi_consecutive_<analysis_id>_summary.csv

A complete table containing every simulation:

    ./Computational_results/model4a30/
        model4a30_phi_consecutive_<analysis_id>_all_simulations.csv

Optional aligned vectors can also be saved, although this is disabled by
default because it can create many large files.

Designed to run directly in a Jupyter notebook:

    %run model4a30_compute_phi_consecutive.py
"""


from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

ABUNDANCE_DIR = Path(ARGS.data) / "Abundances"
OUTPUT_DIR = Path(_RESULTS)

# Select exactly one parameter set when ABUNDANCE_DIR contains several.
#
# Example:
# FILE_FILTER = "nrw_1000_L_10000_q_0.0_fpeak_3_tau_40_delta_15"
FILE_FILTER = ARGS.filter

# Only compare files whose generation difference is exactly one.
# Missing generations are reported and skipped.
REQUIRE_UNIT_TIME_STEP = True

# Numerical tolerance used before arccos.
COSINE_CLIP_TOLERANCE = 1e-12

# Saving every aligned genotype vector is useful for verification but can
# produce a very large number of files. Leave False for routine analysis.
SAVE_ALIGNED_VECTORS = False

ALIGNED_VECTOR_DIR = (
    OUTPUT_DIR
    / "Phi_aligned_vectors"
)


# =============================================================================
# Filename parsing
# =============================================================================

SNAPSHOT_PATTERN = re.compile(
    r"^(?P<base>.+)_t_(?P<time>\d+)\.txt$"
)

ITERATION_PATTERN = re.compile(
    r"_iteration_(?P<iteration>\d+)"
)


def parse_snapshot_filename(filepath: Path) -> dict[str, Any]:
    """
    Extract the per-simulation base name, iteration, and generation.
    """
    snapshot_match = SNAPSHOT_PATTERN.match(
        filepath.name
    )

    iteration_match = ITERATION_PATTERN.search(
        filepath.name
    )

    missing = []

    if snapshot_match is None:
        missing.append("time/base")

    if iteration_match is None:
        missing.append("iteration")

    if missing:
        raise ValueError(
            f"Could not extract {', '.join(missing)} from:\n"
            f"{filepath.name}"
        )

    return {
        "simulation_base": snapshot_match.group("base"),
        "iteration": int(
            iteration_match.group("iteration")
        ),
        "time": int(
            snapshot_match.group("time")
        ),
    }


def parameter_signature(filepath: Path) -> str:
    """
    Remove iteration and time so all independent simulations belonging to
    the same parameter set share one signature.
    """
    signature = ITERATION_PATTERN.sub(
        "_iteration_*",
        filepath.name,
    )

    signature = re.sub(
        r"_t_\d+\.txt$",
        "_t_*.txt",
        signature,
    )

    return signature


def analysis_hash(signature: str) -> str:
    """Stable compact identifier for output filenames."""
    return hashlib.blake2b(
        signature.encode("utf-8"),
        digest_size=6,
        person=b"m4a30phi",
    ).hexdigest()


# =============================================================================
# Population representation
# =============================================================================

def normalize_genotype_text(line: str) -> str:
    """
    Return the canonical genotype identity written by the simulator.
    """
    genotype = line.strip()

    if not genotype or genotype == "--":
        return "--"

    return genotype


def read_population_counter(filepath: Path) -> Counter[str]:
    """
    Read one abundance snapshot as genotype -> abundance.
    """
    counts: Counter[str] = Counter()

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            if not line.strip():
                continue

            genotype = normalize_genotype_text(
                line
            )

            counts[genotype] += 1

    if not counts:
        raise ValueError(
            f"Empty abundance snapshot:\n{filepath}"
        )

    return counts


def align_population_vectors(
    previous_counts: Counter[str],
    current_counts: Counter[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Construct an exact common genotype index for two snapshots.

    The common index is the union of genotype identities. The ordering of the
    union is irrelevant to the cosine as long as both vectors use the same one.

    Returns
    -------
    genotype_index
        Genotype identity at every coordinate.

    previous_vector
        Abundance vector for the earlier snapshot.

    current_vector
        Abundance vector for the later snapshot.
    """
    genotype_index = list(
        previous_counts.keys()
        | current_counts.keys()
    )

    previous_vector = np.fromiter(
        (
            previous_counts.get(
                genotype,
                0,
            )
            for genotype in genotype_index
        ),
        dtype=np.float64,
        count=len(genotype_index),
    )

    current_vector = np.fromiter(
        (
            current_counts.get(
                genotype,
                0,
            )
            for genotype in genotype_index
        ),
        dtype=np.float64,
        count=len(genotype_index),
    )

    return (
        genotype_index,
        previous_vector,
        current_vector,
    )


# =============================================================================
# Pair statistics
# =============================================================================

def effective_genotype_number(
    abundance_vector: np.ndarray,
) -> float:
    """
    Inverse Simpson effective number based on genotype frequencies.

        N_eff = 1 / sum_g p_g^2
    """
    population_size = float(
        abundance_vector.sum()
    )

    if population_size <= 0.0:
        return math.nan

    frequencies = (
        abundance_vector
        / population_size
    )

    concentration = float(
        np.dot(
            frequencies,
            frequencies,
        )
    )

    if concentration <= 0.0:
        return math.nan

    return 1.0 / concentration


def compute_pair_statistics(
    previous_counts: Counter[str],
    current_counts: Counter[str],
) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray]:
    """
    Align two population vectors and calculate their overlap angle.
    """
    (
        genotype_index,
        previous_vector,
        current_vector,
    ) = align_population_vectors(
        previous_counts=previous_counts,
        current_counts=current_counts,
    )

    dot_product = float(
        np.dot(
            previous_vector,
            current_vector,
        )
    )

    norm_previous = float(
        np.linalg.norm(
            previous_vector
        )
    )

    norm_current = float(
        np.linalg.norm(
            current_vector
        )
    )

    if (
        norm_previous <= 0.0
        or norm_current <= 0.0
    ):
        raise ValueError(
            "At least one population abundance vector "
            "has zero Euclidean norm."
        )

    raw_cosine = (
        dot_product
        / (
            norm_previous
            * norm_current
        )
    )

    # Floating-point roundoff can produce values such as 1 + 2e-16.
    if (
        raw_cosine < -1.0 - COSINE_CLIP_TOLERANCE
        or raw_cosine > 1.0 + COSINE_CLIP_TOLERANCE
    ):
        raise ArithmeticError(
            "Cosine overlap lies outside [-1,1] beyond "
            f"the allowed tolerance: {raw_cosine}"
        )

    cosine_phi = float(
        np.clip(
            raw_cosine,
            -1.0,
            1.0,
        )
    )

    phi_radians = float(
        np.arccos(
            cosine_phi
        )
    )

    phi_degrees = float(
        np.degrees(
            phi_radians
        )
    )

    shared_genotypes = (
        previous_counts.keys()
        & current_counts.keys()
    )

    union_genotypes = (
        previous_counts.keys()
        | current_counts.keys()
    )

    population_previous = int(
        previous_vector.sum()
    )

    population_current = int(
        current_vector.sum()
    )

    frequency_previous = (
        previous_vector
        / population_previous
    )

    frequency_current = (
        current_vector
        / population_current
    )

    l1_frequency_distance = float(
        np.sum(
            np.abs(
                frequency_current
                - frequency_previous
            )
        )
    )

    total_variation_distance = (
        0.5
        * l1_frequency_distance
    )

    statistics = {
        "cos_phi": cosine_phi,
        "phi_radians": phi_radians,
        "phi_degrees": phi_degrees,
        "dot_product": dot_product,
        "norm_previous": norm_previous,
        "norm_current": norm_current,
        "norm_squared_previous": (
            norm_previous**2
        ),
        "norm_squared_current": (
            norm_current**2
        ),
        "population_size_previous": (
            population_previous
        ),
        "population_size_current": (
            population_current
        ),
        "n_genotypes_previous": len(
            previous_counts
        ),
        "n_genotypes_current": len(
            current_counts
        ),
        "n_shared_genotypes": len(
            shared_genotypes
        ),
        "n_union_genotypes": len(
            union_genotypes
        ),
        "genotype_jaccard": (
            len(shared_genotypes)
            / len(union_genotypes)
        ),
        "total_variation_distance": (
            total_variation_distance
        ),
        "effective_genotypes_previous": (
            effective_genotype_number(
                previous_vector
            )
        ),
        "effective_genotypes_current": (
            effective_genotype_number(
                current_vector
            )
        ),
    }

    return (
        statistics,
        genotype_index,
        previous_vector,
        current_vector,
    )


# =============================================================================
# Optional aligned-vector export
# =============================================================================

def save_aligned_vectors(
    simulation_base: str,
    time_previous: int,
    time_current: int,
    genotype_index: list[str],
    previous_vector: np.ndarray,
    current_vector: np.ndarray,
) -> Path:
    """
    Save the common genotype index and both abundance vectors.

    Rows for genotypes absent from one population contain zero in that column.
    """
    ALIGNED_VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ALIGNED_VECTOR_DIR
        / (
            f"{simulation_base}"
            f"_t_{time_previous}_to_{time_current}"
            "_aligned_vectors.csv"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file
        )

        writer.writerow([
            "genotype",
            "abundance_previous",
            "abundance_current",
        ])

        for (
            genotype,
            abundance_previous,
            abundance_current,
        ) in zip(
            genotype_index,
            previous_vector,
            current_vector,
        ):
            writer.writerow([
                genotype,
                int(abundance_previous),
                int(abundance_current),
            ])

    return output_path


# =============================================================================
# Locate selected snapshots
# =============================================================================

if not ABUNDANCE_DIR.exists():
    raise FileNotFoundError(
        f"Abundance directory not found:\n"
        f"{ABUNDANCE_DIR.resolve()}"
    )

abundance_files = sorted(
    ABUNDANCE_DIR.glob(
        "dataOut_*_t_*.txt"
    )
)

if FILE_FILTER is not None:
    abundance_files = [
        filepath
        for filepath in abundance_files
        if FILE_FILTER in filepath.name
    ]

if not abundance_files:
    raise FileNotFoundError(
        "No abundance snapshots matched "
        "ABUNDANCE_DIR and FILE_FILTER."
    )

signatures = {
    parameter_signature(filepath)
    for filepath in abundance_files
}

if len(signatures) > 1:
    print("Multiple parameter sets were detected:\n")

    for detected_signature in sorted(
        signatures
    ):
        print(f"  {detected_signature}")

    raise ValueError(
        "\nSet FILE_FILTER so that exactly one "
        "parameter set is selected."
    )

selected_signature = next(
    iter(signatures)
)

selected_analysis_id = analysis_hash(
    selected_signature
)


# =============================================================================
# Group snapshots by independent simulation
# =============================================================================

files_by_simulation: dict[
    str,
    list[tuple[int, Path]]
] = defaultdict(list)

iteration_by_simulation: dict[
    str,
    int
] = {}

for filepath in abundance_files:
    parsed = parse_snapshot_filename(
        filepath
    )

    simulation_base = parsed[
        "simulation_base"
    ]

    files_by_simulation[
        simulation_base
    ].append(
        (
            parsed["time"],
            filepath,
        )
    )

    iteration_by_simulation[
        simulation_base
    ] = parsed["iteration"]

for simulation_base in files_by_simulation:
    files_by_simulation[
        simulation_base
    ].sort(
        key=lambda pair: pair[0]
    )


# =============================================================================
# Compute phi for every consecutive pair
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

all_pair_records: list[
    dict[str, Any]
] = []

skipped_gaps: list[
    tuple[str, int, int]
] = []

for simulation_index, simulation_base in enumerate(
    sorted(files_by_simulation),
    start=1,
):
    time_file_pairs = files_by_simulation[
        simulation_base
    ]

    if len(time_file_pairs) < 2:
        print(
            "Warning: simulation has fewer than two snapshots "
            f"and will be skipped:\n  {simulation_base}"
        )
        continue

    simulation_records: list[
        dict[str, Any]
    ] = []

    time_previous, filepath_previous = (
        time_file_pairs[0]
    )

    previous_counts = read_population_counter(
        filepath_previous
    )

    for (
        time_current,
        filepath_current,
    ) in time_file_pairs[1:]:
        time_step = (
            time_current
            - time_previous
        )

        if (
            REQUIRE_UNIT_TIME_STEP
            and time_step != 1
        ):
            skipped_gaps.append(
                (
                    simulation_base,
                    time_previous,
                    time_current,
                )
            )

            # Continue the chain from the current file.
            current_counts = read_population_counter(
                filepath_current
            )

            previous_counts = current_counts
            time_previous = time_current
            filepath_previous = filepath_current
            continue

        current_counts = read_population_counter(
            filepath_current
        )

        (
            statistics,
            genotype_index,
            previous_vector,
            current_vector,
        ) = compute_pair_statistics(
            previous_counts=previous_counts,
            current_counts=current_counts,
        )

        aligned_vector_filename = ""

        if SAVE_ALIGNED_VECTORS:
            aligned_vector_path = save_aligned_vectors(
                simulation_base=simulation_base,
                time_previous=time_previous,
                time_current=time_current,
                genotype_index=genotype_index,
                previous_vector=previous_vector,
                current_vector=current_vector,
            )

            aligned_vector_filename = (
                aligned_vector_path.name
            )

        record = {
            "iteration": (
                iteration_by_simulation[
                    simulation_base
                ]
            ),
            "simulation_base": simulation_base,
            "time_previous": time_previous,
            "time_current": time_current,
            "time": time_current,
            "dt": time_step,
            **statistics,
            "source_previous": (
                filepath_previous.name
            ),
            "source_current": (
                filepath_current.name
            ),
            "aligned_vector_filename": (
                aligned_vector_filename
            ),
        }

        simulation_records.append(
            record
        )

        all_pair_records.append(
            record
        )

        previous_counts = current_counts
        time_previous = time_current
        filepath_previous = filepath_current

    simulation_output_path = (
        OUTPUT_DIR
        / (
            f"{simulation_base}"
            "_phi_consecutive.csv"
        )
    )

    pd.DataFrame(
        simulation_records
    ).to_csv(
        simulation_output_path,
        index=False,
    )

    if (
        simulation_index == 1
        or simulation_index % 25 == 0
        or simulation_index
        == len(files_by_simulation)
    ):
        print(
            f"Processed {simulation_index}/"
            f"{len(files_by_simulation)} simulations"
        )


# =============================================================================
# Combined and ensemble-summary outputs
# =============================================================================

all_pairs = pd.DataFrame(
    all_pair_records
)

if all_pairs.empty:
    raise RuntimeError(
        "No consecutive population pairs were analyzed."
    )

all_pairs = (
    all_pairs
    .sort_values(
        [
            "iteration",
            "time_current",
        ]
    )
    .reset_index(drop=True)
)

all_simulations_path = (
    OUTPUT_DIR
    / (
        "model4a30_phi_consecutive_"
        f"{selected_analysis_id}"
        "_all_simulations.csv"
    )
)

all_pairs.to_csv(
    all_simulations_path,
    index=False,
)


summary = (
    all_pairs
    .groupby(
        "time_current",
        as_index=False,
    )
    .agg(
        time_previous=(
            "time_previous",
            "first",
        ),
        mean_cos_phi=(
            "cos_phi",
            "mean",
        ),
        std_cos_phi=(
            "cos_phi",
            "std",
        ),
        min_cos_phi=(
            "cos_phi",
            "min",
        ),
        max_cos_phi=(
            "cos_phi",
            "max",
        ),
        mean_phi_radians=(
            "phi_radians",
            "mean",
        ),
        std_phi_radians=(
            "phi_radians",
            "std",
        ),
        mean_phi_degrees=(
            "phi_degrees",
            "mean",
        ),
        std_phi_degrees=(
            "phi_degrees",
            "std",
        ),
        mean_shared_genotypes=(
            "n_shared_genotypes",
            "mean",
        ),
        mean_union_genotypes=(
            "n_union_genotypes",
            "mean",
        ),
        mean_genotype_jaccard=(
            "genotype_jaccard",
            "mean",
        ),
        mean_total_variation_distance=(
            "total_variation_distance",
            "mean",
        ),
        n_simulations=(
            "iteration",
            "nunique",
        ),
    )
    .sort_values(
        "time_current"
    )
    .reset_index(drop=True)
)

for column in [
    "std_cos_phi",
    "std_phi_radians",
    "std_phi_degrees",
]:
    summary[column] = (
        summary[column]
        .fillna(0.0)
    )

# Because arccos is nonlinear, this is generally not identical to the mean
# of the individual angles. Both quantities are saved.
summary[
    "phi_from_mean_cos_radians"
] = np.arccos(
    np.clip(
        summary["mean_cos_phi"],
        -1.0,
        1.0,
    )
)

summary[
    "phi_from_mean_cos_degrees"
] = np.degrees(
    summary[
        "phi_from_mean_cos_radians"
    ]
)

summary_path = (
    OUTPUT_DIR
    / (
        "model4a30_phi_consecutive_"
        f"{selected_analysis_id}"
        "_summary.csv"
    )
)

summary.to_csv(
    summary_path,
    index=False,
)


# =============================================================================
# Report
# =============================================================================

print()
print("Analysis complete")
print("-----------------")
print(
    f"Selected signature:\n  "
    f"{selected_signature}"
)
print(
    f"Independent simulations: "
    f"{all_pairs['iteration'].nunique()}"
)
print(
    f"Consecutive pairs analyzed: "
    f"{len(all_pairs)}"
)
print(
    f"Time range: "
    f"{all_pairs['time_current'].min()}–"
    f"{all_pairs['time_current'].max()}"
)

if skipped_gaps:
    print(
        f"Non-unit generation gaps skipped: "
        f"{len(skipped_gaps)}"
    )

    for (
        simulation_base,
        time_previous,
        time_current,
    ) in skipped_gaps[:10]:
        print(
            f"  {simulation_base}: "
            f"{time_previous} -> {time_current}"
        )

    if len(skipped_gaps) > 10:
        print(
            f"  ... and "
            f"{len(skipped_gaps) - 10} more"
        )

print(
    f"All-simulation results:\n  "
    f"{all_simulations_path.resolve()}"
)
print(
    f"Ensemble summary:\n  "
    f"{summary_path.resolve()}"
)

if SAVE_ALIGNED_VECTORS:
    print(
        f"Aligned pair vectors:\n  "
        f"{ALIGNED_VECTOR_DIR.resolve()}"
    )