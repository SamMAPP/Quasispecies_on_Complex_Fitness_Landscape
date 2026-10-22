#!/usr/bin/env python3
"""
compute_distances.py -- Hamming distances from the origin and from the instantaneous root (multi-peak)

For every abundance snapshot of a multi-peak run (generate_multi_peak.py) computes the Hamming distance
of every individual from
    origin  the ancestral genotype g0 = 0...0, and
    root    the instantaneous root, the most abundant genotype of the snapshot (ties: closest to g0,
            then lexicographically smallest),
and writes
    <data>/Means/<stem>_ref_origin.csv, <stem>_ref_root.csv      t, mean, variance (per iteration)
    <data>/Distribs/<stem>_t_<t>_ref_origin.txt, _ref_root.txt  one distance per individual
    <results>/<stem>_root_log.csv, root_catalog.csv               root genotype, abundance, changes
Snapshots already processed are reused (cache keyed on file size and mtime).
Needed by fig_mean_hamming.py, fig_variance_dispersion.py and compute_phi.py.
Port of the thesis notebook 16-Simulations/old/plots-multipeak.ipynb (cell 0).

    python compute_distances.py --data Data/multi_peak_delta15
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import argparse
_ap = argparse.ArgumentParser(description="Hamming distances from origin and instantaneous root")
_ap.add_argument("--data", default="Data/multi_peak", help="model directory with Abundances/ (Means/, Distribs/ are written there)")
_ap.add_argument("--results", default=None, help="derived tables (default: <data>/results)")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--filter", default=None, help="substring selecting one parameter set, e.g. nrw_1000_L_10000_q_0.0")
ARGS = _ap.parse_args()
_RESULTS = ARGS.results or str(Path(ARGS.data) / "results")

"""
Merged Hamming-distance analysis for model4a30.

Designed to be run directly in a Jupyter notebook with:

    %run model4a30_hamming_analysis.py

or by pasting the complete file into one notebook cell.

For every abundance snapshot in

    ./Data/model4a30/Abundances/

the script computes distances using two references:

1. origin
   The ancestral all-zero genotype.

2. root
   The instantaneous root, defined as the most abundant genotype in the
   current population snapshot.

No plots are produced.

Outputs
=======

A. Per-simulation mean and variance time series
------------------------------------------------
Directory:

    ./Data/model4a30/Means/

For each simulation/iteration, two CSV files are written:

    dataOut_<parameters>_ref_origin.csv
    dataOut_<parameters>_ref_root.csv

The original suffix ``_t_<time>.txt`` is removed. Each CSV contains exactly:

    time,mean_hamming,variance_hamming


B. Per-sequence distance distributions
---------------------------------------
Directory:

    ./Data/model4a30/Distribs/

For every abundance snapshot, two files are written:

    dataOut_<parameters>_t_<time>_ref_origin.txt
    dataOut_<parameters>_t_<time>_ref_root.txt

Each non-empty line contains only the integer Hamming distance of one sequence
from the corresponding reference genotype. Population order is preserved.


C. Reusable instantaneous-root logs
-----------------------------------
Directory:

    ./Computational_results/model4a30/

For each simulation/iteration:

    dataOut_<parameters>_root_log.csv

The log stores the exact root genotype, a stable root ID, its abundance and
frequency, tie information, root changes through time, both sets of moments,
and source-file metadata used to validate the cache.

A global catalog is also created:

    root_catalog.csv

It contains one row per unique root ID found in every root log currently
stored in the results directory.

Caching
=======
Root identification is the expensive part. A snapshot is reused from its
existing root log when:

* cache_version matches;
* source filename, size, and nanosecond modification time match;
* both corresponding distribution files still exist.

Set FORCE_RECOMPUTE = True to ignore the cache.

Root tie breaking
=================
The instantaneous root is the genotype with maximum abundance. If several
modal genotypes are tied, the selected root is:

1. the tied genotype closest to the ancestral origin;
2. if still tied, the lexicographically smallest numeric locus tuple.

The number of tied modal genotypes is stored as ``root_tie_count``.
"""


import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# User configuration
# =============================================================================

ABUNDANCE_DIR = Path(ARGS.data) / "Abundances"
MEANS_DIR = Path(ARGS.data) / "Means"
DISTRIBS_DIR = Path(ARGS.data) / "Distribs"
RESULTS_DIR = Path(_RESULTS)

# Optional substring used to restrict the files analyzed.
# Examples:
# FILE_FILTER = "nrw_1000_L_10000_q_0.0"
# FILE_FILTER = "iteration_0"
FILE_FILTER: str | None = ARGS.filter

# Recalculate snapshots even when a valid cached root-log row and both
# distribution files already exist.
FORCE_RECOMPUTE = False

# Print progress every this many newly calculated snapshots.
PROGRESS_EVERY = 100

# Change this only when the cached schema or calculations are modified.
CACHE_VERSION = 1


# =============================================================================
# Schemas and filename parsing
# =============================================================================

SNAPSHOT_PATTERN = re.compile(
    r"^(?P<base>.+)_t_(?P<time>\d+)\.txt$"
)

MEAN_COLUMNS = [
    "time",
    "mean_hamming",
    "variance_hamming",
]

ROOT_LOG_COLUMNS = [
    "cache_version",
    "simulation_base",
    "time",
    "source_filename",
    "source_size_bytes",
    "source_mtime_ns",
    "population_size",
    "number_of_genotypes",
    "root_id",
    "root_genotype",
    "root_distance_origin",
    "root_abundance",
    "root_frequency",
    "root_tie_count",
    "root_is_unique_mode",
    "root_second_abundance",
    "root_abundance_margin",
    "root_changed",
    "previous_root_id",
    "root_distance_previous",
    "root_epoch",
    "root_run_length",
    "mean_hamming_origin",
    "variance_hamming_origin",
    "mean_hamming_root",
    "variance_hamming_root",
    "origin_distribution_filename",
    "root_distribution_filename",
]

ROOT_CATALOG_COLUMNS = [
    "root_id",
    "root_genotype",
    "root_distance_origin",
    "first_time_observed",
    "last_time_observed",
    "snapshot_count",
    "simulation_count",
    "maximum_abundance",
    "maximum_frequency",
]


def parse_snapshot_filename(filepath: Path) -> tuple[str, int]:
    """Return the filename without ``_t_<time>.txt`` and the generation."""
    match = SNAPSHOT_PATTERN.match(filepath.name)

    if match is None:
        raise ValueError(
            "Filename does not end in '_t_<integer>.txt':\n"
            f"{filepath.name}"
        )

    return match.group("base"), int(match.group("time"))


def origin_distribution_path(
    abundance_filepath: Path,
) -> Path:
    """Output path for distances from the ancestral origin."""
    return DISTRIBS_DIR / f"{abundance_filepath.stem}_ref_origin.txt"


def root_distribution_path(
    abundance_filepath: Path,
) -> Path:
    """Output path for distances from the instantaneous root."""
    return DISTRIBS_DIR / f"{abundance_filepath.stem}_ref_root.txt"


# =============================================================================
# Genotype handling
# =============================================================================


def normalize_genotype_text(line: str) -> str:
    """
    Return the canonical genotype text used by the simulator.

    The ancestral all-zero genotype is represented by ``--``. The simulator
    already writes mutated loci in increasing order.
    """
    genotype = line.strip()

    if not genotype or genotype == "--":
        return "--"

    return genotype


def genotype_length(genotype: str) -> int:
    """Hamming distance from a genotype to the all-zero origin."""
    if genotype == "--":
        return 0

    return genotype.count(",") + 1


def genotype_to_tuple(genotype: str) -> tuple[int, ...]:
    """Convert comma-separated mutated loci into an exact numeric tuple."""
    if genotype == "--":
        return ()

    return tuple(int(value) for value in genotype.split(","))


def stable_root_id(genotype: str) -> str:
    """Return a stable content-derived ID for a root genotype."""
    payload = b"model4a30-root-v1\0" + genotype.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def root_tie_key(genotype: str) -> tuple[Any, ...]:
    """Deterministic ordering for equally abundant modal genotypes."""
    loci = genotype_to_tuple(genotype)
    return len(loci), loci


def distance_to_root(
    genotype: str,
    root_loci: frozenset[int],
    root_length: int,
) -> int:
    """
    Binary Hamming distance from genotype A to root R.

        d_H(A,R) = |A| + |R| - 2 |A intersection R|.
    """
    if genotype == "--":
        return root_length

    genotype_values = genotype.split(",")
    common_loci = sum(int(value) in root_loci for value in genotype_values)

    return len(genotype_values) + root_length - 2 * common_loci


def hamming_between_genotypes(
    genotype_a: str,
    genotype_b: str,
) -> int:
    """Exact Hamming distance between two sorted genotype strings."""
    loci_a = genotype_to_tuple(genotype_a)
    loci_b = genotype_to_tuple(genotype_b)

    index_a = 0
    index_b = 0
    distance = 0

    while index_a < len(loci_a) and index_b < len(loci_b):
        locus_a = loci_a[index_a]
        locus_b = loci_b[index_b]

        if locus_a == locus_b:
            index_a += 1
            index_b += 1
        elif locus_a < locus_b:
            distance += 1
            index_a += 1
        else:
            distance += 1
            index_b += 1

    distance += len(loci_a) - index_a
    distance += len(loci_b) - index_b

    return distance


# =============================================================================
# Atomic output helpers
# =============================================================================


def temporary_output_path(filepath: Path) -> Path:
    """Create a process-specific temporary filename beside an output."""
    return filepath.with_name(f".{filepath.name}.tmp.{os.getpid()}")


def safely_remove(filepath: Path) -> None:
    """Remove a temporary file when present."""
    try:
        filepath.unlink()
    except FileNotFoundError:
        pass


def atomic_write_csv(
    filepath: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write a complete CSV before replacing any older output."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_output_path(filepath)

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        temporary_path.replace(filepath)

    except Exception:
        safely_remove(temporary_path)
        raise


# =============================================================================
# Analysis of one abundance snapshot
# =============================================================================


def analyze_snapshot(
    abundance_filepath: Path,
) -> dict[str, Any]:
    """
    Analyze one abundance snapshot and write both distance distributions.

    First pass
    ----------
    * count exact genotypes;
    * compute moments relative to the ancestral origin;
    * write the origin-reference distribution.

    Second pass
    -----------
    * use the modal genotype identified in the first pass;
    * write the instantaneous-root distribution.

    Root distances are calculated once per distinct genotype and then reused
    for every repeated copy in the population.
    """
    simulation_base, time_value = parse_snapshot_filename(abundance_filepath)
    source_stat = abundance_filepath.stat()

    origin_output = origin_distribution_path(abundance_filepath)
    root_output = root_distribution_path(abundance_filepath)

    DISTRIBS_DIR.mkdir(parents=True, exist_ok=True)

    origin_temporary = temporary_output_path(origin_output)
    root_temporary = temporary_output_path(root_output)

    genotype_counts: Counter[str] = Counter()

    population_size = 0
    origin_sum = 0
    origin_squared_sum = 0

    try:
        # First pass: origin distribution and genotype abundances.
        with (
            abundance_filepath.open("r", encoding="utf-8") as input_file,
            origin_temporary.open("w", encoding="utf-8") as output_file,
        ):
            for line in input_file:
                if not line.strip():
                    continue

                genotype = normalize_genotype_text(line)
                distance = genotype_length(genotype)

                genotype_counts[genotype] += 1
                population_size += 1

                origin_sum += distance
                origin_squared_sum += distance**2

                output_file.write(f"{distance}\n")

        if population_size == 0:
            raise ValueError(f"Empty abundance file:\n{abundance_filepath}")

        # Identify the instantaneous root.
        maximum_abundance = max(genotype_counts.values())

        root_candidates = [
            genotype
            for genotype, abundance in genotype_counts.items()
            if abundance == maximum_abundance
        ]

        root_genotype = min(root_candidates, key=root_tie_key)
        root_tuple = genotype_to_tuple(root_genotype)
        root_loci = frozenset(root_tuple)
        root_length = len(root_tuple)

        root_abundance = genotype_counts[root_genotype]
        root_frequency = root_abundance / population_size

        abundance_ranking = sorted(genotype_counts.values(), reverse=True)
        root_second_abundance = (
            abundance_ranking[1] if len(abundance_ranking) > 1 else 0
        )

        # Calculate the distance once per distinct genotype.
        root_distance_by_genotype = {
            genotype: distance_to_root(
                genotype=genotype,
                root_loci=root_loci,
                root_length=root_length,
            )
            for genotype in genotype_counts
        }

        root_sum = sum(
            genotype_counts[genotype] * root_distance_by_genotype[genotype]
            for genotype in genotype_counts
        )

        root_squared_sum = sum(
            genotype_counts[genotype]
            * root_distance_by_genotype[genotype] ** 2
            for genotype in genotype_counts
        )

        # Second pass: root-reference distribution in population order.
        with (
            abundance_filepath.open("r", encoding="utf-8") as input_file,
            root_temporary.open("w", encoding="utf-8") as output_file,
        ):
            for line in input_file:
                if not line.strip():
                    continue

                genotype = normalize_genotype_text(line)
                output_file.write(f"{root_distance_by_genotype[genotype]}\n")

        # Publish only complete files.
        origin_temporary.replace(origin_output)
        root_temporary.replace(root_output)

    except Exception:
        safely_remove(origin_temporary)
        safely_remove(root_temporary)
        raise

    mean_origin = origin_sum / population_size
    variance_origin = max(
        origin_squared_sum / population_size - mean_origin**2,
        0.0,
    )

    mean_root = root_sum / population_size
    variance_root = max(
        root_squared_sum / population_size - mean_root**2,
        0.0,
    )

    return {
        "cache_version": CACHE_VERSION,
        "simulation_base": simulation_base,
        "time": time_value,
        "source_filename": abundance_filepath.name,
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "population_size": population_size,
        "number_of_genotypes": len(genotype_counts),
        "root_id": stable_root_id(root_genotype),
        "root_genotype": root_genotype,
        "root_distance_origin": root_length,
        "root_abundance": root_abundance,
        "root_frequency": root_frequency,
        "root_tie_count": len(root_candidates),
        "root_is_unique_mode": int(len(root_candidates) == 1),
        "root_second_abundance": root_second_abundance,
        "root_abundance_margin": root_abundance - root_second_abundance,
        "mean_hamming_origin": mean_origin,
        "variance_hamming_origin": variance_origin,
        "mean_hamming_root": mean_root,
        "variance_hamming_root": variance_root,
        "origin_distribution_filename": origin_output.name,
        "root_distribution_filename": root_output.name,
    }


# =============================================================================
# Cache loading and validation
# =============================================================================

INTEGER_FIELDS = {
    "cache_version",
    "time",
    "source_size_bytes",
    "source_mtime_ns",
    "population_size",
    "number_of_genotypes",
    "root_distance_origin",
    "root_abundance",
    "root_tie_count",
    "root_is_unique_mode",
    "root_second_abundance",
    "root_abundance_margin",
    "root_changed",
    "root_distance_previous",
    "root_epoch",
    "root_run_length",
}

FLOAT_FIELDS = {
    "root_frequency",
    "mean_hamming_origin",
    "variance_hamming_origin",
    "mean_hamming_root",
    "variance_hamming_root",
}


def convert_cached_row(row: dict[str, str]) -> dict[str, Any]:
    """Restore numeric values from one root-log CSV row."""
    converted: dict[str, Any] = dict(row)

    for field in INTEGER_FIELDS:
        if field in converted and converted[field] != "":
            converted[field] = int(converted[field])

    for field in FLOAT_FIELDS:
        if field in converted and converted[field] != "":
            converted[field] = float(converted[field])

    return converted


def load_root_log(
    root_log_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load reusable rows indexed by source abundance filename."""
    if not root_log_path.exists():
        return {}

    try:
        with root_log_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)

            if reader.fieldnames is None:
                return {}

            required_fields = {
                "cache_version",
                "simulation_base",
                "source_filename",
                "source_size_bytes",
                "source_mtime_ns",
                "root_id",
                "root_genotype",
                "mean_hamming_origin",
                "variance_hamming_origin",
                "mean_hamming_root",
                "variance_hamming_root",
            }

            if not required_fields.issubset(set(reader.fieldnames)):
                return {}

            return {
                row["source_filename"]: convert_cached_row(row)
                for row in reader
            }

    except (OSError, ValueError, csv.Error):
        return {}


def cached_snapshot_is_valid(
    cached_row: dict[str, Any] | None,
    abundance_filepath: Path,
) -> bool:
    """Validate one cached row and its two distribution files."""
    if cached_row is None:
        return False

    try:
        if int(cached_row["cache_version"]) != CACHE_VERSION:
            return False

        source_stat = abundance_filepath.stat()

        if int(cached_row["source_size_bytes"]) != int(source_stat.st_size):
            return False

        if int(cached_row["source_mtime_ns"]) != int(source_stat.st_mtime_ns):
            return False

        if not origin_distribution_path(abundance_filepath).exists():
            return False

        if not root_distribution_path(abundance_filepath).exists():
            return False

        return True

    except (KeyError, TypeError, ValueError, OSError):
        return False


# =============================================================================
# Root trajectories and output tables
# =============================================================================


def add_root_trajectory_information(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add changes of instantaneous root through time."""
    records = sorted(records, key=lambda record: int(record["time"]))

    previous_root_id = ""
    previous_root_genotype = ""
    root_epoch = 0
    root_run_length = 0

    for index, record in enumerate(records):
        current_root_id = str(record["root_id"])
        current_root_genotype = str(record["root_genotype"])

        if index == 0:
            root_changed = 0
            root_distance_previous = 0
            root_run_length = 1

        elif current_root_id == previous_root_id:
            root_changed = 0
            root_distance_previous = 0
            root_run_length += 1

        else:
            root_changed = 1
            root_epoch += 1
            root_run_length = 1
            root_distance_previous = hamming_between_genotypes(
                current_root_genotype,
                previous_root_genotype,
            )

        record["root_changed"] = root_changed
        record["previous_root_id"] = previous_root_id
        record["root_distance_previous"] = root_distance_previous
        record["root_epoch"] = root_epoch
        record["root_run_length"] = root_run_length

        previous_root_id = current_root_id
        previous_root_genotype = current_root_genotype

    return records


def write_simulation_outputs(
    simulation_base: str,
    records: list[dict[str, Any]],
) -> None:
    """Write the two moment files and one reusable root log."""
    records = add_root_trajectory_information(records)

    origin_rows = [
        {
            "time": record["time"],
            "mean_hamming": record["mean_hamming_origin"],
            "variance_hamming": record["variance_hamming_origin"],
        }
        for record in records
    ]

    root_rows = [
        {
            "time": record["time"],
            "mean_hamming": record["mean_hamming_root"],
            "variance_hamming": record["variance_hamming_root"],
        }
        for record in records
    ]

    atomic_write_csv(
        MEANS_DIR / f"{simulation_base}_ref_origin.csv",
        MEAN_COLUMNS,
        origin_rows,
    )

    atomic_write_csv(
        MEANS_DIR / f"{simulation_base}_ref_root.csv",
        MEAN_COLUMNS,
        root_rows,
    )

    atomic_write_csv(
        RESULTS_DIR / f"{simulation_base}_root_log.csv",
        ROOT_LOG_COLUMNS,
        records,
    )


def rebuild_global_root_catalog() -> None:
    """
    Rebuild a global root catalog from every per-simulation root log.

    This keeps the catalog complete even when the current run uses FILE_FILTER.
    """
    catalog: dict[str, dict[str, Any]] = {}
    simulations_by_root: dict[str, set[str]] = defaultdict(set)

    for root_log_path in sorted(RESULTS_DIR.glob("dataOut_*_root_log.csv")):
        with root_log_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)

            if reader.fieldnames is None:
                continue

            required = {
                "root_id",
                "root_genotype",
                "root_distance_origin",
                "time",
                "root_abundance",
                "root_frequency",
                "simulation_base",
            }

            if not required.issubset(set(reader.fieldnames)):
                continue

            for row in reader:
                root_id = row["root_id"]
                simulation_base = row["simulation_base"]
                time_value = int(row["time"])
                abundance = int(row["root_abundance"])
                frequency = float(row["root_frequency"])

                simulations_by_root[root_id].add(simulation_base)

                if root_id not in catalog:
                    catalog[root_id] = {
                        "root_id": root_id,
                        "root_genotype": row["root_genotype"],
                        "root_distance_origin": int(row["root_distance_origin"]),
                        "first_time_observed": time_value,
                        "last_time_observed": time_value,
                        "snapshot_count": 1,
                        "simulation_count": 0,
                        "maximum_abundance": abundance,
                        "maximum_frequency": frequency,
                    }
                else:
                    entry = catalog[root_id]
                    entry["first_time_observed"] = min(
                        int(entry["first_time_observed"]),
                        time_value,
                    )
                    entry["last_time_observed"] = max(
                        int(entry["last_time_observed"]),
                        time_value,
                    )
                    entry["snapshot_count"] = int(entry["snapshot_count"]) + 1
                    entry["maximum_abundance"] = max(
                        int(entry["maximum_abundance"]),
                        abundance,
                    )
                    entry["maximum_frequency"] = max(
                        float(entry["maximum_frequency"]),
                        frequency,
                    )

    rows = []

    for root_id, entry in catalog.items():
        entry["simulation_count"] = len(simulations_by_root[root_id])
        rows.append(entry)

    rows.sort(
        key=lambda row: (
            int(row["root_distance_origin"]),
            str(row["root_genotype"]),
        )
    )

    atomic_write_csv(
        RESULTS_DIR / "root_catalog.csv",
        ROOT_CATALOG_COLUMNS,
        rows,
    )


# =============================================================================
# Main notebook-friendly workflow
# =============================================================================


def collect_abundance_files() -> list[Path]:
    """Locate all selected abundance snapshots."""
    if not ABUNDANCE_DIR.exists():
        raise FileNotFoundError(
            f"Abundance directory not found:\n{ABUNDANCE_DIR.resolve()}"
        )

    abundance_files = sorted(ABUNDANCE_DIR.glob("dataOut_*_t_*.txt"))

    if FILE_FILTER is not None:
        abundance_files = [
            filepath
            for filepath in abundance_files
            if FILE_FILTER in filepath.name
        ]

    if not abundance_files:
        raise FileNotFoundError(
            "No abundance files matched ABUNDANCE_DIR and FILE_FILTER."
        )

    return abundance_files


def group_files_by_simulation(
    abundance_files: list[Path],
) -> dict[str, list[Path]]:
    """Group snapshots after removing their ``_t_<time>.txt`` suffix."""
    grouped: dict[str, list[Path]] = defaultdict(list)

    for filepath in abundance_files:
        simulation_base, _ = parse_snapshot_filename(filepath)
        grouped[simulation_base].append(filepath)

    for simulation_base in grouped:
        grouped[simulation_base].sort(
            key=lambda filepath: parse_snapshot_filename(filepath)[1]
        )

    return dict(grouped)


def run_analysis() -> None:
    """Run the complete cached analysis without making plots."""
    MEANS_DIR.mkdir(parents=True, exist_ok=True)
    DISTRIBS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    abundance_files = collect_abundance_files()
    grouped_files = group_files_by_simulation(abundance_files)

    records_by_simulation: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_cached = 0
    total_computed = 0
    selected_snapshot_count = len(abundance_files)

    print(f"Abundance snapshots found: {selected_snapshot_count}")
    print(f"Simulation/iteration groups: {len(grouped_files)}")

    for simulation_index, simulation_base in enumerate(
        sorted(grouped_files),
        start=1,
    ):
        simulation_files = grouped_files[simulation_base]
        root_log_path = RESULTS_DIR / f"{simulation_base}_root_log.csv"

        cached_rows = (
            {}
            if FORCE_RECOMPUTE
            else load_root_log(root_log_path)
        )

        seen_times: set[int] = set()

        for filepath in simulation_files:
            _, time_value = parse_snapshot_filename(filepath)

            if time_value in seen_times:
                raise ValueError(
                    "Duplicate generation for the same simulation base:\n"
                    f"{simulation_base}, t={time_value}"
                )

            seen_times.add(time_value)
            cached_row = cached_rows.get(filepath.name)

            if cached_snapshot_is_valid(cached_row, filepath):
                record = cached_row
                total_cached += 1
            else:
                record = analyze_snapshot(filepath)
                total_computed += 1

                if (
                    total_computed == 1
                    or total_computed % PROGRESS_EVERY == 0
                ):
                    print(
                        f"Computed {total_computed} uncached snapshots "
                        f"({total_cached} reused)"
                    )

            records_by_simulation[simulation_base].append(record)

        write_simulation_outputs(
            simulation_base,
            records_by_simulation[simulation_base],
        )

        if (
            simulation_index == 1
            or simulation_index % 25 == 0
            or simulation_index == len(grouped_files)
        ):
            print(
                f"Wrote outputs for {simulation_index}/"
                f"{len(grouped_files)} simulation groups"
            )

    rebuild_global_root_catalog()
    if total_cached + total_computed != selected_snapshot_count:
        raise RuntimeError(
            "Internal accounting error: not every snapshot was processed."
        )

    print()
    print("Analysis complete")
    print("-----------------")
    print(f"Snapshots recomputed: {total_computed}")
    print(f"Snapshots reused from cache: {total_cached}")
    print(f"Mean files:\n  {MEANS_DIR.resolve()}")
    print(f"Distribution files:\n  {DISTRIBS_DIR.resolve()}")
    print(f"Root logs and catalog:\n  {RESULTS_DIR.resolve()}")


# Run immediately when executed with %run or as a script.
run_analysis()