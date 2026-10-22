#!/usr/bin/env python3
"""
fig_topology_knn.py -- Fig. "model-fq_LCC_knn_panel_conditioned_shell_theory_..." (Chapter 4, Sec. 4.2.3)

Average nearest-neighbour degree k_nn(k) of the LCC of the single-peak genotype networks
(N=1e5, L=250, F=3, q=0, lambda=1, t=100; 10 networks pooled), (a) complete, with the uncorrelated
reference <k^2>/<k>, and (b) split by Hamming distance from the selected genotype (m=0, m=1, m>=2),
with the shell-conditioned prediction (same construction as fig_topology_degree.py; the m=1 branch
uses the conditioned shell-2 incidence probabilities, k_nn(k|m=1) ~ kappa_{2|1} + (K_0 - kappa_{2|1})/k,
and the root has k_nn = 1 + 2 S_2 / S_1).

Inputs as for fig_topology_degree.py.
Port of the thesis notebook 16-Simulations/old/plots-model-fq-topology.ipynb (cell 5).

    python fig_topology_knn.py --data Data/single_peak_topology --out Figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import argparse
_ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
_ap.add_argument("--data", default="Data/single_peak",
                 help="searched recursively for */Networks/*.graphml next to */Abundances/*.txt")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--N", type=int, default=100_000)
_ap.add_argument("--L", type=int, default=250)
_ap.add_argument("--F", type=float, default=3.0)
_ap.add_argument("--t", type=int, default=100, help="snapshot time (the t-1 snapshot is also needed)")
ARGS = _ap.parse_args()


from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binom, poisson

try:
    from IPython.display import Markdown, display
except ImportError:  # The analysis still works outside Jupyter.
    Markdown = None
    display = print


# ============================================================
# Configuration
# ============================================================

MODEL = "model-fq"

# Expected hierarchy:
#   Data/model-fq/runs/<run_id>/Networks/<snapshot>.graphml
#   Data/model-fq/runs/<run_id>/Abundances/<snapshot>.txt
NETWORK_SEARCH_ROOT = Path(ARGS.data)

# Used only when SAVE_FIGS = True.
OUTPUT_DIR = Path(ARGS.out)

# ------------------------------------------------------------
# Experiment selection
# ------------------------------------------------------------
# None accepts every available value. A scalar or iterable selects one or
# several values.

TARGET_SEQUENCE_LENGTHS = [ARGS.L]
TARGET_POPULATION_SIZES = [ARGS.N]
TARGET_FPEAK_VALUES = [ARGS.F]

TARGET_Q_VALUES = None              # None = all q values
TARGET_TIMES = [ARGS.t]                # e.g. [99, 100] or None
TARGET_ITERATIONS = None            # None = all independent simulations
TARGET_MU_VALUES = None             # e.g. [1.0]
TARGET_FBG_VALUES = None            # e.g. [1.0]

# If several times are selected:
#   True  -> one panel per q and time
#   False -> pool the selected times within each q
SPLIT_BY_TIME = True

# ------------------------------------------------------------
# Shell-resolved normalization used in panel (b)
# ------------------------------------------------------------
# "joint":
#     empirical:  P(k,c) = sum_s n_s(k,c) / sum_s S_s
#     theory:     P_th(k,c) = sum_s sum_{m in c} S_{s,m} P_s(k|m) / sum_s S_s
#
# "conditional":
#     empirical:  P(k|c) = sum_s n_s(k,c) / sum_s S_{s,c}
#     theory:     P_th(k|c) = sum_s sum_{m in c} S_{s,m} P_s(k|m) / sum_s S_{s,c}
#
# Here c is one of m=0, m=1, or m>=2, S_s is the number of LCC nodes,
# and S_{s,m} is the number of LCC nodes in exact shell m.
SHELL_PROBABILITY_MODE = "joint"

# ------------------------------------------------------------
# Analytical approximation
# ------------------------------------------------------------

ADD_ANALYTICAL_THEORY = True
DISPLAY_ANALYTICAL_DESCRIPTION = True
VALIDATE_POPULATION_SIZE = True

# The homogeneous shell approximation is retained for m=0 and m>=2. For
# m=1, this option uses a genotype-resolved t-1 -> t Poisson-binomial
# approximation based on the finite offspring pool.
USE_DYNAMIC_M1_APPROXIMATION = True

# The network at time t is predicted from the abundance snapshot at t-1.
DYNAMIC_M1_PREVIOUS_TIME_OFFSET = 1

# The default generator saves t=99 and t=100. When True, stop with a clear
# error if the required previous snapshot is missing. Set False to fall back
# to the homogeneous shell approximation for m=1.
REQUIRE_DYNAMIC_M1_INPUT = True

# Mutation transitions above this Hamming distance are neglected only in the
# genotype-resolved m=1 correction. For lambda=1, distance > 6 contributes
# negligibly to one specified target genotype. It must be at least 2 because
# the root can create shell-2 targets by two mutations. Use None for all d.
DYNAMIC_M1_MAX_DISTANCE = 6

# Condition the shell-2 incidence probabilities on the observed number S_2
# of LCC nodes in shell 2. This fixes the exact first-shell mean degree when
# the root and all L first-shell genotypes are present:
#
#     <K>_{m=1} = 1 + 2 S_2 / S_1.
#
# The fitted scalar changes only the global hazard scale; the relative
# heterogeneity among targets {i,j} still comes from the t-1 dynamics.
DYNAMIC_M1_CONDITION_ON_OBSERVED_S2 = True
DYNAMIC_M1_CALIBRATION_RTOL = 1e-11
DYNAMIC_M1_CALIBRATION_MAX_ITERATIONS = 120

# For q>0, recover the deterministic viability of every possible shell-2
# target from run_metadata.json and the same Blake2b hash used by the data
# generator. If metadata is unavailable, the expected landscape factor 1-q
# is used. For q=0 this distinction is irrelevant.
DYNAMIC_M1_USE_EXACT_TARGET_VIABILITY = True

# ------------------------------------------------------------
# Plot and output options
# ------------------------------------------------------------

SAVE_FIGS = True
SHOW_FIGS = False
FIGURE_FORMAT = "pdf"               # "pdf", "png", or "svg"
FIGURE_DPI = 600

FIGSIZE = (12.2, 4.9)
FLOAT_TOLERANCE = 1e-12

# ------------------------------------------------------------
# K_nn analytical and plotting options
# ------------------------------------------------------------
# The degree PMFs retain the same population-derived shell occupancies used
# in the successful P(k) approximation. Expected degrees of neighboring
# shells can instead be conditioned on the observed LCC shell occupancy.
#
#   "lcc_observed"      -> S_m / C(L,m), recommended for K_nn inside the LCC
#   "population_poisson" -> 1-exp[-n_m/C(L,m)] from the abundance snapshot
KNN_NEIGHBOR_OCCUPANCY_SOURCE = "lcc_observed"

# LCC nodes cannot have degree zero unless the LCC itself is a singleton.
# Renormalizing the shell PMFs over k>=1 makes the approximation explicitly
# conditional on membership in a non-trivial connected component.
CONDITION_THEORY_ON_POSITIVE_DEGREE = True

# Horizontal reference in panel (a): <k^2>/<k> for an uncorrelated network
# with the same pooled degree sequence.
SHOW_UNCORRELATED_BASELINE = True

# Plot one panel per q (and per time when SPLIT_BY_TIME=True).
# Both logarithmic options are editable independently.
LOG_X = True
LOG_Y = True

MUTATION_CLASS_ORDER = ["m=0", "m=1", "m>=2"]
MUTATION_CLASS_LABELS = {
    "m=0": r"$m=0$",
    "m=1": r"$m=1$",
    "m>=2": r"$m\geq 2$",
}
MUTATION_CLASS_MARKERS = {
    "m=0": "o",
    "m=1": "s",
    "m>=2": "^",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 9.3,
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
# Filename metadata and selectors
# ============================================================

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

NAME_RE = re.compile(
    rf"^dataOut_mu_(?P<mu>{NUMBER})"
    rf"_nrw_(?P<N>\d+)"
    rf"_L_(?P<L>\d+)"
    rf"_q_(?P<q>{NUMBER})"
    rf"_iteration_(?P<iteration>\d+)"
    rf"_fpeak_(?P<fpeak>{NUMBER})"
    rf"_fbg_(?P<fbg>{NUMBER})"
    rf"_t_(?P<t>\d+)$"
)


def as_list(values):
    """Normalize None, a scalar, or an iterable into None or a list."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return [values]
    if np.isscalar(values):
        return [values]
    return list(values)


def matches_integer(value: int, allowed_values) -> bool:
    allowed = as_list(allowed_values)
    if allowed is None:
        return True
    return int(value) in {int(item) for item in allowed}


def matches_float(
    value: float,
    allowed_values,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    allowed = as_list(allowed_values)
    if allowed is None:
        return True
    return any(
        np.isclose(float(value), float(item), atol=tolerance, rtol=0.0)
        for item in allowed
    )


def parse_network_metadata(network_file: Path) -> dict | None:
    """Parse model-fq parameters from a GraphML filename."""
    match = NAME_RE.fullmatch(network_file.stem)
    if match is None:
        return None

    values = match.groupdict()
    run_directory = (
        network_file.parent.parent
        if network_file.parent.name == "Networks"
        else network_file.parent
    )

    try:
        relative_path = str(
            network_file.resolve().relative_to(NETWORK_SEARCH_ROOT.resolve())
        )
    except ValueError:
        relative_path = str(network_file.resolve())

    return {
        "network": relative_path,
        "network_file": str(network_file.resolve()),
        "network_name": network_file.stem,
        "run_id": run_directory.name,
        "run_directory": str(run_directory.resolve()),
        "mu": float(values["mu"]),
        "N": int(values["N"]),
        "L": int(values["L"]),
        "q": float(values["q"]),
        "iteration": int(values["iteration"]),
        "fpeak": float(values["fpeak"]),
        "fbg": float(values["fbg"]),
        "t": int(values["t"]),
    }


def metadata_is_selected(metadata: dict) -> bool:
    return (
        matches_integer(metadata["N"], TARGET_POPULATION_SIZES)
        and matches_integer(metadata["L"], TARGET_SEQUENCE_LENGTHS)
        and matches_float(metadata["fpeak"], TARGET_FPEAK_VALUES)
        and matches_float(metadata["q"], TARGET_Q_VALUES)
        and matches_integer(metadata["t"], TARGET_TIMES)
        and matches_integer(metadata["iteration"], TARGET_ITERATIONS)
        and matches_float(metadata["mu"], TARGET_MU_VALUES)
        and matches_float(metadata["fbg"], TARGET_FBG_VALUES)
    )


def selector_description() -> str:
    def show(values):
        normalized = as_list(values)
        return "all" if normalized is None else str(normalized)

    return (
        f"N={show(TARGET_POPULATION_SIZES)}, "
        f"L={show(TARGET_SEQUENCE_LENGTHS)}, "
        f"fpeak={show(TARGET_FPEAK_VALUES)}, "
        f"q={show(TARGET_Q_VALUES)}, "
        f"t={show(TARGET_TIMES)}, "
        f"iterations={show(TARGET_ITERATIONS)}, "
        f"mu={show(TARGET_MU_VALUES)}, "
        f"fbg={show(TARGET_FBG_VALUES)}"
    )


def format_number(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "p")


# ============================================================
# Genotype names and shell classes
# ============================================================


def genotype_name(node, attributes: dict) -> str:
    """Return the comma-separated genotype representation of one node."""
    for key in ("genotype", "label"):
        value = attributes.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(node).strip()


def genotype_tuple_from_name(
    name: str,
    *,
    L: int | None = None,
    validate: bool = True,
) -> tuple[int, ...]:
    """Convert ``--`` or a comma-separated node name to a sorted tuple."""
    text = str(name).strip()
    if text in {"", "--"}:
        return ()

    tokens = [token.strip() for token in text.split(",")]
    if any(token == "" for token in tokens):
        raise ValueError(f"Malformed genotype name: {name!r}")

    try:
        loci = tuple(sorted(int(token) for token in tokens))
    except ValueError as error:
        raise ValueError(
            f"Genotype name does not contain integer loci: {name!r}"
        ) from error

    if validate:
        if any(locus < 0 for locus in loci):
            raise ValueError(f"Negative locus in genotype name: {name!r}")
        if len(set(loci)) != len(loci):
            raise ValueError(f"Repeated locus in genotype name: {name!r}")
        if L is not None and loci and loci[-1] >= L:
            raise ValueError(
                f"Locus {loci[-1]} lies outside [0,{L - 1}] in {name!r}."
            )

    return loci


def mutation_class(mutations: int) -> str:
    if mutations == 0:
        return "m=0"
    if mutations == 1:
        return "m=1"
    return "m>=2"


# ============================================================
# Graph and abundance-file reading
# ============================================================


def to_simple_undirected_graph(graph: nx.Graph) -> nx.Graph:
    if graph.is_directed():
        graph = graph.to_undirected()
    if graph.is_multigraph():
        graph = nx.Graph(graph)
    return graph


def largest_component_nodes(graph: nx.Graph) -> tuple[set, int]:
    largest: set | None = None
    n_components = 0

    for component in nx.connected_components(graph):
        n_components += 1
        if largest is None or len(component) > len(largest):
            largest = set(component)

    if largest is None:
        raise ValueError("The graph has no nodes.")

    return largest, n_components


def abundance_stem_at_time(network_file: Path, target_time: int) -> str:
    """Replace the final ``_t_<time>`` field in a snapshot stem."""
    replacement = re.sub(
        r"_t_\d+$",
        f"_t_{int(target_time)}",
        network_file.stem,
    )
    if replacement == network_file.stem and not network_file.stem.endswith(
        f"_t_{int(target_time)}"
    ):
        raise ValueError(
            f"Cannot replace the time field in snapshot name {network_file.stem!r}."
        )
    return replacement


def resolve_abundance_file_at_time(
    network_file: Path,
    metadata: dict,
    *,
    target_time: int,
) -> Path:
    """Locate the abundance snapshot for the same simulation at a chosen time."""
    run_directory = Path(metadata["run_directory"])
    stem = abundance_stem_at_time(network_file, target_time)
    filename = f"{stem}.txt"

    candidates = [
        run_directory / "Abundances" / filename,
        network_file.parent.parent / "Abundances" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(run_directory.rglob(filename))
    matches = [path for path in matches if path.parent.name == "Abundances"]

    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"Several abundance files match {filename}:\n"
            + "\n".join(f"  {path}" for path in matches)
        )

    raise FileNotFoundError(
        "Original abundance snapshot not found for:\n"
        f"  {network_file}\n"
        f"Requested time: t={target_time}\n"
        "Expected:\n"
        f"  {run_directory / 'Abundances' / filename}"
    )


def resolve_abundance_file(network_file: Path, metadata: dict) -> Path:
    """Locate the abundance snapshot corresponding to the GraphML time."""
    return resolve_abundance_file_at_time(
        network_file,
        metadata,
        target_time=int(metadata["t"]),
    )


def read_population_genotype_counts(
    abundance_file: Path,
    *,
    L: int,
    expected_population_size: int | None,
) -> tuple[Counter[tuple[int, ...]], np.ndarray, int]:
    """Read genotype abundances and their population Hamming-shell counts."""
    genotype_counts: Counter[tuple[int, ...]] = Counter()
    shell_counts = np.zeros(L + 1, dtype=np.int64)
    observed_population = 0

    with abundance_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue

            try:
                genotype = genotype_tuple_from_name(text, L=L, validate=True)
            except ValueError as error:
                raise ValueError(
                    f"{error} File {abundance_file}, line {line_number}."
                ) from error

            mutations = len(genotype)
            genotype_counts[genotype] += 1
            shell_counts[mutations] += 1
            observed_population += 1

    if observed_population == 0:
        raise ValueError(f"No individuals found in {abundance_file}.")

    if (
        VALIDATE_POPULATION_SIZE
        and expected_population_size is not None
        and observed_population != expected_population_size
    ):
        raise ValueError(
            f"Population-size mismatch in {abundance_file}: "
            f"expected N={expected_population_size}, read {observed_population}."
        )

    return genotype_counts, shell_counts, observed_population


def load_run_metadata(metadata: dict) -> dict:
    """Read run_metadata.json when it is present."""
    path = Path(metadata["run_directory"]) / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ============================================================
# Shell-dependent analytical approximation
# ============================================================


def shell_log_sizes(L: int) -> np.ndarray:
    """Return log binomial shell sizes log[C(L,m)] for m=0,...,L."""
    m = np.arange(L + 1, dtype=float)
    return gammaln(L + 1.0) - gammaln(m + 1.0) - gammaln(L - m + 1.0)


def shell_representation_probabilities(
    population_shell_counts: np.ndarray,
    *,
    L: int,
) -> np.ndarray:
    r"""
    Poissonized probability that a specific genotype in shell m is observed:

        p_m = 1 - exp[-N x_m / C(L,m)]
            = 1 - exp[-n_m / C(L,m)].
    """
    counts = np.asarray(population_shell_counts, dtype=float)
    if counts.shape != (L + 1,):
        raise ValueError(f"Expected {L + 1} shell counts, received {counts.shape}.")

    log_sizes = shell_log_sizes(L)
    occupancy_means = np.zeros(L + 1, dtype=float)

    positive = counts > 0
    occupancy_means[positive] = np.exp(
        np.log(counts[positive]) - log_sizes[positive]
    )

    # -expm1(-z) is accurate both for very small and very large z.
    probabilities = -np.expm1(-occupancy_means)
    return np.clip(probabilities, 0.0, 1.0)


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """Stable Binomial(n,p) PMF, including deterministic edge cases."""
    if n < 0:
        raise ValueError("Binomial trial count cannot be negative.")
    if n == 0:
        return np.array([1.0], dtype=float)
    if p <= 0.0:
        result = np.zeros(n + 1, dtype=float)
        result[0] = 1.0
        return result
    if p >= 1.0:
        result = np.zeros(n + 1, dtype=float)
        result[n] = 1.0
        return result

    support = np.arange(n + 1)
    result = binom.pmf(support, n, p).astype(float)
    total = result.sum()
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError(f"Invalid Binomial PMF for n={n}, p={p}.")
    return result / total


def mutation_transition_probabilities(
    *,
    L: int,
    mean_mutations: float,
    max_distance: int | None,
) -> np.ndarray:
    r"""Probability of reaching one specified genotype at distance d.

    The generator samples M~Poisson(lambda), applies min(M,L), and chooses the
    mutated loci uniformly without replacement. Thus, for d<L,

        Q_d = Pr(M=d) / C(L,d).

    For d=L, the Poisson tail Pr(M>=L) is used because of the clipping at L.
    """
    lam = float(mean_mutations)
    if lam < 0:
        raise ValueError("Mean mutation number cannot be negative.")

    probabilities = np.zeros(L + 1, dtype=float)
    allowed_max = L if max_distance is None else min(L, int(max_distance))
    if allowed_max < 0:
        raise ValueError("DYNAMIC_M1_MAX_DISTANCE must be non-negative or None.")

    log_sizes = shell_log_sizes(L)
    for distance in range(min(allowed_max, L - 1) + 1):
        log_value = poisson.logpmf(distance, lam) - log_sizes[distance]
        probabilities[distance] = float(np.exp(log_value))

    if allowed_max >= L:
        probabilities[L] = float(poisson.sf(L - 1, lam))

    probabilities[~np.isfinite(probabilities)] = 0.0
    return np.clip(probabilities, 0.0, 1.0)


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """PMF of a sum of independent Bernoulli variables with unequal p_i."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("Poisson-binomial probabilities must be one-dimensional.")
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("Poisson-binomial probabilities contain non-finite values.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    pmf = np.array([1.0], dtype=float)

    for probability in probabilities:
        pmf = np.concatenate((pmf * (1.0 - probability), [0.0])) + np.concatenate(
            ([0.0], pmf * probability)
        )

    pmf = np.clip(pmf, 0.0, None)
    total = pmf.sum()
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("Invalid Poisson-binomial PMF.")
    return pmf / total


def packed_genotype_bytes(genotype: tuple[int, ...], *, L: int) -> bytes:
    """Reproduce the packed uint64 representation used by the generator."""
    n_words = (L + 63) // 64
    genome = np.zeros(n_words, dtype=np.uint64)
    for locus in genotype:
        word_index = int(locus) // 64
        bit_index = int(locus) % 64
        genome[word_index] |= np.uint64(1) << np.uint64(bit_index)
    return genome.view(np.uint8).tobytes()


def deterministic_target_viability(
    genotype: tuple[int, ...],
    *,
    metadata: dict,
    run_metadata: dict,
) -> bool | None:
    """Recover the generator's deterministic viability, when possible."""
    if not genotype:
        return True

    q = float(metadata["q"])
    if q <= 0.0:
        return True
    if q >= 1.0:
        return False
    if not DYNAMIC_M1_USE_EXACT_TARGET_VIABILITY:
        return None

    base_seed = run_metadata.get("base_seed")
    if base_seed is None:
        return None

    landscape_seed = int(base_seed) + int(metadata["iteration"])
    digest = hashlib.blake2b(digest_size=8, person=b"fitness")
    digest.update(int(landscape_seed).to_bytes(16, "little", signed=False))
    digest.update(packed_genotype_bytes(genotype, L=int(metadata["L"])))
    value = int.from_bytes(digest.digest(), "little")
    return (value / 2**64) < (1.0 - q)


def build_parent_reproductive_aggregates(
    genotype_counts: Counter[tuple[int, ...]],
    *,
    L: int,
    f_peak: float,
    f_background: float,
    max_distance: int | None,
) -> dict:
    """Aggregate t-1 reproductive mass by shell, locus, and locus pair.

    For a target shell-2 genotype {i,j}, a parent h contributes according to
    whether it contains both, one, or neither of i and j. These aggregates
    evaluate the sum over all observed parents without an all-pairs loop.
    """
    shell_totals = np.zeros(L + 1, dtype=float)
    detailed_max_shell = L if max_distance is None else min(L, max_distance + 2)
    locus_totals = np.zeros((detailed_max_shell + 1, L), dtype=float)
    pair_totals: list[dict[tuple[int, int], float]] = [
        {} for _ in range(detailed_max_shell + 1)
    ]

    for genotype, abundance in genotype_counts.items():
        shell = len(genotype)
        fitness = float(f_peak if shell == 0 else f_background)
        reproductive_mass = float(abundance) * fitness
        shell_totals[shell] += reproductive_mass

        if reproductive_mass == 0.0 or shell > detailed_max_shell:
            continue

        for locus in genotype:
            locus_totals[shell, locus] += reproductive_mass

        if shell >= 2:
            shell_pairs = pair_totals[shell]
            for first, second in combinations(genotype, 2):
                key = (first, second)
                shell_pairs[key] = shell_pairs.get(key, 0.0) + reproductive_mass

    return {
        "shell_totals": shell_totals,
        "locus_totals": locus_totals,
        "pair_totals": pair_totals,
        "detailed_max_shell": detailed_max_shell,
    }


def expected_viable_offspring_pool_size(
    *,
    shell_reproductive_totals: np.ndarray,
    mutation_probabilities: np.ndarray,
    q: float,
) -> float:
    r"""Mean size of the accepted offspring pool.

    For q=0 this is exact. For q>0, non-root targets are averaged over the
    deterministic landscape, while the always-viable root is retained exactly:

        v_r approximately equals (1-q) + q Q_r.
    """
    q = float(q)
    shell_reproductive_totals = np.asarray(shell_reproductive_totals, dtype=float)
    root_transition = np.asarray(mutation_probabilities, dtype=float)
    mean_viability = (1.0 - q) + q * root_transition
    expected_pool = float(np.dot(shell_reproductive_totals, mean_viability))
    if expected_pool <= 0.0 or not np.isfinite(expected_pool):
        raise FloatingPointError("The predicted viable offspring pool is empty.")
    return expected_pool


def finite_pool_slot_selection_probability(
    *,
    expected_pool_size: float,
    sample_size: int,
) -> float:
    r"""Probability that one pool position is sampled at least once.

    For a pool of size M followed by N draws with replacement,

        alpha_M = 1 - (1 - 1/M)^N
                approximately equals 1 - exp(-N/M).

    The calculation uses the predicted mean viable-pool size as M.
    """
    M = float(expected_pool_size)
    N = int(sample_size)
    if M <= 0.0 or not np.isfinite(M):
        raise ValueError("Expected offspring-pool size must be finite and positive.")
    if N < 0:
        raise ValueError("Sample size cannot be negative.")
    if N == 0:
        return 0.0
    if M <= 1.0:
        return 1.0

    alpha = -np.expm1(float(N) * np.log1p(-1.0 / M))
    return float(np.clip(alpha, 0.0, 1.0))


def target_pair_finite_pool_hazard(
    pair: tuple[int, int],
    *,
    aggregates: dict,
    mutation_probabilities: np.ndarray,
    pool_slot_selection_probability: float,
) -> float:
    r"""Finite-pool occupancy hazard for one shell-2 target {i,j}.

    Every potential offspring position produced by a parent at distance d from
    the target reaches that target with probability Q_d. Conditional on being
    in the pool, that position is sampled at least once with probability alpha.
    The target-absence approximation is therefore

        P_abs(g) approximately equals product_a (1 - alpha Q_{d_a}),

    and the corresponding additive hazard is

        h_g = -sum_a log(1 - alpha Q_{d_a}).

    Parent contributions are evaluated through the precomputed shell/locus/pair
    reproductive aggregates, avoiding an all-parent by all-target loop.
    """
    first, second = pair
    shell_totals = aggregates["shell_totals"]
    locus_totals = aggregates["locus_totals"]
    pair_totals = aggregates["pair_totals"]
    detailed_max_shell = int(aggregates["detailed_max_shell"])
    maximum_transition = len(mutation_probabilities) - 1
    alpha = float(pool_slot_selection_probability)

    def contribution(reproductive_mass: float, distance: int) -> float:
        if reproductive_mass <= 0.0:
            return 0.0
        if distance < 0 or distance > maximum_transition:
            return 0.0
        transition = float(mutation_probabilities[distance])
        if transition <= 0.0 or alpha <= 0.0:
            return 0.0
        selected_target_probability = alpha * transition
        if selected_target_probability >= 1.0:
            return math.inf
        return -float(reproductive_mass) * math.log1p(-selected_target_probability)

    hazard = 0.0
    for shell in range(detailed_max_shell + 1):
        total = float(shell_totals[shell])
        if total <= 0.0:
            continue

        both = (
            float(pair_totals[shell].get(pair, 0.0))
            if shell < len(pair_totals)
            else 0.0
        )
        first_total = float(locus_totals[shell, first])
        second_total = float(locus_totals[shell, second])

        exactly_one = max(0.0, first_total + second_total - 2.0 * both)
        neither = max(0.0, total - first_total - second_total + both)

        hazard += contribution(both, shell - 2)
        hazard += contribution(exactly_one, shell)
        hazard += contribution(neither, shell + 2)

    return max(0.0, float(hazard))


def probabilities_from_scaled_hazards(
    hazards: np.ndarray,
    presence_factors: np.ndarray,
    scale: float,
) -> np.ndarray:
    r"""Return p_g(c)=a_g[1-exp(-c h_g)] stably."""
    hazards = np.asarray(hazards, dtype=float)
    factors = np.asarray(presence_factors, dtype=float)
    if hazards.shape != factors.shape:
        raise ValueError("Hazards and presence factors must have the same shape.")
    if scale < 0.0:
        raise ValueError("Hazard scale cannot be negative.")

    probabilities = factors * (-np.expm1(-float(scale) * hazards))
    probabilities[~np.isfinite(probabilities)] = factors[
        ~np.isfinite(probabilities)
    ]
    return np.clip(probabilities, 0.0, 1.0)


def calibrate_hazard_scale_to_shell2_count(
    *,
    hazards: np.ndarray,
    presence_factors: np.ndarray,
    target_shell2_nodes: int,
) -> tuple[float, np.ndarray, dict]:
    r"""Choose c so sum_g a_g[1-exp(-c h_g)] equals observed S_2.

    The scalar calibration fixes only the total expected shell-2 occupancy.
    Relative differences among targets remain determined by their dynamic
    finite-pool hazards.
    """
    hazards = np.asarray(hazards, dtype=float)
    factors = np.asarray(presence_factors, dtype=float)
    if hazards.ndim != 1 or factors.ndim != 1 or hazards.shape != factors.shape:
        raise ValueError("Hazards and presence factors must be aligned vectors.")
    if np.any(hazards < 0.0) or np.any(~np.isfinite(hazards)):
        raise ValueError("Finite non-negative hazards are required for calibration.")
    if np.any((factors < 0.0) | (factors > 1.0) | ~np.isfinite(factors)):
        raise ValueError("Presence factors must lie in [0,1].")

    target = float(target_shell2_nodes)
    if target < 0.0:
        raise ValueError("Observed S_2 cannot be negative.")

    active = hazards > 0.0
    maximum_attainable = float(factors[active].sum())
    tolerance = max(
        1e-10,
        DYNAMIC_M1_CALIBRATION_RTOL * max(1.0, target, maximum_attainable),
    )

    unconditioned = probabilities_from_scaled_hazards(hazards, factors, 1.0)
    unconditioned_total = float(unconditioned.sum())

    if target <= tolerance:
        probabilities = np.zeros_like(hazards)
        return 0.0, probabilities, {
            "unconditioned_expected_shell2_nodes": unconditioned_total,
            "conditioned_expected_shell2_nodes": 0.0,
            "maximum_attainable_shell2_nodes": maximum_attainable,
            "calibration_iterations": 0,
            "calibration_residual": -target,
        }

    if target > maximum_attainable + tolerance:
        raise ValueError(
            "Observed LCC shell-2 count cannot be attained by the dynamic "
            f"hazards: S2={target:g}, maximum={maximum_attainable:g}. "
            "Increase DYNAMIC_M1_MAX_DISTANCE or verify target viability metadata."
        )

    if abs(unconditioned_total - target) <= tolerance:
        return 1.0, unconditioned, {
            "unconditioned_expected_shell2_nodes": unconditioned_total,
            "conditioned_expected_shell2_nodes": unconditioned_total,
            "maximum_attainable_shell2_nodes": maximum_attainable,
            "calibration_iterations": 0,
            "calibration_residual": unconditioned_total - target,
        }

    # Bracket the unique monotone solution.
    low = 0.0
    high = 1.0
    high_total = unconditioned_total
    while high_total < target and high < 1e16:
        high *= 2.0
        high_total = float(
            probabilities_from_scaled_hazards(hazards, factors, high).sum()
        )

    if high_total < target - tolerance:
        raise FloatingPointError(
            "Could not bracket the conditioned shell-2 hazard scale."
        )

    iterations = 0
    for iterations in range(1, DYNAMIC_M1_CALIBRATION_MAX_ITERATIONS + 1):
        midpoint = 0.5 * (low + high)
        midpoint_total = float(
            probabilities_from_scaled_hazards(hazards, factors, midpoint).sum()
        )

        if abs(midpoint_total - target) <= tolerance:
            low = high = midpoint
            break
        if midpoint_total < target:
            low = midpoint
        else:
            high = midpoint

    scale = 0.5 * (low + high)
    probabilities = probabilities_from_scaled_hazards(hazards, factors, scale)
    conditioned_total = float(probabilities.sum())

    return float(scale), probabilities, {
        "unconditioned_expected_shell2_nodes": unconditioned_total,
        "conditioned_expected_shell2_nodes": conditioned_total,
        "maximum_attainable_shell2_nodes": maximum_attainable,
        "calibration_iterations": int(iterations),
        "calibration_residual": conditioned_total - target,
    }


# ============================================================
# Analytical K_nn(k) approximation
# ============================================================


def lcc_shell_presence_probabilities(
    lcc_shell_counts: np.ndarray,
    *,
    L: int,
) -> np.ndarray:
    r"""Observed fraction of possible shell-m genotypes lying in the LCC.

        u_m = S_m / C(L,m).

    The calculation is performed in logarithmic form because C(L,m) can be
    extremely large even when L is only a few hundred.
    """
    counts = np.asarray(lcc_shell_counts, dtype=float)
    if counts.shape != (L + 1,):
        raise ValueError(f"Expected {L + 1} LCC shell counts, got {counts.shape}.")

    probabilities = np.zeros(L + 1, dtype=float)
    positive = counts > 0
    log_sizes = shell_log_sizes(L)
    probabilities[positive] = np.exp(
        np.log(counts[positive]) - log_sizes[positive]
    )
    return np.clip(probabilities, 0.0, 1.0)


def choose_neighbor_presence_probabilities(
    *,
    population_presence_probabilities: np.ndarray,
    lcc_shell_counts: np.ndarray,
    L: int,
) -> np.ndarray:
    """Select the shell occupancy used for expected neighbor degrees."""
    mode = KNN_NEIGHBOR_OCCUPANCY_SOURCE.lower().strip()
    if mode == "population_poisson":
        return np.asarray(population_presence_probabilities, dtype=float).copy()
    if mode == "lcc_observed":
        return lcc_shell_presence_probabilities(lcc_shell_counts, L=L)
    raise ValueError(
        "KNN_NEIGHBOR_OCCUPANCY_SOURCE must be either "
        "'lcc_observed' or 'population_poisson'."
    )


def condition_pmf_on_positive_degree(
    pmf: np.ndarray,
    knn_weighted_pmf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition P(k) and P(k)K_nn(k) on k>=1."""
    pmf = np.asarray(pmf, dtype=float).copy()
    weighted = np.asarray(knn_weighted_pmf, dtype=float).copy()
    if pmf.shape != weighted.shape:
        raise ValueError("PMF and weighted K_nn arrays must have equal shape.")

    if not CONDITION_THEORY_ON_POSITIVE_DEGREE or len(pmf) <= 1:
        return pmf, weighted

    positive_mass = float(pmf[1:].sum())
    if positive_mass <= 0.0 or not np.isfinite(positive_mass):
        return pmf, weighted

    pmf[0] = 0.0
    weighted[0] = 0.0
    pmf /= positive_mass
    weighted /= positive_mass
    return pmf, weighted


def leave_one_out_poisson_binomial_pmf(
    full_pmf: np.ndarray,
    probability: float,
) -> np.ndarray:
    r"""Recover the PMF after removing one Bernoulli variable.

    If P_s is the full Poisson-binomial PMF and Q_s is the PMF without a
    Bernoulli variable of probability p, then

        P_s = (1-p) Q_s + p Q_{s-1}.

    A forward recurrence is stable for p<=1/2 and a backward recurrence is
    used for p>1/2.
    """
    full = np.asarray(full_pmf, dtype=float)
    p = float(np.clip(probability, 0.0, 1.0))
    n = len(full) - 1
    if n < 1:
        return np.array([1.0], dtype=float)

    if p <= 1e-15:
        result = full[:-1].copy()
    elif p >= 1.0 - 1e-15:
        result = full[1:].copy()
    elif p <= 0.5:
        result = np.zeros(n, dtype=float)
        one_minus_p = 1.0 - p
        result[0] = full[0] / one_minus_p
        for s in range(1, n):
            result[s] = (full[s] - p * result[s - 1]) / one_minus_p
    else:
        result = np.zeros(n, dtype=float)
        result[-1] = full[-1] / p
        one_minus_p = 1.0 - p
        for s in range(n - 1, 0, -1):
            result[s - 1] = (full[s] - one_minus_p * result[s]) / p

    result[np.abs(result) < 1e-14] = 0.0
    result = np.clip(result, 0.0, None)
    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("Invalid leave-one-out Poisson-binomial PMF.")
    return result / total


def marked_poisson_binomial_distribution(
    probabilities: np.ndarray,
    marks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return P(S=s) and E[sum_j a_j B_j 1_{S=s}].

    Here S=sum_j B_j, B_j~Bernoulli(p_j), and ``marks`` contains a_j. The
    marked numerator is

        A_s = sum_j a_j p_j P(S_{-j}=s-1).

    This quantity allows the expected total degree of selected neighbors to be
    calculated while conditioning on the focal node degree.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    marks = np.asarray(marks, dtype=float)
    if probabilities.ndim != 1 or marks.ndim != 1:
        raise ValueError("Probabilities and marks must be one-dimensional.")
    if probabilities.shape != marks.shape:
        raise ValueError("Probabilities and marks must have equal length.")

    full_pmf = poisson_binomial_pmf(probabilities)
    marked = np.zeros_like(full_pmf)

    for probability, mark in zip(probabilities, marks):
        if probability <= 0.0 or mark == 0.0:
            continue
        without = leave_one_out_poisson_binomial_pmf(full_pmf, probability)
        marked[1:] += float(mark) * float(probability) * without

    return full_pmf, marked


def homogeneous_shell_knn_theory(
    *,
    shell: int,
    L: int,
    degree_presence_probabilities: np.ndarray,
    neighbor_presence_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r"""First-order shell prediction for P(k|m) and K_nn(k|m).

    For a shell-m focal node,

        X^- ~ Binomial(m,p_{m-1}),
        X^+ ~ Binomial(L-m,p_{m+1}),
        K = X^- + X^+.

    Conditional on an existing focal edge, the expected degree of an inner or
    outer neighbor is

        kappa^-_m = 1 + (m-1)u_{m-2} + (L-m)u_m,
        kappa^+_m = 1 + m u_m + (L-m-1)u_{m+2}.

    ``degree_presence_probabilities`` supplies p_m and
    ``neighbor_presence_probabilities`` supplies u_m.
    """
    m = int(shell)
    if not 0 <= m <= L:
        raise ValueError(f"Shell m={m} lies outside [0,{L}].")

    p_minus = float(degree_presence_probabilities[m - 1]) if m > 0 else 0.0
    p_plus = float(degree_presence_probabilities[m + 1]) if m < L else 0.0

    lower_pmf = binomial_pmf(m, p_minus)
    upper_pmf = binomial_pmf(L - m, p_plus)
    pmf = np.convolve(lower_pmf, upper_pmf)

    # E[X^- 1_{K=k}] is a convolution with the lower PMF multiplied by r.
    inward_count_weighted = np.convolve(
        np.arange(m + 1, dtype=float) * lower_pmf,
        upper_pmf,
    )

    def occupancy(index: int) -> float:
        if 0 <= index <= L:
            return float(neighbor_presence_probabilities[index])
        return 0.0

    kappa_minus = 1.0 + (m - 1) * occupancy(m - 2) + (L - m) * occupancy(m)
    kappa_plus = 1.0 + m * occupancy(m) + (L - m - 1) * occupancy(m + 2)

    knn_weighted = np.zeros(L + 1, dtype=float)
    for degree in range(1, L + 1):
        inward = float(inward_count_weighted[degree])
        outward = max(0.0, degree * float(pmf[degree]) - inward)
        neighbor_degree_sum = kappa_minus * inward + kappa_plus * outward
        knn_weighted[degree] = neighbor_degree_sum / degree

    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    pmf, knn_weighted = condition_pmf_on_positive_degree(pmf, knn_weighted)

    return pmf, knn_weighted, {
        "shell": m,
        "p_minus": p_minus,
        "p_plus": p_plus,
        "kappa_minus": float(kappa_minus),
        "kappa_plus": float(kappa_plus),
    }


def conditioned_shell2_pair_probabilities(
    *,
    observed_shell2_nodes: int,
    previous_genotype_counts: Counter[tuple[int, ...]],
    metadata: dict,
    run_metadata: dict,
) -> tuple[dict[tuple[int, int], float], dict]:
    """Condition the dynamic shell-2 target probabilities on observed S_2."""
    L = int(metadata["L"])
    mutation_probabilities = mutation_transition_probabilities(
        L=L,
        mean_mutations=float(metadata["mu"]),
        max_distance=DYNAMIC_M1_MAX_DISTANCE,
    )
    aggregates = build_parent_reproductive_aggregates(
        previous_genotype_counts,
        L=L,
        f_peak=float(metadata["fpeak"]),
        f_background=float(metadata["fbg"]),
        max_distance=DYNAMIC_M1_MAX_DISTANCE,
    )
    expected_pool = expected_viable_offspring_pool_size(
        shell_reproductive_totals=aggregates["shell_totals"],
        mutation_probabilities=mutation_probabilities,
        q=float(metadata["q"]),
    )
    alpha = finite_pool_slot_selection_probability(
        expected_pool_size=expected_pool,
        sample_size=int(metadata["N"]),
    )

    all_pairs = list(combinations(range(L), 2))
    hazards = np.zeros(len(all_pairs), dtype=float)
    presence_factors = np.ones(len(all_pairs), dtype=float)
    q = float(metadata["q"])

    for pair_index, pair in enumerate(all_pairs):
        viability = deterministic_target_viability(
            pair,
            metadata=metadata,
            run_metadata=run_metadata,
        )
        if viability is False:
            presence_factors[pair_index] = 0.0
            continue
        if viability is None:
            presence_factors[pair_index] = 1.0 - q

        hazards[pair_index] = target_pair_finite_pool_hazard(
            pair,
            aggregates=aggregates,
            mutation_probabilities=mutation_probabilities,
            pool_slot_selection_probability=alpha,
        )

    if DYNAMIC_M1_CONDITION_ON_OBSERVED_S2:
        scale, probabilities, calibration = calibrate_hazard_scale_to_shell2_count(
            hazards=hazards,
            presence_factors=presence_factors,
            target_shell2_nodes=int(observed_shell2_nodes),
        )
    else:
        scale = 1.0
        probabilities = probabilities_from_scaled_hazards(
            hazards,
            presence_factors,
            scale,
        )
        total = float(probabilities.sum())
        calibration = {
            "unconditioned_expected_shell2_nodes": total,
            "conditioned_expected_shell2_nodes": total,
            "maximum_attainable_shell2_nodes": float(
                presence_factors[hazards > 0.0].sum()
            ),
            "calibration_iterations": 0,
            "calibration_residual": total - float(observed_shell2_nodes),
        }

    pair_probability = {
        pair: float(probability)
        for pair, probability in zip(all_pairs, probabilities)
    }
    diagnostics = {
        "expected_viable_pool_size": float(expected_pool),
        "finite_pool_slot_selection_probability": float(alpha),
        "hazard_calibration_scale": float(scale),
        "observed_lcc_shell2_nodes": int(observed_shell2_nodes),
        **calibration,
    }
    return pair_probability, diagnostics


def conditioned_m1_knn_theory(
    *,
    observed_m1_loci: list[int],
    observed_shell2_nodes: int,
    root_present_in_lcc: bool,
    previous_genotype_counts: Counter[tuple[int, ...]],
    metadata: dict,
    run_metadata: dict,
    neighbor_presence_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], float], dict]:
    r"""Conditioned genotype-resolved prediction for shell m=1.

    For first-shell genotype i,

        K_i = I_0 + sum_{j!=i} B_ij,
        B_ij ~ Bernoulli(p_ij).

    A selected shell-2 neighbor {i,j} has the focal edge, the second inward
    edge when genotype j is present, and approximately (L-2)u_3 outer edges.
    The marked Poisson-binomial distribution then gives K_nn at fixed K_i.
    """
    L = int(metadata["L"])
    loci = sorted(set(int(value) for value in observed_m1_loci))
    if not loci:
        raise ValueError("No first-shell LCC nodes are available.")

    pair_probability, diagnostics = conditioned_shell2_pair_probabilities(
        observed_shell2_nodes=observed_shell2_nodes,
        previous_genotype_counts=previous_genotype_counts,
        metadata=metadata,
        run_metadata=run_metadata,
    )

    observed_set = set(loci)
    root_indicator = 1 if root_present_in_lcc else 0
    root_degree = len(loci) if root_present_in_lcc else 0
    shell3_presence = (
        float(neighbor_presence_probabilities[3]) if L >= 3 else 0.0
    )

    mixture_pmf = np.zeros(L + 1, dtype=float)
    mixture_knn_weighted = np.zeros(L + 1, dtype=float)
    predicted_node_degrees: list[float] = []

    for locus in loci:
        probabilities = []
        neighbor_degree_marks = []

        for other in range(L):
            if other == locus:
                continue
            pair = (locus, other) if locus < other else (other, locus)
            probabilities.append(pair_probability[pair])
            neighbor_degree_marks.append(
                1.0
                + float(other in observed_set)
                + (L - 2) * shell3_presence
            )

        probabilities_array = np.asarray(probabilities, dtype=float)
        marks_array = np.asarray(neighbor_degree_marks, dtype=float)
        upper_pmf, marked_neighbor_sum = marked_poisson_binomial_distribution(
            probabilities_array,
            marks_array,
        )

        predicted_node_degrees.append(
            float(root_indicator + probabilities_array.sum())
        )

        for outer_count, probability_mass in enumerate(upper_pmf):
            degree = root_indicator + outer_count
            if degree > L:
                continue
            mixture_pmf[degree] += probability_mass
            if degree <= 0:
                continue

            root_neighbor_sum = (
                float(root_degree) * probability_mass
                if root_present_in_lcc
                else 0.0
            )
            total_neighbor_sum_weighted = (
                root_neighbor_sum + marked_neighbor_sum[outer_count]
            )
            mixture_knn_weighted[degree] += (
                total_neighbor_sum_weighted / degree
            )

    mixture_pmf /= len(loci)
    mixture_knn_weighted /= len(loci)
    mixture_pmf, mixture_knn_weighted = condition_pmf_on_positive_degree(
        mixture_pmf,
        mixture_knn_weighted,
    )

    root_knn_prediction = float(np.mean(predicted_node_degrees))
    diagnostics = {
        **diagnostics,
        "n_m1_nodes": int(len(loci)),
        "root_present_in_lcc": bool(root_present_in_lcc),
        "root_degree": int(root_degree),
        "shell3_neighbor_presence": float(shell3_presence),
        "mean_predicted_m1_degree": root_knn_prediction,
        "root_knn_prediction": root_knn_prediction,
    }
    return mixture_pmf, mixture_knn_weighted, pair_probability, diagnostics


def analytical_knn_description_markdown() -> str:
    occupancy_text = (
        r"$u_m=S_m/\binom{L}{m}$, measured inside the LCC"
        if KNN_NEIGHBOR_OCCUPANCY_SOURCE.lower().strip() == "lcc_observed"
        else r"$u_m=1-\exp[-n_m/\binom{L}{m}]$ from the population"
    )
    return rf"""
### Shell-dependent prediction for nearest-neighbor degree

For a node $i$ of degree $k_i$, the mean degree of its neighbors is

$$
k_{{\mathrm{{nn}},i}}=\frac{{1}}{{k_i}}
\sum_{{j\in\mathcal{{N}}(i)}}k_j,
$$

and the degree-resolved observable is
$k_\mathrm{{nn}}(k)=\langle k_{{\mathrm{{nn}},i}}\mid k_i=k\rangle$.
A shell-$m$ node has $X_m^-$ inward and $X_m^+$ outward edges. Conditional on
one such edge existing, the expected degrees of its inner and outer neighbors
are approximated by

$$
\kappa_m^-=1+(m-1)u_{{m-2}}+(L-m)u_m,
\qquad
\kappa_m^+=1+m u_m+(L-m-1)u_{{m+2}},
$$

where {occupancy_text}. Conditioning the inward-edge count on the total degree
gives

$$
k_\mathrm{{nn}}(k\mid m)
=\sum_r P(r\mid k,m)
\frac{{r\kappa_m^-+(k-r)\kappa_m^+}}{{k}}.
$$

The complete prediction is mixed over shells with the same expected node
counts used in the degree-distribution theory. Shell $m=1$ is treated with the
conditioned, genotype-resolved probabilities $p_{{ij}}$ obtained from the
$t-1\rightarrow t$ dynamics. The root is treated separately: its degree is the
observed number $S_1$ of first-shell LCC nodes, while its predicted nearest-
neighbor degree is the mean predicted degree of those first-shell nodes.

Panel (a) also shows the configuration-model reference
$\langle k^2\rangle/\langle k\rangle$. A systematic decrease below or across
this line is evidence of disassortative shell organization rather than a
consequence of the degree sequence alone.
"""


# ============================================================
# One-network empirical and analytical measurements
# ============================================================


def aggregate_node_knn_rows(
    *,
    metadata: dict,
    values_by_scope: dict[str, dict[int, list[float]]],
) -> list[dict]:
    rows: list[dict] = []
    for scope, degree_map in values_by_scope.items():
        for degree, values in sorted(degree_map.items()):
            array = np.asarray(values, dtype=float)
            rows.append({
                **metadata,
                "scope": scope,
                "degree": int(degree),
                "node_count": int(len(array)),
                "knn_sum": float(array.sum()),
                "knn_mean_network": float(array.mean()),
            })
    return rows


def append_theory_arrays(
    rows: list[dict],
    *,
    metadata: dict,
    scope: str,
    expected_nodes: float,
    degree_pmf: np.ndarray,
    knn_weighted_pmf: np.ndarray,
    shell: int | None,
) -> None:
    """Append expected counts and expected K_nn sums for one component."""
    for degree, probability in enumerate(np.asarray(degree_pmf, dtype=float)):
        expected_count = float(expected_nodes) * float(probability)
        expected_knn_sum = (
            float(expected_nodes) * float(knn_weighted_pmf[degree])
        )
        if expected_count <= 0.0 and expected_knn_sum <= 0.0:
            continue
        rows.append({
            **metadata,
            "scope": scope,
            "shell": shell,
            "degree": int(degree),
            "expected_count": expected_count,
            "expected_knn_sum": expected_knn_sum,
        })


def read_network_knn_measurements(
    network_file: Path,
    metadata: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Measure empirical K_nn and construct the conditioned shell theory."""
    graph = to_simple_undirected_graph(nx.read_graphml(network_file))
    if graph.number_of_nodes() == 0:
        raise ValueError("empty graph")

    lcc_nodes, n_components = largest_component_nodes(graph)
    lcc = graph.subgraph(lcc_nodes).copy()
    L = int(metadata["L"])

    node_genotypes: dict[object, tuple[int, ...]] = {}
    genotype_to_node: dict[tuple[int, ...], object] = {}
    lcc_shell_counts = np.zeros(L + 1, dtype=np.int64)
    observed_m1_loci: list[int] = []

    for node, attributes in lcc.nodes(data=True):
        genotype = genotype_tuple_from_name(
            genotype_name(node, attributes),
            L=L,
            validate=True,
        )
        node_genotypes[node] = genotype
        genotype_to_node[genotype] = node
        lcc_shell_counts[len(genotype)] += 1
        if len(genotype) == 1:
            observed_m1_loci.append(genotype[0])

    root_node = genotype_to_node.get(())
    root_present = root_node is not None
    degrees = {node: int(value) for node, value in lcc.degree()}

    values_by_scope: dict[str, dict[int, list[float]]] = {
        "all": {},
        **{category: {} for category in MUTATION_CLASS_ORDER},
    }

    for node, degree in degrees.items():
        if degree <= 0:
            continue
        neighbor_degrees = [degrees[neighbor] for neighbor in lcc.neighbors(node)]
        knn_value = float(np.mean(neighbor_degrees))
        category = mutation_class(len(node_genotypes[node]))

        values_by_scope["all"].setdefault(degree, []).append(knn_value)
        values_by_scope[category].setdefault(degree, []).append(knn_value)

    empirical_rows = aggregate_node_knn_rows(
        metadata=metadata,
        values_by_scope=values_by_scope,
    )

    abundance_file = resolve_abundance_file(network_file, metadata)
    _, population_shell_counts, observed_population = read_population_genotype_counts(
        abundance_file,
        L=L,
        expected_population_size=int(metadata["N"]),
    )
    degree_presence = shell_representation_probabilities(
        population_shell_counts,
        L=L,
    )
    neighbor_presence = choose_neighbor_presence_probabilities(
        population_presence_probabilities=degree_presence,
        lcc_shell_counts=lcc_shell_counts,
        L=L,
    )

    shell2_nodes = int(lcc_shell_counts[2]) if L >= 2 else 0
    previous_abundance_file: Path | None = None
    previous_population_size: int | None = None

    if USE_DYNAMIC_M1_APPROXIMATION:
        previous_time = int(metadata["t"]) - int(DYNAMIC_M1_PREVIOUS_TIME_OFFSET)
        if previous_time < 0:
            raise ValueError("The requested previous time is negative.")

        try:
            previous_abundance_file = resolve_abundance_file_at_time(
                network_file,
                metadata,
                target_time=previous_time,
            )
            previous_genotype_counts, _, previous_population_size = (
                read_population_genotype_counts(
                    previous_abundance_file,
                    L=L,
                    expected_population_size=int(metadata["N"]),
                )
            )
            run_metadata = load_run_metadata(metadata)
            m1_pmf, m1_knn_weighted, _, m1_diagnostics = (
                conditioned_m1_knn_theory(
                    observed_m1_loci=observed_m1_loci,
                    observed_shell2_nodes=shell2_nodes,
                    root_present_in_lcc=root_present,
                    previous_genotype_counts=previous_genotype_counts,
                    metadata=metadata,
                    run_metadata=run_metadata,
                    neighbor_presence_probabilities=neighbor_presence,
                )
            )
            m1_diagnostics["method"] = (
                "conditioned_finite_pool_poisson_binomial_tminus1"
            )
        except FileNotFoundError:
            if REQUIRE_DYNAMIC_M1_INPUT:
                raise
            m1_pmf, m1_knn_weighted, _ = homogeneous_shell_knn_theory(
                shell=1,
                L=L,
                degree_presence_probabilities=degree_presence,
                neighbor_presence_probabilities=neighbor_presence,
            )
            mean_m1_degree = float(
                np.dot(np.arange(L + 1, dtype=float), m1_pmf)
            )
            m1_diagnostics = {
                "method": "homogeneous_shell_fallback",
                "hazard_calibration_scale": math.nan,
                "calibration_residual": math.nan,
                "mean_predicted_m1_degree": mean_m1_degree,
                "root_knn_prediction": mean_m1_degree,
            }
    else:
        m1_pmf, m1_knn_weighted, _ = homogeneous_shell_knn_theory(
            shell=1,
            L=L,
            degree_presence_probabilities=degree_presence,
            neighbor_presence_probabilities=neighbor_presence,
        )
        mean_m1_degree = float(
            np.dot(np.arange(L + 1, dtype=float), m1_pmf)
        )
        m1_diagnostics = {
            "method": "homogeneous_shell",
            "hazard_calibration_scale": math.nan,
            "calibration_residual": math.nan,
            "mean_predicted_m1_degree": mean_m1_degree,
            "root_knn_prediction": mean_m1_degree,
        }

    theory_rows: list[dict] = []

    # Root: exact degree within the observed LCC and dynamically predicted mean
    # degree of its first-shell neighbors.
    if root_present:
        root_degree = degrees[root_node]
        root_pmf = np.zeros(L + 1, dtype=float)
        root_weighted = np.zeros(L + 1, dtype=float)
        root_pmf[root_degree] = 1.0
        root_weighted[root_degree] = float(
            m1_diagnostics["root_knn_prediction"]
        )
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="all",
            expected_nodes=1.0,
            degree_pmf=root_pmf,
            knn_weighted_pmf=root_weighted,
            shell=0,
        )
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="m=0",
            expected_nodes=1.0,
            degree_pmf=root_pmf,
            knn_weighted_pmf=root_weighted,
            shell=0,
        )

    # First shell: conditioned finite-pool Poisson-binomial approximation.
    shell1_nodes = int(lcc_shell_counts[1]) if L >= 1 else 0
    if shell1_nodes > 0:
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="all",
            expected_nodes=float(shell1_nodes),
            degree_pmf=m1_pmf,
            knn_weighted_pmf=m1_knn_weighted,
            shell=1,
        )
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="m=1",
            expected_nodes=float(shell1_nodes),
            degree_pmf=m1_pmf,
            knn_weighted_pmf=m1_knn_weighted,
            shell=1,
        )

    # Outer shells: homogeneous adjacent-shell approximation.
    shell_rows: list[dict] = []
    log_sizes = shell_log_sizes(L)
    for shell in range(L + 1):
        shell_rows.append({
            **metadata,
            "shell": int(shell),
            "population_count": int(population_shell_counts[shell]),
            "population_presence_probability": float(degree_presence[shell]),
            "lcc_shell_nodes": int(lcc_shell_counts[shell]),
            "lcc_presence_probability": float(
                lcc_shell_presence_probabilities(lcc_shell_counts, L=L)[shell]
            ),
            "neighbor_presence_probability": float(neighbor_presence[shell]),
            "log_shell_size": float(log_sizes[shell]),
        })

    for shell in range(2, L + 1):
        shell_nodes = int(lcc_shell_counts[shell])
        if shell_nodes <= 0:
            continue
        pmf, weighted, diagnostics = homogeneous_shell_knn_theory(
            shell=shell,
            L=L,
            degree_presence_probabilities=degree_presence,
            neighbor_presence_probabilities=neighbor_presence,
        )
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="all",
            expected_nodes=float(shell_nodes),
            degree_pmf=pmf,
            knn_weighted_pmf=weighted,
            shell=shell,
        )
        append_theory_arrays(
            theory_rows,
            metadata=metadata,
            scope="m>=2",
            expected_nodes=float(shell_nodes),
            degree_pmf=pmf,
            knn_weighted_pmf=weighted,
            shell=shell,
        )
        shell_rows[-(L + 1 - shell)]["kappa_minus"] = diagnostics["kappa_minus"]
        shell_rows[-(L + 1 - shell)]["kappa_plus"] = diagnostics["kappa_plus"]

    degree_array = np.fromiter(degrees.values(), dtype=float)
    degree_sum = float(degree_array.sum())
    degree_square_sum = float(np.square(degree_array).sum())
    uncorrelated_baseline = (
        degree_square_sum / degree_sum if degree_sum > 0 else math.nan
    )

    try:
        assortativity = float(nx.degree_assortativity_coefficient(lcc))
    except Exception:
        assortativity = math.nan

    inventory_row = {
        **metadata,
        "abundance_file": str(abundance_file),
        "previous_abundance_file": (
            str(previous_abundance_file)
            if previous_abundance_file is not None
            else None
        ),
        "observed_population": int(observed_population),
        "full_nodes": int(graph.number_of_nodes()),
        "full_edges": int(graph.number_of_edges()),
        "n_components": int(n_components),
        "lcc_nodes": int(lcc.number_of_nodes()),
        "lcc_edges": int(lcc.number_of_edges()),
        "degree_sum": degree_sum,
        "degree_square_sum": degree_square_sum,
        "uncorrelated_knn_baseline": float(uncorrelated_baseline),
        "degree_assortativity": assortativity,
        "root_present": bool(root_present),
        "root_degree": int(degrees[root_node]) if root_present else 0,
        "nodes_m0": int(lcc_shell_counts[0]),
        "nodes_m1": int(lcc_shell_counts[1]) if L >= 1 else 0,
        "nodes_m2": int(lcc_shell_counts[2]) if L >= 2 else 0,
        "nodes_mge2": int(lcc_shell_counts[2:].sum()) if L >= 2 else 0,
        "m1_theory_method": m1_diagnostics["method"],
        "m1_hazard_scale": m1_diagnostics["hazard_calibration_scale"],
        "m1_calibration_residual": m1_diagnostics["calibration_residual"],
        "m1_predicted_mean_degree": m1_diagnostics["mean_predicted_m1_degree"],
        "root_knn_prediction": m1_diagnostics["root_knn_prediction"],
    }

    return inventory_row, empirical_rows, theory_rows, shell_rows


# ============================================================
# Discovery and aggregation across independent networks
# ============================================================


def collect_selected_knn_networks() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    root = Path(NETWORK_SEARCH_ROOT).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Network search root not found: {root}")

    graphml_files = sorted(root.rglob("dataOut_*.graphml"))
    if not graphml_files:
        raise RuntimeError(
            "No model-fq GraphML files were found below:\n"
            f"  {root}\n"
            "Expected runs/<run_id>/Networks/<snapshot>.graphml."
        )

    catalog: list[dict] = []
    selected: list[tuple[Path, dict]] = []
    for network_file in graphml_files:
        metadata = parse_network_metadata(network_file)
        if metadata is None:
            continue
        catalog.append(metadata)
        if metadata_is_selected(metadata):
            selected.append((network_file, metadata))

    print(f"Discovered GraphML files: {len(graphml_files)}")
    print(f"Parseable model-fq files: {len(catalog)}")
    print(f"Selection: {selector_description()}")
    print(f"Selected independent networks: {len(selected)}")

    if not selected:
        if catalog:
            available = (
                pd.DataFrame(catalog)[
                    ["N", "L", "fpeak", "q", "t", "iteration", "mu", "fbg", "run_id"]
                ]
                .drop_duplicates()
                .sort_values(["N", "L", "fpeak", "q", "t", "run_id", "iteration"])
            )
            raise RuntimeError(
                "No GraphML files matched the configured filters.\n\n"
                + available.head(50).to_string(index=False)
            )
        raise RuntimeError("No parseable model-fq GraphML files were found.")

    inventory_rows: list[dict] = []
    empirical_rows: list[dict] = []
    theory_rows: list[dict] = []
    shell_rows: list[dict] = []
    failures: list[tuple[Path, str]] = []

    for index, (network_file, metadata) in enumerate(selected, start=1):
        try:
            inventory, empirical, theory, shells = read_network_knn_measurements(
                network_file,
                metadata,
            )
        except Exception as error:
            failures.append((network_file, f"{type(error).__name__}: {error}"))
            continue

        inventory_rows.append(inventory)
        empirical_rows.extend(empirical)
        theory_rows.extend(theory)
        shell_rows.extend(shells)

        print(
            f"[{index:>4}/{len(selected)}] q={metadata['q']:g}, "
            f"t={metadata['t']}, run={metadata['run_id']}, "
            f"iteration={metadata['iteration']}: "
            f"LCC={inventory['lcc_nodes']:,}, "
            f"r={inventory['degree_assortativity']:.4g}, "
            f"Knn_unc={inventory['uncorrelated_knn_baseline']:.4g}"
        )

    if failures:
        print("\nSkipped files:")
        for path, reason in failures:
            print(f"  {path}: {reason}")

    if not inventory_rows:
        raise RuntimeError("Every selected GraphML file failed during analysis.")

    inventory_df = pd.DataFrame(inventory_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network"]
    ).reset_index(drop=True)
    empirical_df = pd.DataFrame(empirical_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "scope", "degree"]
    ).reset_index(drop=True)
    theory_df = pd.DataFrame(theory_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "scope", "degree", "shell"]
    ).reset_index(drop=True)
    shell_df = pd.DataFrame(shell_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "shell"]
    ).reset_index(drop=True)
    return inventory_df, empirical_df, theory_df, shell_df


def knn_analysis_group_columns() -> list[str]:
    columns = ["mu", "N", "L", "q", "fpeak", "fbg"]
    if SPLIT_BY_TIME:
        columns.append("t")
    return columns


def aggregate_knn_results(
    inventory: pd.DataFrame,
    empirical: pd.DataFrame,
    theory: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = knn_analysis_group_columns()

    empirical_agg = (
        empirical
        .groupby(group_columns + ["scope", "degree"], as_index=False, dropna=False)
        .agg(
            empirical_node_count=("node_count", "sum"),
            empirical_knn_sum=("knn_sum", "sum"),
        )
    )
    theory_agg = (
        theory
        .groupby(group_columns + ["scope", "degree"], as_index=False, dropna=False)
        .agg(
            theory_expected_count=("expected_count", "sum"),
            theory_expected_knn_sum=("expected_knn_sum", "sum"),
        )
    )
    totals = (
        inventory
        .groupby(group_columns, as_index=False, dropna=False)
        .agg(
            n_networks=("network", "nunique"),
            total_lcc_nodes=("lcc_nodes", "sum"),
            total_degree_sum=("degree_sum", "sum"),
            total_degree_square_sum=("degree_square_sum", "sum"),
            mean_assortativity=("degree_assortativity", "mean"),
            std_assortativity=("degree_assortativity", "std"),
        )
    )
    totals["uncorrelated_knn_baseline"] = (
        totals["total_degree_square_sum"] / totals["total_degree_sum"]
    )

    merged = empirical_agg.merge(
        theory_agg,
        on=group_columns + ["scope", "degree"],
        how="outer",
    ).merge(totals, on=group_columns, how="left")

    for column in (
        "empirical_node_count",
        "empirical_knn_sum",
        "theory_expected_count",
        "theory_expected_knn_sum",
    ):
        merged[column] = merged[column].fillna(0.0)

    merged["empirical_knn"] = np.where(
        merged["empirical_node_count"] > 0,
        merged["empirical_knn_sum"] / merged["empirical_node_count"],
        np.nan,
    )
    merged["theory_knn"] = np.where(
        merged["theory_expected_count"] > 0,
        merged["theory_expected_knn_sum"] / merged["theory_expected_count"],
        np.nan,
    )

    scope_order = pd.CategoricalDtype(
        categories=["all", *MUTATION_CLASS_ORDER],
        ordered=True,
    )
    merged["scope"] = merged["scope"].astype(scope_order)
    return merged.sort_values(
        group_columns + ["scope", "degree"]
    ).reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================


def valid_knn_curve(data: pd.DataFrame, column: str) -> pd.DataFrame:
    curve = data.loc[
        np.isfinite(data[column])
        & (data[column] > 0)
        & (data["degree"] > 0)
    ].copy()
    return curve.sort_values("degree")


def configure_knn_axis(ax: plt.Axes) -> None:
    if LOG_X:
        ax.set_xscale("log")
    if LOG_Y:
        ax.set_yscale("log")
    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    if LOG_X or LOG_Y:
        ax.grid(True, which="minor", linewidth=0.45, alpha=0.10)


def make_knn_figure_tag(metadata: dict) -> str:
    tag = (
        f"{MODEL}_LCC_knn_panel_conditioned_shell_theory"
        f"_mu_{format_number(metadata['mu'])}"
        f"_N_{format_number(metadata['N'])}"
        f"_L_{format_number(metadata['L'])}"
        f"_q_{format_number(metadata['q'])}"
        f"_fpeak_{format_number(metadata['fpeak'])}"
        f"_fbg_{format_number(metadata['fbg'])}"
    )
    if SPLIT_BY_TIME:
        tag += f"_t_{format_number(metadata['t'])}"
    else:
        tag += "_selected_times_pooled"
    return tag


def plot_knn_panels(aggregated: pd.DataFrame) -> list[Path]:
    group_columns = knn_analysis_group_columns()
    saved_paths: list[Path] = []

    for group_key, group_data in aggregated.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(group_columns, group_key))

        fig, axes = plt.subplots(
            1,
            2,
            figsize=FIGSIZE,
            gridspec_kw={"wspace": 0.28},
        )

        # Panel (a): complete LCC K_nn(k).
        all_data = group_data.loc[group_data["scope"] == "all"].copy()
        empirical_all = valid_knn_curve(all_data, "empirical_knn")
        theory_all = valid_knn_curve(all_data, "theory_knn")

        axes[0].plot(
            empirical_all["degree"],
            empirical_all["empirical_knn"],
            linestyle="none",
            marker="o",
            markersize=4.8,
            label="Simulation",
            zorder=4,
        )
        if ADD_ANALYTICAL_THEORY and not theory_all.empty:
            axes[0].plot(
                theory_all["degree"],
                theory_all["theory_knn"],
                linestyle="--",
                linewidth=1.8,
                label="Conditioned shell theory",
                zorder=3,
            )

        baseline = float(group_data["uncorrelated_knn_baseline"].iloc[0])
        if SHOW_UNCORRELATED_BASELINE and np.isfinite(baseline) and baseline > 0:
            axes[0].axhline(
                baseline,
                linestyle=":",
                linewidth=1.45,
                label=r"Uncorrelated $\langle k^2\rangle/\langle k\rangle$",
                zorder=2,
            )

        configure_knn_axis(axes[0])
        axes[0].set_xlabel(r"Node degree, $k$")
        axes[0].set_ylabel(r"Mean nearest-neighbor degree, $k_{\rm nn}(k)$")
        axes[0].set_title(r"$\mathbf{a}$  Complete LCC", loc="left")
        axes[0].legend(frameon=False, loc="best")

        # Panel (b): mutation-class decomposition.
        class_handles: list[Line2D] = []
        for category in MUTATION_CLASS_ORDER:
            class_data = group_data.loc[group_data["scope"] == category].copy()
            empirical_class = valid_knn_curve(class_data, "empirical_knn")
            theory_class = valid_knn_curve(class_data, "theory_knn")
            if empirical_class.empty and theory_class.empty:
                continue

            empirical_line, = axes[1].plot(
                empirical_class["degree"],
                empirical_class["empirical_knn"],
                linestyle="none",
                marker=MUTATION_CLASS_MARKERS[category],
                markersize=4.8,
                zorder=4,
            )
            color = empirical_line.get_color()

            if ADD_ANALYTICAL_THEORY and not theory_class.empty:
                axes[1].plot(
                    theory_class["degree"],
                    theory_class["theory_knn"],
                    linestyle="--",
                    linewidth=1.55,
                    color=color,
                    zorder=3,
                )

            class_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=MUTATION_CLASS_MARKERS[category],
                    linestyle="none",
                    markersize=5.2,
                    label=MUTATION_CLASS_LABELS[category],
                )
            )

        configure_knn_axis(axes[1])
        axes[1].set_xlabel(r"Node degree, $k$")
        axes[1].set_ylabel(r"Mean nearest-neighbor degree, $k_{\rm nn}(k\mid c)$")
        axes[1].set_title(r"$\mathbf{b}$  Mutation-class decomposition", loc="left")

        class_legend = axes[1].legend(
            handles=class_handles,
            title="Mutation class",
            frameon=False,
            loc="best",
        )
        axes[1].add_artist(class_legend)
        style_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", color="black",
                markersize=4.8, label="Simulation",
            )
        ]
        if ADD_ANALYTICAL_THEORY:
            style_handles.append(
                Line2D(
                    [0], [0], linestyle="--", color="black",
                    linewidth=1.55, label="Shell theory",
                )
            )
        axes[1].legend(handles=style_handles, frameon=False, loc="lower left")

        n_networks = int(group_data["n_networks"].max())
        total_lcc_nodes = int(group_data["total_lcc_nodes"].max())
        mean_r = float(group_data["mean_assortativity"].iloc[0])
        std_r = float(group_data["std_assortativity"].iloc[0])

        title = (
            rf"$N={int(metadata['N']):,}$, "
            rf"$L={int(metadata['L'])}$, "
            rf"$q={metadata['q']:g}$, "
            rf"$f_{{\rm peak}}={metadata['fpeak']:g}$, "
            rf"$f_{{\rm bg}}={metadata['fbg']:g}$, "
            rf"$\lambda={metadata['mu']:g}$"
        )
        if SPLIT_BY_TIME:
            title += rf", $t={int(metadata['t'])}$"
        else:
            title += ", selected times pooled"

        assortativity_text = (
            rf"$r={mean_r:.3g}\pm{std_r:.2g}$"
            if np.isfinite(std_r)
            else rf"$r={mean_r:.3g}$"
        )
        fig.suptitle(
            title
            + f"\n{n_networks} independent networks; "
            + f"{total_lcc_nodes:,} pooled LCC nodes; "
            + assortativity_text,
            y=1.04,
            fontsize=13,
        )
        fig.subplots_adjust(top=0.81, bottom=0.15, left=0.085, right=0.985)

        if SAVE_FIGS:
            output_dir = Path(OUTPUT_DIR).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / (
                f"{make_knn_figure_tag(metadata)}.{FIGURE_FORMAT.lstrip('.')}"
            )
            fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")
            saved_paths.append(destination)
            print(f"Saved figure: {destination}")

        if SHOW_FIGS:
            plt.show()
        plt.close(fig)

    return saved_paths


# ============================================================
# Notebook driver
# ============================================================


def run_knn_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    global NETWORK_SEARCH_ROOT, OUTPUT_DIR

    NETWORK_SEARCH_ROOT = Path(NETWORK_SEARCH_ROOT).expanduser().resolve()
    OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()

    figure_format = FIGURE_FORMAT.lower().lstrip(".")
    if figure_format not in {"pdf", "png", "svg"}:
        raise ValueError("FIGURE_FORMAT must be 'pdf', 'png', or 'svg'.")

    if DISPLAY_ANALYTICAL_DESCRIPTION:
        description = analytical_knn_description_markdown()
        if Markdown is not None:
            display(Markdown(description))
        else:
            print(description)

    (
        network_inventory,
        empirical_knn_counts,
        analytical_knn_expected,
        shell_statistics,
    ) = collect_selected_knn_networks()

    aggregated_knn = aggregate_knn_results(
        network_inventory,
        empirical_knn_counts,
        analytical_knn_expected,
    )

    summary_columns = ["q", "t"] if SPLIT_BY_TIME else ["q"]
    summary = (
        network_inventory
        .groupby(summary_columns, as_index=False)
        .agg(
            n_networks=("network", "nunique"),
            mean_lcc_nodes=("lcc_nodes", "mean"),
            mean_assortativity=("degree_assortativity", "mean"),
            std_assortativity=("degree_assortativity", "std"),
            mean_uncorrelated_knn=("uncorrelated_knn_baseline", "mean"),
            mean_root_degree=("root_degree", "mean"),
            mean_root_knn_prediction=("root_knn_prediction", "mean"),
        )
        .sort_values(summary_columns)
        .reset_index(drop=True)
    )

    print("\nK_nn and assortativity summary:")
    try:
        display(summary)
    except Exception:
        print(summary.to_string(index=False))

    saved_paths = plot_knn_panels(aggregated_knn)

    print("\nAnalysis completed.")
    print(f"Neighbor occupancy source: {KNN_NEIGHBOR_OCCUPANCY_SOURCE}")
    print(f"Independent networks analyzed: {network_inventory['network'].nunique()}")
    if SAVE_FIGS:
        print(f"Panel figures exported: {len(saved_paths)}")
    else:
        print("SAVE_FIGS = False: no files were exported.")

    return (
        network_inventory,
        empirical_knn_counts,
        analytical_knn_expected,
        shell_statistics,
        aggregated_knn,
    )


# ============================================================
# Execute in Jupyter
# ============================================================
# Edit the configuration block above and run this cell.

(
    network_inventory,
    empirical_knn_counts,
    analytical_knn_expected,
    shell_statistics,
    aggregated_knn,
) = run_knn_analysis()