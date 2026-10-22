#!/usr/bin/env python3
"""
fig_topology_degree.py -- Fig. "model-fq_LCC_degree_panel_shell_theory_joint_..." (Chapter 4, Sec. 4.2.3)

Degree distribution P(k) of the largest connected component (LCC) of the single-peak genotype
networks (N=1e5, L=250, F=3, q=0, lambda=1, t=100; 10 networks pooled), (a) complete and
(b) split by Hamming distance m from the selected genotype (m=0, m=1, m>=2), with the
shell-conditioned construction of the chapter:

  * m=0 and m>=2: K_m = Bin(m, p_{m-1}) + Bin(L-m, p_{m+1}),  p_m = 1 - exp[-N x_m / C(L,m)];
  * m=1: genotype-resolved Poisson-binomial; the incidence p_ij of every shell-2 genotype {i,j} is
    1 - exp(-c h_ij), where h_ij is the finite-pool hazard computed from the t-1 population and c is
    fixed by the observed number S_2 of shell-2 LCC nodes;
  * P_th(k) = sum_s sum_m S_{s,m} P_s(k|m) / sum_s S_LCC,s.

Inputs (per iteration): <run>/Networks/<stem>_t_100.graphml (common/build_networks.py),
<run>/Abundances/<stem>_t_100.txt and <stem>_t_99.txt. For q > 0 the viability of shell-2
genotypes is recomputed from the base seed in <run>/run_metadata.json.
Port of the thesis notebook 16-Simulations/old/plots-model-fq-topology.ipynb (cell 0).

    python fig_topology_degree.py --data Data/single_peak_topology --out Figures
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

LOG_X = True
LOG_Y = True

SAVE_FIGS = True
SHOW_FIGS = False
FIGURE_FORMAT = "pdf"               # "pdf", "png", or "svg"
FIGURE_DPI = 600

FIGSIZE = (12.2, 4.9)
FLOAT_TOLERANCE = 1e-12

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


def read_population_shell_counts(
    abundance_file: Path,
    *,
    L: int,
    expected_population_size: int | None,
) -> tuple[np.ndarray, int]:
    """Read only the population counts in every Hamming shell."""
    _, shell_counts, observed_population = read_population_genotype_counts(
        abundance_file,
        L=L,
        expected_population_size=expected_population_size,
    )
    return shell_counts, observed_population


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


def analytical_degree_distribution_given_shell(
    *,
    shell: int,
    L: int,
    shell_presence_probabilities: np.ndarray,
) -> np.ndarray:
    r"""
    Calculate P(k|m) from

        K_m = X_m^- + X_m^+,
        X_m^- ~ Binomial(m, p_{m-1}),
        X_m^+ ~ Binomial(L-m, p_{m+1}).

    The PMF is the convolution of the two binomial distributions.
    """
    m = int(shell)
    if not 0 <= m <= L:
        raise ValueError(f"Shell m={m} lies outside [0,{L}].")

    p_minus = (
        float(shell_presence_probabilities[m - 1]) if m > 0 else 0.0
    )
    p_plus = (
        float(shell_presence_probabilities[m + 1]) if m < L else 0.0
    )

    lower_neighbors = binomial_pmf(m, p_minus)
    upper_neighbors = binomial_pmf(L - m, p_plus)
    degree_pmf = np.convolve(lower_neighbors, upper_neighbors)

    # The support is always k=0,...,L.
    if len(degree_pmf) != L + 1:
        raise RuntimeError(
            f"Internal theory error: expected PMF length {L + 1}, "
            f"received {len(degree_pmf)}."
        )

    degree_pmf = np.clip(degree_pmf, 0.0, None)
    total = degree_pmf.sum()
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError(f"Invalid analytical P(k|m={m}).")
    return degree_pmf / total


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


def dynamic_m1_degree_distribution(
    *,
    observed_m1_loci: list[int],
    observed_shell2_nodes: int,
    root_present_in_lcc: bool,
    previous_genotype_counts: Counter[tuple[int, ...]],
    previous_population_size: int,
    metadata: dict,
    run_metadata: dict,
) -> tuple[np.ndarray, dict]:
    r"""Conditioned finite-pool prediction for P(k|m=1).

    For every shell-2 target g_ij={i,j}, the t-1 population determines a
    finite-pool hazard h_ij. A single scalar c is chosen so

        sum_{i<j} p_ij = S_2,
        p_ij = a_ij [1-exp(-c h_ij)],

    where S_2 is the observed number of shell-2 LCC nodes and a_ij encodes
    target viability. For an observed first-shell genotype i,

        K_i = I_root + sum_{j != i} Bernoulli(p_ij),

    giving a genotype-specific Poisson-binomial distribution.
    """
    L = int(metadata["L"])
    if not observed_m1_loci:
        raise ValueError("No observed m=1 LCC nodes were supplied.")
    if observed_shell2_nodes < 0:
        raise ValueError("Observed shell-2 LCC count cannot be negative.")
    if DYNAMIC_M1_MAX_DISTANCE is not None and DYNAMIC_M1_MAX_DISTANCE < 2:
        raise ValueError(
            "DYNAMIC_M1_MAX_DISTANCE must be at least 2 for the conditioned "
            "m=1 approximation, or None."
        )

    unique_loci = sorted(set(int(locus) for locus in observed_m1_loci))
    if len(unique_loci) != len(observed_m1_loci):
        raise ValueError("Repeated m=1 loci were found among distinct LCC nodes.")

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
    exact_landscape_available = q <= 0.0 or run_metadata.get("base_seed") is not None

    for pair_index, pair in enumerate(all_pairs):
        viability = deterministic_target_viability(
            pair,
            metadata=metadata,
            run_metadata=run_metadata,
        )
        if viability is False:
            presence_factors[pair_index] = 0.0
            hazards[pair_index] = 0.0
            continue
        if viability is None:
            # Landscape-averaged fallback: a target is viable with probability
            # 1-q, but conditional on viability its dynamic hazard is unchanged.
            presence_factors[pair_index] = 1.0 - q

        hazards[pair_index] = target_pair_finite_pool_hazard(
            pair,
            aggregates=aggregates,
            mutation_probabilities=mutation_probabilities,
            pool_slot_selection_probability=alpha,
        )

    if DYNAMIC_M1_CONDITION_ON_OBSERVED_S2:
        scale, pair_probabilities, calibration = (
            calibrate_hazard_scale_to_shell2_count(
                hazards=hazards,
                presence_factors=presence_factors,
                target_shell2_nodes=int(observed_shell2_nodes),
            )
        )
        method = "conditioned_finite_pool_poisson_binomial_tminus1"
    else:
        scale = 1.0
        pair_probabilities = probabilities_from_scaled_hazards(
            hazards,
            presence_factors,
            scale,
        )
        total = float(pair_probabilities.sum())
        calibration = {
            "unconditioned_expected_shell2_nodes": total,
            "conditioned_expected_shell2_nodes": total,
            "maximum_attainable_shell2_nodes": float(
                presence_factors[hazards > 0.0].sum()
            ),
            "calibration_iterations": 0,
            "calibration_residual": total - float(observed_shell2_nodes),
        }
        method = "finite_pool_poisson_binomial_tminus1"

    pair_probability = {
        pair: float(probability)
        for pair, probability in zip(all_pairs, pair_probabilities)
    }

    lower_degree = 1 if root_present_in_lcc else 0
    mixture = np.zeros(L + 1, dtype=float)
    all_incident_probabilities: list[float] = []
    node_predicted_means: list[float] = []

    for locus in unique_loci:
        upper_probabilities = np.asarray(
            [
                pair_probability[
                    (locus, other) if locus < other else (other, locus)
                ]
                for other in range(L)
                if other != locus
            ],
            dtype=float,
        )
        all_incident_probabilities.extend(upper_probabilities.tolist())
        upper_pmf = poisson_binomial_pmf(upper_probabilities)

        node_pmf = np.zeros(L + 1, dtype=float)
        stop = lower_degree + len(upper_pmf)
        node_pmf[lower_degree:stop] = upper_pmf
        mixture += node_pmf
        node_predicted_means.append(
            float(lower_degree + upper_probabilities.sum())
        )

    mixture /= len(unique_loci)
    mixture = np.clip(mixture, 0.0, None)
    mixture /= mixture.sum()

    incident_array = np.asarray(all_incident_probabilities, dtype=float)
    all_first_shell_present = len(unique_loci) == L
    incidence_identity_mean = (
        float(lower_degree + 2.0 * observed_shell2_nodes / len(unique_loci))
        if all_first_shell_present
        else math.nan
    )

    diagnostics = {
        "method": method,
        "previous_population_size": int(previous_population_size),
        "expected_viable_pool_size": float(expected_pool),
        "finite_pool_slot_selection_probability": float(alpha),
        "hazard_calibration_scale": float(scale),
        "observed_lcc_shell2_nodes": int(observed_shell2_nodes),
        **calibration,
        "root_neighbor_degree": int(lower_degree),
        "n_m1_nodes": int(len(unique_loci)),
        "all_first_shell_genotypes_present": bool(all_first_shell_present),
        "n_unique_shell2_targets": int(len(all_pairs)),
        "mean_shell2_incidence_probability": (
            float(incident_array.mean()) if incident_array.size else 0.0
        ),
        "mean_predicted_m1_degree": float(np.mean(node_predicted_means)),
        "incidence_identity_mean_degree": incidence_identity_mean,
        "max_mutation_distance": DYNAMIC_M1_MAX_DISTANCE,
        "exact_target_landscape_used": bool(
            DYNAMIC_M1_USE_EXACT_TARGET_VIABILITY and exact_landscape_available
        ),
    }
    return mixture, diagnostics

def analytical_description_markdown() -> str:
    """Markdown explanation shown once in the notebook."""
    mode_text = (
        r"Panel (b) uses the joint normalization $P(k,c)$, so the integrated "
        r"mass of each curve is the LCC fraction in that mutation class."
        if SHELL_PROBABILITY_MODE.lower().strip() == "joint"
        else
        r"Panel (b) uses the conditional normalization $P(k\mid c)$, so each "
        r"mutation-class curve integrates to one."
    )

    conditioned_text = (
        r"""
For the first shell, the network at time $t$ is predicted from the abundance
snapshot at $t-1$. Let $M_{\rm pool}$ denote the predicted viable offspring-pool
size. One position in that pool is sampled at least once with probability

$$
\alpha_M=1-\left(1-\frac{1}{M_{\rm pool}}\right)^N
\simeq 1-e^{-N/M_{\rm pool}}.
$$

For a shell-2 target $g_{ij}=\{i,j\}$, each potential offspring position $a$
reaches the target with probability $Q_{d_a}$. Its finite-pool occupancy hazard
is therefore

$$
h_{ij}=-\sum_a \log\!\left(1-\alpha_M Q_{d_a}\right).
$$

The relative target heterogeneity comes entirely from these dynamic hazards.
A single scalar $c$ is then chosen to impose the observed number $S_2$ of LCC
nodes in shell 2,

$$
\sum_{i<j}p_{ij}=S_2,
\qquad
p_{ij}=1-e^{-c h_{ij}},
$$

with the deterministic viability factor included when $q>0$. For an observed
first-shell genotype $i$,

$$
K_i=I_0+\sum_{j\ne i}B_{ij},
\qquad B_{ij}\sim\operatorname{Bernoulli}(p_{ij}),
$$

so $P(k\mid m=1)$ is an equal-node mixture of genotype-specific
Poisson-binomial distributions. When the root and all $L$ first-shell nodes
are present, this construction enforces the exact incidence identity

$$
\langle K\rangle_{m=1}=1+\frac{2S_2}{S_1}.
$$
"""
        if USE_DYNAMIC_M1_APPROXIMATION
        else ""
    )

    return rf"""
### Conditioned finite-pool shell approximation

For exact shell $m$, the baseline shell theory uses

$$
p_m=1-\exp\!\left[-\frac{{N x_m}}{{\binom{{L}}{{m}}}}\right],
$$

and

$$
K_m=X_m^-+X_m^+,
\qquad
X_m^-\sim\operatorname{{Binomial}}(m,p_{{m-1}}),
\qquad
X_m^+\sim\operatorname{{Binomial}}(L-m,p_{{m+1}}).
$$

The complete LCC degree distribution is mixed with the measured exact-shell
node counts $S_m$,

$$
P(k)=\sum_{{m=0}}^L \frac{{S_m}}{{\sum_r S_r}}P(k\mid m).
$$

{conditioned_text}

The $m\geq2$ prediction remains the weighted mixture of all exact shells,
not one effective shell. {mode_text} Every independent simulation is treated
separately before expected node counts are pooled.
"""


# ============================================================
# One-network measurement and theory
# ============================================================


def read_network_measurements(
    network_file: Path,
    metadata: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Read one GraphML, its abundance snapshot, and the t-1 snapshot."""
    graph = nx.read_graphml(network_file)
    graph = to_simple_undirected_graph(graph)

    if graph.number_of_nodes() == 0:
        raise ValueError("empty graph")

    lcc_nodes, n_components = largest_component_nodes(graph)
    lcc = graph.subgraph(lcc_nodes).copy()

    L = int(metadata["L"])
    abundance_file = resolve_abundance_file(network_file, metadata)
    population_shell_counts, observed_population = read_population_shell_counts(
        abundance_file,
        L=L,
        expected_population_size=int(metadata["N"]),
    )
    presence_probabilities = shell_representation_probabilities(
        population_shell_counts,
        L=L,
    )

    overall_degree_counts: Counter[int] = Counter()
    class_degree_counts: dict[str, Counter[int]] = {
        category: Counter() for category in MUTATION_CLASS_ORDER
    }
    lcc_shell_counts = np.zeros(L + 1, dtype=np.int64)
    degree_values: list[int] = []
    observed_m1_loci: list[int] = []
    root_present_in_lcc = False

    for node, degree in lcc.degree():
        name = genotype_name(node, lcc.nodes[node])
        genotype = genotype_tuple_from_name(name, L=L, validate=True)
        mutations = len(genotype)
        category = mutation_class(mutations)
        degree = int(degree)

        overall_degree_counts[degree] += 1
        class_degree_counts[category][degree] += 1
        lcc_shell_counts[mutations] += 1
        degree_values.append(degree)

        if mutations == 0:
            root_present_in_lcc = True
        elif mutations == 1:
            observed_m1_loci.append(genotype[0])

    if not degree_values:
        raise ValueError("empty largest connected component")

    empirical_rows: list[dict] = []
    for degree, count in sorted(overall_degree_counts.items()):
        empirical_rows.append({
            **metadata,
            "scope": "all",
            "degree": int(degree),
            "node_count": int(count),
        })

    for category in MUTATION_CLASS_ORDER:
        for degree, count in sorted(class_degree_counts[category].items()):
            empirical_rows.append({
                **metadata,
                "scope": category,
                "degree": int(degree),
                "node_count": int(count),
            })

    dynamic_m1_pmf: np.ndarray | None = None
    dynamic_diagnostics: dict = {
        "method": "homogeneous_shell_binomial",
        "previous_abundance_file": None,
        "fallback_reason": None,
    }

    if USE_DYNAMIC_M1_APPROXIMATION and observed_m1_loci:
        previous_time = int(metadata["t"]) - int(DYNAMIC_M1_PREVIOUS_TIME_OFFSET)
        try:
            if previous_time < 0:
                raise ValueError(
                    f"Previous prediction time would be t={previous_time}."
                )

            previous_abundance_file = resolve_abundance_file_at_time(
                network_file,
                metadata,
                target_time=previous_time,
            )
            (
                previous_genotype_counts,
                _,
                previous_population_size,
            ) = read_population_genotype_counts(
                previous_abundance_file,
                L=L,
                expected_population_size=int(metadata["N"]),
            )
            run_metadata = load_run_metadata(metadata)
            dynamic_m1_pmf, computed_diagnostics = dynamic_m1_degree_distribution(
                observed_m1_loci=observed_m1_loci,
                observed_shell2_nodes=int(lcc_shell_counts[2]),
                root_present_in_lcc=root_present_in_lcc,
                previous_genotype_counts=previous_genotype_counts,
                previous_population_size=previous_population_size,
                metadata=metadata,
                run_metadata=run_metadata,
            )
            dynamic_diagnostics = {
                **computed_diagnostics,
                "previous_abundance_file": str(previous_abundance_file),
                "fallback_reason": None,
            }
        except Exception as error:
            if REQUIRE_DYNAMIC_M1_INPUT:
                raise RuntimeError(
                    "The dynamic m=1 approximation could not be calculated for "
                    f"{network_file.name}: {type(error).__name__}: {error}"
                ) from error
            dynamic_diagnostics = {
                "method": "homogeneous_shell_binomial_fallback",
                "previous_abundance_file": None,
                "fallback_reason": f"{type(error).__name__}: {error}",
            }

    expected_overall = np.zeros(L + 1, dtype=float)
    expected_by_class = {
        category: np.zeros(L + 1, dtype=float)
        for category in MUTATION_CLASS_ORDER
    }

    shell_rows: list[dict] = []
    lcc_size = int(lcc.number_of_nodes())
    log_shell_sizes = shell_log_sizes(L)

    for m in range(L + 1):
        population_count = int(population_shell_counts[m])
        shell_lcc_nodes = int(lcc_shell_counts[m])
        theory_method = (
            dynamic_diagnostics["method"]
            if m == 1 and dynamic_m1_pmf is not None
            else "homogeneous_shell_binomial"
        )

        shell_rows.append({
            **metadata,
            "abundance_file": str(abundance_file),
            "previous_abundance_file": dynamic_diagnostics.get(
                "previous_abundance_file"
            ),
            "shell": m,
            "population_count": population_count,
            "population_fraction": float(population_count / observed_population),
            "log_shell_size": float(log_shell_sizes[m]),
            "presence_probability": float(presence_probabilities[m]),
            "lcc_shell_nodes": shell_lcc_nodes,
            "lcc_shell_weight": float(shell_lcc_nodes / lcc_size),
            "theory_method": theory_method,
            "dynamic_expected_viable_pool_size": dynamic_diagnostics.get(
                "expected_viable_pool_size"
            ),
            "dynamic_finite_pool_alpha": dynamic_diagnostics.get(
                "finite_pool_slot_selection_probability"
            ),
            "dynamic_hazard_scale": dynamic_diagnostics.get(
                "hazard_calibration_scale"
            ),
            "dynamic_unconditioned_expected_S2": dynamic_diagnostics.get(
                "unconditioned_expected_shell2_nodes"
            ),
            "dynamic_conditioned_expected_S2": dynamic_diagnostics.get(
                "conditioned_expected_shell2_nodes"
            ),
            "dynamic_mean_shell2_incidence_probability": dynamic_diagnostics.get(
                "mean_shell2_incidence_probability"
            ),
            "dynamic_mean_predicted_m1_degree": dynamic_diagnostics.get(
                "mean_predicted_m1_degree"
            ),
            "dynamic_incidence_identity_mean_degree": dynamic_diagnostics.get(
                "incidence_identity_mean_degree"
            ),
            "dynamic_fallback_reason": dynamic_diagnostics.get("fallback_reason"),
        })

        if shell_lcc_nodes == 0:
            continue

        if m == 1 and dynamic_m1_pmf is not None:
            shell_pmf = dynamic_m1_pmf
        else:
            shell_pmf = analytical_degree_distribution_given_shell(
                shell=m,
                L=L,
                shell_presence_probabilities=presence_probabilities,
            )

        expected_counts = shell_lcc_nodes * shell_pmf
        category = mutation_class(m)
        expected_overall += expected_counts
        expected_by_class[category] += expected_counts

    theory_rows: list[dict] = []
    for degree, expected_count in enumerate(expected_overall):
        theory_rows.append({
            **metadata,
            "scope": "all",
            "degree": degree,
            "expected_count": float(expected_count),
        })

    for category in MUTATION_CLASS_ORDER:
        for degree, expected_count in enumerate(expected_by_class[category]):
            theory_rows.append({
                **metadata,
                "scope": category,
                "degree": degree,
                "expected_count": float(expected_count),
            })

    degree_array = np.asarray(degree_values, dtype=np.int64)
    class_totals = {
        category: int(sum(class_degree_counts[category].values()))
        for category in MUTATION_CLASS_ORDER
    }

    inventory_row = {
        **metadata,
        "abundance_file": str(abundance_file),
        "previous_abundance_file": dynamic_diagnostics.get(
            "previous_abundance_file"
        ),
        "observed_population": int(observed_population),
        "full_nodes": int(graph.number_of_nodes()),
        "full_edges": int(graph.number_of_edges()),
        "n_components": int(n_components),
        "lcc_nodes": lcc_size,
        "lcc_edges": int(lcc.number_of_edges()),
        "lcc_node_fraction": float(lcc_size / graph.number_of_nodes()),
        "mean_lcc_degree": float(degree_array.mean()),
        "variance_lcc_degree": float(degree_array.var(ddof=0)),
        "maximum_lcc_degree": int(degree_array.max()),
        "nodes_m0": class_totals["m=0"],
        "nodes_m1": class_totals["m=1"],
        "nodes_mge2": class_totals["m>=2"],
        "fraction_m0": float(class_totals["m=0"] / lcc_size),
        "fraction_m1": float(class_totals["m=1"] / lcc_size),
        "fraction_mge2": float(class_totals["m>=2"] / lcc_size),
        "m1_theory_method": dynamic_diagnostics.get("method"),
        "m1_theory_expected_pool_size": dynamic_diagnostics.get(
            "expected_viable_pool_size"
        ),
        "m1_theory_finite_pool_alpha": dynamic_diagnostics.get(
            "finite_pool_slot_selection_probability"
        ),
        "m1_theory_hazard_scale": dynamic_diagnostics.get(
            "hazard_calibration_scale"
        ),
        "m1_theory_observed_S2": dynamic_diagnostics.get(
            "observed_lcc_shell2_nodes"
        ),
        "m1_theory_unconditioned_expected_S2": dynamic_diagnostics.get(
            "unconditioned_expected_shell2_nodes"
        ),
        "m1_theory_conditioned_expected_S2": dynamic_diagnostics.get(
            "conditioned_expected_shell2_nodes"
        ),
        "m1_theory_mean_shell2_incidence_probability": dynamic_diagnostics.get(
            "mean_shell2_incidence_probability"
        ),
        "m1_theory_mean_predicted_degree": dynamic_diagnostics.get(
            "mean_predicted_m1_degree"
        ),
        "m1_theory_incidence_identity_mean_degree": dynamic_diagnostics.get(
            "incidence_identity_mean_degree"
        ),
        "m1_theory_calibration_residual": dynamic_diagnostics.get(
            "calibration_residual"
        ),
        "m1_theory_fallback_reason": dynamic_diagnostics.get("fallback_reason"),
    }

    return inventory_row, empirical_rows, theory_rows, shell_rows


# ============================================================
# File discovery and collection
# ============================================================


def collect_selected_networks() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Read selected GraphML files and the matching abundance snapshots."""
    root = Path(NETWORK_SEARCH_ROOT).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Network search root not found: {root}")

    graphml_files = sorted(root.rglob("dataOut_*.graphml"))
    if not graphml_files:
        raise RuntimeError(
            "No model-fq GraphML files were found.\n"
            f"Search root: {root}\n"
            "Expected files below runs/<run_id>/Networks/."
        )

    metadata_catalog: list[dict] = []
    selected_files: list[tuple[Path, dict]] = []

    for network_file in graphml_files:
        metadata = parse_network_metadata(network_file)
        if metadata is None:
            continue
        metadata_catalog.append(metadata)
        if metadata_is_selected(metadata):
            selected_files.append((network_file, metadata))

    print(f"Discovered GraphML files: {len(graphml_files)}")
    print(f"Files with parseable model-fq metadata: {len(metadata_catalog)}")
    print(f"Selection: {selector_description()}")
    print(f"Selected independent networks: {len(selected_files)}")

    if not selected_files:
        if metadata_catalog:
            available = (
                pd.DataFrame(metadata_catalog)[
                    ["N", "L", "fpeak", "q", "t", "iteration", "mu", "fbg", "run_id"]
                ]
                .drop_duplicates()
                .sort_values(["N", "L", "fpeak", "q", "t", "run_id", "iteration"])
                .reset_index(drop=True)
            )
            raise RuntimeError(
                "No GraphML files matched the configured filters.\n\n"
                "Available combinations (first 40):\n"
                + available.head(40).to_string(index=False)
            )
        raise RuntimeError("No parseable model-fq GraphML filenames were found.")

    inventory_rows: list[dict] = []
    empirical_rows: list[dict] = []
    theory_rows: list[dict] = []
    shell_rows: list[dict] = []
    failures: list[tuple[Path, str]] = []

    for index, (network_file, metadata) in enumerate(selected_files, start=1):
        try:
            inventory, empirical, theory, shells = read_network_measurements(
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
            f"[{index:>4}/{len(selected_files)}] "
            f"q={metadata['q']:g}, t={metadata['t']}, "
            f"run={metadata['run_id']}, iteration={metadata['iteration']}: "
            f"LCC={inventory['lcc_nodes']:,}/{inventory['full_nodes']:,}; "
            f"m=0: {inventory['nodes_m0']}, "
            f"m=1: {inventory['nodes_m1']}, "
            f"m>=2: {inventory['nodes_mge2']:,}"
        )

    if failures:
        print("\nSkipped GraphML files:")
        for path, reason in failures:
            print(f"  {path}: {reason}")

    if not inventory_rows:
        raise RuntimeError("Every selected GraphML file failed during loading.")

    inventory_df = pd.DataFrame(inventory_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network"]
    ).reset_index(drop=True)

    empirical_df = pd.DataFrame(empirical_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "scope", "degree"]
    ).reset_index(drop=True)

    theory_df = pd.DataFrame(theory_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "scope", "degree"]
    ).reset_index(drop=True)

    shell_df = pd.DataFrame(shell_rows).sort_values(
        ["q", "t", "run_id", "iteration", "network", "shell"]
    ).reset_index(drop=True)

    return inventory_df, empirical_df, theory_df, shell_df


# ============================================================
# Aggregation across independent simulations
# ============================================================


def analysis_group_columns() -> list[str]:
    columns = ["mu", "N", "L", "q", "fpeak", "fbg"]
    if SPLIT_BY_TIME:
        columns.append("t")
    return columns


def aggregate_empirical_and_theory(
    inventory: pd.DataFrame,
    empirical_counts: pd.DataFrame,
    theory_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Pool observed and analytical expected counts across simulations."""
    group_columns = analysis_group_columns()

    empirical = (
        empirical_counts
        .groupby(group_columns + ["scope", "degree"], as_index=False, dropna=False)
        .agg(empirical_count=("node_count", "sum"))
    )

    theory = (
        theory_counts
        .groupby(group_columns + ["scope", "degree"], as_index=False, dropna=False)
        .agg(theory_expected_count=("expected_count", "sum"))
    )

    totals = (
        inventory
        .groupby(group_columns, as_index=False, dropna=False)
        .agg(
            total_lcc_nodes=("lcc_nodes", "sum"),
            n_networks=("network", "nunique"),
            total_nodes_m0=("nodes_m0", "sum"),
            total_nodes_m1=("nodes_m1", "sum"),
            total_nodes_mge2=("nodes_mge2", "sum"),
        )
    )

    merged = empirical.merge(
        theory,
        on=group_columns + ["scope", "degree"],
        how="outer",
    )
    merged = merged.merge(totals, on=group_columns, how="left")
    merged["empirical_count"] = merged["empirical_count"].fillna(0.0)
    merged["theory_expected_count"] = merged["theory_expected_count"].fillna(0.0)

    def denominator_for_row(row: pd.Series) -> float:
        scope = row["scope"]
        if scope == "all" or SHELL_PROBABILITY_MODE.lower().strip() == "joint":
            return float(row["total_lcc_nodes"])
        if scope == "m=0":
            return float(row["total_nodes_m0"])
        if scope == "m=1":
            return float(row["total_nodes_m1"])
        if scope == "m>=2":
            return float(row["total_nodes_mge2"])
        raise ValueError(f"Unknown scope: {scope!r}")

    merged["normalization"] = merged.apply(denominator_for_row, axis=1)
    merged["empirical_probability"] = np.where(
        merged["normalization"] > 0,
        merged["empirical_count"] / merged["normalization"],
        np.nan,
    )
    merged["theory_probability"] = np.where(
        merged["normalization"] > 0,
        merged["theory_expected_count"] / merged["normalization"],
        np.nan,
    )

    scope_order = pd.CategoricalDtype(
        categories=["all", *MUTATION_CLASS_ORDER],
        ordered=True,
    )
    merged["scope"] = merged["scope"].astype(scope_order)

    return (
        merged
        .sort_values(group_columns + ["scope", "degree"])
        .reset_index(drop=True)
    )


# ============================================================
# Panel plotting
# ============================================================


def make_figure_tag(metadata: dict) -> str:
    mode = SHELL_PROBABILITY_MODE.lower().strip()
    tag = (
        f"{MODEL}_LCC_degree_panel_shell_theory"
        f"_{mode}"
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


def valid_curve(data: pd.DataFrame, column: str) -> pd.DataFrame:
    curve = data.loc[
        np.isfinite(data[column]) & (data[column] > 0)
    ].copy()
    if LOG_X:
        curve = curve.loc[curve["degree"] > 0].copy()
    return curve


def configure_axis(ax: plt.Axes) -> None:
    if LOG_X:
        ax.set_xscale("log")
    if LOG_Y:
        ax.set_yscale("log")

    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    if LOG_X or LOG_Y:
        ax.grid(True, which="minor", linewidth=0.45, alpha=0.10)


def plot_degree_distribution_panels(
    aggregated_distributions: pd.DataFrame,
) -> list[Path]:
    """
    One two-panel figure per selected q (and time when SPLIT_BY_TIME=True).

    Panel (a): complete LCC P(k), empirical and shell-mixture theory.
    Panel (b): m=0, m=1, m>=2, empirical and shell-dependent theory.
    """
    group_columns = analysis_group_columns()
    saved_paths: list[Path] = []
    mode = SHELL_PROBABILITY_MODE.lower().strip()

    for group_key, group_data in aggregated_distributions.groupby(
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

        # ----------------------------------------------------
        # Panel (a): complete P(k)
        # ----------------------------------------------------
        all_data = group_data.loc[group_data["scope"] == "all"].copy()
        empirical_all = valid_curve(all_data, "empirical_probability")
        theory_all = valid_curve(all_data, "theory_probability")

        axes[0].plot(
            empirical_all["degree"],
            empirical_all["empirical_probability"],
            linestyle="none",
            marker="o",
            markersize=4.7,
            label="Simulation",
            zorder=4,
        )

        if ADD_ANALYTICAL_THEORY and not theory_all.empty:
            axes[0].plot(
                theory_all["degree"],
                theory_all["theory_probability"],
                linestyle="--",
                linewidth=1.8,
                label="Conditioned shell theory",
                zorder=3,
            )

        configure_axis(axes[0])
        axes[0].set_xlabel(r"Node degree, $k$")
        axes[0].set_ylabel(r"LCC degree probability, $P(k)$")
        axes[0].set_title(r"$\mathbf{a}$  Complete LCC distribution", loc="left")
        axes[0].legend(frameon=False, loc="best")
        axes[0].set_ylim(bottom=10**-6, top=1.1)

        # ----------------------------------------------------
        # Panel (b): grouped shell distributions
        # ----------------------------------------------------
        class_handles: list[Line2D] = []

        for category in MUTATION_CLASS_ORDER:
            class_data = group_data.loc[group_data["scope"] == category].copy()
            empirical = valid_curve(class_data, "empirical_probability")
            theory = valid_curve(class_data, "theory_probability")

            if empirical.empty and theory.empty:
                continue

            empirical_line, = axes[1].plot(
                empirical["degree"],
                empirical["empirical_probability"],
                linestyle="none",
                marker=MUTATION_CLASS_MARKERS[category],
                markersize=4.8,
                zorder=4,
            )
            color = empirical_line.get_color()

            if ADD_ANALYTICAL_THEORY and not theory.empty:
                axes[1].plot(
                    theory["degree"],
                    theory["theory_probability"],
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

        configure_axis(axes[1])
        axes[1].set_xlabel(r"Node degree, $k$")
        if mode == "joint":
            axes[1].set_ylabel(r"Joint LCC probability, $P(k,c)$")
        else:
            axes[1].set_ylabel(r"Conditional LCC probability, $P(k\mid c)$")
        axes[1].set_title(r"$\mathbf{b}$  Mutation-class decomposition", loc="left")

        class_legend = axes[1].legend(
            handles=class_handles,
            frameon=False,
            title="Mutation class",
            loc="best",
        )
        axes[1].add_artist(class_legend)
        axes[1].set_ylim(bottom=10**-6, top=1.1)

        style_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                color="black",
                markersize=4.8,
                label="Simulation",
            )
        ]
        if ADD_ANALYTICAL_THEORY:
            style_handles.append(
                Line2D(
                    [0], [0],
                    linestyle="--",
                    color="black",
                    linewidth=1.55,
                    label="Shell/conditioned theory",
                )
            )
        axes[1].legend(
            handles=style_handles,
            frameon=False,
            loc="lower left",
        )

        n_networks = int(group_data["n_networks"].max())
        total_lcc_nodes = int(group_data["total_lcc_nodes"].max())

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

        fig.suptitle(
            title
            + f"\n{n_networks} independent networks; "
            + f"{total_lcc_nodes:,} pooled LCC nodes",
            y=1.035,
            fontsize=13,
        )
        fig.subplots_adjust(top=0.82, bottom=0.15, left=0.085, right=0.985)

        if SAVE_FIGS:
            output_dir = Path(OUTPUT_DIR).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / (
                f"{make_figure_tag(metadata)}.{FIGURE_FORMAT.lstrip('.')}"
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


def run_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Read GraphML networks and original abundances, isolate every LCC, compute
    empirical degree distributions and the conditioned finite-pool shell theory,
    aggregate independent simulations, and build the two-panel figures.

    Returns
    -------
    network_inventory
        One row per independent network.
    empirical_degree_counts
        Sparse observed node counts by network, scope, and degree.
    analytical_expected_counts
        Dense shell-theory expected counts by network, scope, and degree.
    shell_statistics
        Per-network x_m, p_m, S_m, and LCC shell weights.
    aggregated_distributions
        P(k), P(k,c)/P(k|c), and their analytical predictions after pooling.
    """
    global NETWORK_SEARCH_ROOT, OUTPUT_DIR

    NETWORK_SEARCH_ROOT = Path(NETWORK_SEARCH_ROOT).expanduser().resolve()
    OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()

    figure_format = FIGURE_FORMAT.lower().lstrip(".")
    if figure_format not in {"pdf", "png", "svg"}:
        raise ValueError("FIGURE_FORMAT must be 'pdf', 'png', or 'svg'.")

    mode = SHELL_PROBABILITY_MODE.lower().strip()
    if mode not in {"joint", "conditional"}:
        raise ValueError(
            "SHELL_PROBABILITY_MODE must be either 'joint' or 'conditional'."
        )

    if DYNAMIC_M1_PREVIOUS_TIME_OFFSET < 1:
        raise ValueError("DYNAMIC_M1_PREVIOUS_TIME_OFFSET must be at least 1.")
    if (
        DYNAMIC_M1_MAX_DISTANCE is not None
        and int(DYNAMIC_M1_MAX_DISTANCE) < 0
    ):
        raise ValueError(
            "DYNAMIC_M1_MAX_DISTANCE must be non-negative or None."
        )

    if DISPLAY_ANALYTICAL_DESCRIPTION:
        description = analytical_description_markdown()
        if Markdown is not None:
            display(Markdown(description))
        else:
            print(description)

    (
        network_inventory,
        empirical_degree_counts,
        analytical_expected_counts,
        shell_statistics,
    ) = collect_selected_networks()

    aggregated_distributions = aggregate_empirical_and_theory(
        network_inventory,
        empirical_degree_counts,
        analytical_expected_counts,
    )

    summary_columns = ["q", "t"] if SPLIT_BY_TIME else ["q"]
    inventory_summary = (
        network_inventory
        .groupby(summary_columns, as_index=False)
        .agg(
            n_networks=("network", "nunique"),
            mean_lcc_nodes=("lcc_nodes", "mean"),
            mean_lcc_fraction=("lcc_node_fraction", "mean"),
            mean_fraction_m0=("fraction_m0", "mean"),
            mean_fraction_m1=("fraction_m1", "mean"),
            mean_fraction_mge2=("fraction_mge2", "mean"),
            mean_lcc_degree=("mean_lcc_degree", "mean"),
            maximum_lcc_degree=("maximum_lcc_degree", "max"),
        )
        .sort_values(summary_columns)
        .reset_index(drop=True)
    )

    print("\nLCC summary by selected experiment group:")
    try:
        display(inventory_summary)
    except Exception:
        print(inventory_summary.to_string(index=False))

    saved_paths = plot_degree_distribution_panels(aggregated_distributions)

    print("\nAnalysis completed.")
    print(f"Shell normalization: {SHELL_PROBABILITY_MODE}")
    print(
        "m=1 analytical method: "
        + (
            "conditioned finite-pool t-1 -> t Poisson-binomial"
            if USE_DYNAMIC_M1_APPROXIMATION
            else "homogeneous shell binomial"
        )
    )
    print(f"Independent networks analyzed: {network_inventory['network'].nunique()}")
    print(
        "Selected q values: "
        + ", ".join(f"{q:g}" for q in sorted(network_inventory["q"].unique()))
    )
    if SAVE_FIGS:
        print(f"Panel figures exported: {len(saved_paths)}")
    else:
        print("SAVE_FIGS = False: no files were exported.")

    return (
        network_inventory,
        empirical_degree_counts,
        analytical_expected_counts,
        shell_statistics,
        aggregated_distributions,
    )


# ============================================================
# Execute in Jupyter
# ============================================================
# Edit the configuration block above and run the cell.

(
    network_inventory,
    empirical_degree_counts,
    analytical_expected_counts,
    shell_statistics,
    aggregated_distributions,
) = run_analysis()