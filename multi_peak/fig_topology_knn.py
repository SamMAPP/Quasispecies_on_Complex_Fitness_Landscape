#!/usr/bin/env python3
"""
fig_topology_knn.py -- Fig. "model4a30_pre_peak_LCC_knn_multicentre_panel_..." (Chapter 4, Sec. 4.3.3)

Average nearest-neighbour degree k_nn(k) of the multi-peak LCCs immediately before successive peak
introductions (t = 39, 79, 119, 159, 199; same runs as fig_topology_degree.py), with the multicentre
shell construction (Eq. model4a30-multicentre-knn), the uncorrelated reference <k^2>/<k> and, in
panel (f), the normalized prediction error. Also prints the degree assortativity r at every time.
Inputs as for fig_topology_degree.py.
Port of the thesis notebook 16-Simulations/old/plots-multipeak-topology.ipynb (cell 5).

    python fig_topology_knn.py --data Data/multi_peak_delta5 --out Figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import argparse
from pathlib import Path
_ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
_ap.add_argument("--data", default="Data/multi_peak", help="model directory with Networks/ and Abundances/")
_ap.add_argument("--out", default="Figures")
_ap.add_argument("--N", type=int, default=100_000)
_ap.add_argument("--L", type=int, default=250)
_ap.add_argument("--delta", type=int, default=5, help="Hamming distance between consecutive peaks")
ARGS = _ap.parse_args()

"""
Pre-peak nearest-neighbor degree correlations for the model4a30 multi-peak
fitness landscape.

For every pre-peak snapshot t = e*tau - GENERATIONS_BEFORE_PEAK, the notebook:

1. discovers matching GraphML networks;
2. extracts the largest connected component (LCC);
3. measures k_nn(k) inside every independent LCC;
4. reconstructs one-centre and effective multi-centre shell theories;
5. pools independent networks with equal weight per degree-k LCC node;
6. produces five pre-peak panels plus a sixth theory-error panel.

No tables or intermediate files are exported. A figure is written only when
SAVE_FIGS = True.

Analytical structure
--------------------
For a local shell m in Voronoi domain alpha, the focal degree is decomposed as

    K_{alpha,m} = X^-_{alpha,m} + X^+_{alpha,m} + Z_{alpha,m},

where X^- and X^+ count internal edges to the adjacent Hamming shells and Z
counts cross-domain bridges. Conditional on total degree k,

    k_nn(k | alpha,m)
      = E[X^- kappa^- + X^+ kappa^+ + Z kappa^x | K=k] / k.

The complete prediction mixes local classes with the same expected degree-k
node counts used in the corresponding P(k) theory.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import re
import string
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

try:
    from scipy.special import gammaln
    from scipy.stats import binom
except ImportError as exc:
    raise ImportError(
        "This notebook requires scipy. Install it with `pip install scipy`."
    ) from exc

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None
    display = print


# ============================================================
# Configuration
# ============================================================

MODEL = "model4a30"

MODEL_ROOT = Path(ARGS.data)
NETWORK_ROOT = MODEL_ROOT / "Networks"
ABUNDANCE_ROOT = MODEL_ROOT / "Abundances"

OUTPUT_DIR = Path(ARGS.out)

# ------------------------------------------------------------
# Experiment selection
# ------------------------------------------------------------

TARGET_N = [ARGS.N]
TARGET_L = [ARGS.L]
TARGET_Q = [0.0]

TARGET_ITERATIONS = None
TARGET_MU = None
TARGET_F_INITIAL = [3.0]
TARGET_F_BACKGROUND = [1.0]
TARGET_PEAK_INCREMENT = None
TARGET_TAU = None
TARGET_DELTA = [ARGS.delta]

TARGET_PRE_PEAK_TIMES = None
TARGET_PEAK_EVENTS = None

# t = e*tau - GENERATIONS_BEFORE_PEAK.
GENERATIONS_BEFORE_PEAK = 1

# ------------------------------------------------------------
# Theory configuration
# ------------------------------------------------------------

ADD_ANALYTICAL_THEORY = True

COMPUTE_SINGLE_CENTRE_BASELINE = True
COMPUTE_MULTI_CENTRE_THEORY = True

# Keep the five event panels visually clean. The single-centre theory remains
# visible in the sixth error panel.
SHOW_SINGLE_CENTRE_IN_EVENT_PANELS = False

ANCHOR_MODE = "most_abundant_lcc"  # or "ancestral"

THEORY_OCCUPANCY_SOURCE = "population_poisson"
# Alternatives:
#   "population_poisson": p_m = 1-exp[-n_m/C(L,m)]
#   "lcc_observed":       p_m = S_m/C(L,m)

CONDITION_THEORY_ON_POSITIVE_DEGREE = True
CONDITION_CENTRE_DEGREE = True
CONDITION_FIRST_SHELL_INCIDENCE = True
CONDITION_CROSS_DOMAIN_DEGREES = True

EXPECTED_CENTRES_MODE = "event_index"
INCLUDE_ANCESTRAL_CENTRE = True
EFFECTIVE_CENTRE_MIN_DISTANCE = None

DISPLAY_THEORY_DESCRIPTION = True

SINGLE_CENTRE_THEORY = "single_centre"
MULTI_CENTRE_THEORY = "multi_centre"

THEORY_LABELS = {
    SINGLE_CENTRE_THEORY: "Single-centre shell theory",
    MULTI_CENTRE_THEORY: "Multi-centre shell theory",
}

# ------------------------------------------------------------
# Figure configuration
# ------------------------------------------------------------

LOG_X = True
LOG_Y = True

SHOW_FIGS = False
SAVE_FIGS = True
FIGURE_FORMAT = "pdf"  # "pdf", "png", or "svg"
FIGURE_DPI = 600

MAX_PANEL_COLUMNS = 3
PANEL_WIDTH = 4.25
PANEL_HEIGHT = 3.65
ADD_DIAGNOSTIC_PANEL = True

COMMON_X_MIN = 0.9
COMMON_X_MAX = None  # None -> inferred from L and observed k_max

# Applied only when explicitly set. Otherwise common positive limits are
# inferred from all empirical and theoretical curves in the family.
COMMON_Y_MIN = None
COMMON_Y_MAX = None

SHOW_EVENT_STATISTICS = True
SHOW_UNCORRELATED_REFERENCE = True
SHOW_THEORY_IN_LEGEND = True

# Sixth panel metric:
#   sqrt[sum_k N(k)(Knn_emp-Knn_th)^2 / sum_k N(k)Knn_emp^2]
DIAGNOSTIC_METRIC = "weighted_nrmse"

FLOAT_TOLERANCE = 1e-12

SIMULATION_COLOR = "#1f77b4"
MULTI_CENTRE_COLOR = "#e67e22"
SINGLE_CENTRE_COLOR = "#7f7f7f"
UNCORRELATED_COLOR = "#4c78a8"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 10.5,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
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
    rf"^(?:dataOut_)?"
    rf"mu_(?P<mu>{NUMBER})"
    rf"_nrw_(?P<N>\d+)"
    rf"_L_(?P<L>\d+)"
    rf"_q_(?P<q>{NUMBER})"
    rf"_iteration_(?P<iteration>\d+)"
    rf"_fpeak_(?P<f_initial>{NUMBER})"
    rf"_fbg_(?P<f_background>{NUMBER})"
    rf"_fstep_(?P<peak_increment>{NUMBER})"
    rf"_tau_(?P<tau>\d+)"
    rf"_delta_(?P<delta>\d+)"
    rf"_t_(?P<t>\d+)$"
)

FAMILY_COLUMNS = [
    "mu",
    "N",
    "L",
    "q",
    "f_initial",
    "f_background",
    "peak_increment",
    "tau",
    "delta",
]

EVENT_COLUMNS = [
    *FAMILY_COLUMNS,
    "t",
    "next_peak_time",
    "peak_event_index",
]

NETWORK_COLUMNS = [
    *EVENT_COLUMNS,
    "network",
    "iteration",
]


def as_list(values):
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


def strip_network_suffixes(path: Path) -> str:
    name = path.name

    for suffix in (".gz", ".graphml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    if name in {"network", "graph", "network_full", "full_network"}:
        return path.parent.name

    return name


def parse_snapshot_metadata(network_file: Path) -> dict | None:
    snapshot_stem = strip_network_suffixes(network_file)
    match = NAME_RE.fullmatch(snapshot_stem)

    if match is None:
        return None

    values = match.groupdict()
    tau = int(values["tau"])
    t = int(values["t"])

    is_pre_peak = (
        t + int(GENERATIONS_BEFORE_PEAK) > 0
        and (t + int(GENERATIONS_BEFORE_PEAK)) % tau == 0
    )

    if is_pre_peak:
        next_peak_time = t + int(GENERATIONS_BEFORE_PEAK)
        peak_event_index = next_peak_time // tau
    else:
        next_peak_time = ((t // tau) + 1) * tau
        peak_event_index = next_peak_time // tau

    try:
        relative_network = str(
            network_file.resolve().relative_to(NETWORK_ROOT.resolve())
        )
    except ValueError:
        relative_network = str(network_file.resolve())

    metadata = {
        "network": relative_network,
        "network_file": str(network_file.resolve()),
        "snapshot_stem": snapshot_stem,
        "mu": float(values["mu"]),
        "N": int(values["N"]),
        "L": int(values["L"]),
        "q": float(values["q"]),
        "iteration": int(values["iteration"]),
        "f_initial": float(values["f_initial"]),
        "f_background": float(values["f_background"]),
        "peak_increment": float(values["peak_increment"]),
        "tau": tau,
        "delta": int(values["delta"]),
        "t": t,
        "is_pre_peak": bool(is_pre_peak),
        "next_peak_time": int(next_peak_time),
        "peak_event_index": int(peak_event_index),
    }

    metadata["active_peak_fitness"] = (
        metadata["f_initial"]
        + max(0, metadata["peak_event_index"] - 1)
        * metadata["peak_increment"]
    )

    return metadata


def metadata_is_selected(metadata: dict) -> bool:
    return (
        metadata["is_pre_peak"]
        and matches_integer(metadata["N"], TARGET_N)
        and matches_integer(metadata["L"], TARGET_L)
        and matches_float(metadata["q"], TARGET_Q)
        and matches_integer(metadata["iteration"], TARGET_ITERATIONS)
        and matches_float(metadata["mu"], TARGET_MU)
        and matches_float(metadata["f_initial"], TARGET_F_INITIAL)
        and matches_float(metadata["f_background"], TARGET_F_BACKGROUND)
        and matches_float(metadata["peak_increment"], TARGET_PEAK_INCREMENT)
        and matches_integer(metadata["tau"], TARGET_TAU)
        and matches_integer(metadata["delta"], TARGET_DELTA)
        and matches_integer(metadata["t"], TARGET_PRE_PEAK_TIMES)
        and matches_integer(metadata["peak_event_index"], TARGET_PEAK_EVENTS)
    )


def selector_description() -> str:
    def show(values):
        normalized = as_list(values)
        return "all" if normalized is None else str(normalized)

    return (
        f"N={show(TARGET_N)}, "
        f"L={show(TARGET_L)}, "
        f"q={show(TARGET_Q)}, "
        f"iterations={show(TARGET_ITERATIONS)}, "
        f"mu={show(TARGET_MU)}, "
        f"f_initial={show(TARGET_F_INITIAL)}, "
        f"f_background={show(TARGET_F_BACKGROUND)}, "
        f"peak_increment={show(TARGET_PEAK_INCREMENT)}, "
        f"tau={show(TARGET_TAU)}, "
        f"delta={show(TARGET_DELTA)}, "
        f"pre_peak_times={show(TARGET_PRE_PEAK_TIMES)}, "
        f"events={show(TARGET_PEAK_EVENTS)}"
    )


def network_priority(path: Path) -> tuple[int, str]:
    lower = path.name.lower()

    if "gc.graphml" in lower or "lcc.graphml" in lower:
        return (99, str(path))

    stem = strip_network_suffixes(path)
    if NAME_RE.fullmatch(stem):
        return (0, str(path))

    if lower in {"network.graphml", "network.graphml.gz"}:
        return (1, str(path))

    return (5, str(path))


def discover_networks() -> pd.DataFrame:
    root = Path(NETWORK_ROOT).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Network root not found: {root}")

    candidates = sorted(
        {
            *root.rglob("*.graphml"),
            *root.rglob("*.graphml.gz"),
        },
        key=lambda path: str(path),
    )

    rows: list[dict] = []

    for path in candidates:
        priority, _ = network_priority(path)
        if priority >= 99:
            continue

        metadata = parse_snapshot_metadata(path)
        if metadata is None:
            continue

        metadata["_priority"] = priority
        rows.append(metadata)

    if not rows:
        return pd.DataFrame()

    inventory = pd.DataFrame(rows)

    dedupe_columns = [
        "snapshot_stem",
        "mu",
        "N",
        "L",
        "q",
        "iteration",
        "f_initial",
        "f_background",
        "peak_increment",
        "tau",
        "delta",
        "t",
    ]

    return (
        inventory
        .sort_values(dedupe_columns + ["_priority", "network_file"])
        .drop_duplicates(dedupe_columns, keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


# ============================================================
# Genotypes and abundances
# ============================================================


def parse_genotype_name(name: str) -> tuple[int, ...]:
    text = str(name).strip()

    if text in {"", "--", "()", "[]"}:
        return ()

    if (
        (text.startswith("(") and text.endswith(")"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        text = text[1:-1].strip()

    if not text:
        return ()

    tokens = [token.strip() for token in text.split(",") if token.strip()]

    try:
        loci = tuple(sorted(int(token) for token in tokens))
    except ValueError as exc:
        raise ValueError(f"Cannot parse genotype name {name!r}.") from exc

    if any(locus < 0 for locus in loci):
        raise ValueError(f"Negative locus in genotype name {name!r}.")
    if len(set(loci)) != len(loci):
        raise ValueError(f"Repeated locus in genotype name {name!r}.")

    return loci


def genotype_name(node, attributes: dict) -> str:
    for key in ("genotype", "label", "name"):
        value = attributes.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return str(node).strip()


def genotype_distance(
    genotype_a: tuple[int, ...],
    genotype_b: tuple[int, ...],
) -> int:
    return len(set(genotype_a).symmetric_difference(genotype_b))


def format_genotype(genotype: tuple[int, ...]) -> str:
    return "--" if not genotype else ",".join(map(str, genotype))


def resolve_abundance_file(metadata: dict) -> Path | None:
    stem = metadata["snapshot_stem"]
    network_file = Path(metadata["network_file"])

    candidates = [
        Path(ABUNDANCE_ROOT) / f"{stem}.txt",
        network_file.parent.parent / "Abundances" / f"{stem}.txt",
        network_file.parent / "Abundances" / f"{stem}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.expanduser().resolve()

    matches = [
        path
        for path in Path(MODEL_ROOT).expanduser().resolve().rglob(f"{stem}.txt")
        if path.parent.name == "Abundances"
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            f"Several abundance snapshots match {stem!r}:\n"
            + "\n".join(f"  {path}" for path in matches)
        )

    return None


def read_abundance_counter(path: Path) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue

            try:
                genotype = parse_genotype_name(text)
            except ValueError as exc:
                raise ValueError(
                    f"{exc} File: {path}, line {line_number}."
                ) from exc

            counts[genotype] += 1

    if not counts:
        raise ValueError(f"No genotypes found in abundance file {path}.")

    return counts


def choose_anchor_genotype(
    *,
    abundance_counts: Counter[tuple[int, ...]],
    lcc_genotypes: set[tuple[int, ...]],
) -> tuple[int, ...]:
    mode = ANCHOR_MODE.lower().strip()

    if mode == "ancestral":
        return ()

    if mode != "most_abundant_lcc":
        raise ValueError(
            "ANCHOR_MODE must be 'ancestral' or 'most_abundant_lcc'."
        )

    if not lcc_genotypes:
        raise ValueError("Cannot choose an anchor from an empty LCC.")

    return max(
        lcc_genotypes,
        key=lambda genotype: (
            int(abundance_counts.get(genotype, 0)),
            -len(genotype),
            tuple(-value for value in genotype),
        ),
    )


# ============================================================
# Graph and centre reconstruction
# ============================================================


def to_simple_undirected_graph(graph: nx.Graph) -> nx.Graph:
    if graph.is_directed():
        graph = graph.to_undirected()
    if graph.is_multigraph():
        graph = nx.Graph(graph)
    return graph


def largest_connected_component(graph: nx.Graph) -> tuple[nx.Graph, int]:
    if graph.number_of_nodes() == 0:
        raise ValueError("The network is empty.")

    components = list(nx.connected_components(graph))
    lcc_nodes = max(components, key=len)
    return graph.subgraph(lcc_nodes).copy(), len(components)


def expected_number_of_centres(metadata: dict) -> int:
    mode = EXPECTED_CENTRES_MODE.lower().strip()

    if mode == "event_index":
        return max(1, int(metadata["peak_event_index"]))

    raise ValueError(
        "EXPECTED_CENTRES_MODE currently supports only 'event_index'."
    )


def effective_centre_minimum_distance(metadata: dict) -> int:
    if EFFECTIVE_CENTRE_MIN_DISTANCE is not None:
        return max(1, int(EFFECTIVE_CENTRE_MIN_DISTANCE))

    return max(2, int(metadata["delta"]) // 2)


def select_effective_centres(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
) -> list[tuple[int, ...]]:
    genotype_to_node = {
        genotype: node
        for node, genotype in node_genotype.items()
    }
    genotypes = list(genotype_to_node)

    if not genotypes:
        raise ValueError("Cannot infer centres from an empty LCC.")

    requested = min(
        expected_number_of_centres(metadata),
        len(genotypes),
    )
    minimum_distance = effective_centre_minimum_distance(metadata)

    def abundance(genotype):
        return int(abundance_counts.get(genotype, 0))

    def degree(genotype):
        return int(lcc.degree(genotype_to_node[genotype]))

    local_maxima: list[tuple[int, ...]] = []

    for genotype, node in genotype_to_node.items():
        value = abundance(genotype)
        neighbor_values = [
            abundance(node_genotype[neighbor])
            for neighbor in lcc.neighbors(node)
        ]

        if not neighbor_values or value >= max(neighbor_values):
            local_maxima.append(genotype)

    ranked_all = sorted(
        genotypes,
        key=lambda genotype: (
            abundance(genotype),
            degree(genotype),
            -len(genotype),
            tuple(-locus for locus in genotype),
        ),
        reverse=True,
    )

    ranked_candidates = sorted(
        set(local_maxima),
        key=lambda genotype: (
            abundance(genotype),
            degree(genotype),
            -len(genotype),
            tuple(-locus for locus in genotype),
        ),
        reverse=True,
    )

    selected = [ranked_all[0]]

    if (
        INCLUDE_ANCESTRAL_CENTRE
        and requested > 1
        and () in genotype_to_node
        and () not in selected
    ):
        selected.append(())

    def can_add(genotype, separation):
        return (
            genotype not in selected
            and all(
                genotype_distance(genotype, centre) >= separation
                for centre in selected
            )
        )

    for genotype in ranked_candidates:
        if len(selected) >= requested:
            break
        if can_add(genotype, minimum_distance):
            selected.append(genotype)

    for genotype in ranked_all:
        if len(selected) >= requested:
            break
        if can_add(genotype, minimum_distance):
            selected.append(genotype)

    for genotype in ranked_all:
        if len(selected) >= requested:
            break
        if genotype not in selected:
            selected.append(genotype)

    return selected[:requested]


def assign_to_nearest_centre(
    genotype: tuple[int, ...],
    centres: list[tuple[int, ...]],
    centre_abundances: list[int],
) -> int:
    distances = np.asarray(
        [genotype_distance(genotype, centre) for centre in centres],
        dtype=int,
    )
    minimum = int(distances.min())
    tied = np.flatnonzero(distances == minimum)

    if len(tied) == 1:
        return int(tied[0])

    return int(
        max(
            tied,
            key=lambda index: (
                centre_abundances[int(index)],
                -int(index),
            ),
        )
    )


def partition_lcc_nodes(
    *,
    node_genotype: dict,
    centres: list[tuple[int, ...]],
    abundance_counts: Counter[tuple[int, ...]],
) -> tuple[dict[int, list], dict]:
    centre_abundances = [
        int(abundance_counts.get(centre, 0))
        for centre in centres
    ]

    domains = {index: [] for index in range(len(centres))}
    domain_by_node = {}

    for node, genotype in node_genotype.items():
        domain = assign_to_nearest_centre(
            genotype,
            centres,
            centre_abundances,
        )
        domains[domain].append(node)
        domain_by_node[node] = domain

    return domains, domain_by_node


def partition_population(
    *,
    abundance_counts: Counter[tuple[int, ...]],
    centres: list[tuple[int, ...]],
) -> dict[int, Counter[tuple[int, ...]]]:
    centre_abundances = [
        int(abundance_counts.get(centre, 0))
        for centre in centres
    ]
    domains = {
        index: Counter()
        for index in range(len(centres))
    }

    for genotype, count in abundance_counts.items():
        domain = assign_to_nearest_centre(
            genotype,
            centres,
            centre_abundances,
        )
        domains[domain][genotype] += int(count)

    return domains


# ============================================================
# Probability helpers
# ============================================================


def shell_log_sizes(L: int) -> np.ndarray:
    m = np.arange(L + 1, dtype=float)
    return gammaln(L + 1.0) - gammaln(m + 1.0) - gammaln(L - m + 1.0)


def shell_occupancy_probabilities(
    *,
    population_shell_counts: np.ndarray,
    lcc_shell_counts: np.ndarray,
    L: int,
) -> np.ndarray:
    mode = THEORY_OCCUPANCY_SOURCE.lower().strip()
    log_sizes = shell_log_sizes(L)

    if mode == "population_poisson":
        counts = np.asarray(population_shell_counts, dtype=float)
        means = np.zeros(L + 1, dtype=float)
        positive = counts > 0

        means[positive] = np.exp(
            np.log(counts[positive]) - log_sizes[positive]
        )
        probabilities = -np.expm1(-means)

    elif mode == "lcc_observed":
        counts = np.asarray(lcc_shell_counts, dtype=float)
        probabilities = np.zeros(L + 1, dtype=float)
        positive = counts > 0

        probabilities[positive] = np.exp(
            np.log(counts[positive]) - log_sizes[positive]
        )

    else:
        raise ValueError(
            "THEORY_OCCUPANCY_SOURCE must be 'population_poisson' "
            "or 'lcc_observed'."
        )

    return np.clip(probabilities, 0.0, 1.0)


def binomial_pmf(n: int, p: float) -> np.ndarray:
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
    result = np.asarray(binom.pmf(support, n, p), dtype=float)
    result = np.clip(result, 0.0, None)

    total = result.sum()
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError(f"Invalid Binomial PMF for n={n}, p={p}.")

    return result / total


def empirical_integer_pmf(
    values,
    *,
    maximum: int,
) -> np.ndarray:
    result = np.zeros(maximum + 1, dtype=float)
    values = np.asarray(list(values), dtype=int)

    if values.size == 0:
        result[0] = 1.0
        return result

    values = np.clip(values, 0, maximum)
    counts = np.bincount(values, minlength=maximum + 1).astype(float)
    return counts / counts.sum()


def truncate_with_tail(values: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(maximum + 1, dtype=float)

    keep = min(len(values), maximum + 1)
    result[:keep] = values[:keep]

    if len(values) > maximum + 1:
        result[maximum] += values[maximum + 1 :].sum()

    return result


def positive_degree_condition_pair(
    pmf: np.ndarray,
    marked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pmf = np.asarray(pmf, dtype=float).copy()
    marked = np.asarray(marked, dtype=float).copy()

    if not CONDITION_THEORY_ON_POSITIVE_DEGREE:
        return pmf, marked

    pmf[0] = 0.0
    marked[0] = 0.0
    mass = pmf.sum()

    if mass <= 0.0:
        return pmf, marked

    return pmf / mass, marked / mass


def marked_component_pmf(
    pmf: np.ndarray,
    neighbor_degree: float,
) -> np.ndarray:
    support = np.arange(len(pmf), dtype=float)
    return support * float(neighbor_degree) * np.asarray(pmf, dtype=float)


def convolve_three_marked(
    pmf_minus: np.ndarray,
    pmf_plus: np.ndarray,
    pmf_cross: np.ndarray,
    *,
    kappa_minus: float,
    kappa_plus: float,
    kappa_cross: float,
    maximum_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return P(K=k) and
        E[(X^- kappa^- + X^+ kappa^+ + Z kappa^x) 1_{K=k}].
    """
    marked_minus = marked_component_pmf(pmf_minus, kappa_minus)
    marked_plus = marked_component_pmf(pmf_plus, kappa_plus)
    marked_cross = marked_component_pmf(pmf_cross, kappa_cross)

    total_pmf = np.convolve(
        np.convolve(pmf_minus, pmf_plus),
        pmf_cross,
    )

    total_marked = (
        np.convolve(
            np.convolve(marked_minus, pmf_plus),
            pmf_cross,
        )
        + np.convolve(
            np.convolve(pmf_minus, marked_plus),
            pmf_cross,
        )
        + np.convolve(
            np.convolve(pmf_minus, pmf_plus),
            marked_cross,
        )
    )

    total_pmf = truncate_with_tail(total_pmf, maximum_degree)
    total_marked = truncate_with_tail(total_marked, maximum_degree)

    total_pmf = np.clip(total_pmf, 0.0, None)
    total_marked = np.clip(total_marked, 0.0, None)

    mass = total_pmf.sum()
    if mass <= 0.0 or not np.isfinite(mass):
        raise FloatingPointError("Invalid total degree PMF.")

    total_pmf /= mass
    total_marked /= mass

    return positive_degree_condition_pair(total_pmf, total_marked)


def knn_from_pmf_and_marked(
    pmf: np.ndarray,
    marked: np.ndarray,
) -> np.ndarray:
    support = np.arange(len(pmf), dtype=float)
    result = np.full(len(pmf), np.nan, dtype=float)

    valid = (support > 0) & (pmf > 0)
    result[valid] = marked[valid] / (support[valid] * pmf[valid])

    return result


# ============================================================
# Domain states and analytical k_nn(k)
# ============================================================


@dataclass
class DomainState:
    domain_id: int
    centre: tuple[int, ...]
    nodes: list
    domain_graph: nx.Graph
    shell_by_node: dict
    nodes_by_shell: dict
    lcc_shell_counts: np.ndarray
    population_shell_counts: np.ndarray
    occupancy: np.ndarray
    cross_degrees_by_shell: dict
    cross_mean_by_shell: np.ndarray
    cross_second_by_shell: np.ndarray
    internal_edge_shell_counts: Counter
    centre_node: object | None
    centre_degree: int | None
    first_shell_outward_probability: float


def shell_edge_counts(
    graph: nx.Graph,
    shell_by_node: dict,
) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()

    for source, target in graph.edges():
        pair = tuple(
            sorted(
                (
                    int(shell_by_node[source]),
                    int(shell_by_node[target]),
                )
            )
        )
        counts[pair] += 1

    return counts


def build_domain_states(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    centres: list[tuple[int, ...]],
    node_domains: dict[int, list],
    population_domains: dict[int, Counter[tuple[int, ...]]],
    L: int,
) -> tuple[
    dict[int, DomainState],
    dict,
    Counter,
]:
    states: dict[int, DomainState] = {}
    class_by_node = {}
    cross_target_counts: Counter = Counter()

    for domain_id, centre in enumerate(centres):
        nodes = list(node_domains[domain_id])
        domain_graph = lcc.subgraph(nodes).copy()

        shell_by_node = {
            node: genotype_distance(node_genotype[node], centre)
            for node in nodes
        }

        nodes_by_shell = {
            shell: []
            for shell in range(L + 1)
        }
        lcc_shell_counts = np.zeros(L + 1, dtype=np.int64)

        for node, shell in shell_by_node.items():
            nodes_by_shell[shell].append(node)
            lcc_shell_counts[shell] += 1
            class_by_node[node] = (domain_id, int(shell))

        population_shell_counts = np.zeros(L + 1, dtype=np.int64)

        for genotype, abundance in population_domains[domain_id].items():
            shell = genotype_distance(genotype, centre)
            if shell <= L:
                population_shell_counts[shell] += int(abundance)

        occupancy = shell_occupancy_probabilities(
            population_shell_counts=population_shell_counts,
            lcc_shell_counts=lcc_shell_counts,
            L=L,
        )

        cross_degrees_by_shell = {
            shell: []
            for shell in range(L + 1)
        }

        for node in nodes:
            shell = int(shell_by_node[node])
            cross_degree = int(
                lcc.degree(node) - domain_graph.degree(node)
            )
            cross_degrees_by_shell[shell].append(cross_degree)

        cross_mean = np.zeros(L + 1, dtype=float)
        cross_second = np.zeros(L + 1, dtype=float)

        for shell in range(L + 1):
            values = np.asarray(
                cross_degrees_by_shell[shell],
                dtype=float,
            )
            if values.size:
                cross_mean[shell] = float(values.mean())
                cross_second[shell] = float(np.mean(values**2))

        genotype_to_node = {
            node_genotype[node]: node
            for node in nodes
        }
        centre_node = genotype_to_node.get(centre)
        centre_degree = (
            int(lcc.degree(centre_node))
            if centre_node is not None
            else None
        )

        internal_edge_counts = shell_edge_counts(
            domain_graph,
            shell_by_node,
        )

        shell1_nodes = int(lcc_shell_counts[1]) if L >= 1 else 0
        shell12_edges = int(internal_edge_counts.get((1, 2), 0))
        denominator = shell1_nodes * max(1, L - 1)

        p_first_out = (
            float(np.clip(shell12_edges / denominator, 0.0, 1.0))
            if denominator > 0
            else 0.0
        )

        states[domain_id] = DomainState(
            domain_id=domain_id,
            centre=centre,
            nodes=nodes,
            domain_graph=domain_graph,
            shell_by_node=shell_by_node,
            nodes_by_shell=nodes_by_shell,
            lcc_shell_counts=lcc_shell_counts,
            population_shell_counts=population_shell_counts,
            occupancy=occupancy,
            cross_degrees_by_shell=cross_degrees_by_shell,
            cross_mean_by_shell=cross_mean,
            cross_second_by_shell=cross_second,
            internal_edge_shell_counts=internal_edge_counts,
            centre_node=centre_node,
            centre_degree=centre_degree,
            first_shell_outward_probability=p_first_out,
        )

    for source, target in lcc.edges():
        source_class = class_by_node[source]
        target_class = class_by_node[target]

        if source_class[0] != target_class[0]:
            cross_target_counts[(source_class, target_class)] += 1
            cross_target_counts[(target_class, source_class)] += 1

    return states, class_by_node, cross_target_counts


def local_internal_component_pmfs(
    *,
    state: DomainState,
    shell: int,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate PMFs for inward and outward internal edges.
    """
    m = int(shell)

    if (
        m == 1
        and CONDITION_FIRST_SHELL_INCIDENCE
    ):
        guaranteed = (
            1
            if state.centre_node is not None
            else 0
        )
        inward = np.zeros(guaranteed + 1, dtype=float)
        inward[guaranteed] = 1.0

        outward = binomial_pmf(
            max(0, L - 1),
            state.first_shell_outward_probability,
        )
        return inward, outward

    p_minus = (
        float(state.occupancy[m - 1])
        if m > 0
        else 0.0
    )
    p_plus = (
        float(state.occupancy[m + 1])
        if m < L
        else 0.0
    )

    return (
        binomial_pmf(m, p_minus),
        binomial_pmf(L - m, p_plus),
    )


def expected_internal_degree_unconditioned(
    *,
    state: DomainState,
    shell: int,
    L: int,
) -> float:
    """
    Expected internal degree of a random node in one class.
    """
    if (
        shell == 0
        and CONDITION_CENTRE_DEGREE
        and state.centre_degree is not None
    ):
        return float(
            state.centre_degree
            - state.cross_mean_by_shell[0]
        )

    inward, outward = local_internal_component_pmfs(
        state=state,
        shell=shell,
        L=L,
    )

    return float(
        np.dot(np.arange(len(inward)), inward)
        + np.dot(np.arange(len(outward)), outward)
    )


def internal_neighbor_degree_constants(
    *,
    state: DomainState,
    shell: int,
    L: int,
) -> tuple[float, float]:
    """
    Expected total degrees of represented inner and outer neighbors.

    The known focal edge contributes one exactly. Cross-domain degrees of the
    destination shell are added through their observed class mean.
    """
    m = int(shell)

    if m > 0:
        destination = m - 1

        if (
            m == 1
            and state.centre_degree is not None
        ):
            kappa_minus = float(state.centre_degree)

        elif (
            destination == 1
            and CONDITION_FIRST_SHELL_INCIDENCE
        ):
            # Reached from shell 2: one known shell-2 edge, the centre edge,
            # the remaining L-2 shell-2 opportunities, and boundary bridges.
            centre_edge = 1.0 if state.centre_node is not None else 0.0
            kappa_minus = (
                1.0
                + centre_edge
                + max(0, L - 2)
                * state.first_shell_outward_probability
                + state.cross_mean_by_shell[destination]
            )

        else:
            p_inward = (
                float(state.occupancy[m - 2])
                if m >= 2
                else 0.0
            )
            p_other_outward = float(state.occupancy[m])

            kappa_minus = (
                1.0
                + max(0, m - 1) * p_inward
                + max(0, L - m) * p_other_outward
                + state.cross_mean_by_shell[destination]
            )
    else:
        kappa_minus = 0.0

    if m < L:
        destination = m + 1

        if (
            destination == 1
            and CONDITION_FIRST_SHELL_INCIDENCE
        ):
            # Reached from the centre: centre edge is known.
            kappa_plus = (
                1.0
                + max(0, L - 1)
                * state.first_shell_outward_probability
                + state.cross_mean_by_shell[destination]
            )

        else:
            p_other_inward = float(state.occupancy[m])
            p_outward = (
                float(state.occupancy[m + 2])
                if m + 2 <= L
                else 0.0
            )

            kappa_plus = (
                1.0
                + m * p_other_inward
                + max(0, L - m - 1) * p_outward
                + state.cross_mean_by_shell[destination]
            )
    else:
        kappa_plus = 0.0

    return float(kappa_minus), float(kappa_plus)


def cross_destination_degree(
    *,
    destination_class: tuple[int, int],
    states: dict[int, DomainState],
    L: int,
) -> float:
    """
    Expected degree of a node reached through a cross-domain edge.

    Cross-edge sampling size-biases the destination cross degree:
        E[Z | reached by a cross edge] = E[Z^2]/E[Z].
    """
    domain_id, shell = destination_class
    state = states[domain_id]

    if (
        shell == 0
        and state.centre_degree is not None
    ):
        return float(state.centre_degree)

    internal_mean = expected_internal_degree_unconditioned(
        state=state,
        shell=shell,
        L=L,
    )

    mean_z = float(state.cross_mean_by_shell[shell])
    second_z = float(state.cross_second_by_shell[shell])

    if mean_z > 0.0:
        size_biased_z = second_z / mean_z
    else:
        size_biased_z = 1.0

    return float(internal_mean + size_biased_z)


def cross_neighbor_degree_constants(
    *,
    states: dict[int, DomainState],
    cross_target_counts: Counter,
    L: int,
) -> dict[tuple[int, int], float]:
    """
    Edge-incidence-weighted expected degree of a cross-domain neighbor.
    """
    result: dict[tuple[int, int], float] = {}

    source_classes = {
        source_class
        for source_class, _ in cross_target_counts
    }

    for source_class in source_classes:
        targets = [
            (target_class, count)
            for (candidate_source, target_class), count
            in cross_target_counts.items()
            if candidate_source == source_class
        ]

        total = sum(count for _, count in targets)

        if total <= 0:
            result[source_class] = 0.0
            continue

        result[source_class] = float(
            sum(
                count
                * cross_destination_degree(
                    destination_class=target_class,
                    states=states,
                    L=L,
                )
                for target_class, count in targets
            )
            / total
        )

    return result


def centre_knn_prediction(
    *,
    state: DomainState,
    lcc: nx.Graph,
    class_by_node: dict,
    states: dict[int, DomainState],
    cross_destination_kappa: dict[tuple[int, int], float],
    L: int,
) -> float:
    """
    Condition the centre degree exactly and predict degrees of its neighbors.
    """
    node = state.centre_node

    if node is None or state.centre_degree is None or state.centre_degree <= 0:
        return np.nan

    neighbor_predictions: list[float] = []

    _, kappa_shell1 = internal_neighbor_degree_constants(
        state=state,
        shell=0,
        L=L,
    )

    for neighbor in lcc.neighbors(node):
        neighbor_class = class_by_node[neighbor]

        if neighbor_class[0] == state.domain_id:
            neighbor_predictions.append(kappa_shell1)
        else:
            neighbor_predictions.append(
                cross_destination_degree(
                    destination_class=neighbor_class,
                    states=states,
                    L=L,
                )
            )

    if not neighbor_predictions:
        return np.nan

    return float(np.mean(neighbor_predictions))


def build_knn_theory_for_centres(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
    centres: list[tuple[int, ...]],
    theory_type: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """
    Construct expected degree-k node counts and the associated k_nn sum.
    """
    L = int(metadata["L"])

    node_domains, domain_by_node = partition_lcc_nodes(
        node_genotype=node_genotype,
        centres=centres,
        abundance_counts=abundance_counts,
    )
    population_domains = partition_population(
        abundance_counts=abundance_counts,
        centres=centres,
    )

    (
        states,
        class_by_node,
        cross_target_counts,
    ) = build_domain_states(
        lcc=lcc,
        node_genotype=node_genotype,
        centres=centres,
        node_domains=node_domains,
        population_domains=population_domains,
        L=L,
    )

    cross_kappa = cross_neighbor_degree_constants(
        states=states,
        cross_target_counts=cross_target_counts,
        L=L,
    )

    expected_counts = np.zeros(L + 1, dtype=float)
    expected_knn_sum = np.zeros(L + 1, dtype=float)
    class_rows: list[dict] = []

    for domain_id, state in states.items():
        for shell in range(L + 1):
            class_nodes = int(state.lcc_shell_counts[shell])

            if class_nodes == 0:
                continue

            source_class = (domain_id, shell)

            if (
                shell == 0
                and CONDITION_CENTRE_DEGREE
                and state.centre_degree is not None
            ):
                degree = int(state.centre_degree)
                pmf = np.zeros(L + 1, dtype=float)
                pmf[degree] = 1.0

                knn_curve = np.full(L + 1, np.nan, dtype=float)
                knn_curve[degree] = centre_knn_prediction(
                    state=state,
                    lcc=lcc,
                    class_by_node=class_by_node,
                    states=states,
                    cross_destination_kappa=cross_kappa,
                    L=L,
                )

            else:
                pmf_minus, pmf_plus = local_internal_component_pmfs(
                    state=state,
                    shell=shell,
                    L=L,
                )

                pmf_cross = (
                    empirical_integer_pmf(
                        state.cross_degrees_by_shell[shell],
                        maximum=L,
                    )
                    if CONDITION_CROSS_DOMAIN_DEGREES
                    else np.array([1.0], dtype=float)
                )

                kappa_minus, kappa_plus = (
                    internal_neighbor_degree_constants(
                        state=state,
                        shell=shell,
                        L=L,
                    )
                )

                kappa_cross = float(
                    cross_kappa.get(source_class, 0.0)
                )

                pmf, marked = convolve_three_marked(
                    pmf_minus,
                    pmf_plus,
                    pmf_cross,
                    kappa_minus=kappa_minus,
                    kappa_plus=kappa_plus,
                    kappa_cross=kappa_cross,
                    maximum_degree=L,
                )

                knn_curve = knn_from_pmf_and_marked(
                    pmf,
                    marked,
                )

            expected_counts += class_nodes * pmf

            valid = np.isfinite(knn_curve)
            expected_knn_sum[valid] += (
                class_nodes
                * pmf[valid]
                * knn_curve[valid]
            )

            class_rows.append({
                **metadata,
                "theory_type": theory_type,
                "domain_id": int(domain_id),
                "centre_genotype": format_genotype(state.centre),
                "shell": int(shell),
                "class_nodes": class_nodes,
                "centre_degree": state.centre_degree,
                "occupancy_probability": float(state.occupancy[shell]),
                "mean_cross_degree": float(
                    state.cross_mean_by_shell[shell]
                ),
                "cross_neighbor_degree": float(
                    cross_kappa.get(source_class, 0.0)
                ),
            })

    diagnostics = {
        "n_effective_centres": int(len(centres)),
        "effective_centres": " | ".join(
            format_genotype(centre)
            for centre in centres
        ),
        "primary_centre": centres[0],
        "domain_sizes": tuple(
            len(node_domains[index])
            for index in range(len(centres))
        ),
    }

    return (
        expected_counts,
        expected_knn_sum,
        class_rows,
        diagnostics,
    )


def build_single_centre_knn_theory(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
):
    centre = choose_anchor_genotype(
        abundance_counts=abundance_counts,
        lcc_genotypes=set(node_genotype.values()),
    )

    return build_knn_theory_for_centres(
        lcc=lcc,
        node_genotype=node_genotype,
        abundance_counts=abundance_counts,
        metadata=metadata,
        centres=[centre],
        theory_type=SINGLE_CENTRE_THEORY,
    )


def build_multi_centre_knn_theory(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
):
    centres = select_effective_centres(
        lcc=lcc,
        node_genotype=node_genotype,
        abundance_counts=abundance_counts,
        metadata=metadata,
    )

    return build_knn_theory_for_centres(
        lcc=lcc,
        node_genotype=node_genotype,
        abundance_counts=abundance_counts,
        metadata=metadata,
        centres=centres,
        theory_type=MULTI_CENTRE_THEORY,
    )


def analytical_description_markdown() -> str:
    occupancy_text = (
        r"$p_{\alpha m}=1-\exp[-n_{\alpha m}/\binom{L}{m}]$"
        if THEORY_OCCUPANCY_SOURCE.lower().strip() == "population_poisson"
        else
        r"$p_{\alpha m}=S_{\alpha m}/\binom{L}{m}$"
    )

    template = r"""
### Conditioned multi-centre prediction for nearest-neighbor degree

The LCC is partitioned into mutational Voronoi domains around the reconstructed
effective centres. For a node in local shell $(\alpha,m)$,

$$
K_{\alpha m}
=
X^-_{\alpha m}
+
X^+_{\alpha m}
+
Z_{\alpha m},
$$

where $X^-$ and $X^+$ count internal links to shells $m-1$ and $m+1$, while
$Z$ counts cross-domain bridges. The shell occupation is estimated with
__OCCUPANCY_TEXT__.

Let $\kappa^-_{\alpha m}$, $\kappa^+_{\alpha m}$, and
$\kappa^\times_{\alpha m}$ denote the expected degrees of a neighbor reached
through an inward, outward, or cross-domain edge. Then

$$
k_{\rm nn}(k\mid\alpha,m)
=
\frac{
\sum_{r+s+z=k}
P_-(r)P_+(s)P_\times(z)
\left(
r\kappa^-_{\alpha m}
+s\kappa^+_{\alpha m}
+z\kappa^\times_{\alpha m}
\right)
}{
k\,P_{\alpha m}(k)
}.
$$

The complete prediction is mixed with the same expected degree-k class counts
used in the multi-centre degree theory:

$$
k_{\rm nn}(k)
=
\frac{
\sum_{\alpha,m}
S_{\alpha m}P_{\alpha m}(k)
k_{\rm nn}(k\mid\alpha,m)
}{
\sum_{\alpha,m}
S_{\alpha m}P_{\alpha m}(k)
}.
$$

Centre degrees, first-shell incidence, cross-domain degree distributions, and
domain-boundary incidence are conditioned on the observed LCC. The internal
shell dependence and the degree-conditioned balance of edge types are
predicted analytically.
"""

    return template.replace("__OCCUPANCY_TEXT__", occupancy_text)


# ============================================================
# One-network measurements
# ============================================================


def empirical_knn_by_degree(
    lcc: nx.Graph,
) -> tuple[list[dict], float, float]:
    degrees = dict(lcc.degree())

    counts: Counter[int] = Counter()
    knn_sums: defaultdict[int, float] = defaultdict(float)

    for node, degree in degrees.items():
        degree = int(degree)

        if degree <= 0:
            continue

        neighbor_degree = float(
            np.mean(
                [
                    degrees[neighbor]
                    for neighbor in lcc.neighbors(node)
                ]
            )
        )

        counts[degree] += 1
        knn_sums[degree] += neighbor_degree

    rows = [
        {
            "degree": int(degree),
            "node_count": int(counts[degree]),
            "knn_sum": float(knn_sums[degree]),
            "knn": float(knn_sums[degree] / counts[degree]),
        }
        for degree in sorted(counts)
    ]

    degree_values = np.asarray(list(degrees.values()), dtype=float)

    if degree_values.sum() > 0:
        uncorrelated = float(
            np.sum(degree_values**2)
            / np.sum(degree_values)
        )
    else:
        uncorrelated = np.nan

    try:
        assortativity = float(
            nx.degree_assortativity_coefficient(lcc)
        )
    except Exception:
        assortativity = np.nan

    return rows, uncorrelated, assortativity


def analyze_one_network(
    network_file: Path,
    metadata: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    graph = nx.read_graphml(network_file)
    graph = to_simple_undirected_graph(graph)
    lcc, n_components = largest_connected_component(graph)

    node_genotype = {}

    for node, attributes in lcc.nodes(data=True):
        node_genotype[node] = parse_genotype_name(
            genotype_name(node, attributes)
        )

    if len(set(node_genotype.values())) != len(node_genotype):
        raise ValueError(
            f"Repeated genotype labels in GraphML {network_file}."
        )

    abundance_file = resolve_abundance_file(metadata)

    if abundance_file is not None:
        abundance_counts = read_abundance_counter(abundance_file)
    else:
        abundance_counts = Counter()

        for node, attributes in lcc.nodes(data=True):
            abundance = attributes.get("abundance", 1)
            try:
                abundance = int(float(abundance))
            except (TypeError, ValueError):
                abundance = 1

            abundance_counts[node_genotype[node]] += max(1, abundance)

        warnings.warn(
            f"Abundance file not found for {network_file}; "
            "GraphML node abundances are being used.",
            RuntimeWarning,
        )

    empirical_basic, uncorrelated, assortativity = empirical_knn_by_degree(lcc)
    empirical_rows = [
        {
            **metadata,
            **row,
        }
        for row in empirical_basic
    ]

    theory_rows: list[dict] = []
    class_rows: list[dict] = []

    single_diagnostics = None
    multi_diagnostics = None

    if ADD_ANALYTICAL_THEORY and COMPUTE_SINGLE_CENTRE_BASELINE:
        (
            counts,
            knn_sum,
            rows,
            single_diagnostics,
        ) = build_single_centre_knn_theory(
            lcc=lcc,
            node_genotype=node_genotype,
            abundance_counts=abundance_counts,
            metadata=metadata,
        )

        class_rows.extend(rows)

        for degree in range(len(counts)):
            theory_rows.append({
                **metadata,
                "theory_type": SINGLE_CENTRE_THEORY,
                "degree": int(degree),
                "expected_count": float(counts[degree]),
                "expected_knn_sum": float(knn_sum[degree]),
            })

    if ADD_ANALYTICAL_THEORY and COMPUTE_MULTI_CENTRE_THEORY:
        (
            counts,
            knn_sum,
            rows,
            multi_diagnostics,
        ) = build_multi_centre_knn_theory(
            lcc=lcc,
            node_genotype=node_genotype,
            abundance_counts=abundance_counts,
            metadata=metadata,
        )

        class_rows.extend(rows)

        for degree in range(len(counts)):
            theory_rows.append({
                **metadata,
                "theory_type": MULTI_CENTRE_THEORY,
                "degree": int(degree),
                "expected_count": float(counts[degree]),
                "expected_knn_sum": float(knn_sum[degree]),
            })

    degrees = np.asarray(
        [degree for _, degree in lcc.degree()],
        dtype=float,
    )

    primary = (
        multi_diagnostics
        if multi_diagnostics is not None
        else single_diagnostics
    )

    if primary is None:
        centre = choose_anchor_genotype(
            abundance_counts=abundance_counts,
            lcc_genotypes=set(node_genotype.values()),
        )
        centre_count = 1
        centre_text = format_genotype(centre)
    else:
        centre = primary["primary_centre"]
        centre_count = int(primary["n_effective_centres"])
        centre_text = str(primary["effective_centres"])

    genotype_to_node = {
        genotype: node
        for node, genotype in node_genotype.items()
    }
    centre_node = genotype_to_node.get(centre)

    inventory_row = {
        **metadata,
        "abundance_file": (
            str(abundance_file)
            if abundance_file is not None
            else None
        ),
        "primary_centre": format_genotype(centre),
        "primary_centre_degree": (
            int(lcc.degree(centre_node))
            if centre_node is not None
            else np.nan
        ),
        "n_effective_centres": centre_count,
        "effective_centres": centre_text,
        "population_read": int(sum(abundance_counts.values())),
        "full_nodes": int(graph.number_of_nodes()),
        "full_edges": int(graph.number_of_edges()),
        "n_components": int(n_components),
        "lcc_nodes": int(lcc.number_of_nodes()),
        "lcc_edges": int(lcc.number_of_edges()),
        "lcc_node_fraction": float(
            lcc.number_of_nodes() / graph.number_of_nodes()
        ),
        "mean_lcc_degree": float(degrees.mean()),
        "maximum_lcc_degree": int(degrees.max()),
        "uncorrelated_knn": uncorrelated,
        "assortativity": assortativity,
    }

    return inventory_row, empirical_rows, theory_rows, class_rows


# ============================================================
# Collection, aggregation, and diagnostics
# ============================================================


def collect_selected_networks() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    catalog = discover_networks()

    if catalog.empty:
        raise RuntimeError(
            "No parseable model4a30 GraphML snapshots were found below:\n"
            f"  {Path(NETWORK_ROOT).expanduser().resolve()}"
        )

    selected = catalog.loc[
        catalog.apply(
            lambda row: metadata_is_selected(row.to_dict()),
            axis=1,
        )
    ].copy()

    print(f"Network root: {Path(NETWORK_ROOT).expanduser().resolve()}")
    print(f"Parseable GraphML snapshots: {len(catalog)}")
    print(f"Available pre-peak snapshots: {int(catalog['is_pre_peak'].sum())}")
    print(f"Selection: {selector_description()}")
    print(f"Selected pre-peak snapshots: {len(selected)}")

    if selected.empty:
        columns = [
            "N",
            "L",
            "q",
            "mu",
            "f_initial",
            "f_background",
            "peak_increment",
            "tau",
            "delta",
            "t",
            "peak_event_index",
            "iteration",
            "network_file",
        ]

        preview = (
            catalog.loc[catalog["is_pre_peak"], columns]
            .head(40)
            .to_string(index=False)
        )

        raise RuntimeError(
            "No pre-peak networks matched the configured selectors.\n\n"
            f"Available examples:\n{preview}"
        )

    inventory_rows = []
    empirical_rows = []
    theory_rows = []
    class_rows = []
    failures = []

    for index, row in enumerate(selected.itertuples(index=False), start=1):
        metadata = row._asdict()
        network_file = Path(metadata["network_file"])

        try:
            inventory, empirical, theory, classes = analyze_one_network(
                network_file,
                metadata,
            )
        except Exception as exc:
            failures.append(
                (
                    str(network_file),
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        inventory_rows.append(inventory)
        empirical_rows.extend(empirical)
        theory_rows.extend(theory)
        class_rows.extend(classes)

        print(
            f"[{index:>4}/{len(selected)}] "
            f"event={metadata['peak_event_index']}, "
            f"t={metadata['t']}, "
            f"iteration={metadata['iteration']}: "
            f"LCC={inventory['lcc_nodes']:,}/"
            f"{inventory['full_nodes']:,}, "
            f"centres={inventory['n_effective_centres']}, "
            f"r={inventory['assortativity']:.4f}"
        )

    if failures:
        print("\nSkipped networks:")
        for path, reason in failures:
            print(f"  {path}: {reason}")

    if not inventory_rows:
        raise RuntimeError("Every selected network failed during analysis.")

    network_inventory = (
        pd.DataFrame(inventory_rows)
        .sort_values(["q", "delta", "t", "iteration", "network"])
        .reset_index(drop=True)
    )

    empirical_knn = (
        pd.DataFrame(empirical_rows)
        .sort_values(
            ["q", "delta", "t", "iteration", "network", "degree"]
        )
        .reset_index(drop=True)
    )

    analytical_knn = (
        pd.DataFrame(theory_rows)
        if theory_rows
        else pd.DataFrame(
            columns=[
                *NETWORK_COLUMNS,
                "theory_type",
                "degree",
                "expected_count",
                "expected_knn_sum",
            ]
        )
    )

    if not analytical_knn.empty:
        analytical_knn = (
            analytical_knn
            .sort_values(
                [
                    "q",
                    "delta",
                    "t",
                    "iteration",
                    "network",
                    "theory_type",
                    "degree",
                ]
            )
            .reset_index(drop=True)
        )

    class_statistics = (
        pd.DataFrame(class_rows)
        if class_rows
        else pd.DataFrame()
    )

    return (
        network_inventory,
        empirical_knn,
        analytical_knn,
        class_statistics,
    )


def aggregate_knn_curves(
    network_inventory: pd.DataFrame,
    empirical_knn: pd.DataFrame,
    analytical_knn: pd.DataFrame,
) -> pd.DataFrame:
    empirical = (
        empirical_knn
        .groupby(EVENT_COLUMNS + ["degree"], as_index=False, dropna=False)
        .agg(
            empirical_count=("node_count", "sum"),
            empirical_knn_sum=("knn_sum", "sum"),
        )
    )
    empirical["empirical_knn"] = (
        empirical["empirical_knn_sum"]
        / empirical["empirical_count"]
    )

    totals = (
        network_inventory
        .groupby(EVENT_COLUMNS, as_index=False, dropna=False)
        .agg(
            n_networks=("network", "nunique"),
            total_lcc_nodes=("lcc_nodes", "sum"),
            mean_lcc_nodes=("lcc_nodes", "mean"),
            mean_degree=("mean_lcc_degree", "mean"),
            maximum_degree=("maximum_lcc_degree", "max"),
            mean_uncorrelated_knn=("uncorrelated_knn", "mean"),
            std_uncorrelated_knn=("uncorrelated_knn", "std"),
            mean_assortativity=("assortativity", "mean"),
            std_assortativity=("assortativity", "std"),
            mean_effective_centres=("n_effective_centres", "mean"),
        )
        .fillna({
            "std_uncorrelated_knn": 0.0,
            "std_assortativity": 0.0,
        })
    )

    if ADD_ANALYTICAL_THEORY and not analytical_knn.empty:
        theory = (
            analytical_knn
            .groupby(
                EVENT_COLUMNS + ["theory_type", "degree"],
                as_index=False,
                dropna=False,
            )
            .agg(
                theory_expected_count=("expected_count", "sum"),
                theory_knn_sum=("expected_knn_sum", "sum"),
            )
        )

        theory["theory_knn"] = np.where(
            theory["theory_expected_count"] > 0,
            theory["theory_knn_sum"]
            / theory["theory_expected_count"],
            np.nan,
        )

        merged = theory.merge(
            empirical,
            on=EVENT_COLUMNS + ["degree"],
            how="left",
        )
    else:
        merged = empirical.copy()
        merged["theory_type"] = "none"
        merged["theory_expected_count"] = np.nan
        merged["theory_knn_sum"] = np.nan
        merged["theory_knn"] = np.nan

    merged = merged.merge(
        totals,
        on=EVENT_COLUMNS,
        how="left",
    )

    return (
        merged
        .sort_values(EVENT_COLUMNS + ["theory_type", "degree"])
        .reset_index(drop=True)
    )


def weighted_nrmse(
    empirical_values: np.ndarray,
    theory_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    empirical = np.asarray(empirical_values, dtype=float)
    theory = np.asarray(theory_values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(empirical)
        & np.isfinite(theory)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not np.any(valid):
        return np.nan

    empirical = empirical[valid]
    theory = theory[valid]
    weights = weights[valid]

    numerator = np.sum(
        weights * (empirical - theory) ** 2
    )
    denominator = np.sum(
        weights * empirical**2
    )

    if denominator <= 0.0:
        return np.nan

    return float(np.sqrt(numerator / denominator))


def compute_network_theory_diagnostics(
    empirical_knn: pd.DataFrame,
    analytical_knn: pd.DataFrame,
) -> pd.DataFrame:
    if analytical_knn.empty:
        return pd.DataFrame()

    empirical_lookup = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in empirical_knn.groupby(
            NETWORK_COLUMNS,
            sort=False,
            dropna=False,
        )
    }

    rows = []

    for group_key, theory_group in analytical_knn.groupby(
        NETWORK_COLUMNS + ["theory_type"],
        sort=True,
        dropna=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        metadata_values = group_key[:-1]
        theory_type = group_key[-1]
        empirical_group = empirical_lookup.get(metadata_values)

        if empirical_group is None:
            continue

        theory_curve = theory_group.loc[
            theory_group["expected_count"] > 0,
            ["degree", "expected_count", "expected_knn_sum"],
        ].copy()

        theory_curve["theory_knn"] = (
            theory_curve["expected_knn_sum"]
            / theory_curve["expected_count"]
        )

        comparison = empirical_group[
            ["degree", "node_count", "knn"]
        ].merge(
            theory_curve[["degree", "theory_knn"]],
            on="degree",
            how="inner",
        )

        metadata = dict(zip(NETWORK_COLUMNS, metadata_values))

        rows.append({
            **metadata,
            "theory_type": theory_type,
            "weighted_nrmse": weighted_nrmse(
                comparison["knn"].to_numpy(dtype=float),
                comparison["theory_knn"].to_numpy(dtype=float),
                comparison["node_count"].to_numpy(dtype=float),
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            EVENT_COLUMNS
            + ["theory_type", "iteration", "network"]
        )
        .reset_index(drop=True)
    )


def summarize_theory_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()

    return (
        diagnostics
        .groupby(
            EVENT_COLUMNS + ["theory_type"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_error=(DIAGNOSTIC_METRIC, "mean"),
            std_error=(DIAGNOSTIC_METRIC, "std"),
            n_networks=(DIAGNOSTIC_METRIC, "count"),
        )
        .fillna({"std_error": 0.0})
        .sort_values(
            EVENT_COLUMNS + ["theory_type"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Plotting
# ============================================================


def safe_number(value) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}".replace("-", "m").replace("+", "p")


def family_figure_tag(metadata: dict) -> str:
    return (
        f"{MODEL}_pre_peak_LCC_knn_multicentre_panel"
        f"_mu_{safe_number(metadata['mu'])}"
        f"_N_{safe_number(metadata['N'])}"
        f"_L_{safe_number(metadata['L'])}"
        f"_q_{safe_number(metadata['q'])}"
        f"_fpeak_{safe_number(metadata['f_initial'])}"
        f"_fbg_{safe_number(metadata['f_background'])}"
        f"_fstep_{safe_number(metadata['peak_increment'])}"
        f"_tau_{safe_number(metadata['tau'])}"
        f"_delta_{safe_number(metadata['delta'])}"
    )


def family_title(metadata: dict) -> str:
    return (
        rf"$N={int(metadata['N']):,}$, "
        rf"$L={int(metadata['L'])}$, "
        rf"$q={metadata['q']:g}$, "
        rf"$f_0={metadata['f_initial']:g}$, "
        rf"$f_{{\rm bg}}={metadata['f_background']:g}$, "
        rf"$\Delta f={metadata['peak_increment']:g}$, "
        rf"$\lambda={metadata['mu']:g}$, "
        rf"$\tau={int(metadata['tau'])}$, "
        rf"$\Delta={int(metadata['delta'])}$"
    )


def configure_knn_axis(ax: plt.Axes) -> None:
    if LOG_X:
        ax.set_xscale("log")
    if LOG_Y:
        ax.set_yscale("log")

    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    if LOG_X or LOG_Y:
        ax.grid(True, which="minor", linewidth=0.42, alpha=0.09)


def valid_curve(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    curve = data.loc[
        np.isfinite(data[column])
        & (data[column] > 0)
    ].copy()

    if LOG_X:
        curve = curve.loc[curve["degree"] > 0].copy()

    return curve


def diagnostic_style(theory_type: str) -> dict:
    if theory_type == MULTI_CENTRE_THEORY:
        return {
            "color": MULTI_CENTRE_COLOR,
            "marker": "s",
            "linestyle": "--",
            "label": THEORY_LABELS[theory_type],
        }

    return {
        "color": SINGLE_CENTRE_COLOR,
        "marker": "o",
        "linestyle": ":",
        "label": THEORY_LABELS[theory_type],
    }


def infer_common_y_limits(family_data: pd.DataFrame) -> tuple[float, float]:
    values = []

    for column in ("empirical_knn", "theory_knn"):
        if column in family_data:
            valid = family_data[column].to_numpy(dtype=float)
            valid = valid[np.isfinite(valid) & (valid > 0)]
            values.extend(valid.tolist())

    if "mean_uncorrelated_knn" in family_data:
        valid = family_data["mean_uncorrelated_knn"].to_numpy(dtype=float)
        valid = valid[np.isfinite(valid) & (valid > 0)]
        values.extend(valid.tolist())

    if not values:
        return 0.8, 1.2

    minimum = min(values)
    maximum = max(values)

    lower = (
        float(COMMON_Y_MIN)
        if COMMON_Y_MIN is not None
        else max(1e-3, minimum / 1.35)
    )
    upper = (
        float(COMMON_Y_MAX)
        if COMMON_Y_MAX is not None
        else maximum * 1.28
    )

    return lower, upper


def plot_diagnostic_panel(
    ax: plt.Axes,
    family_diagnostics: pd.DataFrame,
) -> None:
    for theory_type in (
        MULTI_CENTRE_THEORY,
        SINGLE_CENTRE_THEORY,
    ):
        data = family_diagnostics.loc[
            family_diagnostics["theory_type"] == theory_type
        ].sort_values("peak_event_index")

        if data.empty:
            continue

        style = diagnostic_style(theory_type)

        ax.errorbar(
            data["peak_event_index"],
            data["mean_error"],
            yerr=data["std_error"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            markersize=4.6,
            capsize=2.7,
            label=style["label"],
        )

    ax.set_xlabel("Pre-peak event")
    ax.set_ylabel(r"Weighted normalized error, $\varepsilon_{\rm nn}$")
    ax.set_title(r"$\mathbf{f}$ Theory error", loc="left")
    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    ax.text(
        0.97,
        0.95,
        "lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
    )


def plot_pre_peak_knn_panels(
    aggregated_knn: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
) -> list[Path]:
    saved_paths = []

    for family_key, family_data in aggregated_knn.groupby(
        FAMILY_COLUMNS,
        sort=True,
        dropna=False,
    ):
        if not isinstance(family_key, tuple):
            family_key = (family_key,)

        family_metadata = dict(zip(FAMILY_COLUMNS, family_key))

        event_rows = (
            family_data[
                ["t", "next_peak_time", "peak_event_index"]
            ]
            .drop_duplicates()
            .sort_values(["peak_event_index", "t"])
            .reset_index(drop=True)
        )

        n_events = len(event_rows)
        n_panels = n_events + int(ADD_DIAGNOSTIC_PANEL)
        n_columns = min(MAX_PANEL_COLUMNS, n_panels)
        n_rows = math.ceil(n_panels / n_columns)

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                PANEL_WIDTH * n_columns,
                PANEL_HEIGHT * n_rows,
            ),
            squeeze=False,
        )

        axes_flat = axes.ravel()

        maximum_degree = int(
            max(
                family_metadata["L"],
                family_data["maximum_degree"].max(),
            )
        )
        x_max = (
            float(COMMON_X_MAX)
            if COMMON_X_MAX is not None
            else 1.12 * maximum_degree
        )
        y_min, y_max = infer_common_y_limits(family_data)

        for panel_index, event in event_rows.iterrows():
            ax = axes_flat[panel_index]

            event_data = family_data.loc[
                (family_data["t"] == event["t"])
                & (
                    family_data["peak_event_index"]
                    == event["peak_event_index"]
                )
            ].copy()

            empirical_data = (
                event_data[
                    [
                        "degree",
                        "empirical_knn",
                        "empirical_count",
                    ]
                ]
                .drop_duplicates("degree")
                .sort_values("degree")
            )
            empirical = valid_curve(
                empirical_data,
                "empirical_knn",
            )

            ax.plot(
                empirical["degree"],
                empirical["empirical_knn"],
                linestyle="none",
                marker="o",
                markersize=4.25,
                color=SIMULATION_COLOR,
                label="Simulation",
                zorder=4,
            )

            multi = event_data.loc[
                event_data["theory_type"] == MULTI_CENTRE_THEORY
            ]
            multi = valid_curve(multi, "theory_knn")

            if ADD_ANALYTICAL_THEORY and not multi.empty:
                ax.plot(
                    multi["degree"],
                    multi["theory_knn"],
                    linestyle="--",
                    linewidth=1.65,
                    color=MULTI_CENTRE_COLOR,
                    label=THEORY_LABELS[MULTI_CENTRE_THEORY],
                    zorder=3,
                )

            if SHOW_SINGLE_CENTRE_IN_EVENT_PANELS:
                single = event_data.loc[
                    event_data["theory_type"] == SINGLE_CENTRE_THEORY
                ]
                single = valid_curve(single, "theory_knn")

                if not single.empty:
                    ax.plot(
                        single["degree"],
                        single["theory_knn"],
                        linestyle=":",
                        linewidth=1.6,
                        color=SINGLE_CENTRE_COLOR,
                        label=THEORY_LABELS[SINGLE_CENTRE_THEORY],
                        zorder=2,
                    )

            if SHOW_UNCORRELATED_REFERENCE:
                uncorrelated = float(
                    event_data["mean_uncorrelated_knn"].max()
                )
                if np.isfinite(uncorrelated) and uncorrelated > 0:
                    ax.axhline(
                        uncorrelated,
                        linestyle="-.",
                        linewidth=1.2,
                        color=UNCORRELATED_COLOR,
                        alpha=0.85,
                        zorder=1,
                    )

            configure_knn_axis(ax)
            ax.set_xlim(COMMON_X_MIN, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(r"Node degree, $k$")

            if panel_index % n_columns == 0:
                ax.set_ylabel(
                    r"Mean nearest-neighbor degree, $k_{\rm nn}(k)$"
                )

            event_index = int(event["peak_event_index"])
            pre_peak_time = int(event["t"])
            next_peak_time = int(event["next_peak_time"])

            active_peak_fitness = (
                float(family_metadata["f_initial"])
                + max(0, event_index - 1)
                * float(family_metadata["peak_increment"])
            )

            letter = string.ascii_lowercase[panel_index]

            ax.set_title(
                rf"$\mathbf{{{letter}}}$ "
                rf"Pre-peak {event_index}: $t={pre_peak_time}$"
                "\n"
                rf"next peak at $t={next_peak_time}$; "
                rf"$f_{{\max}}={active_peak_fitness:g}$",
                loc="left",
            )

            if SHOW_EVENT_STATISTICS:
                n_networks = int(event_data["n_networks"].max())
                total_nodes = int(event_data["total_lcc_nodes"].max())
                r_mean = float(event_data["mean_assortativity"].max())
                r_std = float(event_data["std_assortativity"].max())

                annotation = (
                    rf"{n_networks} networks"
                    "\n"
                    rf"{total_nodes:,} LCC nodes"
                    "\n"
                    rf"$r={r_mean:.3f}\pm{r_std:.3f}$"
                )

                ax.text(
                    0.97,
                    0.96,
                    annotation,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.4,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.76,
                        "pad": 2.2,
                    },
                )

        next_panel = n_events

        if ADD_DIAGNOSTIC_PANEL:
            diagnostic_ax = axes_flat[next_panel]

            family_diagnostics = diagnostic_summary.copy()
            for column, value in family_metadata.items():
                family_diagnostics = family_diagnostics.loc[
                    np.isclose(
                        family_diagnostics[column].astype(float),
                        float(value),
                        atol=FLOAT_TOLERANCE,
                        rtol=0.0,
                    )
                ]

            plot_diagnostic_panel(
                diagnostic_ax,
                family_diagnostics,
            )
            next_panel += 1

        for unused_axis in axes_flat[next_panel:]:
            unused_axis.set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=4.8,
                color=SIMULATION_COLOR,
                label="Simulation",
            ),
            Line2D(
                [0],
                [0],
                linestyle="--",
                linewidth=1.65,
                color=MULTI_CENTRE_COLOR,
                label=THEORY_LABELS[MULTI_CENTRE_THEORY],
            ),
        ]

        if SHOW_SINGLE_CENTRE_IN_EVENT_PANELS:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=":",
                    linewidth=1.6,
                    color=SINGLE_CENTRE_COLOR,
                    label=THEORY_LABELS[SINGLE_CENTRE_THEORY],
                )
            )

        if SHOW_UNCORRELATED_REFERENCE:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-.",
                    linewidth=1.2,
                    color=UNCORRELATED_COLOR,
                    label=r"Uncorrelated $\langle k^2\rangle/\langle k\rangle$",
                )
            )

        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(4, len(legend_handles)),
        )

        fig.suptitle(
            "Evolution of pre-peak LCC degree correlations\n"
            + family_title(family_metadata),
            y=0.995,
            fontsize=13,
        )

        fig.subplots_adjust(
            top=0.82,
            bottom=0.11,
            left=0.075,
            right=0.985,
            hspace=0.40,
            wspace=0.25,
        )

        if SAVE_FIGS:
            output_dir = Path(OUTPUT_DIR).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            destination = output_dir / (
                f"{family_figure_tag(family_metadata)}."
                f"{FIGURE_FORMAT.lower().lstrip('.')}"
            )

            fig.savefig(
                destination,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
            saved_paths.append(destination)
            print(f"Saved figure: {destination}")

        if SHOW_FIGS:
            plt.show()

        plt.close(fig)

    return saved_paths


# ============================================================
# Notebook driver
# ============================================================


def validate_configuration() -> None:
    if GENERATIONS_BEFORE_PEAK < 1:
        raise ValueError("GENERATIONS_BEFORE_PEAK must be at least 1.")

    if MAX_PANEL_COLUMNS < 1:
        raise ValueError("MAX_PANEL_COLUMNS must be at least 1.")

    if FIGURE_FORMAT.lower().lstrip(".") not in {"pdf", "png", "svg"}:
        raise ValueError("FIGURE_FORMAT must be 'pdf', 'png', or 'svg'.")

    if ANCHOR_MODE.lower().strip() not in {
        "ancestral",
        "most_abundant_lcc",
    }:
        raise ValueError(
            "ANCHOR_MODE must be 'ancestral' or 'most_abundant_lcc'."
        )

    if THEORY_OCCUPANCY_SOURCE.lower().strip() not in {
        "population_poisson",
        "lcc_observed",
    }:
        raise ValueError(
            "THEORY_OCCUPANCY_SOURCE must be 'population_poisson' "
            "or 'lcc_observed'."
        )

    if DIAGNOSTIC_METRIC != "weighted_nrmse":
        raise ValueError(
            "DIAGNOSTIC_METRIC currently supports only 'weighted_nrmse'."
        )


def run_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    global MODEL_ROOT, NETWORK_ROOT, ABUNDANCE_ROOT, OUTPUT_DIR

    MODEL_ROOT = Path(MODEL_ROOT).expanduser().resolve()
    NETWORK_ROOT = Path(NETWORK_ROOT).expanduser().resolve()
    ABUNDANCE_ROOT = Path(ABUNDANCE_ROOT).expanduser().resolve()
    OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()

    validate_configuration()

    if DISPLAY_THEORY_DESCRIPTION and ADD_ANALYTICAL_THEORY:
        text = analytical_description_markdown()
        if Markdown is not None:
            display(Markdown(text))
        else:
            print(text)

    (
        network_inventory,
        empirical_knn,
        analytical_knn,
        class_statistics,
    ) = collect_selected_networks()

    aggregated_knn = aggregate_knn_curves(
        network_inventory,
        empirical_knn,
        analytical_knn,
    )

    theory_diagnostics = compute_network_theory_diagnostics(
        empirical_knn,
        analytical_knn,
    )
    diagnostic_summary = summarize_theory_diagnostics(
        theory_diagnostics
    )

    summary_table = (
        network_inventory
        .groupby(EVENT_COLUMNS, as_index=False, dropna=False)
        .agg(
            n_networks=("network", "nunique"),
            pooled_lcc_nodes=("lcc_nodes", "sum"),
            mean_lcc_nodes=("lcc_nodes", "mean"),
            mean_degree=("mean_lcc_degree", "mean"),
            maximum_degree=("maximum_lcc_degree", "max"),
            mean_uncorrelated_knn=("uncorrelated_knn", "mean"),
            mean_assortativity=("assortativity", "mean"),
            std_assortativity=("assortativity", "std"),
            mean_effective_centres=("n_effective_centres", "mean"),
        )
        .fillna({"std_assortativity": 0.0})
        .sort_values(
            FAMILY_COLUMNS + ["peak_event_index", "t"]
        )
        .reset_index(drop=True)
    )

    print("\nPre-peak k_nn summary:")
    try:
        display(summary_table)
    except Exception:
        print(summary_table.to_string(index=False))

    if not diagnostic_summary.empty:
        print("\nTheory-error summary:")
        try:
            display(diagnostic_summary)
        except Exception:
            print(diagnostic_summary.to_string(index=False))

    saved_paths = plot_pre_peak_knn_panels(
        aggregated_knn,
        diagnostic_summary,
    )

    print("\nAnalysis completed.")
    print(
        "Independent networks analyzed: "
        f"{network_inventory['network'].nunique()}"
    )
    print(
        "Pre-peak events analyzed: "
        f"{summary_table[['t', 'peak_event_index']].drop_duplicates().shape[0]}"
    )

    if SAVE_FIGS:
        print(f"Panel figures exported: {len(saved_paths)}")
    else:
        print("SAVE_FIGS = False: no files were exported.")

    return (
        network_inventory,
        empirical_knn,
        analytical_knn,
        class_statistics,
        aggregated_knn,
        theory_diagnostics,
        diagnostic_summary,
        summary_table,
    )


# ============================================================
# Execute in Jupyter
# ============================================================

(
    network_inventory,
    empirical_knn,
    analytical_knn,
    class_statistics,
    aggregated_knn,
    theory_diagnostics,
    diagnostic_summary,
    summary_table,
) = run_analysis()