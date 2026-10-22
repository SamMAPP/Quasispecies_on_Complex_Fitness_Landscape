#!/usr/bin/env python3
"""
fig_topology_degree.py -- Fig. "model4a30_pre_peak_LCC_degree_panel_multicentre_..." (Chapter 4, Sec. 4.3.3)

LCC degree distributions of the multi-peak genotype networks immediately before successive peak
introductions (t = 39, 79, 119, 159, 199; N=1e5, L=250, q=0, F0=3, Delta_f=1, tau=40, delta=5;
10 networks pooled per time), compared with
  * the multicentre shell construction: every LCC node is assigned to its nearest centre
    (Eq. model4a30-voronoi-assignment), the degree distribution of each local shell (alpha, m) is the
    single-peak shell construction with occupations p_{alpha m} = 1 - exp[-n_{alpha m}/C(L,m)],
    convolved with the observed cross-domain degrees, and weighted by S_{alpha m}/S_LCC
    (Eqs. model4a30-cross-domain-convolution, model4a30-multicentre-pk);
  * the single-centre baseline; panel (f) gives the Jensen-Shannon divergences.
Inputs: <data>/Networks/<stem>_t_<t>.graphml (common/build_networks.py) and <data>/Abundances/.
Port of the thesis notebook 16-Simulations/old/plots-multipeak-topology.ipynb (cell 1).

    python fig_topology_degree.py --data Data/multi_peak_delta5 --out Figures
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
Jupyter-ready analysis of pre-peak LCC degree distributions for model4a30.

For every pre-peak snapshot t = e*tau - GENERATIONS_BEFORE_PEAK:

1. discover all matching GraphML networks;
2. extract the largest connected component (LCC);
3. pool LCC node degrees across independent simulations;
4. calculate P(k) without producing CCDFs;
5. compare a one-centre shell baseline with a conditioned effective
   multi-centre Voronoi-shell approximation;
6. place all pre-peak events in one thesis-ready panel;
7. use the final panel to quantify theory error through the
   Jensen--Shannon divergence across independent simulations.

No tables or intermediate data are exported. Figures are written only when
SAVE_FIGS = True.

Supported network layouts
-------------------------
The discovery code accepts, among others:

    Data/model4a30/Networks/<snapshot>/network.graphml
    Data/model4a30/Networks/<snapshot>/network.graphml.gz
    Data/model4a30/Networks/<snapshot>.graphml
    Data/model4a30/Networks/<snapshot>.graphml.gz

The matching abundance snapshot is expected at:

    Data/model4a30/Abundances/<snapshot>.txt
"""

from collections import Counter
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
# None accepts every available value. A scalar or iterable selects one or
# several values.

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

# These selectors refer specifically to pre-peak snapshots.
TARGET_PRE_PEAK_TIMES = None
TARGET_PEAK_EVENTS = None

# t = e*tau - GENERATIONS_BEFORE_PEAK.
GENERATIONS_BEFORE_PEAK = 1

# ------------------------------------------------------------
# Analytical approximations
# ------------------------------------------------------------

ADD_ANALYTICAL_THEORY = True

# Compute the original one-centre theory as a diagnostic baseline.
COMPUTE_SINGLE_CENTRE_BASELINE = True

# Compute the corrected effective multi-centre approximation.
COMPUTE_MULTI_CENTRE_THEORY = True

# Show the baseline in the five distribution panels. It remains available in
# the sixth diagnostic panel even when hidden from the event panels.
SHOW_SINGLE_CENTRE_IN_EVENT_PANELS = False

# The one-centre reference is the most abundant genotype represented in the
# LCC. The ancestral genotype can be selected instead if desired.
#
# Alternatives:
#   "ancestral"
#   "most_abundant_lcc"
ANCHOR_MODE = "most_abundant_lcc"

# Occupancy probabilities within a shell:
#
# "population_poisson":
#     p_m = 1 - exp[-n_m / C(L,m)]
#
# "lcc_observed":
#     p_m = S_m / C(L,m)
#
# The population-based choice is the more predictive option.
THEORY_OCCUPANCY_SOURCE = "population_poisson"

# Since only LCC nodes are analyzed, final total-degree PMFs are conditioned on
# k >= 1.
CONDITION_THEORY_ON_POSITIVE_DEGREE = True

# Place each selected centre at its observed total LCC degree.
CONDITION_CENTRE_DEGREE = True

# Condition the local first-shell mean on the observed incidence between local
# shells m=1 and m=2.
CONDITION_FIRST_SHELL_INCIDENCE = True

# Multi-centre correction:
# before pre-peak event e, the model contains e explicit fitness peaks. Because
# the current files do not store their exact sequences, effective centres are
# inferred as abundance local maxima in the LCC.
EXPECTED_CENTRES_MODE = "event_index"

# Keep the ancestral peak among the effective centres whenever it belongs to
# the LCC and more than one centre is requested.
INCLUDE_ANCESTRAL_CENTRE = True

# Minimum Hamming separation between inferred centres. None uses
# max(2, delta//2), which is deliberately less restrictive than delta because
# non-consecutive explicit peaks may be closer than delta.
EFFECTIVE_CENTRE_MIN_DISTANCE = None

# Include observed cross-domain bridge incidences. For each Voronoi domain and
# shell, the local shell PMF is convolved with the empirical distribution of
# cross-domain degrees. This conditions on boundary connectivity without
# fitting the internal degree distribution.
CONDITION_CROSS_DOMAIN_DEGREES = True

DISPLAY_THEORY_DESCRIPTION = True

SINGLE_CENTRE_THEORY = "single_centre"
MULTI_CENTRE_THEORY = "multi_centre"

THEORY_LABELS = {
    SINGLE_CENTRE_THEORY: "Single-centre shell theory",
    MULTI_CENTRE_THEORY: "Multi-centre shell theory",
}

# ------------------------------------------------------------
# Plotting and output
# ------------------------------------------------------------

LOG_X = True
LOG_Y = True

SHOW_FIGS = False
SAVE_FIGS = True
FIGURE_FORMAT = "pdf"       # "pdf", "png", or "svg"
FIGURE_DPI = 600

# The intended thesis figure contains five pre-peak distributions and one
# diagnostic panel. The code remains valid for other event counts.
MAX_PANEL_COLUMNS = 3
PANEL_WIDTH = 4.25
PANEL_HEIGHT = 3.65
ADD_DIAGNOSTIC_PANEL = True

# Common limits make the evolution across events directly comparable.
COMMON_X_MIN = 0.9
COMMON_X_MAX = None         # None -> 1.12 * max(L, observed k_max)
COMMON_Y_MIN = 1e-6
COMMON_Y_MAX = 1.1

SHOW_EVENT_STATISTICS = True
SHOW_THEORY_IN_LEGEND = True

# Sixth panel: mean Jensen--Shannon divergence across independent networks.
# The uncertainty bars are the standard deviation across realizations.
DIAGNOSTIC_METRIC = "jensen_shannon"

FLOAT_TOLERANCE = 1e-12

SIMULATION_COLOR = "#1f77b4"
MULTI_CENTRE_COLOR = "#e67e22"
SINGLE_CENTRE_COLOR = "#7f7f7f"

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
# Filename parsing and selection
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
    """Return the meaningful snapshot stem for direct or nested GraphML files."""
    name = path.name

    for suffix in (".gz", ".graphml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    generic_names = {
        "network",
        "graph",
        "network_full",
        "full_network",
    }

    if name in generic_names:
        return path.parent.name

    return name


def parse_snapshot_metadata(network_file: Path) -> dict | None:
    """Parse model4a30 parameters from a GraphML path."""
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
    """
    Prefer direct snapshot GraphML files, then generic full-network files.

    LCC-specific GraphML files are deliberately excluded during discovery
    because the LCC is recomputed consistently inside this notebook.
    """
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
    """Discover parseable full-network GraphML snapshots recursively."""
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

    parsed_rows: list[dict] = []

    for path in candidates:
        priority, _ = network_priority(path)
        if priority >= 99:
            continue

        metadata = parse_snapshot_metadata(path)
        if metadata is None:
            continue

        metadata["_priority"] = priority
        parsed_rows.append(metadata)

    if not parsed_rows:
        return pd.DataFrame()

    inventory = pd.DataFrame(parsed_rows)

    # Avoid counting both a direct GraphML and a nested generic network file
    # for the same physical snapshot directory.
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

    inventory = (
        inventory
        .sort_values(dedupe_columns + ["_priority", "network_file"])
        .drop_duplicates(dedupe_columns, keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    return inventory


# ============================================================
# Genotype and abundance helpers
# ============================================================


def parse_genotype_name(name: str) -> tuple[int, ...]:
    """Convert '--' or a comma-separated locus list into a canonical tuple."""
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
    """Obtain the stored comma-separated genotype representation."""
    for key in ("genotype", "label", "name"):
        value = attributes.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return str(node).strip()


def genotype_distance(
    genotype_a: tuple[int, ...],
    genotype_b: tuple[int, ...],
) -> int:
    """Binary Hamming distance between two mutated-locus tuples."""
    return len(set(genotype_a).symmetric_difference(genotype_b))


def resolve_abundance_file(metadata: dict) -> Path | None:
    """Locate the original abundance snapshot corresponding to one network."""
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

    model_root = Path(MODEL_ROOT).expanduser().resolve()
    matches = [
        path
        for path in model_root.rglob(f"{stem}.txt")
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
    """Read one abundance snapshot as genotype -> number of individuals."""
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
    """Choose the shell reference according to ANCHOR_MODE."""
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
# Graph reading and shell measurements
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


def shell_edge_counts(
    graph: nx.Graph,
    shell_by_node: dict,
) -> Counter[tuple[int, int]]:
    """Count LCC edges between Hamming shells."""
    counts: Counter[tuple[int, int]] = Counter()

    for source, target in graph.edges():
        shell_a = int(shell_by_node[source])
        shell_b = int(shell_by_node[target])
        pair = tuple(sorted((shell_a, shell_b)))
        counts[pair] += 1

    return counts


# ============================================================
# Shell-mixture analytical approximation
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
    """Return p_m under the configured shell occupancy approximation."""
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
    """Stable Binomial(n,p) probability mass function."""
    if n < 0:
        raise ValueError("The number of binomial trials cannot be negative.")

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


def positive_degree_condition(pmf: np.ndarray) -> np.ndarray:
    """Condition a degree PMF on k >= 1."""
    result = np.asarray(pmf, dtype=float).copy()

    if not CONDITION_THEORY_ON_POSITIVE_DEGREE:
        return result

    if result.size == 0:
        return result

    result[0] = 0.0
    total = result.sum()

    if total <= 0.0:
        return result

    return result / total


def shell_degree_pmf(
    *,
    shell: int,
    L: int,
    occupancy_probabilities: np.ndarray,
    condition_positive: bool = True,
) -> np.ndarray:
    """
    First-order local shell prediction:

        K_m = X_m^- + X_m^+,

        X_m^- ~ Binomial(m, p_{m-1}),
        X_m^+ ~ Binomial(L-m, p_{m+1}).
    """
    m = int(shell)

    p_minus = (
        float(occupancy_probabilities[m - 1])
        if m > 0
        else 0.0
    )
    p_plus = (
        float(occupancy_probabilities[m + 1])
        if m < L
        else 0.0
    )

    lower = binomial_pmf(m, p_minus)
    upper = binomial_pmf(L - m, p_plus)

    pmf = np.convolve(lower, upper)

    if len(pmf) != L + 1:
        raise RuntimeError(
            f"Expected a degree PMF of length {L + 1}, received {len(pmf)}."
        )

    pmf = np.clip(pmf, 0.0, None)
    total = pmf.sum()

    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError(f"Invalid P(k|m={m}).")

    pmf = pmf / total
    return positive_degree_condition(pmf) if condition_positive else pmf


def first_shell_incidence_pmf(
    *,
    L: int,
    shell1_nodes: int,
    shell12_edges: int,
    centre_is_present: bool,
    condition_positive: bool = True,
) -> np.ndarray:
    """
    Condition the local m=1 mean degree on observed m=1--m=2 incidence.

    Every represented first-shell node has one guaranteed internal edge when
    its centre belongs to the domain. Its remaining L-1 potential internal
    edges point to the local second shell.
    """
    pmf = np.zeros(L + 1, dtype=float)

    if shell1_nodes <= 0:
        return pmf

    guaranteed_inner = 1 if centre_is_present else 0
    possible_outward_per_node = max(0, L - 1)

    if possible_outward_per_node == 0:
        pmf[guaranteed_inner] = 1.0
        return positive_degree_condition(pmf) if condition_positive else pmf

    outward_probability = shell12_edges / (
        shell1_nodes * possible_outward_per_node
    )
    outward_probability = float(np.clip(outward_probability, 0.0, 1.0))

    outward_pmf = binomial_pmf(
        possible_outward_per_node,
        outward_probability,
    )

    stop = guaranteed_inner + len(outward_pmf)
    pmf[guaranteed_inner:stop] = outward_pmf

    return positive_degree_condition(pmf) if condition_positive else pmf


def empirical_integer_pmf(values, *, maximum: int) -> np.ndarray:
    """Return the empirical PMF of non-negative integer values."""
    values = np.asarray(list(values), dtype=int)

    pmf = np.zeros(maximum + 1, dtype=float)
    if values.size == 0:
        pmf[0] = 1.0
        return pmf

    if np.any(values < 0):
        raise ValueError("Empirical degrees cannot be negative.")

    clipped = np.minimum(values, maximum)
    counts = np.bincount(clipped, minlength=maximum + 1).astype(float)
    total = counts.sum()

    if total <= 0.0:
        pmf[0] = 1.0
        return pmf

    return counts / total


def convolve_degree_pmfs(
    internal_pmf: np.ndarray,
    bridge_pmf: np.ndarray,
    *,
    maximum_degree: int,
    condition_positive: bool,
) -> np.ndarray:
    """
    Convolve local and cross-domain degree contributions.

    Numerical mass above the binary-space maximum L is accumulated at k=L.
    """
    convolution = np.convolve(
        np.asarray(internal_pmf, dtype=float),
        np.asarray(bridge_pmf, dtype=float),
    )

    result = np.zeros(maximum_degree + 1, dtype=float)
    keep = min(len(convolution), maximum_degree + 1)
    result[:keep] = convolution[:keep]

    if len(convolution) > maximum_degree + 1:
        result[maximum_degree] += convolution[maximum_degree + 1 :].sum()

    result = np.clip(result, 0.0, None)
    total = result.sum()

    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("Invalid convolved degree PMF.")

    result /= total
    return positive_degree_condition(result) if condition_positive else result


def format_genotype(genotype: tuple[int, ...]) -> str:
    return "--" if not genotype else ",".join(map(str, genotype))


def expected_number_of_centres(metadata: dict) -> int:
    """Number of explicit peaks present immediately before event e."""
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
    """
    Infer effective fitness centres from LCC abundance local maxima.

    The number requested equals the number of explicit peaks present before the
    next event. This is a conditioned reconstruction because the simulation
    files do not store the exact peak sequences.
    """
    genotype_to_node = {
        genotype: node
        for node, genotype in node_genotype.items()
    }
    lcc_genotypes = list(genotype_to_node)

    if not lcc_genotypes:
        raise ValueError("Cannot infer centres from an empty LCC.")

    requested = min(
        expected_number_of_centres(metadata),
        len(lcc_genotypes),
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
        lcc_genotypes,
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

    selected: list[tuple[int, ...]] = []

    # The most abundant LCC genotype is always the first effective centre.
    selected.append(ranked_all[0])

    # The ancestral genotype is a known explicit peak and is retained whenever
    # more than one centre is needed and it belongs to the LCC.
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

    # First pass: abundance local maxima with the configured separation.
    for genotype in ranked_candidates:
        if len(selected) >= requested:
            break
        if can_add(genotype, minimum_distance):
            selected.append(genotype)

    # Second pass: all represented genotypes with the same separation.
    for genotype in ranked_all:
        if len(selected) >= requested:
            break
        if can_add(genotype, minimum_distance):
            selected.append(genotype)

    # Final relaxation: preserve the requested number of centres.
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
    """Voronoi assignment with abundance-based tie breaking."""
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
) -> dict[int, list]:
    centre_abundances = [
        int(abundance_counts.get(centre, 0))
        for centre in centres
    ]

    domains = {index: [] for index in range(len(centres))}

    for node, genotype in node_genotype.items():
        domain = assign_to_nearest_centre(
            genotype,
            centres,
            centre_abundances,
        )
        domains[domain].append(node)

    return domains


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


def domain_shell_theory(
    *,
    lcc: nx.Graph,
    domain_nodes: list,
    node_genotype: dict,
    population_counts: Counter[tuple[int, ...]],
    centre: tuple[int, ...],
    L: int,
    domain_id: int,
    theory_type: str,
    metadata: dict,
) -> tuple[np.ndarray, list[dict], dict]:
    """
    Build one conditioned shell theory inside a Voronoi domain.

    Internal shell degrees are predicted analytically. When enabled, the PMF is
    convolved with the observed cross-domain degree distribution of the same
    shell. This uses only boundary incidence, not the internal P(k) shape.
    """
    expected_counts = np.zeros(L + 1, dtype=float)
    shell_rows: list[dict] = []

    if not domain_nodes:
        return expected_counts, shell_rows, {
            "domain_nodes": 0,
            "centre_degree": np.nan,
        }

    domain_graph = lcc.subgraph(domain_nodes).copy()

    shell_by_node = {
        node: genotype_distance(node_genotype[node], centre)
        for node in domain_nodes
    }

    lcc_shell_counts = np.zeros(L + 1, dtype=np.int64)
    for shell in shell_by_node.values():
        lcc_shell_counts[shell] += 1

    population_shell_counts = np.zeros(L + 1, dtype=np.int64)
    for genotype, abundance in population_counts.items():
        shell = genotype_distance(genotype, centre)
        if shell <= L:
            population_shell_counts[shell] += int(abundance)

    internal_edge_shell_counts = shell_edge_counts(
        domain_graph,
        shell_by_node,
    )

    genotype_to_node = {
        node_genotype[node]: node
        for node in domain_nodes
    }
    centre_node = genotype_to_node.get(centre)
    centre_degree = (
        int(lcc.degree(centre_node))
        if centre_node is not None
        else None
    )

    occupancy_probabilities = shell_occupancy_probabilities(
        population_shell_counts=population_shell_counts,
        lcc_shell_counts=lcc_shell_counts,
        L=L,
    )

    cross_degrees_by_shell: dict[int, list[int]] = {
        shell: []
        for shell in range(L + 1)
    }

    for node in domain_nodes:
        shell = int(shell_by_node[node])
        cross_degree = int(
            lcc.degree(node) - domain_graph.degree(node)
        )
        cross_degrees_by_shell[shell].append(cross_degree)

    for shell in range(L + 1):
        shell_nodes = int(lcc_shell_counts[shell])
        population_count = int(population_shell_counts[shell])

        shell_rows.append({
            **metadata,
            "theory_type": theory_type,
            "domain_id": int(domain_id),
            "centre_genotype": format_genotype(centre),
            "centre_abundance": int(population_counts.get(centre, 0)),
            "centre_degree": centre_degree,
            "shell": int(shell),
            "population_shell_count": population_count,
            "domain_shell_nodes": shell_nodes,
            "domain_shell_weight": float(
                shell_nodes / len(domain_nodes)
            ),
            "global_lcc_shell_weight": float(
                shell_nodes / lcc.number_of_nodes()
            ),
            "occupancy_probability": float(
                occupancy_probabilities[shell]
            ),
            "mean_cross_domain_degree": (
                float(np.mean(cross_degrees_by_shell[shell]))
                if cross_degrees_by_shell[shell]
                else 0.0
            ),
        })

        if shell_nodes == 0:
            continue

        if (
            shell == 0
            and CONDITION_CENTRE_DEGREE
            and centre_degree is not None
        ):
            total_pmf = np.zeros(L + 1, dtype=float)
            total_pmf[centre_degree] = 1.0

        else:
            if (
                shell == 1
                and CONDITION_FIRST_SHELL_INCIDENCE
            ):
                internal_pmf = first_shell_incidence_pmf(
                    L=L,
                    shell1_nodes=shell_nodes,
                    shell12_edges=int(
                        internal_edge_shell_counts.get((1, 2), 0)
                    ),
                    centre_is_present=centre_node is not None,
                    condition_positive=False,
                )
            else:
                internal_pmf = shell_degree_pmf(
                    shell=shell,
                    L=L,
                    occupancy_probabilities=occupancy_probabilities,
                    condition_positive=False,
                )

            if CONDITION_CROSS_DOMAIN_DEGREES:
                bridge_pmf = empirical_integer_pmf(
                    cross_degrees_by_shell[shell],
                    maximum=L,
                )
                total_pmf = convolve_degree_pmfs(
                    internal_pmf,
                    bridge_pmf,
                    maximum_degree=L,
                    condition_positive=True,
                )
            else:
                total_pmf = positive_degree_condition(internal_pmf)

        expected_counts += shell_nodes * total_pmf

    diagnostics = {
        "domain_nodes": int(len(domain_nodes)),
        "centre_degree": centre_degree,
        "centre_abundance": int(population_counts.get(centre, 0)),
    }

    return expected_counts, shell_rows, diagnostics


def build_single_centre_theory(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
) -> tuple[np.ndarray, list[dict], dict]:
    lcc_genotypes = set(node_genotype.values())
    centre = choose_anchor_genotype(
        abundance_counts=abundance_counts,
        lcc_genotypes=lcc_genotypes,
    )

    expected_counts, shell_rows, diagnostics = domain_shell_theory(
        lcc=lcc,
        domain_nodes=list(lcc.nodes()),
        node_genotype=node_genotype,
        population_counts=abundance_counts,
        centre=centre,
        L=int(metadata["L"]),
        domain_id=0,
        theory_type=SINGLE_CENTRE_THEORY,
        metadata=metadata,
    )

    diagnostics.update({
        "n_effective_centres": 1,
        "effective_centres": format_genotype(centre),
        "primary_centre": centre,
    })

    return expected_counts, shell_rows, diagnostics


def build_multi_centre_theory(
    *,
    lcc: nx.Graph,
    node_genotype: dict,
    abundance_counts: Counter[tuple[int, ...]],
    metadata: dict,
) -> tuple[np.ndarray, list[dict], dict]:
    centres = select_effective_centres(
        lcc=lcc,
        node_genotype=node_genotype,
        abundance_counts=abundance_counts,
        metadata=metadata,
    )

    node_domains = partition_lcc_nodes(
        node_genotype=node_genotype,
        centres=centres,
        abundance_counts=abundance_counts,
    )
    population_domains = partition_population(
        abundance_counts=abundance_counts,
        centres=centres,
    )

    L = int(metadata["L"])
    expected_counts = np.zeros(L + 1, dtype=float)
    shell_rows: list[dict] = []
    domain_diagnostics: list[dict] = []

    for domain_id, centre in enumerate(centres):
        (
            domain_expected,
            domain_shell_rows,
            diagnostics,
        ) = domain_shell_theory(
            lcc=lcc,
            domain_nodes=node_domains[domain_id],
            node_genotype=node_genotype,
            population_counts=population_domains[domain_id],
            centre=centre,
            L=L,
            domain_id=domain_id,
            theory_type=MULTI_CENTRE_THEORY,
            metadata=metadata,
        )

        expected_counts += domain_expected
        shell_rows.extend(domain_shell_rows)
        domain_diagnostics.append(diagnostics)

    diagnostics = {
        "n_effective_centres": int(len(centres)),
        "effective_centres": " | ".join(
            format_genotype(centre)
            for centre in centres
        ),
        "primary_centre": centres[0],
        "domain_sizes": tuple(
            int(item["domain_nodes"])
            for item in domain_diagnostics
        ),
    }

    return expected_counts, shell_rows, diagnostics


def analytical_description_markdown() -> str:
    """Notebook explanation of the baseline and corrected theory."""
    occupancy_text = (
        r"$p_m=1-\exp[-n_m/\binom{L}{m}]$ from the abundance snapshot"
        if THEORY_OCCUPANCY_SOURCE.lower().strip() == "population_poisson"
        else
        r"$p_m=S_m/\binom{L}{m}$ from the represented LCC shells"
    )

    template = r"""
### Conditioned multi-centre shell approximation

The single-centre baseline measures Hamming shells around the most abundant
LCC genotype. Within each shell it uses __OCCUPANCY_TEXT__ and

$$
K_m=X_m^-+X_m^+,\qquad
X_m^-\sim\mathrm{Binomial}(m,p_{m-1}),\qquad
X_m^+\sim\mathrm{Binomial}(L-m,p_{m+1}).
$$

Before pre-peak event $e$, the dynamics contains $e$ explicit fitness peaks.
Their exact sequences are not stored in the current output, so the corrected
theory reconstructs $e$ effective centres as abundance local maxima in the
LCC. LCC nodes and population genotypes are assigned to their nearest centre,

$$
\alpha^*(g)=\arg\min_\alpha d_{\mathrm H}(g,g_\alpha),
$$

which defines mutational Voronoi domains. A conditioned shell distribution is
computed independently inside every domain and the complete prediction is

$$
P_{\mathrm{multi}}(k)
=
\sum_\alpha
\frac{S_\alpha}{S_{\mathrm{LCC}}}
\sum_m
\frac{S_{\alpha m}}{S_\alpha}
P_\alpha(k\mid m).
$$

The degree of every inferred centre is fixed at its observed LCC value. The
first local shell is conditioned on the observed $m=1$--$m=2$ incidence.
Finally, local shell PMFs are convolved with the observed distribution of
cross-domain bridge degrees. Thus, the approximation conditions on shell
occupation and domain-boundary incidence, but it does not fit the internal
degree distribution.

The sixth panel reports the Jensen--Shannon divergence between the empirical
and theoretical degree distributions. It compares the original one-centre
baseline with the corrected multi-centre construction across successive peak
events.
"""

    return template.replace("__OCCUPANCY_TEXT__", occupancy_text)


# ============================================================
# One-network analysis
# ============================================================


def analyze_one_network(
    network_file: Path,
    metadata: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """
    Analyze one network and construct both theoretical approximations.
    """
    graph = nx.read_graphml(network_file)
    graph = to_simple_undirected_graph(graph)
    lcc, n_components = largest_connected_component(graph)

    node_genotype: dict = {}
    genotype_to_node: dict[tuple[int, ...], object] = {}

    for node, attributes in lcc.nodes(data=True):
        name = genotype_name(node, attributes)
        genotype = parse_genotype_name(name)
        node_genotype[node] = genotype

        if genotype in genotype_to_node:
            raise ValueError(
                f"Repeated genotype {genotype} in GraphML {network_file}."
            )
        genotype_to_node[genotype] = node

    abundance_file = resolve_abundance_file(metadata)
    abundance_counts: Counter[tuple[int, ...]]

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
            f"Abundance file not found for {network_file}. "
            "Using GraphML node abundances when available.",
            RuntimeWarning,
        )

    degrees = dict(lcc.degree())
    empirical_counter = Counter(
        int(degree)
        for degree in degrees.values()
    )

    empirical_rows = [
        {
            **metadata,
            "degree": int(degree),
            "node_count": int(count),
        }
        for degree, count in sorted(empirical_counter.items())
    ]

    theory_rows: list[dict] = []
    shell_rows: list[dict] = []

    single_diagnostics = None
    multi_diagnostics = None

    if ADD_ANALYTICAL_THEORY and COMPUTE_SINGLE_CENTRE_BASELINE:
        (
            single_expected,
            single_shell_rows,
            single_diagnostics,
        ) = build_single_centre_theory(
            lcc=lcc,
            node_genotype=node_genotype,
            abundance_counts=abundance_counts,
            metadata=metadata,
        )

        shell_rows.extend(single_shell_rows)

        for degree, expected_count in enumerate(single_expected):
            theory_rows.append({
                **metadata,
                "theory_type": SINGLE_CENTRE_THEORY,
                "degree": int(degree),
                "expected_count": float(expected_count),
            })

    if ADD_ANALYTICAL_THEORY and COMPUTE_MULTI_CENTRE_THEORY:
        (
            multi_expected,
            multi_shell_rows,
            multi_diagnostics,
        ) = build_multi_centre_theory(
            lcc=lcc,
            node_genotype=node_genotype,
            abundance_counts=abundance_counts,
            metadata=metadata,
        )

        shell_rows.extend(multi_shell_rows)

        for degree, expected_count in enumerate(multi_expected):
            theory_rows.append({
                **metadata,
                "theory_type": MULTI_CENTRE_THEORY,
                "degree": int(degree),
                "expected_count": float(expected_count),
            })

    degree_array = np.fromiter(
        degrees.values(),
        dtype=np.int64,
        count=len(degrees),
    )

    primary_diagnostics = (
        multi_diagnostics
        if multi_diagnostics is not None
        else single_diagnostics
    )

    if primary_diagnostics is None:
        primary_centre = choose_anchor_genotype(
            abundance_counts=abundance_counts,
            lcc_genotypes=set(node_genotype.values()),
        )
        n_effective_centres = 1
        effective_centres = format_genotype(primary_centre)
    else:
        primary_centre = primary_diagnostics["primary_centre"]
        n_effective_centres = int(
            primary_diagnostics["n_effective_centres"]
        )
        effective_centres = str(
            primary_diagnostics["effective_centres"]
        )

    primary_node = genotype_to_node.get(primary_centre)
    primary_degree = (
        int(lcc.degree(primary_node))
        if primary_node is not None
        else np.nan
    )

    inventory_row = {
        **metadata,
        "abundance_file": (
            str(abundance_file)
            if abundance_file is not None
            else None
        ),
        "primary_centre": format_genotype(primary_centre),
        "primary_centre_abundance": int(
            abundance_counts.get(primary_centre, 0)
        ),
        "primary_centre_degree": primary_degree,
        "n_effective_centres": n_effective_centres,
        "effective_centres": effective_centres,
        "population_read": int(sum(abundance_counts.values())),
        "full_nodes": int(graph.number_of_nodes()),
        "full_edges": int(graph.number_of_edges()),
        "n_components": int(n_components),
        "lcc_nodes": int(lcc.number_of_nodes()),
        "lcc_edges": int(lcc.number_of_edges()),
        "lcc_node_fraction": float(
            lcc.number_of_nodes() / graph.number_of_nodes()
        ),
        "mean_lcc_degree": float(degree_array.mean()),
        "variance_lcc_degree": float(degree_array.var(ddof=0)),
        "maximum_lcc_degree": int(degree_array.max()),
    }

    return inventory_row, empirical_rows, theory_rows, shell_rows


# ============================================================
# Collection and aggregation
# ============================================================


def collect_selected_networks() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    inventory_catalog = discover_networks()

    if inventory_catalog.empty:
        raise RuntimeError(
            "No parseable model4a30 GraphML snapshots were found below:\n"
            f"  {Path(NETWORK_ROOT).expanduser().resolve()}"
        )

    selected_catalog = inventory_catalog.loc[
        inventory_catalog.apply(
            lambda row: metadata_is_selected(row.to_dict()),
            axis=1,
        )
    ].copy()

    print(f"Network root: {Path(NETWORK_ROOT).expanduser().resolve()}")
    print(f"Parseable GraphML snapshots: {len(inventory_catalog)}")
    print(
        "Available pre-peak snapshots: "
        f"{int(inventory_catalog['is_pre_peak'].sum())}"
    )
    print(f"Selection: {selector_description()}")
    print(f"Selected pre-peak snapshots: {len(selected_catalog)}")

    if selected_catalog.empty:
        preview_columns = [
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
            inventory_catalog.loc[
                inventory_catalog["is_pre_peak"],
                preview_columns,
            ]
            .head(40)
            .to_string(index=False)
        )

        raise RuntimeError(
            "No pre-peak networks matched the configured selectors.\n\n"
            f"Available examples:\n{preview}"
        )

    inventory_rows: list[dict] = []
    empirical_rows: list[dict] = []
    theory_rows: list[dict] = []
    shell_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for index, row in enumerate(
        selected_catalog.itertuples(index=False),
        start=1,
    ):
        metadata = row._asdict()
        network_file = Path(metadata["network_file"])

        try:
            (
                inventory_row,
                empirical,
                theory,
                shells,
            ) = analyze_one_network(network_file, metadata)
        except Exception as exc:
            failures.append(
                (
                    str(network_file),
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        inventory_rows.append(inventory_row)
        empirical_rows.extend(empirical)
        theory_rows.extend(theory)
        shell_rows.extend(shells)

        print(
            f"[{index:>4}/{len(selected_catalog)}] "
            f"event={metadata['peak_event_index']}, "
            f"t={metadata['t']}, "
            f"iteration={metadata['iteration']}: "
            f"LCC={inventory_row['lcc_nodes']:,}/"
            f"{inventory_row['full_nodes']:,}, "
            f"centres={inventory_row['n_effective_centres']}, "
            f"primary={inventory_row['primary_centre']}, "
            f"k_primary={inventory_row['primary_centre_degree']}"
        )

    if failures:
        print("\nSkipped networks:")
        for path, reason in failures:
            print(f"  {path}: {reason}")

    if not inventory_rows:
        raise RuntimeError("Every selected network failed during analysis.")

    network_inventory = (
        pd.DataFrame(inventory_rows)
        .sort_values(
            ["q", "delta", "t", "iteration", "network"]
        )
        .reset_index(drop=True)
    )

    empirical_degree_counts = (
        pd.DataFrame(empirical_rows)
        .sort_values(
            ["q", "delta", "t", "iteration", "network", "degree"]
        )
        .reset_index(drop=True)
    )

    analytical_expected_counts = (
        pd.DataFrame(theory_rows)
        if theory_rows
        else pd.DataFrame(
            columns=[*EVENT_COLUMNS, "theory_type", "degree", "expected_count"]
        )
    )

    if not analytical_expected_counts.empty:
        analytical_expected_counts = (
            analytical_expected_counts
            .sort_values(
                ["q", "delta", "t", "iteration", "network", "theory_type", "degree"]
            )
            .reset_index(drop=True)
        )

    shell_statistics = (
        pd.DataFrame(shell_rows)
        .sort_values(
            ["q", "delta", "t", "iteration", "network", "shell"]
        )
        .reset_index(drop=True)
    )

    return (
        network_inventory,
        empirical_degree_counts,
        analytical_expected_counts,
        shell_statistics,
    )


def aggregate_degree_distributions(
    network_inventory: pd.DataFrame,
    empirical_degree_counts: pd.DataFrame,
    analytical_expected_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Pool empirical and theoretical node counts across simulations."""
    empirical = (
        empirical_degree_counts
        .groupby(EVENT_COLUMNS + ["degree"], as_index=False, dropna=False)
        .agg(empirical_count=("node_count", "sum"))
    )

    totals = (
        network_inventory
        .groupby(EVENT_COLUMNS, as_index=False, dropna=False)
        .agg(
            n_networks=("network", "nunique"),
            total_lcc_nodes=("lcc_nodes", "sum"),
            mean_lcc_nodes=("lcc_nodes", "mean"),
            mean_degree=("mean_lcc_degree", "mean"),
            std_degree=("mean_lcc_degree", "std"),
            maximum_degree=("maximum_lcc_degree", "max"),
            mean_primary_centre_degree=("primary_centre_degree", "mean"),
            mean_primary_centre_abundance=(
                "primary_centre_abundance",
                "mean",
            ),
            mean_effective_centres=("n_effective_centres", "mean"),
        )
        .fillna({"std_degree": 0.0})
    )

    if (
        ADD_ANALYTICAL_THEORY
        and not analytical_expected_counts.empty
    ):
        theory = (
            analytical_expected_counts
            .groupby(
                EVENT_COLUMNS + ["theory_type", "degree"],
                as_index=False,
                dropna=False,
            )
            .agg(theory_expected_count=("expected_count", "sum"))
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

    merged = merged.merge(
        totals,
        on=EVENT_COLUMNS,
        how="left",
    )

    merged["empirical_count"] = merged["empirical_count"].fillna(0.0)
    merged["empirical_probability"] = (
        merged["empirical_count"] / merged["total_lcc_nodes"]
    )

    merged["theory_probability"] = np.where(
        merged["theory_expected_count"].notna(),
        merged["theory_expected_count"] / merged["total_lcc_nodes"],
        np.nan,
    )

    return (
        merged
        .sort_values(EVENT_COLUMNS + ["theory_type", "degree"])
        .reset_index(drop=True)
    )


def jensen_shannon_divergence(
    empirical_probability: np.ndarray,
    theory_probability: np.ndarray,
) -> float:
    """
    Jensen--Shannon divergence in bits.

    The result lies in [0,1] for two probability distributions.
    """
    p = np.asarray(empirical_probability, dtype=float)
    q = np.asarray(theory_probability, dtype=float)

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return np.nan

    p /= p.sum()
    q /= q.sum()
    mixture = 0.5 * (p + q)

    def kl_divergence(left, right):
        mask = left > 0.0
        return float(
            np.sum(
                left[mask]
                * np.log2(left[mask] / right[mask])
            )
        )

    value = 0.5 * (
        kl_divergence(p, mixture)
        + kl_divergence(q, mixture)
    )
    return float(np.clip(value, 0.0, 1.0))


def total_variation_distance(
    empirical_probability: np.ndarray,
    theory_probability: np.ndarray,
) -> float:
    p = np.asarray(empirical_probability, dtype=float)
    q = np.asarray(theory_probability, dtype=float)

    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return np.nan

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p /= p.sum()
    q /= q.sum()

    return float(0.5 * np.abs(p - q).sum())


def compute_network_theory_diagnostics(
    empirical_degree_counts: pd.DataFrame,
    analytical_expected_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate theory errors separately for every independent realization.
    """
    if analytical_expected_counts.empty:
        return pd.DataFrame()

    network_columns = [
        *EVENT_COLUMNS,
        "network",
        "iteration",
    ]

    empirical_lookup = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in empirical_degree_counts.groupby(
            network_columns,
            sort=False,
            dropna=False,
        )
    }

    rows: list[dict] = []

    for group_key, theory_group in analytical_expected_counts.groupby(
        network_columns + ["theory_type"],
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

        maximum_degree = int(
            max(
                empirical_group["degree"].max(),
                theory_group["degree"].max(),
            )
        )

        empirical_vector = np.zeros(maximum_degree + 1, dtype=float)
        theory_vector = np.zeros(maximum_degree + 1, dtype=float)

        empirical_vector[
            empirical_group["degree"].to_numpy(dtype=int)
        ] = empirical_group["node_count"].to_numpy(dtype=float)

        theory_vector[
            theory_group["degree"].to_numpy(dtype=int)
        ] = theory_group["expected_count"].to_numpy(dtype=float)

        metadata = dict(zip(network_columns, metadata_values))
        rows.append({
            **metadata,
            "theory_type": theory_type,
            "jensen_shannon": jensen_shannon_divergence(
                empirical_vector,
                theory_vector,
            ),
            "total_variation": total_variation_distance(
                empirical_vector,
                theory_vector,
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            EVENT_COLUMNS + ["theory_type", "iteration", "network"]
        )
        .reset_index(drop=True)
    )


def summarize_theory_diagnostics(
    theory_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if theory_diagnostics.empty:
        return pd.DataFrame()

    return (
        theory_diagnostics
        .groupby(
            EVENT_COLUMNS + ["theory_type"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_networks=("network", "nunique"),
            mean_jensen_shannon=("jensen_shannon", "mean"),
            std_jensen_shannon=("jensen_shannon", "std"),
            mean_total_variation=("total_variation", "mean"),
            std_total_variation=("total_variation", "std"),
        )
        .fillna({
            "std_jensen_shannon": 0.0,
            "std_total_variation": 0.0,
        })
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
        f"{MODEL}_pre_peak_LCC_degree_panel_multicentre"
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


def configure_degree_axis(
    ax: plt.Axes,
    *,
    x_limits: tuple[float, float],
) -> None:
    if LOG_X:
        ax.set_xscale("log")
    if LOG_Y:
        ax.set_yscale("log")

    ax.set_xlim(*x_limits)
    ax.set_ylim(COMMON_Y_MIN, COMMON_Y_MAX)

    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    if LOG_X or LOG_Y:
        ax.grid(True, which="minor", linewidth=0.42, alpha=0.08)


def valid_curve(data: pd.DataFrame, column: str) -> pd.DataFrame:
    curve = data.loc[
        np.isfinite(data[column])
        & (data[column] > 0)
    ].copy()

    if LOG_X:
        curve = curve.loc[curve["degree"] > 0].copy()

    return curve.sort_values("degree")


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


def diagnostic_columns() -> tuple[str, str, str]:
    metric = DIAGNOSTIC_METRIC.lower().strip()

    if metric == "jensen_shannon":
        return (
            "mean_jensen_shannon",
            "std_jensen_shannon",
            r"Jensen--Shannon divergence, $D_{\rm JS}$",
        )

    if metric == "total_variation":
        return (
            "mean_total_variation",
            "std_total_variation",
            r"Total variation distance, $D_{\rm TV}$",
        )

    raise ValueError(
        "DIAGNOSTIC_METRIC must be 'jensen_shannon' or "
        "'total_variation'."
    )


def plot_diagnostic_panel(
    ax: plt.Axes,
    *,
    family_diagnostics: pd.DataFrame,
    panel_letter: str,
) -> None:
    mean_column, std_column, ylabel = diagnostic_columns()

    style_by_theory = {
        SINGLE_CENTRE_THEORY: {
            "color": SINGLE_CENTRE_COLOR,
            "marker": "o",
            "linestyle": ":",
            "label": THEORY_LABELS[SINGLE_CENTRE_THEORY],
        },
        MULTI_CENTRE_THEORY: {
            "color": MULTI_CENTRE_COLOR,
            "marker": "s",
            "linestyle": "--",
            "label": THEORY_LABELS[MULTI_CENTRE_THEORY],
        },
    }

    for theory_type in (
        SINGLE_CENTRE_THEORY,
        MULTI_CENTRE_THEORY,
    ):
        data = (
            family_diagnostics.loc[
                family_diagnostics["theory_type"] == theory_type
            ]
            .sort_values("peak_event_index")
            .copy()
        )

        if data.empty:
            continue

        style = style_by_theory[theory_type]

        ax.errorbar(
            data["peak_event_index"],
            data[mean_column],
            yerr=data[std_column],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            markersize=4.8,
            capsize=2.5,
            elinewidth=0.9,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlabel("Pre-peak event")
    ax.set_ylabel(ylabel)
    ax.set_title(
        rf"$\mathbf{{{panel_letter}}}$ Theory error",
        loc="left",
    )
    ax.grid(True, which="major", linewidth=0.7, alpha=0.22)
    ax.set_axisbelow(True)

    if not family_diagnostics.empty:
        events = sorted(
            int(value)
            for value in family_diagnostics["peak_event_index"].unique()
        )
        ax.set_xticks(events)
        ax.set_xlim(min(events) - 0.35, max(events) + 0.35)

        values = family_diagnostics[mean_column].to_numpy(dtype=float)
        errors = family_diagnostics[std_column].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(errors)

        if np.any(finite):
            upper = float(np.max(values[finite] + errors[finite]))
            ax.set_ylim(0.0, max(0.02, 1.18 * upper))

    ax.text(
        0.97,
        0.96,
        "lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )


def plot_pre_peak_degree_panels(
    aggregated_distributions: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
) -> list[Path]:
    """
    Produce one thesis-ready panel per parameter family.

    Event panels show P(k) at each pre-peak time. The final panel quantifies
    the error of the one-centre and corrected multi-centre theories.
    """
    saved_paths: list[Path] = []

    for family_key, family_data in aggregated_distributions.groupby(
        FAMILY_COLUMNS,
        sort=True,
        dropna=False,
    ):
        if not isinstance(family_key, tuple):
            family_key = (family_key,)

        family_metadata = dict(zip(FAMILY_COLUMNS, family_key))

        event_rows = (
            family_data[
                [
                    "t",
                    "next_peak_time",
                    "peak_event_index",
                ]
            ]
            .drop_duplicates()
            .sort_values(["peak_event_index", "t"])
            .reset_index(drop=True)
        )

        family_diagnostics = (
            diagnostic_summary.copy()
            if diagnostic_summary.empty
            else diagnostic_summary.loc[
                np.logical_and.reduce([
                    np.isclose(
                        diagnostic_summary[column].astype(float),
                        float(family_metadata[column]),
                        atol=FLOAT_TOLERANCE,
                        rtol=0.0,
                    )
                    for column in FAMILY_COLUMNS
                ])
            ].copy()
        )

        n_events = len(event_rows)
        n_panels = n_events + (
            1 if ADD_DIAGNOSTIC_PANEL else 0
        )
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

        maximum_observed_degree = int(
            family_data["maximum_degree"].max()
        )
        sequence_length = int(family_metadata["L"])
        x_maximum = (
            float(COMMON_X_MAX)
            if COMMON_X_MAX is not None
            else 1.12 * max(sequence_length, maximum_observed_degree)
        )
        common_x_limits = (float(COMMON_X_MIN), x_maximum)

        for panel_index, event in event_rows.iterrows():
            ax = axes_flat[panel_index]

            event_data = family_data.loc[
                (family_data["t"] == event["t"])
                & (
                    family_data["peak_event_index"]
                    == event["peak_event_index"]
                )
            ].copy()

            empirical_source = (
                event_data[
                    [
                        "degree",
                        "empirical_probability",
                        "n_networks",
                        "total_lcc_nodes",
                        "mean_degree",
                        "maximum_degree",
                    ]
                ]
                .drop_duplicates("degree")
                .sort_values("degree")
            )
            empirical = valid_curve(
                empirical_source,
                "empirical_probability",
            )

            single_theory = valid_curve(
                event_data.loc[
                    event_data["theory_type"]
                    == SINGLE_CENTRE_THEORY
                ],
                "theory_probability",
            )
            multi_theory = valid_curve(
                event_data.loc[
                    event_data["theory_type"]
                    == MULTI_CENTRE_THEORY
                ],
                "theory_probability",
            )

            ax.plot(
                empirical["degree"],
                empirical["empirical_probability"],
                linestyle="none",
                marker="o",
                markersize=4.25,
                color=SIMULATION_COLOR,
                label="Simulation",
                zorder=5,
            )

            if (
                SHOW_SINGLE_CENTRE_IN_EVENT_PANELS
                and not single_theory.empty
            ):
                ax.plot(
                    single_theory["degree"],
                    single_theory["theory_probability"],
                    linestyle=":",
                    linewidth=1.35,
                    color=SINGLE_CENTRE_COLOR,
                    label=THEORY_LABELS[SINGLE_CENTRE_THEORY],
                    zorder=2,
                )

            if not multi_theory.empty:
                ax.plot(
                    multi_theory["degree"],
                    multi_theory["theory_probability"],
                    linestyle="--",
                    linewidth=1.7,
                    color=MULTI_CENTRE_COLOR,
                    label=THEORY_LABELS[MULTI_CENTRE_THEORY],
                    zorder=4,
                )

            configure_degree_axis(
                ax,
                x_limits=common_x_limits,
            )

            row_index = panel_index // n_columns
            column_index = panel_index % n_columns

            if row_index == n_rows - 1:
                ax.set_xlabel(r"Node degree, $k$")
            if column_index == 0:
                ax.set_ylabel(r"LCC degree probability, $P(k)$")

            event_index = int(event["peak_event_index"])
            pre_peak_time = int(event["t"])
            next_peak_time = int(event["next_peak_time"])

            active_peak_fitness = (
                float(family_metadata["f_initial"])
                + max(0, event_index - 1)
                * float(family_metadata["peak_increment"])
            )

            panel_letter = string.ascii_lowercase[panel_index]
            ax.set_title(
                rf"$\mathbf{{{panel_letter}}}$ "
                rf"Pre-peak {event_index}: $t={pre_peak_time}$"
                "\n"
                rf"next peak at $t={next_peak_time}$; "
                rf"$f_{{\max}}={active_peak_fitness:g}$",
                loc="left",
            )

            if SHOW_EVENT_STATISTICS:
                n_networks = int(
                    empirical_source["n_networks"].max()
                )
                total_nodes = int(
                    empirical_source["total_lcc_nodes"].max()
                )
                mean_degree = float(
                    empirical_source["mean_degree"].max()
                )

                annotation = (
                    rf"{n_networks} networks"
                    "\n"
                    rf"{total_nodes:,} LCC nodes"
                    "\n"
                    rf"$\langle k\rangle={mean_degree:.2f}$"
                )

                ax.text(
                    0.97,
                    0.95,
                    annotation,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.35,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 2.0,
                    },
                )

        next_axis_index = n_events

        if ADD_DIAGNOSTIC_PANEL:
            diagnostic_axis = axes_flat[next_axis_index]
            plot_diagnostic_panel(
                diagnostic_axis,
                family_diagnostics=family_diagnostics,
                panel_letter=string.ascii_lowercase[next_axis_index],
            )
            next_axis_index += 1

        for unused_axis in axes_flat[next_axis_index:]:
            unused_axis.set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=SIMULATION_COLOR,
                markersize=4.8,
                label="Simulation",
            ),
        ]

        if SHOW_THEORY_IN_LEGEND:
            if COMPUTE_MULTI_CENTRE_THEORY:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="--",
                        color=MULTI_CENTRE_COLOR,
                        linewidth=1.7,
                        label=THEORY_LABELS[MULTI_CENTRE_THEORY],
                    )
                )

            if COMPUTE_SINGLE_CENTRE_BASELINE:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle=":",
                        color=SINGLE_CENTRE_COLOR,
                        linewidth=1.35,
                        label=THEORY_LABELS[SINGLE_CENTRE_THEORY],
                    )
                )

        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=len(legend_handles),
        )

        fig.suptitle(
            "Evolution of pre-peak LCC degree distributions\n"
            + family_title(family_metadata),
            y=0.995,
            fontsize=13.2,
        )

        fig.subplots_adjust(
            top=0.82,
            bottom=0.10,
            left=0.073,
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

    if EXPECTED_CENTRES_MODE.lower().strip() != "event_index":
        raise ValueError(
            "EXPECTED_CENTRES_MODE currently supports only 'event_index'."
        )

    if not (
        COMPUTE_SINGLE_CENTRE_BASELINE
        or COMPUTE_MULTI_CENTRE_THEORY
    ):
        raise ValueError(
            "At least one analytical theory must be enabled."
        )

    diagnostic_columns()

    if LOG_X and COMMON_X_MIN <= 0:
        raise ValueError("COMMON_X_MIN must be positive for LOG_X=True.")

    if LOG_Y and COMMON_Y_MIN <= 0:
        raise ValueError("COMMON_Y_MIN must be positive for LOG_Y=True.")

    if COMMON_Y_MAX <= COMMON_Y_MIN:
        raise ValueError("COMMON_Y_MAX must exceed COMMON_Y_MIN.")


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
    """
    Run the complete thesis-ready pre-peak LCC analysis.

    Returns
    -------
    network_inventory
        One row per independent GraphML network.

    empirical_degree_counts
        Sparse observed degree counts for every independent network.

    analytical_expected_counts
        Expected degree counts for both theoretical constructions.

    shell_statistics
        Domain, centre, shell, occupancy, and bridge information.

    aggregated_distributions
        Pooled empirical and theoretical P(k) for each pre-peak event.

    theory_diagnostics
        Jensen--Shannon and total-variation errors per realization.

    diagnostic_summary
        Mean and standard deviation of the errors across simulations.

    summary_table
        Compact event-level network summary.
    """
    global MODEL_ROOT, NETWORK_ROOT, ABUNDANCE_ROOT, OUTPUT_DIR

    MODEL_ROOT = Path(MODEL_ROOT).expanduser().resolve()
    NETWORK_ROOT = Path(NETWORK_ROOT).expanduser().resolve()
    ABUNDANCE_ROOT = Path(ABUNDANCE_ROOT).expanduser().resolve()
    OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()

    validate_configuration()

    if DISPLAY_THEORY_DESCRIPTION and ADD_ANALYTICAL_THEORY:
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

    aggregated_distributions = aggregate_degree_distributions(
        network_inventory,
        empirical_degree_counts,
        analytical_expected_counts,
    )

    theory_diagnostics = compute_network_theory_diagnostics(
        empirical_degree_counts,
        analytical_expected_counts,
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
            std_lcc_nodes=("lcc_nodes", "std"),
            mean_lcc_fraction=("lcc_node_fraction", "mean"),
            mean_degree=("mean_lcc_degree", "mean"),
            std_mean_degree=("mean_lcc_degree", "std"),
            mean_maximum_degree=("maximum_lcc_degree", "mean"),
            maximum_degree=("maximum_lcc_degree", "max"),
            mean_primary_centre_degree=(
                "primary_centre_degree",
                "mean",
            ),
            mean_primary_centre_abundance=(
                "primary_centre_abundance",
                "mean",
            ),
            mean_effective_centres=(
                "n_effective_centres",
                "mean",
            ),
        )
        .fillna({
            "std_lcc_nodes": 0.0,
            "std_mean_degree": 0.0,
        })
        .sort_values(
            FAMILY_COLUMNS + ["peak_event_index", "t"]
        )
        .reset_index(drop=True)
    )

    if not diagnostic_summary.empty:
        jsd_wide = (
            diagnostic_summary
            .pivot_table(
                index=EVENT_COLUMNS,
                columns="theory_type",
                values="mean_jensen_shannon",
                aggfunc="first",
            )
            .rename(columns={
                SINGLE_CENTRE_THEORY: "mean_jsd_single_centre",
                MULTI_CENTRE_THEORY: "mean_jsd_multi_centre",
            })
            .reset_index()
        )

        summary_table = summary_table.merge(
            jsd_wide,
            on=EVENT_COLUMNS,
            how="left",
        )

    print("\nPre-peak event summary:")
    try:
        display(summary_table)
    except Exception:
        print(summary_table.to_string(index=False))

    if not diagnostic_summary.empty:
        print("\nTheory diagnostics:")
        try:
            display(diagnostic_summary)
        except Exception:
            print(diagnostic_summary.to_string(index=False))

    saved_paths = plot_pre_peak_degree_panels(
        aggregated_distributions,
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
    print(f"Theory occupancy source: {THEORY_OCCUPANCY_SOURCE}")
    print(
        "Corrected theory: abundance-local-maxima Voronoi domains "
        "with conditioned shell and bridge incidences."
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
        theory_diagnostics,
        diagnostic_summary,
        summary_table,
    )


# ============================================================
# Execute in Jupyter
# ============================================================
# Edit the configuration block above and run this cell.

(
    network_inventory,
    empirical_degree_counts,
    analytical_expected_counts,
    shell_statistics,
    aggregated_distributions,
    theory_diagnostics,
    diagnostic_summary,
    summary_table,
) = run_analysis()