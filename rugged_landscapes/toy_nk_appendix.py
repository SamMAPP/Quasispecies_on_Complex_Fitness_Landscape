#!/usr/bin/env python3
"""
toy_nk_appendix.py -- appendix "Explicit construction of a short NK landscape" and
                      Fig. "toy_NK_evolution_landscape_and_network" (Chapter 4)

Fully enumerable adjacent circular NK landscape, L=3, K=1, quaternary alphabet, landscape seed 12345:
  * Z(g) = [u_1(g1,g2) + u_2(g2,g3) + u_3(g3,g1)] / 3 for all 64 genotypes; exact global optimum by
    brute force, checked against the dynamic-programming optimizer of landscapes.py (Table
    tab:toy_global_verification);
  * threshold f_l from f(g*)/<f>_{9 one-mutants} = exp(2 lambda), lambda = 0.275, by bisection
    (Eqs. toy_threshold_equation, toy_threshold_value; Table tab:toy_TTG_shell);
  * Wright-Fisher evolution, N=1e4, t_max=200, evolution seed 2026, from AAA; successive population
    modes, final abundances and mean scores (Eq. toy_mode_path);
  * figure: (a) effective-fitness landscape on a spring-layout embedding of the Hamming-one graph,
    (b) genotype network at t=200.
Port of the thesis notebook 17-Simulations-NK/suite_detector.ipynb (cell 22).

    python toy_nk_appendix.py --out Figures
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import os
import sys
_ap = argparse.ArgumentParser(description="L=3, K=1 NK appendix example")
_ap.add_argument("--out", default="Figures")
ARGS = _ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from landscapes import NKField, exact_global_maximum, to_string

# =============================================================================
# FIGURE 4 — TOY NK LANDSCAPE AND EVOLUTIONARY EXAMPLE
#
# Fully self-contained appendix figure.
#
# Scientific setup
# ----------------
#   Alphabet             : A, C, G, T
#   L                    : 3
#   K                    : 1
#   Landscape seed       : 12345
#   lambda               : 0.275
#   c                    : 2
#   N                    : 10^4
#   T_MAX                : 200
#   Evolution seed       : 2026
#   Initial genotype     : AAA
#
# For this deterministic realization:
#
#   exact global optimum : TTG
#   population-mode path : AAA -> TAA -> TTG
#
#
# Mapping
# -------
# Raw adjacent/circular NK score:
#
#       Z(g) = (1/L) sum_i u_i(g_i, g_{i+1})
#
# Effective reproductive fitness:
#
#       f(g) = max[0, Z(g) - f_l]
#
# f_l is calibrated from the complete one-mutant shell of the exact
# global optimum using:
#
#       f(g*) / <f(g_1)> = exp(c lambda)
#
#
# Figure
# ------
# (a) Effective-fitness landscape.
#     - complete Hamming-one topology
#     - z = effective reproductive fitness
#     - occupied genotypes emphasized
#     - black line = successive population modes
#
# (b) Genotype network at t=200.
#     - same x,y spring-layout coordinates
#     - node area proportional to log(abundance + 1)
#     - color = Hamming distance to exact global optimum
#
# Special markers
# ---------------
# square  : initial genotype
# circle  : population root at t=200
# diamond : exact global optimum
#
# IMPORTANT:
# The black trajectory in panel (a) represents successive identities of
# the population mode. It is NOT a lineage or ancestral path.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# =============================================================================
# 2. TOY NK LANDSCAPE PARAMETERS
# =============================================================================

ALPHABET = (
    "A",
    "C",
    "G",
    "T",
)

L = 3
K = 1

SEED = 12345

lambda_mut = 0.275

c_localization = 2.0

R_target = np.exp(
    c_localization
    *
    lambda_mut
)


# =============================================================================
# 3. EVOLUTIONARY SIMULATION PARAMETERS
# =============================================================================

N = 10_000

T_MAX = 200

EVOLUTION_SEED = 2026

START_SEQUENCE = "AAA"


# =============================================================================
# 4. FIGURE / LAYOUT PARAMETERS
#
# These reproduce the styling developed in toy_nk_global_peak_demo.
# =============================================================================

SPRING_SEED = 123

SPRING_K = None

SPRING_ITERATIONS = 1000


FIGSIZE = (
    13.6,
    5.8,
)

DPI = 400


DISTANCE_CMAP = "viridis_r"


# Maximum marker AREA in points^2.
#
# Actual area is proportional to log(abundance + 1), normalized so that
# the most abundant genotype receives this maximum graphical area.
MAX_NODE_AREA = 185.0


# Faint complete genotype-space skeleton
FULL_EDGE_WIDTH = 0.45
FULL_EDGE_ALPHA = 0.11


# Stronger edges among occupied genotypes
ACTIVE_EDGE_WIDTH = 0.75
ACTIVE_EDGE_ALPHA = 0.27


LABEL_FONTSIZE = 8.5


SAVE = True


# =============================================================================
# 5. OUTPUT
# =============================================================================

OUTPUT_DIR = Path(ARGS.out)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PDF_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_evolution_landscape_and_network.pdf"
)


PNG_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_evolution_landscape_and_network.png"
)


TRAJECTORY_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_evolution_trajectory.csv"
)


POPULATION_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_evolution_final_population.csv"
)


LANDSCAPE_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_complete_landscape.csv"
)


LAYOUT_PATH = (
    OUTPUT_DIR
    /
    "toy_NK_spring_layout.csv"
)


# =============================================================================
# 6. CONSTRUCT ONE QUENCHED ADJACENT/CIRCULAR NK LANDSCAPE
#
# For K=1:
#
#       u_i = u_i(g_i, g_{i+1})
#
# with periodic indexing.
# =============================================================================

rng_landscape = np.random.default_rng(
    SEED
)


n_letters = len(
    ALPHABET
)


u = rng_landscape.random(
    (
        L,
        n_letters,
        n_letters,
    )
)


letter_to_int = {

    letter:
        i

    for i, letter
    in enumerate(
        ALPHABET
    )
}


# =============================================================================
# 7. RAW NK SCORE
# =============================================================================

def Z_of(
    sequence,
):

    x = [

        letter_to_int[
            base
        ]

        for base
        in sequence
    ]


    total = 0.0


    for i in range(
        L
    ):

        j = (
            i + 1
        ) % L


        total += u[
            i,
            x[i],
            x[j],
        ]


    return (
        total
        /
        L
    )


# =============================================================================
# 8. ENUMERATE COMPLETE GENOTYPE SPACE
# =============================================================================

sequences = [

    "".join(
        letters
    )

    for letters
    in itertools.product(
        ALPHABET,
        repeat=L,
    )
]


Z_values = {

    seq:
        float(
            Z_of(
                seq
            )
        )

    for seq
    in sequences
}


# =============================================================================
# 9. HAMMING DISTANCE
# =============================================================================

def hamming(
    a,
    b,
):

    return sum(

        x != y

        for x, y
        in zip(
            a,
            b,
        )
    )


# =============================================================================
# 10. EXACT GLOBAL OPTIMUM BY BRUTE FORCE
#
# For L=3 there are only 4^3 = 64 sequences, so direct enumeration is exact.
# =============================================================================

peak_sequence = max(

    sequences,

    key=lambda seq:
        Z_values[
            seq
        ],
)


Z_peak = float(
    Z_values[
        peak_sequence
    ]
)


# =============================================================================
# 11. ONE-MUTANT SHELL OF GLOBAL OPTIMUM
# =============================================================================

# Independent check: exact finite-range optimizer (dynamic programming, landscapes.py)
_dp_seq, _dp_z = exact_global_maximum(NKField.from_tables(u.reshape(L, n_letters ** (K + 1)), K))
print(f"Brute force : {peak_sequence}  Z = {Z_peak:.9f}")
print(f"DP optimizer: {to_string(_dp_seq)}  Z = {_dp_z:.9f}")
if to_string(_dp_seq) != peak_sequence or abs(_dp_z - Z_peak) > 1e-12:
    raise RuntimeError("dynamic programming and brute force disagree")


one_mutant_neighbors = [

    seq

    for seq
    in sequences

    if hamming(
        seq,
        peak_sequence,
    ) == 1
]


neighbor_Z = np.asarray(
    [

        Z_values[
            seq
        ]

        for seq
        in one_mutant_neighbors
    ],
    dtype=float,
)


# =============================================================================
# 12. CALIBRATE f_l BY MONOTONE BISECTION
#
# Solve:
#
#                 f(g*)
#       ------------------------- = R_target
#       mean one-mutant fitness
#
# where
#
#       R_target = exp(c lambda)
#
# =============================================================================

def fitness_ratio(
    f_l_trial,
):

    peak_fitness = max(

        0.0,

        Z_peak
        -
        f_l_trial,
    )


    neighbor_fitness = np.maximum(

        0.0,

        neighbor_Z
        -
        f_l_trial,
    )


    mean_neighbor_fitness = float(
        neighbor_fitness.mean()
    )


    if mean_neighbor_fitness <= 0:

        return np.inf


    return (
        peak_fitness
        /
        mean_neighbor_fitness
    )


low = 0.0

high = (
    Z_peak
    -
    1e-12
)


for _ in range(
    200
):

    mid = 0.5 * (
        low
        +
        high
    )


    ratio = fitness_ratio(
        mid
    )


    if ratio < R_target:

        low = mid

    else:

        high = mid


f_l = 0.5 * (
    low
    +
    high
)


# =============================================================================
# 13. EFFECTIVE REPRODUCTIVE FITNESS
# =============================================================================

def fitness_of(
    sequence,
):

    return max(

        0.0,

        Z_values[
            sequence
        ]

        -

        f_l,
    )


# =============================================================================
# 14. BASIC LANDSCAPE INFORMATION
# =============================================================================

print(
    "=" * 82
)

print(
    "TOY ADJACENT/CIRCULAR NK LANDSCAPE"
)

print(
    "=" * 82
)


print(
    f"Alphabet                     : {ALPHABET}"
)

print(
    f"L                            : {L}"
)

print(
    f"K                            : {K}"
)

print(
    f"Landscape seed               : {SEED}"
)

print(
    f"Number of genotypes          : {len(sequences)}"
)


print()


print(
    f"Global optimum               : {peak_sequence}"
)

print(
    f"Z(global)                    : {Z_peak:.9f}"
)

print(
    f"f(global)                    : {fitness_of(peak_sequence):.9f}"
)


print()


print(
    f"lambda                       : {lambda_mut}"
)

print(
    f"c                            : {c_localization}"
)

print(
    f"R_target                     : {R_target:.9f}"
)

print(
    f"f_l                          : {f_l:.9f}"
)

print(
    f"realized localization ratio  : "
    f"{fitness_ratio(f_l):.9f}"
)


# -----------------------------------------------------------------------------
# Verify expected deterministic landscape
# -----------------------------------------------------------------------------

if peak_sequence != "TTG":

    raise RuntimeError(
        "Unexpected toy landscape. "
        f"Global optimum = {peak_sequence}; expected TTG."
    )


# =============================================================================
# 15. COMPLETE NEIGHBOR STRUCTURE / LOCAL MAXIMA
# =============================================================================

neighbors = {

    seq: [

        other

        for other
        in sequences

        if hamming(
            seq,
            other,
        ) == 1
    ]

    for seq
    in sequences
}


local_maxima = []


for seq in sequences:

    if all(

        Z_values[
            seq
        ]

        >=

        Z_values[
            n
        ]

        for n
        in neighbors[
            seq
        ]
    ):

        local_maxima.append(
            seq
        )


# =============================================================================
# 16. INITIAL GENOTYPE
# =============================================================================

if START_SEQUENCE not in sequences:

    raise RuntimeError(
        f"{START_SEQUENCE} is not present in the sequence space."
    )


if fitness_of(
    START_SEQUENCE
) <= 0:

    raise RuntimeError(
        f"{START_SEQUENCE} has zero effective fitness."
    )


print()


print(
    f"Initial genotype             : {START_SEQUENCE}"
)

print(
    f"Z(initial)                   : "
    f"{Z_values[START_SEQUENCE]:.9f}"
)

print(
    f"f(initial)                   : "
    f"{fitness_of(START_SEQUENCE):.9f}"
)

print(
    f"d_H(initial, global)         : "
    f"{hamming(START_SEQUENCE, peak_sequence)}"
)


# =============================================================================
# 17. BASIC ARRAYS
# =============================================================================

seq_to_idx = {

    seq:
        i

    for i, seq
    in enumerate(
        sequences
    )
}


idx_to_seq = {

    i:
        seq

    for seq, i
    in seq_to_idx.items()
}


fitness_array = np.asarray(
    [

        fitness_of(
            seq
        )

        for seq
        in sequences
    ],
    dtype=float,
)


Z_array = np.asarray(
    [

        Z_values[
            seq
        ]

        for seq
        in sequences
    ],
    dtype=float,
)


# =============================================================================
# 18. INITIAL MONOMORPHIC POPULATION
# =============================================================================

rng_evo = np.random.default_rng(
    EVOLUTION_SEED
)


counts = np.zeros(

    len(
        sequences
    ),

    dtype=np.int64,
)


counts[
    seq_to_idx[
        START_SEQUENCE
    ]
] = N


# =============================================================================
# 19. MUTATION FUNCTION
#
# Each offspring receives:
#
#       n ~ Poisson(lambda)
#
# mutation events.
#
# Each event independently:
#   1. chooses one site uniformly;
#   2. replaces its nucleotide by one of the other three bases uniformly.
#
# Multiple events may hit the same site.
# =============================================================================

def mutate_sequence(
    seq,
    n_mutations,
    rng,
):

    child = list(
        seq
    )


    for _ in range(
        n_mutations
    ):

        position = int(
            rng.integers(
                L
            )
        )


        old_letter = child[
            position
        ]


        alternatives = [

            base

            for base
            in ALPHABET

            if base != old_letter
        ]


        child[
            position
        ] = rng.choice(
            alternatives
        )


    return "".join(
        child
    )


# =============================================================================
# 20. EVOLUTION
# =============================================================================

records = []

dominant_history = []


for t in range(
    T_MAX + 1
):

    # -------------------------------------------------------------------------
    # Current population statistics
    # -------------------------------------------------------------------------

    frequencies = (
        counts
        /
        counts.sum()
    )


    dominant_idx = int(
        np.argmax(
            counts
        )
    )


    dominant_seq = idx_to_seq[
        dominant_idx
    ]


    dominant_history.append(
        dominant_seq
    )


    mean_Z = float(
        np.sum(
            frequencies
            *
            Z_array
        )
    )


    mean_fitness = float(
        np.sum(
            frequencies
            *
            fitness_array
        )
    )


    global_fraction = float(
        frequencies[
            seq_to_idx[
                peak_sequence
            ]
        ]
    )


    records.append(
        {

            "generation":
                t,

            "dominant_sequence":
                dominant_seq,

            "dominant_abundance":
                int(
                    counts[
                        dominant_idx
                    ]
                ),

            "dominant_fraction":
                float(
                    frequencies[
                        dominant_idx
                    ]
                ),

            "dominant_Z":
                float(
                    Z_values[
                        dominant_seq
                    ]
                ),

            "dominant_fitness":
                float(
                    fitness_of(
                        dominant_seq
                    )
                ),

            "mean_Z":
                mean_Z,

            "mean_fitness":
                mean_fitness,

            "global_peak_fraction":
                global_fraction,

            "distance_dominant_from_start":
                hamming(
                    dominant_seq,
                    START_SEQUENCE,
                ),

            "distance_dominant_from_global":
                hamming(
                    dominant_seq,
                    peak_sequence,
                ),

            "n_occupied":
                int(
                    np.count_nonzero(
                        counts
                    )
                ),
        }
    )


    if t == T_MAX:

        break


    # =========================================================================
    # SELECTION
    # =========================================================================

    reproductive_weight = (

        counts.astype(
            float
        )

        *

        fitness_array
    )


    total_weight = float(
        reproductive_weight.sum()
    )


    if total_weight <= 0:

        raise RuntimeError(
            "No viable reproductive population remains."
        )


    parent_probability = (
        reproductive_weight
        /
        total_weight
    )


    parent_indices = rng_evo.choice(

        len(
            sequences
        ),

        size=N,

        replace=True,

        p=parent_probability,
    )


    # =========================================================================
    # MUTATION
    # =========================================================================

    mutation_counts = rng_evo.poisson(

        lambda_mut,

        size=N,
    )


    offspring_counts = np.zeros(

        len(
            sequences
        ),

        dtype=np.int64,
    )


    for parent_idx, n_mutations in zip(

        parent_indices,

        mutation_counts,
    ):

        child = idx_to_seq[
            int(
                parent_idx
            )
        ]


        if n_mutations > 0:

            child = mutate_sequence(

                child,

                int(
                    n_mutations
                ),

                rng_evo,
            )


        offspring_counts[
            seq_to_idx[
                child
            ]
        ] += 1


    counts = offspring_counts


# =============================================================================
# 21. RESULTS TABLES
# =============================================================================

trajectory = pd.DataFrame(
    records
)


final_population = pd.DataFrame(
    {

        "sequence":
            sequences,

        "Z":
            [

                Z_values[
                    seq
                ]

                for seq
                in sequences
            ],

        "fitness":
            [

                fitness_of(
                    seq
                )

                for seq
                in sequences
            ],

        "abundance":
            counts,

        "frequency":
            counts
            /
            N,

        "distance_from_start":
            [

                hamming(
                    seq,
                    START_SEQUENCE,
                )

                for seq
                in sequences
            ],

        "distance_from_global_peak":
            [

                hamming(
                    seq,
                    peak_sequence,
                )

                for seq
                in sequences
            ],
    }
).sort_values(

    "abundance",

    ascending=False,

).reset_index(
    drop=True
)


# =============================================================================
# 22. DOMINANT-GENOTYPE CHANGES
# =============================================================================

change_rows = trajectory[

    trajectory[
        "dominant_sequence"
    ]

    !=

    trajectory[
        "dominant_sequence"
    ].shift()

].copy()


print(
    "\n"
    + "=" * 82
)

print(
    "DOMINANT-GENOTYPE CHANGES"
)

print(
    "=" * 82
)


print(

    change_rows[
        [
            "generation",
            "dominant_sequence",
            "dominant_fraction",
            "dominant_Z",
            "dominant_fitness",
            "distance_dominant_from_global",
        ]
    ]
)


# =============================================================================
# 23. DOMINANT PATH
#
# Remove repeated consecutive dominant genotypes.
# =============================================================================

dominant_path = []


for seq in dominant_history:

    if (
        len(
            dominant_path
        ) == 0

        or

        seq
        !=
        dominant_path[
            -1
        ]
    ):

        dominant_path.append(
            seq
        )


print(
    "\nDominant population-mode path:"
)

print(
    "  "
    +
    " -> ".join(
        dominant_path
    )
)


EXPECTED_PATH = [
    "AAA",
    "TAA",
    "TTG",
]


if dominant_path != EXPECTED_PATH:

    print(
        "\nWARNING: the realized population-mode path differs "
        "from the expected pedagogical trajectory."
    )

    print(
        f"Observed: {' -> '.join(dominant_path)}"
    )


# =============================================================================
# 24. COMPLETE HAMMING-ONE GENOTYPE NETWORK
# =============================================================================

G_landscape = nx.Graph()


for seq in sequences:

    G_landscape.add_node(

        seq,

        Z=float(
            Z_values[
                seq
            ]
        ),

        fitness=float(
            fitness_of(
                seq
            )
        ),

        abundance=int(
            counts[
                seq_to_idx[
                    seq
                ]
            ]
        ),
    )


for i, seq1 in enumerate(
    sequences
):

    for seq2 in sequences[
        i + 1:
    ]:

        if hamming(
            seq1,
            seq2,
        ) == 1:

            G_landscape.add_edge(
                seq1,
                seq2,
            )


print(
    "\nComplete genotype network:"
)

print(
    f"  nodes = "
    f"{G_landscape.number_of_nodes()}"
)

print(
    f"  edges = "
    f"{G_landscape.number_of_edges()}"
)


# =============================================================================
# 25. SPRING LAYOUT WITH GLOBAL OPTIMUM FIXED AT ORIGIN
#
# This reproduces the visual construction used in toy_nk_global_peak_demo.
# =============================================================================

initial_pos = nx.random_layout(

    G_landscape,

    seed=SPRING_SEED,
)


# Shift initial positions roughly around zero
for node in initial_pos:

    initial_pos[
        node
    ] = (

        initial_pos[
            node
        ]

        -

        np.asarray(
            [
                0.5,
                0.5,
            ]
        )
    )


# Force exact global optimum to origin
initial_pos[
    peak_sequence
] = np.asarray(
    [
        0.0,
        0.0,
    ]
)


pos = nx.spring_layout(

    G_landscape,

    pos=initial_pos,

    fixed=[
        peak_sequence
    ],

    seed=SPRING_SEED,

    k=SPRING_K,

    iterations=SPRING_ITERATIONS,

    weight=None,
)


# Remove numerical residual offset
offset = np.asarray(
    pos[
        peak_sequence
    ]
)


for seq in pos:

    pos[
        seq
    ] = (

        np.asarray(
            pos[
                seq
            ]
        )

        -

        offset
    )


# -----------------------------------------------------------------------------
# Symmetric plotting limits around global optimum.
#
# This keeps the optimum visually centered in BOTH panels.
# -----------------------------------------------------------------------------

max_abs_coordinate = max(

    max(
        abs(
            float(
                p[
                    0
                ]
            )
        ),

        abs(
            float(
                p[
                    1
                ]
            )
        ),
    )

    for p
    in pos.values()
)


layout_limit = (
    max_abs_coordinate
    *
    1.10
)


# =============================================================================
# 26. CURRENT POPULATION ROOT / TARGET
# =============================================================================

CURRENT_ROOT = idx_to_seq[
    int(
        np.argmax(
            counts
        )
    )
]


TARGET_SEQUENCE = (
    peak_sequence
)


distance_to_target = {

    seq:
        hamming(
            seq,
            TARGET_SEQUENCE,
        )

    for seq
    in sequences
}


max_distance = max(
    distance_to_target.values()
)


print(
    f"\nPopulation root at t={T_MAX}: "
    f"{CURRENT_ROOT}"
)

print(
    f"Exact global optimum       : "
    f"{peak_sequence}"
)


# =============================================================================
# 27. NODE AREA
#
#       A_i ∝ log(a_i + 1)
#
# normalized to MAX_NODE_AREA.
# =============================================================================

max_abundance = int(
    counts.max()
)


if max_abundance > 0:

    log_abundance_normalizer = np.log1p(
        max_abundance
    )

else:

    log_abundance_normalizer = 1.0


node_size = {

    seq:

        (
            MAX_NODE_AREA

            *

            np.log1p(
                counts[
                    seq_to_idx[
                        seq
                    ]
                ]
            )

            /

            log_abundance_normalizer
        )

    for seq
    in sequences
}


# =============================================================================
# 28. FINAL OCCUPIED NETWORK
# =============================================================================

occupied_sequences = [

    seq

    for seq
    in sequences

    if counts[
        seq_to_idx[
            seq
        ]
    ] > 0
]


G_occ = G_landscape.subgraph(
    occupied_sequences
).copy()


# =============================================================================
# 29. RESTRICTED LABELS
#
# Label only:
#   - initial genotype
#   - final population root
#   - exact global optimum
# =============================================================================

label_nodes = []


for seq in [

    START_SEQUENCE,
    CURRENT_ROOT,
    peak_sequence,

]:

    if seq not in label_nodes:

        label_nodes.append(
            seq
        )


# =============================================================================
# 30. COLOR NORMALIZATION
# =============================================================================

norm = Normalize(

    vmin=0,

    vmax=max_distance,
)


cmap = plt.get_cmap(
    DISTANCE_CMAP
)


# =============================================================================
# 31. FIGURE
# =============================================================================

fig = plt.figure(
    figsize=FIGSIZE
)


# =============================================================================
# PANEL (a): FITNESS LANDSCAPE
# =============================================================================

ax1 = fig.add_subplot(

    1,
    2,
    1,

    projection="3d",
)


# -----------------------------------------------------------------------------
# A. Base-plane projection of complete genotype network
# -----------------------------------------------------------------------------

for seq1, seq2 in G_landscape.edges():

    x1, y1 = pos[
        seq1
    ]

    x2, y2 = pos[
        seq2
    ]


    ax1.plot(

        [
            x1,
            x2,
        ],

        [
            y1,
            y2,
        ],

        [
            0.0,
            0.0,
        ],

        color="0.78",

        linewidth=0.40,

        alpha=0.16,

        zorder=0,
    )


# -----------------------------------------------------------------------------
# B. Vertical projections of FINAL occupied genotypes
# -----------------------------------------------------------------------------

for seq in occupied_sequences:

    x, y = pos[
        seq
    ]


    z = fitness_of(
        seq
    )


    ax1.plot(

        [
            x,
            x,
        ],

        [
            y,
            y,
        ],

        [
            0.0,
            z,
        ],

        color="0.55",

        linewidth=0.40,

        alpha=0.20,

        zorder=1,
    )


# -----------------------------------------------------------------------------
# C. Complete landscape edges at actual fitness heights
# -----------------------------------------------------------------------------

for seq1, seq2 in G_landscape.edges():

    x1, y1 = pos[
        seq1
    ]

    x2, y2 = pos[
        seq2
    ]


    z1 = fitness_of(
        seq1
    )

    z2 = fitness_of(
        seq2
    )


    ax1.plot(

        [
            x1,
            x2,
        ],

        [
            y1,
            y2,
        ],

        [
            z1,
            z2,
        ],

        color="0.68",

        linewidth=0.40,

        alpha=0.11,

        zorder=2,
    )


# -----------------------------------------------------------------------------
# D. Ordinary landscape nodes
# -----------------------------------------------------------------------------

special_nodes = {

    START_SEQUENCE,
    CURRENT_ROOT,
    peak_sequence,
}


normal_landscape_nodes = [

    seq

    for seq
    in sequences

    if seq not in special_nodes
]


if normal_landscape_nodes:

    ax1.scatter(

        [
            pos[
                seq
            ][0]

            for seq
            in normal_landscape_nodes
        ],

        [
            pos[
                seq
            ][1]

            for seq
            in normal_landscape_nodes
        ],

        [
            fitness_of(
                seq
            )

            for seq
            in normal_landscape_nodes
        ],

        s=[
            max(
                10.0,
                node_size[
                    seq
                ]
            )

            for seq
            in normal_landscape_nodes
        ],

        c=[
            distance_to_target[
                seq
            ]

            for seq
            in normal_landscape_nodes
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        edgecolors="0.20",

        linewidths=0.35,

        alpha=0.82,

        depthshade=False,

        zorder=5,
    )


# -----------------------------------------------------------------------------
# E. Dominant population-mode trajectory
#
# NOT a lineage.
# -----------------------------------------------------------------------------

path_x = [

    pos[
        seq
    ][0]

    for seq
    in dominant_path
]


path_y = [

    pos[
        seq
    ][1]

    for seq
    in dominant_path
]


path_z = [

    fitness_of(
        seq
    )

    for seq
    in dominant_path
]


if len(
    dominant_path
) > 1:

    ax1.plot(

        path_x,
        path_y,
        path_z,

        color="0.10",

        linewidth=1.25,

        alpha=0.85,

        zorder=10,
    )


# -----------------------------------------------------------------------------
# F. SPECIAL NODES
# -------------------------------------------------------------------------

# Initial genotype — square
if START_SEQUENCE in occupied_sequences:

    ax1.scatter(

        [
            pos[
                START_SEQUENCE
            ][0]
        ],

        [
            pos[
                START_SEQUENCE
            ][1]
        ],

        [
            fitness_of(
                START_SEQUENCE
            )
        ],

        s=max(
            node_size[
                START_SEQUENCE
            ],
            70.0,
        ),

        c=[
            distance_to_target[
                START_SEQUENCE
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        marker="s",

        edgecolors="black",

        linewidths=1.25,

        depthshade=False,

        zorder=20,
    )


# Current population root — circle
if CURRENT_ROOT in occupied_sequences:

    ax1.scatter(

        [
            pos[
                CURRENT_ROOT
            ][0]
        ],

        [
            pos[
                CURRENT_ROOT
            ][1]
        ],

        [
            fitness_of(
                CURRENT_ROOT
            )
        ],

        s=max(
            node_size[
                CURRENT_ROOT
            ],
            70.0,
        ),

        c=[
            distance_to_target[
                CURRENT_ROOT
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        marker="o",

        edgecolors="black",

        linewidths=1.7,

        depthshade=False,

        zorder=21,
    )


# Exact global optimum — diamond
if peak_sequence in occupied_sequences:

    ax1.scatter(

        [
            pos[
                peak_sequence
            ][0]
        ],

        [
            pos[
                peak_sequence
            ][1]
        ],

        [
            fitness_of(
                peak_sequence
            )
        ],

        s=max(
            node_size[
                peak_sequence
            ],
            70.0,
        ),

        c=[
            distance_to_target[
                peak_sequence
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        marker="D",

        edgecolors="black",

        linewidths=1.45,

        depthshade=False,

        zorder=22,
    )


# -----------------------------------------------------------------------------
# G. Restricted sequence labels
# -----------------------------------------------------------------------------

z_range = max(

    fitness_array.max()
    -
    fitness_array.min(),

    1e-9,
)


for seq in label_nodes:

    ax1.text(

        pos[
            seq
        ][0],

        pos[
            seq
        ][1],

        fitness_of(
            seq
        )

        +

        0.025
        *
        z_range,

        seq,

        fontsize=LABEL_FONTSIZE,

        fontweight="semibold",

        ha="center",

        va="bottom",

        zorder=30,
    )


# -----------------------------------------------------------------------------
# H. 3D AXIS STYLE
#
# This follows the toy_nk_global_peak_demo styling.
# -----------------------------------------------------------------------------

ax1.set_title(

    "(a) Fitness landscape",

    fontsize=11,

    pad=8,
)


# No x/y labels or numerical ticks:
# spring-layout coordinates have no biological units.
ax1.set_xlabel(
    ""
)

ax1.set_ylabel(
    ""
)

ax1.set_xticks(
    []
)

ax1.set_yticks(
    []
)


# Keep z scale because effective fitness is a metric quantity.
ax1.set_zlabel(

    "Effective fitness",

    fontsize=10,

    labelpad=7,
)


ax1.tick_params(

    axis="z",

    labelsize=8,
)


# Global optimum centered in x-y domain.
ax1.set_xlim(

    -layout_limit,

    layout_limit,
)


ax1.set_ylim(

    -layout_limit,

    layout_limit,
)


# Side panes unobtrusive
ax1.xaxis.pane.set_alpha(
    0.0
)

ax1.yaxis.pane.set_alpha(
    0.0
)


# Explicit visible floor plane
ax1.zaxis.pane.set_facecolor(
    (
        0.96,
        0.96,
        0.96,
        1.0,
    )
)

ax1.zaxis.pane.set_alpha(
    0.55
)


# Perspective used in the refined toy figure
ax1.view_init(

    elev=24,

    azim=-58,
)


# Less distorted 3D proportions
try:

    ax1.set_box_aspect(
        (
            1.05,
            1.05,
            0.75,
        )
    )

except Exception:

    pass


# =============================================================================
# PANEL (b): FINAL GENOTYPE NETWORK
# =============================================================================

ax2 = fig.add_subplot(

    1,
    2,
    2,
)


# -----------------------------------------------------------------------------
# A. Complete genotype-space skeleton — faint context
# -----------------------------------------------------------------------------

nx.draw_networkx_edges(

    G_landscape,

    pos,

    ax=ax2,

    width=FULL_EDGE_WIDTH,

    alpha=FULL_EDGE_ALPHA,

    edge_color="0.55",
)


# -----------------------------------------------------------------------------
# B. Edges among final occupied genotypes
# -----------------------------------------------------------------------------

nx.draw_networkx_edges(

    G_occ,

    pos,

    ax=ax2,

    width=ACTIVE_EDGE_WIDTH,

    alpha=ACTIVE_EDGE_ALPHA,

    edge_color="0.30",
)


# -----------------------------------------------------------------------------
# C. Ordinary occupied nodes
# -----------------------------------------------------------------------------

normal_nodes = [

    seq

    for seq
    in occupied_sequences

    if seq not in special_nodes
]


if normal_nodes:

    nx.draw_networkx_nodes(

        G_occ,

        pos,

        nodelist=
            normal_nodes,

        node_size=[
            node_size[
                seq
            ]
            for seq
            in normal_nodes
        ],

        node_color=[
            distance_to_target[
                seq
            ]
            for seq
            in normal_nodes
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        edgecolors="0.15",

        linewidths=0.45,

        ax=ax2,
    )


# -----------------------------------------------------------------------------
# D. Initial genotype — square
# -----------------------------------------------------------------------------

if START_SEQUENCE in G_occ:

    nx.draw_networkx_nodes(

        G_occ,

        pos,

        nodelist=[
            START_SEQUENCE
        ],

        node_shape="s",

        node_size=[
            max(
                node_size[
                    START_SEQUENCE
                ],
                70.0,
            )
        ],

        node_color=[
            distance_to_target[
                START_SEQUENCE
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        edgecolors="black",

        linewidths=1.25,

        ax=ax2,
    )


# -----------------------------------------------------------------------------
# E. Current root — emphasized circle
# -----------------------------------------------------------------------------

if CURRENT_ROOT in G_occ:

    nx.draw_networkx_nodes(

        G_occ,

        pos,

        nodelist=[
            CURRENT_ROOT
        ],

        node_shape="o",

        node_size=[
            max(
                node_size[
                    CURRENT_ROOT
                ],
                70.0,
            )
        ],

        node_color=[
            distance_to_target[
                CURRENT_ROOT
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        edgecolors="black",

        linewidths=1.7,

        ax=ax2,
    )


# -----------------------------------------------------------------------------
# F. Exact global optimum — diamond
# -----------------------------------------------------------------------------

if peak_sequence in G_occ:

    nx.draw_networkx_nodes(

        G_occ,

        pos,

        nodelist=[
            peak_sequence
        ],

        node_shape="D",

        node_size=[
            max(
                node_size[
                    peak_sequence
                ],
                70.0,
            )
        ],

        node_color=[
            distance_to_target[
                peak_sequence
            ]
        ],

        cmap=DISTANCE_CMAP,

        vmin=0,

        vmax=max_distance,

        edgecolors="black",

        linewidths=1.45,

        ax=ax2,
    )


# -----------------------------------------------------------------------------
# G. Restricted labels
# -----------------------------------------------------------------------------

network_labels = {

    seq:
        seq

    for seq
    in label_nodes

    if seq
    in G_occ
}


nx.draw_networkx_labels(

    G_occ,

    pos,

    labels=
        network_labels,

    ax=ax2,

    font_size=
        LABEL_FONTSIZE,

    font_weight=
        "semibold",
)


# -----------------------------------------------------------------------------
# H. 2D NETWORK STYLE
# -----------------------------------------------------------------------------

ax2.set_title(

    rf"(b) Genotype network at $t={T_MAX}$",

    fontsize=11,

    pad=8,
)


# Keep global optimum visually centered.
ax2.set_xlim(

    -layout_limit,

    layout_limit,
)


ax2.set_ylim(

    -layout_limit,

    layout_limit,
)


ax2.set_aspect(

    "equal",

    adjustable="box",
)


# No Cartesian axes or border.
ax2.set_axis_off()


# =============================================================================
# 32. COLORBAR
# =============================================================================

sm = ScalarMappable(

    norm=norm,

    cmap=DISTANCE_CMAP,
)

sm.set_array(
    []
)


cbar = fig.colorbar(

    sm,

    ax=[
        ax1,
        ax2,
    ],

    fraction=0.025,

    pad=0.025,

    shrink=0.82,
)


cbar.set_label(

    "Hamming distance to global optimum",

    fontsize=9,

    labelpad=8,
)


cbar.set_ticks(

    np.arange(
        0,
        max_distance + 1,
        1,
    )
)


cbar.ax.tick_params(

    labelsize=8,
)


# =============================================================================
# 33. MARKER LEGEND
#
# Same compact, frameless legend used in toy_nk_global_peak_demo.
# =============================================================================

legend_handles = [

    Line2D(

        [0],
        [0],

        marker="s",

        linestyle="None",

        markerfacecolor="0.75",

        markeredgecolor="black",

        markeredgewidth=1.1,

        markersize=7,

        label="Initial genotype",
    ),


    Line2D(

        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="0.75",

        markeredgecolor="black",

        markeredgewidth=1.5,

        markersize=7,

        label=rf"Population root at $t={T_MAX}$",
    ),


    Line2D(

        [0],
        [0],

        marker="D",

        linestyle="None",

        markerfacecolor="0.75",

        markeredgecolor="black",

        markeredgewidth=1.1,

        markersize=7,

        label="Exact global optimum",
    ),
]


ax2.legend(

    handles=
        legend_handles,

    loc="upper left",

    bbox_to_anchor=(
        0.00,
        1.00,
    ),

    frameon=False,

    fontsize=8,

    handletextpad=0.55,

    borderaxespad=0.0,
)


# =============================================================================
# 34. LAYOUT
#
# Deliberately NO overall figure title.
# =============================================================================

fig.subplots_adjust(

    left=0.035,

    right=0.905,

    bottom=0.055,

    top=0.92,

    wspace=0.055,
)


# =============================================================================
# 35. SAVE OUTPUT
# =============================================================================

landscape_table = pd.DataFrame(
    {

        "sequence":
            sequences,

        "Z":
            [
                Z_values[
                    seq
                ]

                for seq
                in sequences
            ],

        "fitness":
            [
                fitness_of(
                    seq
                )

                for seq
                in sequences
            ],

        "distance_to_global_optimum":
            [
                distance_to_target[
                    seq
                ]

                for seq
                in sequences
            ],

        "is_local_maximum":
            [
                seq in local_maxima

                for seq
                in sequences
            ],

        "is_initial_genotype":
            [
                seq == START_SEQUENCE

                for seq
                in sequences
            ],

        "is_final_root":
            [
                seq == CURRENT_ROOT

                for seq
                in sequences
            ],

        "is_global_optimum":
            [
                seq == peak_sequence

                for seq
                in sequences
            ],
    }
)


layout_table = pd.DataFrame(
    {

        "sequence":
            sequences,

        "x":
            [
                pos[
                    seq
                ][0]

                for seq
                in sequences
            ],

        "y":
            [
                pos[
                    seq
                ][1]

                for seq
                in sequences
            ],
    }
)


if SAVE:

    fig.savefig(

        PDF_PATH,

        bbox_inches="tight",
    )


    fig.savefig(

        PNG_PATH,

        dpi=DPI,

        bbox_inches="tight",
    )


    trajectory.to_csv(

        TRAJECTORY_PATH,

        index=False,
    )


    final_population.to_csv(

        POPULATION_PATH,

        index=False,
    )


    landscape_table.to_csv(

        LANDSCAPE_PATH,

        index=False,
    )


    layout_table.to_csv(

        LAYOUT_PATH,

        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        f"  {PDF_PATH}"
    )

    print(
        f"  {PNG_PATH}"
    )

    print(
        f"  {TRAJECTORY_PATH}"
    )

    print(
        f"  {POPULATION_PATH}"
    )

    print(
        f"  {LANDSCAPE_PATH}"
    )

    print(
        f"  {LAYOUT_PATH}"
    )


# =============================================================================
# 36. FINAL CHECKS
# =============================================================================

print(
    "\n"
    + "=" * 82
)

print(
    "FINAL CHECK"
)

print(
    "=" * 82
)


print(
    f"Global optimum              : {peak_sequence}"
)

print(
    f"Final population root       : {CURRENT_ROOT}"
)

print(
    f"Population-mode trajectory  : "
    f"{' -> '.join(dominant_path)}"
)


change_generations = (
    change_rows[
        [
            "generation",
            "dominant_sequence",
        ]
    ]
    .reset_index(
        drop=True
    )
)


print(
    change_generations
)


plt.show()