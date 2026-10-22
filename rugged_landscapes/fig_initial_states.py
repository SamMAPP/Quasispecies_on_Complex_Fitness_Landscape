#!/usr/bin/env python3
"""
fig_initial_states.py -- Fig. "NK_initial_state_summary_unfiltered_analysis" (Chapter 4, Sec. 4.4.5)

The same quenched NK landscape (replicate 0, base seed 12345, N=1e6) evolved from three initial states:
Local (best of 16 local maxima), Evolved (best genotype of a 5000-generation search) and Global (exact
optimum). For t = 50, 100, 200:
  * degree CCDF of the LCC of the full, unfiltered genotype network of each snapshot;
  * network drawings: genotypes with abundance > 5, positions shared by the three times of a row;
  * last column: Venn diagram of the genotype sets of the three snapshot LCCs and pairwise Jaccard
    indices J_{50,100}, J_{100,200}, J_{50,200}.

Inputs: --local/--evolved/--global, outputs of generate_rugged.py (or the thesis suites), and
--gn-local/--gn-evolved/--gn-global, folders written by export_grand_network.py. A missing
GN_LCC_*_positions.gexf is created with a seeded ForceAtlas2 layout and reused afterwards. NOTE: the
layouts of the thesis figure were produced interactively in Gephi, so the drawings regenerated here have
a different (equivalent) node arrangement; the CCDFs and Jaccard indices are unaffected.
Port of the thesis notebook 17-Simulations-NK/suite_detector.ipynb (cell 7).

    python fig_initial_states.py --local Data/nk_local --evolved Data/nk_evolved --global Data/nk_global \
        --gn-local Figures/GN_local --gn-evolved Figures/GN_evolved --gn-global Figures/GN_global --out Figures
"""
import matplotlib
matplotlib.use("Agg")
import argparse
import os
import sys
_ap = argparse.ArgumentParser(description="Initial-state comparison on one NK landscape")
for _k in ("local", "evolved", "global"):
    _ap.add_argument(f"--{_k}", dest=_k + ("_" if _k == "global" else ""), required=True,
                     help=f"{_k} run (generate_rugged.py output or thesis suite)")
    _ap.add_argument(f"--gn-{_k}", dest=f"gn_{_k}", required=True, help="folder from export_grand_network.py")
_ap.add_argument("--replicate", type=int, default=0)
_ap.add_argument("--out", default="Figures")
ARGS = _ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_analysis import find_snapshot

# =============================================================================
# FIGURE 2 — INITIAL-STATE COMPARISON ON THE SAME NK LANDSCAPE
#
# CORRECTED VERSION
#
# ANALYSIS AND VISUALIZATION ARE NOW COMPLETELY SEPARATED.
#
#
# ============================================================================
# A. ANALYTICAL NETWORKS — READ FROM ORIGINAL SNAPSHOTS
# ============================================================================
#
# For EACH initial condition and EACH passage independently:
#
#     original snapshot
#           ↓
#     reconstruct ALL genotype sequences
#           ↓
#     no abundance filtering
#           ↓
#     construct Hamming-distance-1 network G_t
#           ↓
#     extract LCC(G_t)
#           ↓
#     degree CCDF from LCC(G_t)
#
#
# Temporal Venn/Jaccard:
#
#     node sequences in LCC_50
#     node sequences in LCC_100
#     node sequences in LCC_200
#
# No abundance threshold is applied to either CCDF or Venn.
#
#
# ============================================================================
# B. NETWORK VISUALIZATION — USE EXISTING FILTERED GRAPHML
# ============================================================================
#
# Existing GN-LCC GraphMLs are used ONLY for visualization.
#
# Nodes are drawn with:
#
#     abundance_t > 5
#
# exactly as before.
#
# These GraphMLs do NOT determine the CCDF or Venn.
#
#
# POSITIONS
# ---------
#
# For:
#
#     GN_LCC_xxx.graphml
#
# the matching positions file is:
#
#     GN_LCC_xxx_positions.gexf
#
# in the SAME directory.
#
# If it exists:
#     reuse it.
#
# If it does not exist:
#     generate layout and save it with exactly that name.
#
# =============================================================================


from pathlib import Path
from collections import Counter
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection


# =============================================================================
# MATPLOTLIB-VENN
# =============================================================================

try:

    from matplotlib_venn import venn3

except ImportError as exc:

    raise ImportError(
        "\nThis cell requires matplotlib-venn.\n\n"
        "Install with:\n\n"
        "    python3 -m pip install matplotlib-venn\n"
    ) from exc


# =============================================================================
# 1. GENERAL CONFIGURATION
# =============================================================================

TIMES = (
    50,
    100,
    200,
)


SNAPSHOT_NAMES = {

    50:
        "S1",

    100:
        "S2",

    200:
        "S3",
}


LANDSCAPE = ARGS.replicate


TARGET_ARCHITECTURE = "single_nk"

TARGET_MAPPING = "model_c_linear"

TARGET_LAMBDA = 0.275


# =============================================================================
# 2. ORIGINAL RAW SIMULATION SUITES
#
# Used ONLY for CCDF + Venn/Jaccard.
# =============================================================================


ANALYSIS_CASES = [

    # -------------------------------------------------------------------------
    # LOCAL-PEAK CONDITION
    # -------------------------------------------------------------------------

    {
        "label":
            "Local fitness peak",

        "short":
            "Local",

        "key":
            "local",

        "suite":
            Path(ARGS.local),

        "replicate":
            ARGS.replicate,
    },


    # -------------------------------------------------------------------------
    # EVOLVED HIGH-FITNESS CONDITION
    # -------------------------------------------------------------------------

    {
        "label":
            "Evolved high-fitness state",

        "short":
            "Evolved",

        "key":
            "evolved",

        "suite":
            Path(ARGS.evolved),

        "replicate":
            ARGS.replicate,
    },


    # -------------------------------------------------------------------------
    # EXACT GLOBAL PEAK
    # -------------------------------------------------------------------------

    {
        "label":
            "Exact global fitness peak",

        "short":
            "Global",

        "key":
            "global",

        "suite":
            Path(ARGS.global_),

        "replicate":
            ARGS.replicate,
    },
]


# =============================================================================
# 3. EXISTING FILTERED GRAPHMLS
#
# Used ONLY for visualization.
# =============================================================================

VISUAL_CASES = [

    {
        "label":
            "Local fitness peak",

        "short":
            "Local",

        "key":
            "local",

        "root":
            Path(ARGS.gn_local),
    },


    {
        "label":
            "Evolved high-fitness state",

        "short":
            "Evolved",

        "key":
            "evolved",

        "root":
            Path(ARGS.gn_evolved),
    },


    {
        "label":
            "Exact global fitness peak",

        "short":
            "Global",

        "key":
            "global",

        "root":
            Path(ARGS.gn_global),
    },
]


# =============================================================================
# 4. VISUALIZATION-ONLY FILTER
#
# IMPORTANT:
#
# This threshold is NEVER used for:
#
#     CCDF
#     LCC construction
#     Venn
#     Jaccard
#
# It affects only node drawing.
# =============================================================================

VISUAL_MIN_ABUNDANCE = 5


# =============================================================================
# 5. COLORS
# =============================================================================

TIME_COLORS = {

    50:
        "#EF4444",

    100:
        "#10B981",

    200:
        "#3B82F6",
}


TIME_RGB = {

    50:
        np.asarray(
            [1.0, 0.0, 0.0],
            dtype=float,
        ),

    100:
        np.asarray(
            [0.0, 1.0, 0.0],
            dtype=float,
        ),

    200:
        np.asarray(
            [0.0, 0.0, 1.0],
            dtype=float,
        ),
}


# =============================================================================
# 6. FIGURE STYLE
# =============================================================================

SIZE_SCALE = 5.0

EDGE_WIDTH = 0.10

EDGE_ALPHA = 0.035

NODE_ALPHA = 0.95


CCDF_POINT_SIZE = 8.0

GRID_ALPHA = 0.16

GRID_LINEWIDTH = 0.40


VENN_ALPHA = 0.50


FIGSIZE = (
    14.5,
    11.5,
)


DPI = 400


# =============================================================================
# 7. POSITION GENERATION
# =============================================================================

AUTO_GENERATE_POSITIONS = True

LAYOUT_SEED = 12345

LAYOUT_ITERATIONS = 1000

LAYOUT_SCALING_RATIO = 2.0

LAYOUT_GRAVITY = 1.0

SPRING_ITERATIONS = 500


# =============================================================================
# 8. OUTPUT
# =============================================================================

OUTPUT_DIR = Path(ARGS.out)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PDF = (
    OUTPUT_DIR
    /
    "NK_initial_state_summary_unfiltered_analysis.pdf"
)


OUTPUT_PNG = (
    OUTPUT_DIR
    /
    "NK_initial_state_summary_unfiltered_analysis.png"
)


# =============================================================================
# 9. GENERAL HELPERS
# =============================================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return float(
            default
        )


    if not np.isfinite(
        value
    ):

        return float(
            default
        )


    return value


def read_json(
    path,
):

    with Path(
        path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def abundance(
    attrs,
    t,
):

    return max(

        0.0,

        safe_float(

            attrs.get(
                f"abundance_t{t}",
                0.0,
            )
        ),
    )


def node_size(
    a,
):

    if a <= 0:

        return 0.0


    return (

        SIZE_SCALE

        *

        math.log(
            a + 1.0
        )
    )


# =============================================================================
# ============================================================================
#
# PART A
#
# READ ORIGINAL SNAPSHOTS AND BUILD EACH SNAPSHOT NETWORK INDEPENDENTLY
#
# ============================================================================
# =============================================================================


# =============================================================================
# 10. MODEL-NAME NORMALIZATION
# =============================================================================

def infer_model(
    metadata,
):

    value = str(

        metadata.get(

            "fitness_model",

            metadata.get(
                "mapping",
                "",
            ),
        )

    ).strip()


    aliases = {

        "model_c":
            "model_c_linear",

        "linear":
            "model_c_linear",

        "affine":
            "model_c_linear",

        "manrubia":
            "model_c_linear",

        "holey":
            "model_c_linear",
    }


    return aliases.get(
        value,
        value,
    )


# =============================================================================
# 11. FIND CORRECT RAW RUN
# =============================================================================

def find_raw_run(suite, replicate):
    """Directory holding snapshot_t_{50,100,200}.npz (generate_rugged.py output or a thesis suite)."""
    run_dir = find_snapshot(suite, TARGET_ARCHITECTURE, replicate, TIMES[0]).parent
    missing = [t for t in TIMES if not (run_dir / f"snapshot_t_{t}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing snapshots {missing}")
    return run_dir


def reconstruct_unique_genotypes(
    indptr,
    mutation_ids,
    reference_sequence,
):

    indptr = np.asarray(
        indptr,
        dtype=np.int64,
    )


    mutation_ids = np.asarray(
        mutation_ids,
        dtype=np.int64,
    )


    reference = np.asarray(
        reference_sequence,
        dtype=np.uint8,
    )


    n_genotypes = (
        len(
            indptr
        )
        -
        1
    )


    sequences = np.repeat(

        reference[
            None,
            :
        ],

        n_genotypes,

        axis=
            0,
    )


    mutation_counts = np.diff(
        indptr
    )


    if mutation_ids.size > 0:

        rows = np.repeat(

            np.arange(
                n_genotypes,
                dtype=np.int64,
            ),

            mutation_counts,
        )


        sites = (
            mutation_ids
            //
            3
        )


        relative_states = (

            mutation_ids
            %
            3

        ) + 1


        sequences[
            rows,
            sites,
        ] = (

            (
                reference[
                    sites
                ].astype(
                    np.int64
                )

                +

                relative_states
            )

            %

            4

        ).astype(
            np.uint8
        )


    return sequences


# =============================================================================
# 13. LOAD ORIGINAL SNAPSHOT
#
# NO ABUNDANCE FILTERING.
#
# Every represented genotype in the original snapshot is retained.
# =============================================================================

def load_original_snapshot(
    path,
):

    path = Path(
        path
    )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        counts = np.asarray(
            data[
                "counts"
            ],
            dtype=np.int64,
        )


        reference = np.asarray(
            data[
                "peak_sequence"
            ],
            dtype=np.uint8,
        )


        sequences = reconstruct_unique_genotypes(

            data[
                "indptr"
            ],

            data[
                "mutation_ids"
            ],

            reference,
        )


    if len(
        sequences
    ) != len(
        counts
    ):

        raise RuntimeError(

            "\nSnapshot sequence/count mismatch:\n"

            f"  sequences = {len(sequences):,}\n"
            f"  counts    = {len(counts):,}\n"
            f"  file      = {path}"
        )


    # -------------------------------------------------------------------------
    # Be conservative:
    #
    # retain everything genuinely represented.
    #
    # Usually counts should already all be > 0.
    # -------------------------------------------------------------------------

    present = (
        counts
        >
        0
    )


    sequences = sequences[
        present
    ]


    counts = counts[
        present
    ]


    return {

        "sequences":
            sequences,

        "counts":
            counts,

        "population_size":
            int(
                counts.sum()
            ),

        "n_genotypes":
            int(
                len(
                    counts
                )
            ),

        "reference":
            reference,
    }


# =============================================================================
# 14. BUILD EXACT HAMMING-DISTANCE-1 NETWORK
#
# This is done separately for each snapshot.
#
# NO temporal union.
# NO Grand Network.
# NO abundance filtering.
# =============================================================================

def build_hamming1_graph(
    sequences,
    counts,
):

    sequences = np.asarray(
        sequences,
        dtype=np.uint8,
    )


    counts = np.asarray(
        counts,
        dtype=np.int64,
    )


    if sequences.ndim != 2:

        raise ValueError(

            f"Expected sequences with shape (M,L), got "
            f"{sequences.shape}."
        )


    n_genotypes = int(
        sequences.shape[
            0
        ]
    )


    sequence_length = int(
        sequences.shape[
            1
        ]
    )


    G = nx.Graph()


    if n_genotypes == 0:

        return G


    # =========================================================================
    # Hash every sequence for O(1)-like neighbor lookup.
    # =========================================================================

    key_to_node = {}


    for i, seq in enumerate(
        sequences
    ):

        node = int(
            i
        )


        key = seq.tobytes()


        key_to_node[
            key
        ] = node


        G.add_node(

            node,

            abundance=
                int(
                    counts[
                        i
                    ]
                ),

            sequence_bytes=
                key.hex(),
        )


    # =========================================================================
    # Generate every single-mutant neighbor and check whether it is represented.
    #
    # alphabet = {0,1,2,3}
    # =========================================================================

    for i, sequence in enumerate(
        sequences
    ):

        work = sequence.copy()


        for position in range(
            sequence_length
        ):

            original_state = int(
                work[
                    position
                ]
            )


            for state in range(
                4
            ):

                if state == original_state:

                    continue


                work[
                    position
                ] = np.uint8(
                    state
                )


                other = key_to_node.get(
                    work.tobytes()
                )


                # -------------------------------------------------------------
                # Add each undirected edge only once.
                # -------------------------------------------------------------

                if (

                    other is not None

                    and

                    i
                    <
                    other
                ):

                    G.add_edge(
                        i,
                        other,
                    )


            work[
                position
            ] = np.uint8(
                original_state
            )


    return G


# =============================================================================
# 15. EXTRACT LARGEST CONNECTED COMPONENT
# =============================================================================

def largest_connected_component(
    G,
):

    if G.number_of_nodes() == 0:

        return nx.Graph()


    components = list(
        nx.connected_components(
            G
        )
    )


    if not components:

        return nx.Graph()


    largest_nodes = max(

        components,

        key=
            len,
    )


    return G.subgraph(
        largest_nodes
    ).copy()


# =============================================================================
# 16. EMPIRICAL DEGREE CCDF OF THE SNAPSHOT LCC
#
# P(K >= k)
# =============================================================================

def empirical_ccdf(
    degrees,
):

    degrees = np.asarray(
        degrees,
        dtype=np.int64,
    )


    if degrees.size == 0:

        return (
            np.asarray(
                [],
                dtype=float,
            ),

            np.asarray(
                [],
                dtype=float,
            ),
        )


    kmax = int(
        degrees.max()
    )


    if kmax < 1:

        return (
            np.asarray(
                [],
                dtype=float,
            ),

            np.asarray(
                [],
                dtype=float,
            ),
        )


    counts = np.bincount(

        degrees,

        minlength=
            kmax
            +
            1,
    )


    cumulative = np.cumsum(

        counts[
            ::-1
        ]

    )[
        ::-1
    ]


    k = np.arange(

        1,

        kmax
        +
        1,

        dtype=float,
    )


    ccdf = (

        cumulative[
            1:
            kmax
            +
            1
        ]

        /

        float(
            degrees.size
        )
    )


    valid = (

        np.isfinite(
            ccdf
        )

        &

        (
            ccdf
            >
            0
        )
    )


    return (

        k[
            valid
        ],

        ccdf[
            valid
        ],
    )


# =============================================================================
# 17. SEQUENCE IDENTITY FOR VENN
#
# Use the exact nucleotide-state vector as identity.
#
# bytes are ideal because:
#
#     - immutable
#     - hashable
#     - exact
# =============================================================================

def sequence_identity_set_from_lcc(
    LCC,
    sequences,
):

    return {

        sequences[
            int(
                node
            )
        ].tobytes()

        for node
        in LCC.nodes()
    }


# =============================================================================
# 18. TEMPORAL OVERLAP OF THE THREE INDEPENDENT SNAPSHOT LCCs
# =============================================================================

def overlap_statistics_from_sets(
    sets,
):

    A = sets[
        50
    ]


    B = sets[
        100
    ]


    C = sets[
        200
    ]


    union_all = (
        A
        |
        B
        |
        C
    )


    intersection_all = (
        A
        &
        B
        &
        C
    )


    def jaccard(
        X,
        Y,
    ):

        union = (
            X
            |
            Y
        )


        if not union:

            return np.nan


        return (

            len(
                X
                &
                Y
            )

            /

            len(
                union
            )
        )


    return {

        "sets":
            sets,

        "union":
            len(
                union_all
            ),

        "triple":
            len(
                intersection_all
            ),

        "triple_fraction":
            (

                len(
                    intersection_all
                )

                /

                len(
                    union_all
                )

                if union_all

                else np.nan
            ),

        "J_50_100":
            jaccard(
                A,
                B,
            ),

        "J_100_200":
            jaccard(
                B,
                C,
            ),

        "J_50_200":
            jaccard(
                A,
                C,
            ),
    }


# =============================================================================
# 19. ANALYZE ONE INITIAL CONDITION FROM ORIGINAL SNAPSHOTS
# =============================================================================

def analyze_original_condition(
    case,
):

    run_dir = find_raw_run(

        case[
            "suite"
        ],

        case[
            "replicate"
        ],
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        f"ANALYTICAL NETWORKS — {case['label']}"
    )


    print(
        "=" * 100
    )


    print(
        f"\nRaw run:\n"
        f"  {run_dir}"
    )


    snapshots = {}

    networks = {}

    lccs = {}

    ccdfs = {}

    degree_stats = {}

    lcc_sets = {}


    # =========================================================================
    # EACH SNAPSHOT IS COMPLETELY INDEPENDENT
    # =========================================================================

    for t in TIMES:

        snapshot_path = (

            run_dir
            /
            f"snapshot_t_{t}.npz"
        )


        print(
            "\n"
            +
            "-" * 90
        )


        print(
            f"t = {t}"
        )


        print(
            "-" * 90
        )


        # ---------------------------------------------------------------------
        # Load all represented genotype sequences.
        # ---------------------------------------------------------------------

        snapshot = load_original_snapshot(
            snapshot_path
        )


        sequences = snapshot[
            "sequences"
        ]


        counts = snapshot[
            "counts"
        ]


        print(
            f"Population size     : "
            f"{snapshot['population_size']:,}"
        )


        print(
            f"Genotype types      : "
            f"{snapshot['n_genotypes']:,}"
        )


        print(
            f"Minimum abundance   : "
            f"{counts.min():,}"
        )


        print(
            f"Maximum abundance   : "
            f"{counts.max():,}"
        )


        # ---------------------------------------------------------------------
        # Build full snapshot network from ALL sequences.
        # ---------------------------------------------------------------------

        G = build_hamming1_graph(

            sequences,

            counts,
        )


        print(
            f"Snapshot graph      : "
            f"{G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )


        # ---------------------------------------------------------------------
        # Extract LCC OF THIS SNAPSHOT.
        # ---------------------------------------------------------------------

        LCC = largest_connected_component(
            G
        )


        print(
            f"Snapshot LCC        : "
            f"{LCC.number_of_nodes():,} nodes, "
            f"{LCC.number_of_edges():,} edges"
        )


        if G.number_of_nodes() > 0:

            print(
                f"LCC node fraction   : "
                f"{LCC.number_of_nodes() / G.number_of_nodes():.4f}"
            )


        # ---------------------------------------------------------------------
        # Degree distribution inside THIS LCC.
        # ---------------------------------------------------------------------

        degrees = np.asarray(

            [

                degree

                for _, degree
                in LCC.degree()
            ],

            dtype=np.int64,
        )


        k, ccdf = empirical_ccdf(
            degrees
        )


        # ---------------------------------------------------------------------
        # Sequence identities of THIS LCC for the temporal Venn.
        # ---------------------------------------------------------------------

        lcc_set = sequence_identity_set_from_lcc(

            LCC,

            sequences,
        )


        snapshots[
            t
        ] = snapshot


        networks[
            t
        ] = G


        lccs[
            t
        ] = LCC


        ccdfs[
            t
        ] = (
            k,
            ccdf,
        )


        lcc_sets[
            t
        ] = lcc_set


        degree_stats[
            t
        ] = {

            "full_nodes":
                int(
                    G.number_of_nodes()
                ),

            "full_edges":
                int(
                    G.number_of_edges()
                ),

            "lcc_nodes":
                int(
                    LCC.number_of_nodes()
                ),

            "lcc_edges":
                int(
                    LCC.number_of_edges()
                ),

            "mean_degree":
                (
                    float(
                        degrees.mean()
                    )

                    if degrees.size

                    else np.nan
                ),

            "median_degree":
                (
                    float(
                        np.median(
                            degrees
                        )
                    )

                    if degrees.size

                    else np.nan
                ),

            "max_degree":
                (
                    int(
                        degrees.max()
                    )

                    if degrees.size

                    else 0
                ),
        }


        print(
            f"<k> in LCC          : "
            f"{degree_stats[t]['mean_degree']:.4f}"
        )


        print(
            f"k_max in LCC        : "
            f"{degree_stats[t]['max_degree']}"
        )


    # =========================================================================
    # VENN / JACCARD BETWEEN THE THREE SNAPSHOT LCCs
    # =========================================================================

    overlap = overlap_statistics_from_sets(
        lcc_sets
    )


    return {

        **case,

        "run_dir":
            run_dir,

        "snapshots":
            snapshots,

        "networks":
            networks,

        "lccs":
            lccs,

        "ccdfs":
            ccdfs,

        "lcc_sets":
            lcc_sets,

        "degree_stats":
            degree_stats,

        "overlap":
            overlap,
    }


# =============================================================================
# ============================================================================
#
# PART B
#
# EXISTING FILTERED GRAPHML FOR VISUALIZATION ONLY
#
# ============================================================================
# =============================================================================


# =============================================================================
# 20. TEMPORAL COLOR FOR VISUAL GRAPH
# =============================================================================

def temporal_log_color(
    attrs,
):

    alpha = {

        t:
            math.log(

                abundance(
                    attrs,
                    t,
                )

                +

                1.0
            )

        for t
        in TIMES
    }


    total = sum(
        alpha.values()
    )


    if total <= 0:

        return (
            0.0,
            0.0,
            0.0,
        )


    rgb = np.zeros(
        3,
        dtype=float,
    )


    for t in TIMES:

        rgb += (

            alpha[
                t
            ]

            /

            total

        ) * TIME_RGB[
            t
        ]


    return tuple(

        np.clip(
            rgb,
            0.0,
            1.0,
        )
    )


# =============================================================================
# 21. FIND VISUAL GRAPHML
# =============================================================================

def find_landscape_graphml(
    root,
    landscape,
):

    root = Path(
        root
    )


    if not root.exists():

        raise FileNotFoundError(

            "\nVisualization GraphML directory does not exist:\n"
            f"  {root}"
        )


    tag = (
        f"landscape_{landscape:03d}"
    )


    candidates = [

        path

        for path
        in root.rglob(
            "GN_LCC_*.graphml"
        )

        if (

            tag
            in str(
                path
            )

            or

            f"landscape_{landscape}"
            in str(
                path
            )
        )
    ]


    if not candidates:

        raise FileNotFoundError(

            f"\nNo GraphML found for {tag} under:\n"
            f"  {root}"
        )


    candidates = sorted(
        candidates
    )


    exact = [

        path

        for path
        in candidates

        if (

            tag
            in path.parent.name

            or

            tag
            in path.stem
        )
    ]


    if len(
        exact
    ) == 1:

        return exact[
            0
        ]


    return candidates[
        0
    ]


# =============================================================================
# 22. EXACT POSITION FILE PATH
#
# xxx.graphml
#
# ->
#
# xxx_positions.gexf
# =============================================================================

def position_file(
    graphml,
):

    graphml = Path(
        graphml
    )


    return graphml.with_name(

        graphml.stem

        +

        "_positions.gexf"
    )


# =============================================================================
# 23. GEXF/XML HELPERS
# =============================================================================

def local_name(
    tag,
):

    tag = str(
        tag
    )


    if "}" in tag:

        return tag.rsplit(
            "}",
            1,
        )[
            1
        ]


    if ":" in tag:

        return tag.rsplit(
            ":",
            1,
        )[
            1
        ]


    return tag


def normalize_name(
    value,
):

    return (

        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
        .replace(
            ".",
            "_",
        )
    )


def position_name_score(
    name,
):

    name = normalize_name(
        name
    )


    exact_x = {

        "x",
        "pos_x",
        "position_x",
        "layout_x",
        "coord_x",
        "coordinate_x",
        "x_position",
        "xpos",
    }


    exact_y = {

        "y",
        "pos_y",
        "position_y",
        "layout_y",
        "coord_y",
        "coordinate_y",
        "y_position",
        "ypos",
    }


    if name in exact_x:

        return (
            "x",
            100,
        )


    if name in exact_y:

        return (
            "y",
            100,
        )


    if (

        name.endswith(
            "_x"
        )

        and

        any(

            word
            in name

            for word
            in (
                "pos",
                "position",
                "layout",
                "coord",
            )
        )
    ):

        return (
            "x",
            80,
        )


    if (

        name.endswith(
            "_y"
        )

        and

        any(

            word
            in name

            for word
            in (
                "pos",
                "position",
                "layout",
                "coord",
            )
        )
    ):

        return (
            "y",
            80,
        )


    return (
        None,
        0,
    )


# =============================================================================
# 24. POSITION METHOD 1 — NETWORKX
# =============================================================================

def positions_from_networkx(
    path,
):

    try:

        P = nx.read_gexf(
            path
        )

    except Exception:

        return {}


    positions = {}


    for node, attrs in P.nodes(
        data=True
    ):

        node_id = str(
            node
        )


        viz = attrs.get(
            "viz"
        )


        if isinstance(
            viz,
            dict,
        ):

            pos = viz.get(
                "position"
            )


            if isinstance(
                pos,
                dict,
            ):

                x = safe_float(
                    pos.get(
                        "x"
                    ),
                    np.nan,
                )


                y = safe_float(
                    pos.get(
                        "y"
                    ),
                    np.nan,
                )


                if (

                    np.isfinite(
                        x
                    )

                    and

                    np.isfinite(
                        y
                    )
                ):

                    positions[
                        node_id
                    ] = (
                        float(
                            x
                        ),
                        float(
                            y
                        ),
                    )


                    continue


        for xkey, ykey in [

            (
                "x",
                "y",
            ),

            (
                "pos_x",
                "pos_y",
            ),

            (
                "position_x",
                "position_y",
            ),

            (
                "layout_x",
                "layout_y",
            ),

            (
                "coord_x",
                "coord_y",
            ),

        ]:

            x = safe_float(
                attrs.get(
                    xkey
                ),
                np.nan,
            )


            y = safe_float(
                attrs.get(
                    ykey
                ),
                np.nan,
            )


            if (

                np.isfinite(
                    x
                )

                and

                np.isfinite(
                    y
                )
            ):

                positions[
                    node_id
                ] = (
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                )


                break


    return positions


# =============================================================================
# 25. ATTRIBUTE DEFINITIONS
# =============================================================================

def read_attribute_definitions(
    root,
):

    mapping = {}


    for elem in root.iter():

        if local_name(
            elem.tag
        ) != "attribute":

            continue


        attr_id = elem.attrib.get(
            "id"
        )


        title = elem.attrib.get(

            "title",

            elem.attrib.get(
                "name"
            ),
        )


        if (

            attr_id is not None

            and

            title is not None
        ):

            mapping[
                str(
                    attr_id
                )
            ] = str(
                title
            )


    return mapping


# =============================================================================
# 26. RAW XML POSITION ELEMENTS
# =============================================================================

def positions_from_xml_position_elements(
    root,
):

    positions = {}


    for node in root.iter():

        if local_name(
            node.tag
        ) != "node":

            continue


        node_id = node.attrib.get(
            "id"
        )


        if node_id is None:

            continue


        for child in node.iter():

            name = local_name(
                child.tag
            ).lower()


            if (

                "position"
                not in name

                and

                name
                not in {

                    "pos",
                    "coordinate",
                    "coordinates",
                    "layout",
                }
            ):

                continue


            x = safe_float(
                child.attrib.get(
                    "x"
                ),
                np.nan,
            )


            y = safe_float(
                child.attrib.get(
                    "y"
                ),
                np.nan,
            )


            if (

                np.isfinite(
                    x
                )

                and

                np.isfinite(
                    y
                )
            ):

                positions[
                    str(
                        node_id
                    )
                ] = (
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                )


                break


    return positions


# =============================================================================
# 27. RAW NODE ATTRIBUTES
# =============================================================================

def positions_from_raw_node_attributes(
    root,
):

    positions = {}


    for node in root.iter():

        if local_name(
            node.tag
        ) != "node":

            continue


        node_id = node.attrib.get(
            "id"
        )


        if node_id is None:

            continue


        values = {}


        for key, value in node.attrib.items():

            axis, score = position_name_score(
                local_name(
                    key
                )
            )


            if axis is None:

                continue


            numeric = safe_float(
                value,
                np.nan,
            )


            if not np.isfinite(
                numeric
            ):

                continue


            current = values.get(
                axis
            )


            if (

                current is None

                or

                score
                >
                current[
                    0
                ]
            ):

                values[
                    axis
                ] = (
                    score,
                    float(
                        numeric
                    ),
                )


        if (

            "x"
            in values

            and

            "y"
            in values
        ):

            positions[
                str(
                    node_id
                )
            ] = (

                values[
                    "x"
                ][
                    1
                ],

                values[
                    "y"
                ][
                    1
                ],
            )


    return positions


# =============================================================================
# 28. GEXF ATTVALUES
# =============================================================================

def positions_from_attvalues(
    root,
    definitions,
):

    positions = {}


    for node in root.iter():

        if local_name(
            node.tag
        ) != "node":

            continue


        node_id = node.attrib.get(
            "id"
        )


        if node_id is None:

            continue


        values = {}


        for elem in node.iter():

            if local_name(
                elem.tag
            ) != "attvalue":

                continue


            field_id = (

                elem.attrib.get(
                    "for"
                )

                or

                elem.attrib.get(
                    "id"
                )

                or

                elem.attrib.get(
                    "name"
                )
            )


            if field_id is None:

                continue


            field_name = definitions.get(

                str(
                    field_id
                ),

                str(
                    field_id
                ),
            )


            axis, score = position_name_score(
                field_name
            )


            if axis is None:

                continue


            numeric = safe_float(

                elem.attrib.get(
                    "value"
                ),

                np.nan,
            )


            if not np.isfinite(
                numeric
            ):

                continue


            current = values.get(
                axis
            )


            if (

                current is None

                or

                score
                >
                current[
                    0
                ]
            ):

                values[
                    axis
                ] = (
                    score,
                    float(
                        numeric
                    ),
                )


        if (

            "x"
            in values

            and

            "y"
            in values
        ):

            positions[
                str(
                    node_id
                )
            ] = (

                values[
                    "x"
                ][
                    1
                ],

                values[
                    "y"
                ][
                    1
                ],
            )


    return positions


# =============================================================================
# 29. RECURSIVE XML POSITION SEARCH
# =============================================================================

def positions_from_recursive_search(
    root,
    definitions,
):

    positions = {}


    for node in root.iter():

        if local_name(
            node.tag
        ) != "node":

            continue


        node_id = node.attrib.get(
            "id"
        )


        if node_id is None:

            continue


        candidates = {

            "x":
                [],

            "y":
                [],
        }


        for elem in node.iter():

            element_name = local_name(
                elem.tag
            )


            for key, value in elem.attrib.items():

                axis, score = position_name_score(
                    local_name(
                        key
                    )
                )


                numeric = safe_float(
                    value,
                    np.nan,
                )


                if (

                    axis is None

                    or

                    not np.isfinite(
                        numeric
                    )
                ):

                    continue


                if (
                    "position"
                    in element_name.lower()
                ):

                    score += 50


                candidates[
                    axis
                ].append(
                    (
                        score,
                        float(
                            numeric
                        ),
                    )
                )


            if local_name(
                elem.tag
            ) == "attvalue":

                field_id = (

                    elem.attrib.get(
                        "for"
                    )

                    or

                    elem.attrib.get(
                        "id"
                    )
                )


                if field_id is not None:

                    field_name = definitions.get(

                        str(
                            field_id
                        ),

                        str(
                            field_id
                        ),
                    )


                    axis, score = position_name_score(
                        field_name
                    )


                    numeric = safe_float(

                        elem.attrib.get(
                            "value"
                        ),

                        np.nan,
                    )


                    if (

                        axis is not None

                        and

                        np.isfinite(
                            numeric
                        )
                    ):

                        candidates[
                            axis
                        ].append(
                            (
                                score
                                +
                                20,

                                float(
                                    numeric
                                ),
                            )
                        )


        if (

            candidates[
                "x"
            ]

            and

            candidates[
                "y"
            ]
        ):

            best_x = max(

                candidates[
                    "x"
                ],

                key=lambda z:
                    z[
                        0
                    ],
            )


            best_y = max(

                candidates[
                    "y"
                ],

                key=lambda z:
                    z[
                        0
                    ],
            )


            positions[
                str(
                    node_id
                )
            ] = (

                best_x[
                    1
                ],

                best_y[
                    1
                ],
            )


    return positions


# =============================================================================
# 30. ROBUST POSITION EXTRACTION
# =============================================================================

def robust_extract_positions(
    gexf,
    G,
):

    gexf = Path(
        gexf
    )


    networkx_positions = positions_from_networkx(
        gexf
    )


    tree = ET.parse(
        gexf
    )


    root = tree.getroot()


    definitions = read_attribute_definitions(
        root
    )


    sources = [

        (
            "networkx",
            networkx_positions,
        ),

        (
            "xml_position",
            positions_from_xml_position_elements(
                root
            ),
        ),

        (
            "node_attributes",
            positions_from_raw_node_attributes(
                root
            ),
        ),

        (
            "attvalues",
            positions_from_attvalues(
                root,
                definitions,
            ),
        ),

        (
            "recursive",
            positions_from_recursive_search(
                root,
                definitions,
            ),
        ),
    ]


    merged = {}

    provenance = {}


    for source_name, source in sources:

        for node, xy in source.items():

            if node not in merged:

                merged[
                    node
                ] = xy


                provenance[
                    node
                ] = source_name


    required = {

        str(
            node
        )

        for node
        in G.nodes()
    }


    available = set(
        merged
    )


    missing = sorted(

        required
        -
        available
    )


    print(
        f"\nPosition GEXF:"
    )


    print(
        f"  {gexf}"
    )


    print(
        f"  required nodes : {len(required):,}"
    )


    print(
        f"  matched nodes  : {len(required & available):,}"
    )


    print(
        f"  missing nodes  : {len(missing):,}"
    )


    if provenance:

        print(
            "  sources:"
        )


        for source, count in Counter(
            provenance.values()
        ).most_common():

            print(
                f"    {source:18s}: {count:,}"
            )


    if missing:

        raise RuntimeError(

            "\nCould not recover positions for every GraphML node.\n"
            f"First missing IDs:\n"
            f"  {missing[:20]}"
        )


    return {

        node:
            merged[
                str(
                    node
                )
            ]

        for node
        in G.nodes()
    }


# =============================================================================
# 31. GENERATE LAYOUT ONLY WHEN POSITION GEXF IS MISSING
# =============================================================================

def generate_positions(
    G,
):

    if G.number_of_nodes() == 0:

        raise RuntimeError(
            "Cannot layout an empty graph."
        )


    if hasattr(
        nx,
        "forceatlas2_layout",
    ):

        try:

            return nx.forceatlas2_layout(

                G,

                max_iter=
                    LAYOUT_ITERATIONS,

                scaling_ratio=
                    LAYOUT_SCALING_RATIO,

                gravity=
                    LAYOUT_GRAVITY,

                seed=
                    LAYOUT_SEED,
            )


        except Exception as exc:

            print(
                f"\nForceAtlas2 failed: {exc}"
            )


    print(
        "\nUsing spring_layout fallback."
    )


    return nx.spring_layout(

        G,

        seed=
            LAYOUT_SEED,

        iterations=
            SPRING_ITERATIONS,
    )


# =============================================================================
# 32. SAVE POSITION GEXF
# =============================================================================

def save_positions_gexf(
    path,
    G,
    positions,
):

    P = nx.Graph()


    for node in G.nodes():

        x, y = positions[
            node
        ]


        P.add_node(

            str(
                node
            ),

            x=
                float(
                    x
                ),

            y=
                float(
                    y
                ),

            viz=
                {

                    "position":
                        {

                            "x":
                                float(
                                    x
                                ),

                            "y":
                                float(
                                    y
                                ),

                            "z":
                                0.0,
                        }
                },
        )


    for u, v in G.edges():

        P.add_edge(

            str(
                u
            ),

            str(
                v
            ),
        )


    nx.write_gexf(
        P,
        path,
    )


# =============================================================================
# 33. GET OR CREATE POSITIONS
# =============================================================================

def load_positions(
    graphml,
    G,
):

    gexf = position_file(
        graphml
    )


    if gexf.exists():

        print(
            "\nExisting position file found — reusing it."
        )


        return (

            robust_extract_positions(
                gexf,
                G,
            ),

            gexf,
        )


    if not AUTO_GENERATE_POSITIONS:

        raise FileNotFoundError(

            f"\nMissing position file:\n"
            f"  {gexf}"
        )


    print(
        "\nPosition GEXF not found."
    )


    print(
        "Generating a new layout..."
    )


    positions = generate_positions(
        G
    )


    save_positions_gexf(

        gexf,

        G,

        positions,
    )


    print(
        f"Saved:\n"
        f"  {gexf}"
    )


    return (

        robust_extract_positions(
            gexf,
            G,
        ),

        gexf,
    )


# =============================================================================
# 34. VISUAL POSITION LIMITS
# =============================================================================

def position_limits(
    positions,
    padding=0.025,
):

    xy = np.asarray(

        list(
            positions.values()
        ),

        dtype=float,
    )


    xmin = float(
        xy[
            :,
            0
        ].min()
    )


    xmax = float(
        xy[
            :,
            0
        ].max()
    )


    ymin = float(
        xy[
            :,
            1
        ].min()
    )


    ymax = float(
        xy[
            :,
            1
        ].max()
    )


    dx = max(
        xmax
        -
        xmin,
        1.0,
    )


    dy = max(
        ymax
        -
        ymin,
        1.0,
    )


    return (

        (
            xmin
            -
            padding
            *
            dx,

            xmax
            +
            padding
            *
            dx,
        ),

        (
            ymin
            -
            padding
            *
            dy,

            ymax
            +
            padding
            *
            dy,
        ),
    )


# =============================================================================
# 35. LOAD VISUALIZATION CASE
# =============================================================================

def load_visual_case(
    case,
):

    graphml = find_landscape_graphml(

        case[
            "root"
        ],

        LANDSCAPE,
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        f"VISUALIZATION — {case['label']}"
    )


    print(
        "=" * 100
    )


    print(
        f"\nGraphML:\n"
        f"  {graphml}"
    )


    G = nx.read_graphml(
        graphml
    )


    if G.is_directed():

        G = G.to_undirected()


    positions, positions_gexf = load_positions(

        graphml,

        G,
    )


    xlim, ylim = position_limits(
        positions
    )


    segments = np.asarray(

        [

            (
                positions[
                    u
                ],

                positions[
                    v
                ],
            )

            for u, v
            in G.edges()
        ],

        dtype=float,
    )


    return {

        **case,

        "graphml":
            graphml,

        "positions_gexf":
            positions_gexf,

        "G":
            G,

        "positions":
            positions,

        "xlim":
            xlim,

        "ylim":
            ylim,

        "segments":
            segments,
    }


# =============================================================================
# 36. DRAW FILTERED VISUAL NETWORK
# =============================================================================

def draw_network(
    ax,
    visual_case,
    t,
):

    G = visual_case[
        "G"
    ]


    positions = visual_case[
        "positions"
    ]


    segments = visual_case[
        "segments"
    ]


    # =========================================================================
    # Full GN-LCC skeleton, as before.
    # =========================================================================

    if len(
        segments
    ):

        ax.add_collection(

            LineCollection(

                segments,

                colors=
                    "#AAAAAA",

                linewidths=
                    EDGE_WIDTH,

                alpha=
                    EDGE_ALPHA,

                zorder=
                    1,
            )
        )


    # =========================================================================
    # VISUAL FILTER ONLY:
    #
    #     abundance > 5
    # =========================================================================

    visible_nodes = [

        node

        for node, attrs
        in G.nodes(
            data=True
        )

        if abundance(
            attrs,
            t,
        )
        >
        VISUAL_MIN_ABUNDANCE
    ]


    if visible_nodes:

        xy = np.asarray(

            [

                positions[
                    node
                ]

                for node
                in visible_nodes
            ],

            dtype=float,
        )


        sizes = [

            node_size(

                abundance(
                    G.nodes[
                        node
                    ],
                    t,
                )
            )

            for node
            in visible_nodes
        ]


        colors = [

            temporal_log_color(
                G.nodes[
                    node
                ]
            )

            for node
            in visible_nodes
        ]


        ax.scatter(

            xy[
                :,
                0
            ],

            xy[
                :,
                1
            ],

            s=
                sizes,

            c=
                colors,

            alpha=
                NODE_ALPHA,

            edgecolors=
                "none",

            linewidths=
                0,

            rasterized=
                True,

            zorder=
                2,
        )


    ax.set_xlim(
        *visual_case[
            "xlim"
        ]
    )


    ax.set_ylim(
        *visual_case[
            "ylim"
        ]
    )


    ax.set_aspect(
        "equal",
        adjustable=
            "box",
    )


    ax.set_axis_off()


# =============================================================================
# ============================================================================
#
# PART C
#
# RUN ANALYSIS + LOAD VISUALIZATION
#
# ============================================================================
# =============================================================================


# =============================================================================
# 37. ANALYTICAL NETWORKS
# =============================================================================

analysis_results = {}


for case in ANALYSIS_CASES:

    result = analyze_original_condition(
        case
    )


    analysis_results[
        case[
            "key"
        ]
    ] = result


# =============================================================================
# 38. VISUAL NETWORKS
# =============================================================================

visual_results = {}


for case in VISUAL_CASES:

    result = load_visual_case(
        case
    )


    visual_results[
        case[
            "key"
        ]
    ] = result


# =============================================================================
# 39. COMMON CCDF LIMITS
#
# Derived ONLY from the independently reconstructed snapshot LCCs.
# =============================================================================

all_k = []

all_y = []


for result in analysis_results.values():

    for t in TIMES:

        k, y = result[
            "ccdfs"
        ][
            t
        ]


        if len(
            k
        ):

            all_k.extend(
                k.tolist()
            )


            all_y.extend(
                y.tolist()
            )


if not all_k:

    raise RuntimeError(
        "No analytical CCDF data were produced."
    )


CCDF_XLIM = (

    max(

        0.8,

        min(
            all_k
        )
        *
        0.8,
    ),

    max(
        all_k
    )
    *
    1.25,
)


CCDF_YLIM = (

    max(

        min(
            all_y
        )
        *
        0.6,

        1e-8,
    ),

    min(

        1.2,

        max(
            all_y
        )
        *
        1.25,
    ),
)


# =============================================================================
# 40. DRAW ANALYTICAL CCDF
# =============================================================================

def draw_ccdf(
    ax,
    k,
    ccdf,
    t,
):

    if len(
        k
    ):

        ax.scatter(

            k,
            ccdf,

            s=
                CCDF_POINT_SIZE,

            color=
                TIME_COLORS[
                    t
                ],

            alpha=
                0.80,

            edgecolors=
                "none",

            linewidths=
                0,

            rasterized=
                True,
        )


    ax.set_xscale(
        "log"
    )


    ax.set_yscale(
        "log"
    )


    ax.set_xlim(
        *CCDF_XLIM
    )


    ax.set_ylim(
        *CCDF_YLIM
    )


    ax.grid(

        True,

        which=
            "major",

        linewidth=
            GRID_LINEWIDTH,

        alpha=
            GRID_ALPHA,
    )


    ax.tick_params(

        axis=
            "both",

        which=
            "both",

        labelsize=
            6.5,

        width=
            0.5,

        length=
            2.5,
    )


    ax.set_xlabel(

        r"Degree $k$",

        fontsize=
            7.5,

        labelpad=
            1,
    )


    ax.set_ylabel(

        r"$P(K\geq k)$",

        fontsize=
            7.5,

        labelpad=
            1,
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            0.6
        )


# =============================================================================
# 41. DRAW ANALYTICAL VENN
#
# Sets = genotype sequences belonging to the independently computed LCC
# at t=50,100,200.
# =============================================================================

def draw_overlap(
    ax,
    analysis_case,
):

    sets_dict = analysis_case[
        "lcc_sets"
    ]


    sets = [

        sets_dict[
            t
        ]

        for t
        in TIMES
    ]


    v = venn3(

        subsets=
            sets,

        set_labels=
            (
                "S1",
                "S2",
                "S3",
            ),

        set_colors=
            (

                TIME_COLORS[
                    50
                ],

                TIME_COLORS[
                    100
                ],

                TIME_COLORS[
                    200
                ],
            ),

        alpha=
            VENN_ALPHA,

        ax=
            ax,
    )


    if v is not None:

        if v.set_labels:

            for label, t in zip(
                v.set_labels,
                TIMES,
            ):

                if label is not None:

                    label.set_color(
                        TIME_COLORS[
                            t
                        ]
                    )


                    label.set_fontsize(
                        8
                    )


                    label.set_fontweight(
                        "bold"
                    )


        if v.subset_labels:

            for label in v.subset_labels:

                if label is not None:

                    label.set_fontsize(
                        7
                    )


    stats = analysis_case[
        "overlap"
    ]


    def fmt(
        value,
    ):

        if np.isfinite(
            value
        ):

            return f"{value:.2f}"


        return "NA"


    summary_text = (

        rf"$J_{{50,100}}={fmt(stats['J_50_100'])}$"
        "\n"

        rf"$J_{{100,200}}={fmt(stats['J_100_200'])}$"
        "\n"

        rf"$J_{{50,200}}={fmt(stats['J_50_200'])}$"
    )


    ax.text(

        0.50,

        -0.08,

        summary_text,

        transform=
            ax.transAxes,

        ha=
            "center",

        va=
            "top",

        fontsize=
            7.5,
    )


    ax.set_axis_off()


# =============================================================================
# ============================================================================
#
# PART D
#
# CREATE FIGURE
#
# ============================================================================
# =============================================================================


# =============================================================================
# 42. FIGURE LAYOUT
# =============================================================================

fig = plt.figure(

    figsize=
        FIGSIZE,

    facecolor=
        "white",
)


outer = fig.add_gridspec(

    nrows=
        3,

    ncols=
        4,

    width_ratios=
        [
            1.0,
            1.0,
            1.0,
            0.78,
        ],

    height_ratios=
        [
            1.0,
            1.0,
            1.0,
        ],

    wspace=
        0.10,

    hspace=
        0.23,
)


# =============================================================================
# 43. HEADERS
# =============================================================================

COLUMN_HEADERS = [

    r"S1, $t=50$",

    r"S2, $t=100$",

    r"S3, $t=200$",

    "Temporal overlap",
]


HEADER_X = [

    0.205,

    0.445,

    0.685,

    0.895,
]


for x, text in zip(

    HEADER_X,

    COLUMN_HEADERS,
):

    fig.text(

        x,

        0.975,

        text,

        ha=
            "center",

        va=
            "top",

        fontsize=
            12,
    )


# =============================================================================
# 44. ROW INFORMATION
# =============================================================================

PANEL_LABELS = [

    "(a)",

    "(b)",

    "(c)",
]


ROW_CENTERS = [

    0.82,

    0.50,

    0.18,
]


ROW_KEYS = [

    "local",

    "evolved",

    "global",
]


# =============================================================================
# 45. DRAW ROWS
# =============================================================================

for row, key in enumerate(
    ROW_KEYS
):

    analysis_case = analysis_results[
        key
    ]


    visual_case = visual_results[
        key
    ]


    # -------------------------------------------------------------------------
    # Panel label
    # -------------------------------------------------------------------------

    fig.text(

        0.018,

        ROW_CENTERS[
            row
        ]
        +
        0.125,

        PANEL_LABELS[
            row
        ],

        ha=
            "left",

        va=
            "center",

        fontsize=
            10,

        fontweight=
            "semibold",
    )


    # -------------------------------------------------------------------------
    # Condition label
    # -------------------------------------------------------------------------

    fig.text(

        0.025,

        ROW_CENTERS[
            row
        ],

        analysis_case[
            "label"
        ],

        rotation=
            90,

        ha=
            "center",

        va=
            "center",

        fontsize=
            11,

        fontweight=
            "semibold",
    )


    # =========================================================================
    # t=50 / t=100 / t=200
    # =========================================================================

    for col, t in enumerate(
        TIMES
    ):

        inner = outer[
            row,
            col
        ].subgridspec(

            2,
            1,

            height_ratios=
                [
                    3.6,
                    1.15,
                ],

            hspace=
                0.035,
        )


        # ---------------------------------------------------------------------
        # NETWORK VISUALIZATION
        #
        # Source:
        #
        #     existing filtered GN-LCC GraphML
        #
        # Node display threshold:
        #
        #     abundance > 5
        # ---------------------------------------------------------------------

        ax_net = fig.add_subplot(

            inner[
                0,
                0
            ]
        )


        draw_network(

            ax_net,

            visual_case,

            t,
        )


        ax_net.text(

            0.02,

            0.98,

            SNAPSHOT_NAMES[
                t
            ],

            transform=
                ax_net.transAxes,

            ha=
                "left",

            va=
                "top",

            fontsize=
                7.5,

            fontweight=
                "bold",

            color=
                TIME_COLORS[
                    t
                ],
        )


        # ---------------------------------------------------------------------
        # CCDF
        #
        # Source:
        #
        #     ORIGINAL RAW SNAPSHOT
        #
        # Network:
        #
        #     independently reconstructed G_t
        #
        # Distribution:
        #
        #     degree distribution of LCC(G_t)
        #
        # NO abundance filter.
        # ---------------------------------------------------------------------

        ax_ccdf = fig.add_subplot(

            inner[
                1,
                0
            ]
        )


        k, ccdf = analysis_case[
            "ccdfs"
        ][
            t
        ]


        draw_ccdf(

            ax_ccdf,

            k,

            ccdf,

            t,
        )


    # =========================================================================
    # TEMPORAL OVERLAP
    #
    # Sets:
    #
    #     sequences in LCC(G_50)
    #     sequences in LCC(G_100)
    #     sequences in LCC(G_200)
    #
    # NO abundance filter.
    # =========================================================================

    ax_overlap = fig.add_subplot(

        outer[
            row,
            3
        ]
    )


    draw_overlap(

        ax_overlap,

        analysis_case,
    )


# =============================================================================
# 46. FINAL LAYOUT
# =============================================================================

fig.subplots_adjust(

    left=
        0.085,

    right=
        0.985,

    top=
        0.945,

    bottom=
        0.045,
)


# =============================================================================
# 47. SAVE
# =============================================================================

fig.savefig(

    OUTPUT_PDF,

    bbox_inches=
        "tight",

    facecolor=
        "white",
)


fig.savefig(

    OUTPUT_PNG,

    dpi=
        DPI,

    bbox_inches=
        "tight",

    facecolor=
        "white",
)


print(
    "\nSaved:"
)


print(
    f"  {OUTPUT_PDF}"
)


print(
    f"  {OUTPUT_PNG}"
)


# =============================================================================
# 48. DETAILED ANALYTICAL SUMMARY
# =============================================================================

print(
    "\n\n"
    +
    "=" * 100
)


print(
    "ANALYTICAL NETWORK SUMMARY"
)


print(
    "=" * 100
)


print(
    "\nCCDF definition:"
)


print(
    "  original snapshot"
)


print(
    "  -> all represented sequences"
)


print(
    "  -> independent Hamming-1 graph"
)


print(
    "  -> snapshot LCC"
)


print(
    "  -> degree CCDF"
)


print(
    "\nVenn definition:"
)


print(
    "  genotype sets belonging to the three independent snapshot LCCs"
)


for key in ROW_KEYS:

    result = analysis_results[
        key
    ]


    print(
        "\n"
        +
        "-" * 88
    )


    print(
        result[
            "label"
        ]
    )


    print(
        "-" * 88
    )


    for t in TIMES:

        stats = result[
            "degree_stats"
        ][
            t
        ]


        print(
            f"\nt={t}"
        )


        print(
            f"  full snapshot network : "
            f"{stats['full_nodes']:,} nodes, "
            f"{stats['full_edges']:,} edges"
        )


        print(
            f"  snapshot LCC          : "
            f"{stats['lcc_nodes']:,} nodes, "
            f"{stats['lcc_edges']:,} edges"
        )


        print(
            f"  <k> LCC               : "
            f"{stats['mean_degree']:.4f}"
        )


        print(
            f"  k_max LCC             : "
            f"{stats['max_degree']}"
        )


    overlap = result[
        "overlap"
    ]


    print(
        "\nTemporal overlap of snapshot LCCs:"
    )


    for t in TIMES:

        print(

            f"  t={t:3d}: "
            f"{len(overlap['sets'][t]):,} sequences"
        )


    print(

        f"  union                  = "
        f"{overlap['union']:,}"
    )


    print(

        f"  present in all 3 LCCs = "
        f"{overlap['triple']:,}"
    )


    print(

        f"  triple / union         = "
        f"{overlap['triple_fraction']:.4f}"
    )


    print(

        f"  J(50,100)              = "
        f"{overlap['J_50_100']:.4f}"
    )


    print(

        f"  J(100,200)             = "
        f"{overlap['J_100_200']:.4f}"
    )


    print(

        f"  J(50,200)              = "
        f"{overlap['J_50_200']:.4f}"
    )


# =============================================================================
# 49. VISUALIZATION SUMMARY
# =============================================================================

print(
    "\n\n"
    +
    "=" * 100
)


print(
    "VISUALIZATION SUMMARY"
)


print(
    "=" * 100
)


print(
    f"\nNode-display threshold:"
)


print(
    f"  abundance > {VISUAL_MIN_ABUNDANCE}"
)


print(
    "\nThis threshold affects ONLY the network drawings."
)


print(
    "It does NOT affect CCDF, LCC construction, Venn, or Jaccard."
)


for key in ROW_KEYS:

    result = visual_results[
        key
    ]


    print(
        f"\n{result['label']}"
    )


    print(
        f"  GraphML   : {result['graphml']}"
    )


    print(
        f"  positions : {result['positions_gexf']}"
    )


plt.show()