#!/usr/bin/env python3
"""
export_grand_network.py -- "grand network" GraphML used to draw Fig. NK_initial_state_summary
==========================================================================================

For one replicate of a rugged-landscape run, takes the snapshots at t = 50, 100, 200, keeps every genotype
with abundance > --min-abundance (default 5) in at least one of them, joins all genotypes at Hamming
distance one, and writes the largest connected component as

    <out>/<architecture>/model_c_linear/lambda_<lam>/landscape_<rep>/GN_LCC_<...>.graphml

with node attributes abundance_t50/t100/t200, present_t*, is_root_t*, distance_to_origin and a
temporal colour. This network is used ONLY for the drawings of fig_initial_states.py; the CCDFs and
Jaccard indices of that figure are computed from the unfiltered snapshots.
Core functions extracted from old_codes/export_GN_LCC_graphml_only.py.

    python export_grand_network.py --data Data/nk_local   --out Figures/GN_local   --replicate 0
    python export_grand_network.py --data Data/nk_evolved --out Figures/GN_evolved --replicate 0
    python export_grand_network.py --data Data/nk_global  --out Figures/GN_global  --replicate 0
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
import numpy as np
import networkx as nx
SNAPSHOT_TIMES = (50, 100, 200)
TIME_COLORS = {
    50: "#3B82F6",   # blue
    100: "#10B981",  # green
    200: "#EF4444",  # red
}
DNA_ALPHABET = "ACGT"
def safe_name(value: Any) -> str:
    text = str(value).replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def lambda_tag(lam: float) -> str:
    return safe_name(f"{float(lam):.8g}")


def scalar_from_npz(z, key: str, default=None):
    if key not in z.files:
        return default
    a = np.asarray(z[key])
    if a.ndim == 0:
        return a.item()
    if a.size == 1:
        return a.reshape(-1)[0].item()
    return a


def sequence_string(seq: np.ndarray) -> str:
    seq = np.asarray(seq, dtype=np.uint8)
    return "".join(DNA_ALPHABET[int(x)] for x in seq)


def mutations_from_reference(seq: np.ndarray, ref: np.ndarray) -> str:
    seq = np.asarray(seq, dtype=np.uint8)
    ref = np.asarray(ref, dtype=np.uint8)

    idx = np.flatnonzero(seq != ref)
    if len(idx) == 0:
        return ""

    return ";".join(
        f"{DNA_ALPHABET[int(ref[i])]}{int(i)+1}{DNA_ALPHABET[int(seq[i])]}"
        for i in idx
    )


def hex_to_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.asarray(
        [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=float,
    )


def rgb01_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    vals = np.rint(rgb * 255.0).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*vals)


TIME_RGB = {
    t: hex_to_rgb01(c)
    for t, c in TIME_COLORS.items()
}
def temporal_barycentric_color(abundance_by_time: dict[int, int]) -> str:
    """
    Abundance-weighted mixture of blue (50), green (100), red (200).
    """
    weights = np.asarray(
        [
            max(0.0, float(abundance_by_time.get(t, 0)))
            for t in SNAPSHOT_TIMES
        ],
        dtype=float,
    )

    if weights.sum() <= 0:
        return "#6B7280"

    weights /= weights.sum()

    rgb = np.zeros(3, dtype=float)
    for w, t in zip(weights, SNAPSHOT_TIMES):
        rgb += w * TIME_RGB[t]

    return rgb01_to_hex(rgb)


def reconstruct_unique_genotypes(
    indptr: np.ndarray,
    mutation_ids: np.ndarray,
    peak_sequence: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct occupied genotypes from sparse mutation encoding relative to g0.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    mutation_ids = np.asarray(mutation_ids, dtype=np.int64)
    peak = np.asarray(peak_sequence, dtype=np.uint8)

    M = len(indptr) - 1
    unique = np.repeat(peak[None, :], M, axis=0)

    lengths = np.diff(indptr)

    if mutation_ids.size:
        rows = np.repeat(np.arange(M, dtype=np.int64), lengths)

        ids = mutation_ids
        sites = ids // 3
        relative_states = (ids % 3) + 1

        unique[rows, sites] = (
            (peak[sites].astype(np.int64) + relative_states) % 4
        ).astype(np.uint8)

    return unique


def load_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        counts = np.asarray(z["counts"], dtype=np.int64)
        origin = np.asarray(z["peak_sequence"], dtype=np.uint8)

        unique = reconstruct_unique_genotypes(
            z["indptr"],
            z["mutation_ids"],
            origin,
        )

        if len(unique) != len(counts):
            raise RuntimeError(
                f"counts/genotypes mismatch in {path}: "
                f"{len(counts)} != {len(unique)}"
            )

        dom_idx = int(np.argmax(counts))
        root = unique[dom_idx].copy()

        return {
            "path": path,
            "counts": counts,
            "unique": unique,
            "origin": origin,
            "root": root,
            "time": int(scalar_from_npz(z, "t", -1)),
            "mapping": str(scalar_from_npz(z, "fitness_model", "")),
            "lambda": float(scalar_from_npz(z, "av_muts", np.nan)),
            "replicate": int(scalar_from_npz(z, "replicate", -1)),
            "field_seed": int(scalar_from_npz(z, "field_seed", -1)),
            "architecture": str(scalar_from_npz(z, "architecture", "")),
            "L": int(scalar_from_npz(z, "L", unique.shape[1])),
            "N": int(scalar_from_npz(z, "N", counts.sum())),
        }


def build_grand_network(
    snapshots: dict[int, dict[str, Any]],
    min_abundance: int,
) -> nx.Graph:
    """
    Union genotype identities across t=50,100,200, then recompute all exact
    Hamming-distance-1 edges on the union.
    """
    records: dict[bytes, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Union genotype nodes
    # ------------------------------------------------------------------
    for t in SNAPSHOT_TIMES:
        snap = snapshots[t]
        unique = snap["unique"]
        counts = snap["counts"]

        for seq, abundance in zip(unique, counts):
            abundance = int(abundance)

            # Strict condition requested: abundance > 5.
            if abundance <= int(min_abundance):
                continue

            key = seq.tobytes()

            if key not in records:
                records[key] = {
                    "sequence_array": seq.copy(),
                    "abundance_by_time": {
                        tt: 0 for tt in SNAPSHOT_TIMES
                    },
                }

            records[key]["abundance_by_time"][t] = abundance

    G = nx.Graph()

    if not records:
        return G

    # All snapshots in one simulation should share the same original genotype.
    origin = snapshots[SNAPSHOT_TIMES[0]]["origin"]

    ordered = sorted(
        records.items(),
        key=lambda kv: sequence_string(kv[1]["sequence_array"]),
    )

    key_to_node: dict[bytes, str] = {}
    node_to_seq: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Add nodes + all temporal attributes
    # ------------------------------------------------------------------
    for i, (key, rec) in enumerate(ordered):
        seq = rec["sequence_array"]
        abund = rec["abundance_by_time"]

        node_id = f"g{i:07d}"

        key_to_node[key] = node_id
        node_to_seq[node_id] = seq.copy()

        total_abundance = int(sum(abund.values()))

        attrs = {
            "sequence": sequence_string(seq),

            # Aggregate abundance
            "abundance_total": total_abundance,
            "abundance": total_abundance,
            "log_abundance": float(np.log1p(total_abundance)),

            # Origin information
            "distance_to_origin": int(np.count_nonzero(seq != origin)),
            "mutations_from_origin": mutations_from_reference(seq, origin),

            # Temporal color
            "color_hex": temporal_barycentric_color(abund),
        }

        for t in SNAPSHOT_TIMES:
            a = int(abund[t])

            attrs[f"abundance_t{t}"] = a
            attrs[f"present_t{t}"] = bool(
                a > int(min_abundance)
            )

            attrs[f"is_root_t{t}"] = bool(
                np.array_equal(
                    seq,
                    snapshots[t]["root"],
                )
            )

        G.add_node(node_id, **attrs)

    # ------------------------------------------------------------------
    # Recompute exact Hamming-1 edges on the genotype UNION.
    # ------------------------------------------------------------------
    L = next(iter(node_to_seq.values())).shape[0]

    for node_id, seq in node_to_seq.items():
        work = seq.copy()

        for locus in range(L):
            old = int(work[locus])

            for new_state in range(4):
                if new_state == old:
                    continue

                work[locus] = np.uint8(new_state)
                other = key_to_node.get(work.tobytes())

                # node_id < other avoids adding the same undirected edge twice.
                if other is not None and node_id < other:
                    common_times = []

                    for t in SNAPSHOT_TIMES:
                        if (
                            G.nodes[node_id][f"present_t{t}"]
                            and G.nodes[other][f"present_t{t}"]
                        ):
                            common_times.append(t)

                    G.add_edge(
                        node_id,
                        other,
                        hamming_distance=1,
                        common_snapshot_times=",".join(
                            str(t) for t in common_times
                        ),
                        grand_only=bool(len(common_times) == 0),
                    )

            work[locus] = np.uint8(old)

    return G


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()

    nodes = max(
        nx.connected_components(G),
        key=len,
    )

    return G.subgraph(nodes).copy()


def add_final_graphml_attributes(
    G: nx.Graph,
    simulation: dict[str, Any],
    min_abundance: int,
) -> None:
    for n in G.nodes:
        G.nodes[n]["degree"] = int(G.degree[n])

        # Convenient size attribute for later Gephi/Gephi-Lite use.
        G.nodes[n]["plot_size_log_abundance"] = float(
            np.log1p(
                float(G.nodes[n]["abundance_total"])
            )
        )

        G.nodes[n]["architecture"] = simulation["architecture"]
        G.nodes[n]["mapping"] = simulation["mapping"]
        G.nodes[n]["lambda"] = float(simulation["lambda"])
        G.nodes[n]["replicate"] = int(simulation["replicate"])
        G.nodes[n]["field_seed"] = int(simulation["field_seed"])

    G.graph["architecture"] = simulation["architecture"]
    G.graph["mapping"] = simulation["mapping"]
    G.graph["lambda"] = float(simulation["lambda"])
    G.graph["replicate"] = int(simulation["replicate"])
    G.graph["field_seed"] = int(simulation["field_seed"])

    G.graph["grand_network_times"] = "50,100,200"
    G.graph["abundance_threshold_strict_gt"] = int(min_abundance)

    G.graph["color_t50"] = TIME_COLORS[50]
    G.graph["color_t100"] = TIME_COLORS[100]
    G.graph["color_t200"] = TIME_COLORS[200]

    G.graph["color_rule"] = (
        "abundance-weighted barycentric RGB mixture of t50/t100/t200 colors"
    )

    G.graph["edge_rule"] = (
        "exact Hamming-distance-1 edges recomputed on union of retained genotypes"
    )

    G.graph["grand_only_definition"] = (
        "True when endpoint genotypes never co-occur above threshold "
        "in the same one of t=50,100,200"
    )

    # Explicitly no layout information.
    G.graph["layout"] = "none"


def output_path(
    output_root: Path,
    simulation: dict[str, Any],
) -> Path:
    arch = safe_name(simulation["architecture"])
    mapping = safe_name(simulation["mapping"])
    lam = lambda_tag(simulation["lambda"])
    rep = int(simulation["replicate"])

    outdir = (
        output_root
        / arch
        / mapping
        / f"lambda_{lam}"
        / f"landscape_{rep:03d}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    condition = (
        f"{arch}"
        f"__{mapping}"
        f"__lambda_{lam}"
        f"__landscape_{rep:03d}"
    )

    return outdir / f"GN_LCC_{condition}.graphml"




def main():
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from network_analysis import find_snapshot
    ap = argparse.ArgumentParser(description="Grand-network LCC GraphML for the network drawings")
    ap.add_argument("--data", required=True, help="output of generate_rugged.py (or a thesis suite)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--architecture", default="single_nk")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.275)
    ap.add_argument("--min-abundance", type=int, default=5)
    a = ap.parse_args()
    paths = {t: find_snapshot(a.data, a.architecture, a.replicate, t) for t in SNAPSHOT_TIMES}
    snapshots = {t: load_snapshot(p) for t, p in paths.items()}
    if len({s["origin"].tobytes() for s in snapshots.values()}) != 1:
        raise RuntimeError("the initial genotype differs between snapshots")
    G = build_grand_network(snapshots, min_abundance=a.min_abundance)
    lcc = largest_connected_component(G)
    sim = dict(architecture=a.architecture, mapping="model_c_linear", **{"lambda": a.lam}, replicate=a.replicate,
               field_seed=int(scalar_from_npz(np.load(paths[SNAPSHOT_TIMES[0]]), "field_seed", -1)))
    add_final_graphml_attributes(lcc, sim, min_abundance=a.min_abundance)
    out = output_path(Path(a.out), sim)
    nx.write_graphml(lcc, out)
    print(f"{out}: {lcc.number_of_nodes()} nodes, {lcc.number_of_edges()} edges "
          f"(grand network {G.number_of_nodes()} nodes)")


if __name__ == "__main__":
    main()
