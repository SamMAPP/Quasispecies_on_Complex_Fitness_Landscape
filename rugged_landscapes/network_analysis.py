"""
network_analysis.py -- genotype networks of quaternary populations (rugged landscapes)
=====================================================================================

analyze_snapshot(path) reads one compact snapshot (snapshot_t_<t>.npz written by generate_rugged.py,
or by the thesis scripts), builds the exact network of occupied genotypes (edges between genotypes at
Hamming distance one), extracts its largest connected component (LCC) and returns, among others,
the LCC node degrees, the node-level k_nn, abundances (rank-abundance curve), radial shells and
assortativity. Used by fig_hoc_vs_nk.py and fig_initial_states.py.

Extracted verbatim from old_codes/analyze_hoc_nk_final_qbeta.py (the analyzer used for the thesis
figures), keeping only the functions analyze_snapshot needs; find_snapshot() is new.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
def grouped_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    ox, oy = [], []
    for xx in np.unique(x):
        z = y[(x == xx) & np.isfinite(y)]
        if len(z):
            ox.append(float(xx))
            oy.append(float(np.mean(z)))
    return np.asarray(ox, float), np.asarray(oy, float)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def radial_moments_from_dataframe(df: pd.DataFrame) -> tuple[float, float, float]:
    """Return abundance-weighted mean, variance, and dispersion of a radial curve."""
    if df.empty or "m" not in df or "P_read" not in df:
        return np.nan, np.nan, np.nan
    m = df["m"].to_numpy(float)
    p = df["P_read"].to_numpy(float)
    z = np.isfinite(m) & np.isfinite(p) & (p >= 0)
    if not np.any(z):
        return np.nan, np.nan, np.nan
    m, p = m[z], p[z]
    norm = float(np.sum(p))
    if norm <= 0:
        return np.nan, np.nan, np.nan
    p = p / norm
    mean = float(np.sum(p * m))
    var = float(np.sum(p * (m - mean) ** 2))
    disp = float(var / mean) if mean > 0 else np.nan
    return mean, var, disp


def log_shell_size(L: int, m: int) -> float:
    if m < 0 or m > L:
        return -np.inf
    return (
        m * math.log(3.0)
        + math.lgamma(L + 1.0)
        - math.lgamma(m + 1.0)
        - math.lgamma(L - m + 1.0)
    )


def shell_fraction_from_log_numerator(value: float, log_omega: float) -> float:
    if value <= 0 or not np.isfinite(log_omega):
        return 0.0
    logv = math.log(float(value)) - log_omega
    # exp() underflow to zero is harmless and scientifically appropriate here.
    return math.exp(logv) if logv > -745.0 else 0.0


def load_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        indptr = np.asarray(z["indptr"], dtype=np.int64)
        mids = np.asarray(z["mutation_ids"], dtype=np.int64)
        counts = np.asarray(z["counts"], dtype=np.int64)
        L = int(z["L"])
        N = int(z["N"])
        if "hamming_to_initial_peak" in z.files:
            h0 = np.asarray(z["hamming_to_initial_peak"], dtype=np.int32)
        else:
            h0 = None

    keys = [
        tuple(int(x) for x in mids[indptr[i] : indptr[i + 1]])
        for i in range(len(counts))
    ]
    if h0 is None:
        h0 = np.asarray([len(k) for k in keys], dtype=np.int32)
    return {"keys": keys, "counts": counts, "h0": h0, "L": L, "N": N}


def hamming_keys(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    ia = ib = d = 0
    while ia < len(a) and ib < len(b):
        sa, sb = a[ia] // 3, b[ib] // 3
        if sa < sb:
            d += 1
            ia += 1
        elif sb < sa:
            d += 1
            ib += 1
        else:
            if a[ia] != b[ib]:
                d += 1
            ia += 1
            ib += 1
    return d + (len(a) - ia) + (len(b) - ib)


def build_graph(
    keys: list[tuple[int, ...]], L: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {k: i for i, k in enumerate(keys)}
    eu, ev, es = [], [], []

    for i, key in enumerate(keys):
        for pos, mid in enumerate(key):
            site = mid // 3

            parent = key[:pos] + key[pos + 1 :]
            j = index.get(parent)
            if j is not None:
                eu.append(i)
                ev.append(j)
                es.append(site)

            base = 3 * site
            for alt in range(base, base + 3):
                if alt <= mid:
                    continue
                kk = key[:pos] + (alt,) + key[pos + 1 :]
                j = index.get(kk)
                if j is not None:
                    eu.append(i)
                    ev.append(j)
                    es.append(site)

    return (
        np.asarray(eu, dtype=np.int32),
        np.asarray(ev, dtype=np.int32),
        np.asarray(es, dtype=np.int32),
    )


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.sz = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = int(self.p[x])
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def largest_component_mask(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=bool)
    uf = UnionFind(n)
    for a, b in zip(u.tolist(), v.tolist()):
        uf.union(int(a), int(b))
    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    sizes = np.bincount(labels)
    return labels == int(np.argmax(sizes))


def node_degrees(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.zeros(n, dtype=np.int32)
    if len(u):
        d += np.bincount(u, minlength=n).astype(np.int32)
        d += np.bincount(v, minlength=n).astype(np.int32)
    return d


def eigenvector_centrality_lcc(
    n: int, u: np.ndarray, v: np.ndarray, max_iter: int = 1000, tol: float = 1e-11
) -> np.ndarray:
    """Max-normalized eigenvector centrality for one already-isolated LCC."""
    if n == 0:
        return np.zeros(0, float)
    if n == 1:
        return np.ones(1, float)
    data = np.ones(2 * len(u), float)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    x = np.full(n, 1.0 / math.sqrt(n), float)
    # +I is a harmless shift that improves power-iteration behavior for bipartite-ish graphs.
    for _ in range(max_iter):
        y = A @ x + x
        norm = float(np.linalg.norm(y))
        if norm <= 0:
            return np.ones(n, float)
        y /= norm
        if np.linalg.norm(y - x) < tol:
            x = y
            break
        x = y
    mx = float(np.max(x))
    return x / mx if mx > 0 else np.ones(n, float)


def exact_triangles_from_edge_sites(
    n: int, u: np.ndarray, v: np.ndarray, sites: np.ndarray, L: int
) -> np.ndarray:
    tri = np.zeros(n, dtype=np.int64)
    if not len(u):
        return tri
    codes = np.concatenate(
        [
            u.astype(np.int64) * int(L) + sites.astype(np.int64),
            v.astype(np.int64) * int(L) + sites.astype(np.int64),
        ]
    )
    uq, c = np.unique(codes, return_counts=True)
    nodes = (uq // int(L)).astype(np.int32)
    contrib = (c * (c - 1) // 2).astype(np.int64)
    np.add.at(tri, nodes, contrib)
    return tri


def radial_shells(
    distances: np.ndarray, counts: np.ndarray, L: int, N: int
) -> pd.DataFrame:
    distances = np.asarray(distances, dtype=int)
    counts = np.asarray(counts, dtype=float)
    maxm = int(np.max(distances)) if len(distances) else 0
    H = np.bincount(distances, minlength=maxm + 1).astype(float)
    A = np.bincount(distances, weights=counts, minlength=maxm + 1).astype(float)

    rows = []
    for m in range(maxm + 1):
        logom = log_shell_size(L, m)
        p_read = A[m] / float(N) if N > 0 else 0.0
        q_shell = shell_fraction_from_log_numerator(H[m], logom)
        c_per_read = shell_fraction_from_log_numerator(p_read, logom)
        rows.append(
            {
                "m": float(m),
                "H": H[m],
                "A": A[m],
                "P_read": p_read,
                "q_shell": q_shell,
                "C_shell_per_read": c_per_read,
                "log_Omega": logom,
            }
        )
    return pd.DataFrame(rows)


def analyze_snapshot(path: Path, replicate: int = -1) -> dict[str, Any]:
    s = load_snapshot(path)
    keys = s["keys"]
    counts = np.asarray(s["counts"], dtype=np.int64)
    h0 = np.asarray(s["h0"], dtype=np.int32)
    L, N, M = int(s["L"]), int(s["N"]), len(keys)

    uall, vall, sall = build_graph(keys, L)

    # -------------------------------------------------------------------------
    # FULL OCCUPIED NETWORK: every occupied genotype, including all disconnected
    # components and isolated nodes.  This is the new topology scope requested.
    # -------------------------------------------------------------------------
    degree_all = node_degrees(M, uall, vall)

    neighbour_degree_sum_all = np.zeros(M, dtype=float)
    if len(uall):
        np.add.at(neighbour_degree_sum_all, uall, degree_all[vall])
        np.add.at(neighbour_degree_sum_all, vall, degree_all[uall])

    knn_node_all = np.full(M, np.nan, dtype=float)
    good_all = degree_all > 0
    knn_node_all[good_all] = (
        neighbour_degree_sum_all[good_all] / degree_all[good_all]
    )

    triangles_all = exact_triangles_from_edge_sites(
        M, uall, vall, sall, L
    ).astype(float)

    # The same sparse power iteration works on the disconnected full graph.
    # Components with smaller spectral radius naturally receive very small
    # eigenvector weight; isolated nodes can remain finite because of the +I shift.
    eig_all = eigenvector_centrality_lcc(M, uall, vall)

    k_knn_all, mean_knn_all = grouped_mean(
        degree_all, knn_node_all
    )
    maxk_all = int(np.max(degree_all)) if M else 0
    k_ccdf_all = np.arange(0, maxk_all + 1, dtype=float)
    ccdf_all = (
        np.asarray(
            [np.mean(degree_all >= k) for k in k_ccdf_all],
            dtype=float,
        )
        if M
        else np.zeros(0)
    )
    k_tri_all, mean_tri_all = grouped_mean(
        degree_all, triangles_all
    )

    # Connected-component labels/sizes for inspection of the full graph.
    if M:
        uf_all = UnionFind(M)
        for a, b in zip(uall.tolist(), vall.tolist()):
            uf_all.union(int(a), int(b))
        roots_all = np.asarray(
            [uf_all.find(i) for i in range(M)],
            dtype=np.int32,
        )
        _, component_label_all = np.unique(
            roots_all,
            return_inverse=True,
        )
        component_sizes_all = np.bincount(component_label_all)
        n_components_all = int(len(component_sizes_all))
    else:
        component_label_all = np.zeros(0, dtype=np.int32)
        component_sizes_all = np.zeros(0, dtype=np.int32)
        n_components_all = 0

    # -------------------------------------------------------------------------
    # LCC TOPOLOGY: preserve the original analysis unchanged.
    # -------------------------------------------------------------------------
    lcc = largest_component_mask(M, uall, vall)
    em = lcc[uall] & lcc[vall] if len(uall) else np.zeros(0, dtype=bool)
    u_global, v_global, sites = uall[em], vall[em], sall[em]

    degree = node_degrees(M, u_global, v_global)
    neighbour_degree_sum = np.zeros(M, dtype=float)
    if len(u_global):
        np.add.at(neighbour_degree_sum, u_global, degree[v_global])
        np.add.at(neighbour_degree_sum, v_global, degree[u_global])
    knn_node = np.full(M, np.nan, dtype=float)
    good = degree > 0
    knn_node[good] = neighbour_degree_sum[good] / degree[good]

    triangles = exact_triangles_from_edge_sites(M, u_global, v_global, sites, L)
    lcc_nodes = np.flatnonzero(lcc)
    lcc_deg = degree[lcc]
    lcc_knn = knn_node[lcc]
    lcc_tri = triangles[lcc].astype(float)

    # Remap LCC edges to contiguous local indices for eigenvector centrality.
    remap = np.full(M, -1, dtype=np.int32)
    remap[lcc_nodes] = np.arange(len(lcc_nodes), dtype=np.int32)
    lu = remap[u_global] if len(u_global) else np.zeros(0, np.int32)
    lv = remap[v_global] if len(v_global) else np.zeros(0, np.int32)
    eig_lcc = eigenvector_centrality_lcc(len(lcc_nodes), lu, lv)

    k_knn, mean_knn = grouped_mean(lcc_deg, lcc_knn)
    maxk = int(np.max(lcc_deg)) if len(lcc_deg) else 0
    k_ccdf = np.arange(0, maxk + 1, dtype=float)
    ccdf = (
        np.asarray([np.mean(lcc_deg >= k) for k in k_ccdf], dtype=float)
        if len(lcc_deg)
        else np.zeros(0)
    )
    k_tri, mean_tri = grouped_mean(lcc_deg, lcc_tri)

    # Instantaneous root = most abundant genotype at this snapshot.
    dom_idx = int(np.argmax(counts)) if M else 0
    dom_key = keys[dom_idx] if M else ()
    hdom = np.asarray([hamming_keys(k, dom_key) for k in keys], dtype=np.int32)

    radial_dom = radial_shells(hdom, counts, L, N)
    radial_origin = radial_shells(h0, counts, L, N)

    abundance = np.sort(counts.astype(float))[::-1]
    rank = np.arange(1, len(abundance) + 1, dtype=float)
    rank_p = abundance / float(N) if N else np.zeros_like(abundance)

    # Rank of each genotype by abundance, retained at node level for the
    # alpha-gamma-eta consistency test. Ties are assigned a deterministic order.
    order = np.argsort(-counts, kind="stable") if M else np.zeros(0, dtype=int)
    rank_by_index = np.empty(M, dtype=float)
    if M:
        rank_by_index[order] = np.arange(1, M + 1, dtype=float)

    pdom = radial_dom["P_read"].to_numpy(float)
    mdom = radial_dom["m"].to_numpy(float)
    mean_dom, var_dom, disp_dom = radial_moments_from_dataframe(radial_dom)
    mean_origin, var_origin, disp_origin = radial_moments_from_dataframe(radial_origin)
    dominant_key = "WT" if len(dom_key) == 0 else ",".join(str(int(x)) for x in dom_key)

    node_table_all = pd.DataFrame(
        {
            "degree": degree_all.astype(float),
            "knn": knn_node_all.astype(float),
            "triangles": triangles_all.astype(float),
            "relative_abundance": (
                counts.astype(float) / float(N) if N else np.zeros(M)
            ),
            "abundance_rank": (
                rank_by_index.astype(float) if M else np.zeros(0, dtype=float)
            ),
            "eigenvector": eig_all.astype(float),
            "distance_dominant": hdom.astype(float),
            "distance_origin": h0.astype(float),
            "component": component_label_all.astype(float),
            "component_size": (
                component_sizes_all[component_label_all].astype(float)
                if M else np.zeros(0, dtype=float)
            ),
            "in_lcc": lcc.astype(bool),
            "replicate": int(replicate),
        }
    )

    node_table = pd.DataFrame(
        {
            "degree": lcc_deg.astype(float),
            "knn": lcc_knn.astype(float),
            "triangles": lcc_tri.astype(float),
            "relative_abundance": (
                counts[lcc].astype(float) / float(N) if N else np.zeros(len(lcc_deg))
            ),
            "abundance_rank": (
                rank_by_index[lcc].astype(float) if M else np.zeros(len(lcc_deg))
            ),
            "eigenvector": eig_lcc.astype(float),
            "distance_dominant": hdom[lcc].astype(float),
            "distance_origin": h0[lcc].astype(float),
            "replicate": int(replicate),
        }
    )

    return {
        "L": L,
        "N": N,
        "scalar": {
            "M": M,
            "N": N,
            "L": L,
            "lcc_nodes": int(np.sum(lcc)),
            "lcc_node_fraction": float(np.mean(lcc)) if M else np.nan,
            "lcc_read_fraction": float(np.sum(counts[lcc]) / N) if N else np.nan,
            "dominance": float(counts[dom_idx] / N) if N and M else np.nan,
            "dominant_key": dominant_key,
            "mean_m_dominant": mean_dom,
            "variance_m_dominant": var_dom,
            "dispersion_m_dominant": disp_dom,
            "mean_m_origin": mean_origin,
            "variance_m_origin": var_origin,
            "dispersion_m_origin": disp_origin,
            "P_m_le3": (
                float(np.sum(pdom[: min(4, len(pdom))])) if len(pdom) else np.nan
            ),
            "dominant_distance_from_origin": int(h0[dom_idx]) if M else 0,
            "mean_degree_lcc": float(np.mean(lcc_deg)) if len(lcc_deg) else np.nan,
            "max_degree_lcc": int(np.max(lcc_deg)) if len(lcc_deg) else 0,

            # Full occupied-network topology.
            "all_nodes": int(M),
            "all_edges": int(len(uall)),
            "all_components": int(n_components_all),
            "all_isolated_nodes": int(np.sum(degree_all == 0)) if M else 0,
            "all_isolated_fraction": (
                float(np.mean(degree_all == 0)) if M else np.nan
            ),
            "mean_degree_all": (
                float(np.mean(degree_all)) if M else np.nan
            ),
            "max_degree_all": (
                int(np.max(degree_all)) if M else 0
            ),
            "abundance_degree_spearman_all": (
                spearman_corr(
                    node_table_all["relative_abundance"],
                    node_table_all["degree"],
                )
                if len(node_table_all)
                else np.nan
            ),
            "degree_eigenvector_spearman_all": (
                spearman_corr(
                    node_table_all["degree"],
                    node_table_all["eigenvector"],
                )
                if len(node_table_all)
                else np.nan
            ),
            "abundance_eigenvector_spearman_all": (
                spearman_corr(
                    node_table_all["relative_abundance"],
                    node_table_all["eigenvector"],
                )
                if len(node_table_all)
                else np.nan
            ),

            "abundance_degree_spearman": (
                spearman_corr(node_table["relative_abundance"], node_table["degree"])
                if len(node_table)
                else np.nan
            ),
            "degree_eigenvector_spearman": (
                spearman_corr(node_table["degree"], node_table["eigenvector"])
                if len(node_table)
                else np.nan
            ),
            "abundance_eigenvector_spearman": (
                spearman_corr(
                    node_table["relative_abundance"], node_table["eigenvector"]
                )
                if len(node_table)
                else np.nan
            ),
        },
        # Original LCC topology (backwards compatible names).
        "knn": (k_knn, mean_knn),
        "ccdf": (k_ccdf, ccdf),
        "triangles": (k_tri, mean_tri),
        "nodes": node_table,

        # New full occupied-network topology.
        "knn_all": (k_knn_all, mean_knn_all),
        "ccdf_all": (k_ccdf_all, ccdf_all),
        "triangles_all": (k_tri_all, mean_tri_all),
        "nodes_all": node_table_all,

        # Population-wide quantities (independent of graph scope).
        "rank": (rank, rank_p),
        "radial_dom": radial_dom,
        "radial_origin": radial_origin,
    }




def find_snapshot(root, architecture, replicate, t):
    """Path of snapshot_t_<t>.npz for one architecture/replicate.

    Accepts the layout of generate_rugged.py (<root>/<architecture>/replicate_NNN/) and that of the
    thesis suites (<root>/architecture_<architecture>/replicate_NNN/treatment_ordinary/
    fitness_model_model_c_linear/.../run_000/).
    """
    root = Path(root)
    for base in (root / architecture, root / f"architecture_{architecture}"):
        rep = base / f"replicate_{int(replicate):03d}"
        hits = sorted(p for p in rep.rglob(f"snapshot_t_{int(t)}.npz")
                      if "fitness_model_" not in str(p) or "model_c_linear" in str(p))
        if hits:
            if len(hits) > 1:
                raise RuntimeError(f"several snapshots match {rep}: {hits}")
            return hits[0]
    raise FileNotFoundError(f"no snapshot_t_{t}.npz for {architecture} replicate {replicate} under {root}")
