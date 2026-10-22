# -*- coding: utf-8 -*-
"""
landscapes.py -- quenched rugged landscapes of Chapter 4, Sec. 4.4
==================================================================

Genotypes are quaternary sequences g in {0,1,2,3}^L (A,C,G,T), stored as uint8 rows.

Raw landscape scores (both have the one-point law of a mean of L Uniform(0,1) variables)

  NK (adjacent, circular):  Z_NK(g) = (1/L) sum_i u_i(g_i, g_{i+1}, ..., g_{i+K}),
      u_i(.) ~ U(0,1) quenched lookup tables (L x 4^{K+1}, stored as float32);
  House of Cards:           Z_HoC(g) = (1/L) sum_i U_i(g),  U_i(g) ~ U(0,1) iid for every g,
      generated lazily from a 64-bit hash of (seed, g) (the K = L-1 limit of NK).

Reproductive fitness (linear-threshold, "Model C" in the old scripts)

  f(g) = max(0, Z(g) - f_l),   f_l chosen so that
  f(Z_ref) / mean_{j in 3L one-mutants} f(Z_1j) = R_loc = exp(c * lambda)      (bisection)

Peaks

  * local maxima from steepest-ascent adaptive walks started at random sequences;
  * the exact global maximum of an adjacent NK field by dynamic programming over the
    4^K boundary states (cyclic closure + backtracking).

Every numerical step follows the scripts that produced the thesis data
(hoc_nk_final_10rep.py, hoc_nk_global_peak.py, hoc_nk_global_peak_optimized.py), so that
the same seeds give the same landscapes, peaks and thresholds.
"""
import math

import numpy as np

A_SIZE = 4
ALPHABET = "ACGT"

# 64-bit hash constants of the House-of-Cards field
ROLL_BASE = 0x9E3779B185EBCA87
SPLITMIX_A = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX_B = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_C = np.uint64(0x94D049BB133111EB)
TWO53 = float(1 << 53)


# --------------------------------------------------------------------------- sequences
def to_string(g):
    return "".join(ALPHABET[int(x)] for x in np.asarray(g, dtype=np.uint8))


def from_string(s):
    lookup = {c: i for i, c in enumerate(ALPHABET)}
    return np.asarray([lookup[c] for c in s.strip()], dtype=np.uint8)


def one_mutant_neighbors(g):
    """All 3L one-mutant neighbours, ordered by site and then by +1, +2, +3 (mod 4)."""
    g = np.asarray(g, dtype=np.uint8)
    L = g.size
    nb = np.repeat(g[None, :], 3 * L, axis=0)
    rows, sites = np.arange(3 * L), np.repeat(np.arange(L), 3)
    nb[rows, sites] = (nb[rows, sites] + np.tile(np.asarray([1, 2, 3], dtype=np.uint8), L)) & np.uint8(3)
    return nb


def hamming(a, b):
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


# --------------------------------------------------------------------------- fields
class NKField:
    """Adjacent circular NK field (Aguirre & Manrubia 2015, Eq. 1)."""

    kind = "single_nk"

    def __init__(self, L, K, seed):
        if not 0 <= K < L:
            raise ValueError("K must satisfy 0 <= K < L")
        self.L, self.K, self.seed = int(L), int(K), int(seed)
        self.window = self.K + 1
        rng = np.random.default_rng(self.seed)
        self.tables = rng.random(size=(self.L, A_SIZE ** self.window)).astype(np.float32)
        self.neighborhoods = (np.arange(self.L)[:, None] + np.arange(self.window)[None, :]) % self.L
        self.powers = A_SIZE ** np.arange(self.window - 1, -1, -1, dtype=np.int64)

    @classmethod
    def from_tables(cls, tables, K):
        """NK field with given lookup tables, shape (L, 4^{K+1}); used for the L=3 appendix example."""
        f = cls.__new__(cls)
        f.L, f.K, f.seed, f.window = tables.shape[0], int(K), None, int(K) + 1
        f.tables = np.asarray(tables)
        f.neighborhoods = (np.arange(f.L)[:, None] + np.arange(f.window)[None, :]) % f.L
        f.powers = A_SIZE ** np.arange(f.window - 1, -1, -1, dtype=np.int64)
        return f

    def scores(self, G):
        """Z of every row of G (shape M x L)."""
        G = np.asarray(G, dtype=np.uint8)
        z = np.zeros(G.shape[0], dtype=np.float64)
        for i in range(self.L):
            idx = G[:, self.neighborhoods[i]].astype(np.int64, copy=False) @ self.powers
            z += self.tables[i, idx]
        return z / float(self.L)

    def single_mutant_scores(self, parents, parent_z, parent_idx, children, loci, batch=20_000):
        """Z of one-mutant children from the parental Z: only the K+1 terms that contain the
        mutated locus are updated (the thesis runs used this update; see dynamics.py)."""
        out = np.empty(len(loci), dtype=np.float64)
        offsets = np.arange(self.window, dtype=np.int64)
        for s in range(0, len(loci), batch):
            e = min(len(loci), s + batch)
            pidx, loc = parent_idx[s:e], loci[s:e]
            terms = (loc[:, None] - offsets[None, :]) % self.L
            nbh = self.neighborhoods[terms]                                 # (B, K+1, K+1)
            old = np.sum(parents[pidx[:, None, None], nbh].astype(np.int64) * self.powers, axis=2, dtype=np.int64)
            new = np.sum(children[s:e][np.arange(e - s)[:, None, None], nbh].astype(np.int64) * self.powers,
                         axis=2, dtype=np.int64)
            delta = np.sum(self.tables[terms, new].astype(np.float64) - self.tables[terms, old].astype(np.float64),
                           axis=1, dtype=np.float64) / float(self.L)
            out[s:e] = parent_z[pidx] + delta
        return out


def _splitmix64(x):
    z = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + SPLITMIX_A
        z = (z ^ (z >> np.uint64(30))) * SPLITMIX_B
        z = (z ^ (z >> np.uint64(27))) * SPLITMIX_C
        z = z ^ (z >> np.uint64(31))
    return z


def _open_unit(x):
    return ((np.asarray(x, dtype=np.uint64) >> np.uint64(11)).astype(np.float64) + 0.5) / TWO53


class HoCField:
    """House of Cards: independent mean-of-L-uniforms score for every genotype."""

    kind = "house_of_cards"

    def __init__(self, L, seed):
        self.L, self.seed = int(L), int(seed)
        rng = np.random.default_rng(self.seed)
        self.locus_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=self.L, dtype=np.uint64)

    def scores(self, G):
        G = np.asarray(G, dtype=np.uint8)
        key = np.full(G.shape[0], np.uint64(self.seed & ((1 << 64) - 1)), dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(G.shape[1]):
                key = key * np.uint64(ROLL_BASE) + (G[:, j].astype(np.uint64) + np.uint64(1))
        z = np.zeros(G.shape[0], dtype=np.float64)
        for s in self.locus_seeds:
            z += _open_unit(_splitmix64(key ^ np.uint64(s)))
        return z / float(self.L)


def build_field(architecture, L, K, seed):
    if architecture == "single_nk":
        return NKField(L, K, seed)
    if architecture == "house_of_cards":
        return HoCField(L, seed)
    raise ValueError(f"unknown architecture {architecture!r}")


class Landscape:
    """A field plus a bounded cache of visited scores (the cache never changes a value)."""

    def __init__(self, field, cache_max=150_000):
        self.field, self.L = field, field.L
        self.cache, self.cache_max = {}, cache_max

    def scores(self, G):
        G = np.asarray(G, dtype=np.uint8)
        out = np.empty(len(G), dtype=np.float64)
        miss = []
        for j, row in enumerate(G):
            v = self.cache.get(row.tobytes())
            if v is None:
                miss.append(j)
            else:
                out[j] = v
        if miss:
            vals = self.field.scores(G[miss])
            if len(self.cache) + len(miss) > self.cache_max:
                self.cache.clear()
            for j, v in zip(miss, vals):
                out[j] = v
                self.cache[G[j].tobytes()] = float(v)
        return out


# --------------------------------------------------------------------------- fitness
def log_fitness(z, f_l):
    """log f with f = max(0, Z - f_l); lethal genotypes get -inf (exactly zero weight)."""
    f = np.maximum(0.0, np.asarray(z, dtype=np.float64) - f_l)
    out = np.full_like(f, -np.inf)
    viable = f > 0.0
    out[viable] = np.log(f[viable])
    return out


def calibrate_threshold(z_ref, shell_z, lam, c=2.0, n_bisect=100):
    """f_l such that f(Z_ref) / mean_j f(Z_1j) = exp(c*lam) (Eq. nk_localization_condition).

    The ratio is non-decreasing in f_l for a strict local maximum; the root is bracketed in
    [0, max(Z_1)) and found by bisection. If the raw scores already exceed the target ratio
    at f_l = 0, f_l = 0 is returned.
    """
    shell = np.asarray(shell_z, dtype=np.float64).ravel()
    R = math.exp(c * lam)
    if not z_ref > shell.max():
        raise ValueError("the reference genotype must be a strict local maximum")

    def ratio(f_l):
        m = float(np.mean(np.maximum(0.0, shell - f_l)))
        return max(0.0, z_ref - f_l) / m if m > 0.0 else float("inf")

    if ratio(0.0) >= R:
        return 0.0
    lo, hi = 0.0, float(np.nextafter(shell.max(), -np.inf))
    if ratio(hi) < R:
        raise RuntimeError("could not bracket f_l")
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < R:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- local maxima
def adaptive_walk(land, start, max_steps=300):
    """Steepest ascent: move to the best one-mutant neighbour while it improves Z."""
    g = np.asarray(start, dtype=np.uint8).copy()
    for step in range(max_steps + 1):
        z = float(land.scores(g[None, :])[0])
        nb = one_mutant_neighbors(g)
        z_nb = land.scores(nb)
        best = int(np.argmax(z_nb))
        if z_nb[best] <= z + 1e-14:
            return dict(genotype=g, z=z, shell_mean=float(z_nb.mean()), n_steps=step)
        g = nb[best].copy()
    raise RuntimeError(f"adaptive walk did not converge in {max_steps} steps")


def distinct_local_maxima(land, n, seed, max_steps=300, attempt_multiplier=25):
    """The first n distinct local maxima reached from uniformly random starting sequences."""
    rng = np.random.default_rng(seed)
    found, seen, attempts = [], set(), 0
    while len(found) < n and attempts < max(attempt_multiplier * n, n):
        attempts += 1
        res = adaptive_walk(land, rng.integers(0, 4, size=land.L, dtype=np.uint8), max_steps)
        key = res["genotype"].tobytes()
        if key not in seen:
            seen.add(key)
            found.append(res)
    if len(found) < n:
        raise RuntimeError(f"only {len(found)} distinct maxima found in {attempts} walks")
    return found


def best_of(maxima):
    """Highest Z; ties broken by the mean score of the one-mutant shell."""
    return max(maxima, key=lambda r: (r["z"], r["shell_mean"]))


# --------------------------------------------------------------------------- exact NK optimum
def _base4_digits(n_states, width):
    out = np.empty((n_states, width), dtype=np.uint8)
    tmp = np.arange(n_states, dtype=np.int64)
    for j in range(width - 1, -1, -1):
        out[:, j] = tmp % A_SIZE
        tmp //= A_SIZE
    return out


def exact_global_maximum(field):
    """Exact argmax of an adjacent circular NK field by dynamic programming.

    State = the last K bases. All 4^K initial states are propagated simultaneously
    (dp[start, current]); the K wrap-around terms are added at closure and the optimal
    sequence is recovered by backtracking. Cost O(L 4^{2K+1}), memory O(L 4^{2K}).
    """
    L, K = field.L, field.K
    if K == 0:
        seq = np.argmax(field.tables[:, :A_SIZE].astype(np.float64), axis=1).astype(np.uint8)
        return seq, float(field.scores(seq[None, :])[0])
    n_states, drop_place = A_SIZE ** K, A_SIZE ** (K - 1)
    states = np.arange(n_states, dtype=np.int64)
    digits = _base4_digits(n_states, K)
    pred = (states // A_SIZE)[:, None] + np.arange(A_SIZE)[None, :] * drop_place   # predecessors of each state
    config = pred * A_SIZE + (states % A_SIZE)[:, None]                             # (K+1)-mer table index
    n_steps = L - K
    dp = np.full((n_states, n_states), -np.inf)
    np.fill_diagonal(dp, 0.0)
    back = np.empty((n_steps, n_states, n_states), dtype=np.uint8)
    for step in range(n_steps):
        cand = dp[:, pred] + field.tables[step, config].astype(np.float64)[None, :, :]
        arg = np.argmax(cand, axis=2).astype(np.uint8)
        dp = np.take_along_axis(cand, arg[:, :, None], axis=2)[:, :, 0]
        back[step] = arg
    closure = np.zeros((n_states, n_states))
    for j in range(K):                       # term L-K+j sees suffix[j:K] + prefix[0:j+1]
        suffix = np.zeros(n_states, dtype=np.int64)
        for col in range(j, K):
            suffix = suffix * A_SIZE + digits[:, col]
        prefix = np.zeros(n_states, dtype=np.int64)
        for col in range(j + 1):
            prefix = prefix * A_SIZE + digits[:, col]
        closure += field.tables[L - K + j, suffix[None, :] * A_SIZE ** (j + 1) + prefix[:, None]].astype(np.float64)
    total = dp + closure
    start, end = np.unravel_index(int(np.argmax(total)), total.shape)
    seq = np.empty(L, dtype=np.uint8)
    seq[:K] = digits[start]
    cur = int(end)
    for step in range(n_steps - 1, -1, -1):
        seq[K + step] = cur % A_SIZE
        cur = cur // A_SIZE + int(back[step, start, cur]) * drop_place
    if cur != int(start):
        raise RuntimeError("backtracking did not close on the initial state")
    z = float(field.scores(seq[None, :])[0])
    if abs(float(total[start, end]) / L - z) > 5e-12:
        raise RuntimeError("DP optimum and direct score disagree")
    return seq, z
