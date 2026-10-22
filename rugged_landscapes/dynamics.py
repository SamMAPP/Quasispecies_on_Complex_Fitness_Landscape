# -*- coding: utf-8 -*-
"""
dynamics.py -- haploid Wright-Fisher dynamics on a quenched rugged landscape
============================================================================

One generation (Sec. 4.4.3 of the chapter)

  1. selection: N offspring choose their parents with replacement, with probability
     proportional to n_g f(g)  (one multinomial draw over the occupied genotypes);
  2. mutation: every offspring receives K ~ Poisson(lambda) point substitutions at
     distinct sites; each substitution replaces the base by one of the other three
     with equal probability. Lethal genotypes (f = 0) may be produced but never reproduce.

Two implementations of exactly this process were used for the thesis data; they differ only
in how random numbers are consumed and in how scores are evaluated, and both are kept so
that the published seeds reproduce the published populations:

  CompressedWF  (Local runs and HoC-vs-NK comparison; also the 5000-generation search)
      the population is stored as (unique genotypes, counts, Z). After the multinomial,
      the number of mutated offspring of each genotype is Binomial(n_sel, 1-e^{-lambda}),
      their mutation numbers are Poisson conditioned on > 0, and only mutated offspring
      are materialised. One-mutant NK children are scored by updating the K+1 affected
      terms of the parental Z (exact up to floating-point rounding).
  ExplicitWF    (Evolved and Global production runs)
      the N selected parents are materialised as an N x L array, every row draws a Poisson
      number of mutations, and the resulting unique genotypes are scored from scratch.

Mutation sites: one-hit offspring draw one uniform site; two-hit offspring draw a and b != a;
offspring with >= 3 hits draw their sites with choice(L, k, replace=False). Each hit adds
1, 2 or 3 (mod 4) to the base.
"""
import math

import numpy as np

from landscapes import NKField, log_fitness


# --------------------------------------------------------------------------- shared steps
def wright_fisher_counts(counts, logf, N, rng):
    """Multinomial selection: how many of the N offspring descend from each genotype."""
    logw = np.log(np.asarray(counts, dtype=np.float64)) + logf
    finite = np.isfinite(logw)
    if not finite.any():
        raise RuntimeError("all occupied genotypes are lethal")
    w = np.zeros_like(logw)
    w[finite] = np.exp(np.maximum(logw[finite] - float(np.max(logw[finite])), -745.0))
    return rng.multinomial(int(N), w / float(w.sum())).astype(np.int64, copy=False)


def _substitute(G, rows, sites, rng):
    """One substitution per listed row: base -> base + {1,2,3} (mod 4)."""
    if len(rows):
        G[rows, sites] = (G[rows, sites] + rng.integers(1, 4, size=len(rows), dtype=np.uint8)) & np.uint8(3)


def mutate_rows(G, nmut, L, rng):
    """Apply nmut[r] substitutions at distinct sites to row r of G (in place).

    Returns (rows with one hit, their sites), used by the incremental NK scoring."""
    rows1 = np.flatnonzero(nmut == 1)
    loci1 = np.empty(0, dtype=np.int64)
    if len(rows1):
        loci1 = rng.integers(0, L, size=len(rows1), dtype=np.int64)
        _substitute(G, rows1, loci1, rng)
    rows2 = np.flatnonzero(nmut == 2)
    if len(rows2):
        a = rng.integers(0, L, size=len(rows2), dtype=np.int64)
        b = rng.integers(0, L - 1, size=len(rows2), dtype=np.int64)
        b = b + (b >= a)
        _substitute(G, rows2, a, rng)
        _substitute(G, rows2, b, rng)
    for r in np.flatnonzero(nmut >= 3):
        k = int(nmut[r])
        sites = rng.choice(L, size=k, replace=False)
        G[r, sites] = (G[r, sites] + rng.integers(1, 4, size=k, dtype=np.uint8)) & np.uint8(3)
    return rows1, loci1


def _pack(G):
    """Two bits per base, earlier loci in the high bits: same lexicographic order as G."""
    packed = np.zeros((G.shape[0], (G.shape[1] + 3) // 4), dtype=np.uint8)
    for off, shift in enumerate((6, 4, 2, 0)):
        part = G[:, off::4]
        packed[:, :part.shape[1]] |= part << np.uint8(shift)
    return packed


# --------------------------------------------------------------------------- populations
class Population:
    """Occupied genotypes (lexicographically sorted), their abundances and raw scores."""

    def __init__(self, unique, counts, z, f_l):
        self.unique, self.counts, self.z = unique, counts, z
        self.logf = log_fitness(z, f_l)

    @property
    def N(self):
        return int(self.counts.sum())

    def summary(self, ref):
        """Observables used in the trajectories (ref = initial genotype g0)."""
        p = self.counts / float(self.N)
        h0 = np.count_nonzero(self.unique != ref[None, :], axis=1).astype(np.float64)
        m0 = float(np.dot(p, h0))
        dom = int(np.argmax(self.counts))
        return dict(richness=len(self.counts), dominance=float(p[dom]), mean_hamming_from_g0=m0,
                    variance_hamming_from_g0=float(np.dot(p, (h0 - m0) ** 2)),
                    dominant_distance_from_g0=int(h0[dom]), mean_latent_z=float(np.dot(p, self.z)),
                    max_latent_z=float(np.max(self.z)), dominant_latent_z=float(self.z[dom]))


class CompressedWF:
    """Wright-Fisher on (unique, counts, Z); see module docstring."""

    def __init__(self, land, f_l, lam, rng_sel, rng_mut):
        self.land, self.f_l, self.lam = land, f_l, lam
        self.rng_sel, self.rng_mut = rng_sel, rng_mut

    def start(self, g0, N, z0):
        return Population(g0[None, :].copy(), np.asarray([N], dtype=np.int64), np.asarray([z0]), self.f_l)

    def step(self, pop):
        L, rng = self.land.L, self.rng_mut
        sel = wright_fisher_counts(pop.counts, pop.logf, pop.N, self.rng_sel)

        n_mut = rng.binomial(sel, -math.expm1(-self.lam)).astype(np.int64, copy=False)
        stay = sel - n_mut
        n_m = int(n_mut.sum())
        active = np.flatnonzero(n_mut)
        parent = np.repeat(active, n_mut[active]).astype(np.int64)
        children = pop.unique[parent].copy()
        k = rng.poisson(self.lam, size=n_m).astype(np.int32)       # Poisson conditioned on k > 0
        while (k == 0).any():
            k[k == 0] = rng.poisson(self.lam, size=int(np.count_nonzero(k == 0)))
        np.minimum(k, L, out=k)

        child_z = np.empty(n_m, dtype=np.float64)
        rows1, loci1 = mutate_rows(children, k, L, rng)
        if len(rows1):
            if isinstance(self.land.field, NKField):
                child_z[rows1] = self.land.field.single_mutant_scores(
                    pop.unique, pop.z, parent[rows1], children[rows1], loci1)
            else:
                child_z[rows1] = self.land.scores(children[rows1])
        rest = np.flatnonzero(k != 1)
        if len(rest):
            child_z[rest] = self.land.scores(children[rest])

        keep = stay > 0
        rows = np.concatenate((pop.unique[keep], children))
        weights = np.concatenate((stay[keep], np.ones(n_m, dtype=np.int64)))
        z = np.concatenate((pop.z[keep], child_z))
        _, first, inverse = np.unique(_pack(rows), axis=0, return_index=True, return_inverse=True)
        counts = np.bincount(inverse.reshape(-1), weights=weights.astype(np.float64),
                             minlength=len(first)).astype(np.int64)
        return Population(rows[first].copy(), counts, z[first].astype(np.float64, copy=True), self.f_l)


class ExplicitWF:
    """Wright-Fisher on an explicit N x L array; see module docstring."""

    def __init__(self, land, f_l, lam, rng_sel, rng_mut):
        self.land, self.f_l, self.lam = land, f_l, lam
        self.rng_sel, self.rng_mut = rng_sel, rng_mut

    def from_rows(self, G):
        unique, counts = np.unique(G, axis=0, return_counts=True)
        return Population(unique, counts.astype(np.int64), self.land.scores(unique), self.f_l)

    def start(self, g0, N, z0=None):
        return self.from_rows(np.repeat(g0[None, :], N, axis=0))

    def step(self, pop):
        sel = wright_fisher_counts(pop.counts, pop.logf, pop.N, self.rng_sel)
        keep = sel > 0
        G = np.repeat(pop.unique[keep], sel[keep], axis=0)
        k = np.minimum(self.rng_mut.poisson(self.lam, size=len(G)), self.land.L).astype(np.int32)
        mutate_rows(G, k, self.land.L, self.rng_mut)
        return self.from_rows(G)


# --------------------------------------------------------------------------- output
def save_snapshot(path, pop, ref, t, **meta):
    """Compact snapshot (same keys as the thesis data, read by analysis/common.py).

    Genotype r is stored sparsely relative to the initial genotype `ref`: its mutation ids
    are mutation_ids[indptr[r]:indptr[r+1]], with id = 3*site + (state - ref_state - 1) mod 4.
    """
    rel = ((pop.unique.astype(np.int16) - ref.astype(np.int16)) % 4).astype(np.uint8)
    rows, sites = np.nonzero(rel)
    ids = (3 * sites.astype(np.uint32) + (rel[rows, sites].astype(np.uint32) - 1))
    ids = ids.astype(np.uint16 if 3 * len(ref) <= 65535 else np.uint32)
    indptr = np.zeros(len(pop.unique) + 1, dtype=np.uint64)
    np.cumsum(np.bincount(rows, minlength=len(pop.unique)), out=indptr[1:])
    N = pop.N
    dom = int(np.argmax(pop.counts))
    np.savez_compressed(
        path, indptr=indptr.astype(np.uint32), mutation_ids=ids,
        counts=pop.counts.astype(np.uint16 if N <= 65535 else np.uint32),
        latent_z=pop.z.astype(np.float32), log_fitness=pop.logf,
        hamming_to_initial_peak=np.count_nonzero(rel, axis=1).astype(np.uint16),
        peak_sequence=ref.astype(np.uint8), dominant_sequence=pop.unique[dom],
        t=np.asarray(t, dtype=np.int32), N=np.asarray(N, dtype=np.int32),
        L=np.asarray(len(ref), dtype=np.int32), **{k: np.asarray(v) for k, v in meta.items()})
