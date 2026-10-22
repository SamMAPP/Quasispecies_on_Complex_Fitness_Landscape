#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_single_peak.py -- single-peak holey landscape (Model fq, Chapter 4, Sec. 4.2)
=====================================================================================

Landscape (Eq. fq_landscape of the chapter)

    f(g) = F            if g = g0 = 0...0  (the selected genotype),
         = 0            with probability q   (lethal),     g != g0,
         = f_bg = 1     with probability 1-q (viable),     g != g0.

  The viable/lethal state of a genotype is quenched: it is a deterministic hash of
  (landscape_seed, genotype), so no table of visited genotypes is stored.
  A genotype g != g0 is viable iff  blake2b(landscape_seed, g) / 2^64 <= 1-q.

Dynamics (one generation)

  * every individual with fitness f > 0 produces f offspring (an integer);
  * mutations: K ~ Poisson(lambda) distinct loci drawn uniformly (binary genotypes,
    a mutation flips the bit). Two schemes were used for the thesis data:
        --mutations per-parent     one draw per parent, shared by its f offspring
                                   (the convention of Chapter 3 / generate_data.py);
                                   used for the transition runs with q = 0, 0.1, ..., 1.0;
        --mutations per-offspring  one independent draw per offspring;
                                   used for the transition runs with q = 0.05, 0.15, ..., 0.95
                                   and for the topology runs (N=1e5, L=250, F=3).
    Both give the same mean-field recursion x_{t+1} = r_q x_t / [1+(F-1) x_t];
  * offspring with fitness 0 are discarded; the viable offspring form the pool;
  * the next generation is N draws with replacement from the pool (Wright-Fisher).

  The population starts monomorphic at g0. Iteration i uses
  iteration_seed = base_seed + i both for the random numbers and as landscape seed.

Reproducibility

  The random numbers are consumed in exactly the order of the scripts that produced the
  thesis data: for every parent in population order, one rng.poisson(lambda) and, if
  K > 0, one rng.choice(L, K, replace=False) -- once per parent (per-parent) or once per
  offspring (per-offspring) -- and finally one rng.integers(0, pool_size, N).
  Hence the same base_seed and parameters give the same trajectories, bit by bit.
  The remaining work (bit flips, hashing, statistics) is vectorised.

Output (<out>/ is created if needed)

    <out>/Means/<stem>.csv            t, mean_hamming, variance_hamming   (every t)
    <out>/Abundances/<stem>_t_{t}.txt one genotype per line: comma-separated mutated loci, '--' = g0
    <out>/Distribs/<stem>_t_{t}.txt   one Hamming distance (to g0) per line
    <out>/run_metadata_<stem without iteration>.json   all parameters and the base seed

    stem = dataOut_mu_{lam}_nrw_{N}_L_{L}_q_{q}_iteration_{i}_fpeak_{F}_fbg_{f_bg}

Chapter runs (base seeds of the thesis data are listed in README.md)

  localization transition, Figs. "mean_hamming_and_beta" and "Qk" (N=1e3, L=1e4, F=2), two batches:
    python generate_single_peak.py --q 0.0 0.1 ... 1.0 --mutations per-parent --F 2 --N 1000 \
           --L 10000 --t-max 100 --n-iters 100 --abund-times 99 100 --distrib-times 100 \
           --seed <one seed per q>
    python generate_single_peak.py --q 0.05 0.15 ... 0.95 --mutations per-offspring  (same options)
  topology, Figs. "LCC_degree" and "LCC_knn" (N=1e5, L=250, F=3, q=0):
    python generate_single_peak.py --q 0 --F 3 --N 100000 --L 250 --mutations per-offspring --t-max 100 \
           --n-iters 10 --abund-times 99 100 --distrib-times 100 --seed 1784889873066522343
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

assert sys.byteorder == "little", "packed genotypes assume a little-endian machine"

BYTE_BITCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------- genotypes
def n_words(L):
    return (L + 63) // 64


def popcount_rows(G):
    """Hamming distance of every packed genotype (row) to g0."""
    return BYTE_BITCOUNT[np.ascontiguousarray(G).view(np.uint8)].sum(axis=1).astype(np.int64)


def row_to_loci(row, L):
    bits = np.unpackbits(row.view(np.uint8), bitorder="little")[:L]
    return np.flatnonzero(bits)


def fmt_num(x):
    """Number label used in file names: 1.0 -> '1.0', 0.5 -> '0.5', 3 -> '3'."""
    return str(x)


def file_stem(lam, N, L, q, iteration, F, f_bg):
    return (f"dataOut_mu_{float(lam)}_nrw_{N}_L_{L}_q_{float(q)}_iteration_{iteration}"
            f"_fpeak_{int(F)}_fbg_{int(f_bg)}")


# --------------------------------------------------------------------------- landscape
class HashLandscape:
    """Quenched single-peak holey landscape, Eq. (fq_landscape)."""

    def __init__(self, seed, q, F, f_bg):
        self.q, self.F, self.f_bg = float(q), int(F), int(f_bg)
        self.prefix = hashlib.blake2b(digest_size=8, person=b"fitness")
        self.prefix.update(int(seed).to_bytes(16, "little", signed=False))

    def fitness_rows(self, G):
        """Fitness of every packed genotype (row of G)."""
        f = np.empty(G.shape[0], dtype=np.int64)
        for j, row in enumerate(G):
            if not row.any():
                f[j] = self.F
                continue
            h = self.prefix.copy()
            h.update(row.tobytes())
            u = int.from_bytes(h.digest(), "little") / 2.0 ** 64
            f[j] = self.f_bg if u <= 1.0 - self.q else 0
        return f


# --------------------------------------------------------------------------- one generation
def mutate(G, fit, rng, lam, L, land):
    """One mutation draw per row of G: returns the mutated copies and their fitness.

    The random numbers are drawn row by row (poisson, then choice if K > 0), as in the
    original scripts; rows that receive no mutation keep the fitness in `fit`.
    """
    n = G.shape[0]
    k = np.empty(n, dtype=np.int64)
    loci = []
    for j in range(n):
        kj = min(int(rng.poisson(lam)), L)
        k[j] = kj
        if kj:
            loci.append(rng.choice(L, size=kj, replace=False))

    OG = G.copy()
    off_fit = fit.copy()
    mutated = np.flatnonzero(k)
    if mutated.size:
        ind = np.repeat(mutated, k[mutated])
        loc = np.concatenate(loci).astype(np.int64)
        np.bitwise_xor.at(OG, (ind, loc >> 6), np.uint64(1) << (loc & 63).astype(np.uint64))
        off_fit[mutated] = land.fitness_rows(OG[mutated])
    return OG, off_fit


def one_generation(G, fit, rng, lam, L, land, mutations):
    """Advance the population (G: packed genotypes, fit: their fitness) by one generation.

    Returns the new (G, fit), or None if the pool of viable offspring is empty.
    """
    n_desc = np.maximum(fit, 0)
    if mutations == "per-offspring":
        parent = np.repeat(np.arange(G.shape[0]), n_desc)      # one row per offspring
        OG, off_fit = mutate(G[parent], fit[parent], rng, lam, L, land)
    elif mutations == "per-parent":
        rep = np.flatnonzero(n_desc)                            # one row per reproducing parent
        MG, m_fit = mutate(G[rep], fit[rep], rng, lam, L, land)
        OG, off_fit = np.repeat(MG, n_desc[rep], axis=0), np.repeat(m_fit, n_desc[rep])
    else:
        raise ValueError(f"unknown mutation scheme {mutations!r}")
    if off_fit.size == 0:
        return None

    pool = np.flatnonzero(off_fit > 0)
    if pool.size == 0:
        return None
    chosen = pool[rng.integers(0, pool.size, size=G.shape[0])]
    return OG[chosen], off_fit[chosen]


# --------------------------------------------------------------------------- I/O
def write_abundances(path, G, L):
    cache = {}
    with open(path, "w") as f:
        for row in G:
            key = row.tobytes()
            line = cache.get(key)
            if line is None:
                loci = row_to_loci(row, L)
                line = "--\n" if loci.size == 0 else ",".join(map(str, loci.tolist())) + "\n"
                cache[key] = line
            f.write(line)


def write_distribs(path, m):
    with open(path, "w") as f:
        f.write("\n".join(map(str, m.tolist())) + "\n")


# --------------------------------------------------------------------------- one realization
def run_iteration(job):
    lam, N, L, q, F, f_bg = (job[k] for k in ("lam", "N", "L", "q", "F", "f_bg"))
    it, t_max, out = job["iteration"], job["t_max"], job["out"]
    stem = file_stem(lam, N, L, q, it, F, f_bg)
    seed = job["base_seed"] + it
    rng = np.random.default_rng(seed)
    land = HashLandscape(seed, q, F, f_bg)

    G = np.zeros((N, n_words(L)), dtype=np.uint64)
    fit = np.full(N, F, dtype=np.int64)
    status = "completed"
    tmp = os.path.join(out, "Means", stem + ".csv.part")
    with open(tmp, "w") as fm:
        fm.write("t,mean_hamming,variance_hamming\n")
        fm.write("0,0.0,0.0\n")
        for t in range(1, t_max + 1):
            new = one_generation(G, fit, rng, lam, L, land, job["mutations"])
            if new is None:
                fm.write(f"{t},nan,nan\n")
                status = f"extinct at t={t}"
                break
            G, fit = new
            m = popcount_rows(G)
            mf = m.astype(np.float64)                # float64 first: same rounding as the original scripts
            fm.write(f"{t},{float(mf.mean())!r},{float(mf.var())!r}\n")
            if t in job["abund_times"]:
                write_abundances(os.path.join(out, "Abundances", f"{stem}_t_{t}.txt"), G, L)
            if t in job["distrib_times"]:
                write_distribs(os.path.join(out, "Distribs", f"{stem}_t_{t}.txt"), m)
    os.replace(tmp, tmp[:-len(".part")])
    return it, status


# --------------------------------------------------------------------------- driver
def generate(q, F=2, f_bg=1, N=1000, L=10000, lam=1.0, mutations="per-parent", t_max=100, n_iters=100, iter_start=0,
             base_seed=None, out="./Data/single_peak", abund_times=(99, 100), distrib_times=(100,),
             n_cores=1, verbose=True):
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    base_seed = time.time_ns() if base_seed is None else int(base_seed)
    for sub in ("Means", "Abundances", "Distribs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    meta = dict(model="single_peak (model fq)", q=float(q), F=int(F), f_bg=int(f_bg), N=int(N),
                L=int(L), lam=float(lam), mutations=mutations, t_max=int(t_max), iterations=[iter_start, iter_start + n_iters - 1],
                base_seed=base_seed, seed_rule="iteration_seed = base_seed + iteration",
                abund_times=sorted(abund_times), distrib_times=sorted(distrib_times))
    tag = f"q={q} F={F} N={N} L={L}"
    with open(os.path.join(out, f"run_metadata_{file_stem(lam, N, L, q, 'X', F, f_bg)}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    jobs = [dict(lam=float(lam), N=N, L=L, q=float(q), F=F, f_bg=f_bg, mutations=mutations, iteration=i, t_max=t_max,
                 out=out, base_seed=base_seed, abund_times=set(abund_times),
                 distrib_times=set(distrib_times))
            for i in range(iter_start, iter_start + n_iters)]
    t0 = time.time()
    if n_cores <= 1:
        results = map(run_iteration, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=n_cores)
        results = (fut.result() for fut in as_completed([ex.submit(run_iteration, j) for j in jobs]))
    for done, (it, status) in enumerate(results, 1):
        if verbose:
            print(f"  [{tag}] {done}/{n_iters} iteration {it}: {status}  ({time.time() - t0:.0f} s)", flush=True)
    return base_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=float, nargs="+", required=True, help="lethal fraction (one or several values)")
    p.add_argument("--F", type=int, default=2, help="fitness of the selected genotype g0")
    p.add_argument("--f-bg", type=int, default=1, help="fitness of viable background genotypes")
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--L", type=int, default=10000)
    p.add_argument("--lam", type=float, default=1.0, help="mean number of mutations per replication")
    p.add_argument("--mutations", choices=("per-parent", "per-offspring"), default="per-parent",
                   help="one mutation draw per parent (shared by its F offspring) or per offspring")
    p.add_argument("--t-max", type=int, default=100)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--iter-start", type=int, default=0)
    p.add_argument("--seed", type=int, nargs="+", default=None,
                   help="base seed (one, or one per --q value; default: time_ns)")
    p.add_argument("--out", default="./Data/single_peak")
    p.add_argument("--abund-times", type=int, nargs="*", default=[99, 100])
    p.add_argument("--distrib-times", type=int, nargs="*", default=[100])
    p.add_argument("--n-cores", type=int, default=1)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    seeds = a.seed or [None]
    if len(seeds) not in (1, len(a.q)):
        raise SystemExit("--seed takes one value or one value per --q")
    for j, q in enumerate(a.q):
        generate(q, F=a.F, f_bg=a.f_bg, N=a.N, L=a.L, lam=a.lam, mutations=a.mutations, t_max=a.t_max, n_iters=a.n_iters,
                 iter_start=a.iter_start, base_seed=seeds[j if len(seeds) > 1 else 0], out=a.out,
                 abund_times=a.abund_times, distrib_times=a.distrib_times, n_cores=a.n_cores)


if __name__ == "__main__":
    main()
