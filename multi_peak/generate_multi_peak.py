#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_multi_peak.py -- traveling localized quasispecies (multiple peaks, Chapter 4, Sec. 4.3)
================================================================================================

Landscape

  * explicit peaks: the ancestor g0 = 0...0 has fitness F_0; at t = tau, 2 tau, ... a new
    peak g_{j+1} is created exactly delta mutations away from the PREVIOUS PEAK g_j
    (delta distinct loci drawn uniformly and flipped; redrawn if it coincides with an
    older peak), with fitness F_{j+1} = F_j + Delta_f. Older peaks keep their fitness;
  * background: every other genotype has fitness 0 with probability q and f_bg otherwise,
    quenched through a hash of (landscape_seed, genotype)
    (viable iff blake2b(landscape_seed, sorted loci as <u4) / 2^64 >= q).

Dynamics (one generation t -> t+1)

  * parents are processed in population order; a parent of fitness f > 0 draws ONE
    mutation set, K ~ Poisson(lambda) distinct loci, shared by its f offspring;
  * at an introduction time (t+1 multiple of tau) the first offspring produced in that
    generation is replaced by one copy of the new peak;
  * viable offspring form the pool. If the pool has exactly N members it becomes the new
    population; otherwise (always, in practice) positions 1, ..., N-1 are resampled with
    replacement from the pool while POSITION 0 IS KEPT. Position 0 holds g0 until the first
    introduction and afterwards the most recently introduced peak (it is overwritten at
    every introduction), so the newest peak always keeps one protected copy.

  These rules reproduce 16-Simulations/old/multi_peak_quasispecies.py, which generated the
  thesis data (Data/model4a30). The chapter text describes the peak as introduced at
  distance delta from the instantaneous root and does not mention the protected copy; see
  CHAPTER_NOTES.md.

Reproducibility

  Iteration i uses iteration_seed = base_seed + i for: numpy default_rng (Poisson numbers),
  random.Random (mutated loci, new peaks, resampling) and the landscape hash. The calls are
  made in the same order as in the original script, so the same seed and parameters give
  identical output. The base seeds of the thesis runs were not recorded (the original
  script used time_ns() without saving it), so those exact realizations cannot be
  regenerated; new runs are statistically equivalent.

Output

    <out>/Abundances/dataOut_<tag>_t_{t}.txt   one individual per line (line 1 = position 0):
                                               comma-separated mutated loci, '--' for g0
    <out>/run_metadata_<tag without iteration>.json
    tag = mu_{lam}_nrw_{N}_L_{L}_q_{q}_iteration_{i}_fpeak_{F0}_fbg_{f_bg}_fstep_{Delta_f}_tau_{tau}_delta_{delta}

Chapter runs

  dynamics (Figs. mean_hamming, variance_dispersion, phi):
    python generate_multi_peak.py --q 0 --delta 15 --N 1000 --L 10000 --t-max 200 --n-iters 100
  topology (Figs. pre_peak_LCC_degree, pre_peak_LCC_knn):
    python generate_multi_peak.py --q 0 --delta 5 --N 100000 --L 250 --t-max 200 --n-iters 10
"""
import argparse
import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

MAX_SEED = 2 ** 128 - 1


# --------------------------------------------------------------------------- genotypes
# A genotype is the sorted tuple of its mutated loci (relative to g0 = ()).
def apply_muts(genotype, loci):
    return tuple(sorted(set(genotype).symmetric_difference(loci))) if len(loci) else genotype


def new_peak(source, delta, L, rng, existing, max_attempts=10000):
    """A genotype exactly delta mutations from `source`, distinct from all older peaks."""
    for _ in range(max_attempts):
        candidate = apply_muts(source, rng.sample(range(L), k=delta))
        if candidate not in existing:
            return candidate
    raise RuntimeError(f"no new peak found after {max_attempts} attempts; try another delta or larger L")


# --------------------------------------------------------------------------- landscape
class MultiPeakLandscape:
    def __init__(self, seed, q, F0, f_bg):
        self.q, self.f_bg = float(q), int(f_bg)
        self.peaks = {(): int(F0)}                  # explicit peaks override the background
        self.current = ()
        self.prefix = hashlib.blake2b(digest_size=8, person=b"fitness")
        self.prefix.update(int(seed).to_bytes(16, "little", signed=False))

    def fitness(self, g):
        f = self.peaks.get(g)
        if f is not None:
            return f
        if self.f_bg == 0 or self.q >= 1.0:
            return 0
        if self.q <= 0.0:
            return self.f_bg
        h = self.prefix.copy()
        h.update(np.asarray(g, dtype="<u4").tobytes() if g else b"")
        u = int.from_bytes(h.digest(), "little") / 2.0 ** 64
        return 0 if u < self.q else self.f_bg

    def introduce_peak(self, delta, f_step, L, rng):
        g = new_peak(self.current, delta, L, rng, self.peaks)
        self.peaks[g] = self.peaks[self.current] + f_step
        self.current = g
        return g


# --------------------------------------------------------------------------- one generation
def one_generation(pop, land, t, prm, np_rng, py_rng):
    """Advance the population from t-1 to t (pop is a list of genotypes, modified in place)."""
    L, lam, N = prm["L"], prm["lam"], prm["N"]
    introduce = t % prm["tau"] == 0
    pool = []
    for parent in pop[:N]:
        n_desc = land.fitness(parent)
        if n_desc <= 0:
            continue
        k = min(int(np_rng.poisson(lam)), L)
        loci = py_rng.sample(range(L), k=k) if k > 0 else []
        child = apply_muts(parent, loci)
        for _ in range(n_desc):
            off = child
            if introduce:                            # first offspring of the generation
                off = land.introduce_peak(prm["delta"], prm["f_step"], L, py_rng)
                pop[0] = off                         # the protected position 0
                introduce = False
            if land.fitness(off) > 0:
                pool.append(off)
    if not pool:
        raise RuntimeError(f"extinction at t={t}")
    if len(pool) == N:
        pop[:] = pool
    else:
        for i in range(1, N):
            pop[i] = py_rng.choice(pool)


# --------------------------------------------------------------------------- I/O
def tag(prm, iteration):
    return (f"mu_{prm['lam']}_nrw_{prm['N']}_L_{prm['L']}_q_{prm['q']}_iteration_{iteration}"
            f"_fpeak_{prm['F0']}_fbg_{prm['f_bg']}_fstep_{prm['f_step']}_tau_{prm['tau']}_delta_{prm['delta']}")


def write_abundances(path, pop):
    with open(path, "w") as f:
        for g in pop:
            f.write(",".join(map(str, g)) + "\n" if g else "--\n")


# --------------------------------------------------------------------------- one realization
def run_iteration(prm, iteration):
    seed = (prm["base_seed"] + iteration) % (MAX_SEED + 1)
    py_rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    land = MultiPeakLandscape(seed, prm["q"], prm["F0"], prm["f_bg"])
    pop = [() for _ in range(prm["N"])]
    stem = os.path.join(prm["out"], "Abundances", f"dataOut_{tag(prm, iteration)}_t_")
    times = prm["abund_times"]
    if times is None or 0 in times:
        write_abundances(f"{stem}0.txt", pop)
    for t in range(1, prm["t_max"] + 1):
        one_generation(pop, land, t, prm, np_rng, py_rng)
        if times is None or t in times:
            write_abundances(f"{stem}{t}.txt", pop)
    return iteration, {",".join(map(str, g)) or "--": f for g, f in land.peaks.items()}


# --------------------------------------------------------------------------- driver
def generate(q=0.0, delta=15, N=1000, L=10000, lam=1.0, F0=3, f_bg=1, f_step=1, tau=40, t_max=200,
             n_iters=100, iter_start=0, base_seed=None, out="./Data/multi_peak", abund_times=None,
             n_cores=1, verbose=True):
    if not 0.0 <= q <= 1.0 or not 1 <= delta <= L or tau < 1 or N < 1:
        raise ValueError("invalid parameters")
    base_seed = time.time_ns() if base_seed is None else int(base_seed)
    prm = dict(q=float(q), delta=int(delta), N=int(N), L=int(L), lam=float(lam), F0=int(F0), f_bg=int(f_bg),
               f_step=int(f_step), tau=int(tau), t_max=int(t_max), base_seed=base_seed, out=out,
               abund_times=None if abund_times is None else set(abund_times))
    os.makedirs(os.path.join(out, "Abundances"), exist_ok=True)
    peaks = {}
    t0 = time.time()
    its = range(iter_start, iter_start + n_iters)
    if n_cores <= 1:
        results = (run_iteration(prm, i) for i in its)
    else:
        ex = ProcessPoolExecutor(max_workers=n_cores)
        results = (fut.result() for fut in as_completed([ex.submit(run_iteration, prm, i) for i in its]))
    for done, (it, pk) in enumerate(results, 1):
        peaks[it] = pk
        if verbose:
            print(f"  [delta={delta} N={N} L={L}] {done}/{n_iters} iteration {it}  ({time.time() - t0:.0f} s)", flush=True)
    meta = {k: (sorted(v) if isinstance(v, set) else v) for k, v in prm.items()}
    meta.update(model="multi_peak (model4a30)", iterations=[iter_start, iter_start + n_iters - 1],
                seed_rule="iteration_seed = base_seed + iteration", peaks_by_iteration=peaks)
    name = tag(prm, "X").replace("_iteration_X", "")
    with open(os.path.join(out, f"run_metadata_{name}_seed_{base_seed}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return base_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=float, default=0.0, help="lethal fraction of the background")
    p.add_argument("--delta", type=int, required=True, help="Hamming distance between consecutive peaks")
    p.add_argument("--tau", type=int, default=40, help="generations between peak introductions")
    p.add_argument("--F0", type=int, default=3, help="fitness of the ancestral peak")
    p.add_argument("--f-step", type=int, default=1, help="fitness increment Delta_f of each new peak")
    p.add_argument("--f-bg", type=int, default=1, help="fitness of viable background genotypes")
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--L", type=int, default=10000)
    p.add_argument("--lam", type=float, default=1.0, help="mean number of mutations per replication")
    p.add_argument("--t-max", type=int, default=200)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--iter-start", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="base seed (default: time_ns, saved in the metadata)")
    p.add_argument("--out", default="./Data/multi_peak")
    p.add_argument("--abund-times", type=int, nargs="*", default=None,
                   help="generations with an abundance snapshot (default: every generation)")
    p.add_argument("--n-cores", type=int, default=1)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    seed = generate(q=a.q, delta=a.delta, N=a.N, L=a.L, lam=a.lam, F0=a.F0, f_bg=a.f_bg, f_step=a.f_step,
                    tau=a.tau, t_max=a.t_max, n_iters=a.n_iters, iter_start=a.iter_start, base_seed=a.seed,
                    out=a.out, abund_times=a.abund_times, n_cores=a.n_cores)
    print(f"base seed: {seed}")


if __name__ == "__main__":
    main()
