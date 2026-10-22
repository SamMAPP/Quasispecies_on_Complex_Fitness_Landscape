#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rugged.py -- quasispecies on House-of-Cards and correlated NK landscapes (Chapter 4, Sec. 4.4)
======================================================================================================

For every architecture (house_of_cards, single_nk) and replicate:

  1. build the quenched field from field_seed (see seeds below);
  2. choose the initial genotype g0 according to --condition
       local    the highest of --n-peak-search distinct local maxima reached by
                steepest-ascent adaptive walks from random sequences;
       evolved  as `local`, then a Wright-Fisher search of --search-generations generations
                started at that maximum (f_l calibrated at the starting maximum) keeps the
                highest-Z genotype ever occupied; --validation-walks further adaptive walks
                may replace it if they find a higher maximum; finally a steepest-ascent
                "polish" makes it a strict local maximum. The search trajectory is saved
                (search_progress.csv, search_records.csv; Fig. NK_evolutionary_search);
       global   the exact global maximum of the NK field (dynamic programming; NK only);
  3. calibrate the threshold f_l at g0: f(g0) / <f>_{3L one-mutants} = exp(c * lambda);
  4. evolve N copies of g0 for t_max generations and save snapshots.

Conventions inherited from the thesis runs (needed to reproduce them, see README.md)

  * --dynamics: compressed or explicit implementation of the same Wright-Fisher process
    (dynamics.py). Default: compressed for local, explicit for evolved/global. The HoC-vs-NK
    comparison was run with --dynamics explicit;
  * condition local: t = 0 is the initial population, the snapshot at t follows t
    generations; conditions evolved/global: t = 1 is the initial population, the snapshot
    at t follows t-1 generations;
  * f_l is recalibrated at each condition's own g0 (it is NOT shared across conditions);
  * seeds depend on the position of the architecture in --architectures (rank r = 0, 1, ...):
        field_seed       = base + (r+1)*4000000037 + (rep+1)*1000000007
        peak_search_seed = base + (r+1)*4000000037 + (rep+1)*2000000033 + 17
        selection/mutation streams of production
                         = base + (r+1)*4000000037 + (rep+1)*10000019 + 1000003 + 100003 + {10007, 20011}
        selection/mutation streams of the evolutionary search
                         = base + (r+1)*4000000037 + (rep+1)*2000000033 + {40009, 50021}
        validation walks = base + (r+1)*4000000037 + (rep+1)*60013

Thesis runs (base seed 12345, L=250, K=4, lambda=0.275, c=2, t_max=200)

  HoC vs NK (Fig. hoc_nk_compact_comparison, replicates 0-2 of 10):
    python generate_rugged.py --condition local --dynamics explicit \
           --architectures house_of_cards single_nk \
           --N 100000 --replicates 10 --snapshots 50 100 150 200 --out Data/hoc_vs_nk
  Local / Evolved / Global on the same NK landscape (Fig. NK_initial_state_summary, replicate 0):
    python generate_rugged.py --condition local   --architectures single_nk --N 1000000 \
           --replicates 5 --snapshots 50 100 150 200   --out Data/nk_local
    python generate_rugged.py --condition evolved --architectures single_nk --N 1000000 \
           --replicates 5 --snapshots 1 50 100 150 200 --search-generations 5000 --out Data/nk_evolved
    python generate_rugged.py --condition global  --architectures single_nk --N 1000000 \
           --replicates 5 --snapshots 1 50 100 150 200 --out Data/nk_global

Output: <out>/<architecture>/replicate_XXX/
    peak.json                 g0, Z(g0), f_l, seeds and parameters
    snapshot_t_{t}.npz        occupied genotypes (sparse, relative to g0), counts, Z, log f
    trajectory.csv            richness, dominance, distances and scores at every t
    search_progress.csv, search_records.csv    (condition evolved only)
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from landscapes import (Landscape, adaptive_walk, best_of, build_field, calibrate_threshold,
                        distinct_local_maxima, exact_global_maximum, hamming, one_mutant_neighbors, to_string)
from dynamics import CompressedWF, ExplicitWF, save_snapshot

ARCH_STRIDE, FIELD_STRIDE, PEAK_STRIDE, REP_STRIDE = 4_000_000_037, 1_000_000_007, 2_000_000_033, 10_000_019
STRENGTH_STRIDE, LAMBDA_STRIDE = 1_000_003, 100_003
SEL_OFFSET, MUT_OFFSET = 10_007, 20_011
SEARCH_SEL_OFFSET, SEARCH_MUT_OFFSET, VALIDATION_STRIDE = 40_009, 50_021, 60_013


def seeds_for(base, rank, rep):
    a = base + (rank + 1) * ARCH_STRIDE
    pop = a + (rep + 1) * REP_STRIDE + STRENGTH_STRIDE + LAMBDA_STRIDE
    search = a + (rep + 1) * PEAK_STRIDE
    return dict(field=a + (rep + 1) * FIELD_STRIDE, peak_search=search + 17,
                selection=pop + SEL_OFFSET, mutation=pop + MUT_OFFSET,
                search_selection=search + SEARCH_SEL_OFFSET, search_mutation=search + SEARCH_MUT_OFFSET,
                validation=a + (rep + 1) * VALIDATION_STRIDE)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- evolutionary search
def evolutionary_search(land, root, prm, sd, out_dir):
    """Long Wright-Fisher search from `root`; returns the highest-Z genotype ever occupied."""
    z_root = float(land.scores(root[None, :])[0])
    f_l = calibrate_threshold(z_root, land.scores(one_mutant_neighbors(root)), prm["lam"], prm["c"])
    wf = CompressedWF(land, f_l, prm["lam"], np.random.default_rng(sd["search_selection"]),
                      np.random.default_rng(sd["search_mutation"]))
    pop = wf.start(root, prm["N"], z_root)
    best, best_z, found_at = root.copy(), z_root, 0
    records = [dict(search_generation=0, record_index=1, best_z=best_z, improvement=0.0,
                    distance_from_search_root=0, peak_sequence=to_string(best))]
    progress = []

    def emit(gen):
        s = pop.summary(root)
        z1 = land.scores(one_mutant_neighbors(best))
        mask = np.all(pop.unique == best[None, :], axis=1)
        progress.append(dict(
            search_generation=gen, search_N=prm["N"], richness=s["richness"], dominance=s["dominance"],
            population_mean_z=s["mean_latent_z"], max_z_occupied=s["max_latent_z"], dominant_z=s["dominant_latent_z"],
            mean_hamming_from_search_root=s["mean_hamming_from_g0"],
            dominant_distance_from_search_root=s["dominant_distance_from_g0"], best_z_ever=best_z,
            best_first_found_generation=found_at, record_count=len(records),
            best_distance_from_search_root=hamming(best, root),
            best_fraction_in_population=float(pop.counts[mask].sum() / prm["N"]),
            best_d1_mean_z=float(z1.mean()), best_d1_max_z=float(z1.max()), search_model_c_f_l=f_l))

    emit(0)
    G = prm["search_generations"]
    for gen in range(1, G + 1):
        pop = wf.step(pop)
        i = int(np.argmax(pop.z))
        improved = pop.z[i] > best_z + 1e-14
        if improved:
            old, best_z, best, found_at = best_z, float(pop.z[i]), pop.unique[i].copy(), gen
            records.append(dict(search_generation=gen, record_index=len(records) + 1, best_z=best_z,
                                improvement=best_z - old, distance_from_search_root=hamming(best, root),
                                peak_sequence=to_string(best)))
        if improved or gen == 1 or gen % prm["diagnostic_every"] == 0 or gen == G:
            emit(gen)
        if gen % 500 == 0:
            print(f"    search g={gen}/{G} best Z={best_z:.6f}", flush=True)
    write_csv(os.path.join(out_dir, "search_progress.csv"), progress)
    write_csv(os.path.join(out_dir, "search_records.csv"), records)
    return best, best_z, f_l


# --------------------------------------------------------------------------- initial genotype
def choose_g0(land, arch, prm, sd, out_dir):
    info = {}
    if prm["condition"] == "global":
        if arch != "single_nk":
            raise ValueError("condition 'global' needs the NK field (exact dynamic programming)")
        g0, z0 = exact_global_maximum(land.field)
        return g0, z0, info

    maxima = distinct_local_maxima(land, prm["n_peak_search"], sd["peak_search"], prm["max_walk_steps"],
                                   prm["attempt_multiplier"])
    pre = best_of(maxima)
    info["local_maxima_z"] = [m["z"] for m in maxima]
    if prm["condition"] == "local":
        return pre["genotype"], pre["z"], info

    g0, z0, f_search = evolutionary_search(land, pre["genotype"], prm, sd, out_dir)
    info.update(preliminary_sequence=to_string(pre["genotype"]), preliminary_z=pre["z"],
                search_best_z=z0, search_f_l=f_search, source="evolutionary search")
    if prm["validation_walks"] > 0:
        val = best_of(distinct_local_maxima(land, prm["validation_walks"], sd["validation"],
                                            prm["max_walk_steps"], prm["attempt_multiplier"]))
        if val["z"] > z0 + 1e-14:
            g0, z0, info["source"] = val["genotype"], val["z"], "validation walk"
    pol = adaptive_walk(land, g0, prm["max_walk_steps"])
    if pol["z"] > z0 + 1e-14:
        g0, z0, info["source"] = pol["genotype"], pol["z"], info["source"] + " + polish"
    return g0, z0, info


# --------------------------------------------------------------------------- one replicate
def run_replicate(prm, arch, rank, rep):
    out_dir = os.path.join(prm["out"], arch, f"replicate_{rep:03d}")
    os.makedirs(out_dir, exist_ok=True)
    sd = seeds_for(prm["base_seed"], rank, rep)
    land = Landscape(build_field(arch, prm["L"], prm["K"], sd["field"]))
    t0 = time.time()

    g0, z_ref, info = choose_g0(land, arch, prm, sd, out_dir)
    f_l = calibrate_threshold(z_ref, land.scores(one_mutant_neighbors(g0)), prm["lam"], prm["c"])
    with open(os.path.join(out_dir, "peak.json"), "w") as f:
        json.dump(dict(architecture=arch, replicate=rep, condition=prm["condition"], peak_sequence=to_string(g0),
                       z_peak=z_ref, f_l=f_l, R_loc=float(np.exp(prm["c"] * prm["lam"])), seeds=sd,
                       parameters={k: v for k, v in prm.items() if k != "out"}, **info), f, indent=1)

    rng_sel, rng_mut = np.random.default_rng(sd["selection"]), np.random.default_rng(sd["mutation"])
    wf = (ExplicitWF if prm["dynamics"] == "explicit" else CompressedWF)(land, f_l, prm["lam"], rng_sel, rng_mut)
    pop = wf.start(g0, prm["N"], float(land.scores(g0[None, :])[0]))
    t_first = 0 if prm["condition"] == "local" else 1    # label of the initial population
    meta = dict(architecture=arch, replicate=rep, condition=prm["condition"], av_muts=prm["lam"],
                model_c_lethality_threshold=f_l, field_seed=sd["field"])
    traj = []
    for t in range(t_first, prm["t_max"] + 1):
        if t > t_first:
            pop = wf.step(pop)
        traj.append(dict(t=t, **pop.summary(g0)))
        if t in prm["snapshots"]:
            save_snapshot(os.path.join(out_dir, f"snapshot_t_{t}.npz"), pop, g0, t, **meta)
    write_csv(os.path.join(out_dir, "trajectory.csv"), traj)
    return f"{arch} replicate {rep}: Z(g0)={z_ref:.6f} f_l={f_l:.6f} ({time.time() - t0:.0f} s)"


# --------------------------------------------------------------------------- driver
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=("local", "evolved", "global"), required=True)
    p.add_argument("--dynamics", choices=("compressed", "explicit"), default=None,
                   help="Wright-Fisher implementation (default: compressed for local, explicit otherwise)")
    p.add_argument("--architectures", nargs="+", default=["house_of_cards", "single_nk"],
                   choices=("house_of_cards", "single_nk"), help="order matters for the seeds")
    p.add_argument("--replicates", type=int, default=10, help="number of replicates (0 ... n-1)")
    p.add_argument("--only-replicates", type=int, nargs="*", default=None, help="run only these replicate ids")
    p.add_argument("--N", type=int, default=100_000)
    p.add_argument("--L", type=int, default=250)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.275, help="mean number of substitutions per offspring")
    p.add_argument("--c", type=float, default=2.0, help="localization factor: R_loc = exp(c lambda)")
    p.add_argument("--t-max", type=int, default=200)
    p.add_argument("--snapshots", type=int, nargs="*", default=[50, 100, 150, 200])
    p.add_argument("--seed", type=int, default=12345, help="base seed")
    p.add_argument("--n-peak-search", type=int, default=16)
    p.add_argument("--attempt-multiplier", type=int, default=25)
    p.add_argument("--max-walk-steps", type=int, default=300)
    p.add_argument("--search-generations", type=int, default=5000)
    p.add_argument("--validation-walks", type=int, default=64)
    p.add_argument("--diagnostic-every", type=int, default=50)
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--out", default="./Data/rugged")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    dynamics = a.dynamics or ("compressed" if a.condition == "local" else "explicit")
    prm = dict(condition=a.condition, dynamics=dynamics, N=a.N, L=a.L, K=a.K, lam=a.lam, c=a.c, t_max=a.t_max,
               snapshots=sorted(set(a.snapshots)), base_seed=a.seed, n_peak_search=a.n_peak_search,
               attempt_multiplier=a.attempt_multiplier, max_walk_steps=a.max_walk_steps,
               search_generations=a.search_generations, validation_walks=a.validation_walks,
               diagnostic_every=a.diagnostic_every, architectures=a.architectures, out=a.out)
    reps = a.only_replicates if a.only_replicates is not None else range(a.replicates)
    jobs = [(arch, rank, rep) for rep in reps for rank, arch in enumerate(a.architectures)]
    if a.n_cores <= 1:
        for job in jobs:
            print(run_replicate(prm, *job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.n_cores) as ex:
            for fut in as_completed([ex.submit(run_replicate, prm, *job) for job in jobs]):
                print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
