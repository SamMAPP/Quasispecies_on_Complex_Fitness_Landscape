# Quasispecies on complex fitness landscapes

Code for Chapter 4 of the thesis, *From selective peaks to rugged correlated fitness landscapes*. It
contains the simulation generators for the three model families of the chapter, and the analysis
scripts that make every figure of the chapter from the simulation output:

1. a single selected genotype on a random holey landscape (Sec. 4.2);
2. a sequence of successively introduced fitter peaks, the "traveling localized quasispecies"
   (Sec. 4.3);
3. quenched rugged landscapes: House of Cards and correlated NK (Sec. 4.4, plus the appendix
   example).

Every generator consumes random numbers in exactly the same order as the scripts that produced the
thesis data. **The same base seed and parameters give the same populations.** This was checked
against the thesis data (see [Verification](#verification)). The analysis scripts were checked by
regenerating the published figures from the thesis data.

`CHAPTER_NOTES.md` lists the places where the chapter text and the simulations disagree, and details
the text leaves out.

---

## Layout

```
single_peak/
    generate_single_peak.py      single-peak holey landscape (Model fq)
    fig_localization_region.py   Fig. localization_parameter_region       (analytical)
    fig_radial_spreading.py      Fig. model-fq_mean_hamming_and_beta_panel_last_15_points
    fig_stationary_Qk.py         Fig. model-fq_Qk_analytical_exponential_clean_t_100
    fig_topology_degree.py       Fig. model-fq_LCC_degree_panel_shell_theory_joint_...
    fig_topology_knn.py          Fig. model-fq_LCC_knn_panel_conditioned_shell_theory_...
multi_peak/
    generate_multi_peak.py       successively introduced peaks (model4a30)
    compute_distances.py         distances to origin / instantaneous root  (needed by the next three)
    compute_phi.py               overlap angle between consecutive populations
    fig_mean_hamming.py          Fig. model4a30_mean_hamming_origin_root_panel
    fig_variance_dispersion.py   Fig. model4a30_variance_dispersion_origin_root_panel
    fig_phi.py                   Fig. model4a30_phi_vs_time_with_theory
    fig_topology_degree.py       Fig. model4a30_pre_peak_LCC_degree_panel_multicentre_...
    fig_topology_knn.py          Fig. model4a30_pre_peak_LCC_knn_multicentre_panel_...
rugged_landscapes/
    landscapes.py                NK and HoC fields, linear-threshold fitness, f_l calibration,
                                 adaptive walks, exact NK optimum (dynamic programming)
    dynamics.py                  Wright-Fisher selection + mutation; snapshot format
    generate_rugged.py           simulations: --condition local | evolved | global
    network_analysis.py          LCC degree, k_nn, rank abundance of a snapshot
    export_grand_network.py      filtered union network used for the drawings
    fig_hoc_vs_nk.py             Fig. hoc_nk_compact_comparison_t200
    fig_initial_states.py        Fig. NK_initial_state_summary_unfiltered_analysis
    fig_evolutionary_search.py   Fig. NK_evolutionary_search_examples
    toy_nk_appendix.py           appendix (L=3, K=1) and Fig. toy_NK_evolution_landscape_and_network
common/
    build_networks.py            GraphML genotype networks of the binary models (single and multi peak)
```

The generators were written for this repository. They are compact, vectorized and commented, with
the model equations in each module docstring. The `fig_*`/`compute_*` scripts of `single_peak/` and
`multi_peak/`, and `fig_hoc_vs_nk.py`, `fig_initial_states.py`, `fig_evolutionary_search.py`
and `toy_nk_appendix.py`, are close ports of the notebooks that made the thesis figures. Each was
changed only in its input/output configuration (now command-line options) and by removing unused
code, so that it reproduces the published figures. Each says in its header which notebook cell it
comes from.

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`, `networkx` (≥ 3.4, for
`forceatlas2_layout`) and `matplotlib-venn`; see `requirements.txt`.

**numpy version and bit-for-bit reproduction.** The random streams (numpy `PCG64`, Python `random`)
are the same in numpy 1.26 and 2.x, but `exp`, `log` and some reductions round differently in the
last bit. In the rugged-landscape runs a one-ulp change in a selection probability eventually flips a
multinomial draw, so the trajectories diverge after some generations. **Use numpy 1.26.4** to
reproduce all thesis data exactly (`pip install numpy==1.26.4`). Every verification below passes with
it. The HoC-vs-NK and single/multi-peak simulations are also identical under numpy 2.5; the
Local/Evolved/Global runs at `N = 10⁶` are not. With any other version the results are statistically
equivalent.

---

## 1. Single peak (Sec. 4.2)

`f(g0) = F`, `f(g) = 0` with probability `q` and `1` otherwise (quenched by hashing the genotype),
`f` offspring per individual, Poisson(`λ`) mutations, `N` draws with replacement from the viable
offspring. Binary genotypes, ancestor `g0 = 0…0`.

```bash
cd single_peak
# localization transition (N=1e3, L=1e4, F=2, lambda=1, 100 iterations, t=0..100).
# The thesis sweep used two generator versions; see CHAPTER_NOTES.md (A3):
#   q = 0, 0.1, ..., 1.0       --mutations per-parent
#   q = 0.05, 0.15, ..., 0.95  --mutations per-offspring
python generate_single_peak.py --q 0.3 --seed 1783604226783417763 --mutations per-parent \
       --F 2 --N 1000 --L 10000 --t-max 100 --n-iters 100 --abund-times 99 100 --distrib-times 100 \
       --out ../Data/single_peak --n-cores 8          # repeat for every q with its seed (table below)
# topology (N=1e5, L=250, F=3, q=0, 10 iterations)
python generate_single_peak.py --q 0 --seed 1784889873066522343 --mutations per-offspring \
       --F 3 --N 100000 --L 250 --t-max 100 --n-iters 10 --abund-times 99 100 --distrib-times 100 \
       --out ../Data/single_peak_topology --n-cores 10
python ../common/build_networks.py ../Data/single_peak_topology --times 100 --n-cores 10

python fig_localization_region.py --lam 0.2 --out ../Figures      # the thesis figure uses lambda=0.2
python fig_radial_spreading.py --data ../Data/single_peak --out ../Figures
python fig_stationary_Qk.py    --data ../Data/single_peak --out ../Figures
python fig_topology_degree.py  --data ../Data/single_peak_topology --out ../Figures
python fig_topology_knn.py     --data ../Data/single_peak_topology --out ../Figures
```

Base seeds of the thesis localization-transition runs (`iteration_seed = base_seed + iteration`):

| q | base seed | | q | base seed |
|---|---|---|---|---|
| 0.0 | 1783604226853408180 | | 0.05 | 1784623634615959534 |
| 0.1 | 1783604226892754179 | | 0.15 | 1784623634456135775 |
| 0.2 | 1783604226760336437 | | 0.25 | 1784623634433148753 |
| 0.3 | 1783604226783417763 | | 0.35 | 1784623635391361735 |
| 0.4 | 1783604226658051499 | | 0.45 | 1784623634378339825 |
| 0.5 | 1783604226735018922 | | 0.55 | 1784623634972665300 |
| 0.6 | 1783604226652850852 | | 0.65 | 1784623634536101880 |
| 0.7 | 1783604227126006364 | | 0.75 | 1784623634520506102 |
| 0.8 | 1783604226615068273 | | 0.85 | 1784623634442033720 |
| 0.9 | 1783604227428101672 | | 0.95 | 1784623634690848740 |
| 1.0 | 1783605555135981061 | | | |

Left column: `--mutations per-parent`; right column: `--mutations per-offspring`.

## 2. Multiple peaks (Sec. 4.3)

Ancestral peak `F0 = 3`, background fitness 1 (`q = 0`). Every `τ = 40` generations a new peak with
fitness `F_j + 1` is created `δ` mutations from the previous peak and inserted as one individual;
position 0 of the population (the newest peak) is never resampled. See the module docstring and
`CHAPTER_NOTES.md` (A2).

```bash
cd multi_peak
# dynamics (N=1e3, L=1e4, delta=15, 100 iterations, t=0..200)
python generate_multi_peak.py --delta 15 --N 1000 --L 10000 --t-max 200 --n-iters 100 \
       --out ../Data/multi_peak_delta15 --n-cores 8
python compute_distances.py       --data ../Data/multi_peak_delta15
python compute_phi.py             --data ../Data/multi_peak_delta15
python fig_mean_hamming.py        --data ../Data/multi_peak_delta15 --out ../Figures
python fig_variance_dispersion.py --data ../Data/multi_peak_delta15 --out ../Figures
python fig_phi.py                 --data ../Data/multi_peak_delta15 --out ../Figures
# topology (N=1e5, L=250, delta=5, 10 iterations; snapshots just before each introduction)
python generate_multi_peak.py --delta 5 --N 100000 --L 250 --t-max 200 --n-iters 10 \
       --abund-times 39 79 119 159 199 --out ../Data/multi_peak_delta5 --n-cores 10
python ../common/build_networks.py ../Data/multi_peak_delta5 --times 39 79 119 159 199 --n-cores 10
python fig_topology_degree.py --data ../Data/multi_peak_delta5 --delta 5 --out ../Figures
python fig_topology_knn.py    --data ../Data/multi_peak_delta5 --delta 5 --out ../Figures
```

The base seeds of the thesis multi-peak runs were not recorded: the original script used `time_ns()`
without saving it. Those exact realizations therefore cannot be regenerated. New runs are statistically
equivalent, and `generate_multi_peak.py` now saves its seed (and the true peak sequences) in
`run_metadata_*.json`. If a data folder holds several parameter sets, select one with
`--filter <substring of the file names>`.

## 3. House of Cards and correlated NK (Sec. 4.4)

Quaternary genotypes, `L = 250`. Raw score `Z` (NK with adjacent circular neighborhoods, `K = 4`, or
House of Cards), fitness `f = max(0, Z − f_l)` with `f_l` calibrated so that
`f(g0) / ⟨f⟩_{3L one-mutants} = e^{2λ}`, `λ = 0.275`. Haploid Wright–Fisher dynamics with Poisson
point substitutions. All thesis runs used `--seed 12345`.

```bash
cd rugged_landscapes
# HoC vs NK (Fig. hoc_nk_compact_comparison_t200): 10 replicates, N=1e5 (displayed: 0, 1, 2)
python generate_rugged.py --condition local --dynamics explicit \
       --architectures house_of_cards single_nk --N 100000 --replicates 10 \
       --snapshots 50 100 150 200 --out ../Data/hoc_vs_nk --n-cores 6
python fig_hoc_vs_nk.py --data ../Data/hoc_vs_nk --out ../Figures

# Same NK landscape, three initial states, N=1e6 (replicate 0 is shown; the
# Evolved search takes about 8 h per replicate)
python generate_rugged.py --condition local   --architectures single_nk --N 1000000 \
       --only-replicates 0 1 --snapshots 50 100 150 200   --out ../Data/nk_local
python generate_rugged.py --condition evolved --architectures single_nk --N 1000000 \
       --only-replicates 0 1 --snapshots 1 50 100 150 200 --search-generations 5000 --out ../Data/nk_evolved
python generate_rugged.py --condition global  --architectures single_nk --N 1000000 \
       --only-replicates 0 1 --snapshots 1 50 100 150 200 --out ../Data/nk_global
for c in local evolved global; do
    python export_grand_network.py --data ../Data/nk_$c --out ../Figures/GN_$c --replicate 0
done
python fig_initial_states.py --local ../Data/nk_local --evolved ../Data/nk_evolved --global ../Data/nk_global \
       --gn-local ../Figures/GN_local --gn-evolved ../Figures/GN_evolved --gn-global ../Figures/GN_global \
       --out ../Figures
python fig_evolutionary_search.py --evolved ../Data/nk_evolved --global ../Data/nk_global --out ../Figures

# appendix: L=3, K=1, fully enumerable
python toy_nk_appendix.py --out ../Figures
```

Conventions inherited from the thesis runs, which the commands above already follow:
- the HoC-vs-NK suite used the explicit Wright–Fisher implementation and the Local suite the
  compressed one;
- the initial population is `t = 0` for `local` and `t = 1` for `evolved`/`global`;
- seeds depend on the position of each architecture in `--architectures`.

All three are documented in `generate_rugged.py` and `dynamics.py`.

The node layouts of the published `NK_initial_state_summary` drawings were made in Gephi. This code
produces a seeded ForceAtlas2 layout instead (`CHAPTER_NOTES.md`, B8).

---

## Verification

Each generator was compared against the thesis data. "Identical" means byte-identical files or
identical arrays.

| model / suite | compared | result |
|---|---|---|
| single peak, transition sweep | iteration 0 of all 21 `q`, plus iterations 0–1 at `q = 0, 0.5, 0.9`: `Means`, `Distribs`, `Abundances` at t = 99, 100 | identical |
| single peak, topology | iterations 0–1: `Abundances` at t = 99, 100 | identical (`Means` identical with numpy 1.26) |
| multi peak | 344 snapshots, compared with the original script run with the same seed (q = 0 and 0.3) | identical |
| HoC vs NK, replicate 0 (HoC and NK) | landscape seed, selected peak, `Z`, `f_l`, snapshot t = 50 | identical |
| NK Local, `N = 10⁶`, replicate 0 | peak, `f_l`, snapshot t = 50 | identical |
| NK Evolved, replicate 0 | first 20 search generations (all records and progress rows), final `f_l` | identical |
| NK Global, replicate 0 | DP optimum, `f_l`, snapshots t = 1, 50 | identical |

Every analysis script was run on the thesis data and its figure compared with the published one.
Data, theory curves and quoted numbers are the same; the only differences are tick-label
formatting from the matplotlib version, the two fixes listed in `CHAPTER_NOTES.md` (B9), and the
node layouts (B8). Examples: α_k = 2.69 ± 0.20, β_nn = 0.77 ± 0.08, γ = 0.75 ± 0.06;
Jaccard indices 0.07 / 0.03 / 0.00, 0.18 / 0.05 / 0.05, 0.18 / 0.18 / 0.18; `Z_best(5000)` =
0.847 / 0.842 vs `Z_global` = 0.899 / 0.898; and the appendix values (`TTG`,
`f_l = 0.243707040`, `n(TTG) = 4632`).
