# Suggested corrections to Chapter 4 (`chap-4-peaks-landscapes-simulations.tex`)

While tracing every figure of the chapter back to the code and data that produced it, I found the
points below where the text and the simulations disagree, or where the text leaves out a detail a
reader would need to reproduce the result. Each item gives the evidence and a suggested fix.
Line numbers refer to `old_codes/chap-4-peaks-landscapes-simulations.tex`.

Items are grouped by how much they matter:

- **A**: the text describes something different from what was simulated.
- **B**: information is missing for reproduction.
- **C**: answers to the `% CONFIRM` notes already in the source.

---

## A. Text and simulation disagree

### A1. Rugged landscapes: the threshold `f_l` is *not* shared by the Local, Evolved and Global runs (l. 1514)

The text says: *"when different initial populations are compared on the same quenched landscape, the
threshold f_l is determined once and then kept fixed."*

In the data, the three conditions use the same quenched NK field (`base_seed = 12345`, replicate 0,
`field_seed = 5000012389`), but each run recalibrated `f_l` at its own initial genotype:

| condition | initial genotype `g0` | `Z(g0)` | `f_l` |
|---|---|---|---|
| Local   | best of 16 local maxima          | 0.79108 | 0.77656 |
| Evolved | best genotype of the 5000-generation search | 0.84743 | 0.83057 |
| Global  | exact global maximum (dynamic programming)  | 0.89911 | 0.87991 |

So the *effective* landscape `f = max(0, Z - f_l)` differs between the three rows of
Fig. `NK_initial_state_summary`, which is exactly the confound the sentence says was avoided. In each
case `f(g0)/<f>_{one-mutants} = e^{0.55}`, so each population starts with the same local contrast.

Suggested fix (text only, results unchanged): replace the sentence with something like
*"In each condition the threshold is calibrated at the corresponding initial genotype, so that all
three populations start with the same local selective contrast R_loc. The raw landscape Z is the same
in the three conditions, but the effective fitness landscape is not."* The discussion in
Sec. 4.4.5 (turnover, Jaccard indices, CCDF shapes) should then note that part of the difference
between conditions may come from the different thresholds.

Alternatively, rerun the Evolved and Global conditions with the Local `f_l`. This would need a small
code change (an `--f-l` override) and new figures.

### A2. Multiple peaks: where the new peak is placed, and the protected copy (l. 862, 1102)

The text says each new genotype is introduced *"at Hamming distance δ from the instantaneous population
root (the genotype with maximum abundance)"*. The generator that produced `Data/model4a30`
(`multi_peak_quasispecies.py`, reproduced by `multi_peak/generate_multi_peak.py`) does the following
instead:

1. The new peak is placed exactly δ mutations from the **previously introduced peak** (`g_j`), not from
   the most abundant genotype. Usually the two coincide, because the population is localized on the
   current peak. They need not coincide during a replacement.
2. The new peak enters the population as **one copy**: at generation `t = jτ` the first offspring
   produced is replaced by the new peak.
3. **Position 0 of the population is never resampled.** It holds `g0` until the first introduction
   and the newest peak afterwards. So the newest peak always keeps at least one copy and cannot be
   lost by drift. In particular, every introduced peak is guaranteed to establish. This does not hold
   in a plain Wright–Fisher model, where a single beneficial copy with relative advantage
   `1 + Δf/F_j` is lost with appreciable probability.

Suggested fix: describe the protocol as implemented, e.g. *"Every τ generations a new genotype
g_{j+1}, at Hamming distance δ from the previous peak g_j, is inserted as a single individual and
assigned fitness F_{j+1} = F_j + Δf. One copy of the most recently introduced peak is kept in the
population at all times (it is exempt from resampling), so that every introduced peak establishes."*
Point 3 also matters for Fig. `phi_vs_time` and the replacement times.

### A3. Single peak: two mutation conventions were mixed (Sec. 4.2)

Two versions of the single-peak generator were used. They differ in how the `F` offspring of a parent
receive mutations:

- **per parent**: one Poisson draw per parent, shared by all its `F` offspring (as in the Chapter 3
  models);
- **per offspring**: one independent Poisson draw for each offspring.

Which version produced each dataset (checked by regenerating iteration 0 of every `q`
bit-for-bit with both versions):

| data | convention |
|---|---|
| transition sweep, `q = 0, 0.1, 0.2, …, 1.0` (generated 2026-07-09) | per parent |
| transition sweep, `q = 0.05, 0.15, …, 0.95` (generated 2026-07-21) | per offspring |
| topology runs, `N=10⁵, L=250, F=3, q=0` | per offspring |

So the curves of Figs. `mean_hamming_and_beta` and `Qk` (`N=10³, L=10⁴, F=2`) alternate between the
two conventions from one `q` value to the next. In Fig. `Qk`, `q = 0.3, 0.6, 0.9` are per parent and
`q = 0.75` is per offspring.

The mean-field recursion `x_{t+1} = r_q x_t / [1 + (F-1) x_t]`, `β_q`, `x_*`, `m_*` and the generating
function `G_*` are expectations and are identical under both conventions. Fluctuations, and the joint
occupancy of neighboring genotypes, are not. The finite-pool `m = 1` construction
(Eq. `model-fq-conditioned-incidence`) assumes independent offspring, which matches the topology data.
No visible discontinuity between neighboring `q` values appears in the figures, which is consistent
with the expectations being equal.

Suggested fixes (choose one):
1. state the replication rule in Sec. 4.2.1 and add a footnote that the transition sweep combined
   both conventions; or
2. regenerate the 10 per-parent `q` values with `--mutations per-offspring` (about 10 min in total at
   `N=10³`), so that the whole chapter uses independent offspring mutations.

---

## B. Missing for reproduction

### B1. `N` in Secs. 4.4.5–4.4.6 is `10⁶`
The Local, Evolved and Global production runs and the 5000-generation search all used
`N = 1 000 000` (suite metadata). The HoC-vs-NK comparison (Sec. 4.4.4) used `N = 100 000`. This answers
the `% CONFIRM` note at l. 1692.

### B2. Time labels differ by one generation between conditions
In the Local runs the initial population is `t = 0`, so the snapshot labeled `t` follows `t` generations.
In the Evolved and Global runs the initial population is labeled `t = 1`, so the snapshot labeled `t`
follows `t − 1` generations. Figure `NK_initial_state_summary` therefore compares, for example,
`t = 50` (Local) with 49 generations (Evolved, Global). The effect is negligible, but worth a footnote.

### B3. What "Evolved" means
Production for the Evolved condition does **not** continue the population of the 5000-generation
search. It starts again from `N` identical copies of the **highest-Z genotype ever occupied** during
the search (`Z = 0.84743`, first found at generation 4493, 66 mutations from the search root). It then
recalibrates `f_l` at that genotype (A1). The search itself:
- starts from the same local maximum as the Local condition;
- uses `N = 10⁶`;
- keeps `f_l` fixed at the value calibrated at that starting maximum (0.77656) for all 5000 generations.

After the search, 64 extra adaptive walks and a final steepest-ascent step are run. They did not change
the genotype (`candidate_source = long_evolution_best_ever`).

Suggested wording for l. 1683: *"...denoted Evolved, starts from N copies of the highest-scoring genotype
found during a 5000-generation Wright–Fisher search (N = 10⁶) that started at the Local peak."*

### B4. How the "selected local fitness maximum" is chosen (l. 1549, 1682)
In every run, the local maximum is the highest of the first **16 distinct local maxima** reached by
steepest-ascent adaptive walks (moving each step to the best of the `3L` one-mutant neighbors) that
start from uniformly random sequences. Ties are broken by the mean score of the one-mutant shell.
This is not stated anywhere. One sentence in Sec. 4.4.4 would do.

### B5. Fig. `localization_parameter_region` uses `λ = 0.2`
The figure is drawn with `lambda_mut = 0.20` (`16-Simulations/old/final_figures.ipynb`). The
surrounding text uses `λ = 1` (where `F_c = e ≈ 2.72`), and the caption gives no value. Either state
`λ = 0.2` in the caption, or redraw with `λ = 1` (`single_peak/fig_localization_region.py --lam 1`).

### B6. HoC-vs-NK figure
The three displayed realizations are replicates 0, 1 and 2 of the 10 simulated per architecture
(`base_seed = 12345`). The `±` values of `ᾱ_k`, `β̄_nn` and `γ̄` are **sample standard deviations
(ddof = 1) over the three displayed realizations**; this answers the `% CONFIRM` at l. 1629. The
regenerated values are 2.6887 ± 0.2046, 0.7731 ± 0.0840 and 0.7462 ± 0.0675.

### B7. Multi-center analysis uses inferred centers (Sec. 4.3.3, Eq. `model4a30-voronoi-assignment`)
The text writes the centers as `C_e = {g_0, …, g_{e−1}}`, i.e. the introduced peaks. The old generator
never saved the peak sequences, so the analysis **infers** `e` effective centers from each snapshot:
- the most abundant LCC genotype;
- the ancestral genotype, if it is in the LCC;
- then abundance local maxima (ranked by abundance, then degree) at least `max(2, δ/2)` mutations
  from the centers already chosen.

Suggested fix: one sentence saying the centers are reconstructed from abundance maxima. (The new
`multi_peak/generate_multi_peak.py` saves the true peak sequences in its metadata, so a future
analysis could use them directly.)

### B8. Fig. `NK_initial_state_summary`: node layouts
The network drawings use node positions laid out interactively in Gephi (the
`GN_LCC_*_positions.gexf` files). They cannot be regenerated by script. The repository code produces a
seeded ForceAtlas2 layout instead, so the drawings look different, while the CCDFs, Venn counts and
Jaccard indices are identical.

### B9. Minor issues in the thesis figures
- `model-fq_mean_hamming_and_beta_panel`: the plotting script loaded every file in `Means/`, which also
  held 5 runs with `N = 10⁵, L = 250` at `q = 0.7`. Those runs are averaged into the `q = 0.7` curve
  and its β distribution (mean β changes by 2·10⁻⁴; invisible). The repository script keeps only
  `N = 10³, L = 10⁴`.
- `model4a30_variance_dispersion_origin_root_panel`: the panel labels are printed as the literal
  text `\textbf{a}` … `\textbf{d}` (LaTeX was not enabled). Fixed in the repository script, which now
  prints `(a)` … `(d)`.

---

## C. Answers to the `% CONFIRM` notes

- l. 1692 (population size of Secs. 4.4.5–4.4.6): `N = 10⁶`; see B1.
- l. 1460 (origin of `λ = 0.275`): `λ = 0.275` is the default of the generators (`DEFAULT_LAMBDAS`) and
  of every suite. The code records no derivation for this value. With `c = 2` it gives
  `R_loc = e^{0.55} = 1.733` and `p_0 R_loc = e^{0.275} = 1.32 > 1`.
