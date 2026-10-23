---
title: "Choosing the number of latent dimensions d in GLLVMs and factor models"
date: 2026-09-26
source: "notebooklm c2c564e3-b5b1-44a6-9a5b-fee9f0ebbf7e"
tags: [gllvmTMB, GLLVM.jl, latent-variable-selection, model-selection, factor-analysis, JSDM]
---

# Auto-d: choosing the number of latent dimensions in GLLVMs and factor models

Decision this informs: whether GLLVModels.jl (Julia) and gllvmTMB (R) should estimate the
number of latent dimensions `d` automatically, and by which default rule. This note gathers
**inputs** to that decision; it does not make the call.

## Verdict

The corpus confirms the two-web-report picture and sharpens it with numbers. **Classical
linear/Gaussian factor-number selectors (Bai–Ng `PC_p`/`IC_p`, Ahn–Horenstein eigenvalue-ratio
and growth-ratio, Dobriban–Owen deterministic parallel analysis) are extremely well validated
by simulation** — near-100% exact recovery once `N` and `T` (or `n`, `p`) are in the hundreds,
degrading sharply only when factors are weak (low signal-to-noise) or dimensions are small.
**None of these classical selectors has been benchmarked on discrete (count/binary) GLLVM data
in the corpus** — every recovery table found is for continuous Gaussian factor models. The one
method purpose-built for non-Gaussian GLLVM order selection, Hui–Tanaka–Warton's Ordered Factor
LASSO (OFAL, 2018), has a strong qualitative simulation claim in its own abstract ("simulation
shows that the OFAL penalty performs strongly compared with standard methods... for order
selection... in GLLVMs") but **the corpus does not contain OFAL's actual numeric recovery
table** (full text stayed inaccessible even via a Monash repository mirror). Chen & Li's (2022)
joint-likelihood information criterion is the other GLLVM-relevant method with a genuine
high-dimensional generalized (binary/count) consistency proof, but likewise **no numeric
recovery table from Chen & Li surfaced in this corpus** (their arXiv preprint's own text did not
resurface the simulation tables in retrieved chunks, only theory/abstract). The ecological
software actually in use (`gllvm`, `boral`, `HMSC`, glmmTMB `rr()`) universally defers `d`-choice
to the user (default `num.lv = 2` in `gllvm`/`boral`, user-supplied rank in `rr()`), recommending
a refit-and-compare-by-AIC/BIC/cross-validation workflow that **has no simulation-validated
recovery guarantee anywhere in this corpus for non-Gaussian GLLVMs.** The naive LRT for `d` vs
`d+1` is confirmed non-standard (boundary/singularity breaks Wilks' theorem; Drton 2009 ties the
limiting distribution to Wishart eigenvalues), and the corpus's only valid fallback is
resampling (parametric/modified-parametric bootstrap), not a closed-form reference distribution.
For sparsity/shrinkage, OFAL (group-LASSO on Λ-columns) is qualitatively distinct from ridge/
Gaussian-prior shrinkage (never zeroes) and from MGP/CUSP Bayesian shrinkage (automatic,
MCMC-native, no refit step) — the corpus is consistent with, but does not extend, the web
reports' comparison table.

## Sub-question 1 — Has anyone chosen d automatically, with what recovery evidence?

Corpus-backed, with citations:

- **Bai & Ng (2002)**, panel `PC_p`/`IC_p` criteria, linear approximate factor model, `N, T →
  ∞`. 1,000 Monte Carlo replications, `k_max = 8`. At true `r = 1`: `PC_p2, IC_p1, IC_p2, IC_p3`
  give exact mean k̂ = 1.00 (SD 0) once `T ≥ 60`, `N = 100`. At true `r = 3`: same criteria give
  k̂ = 3.00 (SD 0) for `N=100, T≥60`. At true `r = 5`: k̂ ≈ 4.98–5.00 at `T=40` (slight
  underestimation), exact 5.00 by `T ≥ 60`. AIC/BIC-style criteria without the Bai–Ng
  cross-section correction (their `AIC1..3`, `BIC1..3` comparators) frequently hit the ceiling
  `k_max = 8` — i.e., **plain AIC/BIC over-select** in this design; only the `(N+T)/(NT)`-scaled
  panel penalties recover cleanly. [DOI 10.1111/1468-0262.00273]
- **Ahn & Horenstein (2013)**, eigenvalue-ratio (ER) and growth-ratio (GR) tests. 1,000
  replications, `k_max=8`. At true `r=1` or `r=5` with iid errors and `N,T=100`: ER exact
  recovery ~100% (SD 0.00–0.03); at `r=3` with correlated errors: 100%. **Weak-factor regime**
  (`N=1000, T=60, r=3`, varying SNR on the 3rd factor): ER recovers the true `r` in only 0.1% of
  runs at SNR=0.10, rising to 88.6% at SNR=0.45; GR does slightly better (0.3%→95.8%); a
  competing Onatski threshold test is *more* robust at very low SNR (47.4% at SNR=0.10) but
  plateaus lower at high SNR (~66%) than ER/GR. This is the clearest evidence in the corpus that
  **factor strength, not just n/p, drives whether any of these criteria recover d** — none is
  uniformly best. [DOI 10.3982/ECTA10574]
- **Dobriban & Owen (2019)**, deterministic parallel analysis (DPA/DDPA/DDPA+). Gaussian
  1-factor `n=500,p=300`: 100% exact recovery (SD 0). 2-/3-factor with one dominant factor:
  classical PA/DPA suffer "shadowing" (underselect), corrected by deflation (DDPA) and a raised
  threshold (DDPA+). Non-Gaussian Bernoulli-noise check at `n=75, p=300, d=1`: DPA still
  recovers `d=1` once signal clears the Marchenko–Pastur noise edge — the only non-Gaussian
  robustness check found for a classical selector, and it is still linear/PCA-based, not a
  GLLVM likelihood fit. [arXiv:1711.04155 mirror]
- **Chen & Li (2022)**, joint-likelihood information criterion (JIC) for high-dimensional
  generalized (binary/count) latent factor models — genuinely GLLVM-relevant and the closest
  match to gllvmTMB's setting. Per the original web-search report (A1), simulated true `K*=3`,
  candidates 1–5, `N∈{J,5J}`, `J∈{100,...,400}`: near-perfect recovery by `J=300–400`,
  underselection only for weak factors/small J. **The notebook corpus itself surfaced only the
  abstract/theory** for this paper (arXiv:2010.02326 added), not a re-confirmed numeric table —
  treat the A1 web-report's numbers as the authoritative record for Chen & Li, not this corpus
  pass.
- **Choi, Zou & Oehlert (2010)**, L1/adaptive-LASSO sparse factor analysis, Gaussian, `p=12,
  q=4`: validation-loss selection of `q` correct in 100/100 replications — but linear/Gaussian,
  small and idealized (`p=12`).
- **Hui, Tanaka & Warton (2018)**, OFAL — abstract claims strong simulated order-selection
  performance in GLLVMs specifically, but **no numeric recovery table surfaced in this corpus**
  even via full-text mirror attempts (Monash repository page returned metadata, not the table).

What the corpus does **not** answer: no Monte Carlo study anywhere in the corpus directly
compares AIC/BIC/EBIC/CV/Chen–Li/OFAL/parallel-analysis head-to-head on the same discrete
(Poisson/negative-binomial/Bernoulli) GLLVM data-generating process at ecological `n,p`.

## Sub-question 2 — Which criteria recover d best for count/binary data at ecological n, p?

Corpus-backed: at ecological scale (`gllvm 2.0`'s own worked example, Scottish ground beetles,
`n=88` sites, `m=68` species), **AIC and BIC disagree** — AIC/AICc select `num.lv.c=3`
(AIC=18,107.78) while BIC selects `num.lv.c=2` (BIC=20,368.82, vs 20,681.66 for d=3); the
practitioner guide ("One Toolbox, Many Tools") explicitly recommends AIC over BIC for this kind
of exploratory ordination choice, citing Aho et al. 2014 — a general model-selection philosophy
citation, not a GLLVM-specific recovery simulation. Chen & Li's JIC has the only real
consistency proof for count/binary data as `n,p→∞`, but (see above) no numeric table reconfirmed
in this pass. Bai–Ng and Ahn–Horenstein criteria are validated only on continuous Gaussian data
in this corpus; **the corpus contains no evidence that they have ever been applied to discrete
GLLVM data.** ERIC/EBIC is not evaluated for GLLVM dimension selection anywhere in the corpus at
all (the only "ERIC" hit is an unrelated database accession record). Cross-validation is used in
GLLVM packages for held-out deviance/variance-explained (e.g. Niku et al. 2019's amoebae/bird
count examples), not for direct integer-`d` recovery-rate reporting.

Absence, stated plainly: **there is no published head-to-head benchmark, in this corpus, of
which criterion best recovers true `d` for count/binary GLLVM data at n,p in the ecological
range (tens to low hundreds).** This is the single largest evidentiary gap for the decision.

## Sub-question 3 — Why is the LRT for d vs d+1 non-standard, and what is valid instead?

Corpus-backed, quoted:

- Hayashi, Bentler & Yuan (2007): *"when the number of factors exceeds the true number of
  factors, the likelihood ratio test statistic no longer follows the chi-square distribution due
  to a problem of rank deficiency and nonidentifiability of model parameters."*
- Drton (2009): *"At boundary points or singularities, the tangent cone need not be a linear
  space and limiting distributions other than chi-square distributions may arise... While
  boundary points often lead to mixtures of chi-square distributions, singularities give rise to
  nonstandard limits... in a study of the factor analysis model with one factor, we reveal
  connections to eigenvalues of Wishart matrices."*

Mechanism: under `H0: d factors`, the `(d+1)`-th factor's loadings are unidentified (arbitrary
rotation/rescaling gives the same likelihood), collapsing the Fisher information rank and putting
the true parameter at a **boundary/singularity** of the larger model's parameter space — this is
what breaks Wilks' theorem, not merely small-sample noise. The corpus's valid fallback,
consistent across both cited papers and general theory (Chernoff 1954, Chen/Moustaki/Zhang
2020b on latent-variable LRTs specifically), is **resampling**: a parametric or modified
parametric bootstrap that regenerates data under the fitted null model and recomputes the test
statistic empirically, rather than relying on an intractable analytical reference distribution.
No closed-form corrected reference distribution for the GLLVM case specifically was found.

Absence: no GLLVM- or gllvm-package-specific implementation of this bootstrap procedure was
surfaced in the corpus; the citations are general latent-variable/factor-analysis theory, not
gllvm/gllvmTMB-specific software.

## Sub-question 4 — OFAL vs ridge/Gaussian priors vs MGP/CUSP; tuning; post-selection inference

Corpus-backed:

| | OFAL (group-LASSO on Λ columns) | Ridge / Gaussian prior | MGP / CUSP |
|---|---|---|---|
| Mechanism | Hierarchical group-LASSO forces column `h+1` inactive unless column `h` active; adaptive-LASSO within active columns for entrywise sparsity | Quadratic `L2` penalty / N(0,τ²) prior, continuous shrinkage | Increasing-shrinkage prior (`τ_h = Π δ_l`, `δ_l>1`) or spike-and-slab sequence with rising spike mass |
| Zeroes a column exactly? | Yes — this *is* how it selects `d` | No | Effectively yes via shrinkage-to-near-zero / spike mass, not a hard L1 zero |
| Selection is automatic in one fit? | Yes | No (must be paired with refit-and-compare) | Yes (posterior does the work) |

Tuning (quoted): *"The OFAL penalty is the first penalty developed specifically for order
selection in latent variable models... In conjunction with using an information criterion which
promotes aggressive shrinkage, simulation shows that the OFAL penalty performs strongly compared
with standard methods and penalties for order selection..."* — i.e. OFAL's own paper pairs the
penalty-strength grid with an IC that "promotes aggressive shrinkage" (consistent with the A1b
web report's point that plain BIC/CV under-penalizes post-LASSO selection and that ERIC-style
corrected criteria exist for this purpose in the broader SEM literature, though not
re-confirmed as GLLVM-specific in this notebook pass).

Relaxed refit: the two-step adaptive-LASSO logic (quoted from Choi/Zou/Oehlert-adjacent material
in the corpus) — compute the penalized estimator, then re-weight and refit — is the corpus's
concrete mechanic for the general principle (fix the selected zero/nonzero pattern, then
re-estimate unpenalized on the active set to remove L1 shrinkage bias). No GLLVM-specific
worked relaxed-refit recipe (e.g. inside `gllvm` or a TMB fit) was found.

Post-selection inference: the corpus's only concrete valid mechanisms are (a) debiased/
decorrelated-score estimators from the hidden-confounder GLM literature (Ouyang et al. 2023-type
setting — regression coefficients *after* factor estimation, not the factor count itself), and
(b) the same modified-parametric-bootstrap logic as sub-question 3. Bayesian MGP/CUSP sidesteps
the problem by never doing a hard post-hoc selection step — inference is already marginal over
dimensionality in the posterior. **No paper in this corpus addresses valid inference specifically
for "how many latent dimensions" as the selected quantity** (as opposed to inference on
loadings/coefficients given a selected d) — this matches the A1 web report's original absence
finding and the corpus does not close that gap.

## Sub-question 5 — Package defaults

Corpus-backed, quoted where available:

- **gllvm**: `num.lv` defaults to 2; *"num.lv defaults to two latent variables unless we already
  include constrained... or informed latent variables."* Selection across a grid + AIC/AICc/BIC
  is the documented workflow, not an automatic default.
- **boral**: `num.lv = 2` by default, explicitly chosen for 2-D ordination plotting ("consistent
  with distance-based techniques like Non-metric Multidimensional Scaling"), not for
  data-driven accuracy.
- **HMSC**: corpus material describes a Bayesian increasing-shrinkage (MGP-style) prior as
  automatically down-weighting unneeded factors — **flag: the quoted sentence supporting this
  in this notebook pass comes from the general MGP/CUSP shrinkage literature, not a
  package-specific HMSC citation**, so treat "HMSC's exact current default mechanism" as
  **still unconfirmed for the actual `Hmsc` R package**, consistent with the A1 web report's own
  flagged absence on this point.
- **glmmTMB `rr()`**: rank `d` is user-supplied in the model formula (`rr(..., d)`); no
  automatic default rank; guidance is the refit-across-a-grid-and-compare-by-AIC/CV workflow
  (matches A1 web report, arXiv:2411.04411).
- **sjSDM**: does not use a low-rank latent-variable structure by default at all — fits a full
  (non-latent) covariance matrix, sidestepping the "choose d" problem, confirmed directly
  ("The package estimates a full (i.e. non-latent) jSDM...").
- **VAST / tinyVAST**: rank/structure set via user-specified SEM/RAM "arrow notation"
  (`make_sem_ram`) or a spatial-graph precision structure; the reduced-rank projection is
  explicitly user-parameterized, not auto-selected.

## Recovery evidence table

| Method | Model class | n / p (or N / T) tested | True d | Recovery | Source |
|---|---|---|---|---|---|
| Bai–Ng PCp2/ICp1-3 | Linear Gaussian factor | N=100,T≥60 | 1,3 | 100% (SD 0) | Bai & Ng 2002 |
| Bai–Ng PCp/ICp | Linear Gaussian factor | N=100,T=40–60 | 5 | 98–100% | Bai & Ng 2002 |
| Ahn–Horenstein ER/GR | Linear Gaussian factor | N=T=100, iid errors | 1,5 | ~100% | Ahn & Horenstein 2013 |
| Ahn–Horenstein ER/GR | Linear Gaussian factor, weak factor | N=1000,T=60 | 3 | 0.1%→88.6% (ER) as SNR 0.10→0.45 | Ahn & Horenstein 2013 |
| Dobriban–Owen DPA | Linear Gaussian factor | n=500,p=300 | 1 | 100% (SD 0) | Dobriban & Owen 2019 |
| Dobriban–Owen DPA | Non-Gaussian (Bernoulli) | n=75,p=300 | 1 | Recovers once above MP noise edge | Dobriban & Owen 2019 |
| Choi/Zou/Oehlert L1-FA | Linear Gaussian, sparse | p=12 | 4 | 100/100 | Choi, Zou & Oehlert 2010 |
| Chen & Li JIC | Generalized (binary/count) high-dim | J=100–400 (per A1 web report) | 3 | Near-perfect by J=300–400 | Chen & Li 2022 (numbers per A1 web report, not re-confirmed in this notebook pass) |
| OFAL | GLLVM (binary/count) | not extracted | — | Claimed strong, no table found | Hui, Tanaka & Warton 2018 |
| AIC vs BIC (gllvm 2.0 example) | Real GLLVM, negative binomial counts | n=88, m=68 | unknown (real data) | AIC/AICc→d=3, BIC→d=2 (disagreement, not recovery) | Korhonen et al. 2025 / gllvm 2.0 |

## Recommendation inputs for the decision (not the decision)

- Any automatic default rule for gllvmTMB/GLLVModels.jl that leans on AIC/BIC alone inherits a
  method with **zero discrete-data recovery evidence** in this corpus, and known disagreement
  even on a single real dataset (beetles: AIC picks 3, BIC picks 2).
- Chen & Li's JIC is the only criterion in the literature with a consistency proof matching the
  actual data type (binary/count, high-dimensional); its lack of a re-confirmed numeric table in
  this corpus pass is a gap to close by reading the paper directly (arXiv:2010.02326 is in the
  notebook) before relying on it as "validated."
- OFAL is the only method that estimates `d` and structure jointly in one fit for GLLVMs, but
  has no confirmed existing software implementation (per A1b web report) and no TMB/Laplace
  compatibility precedent found anywhere (non-smooth penalty at a non-smooth identifiability
  boundary).
- If any LRT-style d-vs-d+1 test is exposed to users, it must be bootstrap-based; a naive
  chi-square p-value is confirmed wrong by two independent theoretical sources.
- Eigenvalue-ratio/parallel-analysis methods are cheap (no refitting across a grid) but their
  only tested failure mode (weak factors) is exactly the ecological regime (small, noisy,
  ecological count matrices) where this decision most needs a validated default.

## Absences

- No Monte Carlo benchmark, anywhere in this corpus, of criterion recovery accuracy for `d` in
  a discrete (count/binary) GLLVM at ecological n, p — the central open question is unanswered.
- No numeric recovery table for OFAL (Hui, Tanaka & Warton 2018) despite two access attempts
  (DOI redirect, Monash institutional repository).
- No re-confirmed numeric recovery table for Chen & Li (2022) in this notebook pass (available
  only via the earlier A1 web-search report, not reproduced here from the arXiv source text).
- HMSC's actual current default latent-factor-count mechanism remains unconfirmed from
  package-specific material — the corpus surfaced only general MGP/CUSP shrinkage-prior theory,
  not an HMSC-authored statement.
- No GLLVM-specific post-selection-inference paper for the *number of latent variables itself*
  (as opposed to inference on coefficients/loadings after selection).
- ERIC/Extended-BIC is not evaluated for latent-dimension selection in any GLLVM/JSDM context in
  this corpus.
- Ridge/L2 vs LASSO/L1 rotation-invariance interaction with the lower-triangular TMB
  identifiability constraint (flagged as an open engineering risk in the A1b web report) was not
  re-investigated in this notebook pass.

## Sources

- Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models.
  Econometrica 70(1):191–221. DOI 10.1111/1468-0262.00273.
- Ahn, S.C. & Horenstein, A.R. (2013). Eigenvalue Ratio Test for the Number of Factors.
  Econometrica 81(3):1203–1227. DOI 10.3982/ECTA10574.
- Dobriban, E. & Owen, A.B. (2019/2020). Deterministic parallel analysis: An improved method for
  selecting the number of factors and principal components. arXiv:1711.04155.
- Chen, Y. & Li, X. (2022). Determining the Number of Factors in High-dimensional Generalized
  Latent Factor Models. Biometrika 109(3):769–782. DOI 10.1093/biomet/asab044; arXiv:2010.02326.
- Hui, F.K.C., Tanaka, E. & Warton, D.I. (2018). Order selection and sparsity in latent variable
  models via the ordered factor LASSO. Biometrics 74(4):1311–1319. DOI 10.1111/biom.12888.
- Choi, J., Zou, H. & Oehlert, G. (2010). A penalized maximum likelihood approach to sparse
  factor analysis. Statistics and Its Interface 3(4):429–436. DOI 10.4310/SII.2010.v3.n4.a1.
- Bhattacharya, A. & Dunson, D.B. (2011). Sparse Bayesian infinite factor models. Biometrika
  98(2):291–306. DOI 10.1093/biomet/asr013.
- Legramanti, S., Durante, D. & Dunson, D.B. (2020). Bayesian cumulative shrinkage for infinite
  factorizations. Biometrika 107(3):745–752. DOI 10.1093/biomet/asaa008; arXiv:1902.04349.
- Hayashi, K., Bentler, P.M. & Yuan, K.-H. (2007). On the Likelihood Ratio Test for the Number of
  Factors in Exploratory Factor Analysis. Structural Equation Modeling 14:505–526.
- Drton, M. (2009). Likelihood ratio tests and singularities. Annals of Statistics 37:979–1012.
  arXiv:math/0703360.
- Yuan, M. & Lin, Y. (2006). Model Selection and Estimation in Regression with Grouped
  Variables. JRSS-B 68(1):49–67. DOI 10.1111/j.1467-9868.2005.00532.x.
- Jacobucci, R., Grimm, K.J. & McArdle, J.J. (2016). regsem: Regularized Structural Equation
  Modeling. arXiv:1703.08489.
- Huang, P.-H. (2020). lslx: Semi-Confirmatory Structural Equation Modeling via Penalized
  Likelihood. Journal of Statistical Software 93(7). DOI 10.18637/jss.v093.i07.
- Park, T. & Casella, G. (2008). The Bayesian Lasso. JASA 103(482):681–686.
- Niku, J., Brooks, W., Herliansyah, R., Hui, F.K.C., Taskinen, S. & Warton, D.I. (2019).
  Efficient estimation of generalized linear latent variable models. PLOS ONE.
  DOI 10.1371/journal.pone.0216129.
- Niku, J., Hui, F.K.C., Taskinen, S. & Warton, D.I. (2019). gllvm: Fast analysis of multivariate
  abundance data with generalized linear latent variable models in R. Methods in Ecology and
  Evolution. DOI 10.1111/2041-210X.13303.
- Hui, F.K.C. (2016). boral – Bayesian Ordination and Regression Analysis of Multivariate
  Abundance Data in R. Methods in Ecology and Evolution. DOI 10.1111/2041-210X.12514.
- Tikhonov, G. et al. (2020). Joint species distribution modelling with the R-package Hmsc.
  Methods in Ecology and Evolution. DOI 10.1111/2041-210X.13345.
- van der Veen, B. et al. Parsimoniously Fitting Large Multivariate Random Effects in glmmTMB.
  arXiv:2411.04411.
- Warton, D.I. et al. (2015). So Many Variables: Joint Modeling in Community Ecology. Trends in
  Ecology & Evolution 30(12):766–779. DOI 10.1016/j.tree.2015.09.007.
- Ovaskainen, O., Abrego, N. et al. Latent Factor Models: A Tool for Dimension Reduction in Joint
  Species Distribution Models. In Statistical Approaches for Hidden Variables in Ecology, Wiley.
  DOI 10.1002/9781119902799.ch7.
- Kidziński, Ł., Hui, F.K.C., Warton, D.I. & Hastie, T. (2022). Generalized Matrix Factorization.
  JMLR 23. arXiv:2010.02469.
- sjSDM package (CRAN/GitHub, TheoreticalEcology/s-jSDM).
- tinyVAST package documentation and preprint (arXiv:2401.10193 / NOAA repository).
- Ouyang et al. (2023) and related GEE-assisted / hidden-confounder GLM factor-selection papers
  (imported via narrow research query; used only for the parallel-analysis-for-confounders
  cross-reference in sub-question 2).

> Related: [[projects/deep-research/README|deep-research]] · [[GLLVM.jl]] · [[projects/gllvmTMB]]
