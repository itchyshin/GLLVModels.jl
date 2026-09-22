# GLLVModels.jl speed lane S4–S8: phylo EM scaling class change, Poisson gradient hoists, grouped-route acceleration, analytic grouped gradient

## Numbers

| Slice | Fixture / Model | Metric | Before | After | Ratio | Derived from | Source file |
|-------|---|---|---|---|---|---|---|
| **S7** | EM phylo p=200 | loglik_check_ms | 1.3 | 0.5 | 0.38 | Table 3, item 1 | checkpoint.md:206–210 |
| **S7** | EM phylo p=200 | estep_ms | 0.7 | 0.3 | 0.43 | Table 3, item 1 | checkpoint.md:206–210 |
| **S7** | EM phylo p=200 | driver_ms_per_iter | 3.4 | 3.5 | 1.03 | Table 3, item 1 | checkpoint.md:206–210 |
| **S7** | EM phylo p=1000 | loglik_check_ms | 35 | 2.1 | 0.06 | Table 3, item 2 | checkpoint.md:206–210 |
| **S7** | EM phylo p=1000 | estep_ms | 14.5 | 1.1 | 0.08 | Table 3, item 2 | checkpoint.md:206–210 |
| **S7** | EM phylo p=1000 | driver_ms_per_iter | 83 | 3.2 | 0.04 | Table 3, item 2 | checkpoint.md:206–210 |
| **S7** | EM phylo p=5000 | loglik_check_ms | 2517 | 13.1 | 0.005 | Table 3, item 3 | checkpoint.md:206–210 |
| **S7** | EM phylo p=5000 | estep_ms | 1114 | 5.8 | 0.005 | Table 3, item 3 | checkpoint.md:206–210 |
| **S7** | EM phylo p=5000 | driver_ms_per_iter | 8235 | 17.9 | 0.002 | Table 3, item 3 | checkpoint.md:206–210 |
| **S7** | EM phylo scaling exponent | fitted exponent (p=200,1000,5000) | 2.42 | 0.51 | — | "EM_SCALING p^2.4" → "EM_SCALING p^1" | em_phylo_scaling_d807d96a7.tsv (before), em_phylo_after_1f4ae6632.tsv (after) |
| **S7b** | Grouped Poisson GLMM 200×5 | warm median wall (5 reps) | 0.192 | 0.192–0.195 | 1.0 | S7b before/after; vs Latte 0.015 s | grouped_warm_68c2f067c.tsv; checkpoint.md:80 |
| **S7b** | Grouped Poisson GLMM 200×5 | fresh CHOLMOD symbolic analyses | 1540 | 236 | 0.15 | S7b item 1: "2 per call, 0 thereafter" | checkpoint.md:82 |
| **S7b** | Grouped Poisson GLMM 200×5 | reused (cholesky!) | 0 | 1304 | — | "6.5× fewer fresh, 0 fallbacks" | checkpoint.md:83 |
| **S7b** | Grouped Poisson GLMM 200×5 | inner Laplace-fit calls | 118 | 118 | 1.0 | Optimiser unchanged; S4 corrected count | checkpoint.md:81 |
| **S7c** | Grouped Poisson GLMM 200×5 warm-started | before_warm_wall_s | 0.18472 | 0.15038 | 0.81 | 5 reps, warm-start cache threaded through | grouped_warm_68c2f067c.tsv:9 |
| **S7c** | Grouped Poisson GLMM 200×5 | cold-vs-warm loglik rel gap | — | 1.219e–13 | — | "shadow loglik gap" G7c.2 | leaf-S7c.md:15 |
| **S4** | Poisson Laplace gradient p=50 | gradient_wall_ms (banked, pre-R2 repair) | 170.5 | 95.6 | 0.56 | R2 repair already fixed; current baseline | checkpoint.md:21–25; docs/dev-log/core070/poisson-perf-diagnosis.md |
| **S4** | Poisson Laplace gradient p=50 (current) | Measured on origin/main 69a69b0a0 | 95.6 | 95.6 | 1.0 | Baseline for S6 gate G6.5 | checkpoint.md:21; Table 1 |
| **S6** | Poisson Laplace gradient p=20 | grad_wall_ms | 23.714 | 17.891 | 0.75 | G6.5, before from a detached worktree at 69a69b0a0 | laplace_after_020056f92.tsv |
| **S6** | Poisson Laplace gradient p=50 | grad_wall_ms | 102.569 | 83.892 | 0.82 | G6.5, same run | laplace_after_020056f92.tsv |
| **S6** | Poisson Laplace gradient p=50 | gradient/value ratio | 11.11x | 8.72x | 0.78 | G6.5, both already under the banked 24.3x | laplace_after_020056f92.tsv |
| **S6** | Poisson Laplace gradient p=20/50 | Optim iterations | 63 / 81 | 63 / 81 | 1.0 | Bit-identical Optim trajectory | leaf-S6.md:30 |
| **S8** | Grouped Poisson GLMM 200x5 (nθ=2) | driver wall, median 5 reps | 0.149013 | **0.122395** | **0.82** | GB.5, both settings in one process | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3, 500 groups (nθ=6) | driver wall, median 3 reps | 10.566280 | **5.928687** | **0.56** | GB.5, same run | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 200x5 | objective calls | 116 | 84 | 0.72 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3 | objective calls | 512 | 284 | 0.55 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 200x5 | inner Laplace-fit calls | 116 | 92 | 0.79 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3 | inner Laplace-fit calls | 512 | 303 | 0.59 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 200x5 | summed inner Newton iterations | 551 | 407 | 0.74 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3 | summed inner Newton iterations | 2735 | 1437 | 0.53 | GB.5 | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3 | outer gradient section, seconds | 5.7901 (20 FD invocations, 240 objective calls) | **0.6441** (19 analytic invocations) | **0.11** | GB.5; this is the section the slice targets | grouped_sections_after_6f2a98f36.tsv |
| **S8** | Grouped Poisson GLMM 5000x3 | FD-attributable share of wall, before | 0.954 | — | — | GA.2, the arc's go/no-go | grouped_sections_a2dc58557.tsv |

## Gates

**S8** (analytic outer gradient for the grouped non-Gaussian Laplace route). **5 of 8 gates ticked; the 3 that are not all wait on ONE open decision, see Known exceptions 5**:
- **GA.1 PASS**: a second fixture with nθ >= 6 and a section partition; sections sum to within 10% of the measured wall on both fixtures (0.082 small, 0.000 large), and the driver's loglik matches a real `fit_gllvm` call at rel 0.000e+00, so the partition is of the shipped code | leaf-S8.md:GA.1
- **GA.2 PROCEED**: the go/no-go. FD-attributable share on the larger fixture **0.954** against a 0.25 threshold (FD gradient 53.7%, Nelder-Mead 25.6%, final FD Hessian 16.0%, BFGS line search only 4.6%); 0.829 on the small fixture. This refuted the planning premise that the FD gradient was cheap at nθ = 2: that reasoning was per evaluation rather than cumulative across BFGS iterations | leaf-S8.md:GA.2
- **GB.1 PASS**: `docs/design/grouped-analytic-gradient.md` derives the gradient BEFORE any code (file mtime precedes the src edits), with a 28-row symbolic-alignment table mapping every symbol to a `file:line`. Only the envelope term dies at the mode; the log-det's implicit term through b-hat survives, and dropping it is the cheap wrong answer | leaf-S8.md:GB.1
- **GB.2 PASS**: the analytic gradient agrees with the existing central-difference gradient **per coordinate** at 20 random theta on **five** fixtures, rtol 1e-6, `inner_tol = 1e-10`, zero fallbacks to the FD path. Worst per-coordinate relative disagreement 7.281e-08 / 6.211e-09 / 1.986e-07 / 2.890e-08 / 1.571e-07; no coordinate had its FD reference below 1e-6, so none is a small-denominator artefact. The gate was AMENDED twice during the slice and **mutation-tested** each time. Zeroing the log-det implicit term fails all five (worst rel up to 1.79, with Beta and NB2 dispersion coordinates flipping sign), and halving the design-derivative trace term fails all five | leaf-S8.md:GB.2
- **GB.5 PASS**: objective calls, inner Laplace-fit calls and summed inner Newton iterations reported before and after on both fixtures; fixture A's wall recorded against the S7c-banked 0.150383 s and Latte's 0.015 s; the larger fixture against its own GA.1 baseline. See the Numbers table. Integrity: sections sum within 0.2% of wall on all four measurements, after-vs-before loglik rel 4.154e-15 and 2.011e-15 (rtol 1e-8), loglik vs a real `fit_gllvm` call rel 0.000e+00, converged everywhere, zero FD fallbacks, call counts identical across every rep | leaf-S8.md:GB.5
- **GB.3 HALF PASS, HALF OPEN**: fitted parameters and logLik equal origin/main 69a69b0a0 within rtol 1e-8, verified against a real detached worktree at that commit, given a byte copy of the lane's `Manifest.toml` so both sides resolve identical dependency versions; worst disagreement anywhere across four fixtures **3.753e-09** against the 1e-8 bound. The other half is the open decision: `test_grouped_laplace_identity.jl` reports 16 passed, 1 failed, and the single failure is a call-COUNT invariant, not a number | leaf-S8.md:GB.3
- **GB.4 NOT TICKED**: the warm start is still confined to Nelder-Mead. Running this gate's CHECK before making its change is what found the multi-term crash (Known exceptions 6); after the fix it reports **PASS 21/21**, with fixture D's cold and warm gradient norms bit-identical at 7.815970093361102e-8. That establishes the weaker fact that S8 does not break S7c's identity; UNCONFINING is not done and waits on the decision | leaf-S8.md:GB.4
- **GB.6 NOT TICKED, and blocked by the decision alone**: full `Pkg.test()` run to completion, **16320 passed, 2 failed, 0 errored, 19 broken, 16341 total, 94m37s**. The two failures are the pre-existing `test_em_louis.jl:127` flake this gate explicitly allows, and the one call-count assertion of the open decision. The gate permits exactly one failure, so it stays unticked, but **nothing else in 16,341 tests regressed**. Cross-check against S7c's banked suite (16249 passed, 1 failed, 19 broken): errors still zero, broken identical, passes up 71, failures up by exactly one | leaf-S8.md:GB.6

**S6** (per-site Poisson Laplace gradient: one `GradientConfig` per fit, shared mode solve, allocation-free site kernel). **All 6 gates PASS**; this slice HAS landed on the branch, which the earlier draft's "Not in this branch" section got wrong:
- **G6.1 PASS**: the new gradient equals the origin/main gradient at 50 random theta on p=5/20/50; verified **bitwise** (maxabsdiff == 0.0) at all 36 (p, theta) probes, with and without the shared config at chunk 12/24/32 | leaf-S6.md:10
- **G6.2 PASS**: fitted parameters and logLik equal origin/main within rtol 1e-6 at p=5/20/50; `test_poisson_grad_perf.jl`'s BASELINE_LOGLIK -14604.017303313138 holds at atol 1e-8, with that file now included by `test/runtests.jl` | leaf-S6.md:15
- **G6.3 PASS**: Optim `f_calls`, `g_calls` and iterations are not above the origin/main counts. Deviation flagged in the ledger: the gate self-derives its own pre-change baseline rather than using the banked 31/92/140, which names no fixture | leaf-S6.md:20
- **G6.4 PASS**: logLik agreement with the frozen gllvmTMB oracle holds. Deviation flagged: it reuses the ONE frozen Poisson oracle actually banked in this repo (NATIVE-03-POISSON, r_loglik = -634.1712844104393), not ten grid cells | leaf-S6.md:25
- **G6.5 PASS**: p=20 gradient wall 23.714 -> 17.891 ms, p=50 102.569 -> 83.892 ms, gradient/value ratio at p=50 11.11x -> 8.72x (both already under the banked 24.3x), Optim iterations unchanged at 63/81. Newton-solves-per-iteration == 1 verified **by code construction**, not by an external counter, because that counter lives outside this leaf's OWNS list; stated rather than glossed | leaf-S6.md:30
- **G6.6 PASS**: protected files unchanged from 69a69b0a0, and the full suite **16313 passed, 1 failed, 0 errored, 19 broken**, the single failure the same pre-existing `test_em_louis.jl:127` flake | leaf-S6.md:41

**S7c** (warm-started inner Laplace fits across outer FD evaluations):
- **G7c.1 PASS**: Grouped Poisson fitted params & logLik equal origin/main within rtol 1e–8; rel_ll = 1.22e–13, rel_par = 1.31e–6 | leaf-S7c.md:10
- **G7c.2 PASS**: Inner Newton iteration count falls from 711 (corrected S4) to reported number; cold-vs-warm shadow loglik gap rel = 1.219e–13 | leaf-S7c.md:15
- **G7c.3 PASS**: Warm median wall on 200×5 recorded; TSV bench/results/grouped_warm_68c2f067c.tsv written | leaf-S7c.md:20
- **G7c.4 PASS**: Full suite green except pre-existing em_louis flake: 16249 passed, 1 failed (em_louis.jl:127), 0 errored, 19 broken | leaf-S7c.md:25

**S7** (sparse-phylo EM hoists: items 1–4 landed):
- **G7.1 PASS**: Marginal logLik at fixed params equals origin/main within rtol 1e–12 (p=200/1000) | leaf-S7.md:10
- **G7.2 PASS**: Sparse-phylo & node gradients bitwise-equal to origin/main; takahashi_diag call count asserted == 1 | leaf-S7.md:15
- **G7.3 PASS**: E-step moments within rtol 1e–10 of dense; 50-iteration EM trajectory within rtol 1e–10 (p=200) | leaf-S7.md:20
- **G7.4 PASS**: Sparse monotonicity check flags same iterations as dense; SQUAREM decisions identical | leaf-S7.md:25
- **G7.5 PASS**: Fallback count zero over one EM fit; bench/results/em_phylo_after_1f4ae6632.tsv written; full suite 16228 passed, 1 failed (em_louis.jl:127), 0 errored, 19 broken | leaf-S7.md:28

**S7b** (CHOLMOD symbolic reuse, grouped):
- **G7b.1 PASS**: Grouped Poisson fitted params & logLik equal origin/main within rtol 1e–8; fresh symbolic 2→0 after first, zero fallbacks | leaf-S7b.md:10
- **G7b.2 PASS**: Warm median wall recorded; TSV bench/results/grouped_after_43db3ec85.tsv written | leaf-S7b.md:15
- **G7b.3 ABANDONED**: Full suite 16206 passed, 1 failed (em_louis.jl:127, pre-existing), 0 errored, 19 broken; test_grouped_laplace.jl unchanged | leaf-S7b.md:20

**S4** (profiling baseline):
- **G4.1 PASS**: Laplace gradient wall split at p=20/50; baseline 95.6 ms at p=50 confirmed on origin/main | leaf-S4.md:10
- **G4.2 PASS**: ForwardDiff chunk-pass count measured; 13 at p=50 with chunk=12, n_theta=149 | leaf-S4.md:15
- **G4.3 PASS**: Grouped 200×5 GLMM warm median wall 0.15–0.25 s (banked 0.192 s); split recorded | leaf-S4.md:20
- **G4.4 PASS**: EM phylo wall per iteration at p=200/1000/5000 measured; fitted exponent 2.42 (p^2.4 verdict); TSV bench/results/em_phylo_scaling_d807d96a7.tsv | leaf-S4.md:25
- **G4.5 PASS**: No src/ file differs from origin/main | leaf-S4.md:30

## Commits by slice

**S4** (bench/profile scripts, baseline):
- d56bccea4: lane(speed78-20260919): scaffold
- d1ef337e9: handover: leaf-S4 checkpoint
- eb8b868ac: bench: leaf-S4 profiling scripts
- d807d96a7: bench(profile_laplace_allocs): p=50 gradient baseline 170.5 → 95.6 ms (stale bank; current is 95.6)
- fb1d4f42e: lane(speed78-20260919): S4 done; order re-set by leverage to S7b, S7, S6

**S7b** (grouped Laplace CHOLMOD symbolic reuse):
- 3ff8bfbeb: test(grouped-laplace): S7b identity gate, pinned pre-fix baseline
- 4a11bd4b4: fix(grouped-laplace): reuse the CHOLMOD symbolic factorisation across one fit's inner Newton loop
- e7ca4ff49: bench(profile_grouped_glmm): add --gate after for the S7b before/after comparison
- e9c5fa16d: fix(test): drop DelimitedFiles from test_grouped_laplace_identity.jl
- 43db3ec85: docs(core070): re-bank the p=50 gradient wall, 170.5 ms predates the R2 repair
- 55c7c0b97: lane(speed78-20260919): S7b outcome; S7c warm-started inner fits added after S7
- 12de911e3: handover: S7b checkpoint — gates, before/after numbers, and the pre-existing test_em_louis flake

**S7** (sparse-phylo EM: Woodbury reuse, cached X_G, sparse loglik routing, takahashi_diag cache, CHOLMOD reuse):
- 1f472cab7: perf(em_phylo): Woodbury reuse, cached X_G, sparse loglik routing, CHOLMOD symbolic reuse (items 1/2/4)
- 5369d5ae9: perf(node_gradient): compute takahashi_diag once per gradient, not twice (item 3)
- 4d3508c5d: test(sparse-phy): identity gates G7.1–G7.4 for the S7 sparse-phylo EM changes
- cfdaf7477: bench(em_phylo_scaling): add --gate after mode; fix two measurement artifacts
- a83d8eac1: checkpoint: S7 fully closed — G7.5 full Pkg.test() completed, PASS modulo the known test_em_louis flake
- ad578e52c: fix(test): rename _RESULTS/_REASONS/_check! to avoid Main-namespace collision
- 97db31fc3: checkpoint: S7 done (items 1–4), G7.1–G7.4 PASS, G7.5 full-suite pending

**S7c** (warm-started inner Laplace fits across outer FD evaluations):
- fb30922dd: feat(grouped_laplace): b_init warm-start keyword on joint_grouped_laplace_loglik
- de57fb3a0: feat(grouped-nongaussian): thread a warm-start cache through the outer fit; default on
- 4cc962e9a: fix(grouped-nongaussian): confine warm-starting to Nelder-Mead; BFGS/FD stay cold
- 68c2f067c: bench(grouped_glmm): --gate warm (G7c.2) and --gate after_warm (G7c.3)
- e05a75365: test(grouped-laplace): fixture D regression test for the Nelder-Mead-only warm-start fix
- 388f0b135: bench(grouped_glmm): mirror the Nelder-Mead-only warm-start split in the shadow driver
- 10d0bfb11: test(sparse-phy-identities): rename _GATES to _S7_IDENTITY_GATES; it collided with test_grouped_laplace_identity.jl's constant

**S6** (per-site Poisson Laplace gradient):
- a4b40794e: test(laplace-grad): identity gates G6.1-G6.4 for the S6 Poisson Laplace perf change (tests first)
- 16486a4e9: test(runtests): wire test_poisson_grad_perf.jl (S6 item 4)
- 184523155: perf(laplace_grad): allocation-free _poisson_site_diffable (S6 item 3)
- 020056f92: perf(laplace_grad,poisson-fit): one GradientConfig per fit + shared mode solve (S6 items 2 and 1)
- 9382c00bc: checkpoint: S6 done (items 1-4), G6.1-G6.5 PASS, G6.6 full suite pending
- c825f68f9: fix(poisson-fit): the shared mode solve honours the fit's newton_maxiter / newton_tol (S6, Gauss review)
- 2fea78d78: docs(laplace_grad, node_gradient): reattach the poisson_laplace_grad and node_grad docstrings
- a2dc58557: docs(low-level-reference): list _grouped_nongaussian_objective (S7c gave it a docstring)

**S8** (analytic outer gradient for the grouped non-Gaussian Laplace route):
- f59757a4f: bench(profile_grouped_glmm): section partition and a second nθ>=6 fixture (slice A, measure first)
- 7ae2f8cbf: feat(grouped): analytic outer gradient for the non-Gaussian Laplace route (S8, GB.1-GB.3)
- ec76a2090: fix(grouped-analytic): dk W placeholder width, and the fixture that catches it
- 47da8c29e: test(grouped-analytic): run the S8 gate file inside the suite, and let it load
- 6f2a98f36: test(grouped-analytic): a fifth fixture, loadings and placeholders together
- bfa4ee6bf: bench(profile_grouped_glmm): --gate sections_after, and GB.5 measured

The order is deliberate and is the slice's own discipline: **measure first** (f59757a4f, which produced the
25% go/no-go), **derive before coding** (the design document landed with 7ae2f8cbf but was written before the
src edits, which GB.1 checks by file mtime), then implement, then fix what a gate caught, then widen the gate
so it could have caught it.

## Known exceptions

1. **Pre-existing test_em_louis.jl:127 flake**: All full-suite runs in S7b, S7, and S7c report exactly one failure at test_em_louis.jl:127 ("SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs (p=10)", rel = 0.0010560 vs threshold 0.001). This is a pre-existing, order-dependent, deterministic flake confirmed NOT caused by any arc in this lane: (a) test_em_louis.jl never touches grouped_laplace.jl or sparse-phylo code, (b) it runs at position 54 in runtests.jl, strictly BEFORE test_grouped_laplace_identity.jl at position 97, (c) it passes 147/147 standalone on origin/main 69a69b0a0. Documented in leaf-S7b.md:127–152 and leaf-S7.md:239–242.

2. **S7b warm wall: no proportional gain despite 6.5× CHOLMOD reduction**: The CHOLMOD symbolic reuse reduced fresh analyses from 1540 → 236 (0.15 ratio) and added 1304 reused factorizations with zero fallbacks, an identity-verified, mechanically correct optimization. Wall clock before/after: 0.192 s (both runs). This is surfaced in checkpoint.md:88–102: symbolic analysis for this fixture's ~200×200 sparse system is already sub-millisecond fresh; the ~0.19 s wall is dominated by per-Newton-iteration GLM state/score/curvature computation in `_joint_grouped_state`/`_joint_grouped_components` (O(n=1000) loop over all observations, regardless of m=200 random-effect dimension). Outside this arc's scope; surfaced for future investigation.

3. **S7c warm-start unconfined path NOT shipped**: The unconfined warm-start experimental run measured before_warm_wall = 0.099 s (checkpoint.md during drafting, not the final TSV). This path is NOT included in the landing commits; commit 4cc962e9a ("confine warm-starting to Nelder-Mead; BFGS/FD stay cold") restricts warm-start to the Nelder-Mead-only portion of the fit. The shipped path delivers 0.1847 → 0.1504 s (0.81 ratio, 18% gain).

4. **19 pre-existing broken tests**: All full-suite runs report 19 broken (pre-existing @test_broken entries), unrelated to any change in this lane.

5. **ONE OPEN DECISION, AND IT IS THE ONLY THING BETWEEN S8 AND GREEN.** `test/test_grouped_laplace_identity.jl:49` asserts that fixture A takes exactly **118** inner Laplace-fit calls. With the analytic gradient on it takes **94**. Every numeric identity in that file still passes at rtol 1e-8 (logLik, logdet_precision, fitted parameters) and only the COUNT moved. That assertion was banked for slice S7b, a CHOLMOD-reuse change that had to be numerically AND procedurally invisible; its own comment says "the reuse must not change the optimiser's path". S8 exists to change that path, and a 20.3% cut in inner Laplace fits is the first direct evidence it does what it was built to do. The invariant is right for S7b and wrong for S8, and the two slices now disagree about what "identity" means. Three options, **none taken here**: (a) make the invariant conditional on `analytic_gradient`; (b) rebank it at 94 and say why; (c) ship S8 with `analytic_gradient = false` by default until (a) or (b). Not decided in this arc because it changes a gate another slice depends on, and that file is not in leaf-S8's OWNS list. **Nothing was edited to make it green: no tolerance widened, no assertion touched.** GB.3's second half, GB.4 and GB.6 all wait on this one choice.

6. **A real defect the arc found and fixed in itself: the analytic gradient CRASHED on any model with two or more grouping terms.** Found by running GB.4's CHECK before making GB.4's change, purely to learn whether it was green. `DimensionMismatch` thrown from `_grouped_analytic_loglik_gradient` at `dW * bhat`, on fixture D's 4-source `common = true` design, reached through the ordinary public `fit_gllvm`. It threw rather than falling back to the FD gradient, so it was a hard user-facing regression sitting on the branch. Root cause, proved by direct probe before any edit: the zero PLACEHOLDER block pushed in `_grouped_laplace_design_jacobian` for each grouping term the coordinate does not belong to used the trait-factor `width`, where the real block is `kron(incidences[s], Lstar)` and so is `size(incidences[s], 2) * width` columns wide. **With one grouping term no placeholder is ever pushed**, which is exactly why the three original GB.2 fixtures, the whole of GB.3 and the first speed measurement were all blind to it. Fixed with one line, and GB.2 was then amended to require a fixture with two or more terms AND unequal group counts, because equal counts would let a coincidental width match hide the same defect. Mutation-tested: reverting the fix leaves the three original fixtures passing at their identical worsts and fails only the new one.

7. **Two wiring gaps, both of which meant the S8 gate file had never actually run where it mattered.** (a) `test/test_grouped_analytic_grad.jl` was never included in `test/runtests.jl`, so the file GB.2 and GB.3 both name as their CHECK had never executed inside the suite or in CI — only when invoked by hand. (b) Once included it could not load: it `using`s `Printf`, which resolves under `julia --project=.` but was absent from `test/Project.toml`, so `Pkg.test()` threw `ArgumentError: Package Printf not found` out of the `@testset` and **every file after `runtests.jl:192` silently never ran**, producing an 11-minute "3847 pass" line that reads like an almost-clean suite and is a partial count. Fixed by adding the stdlib, following this repo's own precedent `f15ae2f52`. **OWNS extension declared, not slipped in:** `test/Project.toml` is not in leaf-S8's OWNS list and was edited anyway for one line, after checking that no live lane had touched it since origin/main.

8. **A withdrawn number, recorded because it was quoted before it was reproduced.** An interim measurement during the slice reported the small fixture at 0.085829 s / 1.64x. It does not reproduce. Three later measurements put it at 0.120141 s, 0.122395 s and 0.117778 s, the last through a separate cross-check calling `fit_grouped_nongaussian` directly, and the machine was LESS loaded for those runs, so contention cannot explain the direction. **The gated number is 0.122395 s / 1.217x** and that is what the Numbers table carries. The LARGE fixture's interim 1.78x did reproduce (1.772x on the independent cross-check) and stands.

9. **Read every S8 speedup as a FLOOR, not a ceiling.** Two of the three things GA.2 measured are still paid in full on the shipped path: the S7c warm start is still confined to Nelder-Mead (GB.4 not done) and the final O(nθ²) FD Hessian is still computed for the diagnostics, which GA.2 put at 16.0% of the large-fixture wall on its own. On that fixture the gradient section itself fell 9.0x (5.7901 s -> 0.6441 s); the wall fell 1.78x because Nelder-Mead (2.85 s) and the Hessian (1.69 s) did not move.

10. **Latte.jl is still ahead on the 200x5 fixture and no claim is made otherwise.** 0.122395 s against Latte's 0.015 s: the gap closes from about 10.0x to about 8.2x. The arc's stated goal was "materially faster than 0.150 s on that fixture", which is met by about 19%.

## Still to write before this body ships

- The **S6 section above was reconstructed from `leaf-S6.md` by a later session**, not by the session that ran the slice. Its gate lines and the two G6.5 rows in the Numbers table are quoted from that ledger and are accurate to it, but S6's own checkpoint prose (the kind the S7/S7b/S7c sections have) has not been written, and the ledger records four non-blocking Gauss-review follow-ups (G6.1's rtol looser than the measured bitwise; G6.3's mirror `fg!` measuring near-unchanged code; `_grad_gcfg` as shared mutable per-fit state; and the bench script's unrelated G4.1 sub-check reading FAIL as a side effect, out of OWNS). Those belong in Known exceptions before this ships.
- **This body must not be posted while the open decision in Known exceptions 5 is open**, because CI on it would go red for something that is not a defect.

## Not in this branch

Nothing. The earlier draft of this document listed **S6** here with all gates "PENDING" and the arc "not started". That was true when it was written on 2026-09-20 and is now wrong on both counts: S6 landed in commits a4b40794e through c825f68f9, and all six of its gates are ticked with evidence in `leaf-S6.md`. Corrected rather than left standing, because a reviewer reading the old text would have looked for S6's changes on a different branch.
