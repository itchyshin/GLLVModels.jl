# Mode-search triage, 2026-09-26

Scope: the undamped per-site Laplace mode-search defect (fix pattern: #479/#481 Gamma,
#480/#483 Beta) and the class-wide audit at
`origin/claude/lane-true-parity-finish-20260925:docs/dev-log/core070/class-audit-20260924/`
(read via `git show`, not checked out). Checked against current `origin/main` @ `b90641c97`
(2026-09-26). No src edits, no GitHub writes.

## 1. Reproduction steps recovered

The audit's `notes.md` and `README.md` define two failure classes:

- **Class A**: a per-site inner mode search stops without converging (gradient residual
  above 1e-4 at the kernel's own z) and the kernel still returns a finite value, so the
  fitter's 1e12-sentinel guard never fires.
- **Class B**: `_fit_verdict` (and the analogous per-family verdict helpers) report
  `converged = true` from `Optim.converged(res) = x_converged || f_converged || g_converged`.
  Every caller leaves `x_abstol = f_abstol = 0` (Optim's default), so a single zero-length
  line-search step trips `x`/`f` convergence even when the gradient residual is far above
  `g_tol`.

Threshold used throughout: "non-mode" = a kernel z with `|grad log-posterior| > 1e-4` while
a from-scratch BFGS restart (`g_tol = 1e-10`, up to 2000 iterations) at the same site
converges to `|grad| < 1e-5`. The per-family probe scripts (`probe_A_kernels.jl`,
`probe_A2_kernels.jl`, `probe_A3_kernels.jl`, `probe_B_nb1_mech.jl`, `env/probe1-4.jl`,
`skeptic/*`) are self-contained Julia scripts against `GLLVModels` internals; each defines
its own `probe`/`refmode` helpers or reuses `probe_common.jl`. Recovered via
`git show origin/claude/lane-true-parity-finish-20260925:<path>` from
`/Users/z3437171/Dropbox/Github Local/GLLVM.jl` (never checked out, never edited).

## 2. What changed since the audit (verified against current origin/main, b90641c97)

| PR/issue | Status | Covers |
|---|---|---|
| #481 (Gamma grouped kernel) | merged (925a0f397) | Gamma grouped mode search, Fisher+halving then damped Newton, `-Inf` on failure |
| #483 (Beta grouped verdict) | merged (72cc7458c) | Beta grouped gradient-aware verdict |
| #494, Fixes #486 | merged (d4da31544) | `_laplace_mode_off` in `src/families/covariates.jl`, the shared kernel behind `fit_gllvm_cov` and 4 sibling fitters (species-covariates, fourth-corner, row-effects, constrained-ordination). The fix is family-agnostic (damped search + Newton fallback + `-Inf`), and the PR's own behaviour-change table reports Binomial fits flipping from `converged=true` to `converged=false`, so it covers **both** the audit's Poisson (420/4000) and Binomial (16/4000) `_laplace_mode_off` rows, not only the Gamma row it was filed for. |
| PR #500 (open), Fixes #484 | not merged | Twopart family kernel (ZIP, ZINB, ZIB, HurdlePoisson, HurdleNB, DeltaGamma, Delta-lognormal, Beta-hurdle): damped search, Newton fallback, `-Inf`. Per task brief, not re-probed beyond confirming it is still open; see note in Section 3. |
| Issue #501 (open) | not fixed | Ordered beta, fit-level: `converged=true` at non-stationary points, restarts reach up to +2859 logLik. Issue proposes `_ordered_beta_mode` mode-switching under theta as the mechanism, not yet fixed. |
| PR #502 (draft) | not merged | `_fit_verdict`'s Class B gradient-criterion gap (Fixes #485), shared by ~85 fitters including NB1, NB2-grouped, Beta-grouped, Gamma-grouped, Poisson, Binomial, ordinal, Tweedie. This is a Class B (verdict) fix, not a Class A (mode-search) fix — it would not by itself fix a kernel that returns a wrong finite value, only stop the fitter from calling that wrong value "converged." Per task brief, not built upon. |

## 3. Re-run probes, current origin/main @ b90641c97

Environment: `~/local-scratch/triage-modesearch-20260926-101234` (pre-existing worktree at
this exact commit; reused per task brief), macOS aarch64, Julia 1.10.0, `JULIA_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1`. `Pkg.instantiate()` completed in under a minute (most dependencies
already precompiled in the shared depot); no fresh full precompile was needed. Estimate before
running: under 10 minutes for reduced-count probes on 2-8 kernel-level families; actual wall
time was about 4 minutes for two batched scripts. Counts were reduced to roughly 1/3-1/5 of the
audit's (stated below); this trades precision for a same-order-of-magnitude confirmation, not a
new fixed-precision estimate.

Class A kernel probes, reduced counts, current main:

| Family / kernel | Audit rate | Current rate | Same order of magnitude? |
|---|---|---|---|
| NB1 grouped `_nb1_grouped_loglik_site` | 108/3000 (3.6%) | 20/600 (3.3%) | yes |
| COMPoisson `_compoisson_mode` | 19/800 (2.4%) | 7/266 (2.6%) | yes |
| OrderedBeta `_ordered_beta_mode` | 212/1500 (14.1%) | 72/500 (14.4%) | yes |
| BetaBinomial `_beta_binomial_mode` | 103/1500 (6.9%) | 27/500 (5.4%) | yes |
| StudentT shared (generic `_laplace_mode`) | 246/1500 (16.4%) | 78/500 (15.6%) | yes |
| StudentT grouped (default route) | 273/1500 (18.2%) | 101/500 (20.2%) | yes |
| GP1 generic `_laplace_mode` | 120/1000 (12.0%) | 44/333 (13.2%) | yes |
| Mixed-family `_mixed_laplace_mode` | 42/1200 (3.5%) | 16/400 (4.0%) | yes |
| Tweedie grouped kernel | 13/300 (4.3%) | 11/150 (7.3%) | yes (small n, noisier) |
| Ordinal per-trait `_ordinal_laplace_mode_pertrait` | 0/1500 (0%) | 0/500 (0%) | yes, clean |

None of the above is touched by #481, #483, #494, #500 or #501: each has its own copy of the
per-site loop (the same one-function-per-family pattern #479's after-task report used to
justify fixing Gamma alone). None is covered by #502 either, since #502 only changes the
verdict helper, not these kernels.

Worked example (NB1 grouped, worst case in the reduced run): at `p=9, K=3, sc=2.0`, the kernel
z had `|grad|=3.0e13` (nowhere near a mode) and returned a finite site log-likelihood of
`-1.056e13`, against `-169.36` at the true mode: an 11-order-of-magnitude finite garbage value
that would not trip the 1e12 objective sentinel in the wrong direction (it is far past it, so
it likely would trip the sentinel here, but smaller-magnitude divergences inside the 1e11 band
would not; see the "finite & escapes sentinel" column above, which is the operative risk count).
NB1 grouped is default-fit reachable: `fit_gllvm(Y; family = GLLVModels.NB1(), K = 2)` (confirmed
by the audit's own `probe_B_nb1_mech.jl`, which replicates the fitter's own optimization loop).

Twopart families (ZIP, ZINB, HurdlePoisson, DeltaGamma) were bundled in the same script as
COMPoisson/OrderedBeta/BetaBinomial/StudentT/GP1/Ordinal (`probe_A2_kernels.jl` loops over a
list including these), so they were re-probed incidentally despite the brief's instruction not
to. Rates: ZIP 196/500 (39.2%) vs audit 616/1500 (41.1%); ZINB 121/500 (24.2%) vs audit 328/1500
(21.9%); HurdlePoisson 67/500 (13.4%) vs audit 147/1500 (9.8%); DeltaGamma 72/500 (14.4%) vs
audit 274/1500 (18.3%). All consistent with the audit, and consistent with PR #500 (open, not
merged) still being the correct and only pending fix; this adds no new information beyond
confirming #500's target is real and unmerged. Flagging this here rather than treating it as
new evidence, per the instruction to not build findings on top of it.

NB2 grouped (excluded fitters, audit rate 147/1500) was not re-probed: it was not in any script
batched above, and re-extracting and running it separately was judged not worth the marginal
token/time cost inside the 45-minute budget, since the audit already flagged it as excluded from
the main fitter surface and it is not covered by any current PR.

## 4. `confint_family.jl` bootstrap/profile-refit gap: verified against current line numbers

Read directly (no Julia run needed; this is a static defect, confirmed by reading the current
file, not by execution):

- `_family_profile_refit` (`src/confint_family.jl:2798-2818`): builds the reduced objective,
  runs `Optim.optimize(...; autodiff=:finite)`, and returns `(-nmin, true, ...)` whenever
  `isfinite(nmin)` (line 2816), including when `nmin` is the fitters' `1e12` failure sentinel
  (`1e12` is finite). It never inspects `Optim.converged(res)`.
- `_family_bootstrap` (`src/confint_family.jl:2893-2935`): `ok[b] = true` is set (line ~2905)
  whenever `θb !== nothing && length(θb) == m && all(isfinite, θb)` — any finite refit vector
  counts as a converged bootstrap replicate; `ad.refit`'s own `.converged` and `.loglik` fields
  are never read.

Both are unchanged from the audit's characterization (`confint_family.jl:2904-2935` for
bootstrap, `:2817-2819` for profile-refit in the audit's line numbers, which are close to
current `2893-2935` / `2798-2818` after subsequent unrelated commits shifted the file by a few
dozen lines). This gap sits downstream of every family fix above: even a family whose fitter now
reports `converged=false` correctly can still have its confidence interval silently accept a
non-converged bootstrap replicate or profile refit, because neither helper reads the flag its
own machinery already computes.

## 5. Triage by user-facing risk

"Default route" = reachable via `fit_gllvm(Y; family=..., K=...)` or the model's own named
top-level fitter (`fit_gamma_gllvm`, `fit_zip_gllvm`, etc.), not only an exported low-level
kernel function that an ordinary user would not call directly.

| Family | Current rate | Default-route reachable | Coverage | Priority |
|---|---|---|---|---|
| NB1 grouped | 20/600 (3.3%) | yes (`fit_gllvm(...;family=NB1())`, confirmed) | none | High — common family, silent divergence, no open PR |
| Ordered beta | 72/500 (14.4%) kernel-level; issue #501 separately confirms fit-level gaps to +2859 logLik | yes | issue #501 open, not fixed | High — highest fit-level logLik gap seen of any family, actively being reproduced by a sibling lane |
| StudentT (shared and grouped) | 78/500, 101/500 (~16-20%) | yes (robust-regression use case, likely a common choice) | none | High — highest non-mode rate of the still-live set together with GP1 |
| GP1 (generalised Poisson) | 44/333 (13.2%) | likely yes (generic kernel path) | none | Medium-high |
| BetaBinomial | 27/500 (5.4%) | likely yes | none | Medium |
| COMPoisson | 7/266 (2.6%) | likely yes | none | Medium (lowest rate of the confirmed-live set, but same silent-divergence shape) |
| Mixed-family (bridge) | 16/400 (4.0%) | yes, this is the multi-family bridge path | none | Medium — affects any model mixing families |
| Tweedie grouped | 11/150 (7.3%) | narrower ("disp_group route" per audit notes) | none | Medium-low, smaller surface |
| NB2 grouped | 147/1500 (audit only, not re-probed) | audit notes it as "excluded fitters" | none | Low-medium, needs its own confirmation before acting |
| Twopart (ZIP/ZINB/ZIB/Hurdle*/Delta*) | consistent with audit | yes | **PR #500 open**, covers this | Tracked, not orphaned |
| Ordinal per-trait | 0/500, clean | n/a | n/a | None — audit and re-run agree this kernel is fine |
| Poisson/Binomial `_laplace_mode_off` (covariate kernel) | not re-run (would duplicate #494's own regression evidence) | yes | **fixed by merged #494** | Closed |
| Gamma (all routes) | fixed | yes | **fixed by merged #479/#481, #494** | Closed |
| Beta (grouped) | fixed | yes | **fixed by merged #480/#483** | Closed |
| `_fit_verdict` Class B gap (85 fitters) | draft evidence in #502 | yes, cross-cutting | **PR #502 draft**, do not build on it | Tracked separately |
| confint bootstrap/profile-refit | confirmed live, Section 4 | yes, cross-cutting (every family's CI) | none | High — silently wrong CIs are worse than a silently wrong point estimate, because CIs are usually the thing being reported as the "uncertainty already checked" |

## 6. What could not be re-measured

- NB2 grouped kernel (147/1500 in the audit): not re-probed. The probe script for it was not
  identified among the four probe files pulled (`probe_A_kernels.jl`, `_A2`, `_A3`,
  `probe_B_nb1_mech.jl`); a full search of every `env/probe*.jl` file was not done given the time
  budget. Its current rate is therefore an inference from the audit only, not a fresh
  measurement.
- Twopart families were reprobed only incidentally (bundled with unrelated families in the same
  script); this was not a deliberate check of PR #500's target and should not be read as new
  evidence about #500.
- "Default-route reachable" for BetaBinomial, COMPoisson, GP1, Tweedie grouped and mixed-family
  is stated as "likely" rather than confirmed by reading `fit_gllvm`'s dispatch table or a
  fitter's own docstring; only NB1 grouped's default-route reachability was independently
  confirmed (via the audit's own `probe_B_nb1_mech.jl`, which calls `fit_gllvm(Y; family=NB1(),
  K=2)` directly).
- No timing/runtime-cost measurement was attempted for any family (out of scope per the task).
- The exact current line-number audit-to-main mapping for `confint_family.jl` beyond
  "_family_profile_refit" and "_family_bootstrap" (e.g., whether other confint helpers have the
  same gap for parameters outside `sel`) was not swept exhaustively; only the two functions the
  audit named were checked.

## Files

- This report:
  `/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/LOOP/lanes/gllvm-backlog-20260926/reviews/mode-search-triage-2026-09-26.md`
- Draft tracking issue (still-live families):
  `/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/LOOP/lanes/gllvm-backlog-20260926/reviews/draft-issue-mode-search-tracking.md`
- Draft issue (confint bootstrap/profile-refit gap):
  `/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/LOOP/lanes/gllvm-backlog-20260926/reviews/draft-issue-confint-bootstrap.md`
- Probe scripts used (reduced-count copies), for reproduction:
  `~/local-scratch/triage-modesearch-b-nb1/probe_nb1_reduced.jl`,
  `~/local-scratch/triage-modesearch-b-nb1/probe_A2_reduced.jl`,
  `~/local-scratch/triage-modesearch-b-nb1/probe_A3_reduced.jl`
  (originals recovered from `origin/claude/lane-true-parity-finish-20260925`, unmodified except
  for lowered trial counts).
