# After-task: direct outer optimizer selection

## 1. Goal

Allow users to choose LBFGS or BFGS directly on qualified ordinary GLLVModels fit routes, retain the existing `optimizer=nothing` behavior, and make every outer attempt inspectable on the returned fit.

## 2. Implemented

Added `optimizer = nothing`, `:lbfgs`, `:bfgs`, and configured Optim method support to the exact Gaussian, Poisson Laplace, shared-r NB2 Laplace, Gaussian source, and default per-trait NB2 Laplace pipelines. Added `OptimizerStage`, `OptimizerRecord`, and `optimizer_info(fit)`. Receipts retain requested and actual methods, method objects, derivative route, native `Optim.Options`, execution controls, acceptance, and each raw outer result. Explicit Poisson/NB2 choices bypass default-only BFGS continuation. Per-trait NB2 explicit LBFGS skips the legacy BFGS ridge polish; explicit BFGS uses BFGS for each executed stage. Formula and `select_lv` forward supported requests. Unsupported high-level routes and interval refits that cannot forward an explicit choice fail with `ArgumentError`. Old positional fit constructors have compatibility methods.

The pre-implementation test run failed on unsupported optimizer keywords and missing receipts as expected. The implementation focused suite passed 61/61 plus 1/1 spy-cleanup check and printed `OPTIMIZER_SELECTION_PASS`. A follow-up adds selected-method propagation and retained receipts for Gaussian mask/offset refits; the focused suite was rerun afterward. Seven of nine named neighbor files completed in the final logged batch before the 5-minute estimate was exceeded at `test_formula.jl`; the missing two are explicitly incomplete. The targeted Gaussian source-design regression passed 34/34. The paired fixed-start Poisson README/tutorial example ran both LBFGS and BFGS successfully with the same inputs and log-likelihoods -52.81656880323053 and -52.81656880323046. The final local docs attempt rendered VitePress pages, but the Julia wrapper did not emit a final exit receipt, so docs validation remains unconfirmed. The broad suite was stopped with SIGINT at `test/test_phylo_beta_xlv.jl:158` after exceeding the 5–10 minute estimate. It had reached later truncated-NB2 cases and printed no final aggregate denominator, so it is incomplete evidence. `Pkg.test()` was not run. The parent requested stopping the run after the estimate overrun; it was not restarted.

## 3a. Decisions and Rejected Alternatives

- Shorthand symbols resolve centrally to native `Optim.LBFGS()` / `Optim.BFGS()` objects, so users need not import Optim just to choose a method. Configured Optim objects retain their settings in stage receipts.
- The existing three-step per-trait NB2 default remains the default. Explicit LBFGS skips only the legacy BFGS ridge polish; explicit BFGS uses the chosen method in that stage. This preserves the fitted NB2 model and avoids silently running a different optimizer.
- Profile and bootstrap intervals refuse explicit-selected fits because their internal refits do not yet propagate the chosen solver. Forwarding a method into those nested refits would need a separate inference-path audit.
- Assumptions made without asking: `optimizer_info(fit).actual` reports the Optim algorithm family (`Optim.LBFGS` or `Optim.BFGS`), while configured algorithm details live in each stage's method object. If callers require the full parameterized Julia type as `actual`, the field could be made more specific without changing the fitted result.

## 4. Files Touched

- `src/optimizer_selection.jl`
- `src/GLLVModels.jl`
- `src/fit.jl`
- `src/gaussian_intercept.jl`
- `src/source_fit.jl`
- `src/families/aghq_gaussian_fit.jl`
- `src/families/aghq_poisson_fit.jl`
- `src/families/aghq_nb2.jl`
- `src/families/poisson.jl`
- `src/families/negbin.jl`
- `src/families/grouped_dispersion.jl`
- `src/families/fit_gllvm.jl`
- `src/formula.jl`
- `src/model_selection.jl`
- `src/laplace_grad.jl`
- `src/confint.jl`
- `src/confint_family.jl`
- `test/runtests.jl`
- `test/test_optimizer_selection.jl`
- `README.md`
- `CHANGELOG.md`
- `docs/src/api.md`
- `docs/src/changelog.md`
- `docs/src/tutorial.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/after-task/2026-10-18-optimizer-selection.md`

## 5. Checks Run

- Focused test command: 61/61 plus spy cleanup 1/1; printed `OPTIMIZER_SELECTION_PASS`. Retained logs: `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-optimizer-review-red.log` and `.../gllvmodels-optimizer-review-green.log`.
- Final neighbor batch: seven of nine files reached exit 0 before the 5-minute estimate at `test_formula.jl`; `formula_sources` was not run. Full retained per-file evidence is `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-neighbours-final.log`. A separate targeted Gaussian source-design regression passed 34/34 at `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-gaussian-source-final.log`.
- The paired fixed-start Poisson example passed at `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-paired-example-final.log`. Docs VitePress rendering completed but Julia wrapper exit is unconfirmed; see `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-docs-final.log`.
- Local docs command `julia --project=docs docs/make.jl --local`: completed. Documenter doctests, reference checks, and VitePress rendering passed. VitePress reported an advisory large chunk; deployment was skipped for a local build.
- `git diff --check` passed at the pre-final-integrity check.
- Full suite `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=2 julia --project=test test/runtests.jl`: interrupted at `test/test_phylo_beta_xlv.jl:158`; no final aggregate counts.
- `Pkg.test()`: not run.
- PR CI and independent parent review: pending.

## 6. Tests of the Tests

Before implementation, the optimizer-selection test file failed because the public fitters rejected the new keyword and fit objects had no optimizer receipt. After implementation, the same cases passed. Tests assert direct method objects on all recorded stages, detect the default Poisson BFGS continuation, verify explicit unconverged LBFGS does not continue, and check selected method forwarding through `fit_gllvm`, `gllvm`, and `select_lv`. They also exercise unsupported routes and the profile-refit guard, and reconstruct old positional forms of each changed fit type.

## 7a. Issue Ledger

- Fixed: selected optimizers initially recorded parameterized Optim types where the user-facing method family was expected. `OptimizerRecord.actual` now reports the algorithm family; the configured object remains in every stage.
- Fixed: an initial test run used result variables across nested testsets and a formula fixture with the wrong site count; fixtures now match Julia testset scope and the formula's site axis.
- Deferred: full package and quality-tool test outcomes. The broad core run was stopped after its estimated runtime was exceeded; no aggregate pass/fail/broken count was emitted. `Pkg.test()` remains unrun.

## 8. Consistency Audit

Checked direct Gaussian and Gaussian-source solves, the Poisson conditional continuation, shared NB2, per-trait NB2 initial/boundary/ridge stages, Gaussian sign-flip refinements, formula routing, automatic-rank selection, and interval refit paths. Existing neighboring regression files passed. The default per-trait NB2 route keeps its species-specific dispersion; documentation warns that optimizer choice does not alter that estimand. No likelihood, parameterization, or formula grammar changed.

## 9. What Did Not Go Smoothly

Julia's compilation cache required escalated file access. The docs environment had no manifest and could not resolve this unregistered package until it was temporarily developed from the local checkout; the local build then completed. The hub `closeout.py` helper targets the Shinichi brain repository regardless of the current package checkout, so it could not generate the report here; the package-local report follows the required section structure manually. Temporary docs manifests must be removed before delivery.

## 10. Known Residuals

Explicit optimizer selection does not cover AGHQ/adaptive, variational, other grouped/covariance, mixed, phylogenetic, row-effect, pervar, or other response-family routes. Profile and bootstrap intervals refuse explicitly selected fits until their nested refits forward the solver. The full core suite is incomplete (interrupted at `test/test_phylo_beta_xlv.jl:158`, no final aggregate counts); `Pkg.test()`, independent parent review, and PR CI remain unverified.

## 11. Team Learning

Memory receipt: the package-local `AGENTS.md`, parent-provided cross-package goal/design notes, and the shared lane ledger shaped the work. `route.py` returned no LOAD-FIRST manifest for the isolated checkout or registered GLLVM path. Golden Set: not run; no existing Golden Set category was directly in scope. The acceptance gate records the exact route matrix and commands.

Rose audit: the final logged neighbor batch is incomplete at seven of nine files; earlier terminal-only checks remain distinct from this receipt. Independent parent review and PR CI remain pending. No merge-readiness claim is made.

## 12. Cross-Product Coverage

Covers: Gaussian exact marginal, mask/offset observed-data refit, and sign-flip stages; Gaussian source outer solve; Poisson Laplace; shared-r NB2 Laplace; the default per-trait NB2 formula route and its initial, boundary-restart, and ridge-polish stages; matrix/formula/model-selection forwarding; immutable receipts; legacy constructor compatibility; clear unsupported-route/refit errors.

Does NOT cover: other count or continuous families; AGHQ/adaptive or variational solvers; non-NB2 grouped/covariance fits; grouped/mixed/phylogenetic/row-effect/pervar models; bootstrap/profile refit forwarding; full package or platform/CI evidence; R parity or release readiness.
