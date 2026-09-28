# After-task: NB2 bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:nb-boot-verdict-504:6081`, branch `claude/nb-boot-verdict-504`,
stacked on `claude/bb-boot-verdict-542` (#550) @ `1bee7aa0a`, worktree `.worktrees/nb-boot-verdict-504`.
Started on Shinichi's instruction "start on migrating the next family for #504"; finished while he
was away ("keep going autonomously").

## 1. Goal

Move the NB2 bootstrap refit closures onto the #542 contract (option 3), so a non-converged refit is
excluded and an `r` at the Poisson limit is flagged and can give an `Inf` upper bound.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `NBFit`, `NBGroupedFit` and
  `NBGroupedCovFit` return `(θ, converged, loglik, upper_boundary)`. New `_nb_r_upper_boundary(θ, nr)`
  flags each of the last `nr` entries (`log r`) with `exp(θ[i]) > 1e6`, the same comparison as the
  upper end of `_dispersion_group_boundary`. No change to `_family_bootstrap` (it already reads the
  field, from #550).
- `test/test_confint_bootstrap_verdict_nb.jl` (new, registered after the beta-binomial file).
- `test/fixtures/nb_boot_boundary_504.toml` (new): NB(r = 3) seeds 1 and 2, Poisson seeds 1 and 2,
  drawn on Julia 1.10.12, sha256-guarded.
- `CHANGELOG.md`: entry after the #542 one.

## 3a. Decisions and Rejected Alternatives

- **Why NB2 next.** `NegativeBinomial()` is the default route for the most used count family, and its
  grouped point fits already report `converged = false` at the `r` boundary, so the bootstrap had the
  same gap as beta-binomial.
- **Stacked on #550**, because it needs `_bootstrap_upper_boundary`. The PR's base is the #550 branch.
- **Upper boundary only.** `r < 1e-6` is not flagged; the grouped fitters report it as not converged
  and it is simply excluded. A lower-boundary flag (lower bound 0) would be a new design choice.
- **Same threshold as the point fit** (`> 1e6`, strict), so the bootstrap and the grouped verdict agree.

## 4. Files Touched

- `src/confint_family.jl` (modified)
- `test/test_confint_bootstrap_verdict_nb.jl` (new)
- `test/fixtures/nb_boot_boundary_504.toml` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-nb-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- Probe (Julia 1.10.12, 6 datasets each, p = 6, n = 120, K = 2): on NB(r = 3) data the shared-`r` fit
  recovers r 2.7 to 3.8 (6/6 converged); the per-species grouped fit, the `NegativeBinomial()` default,
  reports `converged = false` on 6/6 with 1 or 2 species' r at 1e9 to 1e17. On Poisson data the
  shared-`r` fit reports `converged = true` with r 7.7e6 to 6.3e10 on 4/6.
- RED, new file on the base (1bee7aa0a): 13 pass, 17 fail, 20 error of 50.
- GREEN: 50/50 on Julia 1.10.12 and 50/50 on Julia 1.13.0.
- Neighbours on 1.10.12, each file alone: `test_confint_hessian_consistency.jl` 12/12,
  `test_grouped_hessian_consistency.jl` 23/23, `test_bridge_grouped_dispersion.jl` 129/129,
  `test_bridge_x.jl` 200/200, `test_bridge_capabilities.jl` 242/242, `test_confint_family.jl` 341/341.
- A live `confint(...; method = :bootstrap, n_boot = 50)` on a per-species grouped fit of nb_r3_seed_1
  was still running when this report was committed (machine load average above 100); its result is
  added to the PR when it finishes.

## 6. Tests of the Tests

- The Poisson-limit testset runs the bootstrap twice on the same data: with the flag every replicate is
  excluded and the `r` upper bound is `Inf`; with a bare-vector wrapper (main's contract) all 10 are
  counted and the bound is finite. It asserts `fp.r > 1e6` first, so it cannot pass vacuously.
- The endpoint-equality testset uses `n_boot = 10` and asserts finite bounds before comparing.

## 7a. Issue Ledger

- #504 stays open (part of). No issue filed for the two point-fit findings below (needs Shinichi).

## 8. Consistency Audit

- No other ref or open PR migrates the NB closures (#518 touches `confint_family.jl` elsewhere).
- No test calls `ad.refit` on an NB fit and expects a vector (grep of `test/`).

## 9. What Did Not Go Smoothly

- zsh read `$B:src` as a history modifier in two shell loops until braced.
- My first fixture-generator invocation read an unset environment variable.

## 10. Known Residuals

- **`fit_nb_gllvm` (shared `r`) has no boundary verdict**: `converged = true` at r = 7.7e6 to 6.3e10 on
  Poisson data (4/6). The bootstrap flag covers the bootstrap; the point fit itself is unchanged here.
  Same class as #515. Not filed.
- **The default NB route rarely converges on genuine NB data** at this size (6/6 not converged). Likely
  the two latent variables absorb the extra-Poisson variance so a species' `r` is not identified; the
  repo already records NB2 boundary cases (#477, NATIVE-06). Its bootstraps will now report `Inf` upper
  bounds for those species' `r`.
- Lower-boundary `r` not flagged (see 3a).

## 11. Team Learning

Before migrating a family's bootstrap, probe its point fits on data from the boundary model (Poisson
for NB, Binomial for beta-binomial): both families had a route that reported `converged = true` there.

## 12. Cross-Product Coverage

Covered: all three NB2 fit types; Julia 1.10.12 and 1.13.0; aarch64 macOS. Not covered: Linux and
Windows (CI), non-log links, masked data, `hessian = :fisher` refits, `parallel = true`.
