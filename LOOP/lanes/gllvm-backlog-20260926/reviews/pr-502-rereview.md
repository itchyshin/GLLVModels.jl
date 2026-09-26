# PR #502 re-review: reworked draft, per-family verdict (maintainer decision (d))

PR: itchyshin/GLLVModels.jl #502, branch `claude/fit-verdict-gradient-485`, head
`be18b9236` (reworked; previous review at `pr-502-correctness.md` covered the earlier,
reverted head `802f3965e`, which widened the shared `_fit_verdict` and broke five
suites — the finding that drove the maintainer's per-family-verdict decision). PR is
still marked **draft**.

Reviewer lane: read-only on GitHub, no pushes/edits to the PR branch. Throwaway
worktrees `~/local-scratch/verify-502` (detached at `be18b9236`) and
`~/local-scratch/verify-502-main` (detached at `origin/main`, `b90641c97`), both
removed after this review. Toolchain: Julia 1.10.0/1.10.12 (juliaup `+1.10`),
`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, Mac Studio (macOS ARM64).

## Verdict: MERGE_AFTER_FIXES

The rework does what the maintainer decision asked: it reverts the shared helper and
adds a narrowly-scoped, precedent-matching verdict to the one fitter that had the
bug. All five previously-broken `@test fit.converged` assertions pass again, and the
new regression test correctly reproduces #485 against `origin/main`. One real risk
remains unresolved: the new test's seed-101 case hard-asserts a specific outcome that
prior, directly-applicable evidence says is platform-fragile (section 5) — plus CI has
not finished, and CHANGELOG/rebase housekeeping is outstanding.

## 1. `src/fit_verdict.jl` — byte-identical to `origin/main`

```
git diff origin/main -- src/fit_verdict.jl   # empty
```

Confirmed: no output. The shared, ~90-fitter-wide helper is exactly `origin/main`'s
version — the blanket change from the previous head is fully reverted, including the
incorrect "callers all leave x/f tolerances at 0" comment the prior review flagged
(reverting to `origin/main` restores the version that never made that claim).

## 2. `_nb1_grouped_g_met` matches `_beta_grouped_g_met`; wired only into `fit_nb1_gllvm_grouped`

`git diff --stat origin/main...be18b9236`:

```
 docs/dev-log/.../2026-09-26-fit-verdict-gradient-485.md          | 371 ++++++++
 docs/dev-log/.../2026-09-26-nb1-grouped-verdict-rework-502b.md    | 314 ++++++
 src/families/grouped_dispersion.jl                                |  12 +
 test/runtests.jl                                                  |   1 +
 test/test_fit_verdict_gradient.jl                                 | 144 +++++
 5 files changed, 842 insertions(+)
```

Only `src/families/grouped_dispersion.jl` changed in `src/` — nothing else in `src/`
is touched. The added lines:

```julia
_nb1_grouped_g_met(res, g_tol) = (gres = Optim.g_residual(res);
    isfinite(gres) && gres <= max(g_tol, g_tol * abs(Optim.minimum(res))))
```
placed right before `fit_nb1_gllvm_grouped`, and inside that function:
```julia
loglik, conv, iters = _fit_verdict(res)
conv = conv && _nb1_grouped_g_met(res, g_tol)   # #485: a zero-length step is not convergence
```

This is character-for-character `_beta_grouped_g_met` (`grouped_dispersion.jl:762-763`),
and the wiring (`conv = conv && helper(res, g_tol)` immediately after `_fit_verdict`)
exactly mirrors `fit_beta_gllvm_grouped`'s own `conv = conv && _beta_grouped_g_met(res, g_tol)`
(line 849). `g_tol` is a proper keyword argument of `fit_nb1_gllvm_grouped` (default
`1e-5`), not a free/global variable — the scoping is correct. `fit_nb1_gllvm`
(scalar/shared-dispersion) and `fit_nb1_gllvm_grouped_cov` (covariate route) are
untouched, confirmed by `grep -rn "_nb1_grouped_g_met"` finding only the one
definition and one call site (plus the two references in the new test).

## 3. The five previously-broken test files, run individually on the branch

| file | result |
|---|---|
| `test_twolevel.jl` | **75/75 pass** (includes line 107) |
| `test_phylo_poisson_xlv.jl` | **9/9 + 27/27 pass** (includes line 158) |
| `test_phylo_beta_xlv.jl` | **13/13 + 25/25 pass** (includes line 183) |
| `test_phylo_binomial_xlv.jl` | 1st testset (incl. line 182 `@test fit.converged`) **passes in full**; 2nd testset 16/22 pass, 6 fail at lines 194-198, 203 (`prof.pd_hessian` / profile-eta-realized assertions) |
| `test_phylo_gamma_xlv.jl` | 1st testset 11/12 pass, 1 fail at line 123 (`isapprox(ll_sparse_aug, ll_dense_leaf; atol=1e-6)`, evaluated `-74.177` vs `-74.259`); 2nd testset (incl. line 172 `@test fit.converged`) run standalone: 19/25 pass, 6 fail at lines 185-189, 195 (again `prof.pd_hessian`/profile assertions) |

All five originally-cited `@test fit.converged` assertions (twolevel:107,
poisson:158, beta:183, binomial:182, gamma:172) **pass on this branch.** The
remaining failures in the binomial and gamma files are a different family of
assertion (a `prof.pd_hessian`/profile-CI check downstream of the fit, and a
sparse-vs-dense-leaf loglik cross-check) at line numbers that match the *previous*
review's own "fail under both verdicts (not the PR)" list (`phylo_binomial_xlv:194-203`,
`phylo_gamma_xlv:123, 185-195`) — i.e. these are pre-existing, unrelated failures that
reproduce identically regardless of the verdict rule, not a regression from this PR.
(`test_phylo_gamma_xlv.jl`'s two `@testset`s are not nested under one enclosing
testset, so a failure in the first one throws and aborts the `include` before the
second runs — the second was re-run standalone, header + `@testset "...canary"...end`
copied verbatim, to get independent evidence on line 172.)

## 4. `test/test_fit_verdict_gradient.jl` on the branch, and reproduction on `origin/main`

Branch: **9/9 pass** (Julia 1.10.12, macOS ARM64), ~4 minutes.

Independent reproduction on `origin/main` (unfixed `fit_nb1_gllvm_grouped`), using the
test's own `_nb1_verdict_fixture`/`_nb1_verdict_max_grad` helpers copied verbatim into
a standalone probe script, seeds 101-110:

| seed | converged (main) | max FD gradient | flip? |
|---|---|---|---|
| 101 | true | 3.85 | **yes** |
| 102 | true | 120.1 | **yes** |
| 103 | true | 12.37 | **yes** |
| 104 | true | 16.70 | **yes** |
| 105 | true | 6.67e-6 | no (genuinely converged) |
| 106 | true | 6.87e-6 | no |
| 107 | true | 7.16e-6 | no |
| 108 | true | 0.495 | **yes** |
| 109 | true | 7.67e17 | **yes** (FD stencil hit the failure-value plateau) |
| 110 | false | 3.22 | no (already correctly unconverged on main) |

**6 of 10 flip** — matches the builder's claim exactly (`docs/dev-log/.../2026-09-26-fit-verdict-gradient-485.md`),
including seed 101 specifically, on this platform.

## 5. Platform robustness of the new test — BLOCKING

The new test's case (a) hard-asserts, for a fixed seed:
```julia
Y = _nb1_verdict_fixture(101)
fit = GLLVModels.fit_nb1_gllvm_grouped(Y; K = 2, group = collect(1:5))
...
@test !fit.converged
```
`_nb1_verdict_fixture` is **byte-identical** to the fixture in the *previous*,
already-reviewed head's test file (`git show a6e68ab19:test/test_fit_verdict_gradient.jl`
— same function body, same defaults, same seed 101 call). That earlier review
(`pr-502-correctness.md`, §"The PR's own new regression test fails on CI") directly
measured, for this exact fixture/seed/fitter, that **on Linux Julia 1.13.0 the fit
converges** (`!fit.converged` fails there), even though it does not on macOS/other
combinations tested. `.github/workflows/CI.yml:67` runs `version: ['1.10', '1']` on
`ubuntu-latest`, and `'1'` resolves to 1.13.0 per this machine's `juliaup list` — the
same combination.

This is not an artifact of the old, reverted blanket rule. The new code applies the
identical formula (`gres <= max(g_tol, g_tol*|nll|)`) to the identical `Optim`
result from the identical fitter call — the *only* thing that changed is which
function owns the AND (`_fit_verdict` before, `_nb1_grouped_g_met` now). The
Optim LBFGS finite-difference path itself — which BLAS build, which zero-length step
it stalls on for seed 101 — is unaffected by that refactor. So the previously-measured
Linux/1.13.0 divergence is expected to reproduce on this branch's own CI (the
"Julia 1 — ubuntu-latest shard 3/4" job; test_fit_verdict_gradient.jl is the 119th
`_shard_include` call, `118 % 4 == 2` → shard 3/4 of 4). The rework's own after-task
report (`2026-09-26-nb1-grouped-verdict-rework-502b.md:257-259`) already flags this
gap: *"Only one platform/BLAS combination verified here... the review's own evidence
that Linux takes a different optimiser path on the same seed is taken on trust, not
re-verified."* I could not close that gap either — CI was still queued for this
entire review (§6).

**Judgment: treat as BLOCKING.** Fix: turn case (a) into a relation, the same way
case (b) already is, rather than a fixed per-seed truth value. Two ways to do it,
either is sufficient:
- Existence form: replace "seed 101 must be `!converged`" with "at least one seed in
  101:110 is `!converged` with `_nb1_verdict_max_grad(...) > 1e-3`" (a disjunction over
  the sweep, not a single seed).
- Relation form matching (b)'s own direction: assert the converse relation directly —
  "whenever `_nb1_verdict_max_grad(...) > 1e-3` (genuinely non-stationary, independently
  computed), `!fit.converged`" — for every seed in the sweep, not just checking the
  implication when `fit.converged` is true (as (b) does) but also the reverse when it
  is provably not stationary. This preserves the "reproduces the real #485 defect"
  intent without hard-coding which seed stalls on a given platform.

Cases (b) (the converged⟹small-gradient relation over 10 seeds) and (c) (seed 703
stationary-stays-converged) are already correctly platform-robust as written — no
change needed there.

## 6. CI on head `be18b9236`

Checked `gh pr checks 502` repeatedly across this review (multiple checks spanning
>20 minutes from the PR's CI trigger); state was unchanged throughout:

| check | status |
|---|---|
| Documenter | **pass** (3m16s) |
| Julia 1 / Julia 1.10 — ubuntu-latest, shards 1-4 (8 jobs) | **pending** (queued, not started) |
| Frozen R 0.7.0 family smoke (advisory) | **pending** |

No shard has reported a result (pass or fail) beyond Documenter. Per the task's
10-minute cap, I stopped waiting and am reporting this as-is: **CI is not yet
resolved**, and in particular the shard that would directly confirm or refute §5's
platform-fragility concern (Julia 1 / ubuntu-latest shard 3/4) has not run. This must
be checked (specifically that shard, that job) before merge.

## 7. CHANGELOG — not a code defect, but a required follow-up

`git diff origin/main -- CHANGELOG.md` shows what looks like a 5-line *removal*, but
it is a rebase artifact, not an edit: the branch is **13 commits behind
`origin/main`** (`git log --oneline $(git merge-base origin/main be18b9236)..origin/main`),
including `3f099e2fd fix: fail closed when GLLVM_PARITY_R_LIBS lacks gllvmTMB (parity
tools)`, which added the CHANGELOG entry the diff appears to remove. This branch
simply predates that commit. Two follow-ups, neither a defect in this PR's own logic:
- Rebase onto current `origin/main` (will need manual resolution of the CHANGELOG.md
  conflict — the parity-tools entry and this PR's own entry both belong in the final
  file).
- Add a CHANGELOG entry for this fix itself (`fit_nb1_gllvm_grouped`'s gradient
  criterion) — none exists yet on this branch, consistent with the lease-contention
  note in the task.

## Blocking

1. `test_fit_verdict_gradient.jl`'s seed-101 case (a) hard-asserts `!fit.converged`
   for a fixture already shown, on the identical seed/fitter, to converge on Linux
   Julia 1.13.0 in the previous review. Convert it to a relation (§5) before merge.
2. CI has not finished (all 8 Julia shards + the advisory R leg still queued after
   >20 minutes); confirm the branch is green — especially shard 3/4, which carries
   the new test — once it reports.

## Should fix (non-blocking)

- Rebase onto current `origin/main` (13 commits behind) and resolve the CHANGELOG.md
  conflict.
- Add the fix's own CHANGELOG entry.
- PR is still `draft` — undraft once the above are resolved.

## What's solid

- `src/fit_verdict.jl` fully reverted, byte-identical to `origin/main` — confirmed by
  empty diff, not by reading a claim.
- The new helper is a verbatim copy of the already-shipped `_beta_grouped_g_met`
  pattern, wired only into the one fitter named in the task scope.
- All five previously-broken `@test fit.converged` assertions pass; the failures that
  remain in those same files are pre-existing and unrelated (independently confirmed
  against the prior review's own classification, not just cited from it).
- The bug itself reproduces cleanly against unfixed `origin/main` (6/10 seeds flip,
  matching the builder's claim including the specific seed used in the regression
  test), and the fix's own test suite is green locally (9/9).
