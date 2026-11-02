# After Task: Data-relative floors and SQUAREM `hit_floor`

## Goal

Unblock draft PR #868 after the review of `4fa4848c9`: make the #754 / #753
boundary checks relative to the data scale (including the EM clamp), carry
`hit_floor` through SQUAREM's main path, and file a `fit_relaxed_clock`
follow-up (refs #753). Stay on the same branch; keep the PR a draft.

## Implemented

`_BRANCH_RE_VARIANCE_BOUNDARY_TOL = 1e-8` is now a fraction of `var(y)`.
Phylo EM's `_sigma_eps2_data_floor` is `1e-8 * sum(y.^2)/(n*p)` (same
fraction), and `_mstep_dense` clamps to `max(floor, eps())`. Both EM and
SQUAREM latch `sigma_eps_boundary` so a later clamp-up to `eps()` on tiny
`y` cannot clear the flag. SQUAREM's increment stop sets
`converged = !sigma_eps_boundary`; nothing hard-codes `true`.
`fit_relaxed_clock` is unchanged; issue create returned 403.

## Mathematical Contract

Unchanged likelihoods. The boundary test is now a stated fraction of the
response scale (`var(y)` or `mean(y.^2)`), so multiplying `y` by a positive
constant cannot flip `converged` or the boundary field.

## Files Changed

- `src/phylo_branch_re.jl` — compare `σ²` / `σ²_eps` to `1e-8 * var(y)`
- `src/em_phylo.jl` — `1e-8 * mean(y.^2)` floor and clamp; latch the flag
- `src/em_squarem.jl` — return and latch `hit_floor`; veto `converged`
- `test/test_branch_re_convergence.jl` — `y` / `y*1e-5` / `y*1e5` flags
- `test/test_em_phylo_sigma_eps_floor.jl` — same rescales + SQUAREM main path
- `test/test_em_squarem.jl`, `bench/em_squarem_bench.jl` — unpack `(θ′, hit)`
- `CHANGELOG.md`, `docs/src/changelog.md`, `AGENTS.md`
- `docs/dev-log/owed/2026-10-11-relaxed-clock-false-converged-issue.md`

## Tests Added

Scale-invariance and SQUAREM-main-path assertions in the two existing
files (boundary clause; would have failed on `4fa4848c9` for tiny / huge
`y` and for SQUAREM without fallback).

## Benchmark Numbers

N/A — no hot-path change. The M-step gained a scale multiply; SQUAREM
gained a boolean latch.

## R-Parity Verdict

Parity: N/A — change does not touch the parity surface. Flags only.

## JET / Allocs / Aqua Verdicts

- JET: not rerun (no `Pkg.test()` merge env in this session)
- Allocs: N/A — no inner-loop rewrite
- Aqua: not rerun locally; CI shard 1 includes `test_quality.jl`

## Checks Run

Julia 1.10.12, `--project=.`, this VM, one process:

```
test/test_branch_re_convergence.jl          16 pass / 0 fail
test/test_em_phylo_sigma_eps_floor.jl       28 pass / 0 fail
test/test_em_squarem.jl                    27 pass / 0 fail
test/test_zinb_cov_iterations.jl            10 pass / 0 fail
test/test_em_phylo.jl                      27 pass / 0 fail
test/test_nparams_em_branch_re.jl            8 pass / 0 fail
test/test_phylo_branch_re.jl               44 pass / 0 fail
```

CI on the review-fix head was still pending when this note was written.

## Consistency Audit

`rg "eps\\(\\) \\* mean\\(y"` across `src/` and the PR changelog entries:
no remaining `eps() * mean(y.^2)` floor claim. `docs/src/changelog.md` and
`CHANGELOG.md` now say `1e-8 * mean(y.^2)`. Historical after-task
`2026-10-10-false-converged-754-753-781.md` still describes the first
slice and is left as written.

## GitHub Issue Maintenance

`issue_write` for `fit_relaxed_clock` returned 403. Full issue body
(with file:line, minimal repro, refs #753, no closing keywords) is in
`docs/dev-log/owed/2026-10-11-relaxed-clock-false-converged-issue.md`
and `/opt/cursor/artifacts/2026-10-11-relaxed-clock-false-converged-issue.md`.
PR #868 body uses only `refs #N`.

## What Did Not Go Smoothly

A 64-ulp floor (`64 * eps() * scale`) still missed `y*1e-5` on the #753
rank-1 path: residual / scale sat at `9.5e-9`, above ~1e-14. The 1e-8
fraction that branch-RE already used catches that stall. After the
relative clamp, rank-1 EM often runs to `max_iter` with the latched
boundary flag rather than stopping in a handful of iterations; the
user-visible contract is the flags, not the iteration count. Graft is
still not installed in this image.

## Team Learning

A variance floor stated in absolute units will flip under a change of
`y` units. Write the test that rescales `y` before calling the floor done.

## Remaining Risks

- `fit_relaxed_clock` still sets `converged = true` from an objective
  increment and has a `τ²` floor plus an absolute `1e-8` `σ²_eps` clamp
  (follow-up issue not filed; body in the owed artifact).
- #752 xlv constrained refit still treats constraint feasibility as
  convergence; next PR.
- Rank-1 EM at the new floor can hit `max_iter` with `converged = false`
  and `sigma_eps_boundary = true`; that is a boundary, not an interior stop.
- CI on this head was pending at write time.

## Known Limitations

The stored clamp on tiny `y` is `max(1e-8 * mean(y.^2), eps())`, so the
stored `σ_eps²` can sit above the relative floor when `y` is tiny. The
latched flag is then the record of the clamp. `fit_relaxed_clock` is not
fixed here.

## Next Command

Do not merge from the agent. Maintainer pastes the owed issue if wanted,
reviews #868, then #752.

## Rose Verdict

Rose verdict: PASS WITH NOTES — `fit_relaxed_clock` issue was 403 so the
body is an artifact; CI was pending when this note was written; #752
remains the next sibling PR.
