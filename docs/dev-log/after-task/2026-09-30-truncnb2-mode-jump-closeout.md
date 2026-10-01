# After-task: truncated-NB2 mode jump, r intervals and follow-ups (2026-09-28 to 2026-09-30)

## 1. Goal

Fix the truncated-NB2 Laplace objective jump (a 1e-5 step in log r moved the objective by 5.5e-4 one way and 1.5e-8 the other) and check whether the broken r intervals from the #581 review (Wald [0.131, 0.136], profile `:failed`) become sane. Then land the follow-ups the maintainer asked for along the way.

## 2. Implemented

Seven PRs, all merged to `main`:

| PR | Merge commit | What it does |
|---|---|---|
| #601 | `e794b4cb2` | `_grouped_laplace_mode` (grouped_dispersion.jl): in the backtracking branch (NB1, truncated NB2, truncated and censored Poisson), a trial step is halved while the gradient along it has turned past -1/2 of its starting value, on small and large steps. Stops the Fisher-scoring 2-cycle. A full step that passes is bit-identical to the old update. |
| #613 | `d337a2302` | Truncated-NB2 frozen-R parity cell: `r_gradient_max <= 1e-4` becomes a printed "recorded, not a gate" value (the #608 rule). |
| #621 | `be25621c5` | Both truncated-NB2 fitters: any r below 1e-6 gives `converged = false` with a warning; r above 1e6 only warns (Poisson limit). Helper `_truncnb2_dispersion_verdict`. |
| #627 | `0f31f6baf` | Per-trait truncated-NB2 fitter calls `_nb_boundary_restart` (#477): a trait stalled at the boundary is restarted from r = 1, kept only if better. |
| #605 | `efdc8be31` | `_family_profile` (confint_family.jl): a log-scale parameter whose profile deviance stays below the cutoff down to 1e-6 x the estimate gets `lower = 0` instead of NaN. |
| #607 | `7e25cbb51` | Same file: that floor check runs before the lower bracket search, so the search is skipped when the end is open (draw 104: 583 s to 20.6 s, same bounds). |
| #634 | `ad733dc28` | Not mine (another lane); reviewed only. `family_formula_cases.jl` records R's gradient instead of gating it, for every family. |

Also posted the r-interval results on #581 and #601 (comment on the #581 fixture draw 104, before and after #601).

## 3a. Decisions and Rejected Alternatives

- **Fix in `_grouped_laplace_mode`, not in `laplace.jl`.** `_laplace_mode` is shared and off-limits without the maintainer. The guard is limited to the backtracking families, so NB2, Beta and Gamma grouped `getLV` stayed bit-identical (no collision with #529).
- **Gradient-based overshoot test, not an observed-curvature Newton fallback.** The truncated-NB2 observed weight can be negative; the gradient test is family-agnostic.
- **Guard both small and large steps.** The first version guarded small steps only; the #605 profile then hit a large-step 2-cycle (site 98, error ratio -0.96) and returned NaN. Extending the guard fixed it.
- **Upper-end dispersion rule (maintainer's decision, reversed once on evidence).** Flagging r > 1e6 as not converged was chosen, then reverted to warn-only after gllvmTMB 0.7.1 was checked on five draws: gllvmTMB's own best fits put one trait's phi above 1e6 on 4 of 5 seeds, so flagging would mark about 80% of ordinary per-trait fits "not converged" in both engines. The other NB routes (#622, #630) now follow the same warn-only rule; NB1 flags the opposite end because its Poisson limit is phi -> 0.
- **Direct test for #621, not a data trigger.** One count of 10^13 drove the shared-r fit to r = 3.1e-46 on macOS but to r = 0.98 on a Linux runner. The test now calls the verdict helper directly and wires each fitter with r started at the boundary and zero iterations.
- **#605 search-first vs #607 floor-first.** Both kept: #605 is the conservative behaviour, #607 the fast one. #607's risk (a non-monotone profile that dips below the cutoff near the floor) is stated in its PR body; none was seen.
- **No second PR for the formula-case gradient.** #634 already did it, for every family.

## 4. Files Touched

- `src/families/grouped_dispersion.jl` (#601)
- `src/families/truncated_nbinom2.jl` (#621, #627)
- `src/confint_family.jl` (#605, #607)
- `test/test_truncnb2_mode_search.jl`, `test/fixtures/truncnb2_mode_search_oscillation.toml` (#601)
- `test/test_truncnb2_dispersion_boundary.jl`, `test/fixtures/truncnb2_dispersion_boundary.toml` (#621; wiring test adjusted in #627)
- `test/test_truncnb2_pertrait_boundary_restart.jl`, `test/fixtures/truncnb2_pertrait_boundary_restart.toml` (#627)
- `test/test_family_profile_open_lower.jl` (#605, #607)
- `test/parity/test_truncated_nbinom2_parity.jl` (#613)
- `test/runtests.jl`, `CHANGELOG.md` (#601, #605, #607, #621, #627)
- `docs/dev-log/after-task/2026-09-30-truncnb2-mode-jump-closeout.md` (this report)
- `docs/dev-log/handover/2026-09-30-claude-handover-truncnb2-mode-jump.md` (handover)

## 5. Checks Run

- Red on origin/main, green with the fix, on Julia 1.10.12 and 1.13.0, for every new test file: #601 (6 of 14 checks red), #605 (2 of 7), #607 (1 of 9), #621 (5 of 9 in the data-trigger version; the direct version cannot run on main because the helper does not exist there), #627 (2 of 2).
- Affected-file sweeps, both Julia versions, 0 failures each: #601 20 files; #605 and #607 13 files (797/799 pass); #621 12 files (729 pass); #627 13 files (742 pass). The only non-passes were existing `@test_skip` / `@test_broken` markers.
- After each rebase: the new test plus the core neighbours on the rebased code (for example #605 349/349, #607 351/351, #627 49/49 then 19/19).
- CI at merge (verified per head SHA, merged with `--match-head-commit`, never `--auto`, since auto-merge is off on this repo): 8/8 test shards, P1 twin tests and Documenter green on #605, #607, #621, #627. The frozen-R job is advisory; its reds were Student-t (#610), NB2 (#608/#631), and the runner-dependent R gradient (#613, #634).
- `main`'s own CI and Documenter on the final merge (`7e25cbb51`, #607): both completed green.
- gllvmTMB 0.7.1 (R 4.6.0) run locally on five #585 per-trait draws to check the Poisson-limit question (table in #627's PR body).

## 6. Tests of the Tests

Every new test was run against origin/main first and failed there for the stated reason: the mode off by 6.9e-6 and 1.3e-3 (#601); `lower = NaN, :partial` (#605); 12 lower-side probes the floor-first order must skip (#607); `converged = true` at r = 3.1e-46 (#621, first version); log-likelihood 0.35 and 2.0 below gllvmTMB (#627). The reference values come from code that shares nothing with the kernels: a closed-form truncated-NB2 score with bisection (#601), synthetic adapters with analytic bounds (#605/#607), and gllvmTMB's own fits (#627).

## 7a. Issue Ledger

- Fixed: truncated-NB2 mode 2-cycle and objective jump (#601); large-step 2-cycle found through the #605 profile (#601); open lower profile end reported as NaN (#605); slow open-end search (#607); degenerate r reported converged (#621); per-trait fit stuck at a worse optimum with the wrong trait at the Poisson limit (#627); runner-dependent R-gradient gate in the parity cell (#613).
- Diagnosed, fixed by other lanes: `parity_nb2_smoke_Y` undefined in the NB2 formula case, from #608 (fixed in #631); the R-gradient gate in `family_formula_cases.jl` (#634).
- Deferred: truncated-NB2 `_family_ci` adapters still pass `boundary = false` for r; they lack the `dispersion_boundary` field other NB routes have. Not started.

## 8. Consistency Audit

- Callers of `_grouped_laplace_mode` (`_grouped_getLV`, the truncated-NB2 per-trait site) and every family reaching the backtracking branch were swept before and after #601.
- The two truncated-NB2 fitters and the formula routes (`fit_gllvm`, wide and long `gllvm`) were checked together for #621 and #627.
- Other `r_gradient_max` gates in `test/` were listed during the #634 review; none remain in these family cells.
- The combination of #621 and #627 was retested after the rebase, and exposed one interaction (the restart rescues a fit started at r = 1e-9), fixed in #627's test.

## 9. What Did Not Go Smoothly

- The first CI sweep passed all file names as one argument (zsh does not split an unquoted `$FILES`); fixed with `${=FILES}`.
- Draw 104 was first regenerated from `MersenneTwister(104)` instead of read from #581's fixture, so the first r-interval run tested the wrong data.
- #621's first test depended on where a fit ended; a Linux runner reached a different optimum. Replaced by a direct test.
- `pgrep -f` watchers matched their own command lines and never exited; two waiting shells had to be killed by PID.
- A zsh `$VAR:r` modifier garbled a push refspec in the #607 chain script. The push failed and the gate refused to merge, so nothing wrong landed; #607 was pushed by hand.
- GitHub did not start CI on a conflicting PR (#621, #627 after `main` moved), and once started #605's CI about 9 hours after the push for an unknown reason.
- A relayed "both ends" decision reversed a decision the maintainer had given directly; it was confirmed with him before any change, then reversed again on the gllvmTMB evidence.

## 10. Known Residuals

- The broken r intervals on #581's guarded draw are explained and fixed with #601. The Wald interval is symmetric on the log scale, so it still differs from the profile there: [0.034, 0.530] against (0, 0.331].
- The profile's open lower end at r -> 0 is a property of the Laplace objective. The review's table suggests the exact marginal would close it at a positive r; not tested.
- #607's floor-first order could report 0 on a non-monotone profile; no such case was looked for.
- The shared-r truncated-NB2 fitter has no boundary restart; it was not probed for the #627 stall.
- Five seeds of one simulation design back the #627 stall rate and the 80% Poisson-limit figure.

## 11. Team Learning

- Before flagging a boundary estimate as a failure, fit the same model in the reference engine. Here gllvmTMB showed the Poisson limit was normal and, separately, exposed a real wrong-optimum bug.
- A test premise that depends on where an optimiser ends is machine-dependent. Test the rule directly and wire it with a deterministic start (zero iterations).
- Merge gates must key on `status == COMPLETED` per head SHA and must not use `--auto` where auto-merge is off. The hand-rolled gate with `--match-head-commit` worked unattended across an app restart.
- In zsh scripts write `${VAR}` next to a colon, and never `pgrep -f` for a string your own watcher contains.

Memory receipt: loaded the GLLVM.jl LOAD-FIRST manifest (`route.py GLLVM.jl`), the repo `CLAUDE.md`, the `merge-when-green` skill (its `status == COMPLETED` rule and `--auto` warning shaped every merge), and the lane preflight each session. Recall of prior decisions came through other lanes' messages and the repo, not through the brain MCP.

Golden Set: not run; no listed known-mistake class was in scope beyond the merge-gate rule, which was followed.

## 12. Cross-Product Coverage

Covers: the truncated-NB2 shared-r and per-trait fitters (`converged` verdict, per-trait restart), the grouped mode search for NB1, truncated NB2, truncated Poisson and censored Poisson, the family profile intervals for log-scale parameters in `_family_profile`, and the truncated-NB2 frozen-R parity cell.

Does NOT cover: the Gaussian `profile_ci` and `phylo_beta_xlv.jl` (they share `_profile_bisect_side`, which was not changed, so they still report NaN for an open end); the `_family_ci` boundary flag for truncated NB2; the shared-r truncated-NB2 fitter's stall behaviour; `laplace.jl`'s `_laplace_mode`; NB2, Beta and Gamma grouped `getLV` (deliberately unchanged); exact-marginal intervals.
