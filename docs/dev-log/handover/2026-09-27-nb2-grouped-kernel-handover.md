# Handover: NB2 grouped kernel + grouped-fitter init keywords (2026-09-27)

From: Claude (session in the glmmTMB folder, worktree `../GLLVM.jl-worktrees/nb-grouped-init`).
To: whichever Claude session picks up draft PR #521. Shinichi confirmed (2026-09-27, in chat) that this lane owns the NB2 grouped kernel `_nb_grouped_loglik_site`. The backlog lane stepped back from it.

## Landing state

| Item | State | Where | Resume |
|---|---|---|---|
| Kernel fix + init keywords + tests + fixtures | LANDED on branch, draft PR | #521, `claude/nb-grouped-init-v2` @ e22221eb0 (plus this file) | `git fetch origin && git worktree add ../nb2 origin/claude/nb-grouped-init-v2` |
| `test/runtests.jl` registration of the two new test files | CARRIED-OVER | not done | The auto-d lane released its lease (17:1xZ). Add two `_shard_include` lines next to `test_nb1_grouped_mode_search.jl` (around line 212) once the lease is free (`tools/lane_lease.sh --list GLLVM.jl`) |
| CHANGELOG entry | CARRIED-OVER | not done | Lease now free. Put the entry after #507's NB1 grouped entry |
| Merge | GATED on Shinichi | | It changes fitted NB2 results on the default route and adds public keywords (D-290). Draft only; never merge from this lane |
| Independent review | OWED | stopped before reporting when that session closed | Handed to the new true-parity lane (GLLVM.jl folder), which carries the four review points. The auto-d branch (#518) also adds a runtests line and CHANGELOG entries, so whichever merges second rebases |

FINDINGS-OF-RECORD: the NB2 Fisher-scoring 2-cycle mechanism below. It is recorded in the PR body and in the comment above `_nb_grouped_mode`. vault-note: AGENT_LOG entry 2026-09-27 (NB2 grouped kernel).

## What was found (the durable part)

- Mechanism: where y ≫ μ, the NB2 Fisher weight μr/(r+μ) is smaller than the observed curvature μr(r+y)/(r+μ)² by the factor (r+y)/(r+μ). The Fisher step overshoots and 2-cycles. The step is small enough to fall under the `norm(Δ) <= 1e-3(1+|z|)` shortcut, so step-halving never triggers. Damping alone is therefore not enough: without the observed fallback, K=2 on the seed-1 data returns -Inf at the warm start.
- For NB2/log the observed weight is always positive, so observed Newton is exact and SPD. It is the fallback, and it runs only where Fisher failed.
- Main's poor optima also had inflated logliks: a non-mode z can raise the Laplace value through the log-det term. Both "lower" panel cells trace to this (0.0015 and 0.0067 overstatement).

## Numbers (against origin/main d55a8e3af)

- Seed-1 (p=20, n=300, K_true=3) `fit_nb_gllvm_grouped` loglik, main then branch: K=1 -19473.13 then -18400.45; K=2 -18874.03 then -17617.63; K=3 -19113.13 then -16743.17; K=4 -20240.21 then -16727.98 (monotone through K=4, max loading-row norm 2.44).
- Default-route panel, 48 NB2 cells: 18 higher (up to +5,340), 28 unchanged, 2 lower (both main-overstatement artefacts). NB1/Beta grouped: 8/8 identical.
- Ill-conditioned panel (loading scale 2.5, r ∈ {0.3, 50}): 7 higher (by 50 to 1,141), 1 unchanged, none lower.
- 13 panel cells where the branch reports `converged = false`: every one is the existing dispersion-boundary flag (one species' r > 1e6), not an optimizer failure.
- Runtime: the panel took 70% longer in total on the branch.
- Local tests: 27 files green (every `test_grouped*.jl`, `test_model_selection.jl`, and the NB2 grouped callers). The slow seed-1 test (`GLLVM_SLOW_TESTS=1`) passed.

## Open questions for Shinichi (drafted replies)

1. More NB fits will now report `converged = false` with the boundary warning. Suggested reply: "Accept; the boundary flag is honest and the fits are better."
2. +70% runtime on the NB default route. Suggested reply: "Accept for correctness; open a speed issue for making observed Newton the first step on NB2/log." That step would change healthy-site bits at about 1e-10, which is why it was not done here.
3. The public keywords `β_init` / `Λ_init` on three grouped fitters. Suggested reply: "Approve; they match fit_nb_gllvm."

## Not covered

- `_grouped_laplace_mode` (a shared kernel, used by `getLV` for these fits) has the same small-step shortcut and may return a non-mode z at 2-cycle sites.
- `fit_nb_gllvm_grouped_cov` uses the same kernel, so it is affected, but it was not panel-tested separately.
- Non-log links reach only the 20x Fisher retry. Not measured.
- `fit_gamma_gllvm_grouped` and the Tweedie grouped fitters did not get the init keywords.
- The shared (non-grouped) NB route `_laplace_mode` probably has the same 2-cycle. It was not measured, and that kernel is Shinichi's call.
