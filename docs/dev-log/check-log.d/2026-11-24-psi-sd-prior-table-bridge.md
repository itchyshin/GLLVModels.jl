# Psi SD prior-table bridge check log

Source under validation: shared-table base `a2d5267a8fefed301cfd14615acf6e9b53aec9c3` plus the isolated Psi bridge commit `290af8a2280a0746012bff5f41335ddf5bcf922c` and subsequent test-only coverage for kernel-order reversal.

## Commands run

- `git diff --check` — passed after the source bridge and after the kernel-order test addition.
- `git status --short --branch` — confirmed the validation branch contains only Psi-owned files before each local commit.
- `git worktree add -b codex/psi-sd-prior-table-validate-20261124 ... a2d5267a8fefed301cfd14615acf6e9b53aec9c3` — created a separate validation worktree from the frozen shared-table source.

## Tests deliberately not run yet

- `test/auto/test_psi_sd_prior.jl` and `test/auto/test_prior_table_source_fit.jl` — Totoro's DRModels Julia `Pkg.test()` remained active (PID 2698092 at 2026-11-24 03:46 UTC); the Mac also had an active JuliaCall bridge test and R package checks. No process was stopped or overlapped.
- `check-table-derivatives.jl` — deferred until the Julia slot is released; this is the required five-point public-table objective, gradient, and full-Hessian comparison against the frozen oracle and central differences.
- Full `Pkg.test()` — deferred until focused checks pass and a single-thread slot is available.
- The current-main gllvmTMB public-table probe and full R package suite — deferred until the main lane provides a frozen current-main R integration pin. The earlier dbeff133 result is not current-main evidence.

## Scope notes

- No reader-facing docs, formula route, Paper Two analysis, push, merge, or release was changed.
- No exact rg stale-wording scan applies because this bridge did not change public documentation wording.
- The coevolution two-kernel formula route remains unsupported pending its own unique-source model route and independent validation.
