# After-task: #458 grouped-analytic identity (2026-09-23)

Active lenses: Shannon, Ada, Noether, Rose (perspectives). Spawned subagents: none.

Lane: `cursor/458-grouped-analytic-identity-20260923` at
`~/local-scratch/lanes/GLLVM.jl-458-identity-20260923`.

## Scope

Close GitHub #458: Totoro reverify (PR #455) printed `GATE GB.3 FAIL` /
G9.1 on tip `8d58a0c94`. Do not demote Latte or Nelder-Mead defaults.

## Failing fixture

`poisson_percoord` (Poisson, `:indep` with `common=false`).

Reproduced on Julia 1.12 x64 locally (Totoro class). aarch64 stayed green
before the fix because beta rel stayed under `1e-8`.

## Root cause

Not a wrong analytic gradient (GB.2 already PASS; loglik agrees at ~1e-13).

`gate_identity` required a hard `dbeta <= 1e-8` **and** `_s9c_theta_verdict`.
On x64, beta coords over that bare rtol were already FLAT on the cold
objective, so the theta verdict correctly passed while the hard beta check
failed. The printed "theta offenders not explained by flatness" line was
misleading: those coords had been adjudicated flat.

## Fix

In `test/test_grouped_analytic_grad.jl` only:

- `gate_identity` / `gate_mixed`: drop the hard `dbeta` conjunct; keep dll,
  `tv.pass`, and matching `converged`.
- `_s9c_theta_verdict`: return only non-flat offenders.

No `src/` edit. Latte `diag_precision_kernel` default unchanged. NM default
unchanged.

## Checks

- Julia 1.12 x64: `--gate identity` → `GATE GB.3 PASS`, exit 0.
- Julia 1.12 aarch64: same.
- Compound GB.3 shell (analytic identity && laplace identity): exit 0.

## Rose

Claim is gate honesty for flat ridges, not a new speed or Latte claim.
G9.3 / G9c.2 mixed-path `calls=0` stays open and separate.
