# AGHQ P1 twin compares fixed points, not maximum-likelihood estimates

Date: 2026-10-01. Documentation only. No numbers, code paths or tolerances changed.

## Finding

The AGHQ P1 twin (itchyshin/GLLVModels.jl#651) compares Julia's adaptive Gauss-Hermite fits with R's gllvmTMB fits. An earlier comment in `test/fixtures/aghq_p1/gen_aghq_p1.R` said that on the four rejected Poisson seeds (20260103, 20260107, 20260110, 20260111) Julia ends below R's logLik. The sign was wrong. On those seeds Julia ends above R by 2.8e-5 to 2.2e-4 logLik, and stops with reason `no_merit_descent`. The comment is now corrected.

## Cause

Both engines run the same loop. gllvmTMB (`fit-multi.R`, roughly lines 8270 to 8600 at 9539352f6) matches `src/families/aghq_outer.jl`. Each step is accepted on the AGHQ objective after the nodes are re-centred (adapted) at the current point. Convergence is certified on the gradient with the nodes held frozen. R's reported optimum is therefore a fixed point of the adapt-then-step loop. It is not a minimum of the adaptively re-centred objective.

## Evidence

Scripts are in `~/local-scratch/aghq-stall-work/` (gen.R, rtrace.R, repro.jl, diag.jl, runs.jl) on the maintainer's machine, not in the repo.

- The true gradient of the objective at R's reported point is 0.09 to 0.14, not near zero.
- Direct minimisation beats R's logLik by 7.6e-5 to 3.5e-4 on every tested seed.
- That includes seed 20260113, where the twin matches R to 2e-7. Agreement with R there does not mean both sit at the maximum.

## What the twin does and does not establish

It establishes that Julia reproduces R's adapt-then-step algorithm closely, and that on accepted seeds the two end at the same fixed point. It does not establish that either engine returns the AGHQ maximum-likelihood estimate. The logLik gaps (below 4e-4) are numerically small, but the parameters differ by about 1e-3 on the rejected seeds, so the shortfall is not purely cosmetic.

## Open contract decision

Two options, for the maintainer to choose:

1. Certify convergence on the true gradient of the adapted objective in both engines. Both would then return real maximum-likelihood estimates and the twin would compare estimates. This means changing gllvmTMB as well as this package, and the rejected-seed list could probably be dropped.
2. Keep fixed-point comparison as the contract and say so plainly in the twin documentation. This is cheaper and agreement with R stays the target, but the AGHQ results should not be described as maximum-likelihood estimates.

Until one is chosen, read the twin as option 2.
