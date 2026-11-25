# After-task: #1051 masked Gaussian BFGS polish

## Goal

`test/test_masked_gaussian_converged_735.jl` failed on some Linux/OpenBLAS targets with
`converged=false` and `g_residual` marginally above `g_tol=1e-6` (not test-order or RNG
leakage; see issue #1051 comments).

## Root cause

After mask/offset refit, L-BFGS polish restarts stall on a flat NLL plateau: each restart
does one step, leaves `g_residual` about 1.2–1.9e-6. A BFGS pass from the same point
reaches `g_tol` in a few iterations but was never run. When it was added, strict
`minimum(nxt) <= minimum(res)` rejected BFGS because NLL worsened by about 4.5e-13
(rounding).

## Change

`src/families/aghq_gaussian_fit.jl`: optional `:mask_offset_bfgs_polish` after the
existing L-BFGS polish loop when the outer method is not already BFGS; accept the stage
when the gradient test passes and any NLL regression is within numerical noise
(`_masked_gaussian_polish_accept`).

## Tests (this VM, Julia 1.13.4, `--project=test`)

- `test/test_masked_gaussian_converged_735.jl`: 5 passed, 0 failed, ~19s
- `test/test_optimizer_selection.jl`: 62 passed, 0 failed, ~92s

Full `Pkg.test()` not run here (~50 min).

## Decisions needed

None — `g_tol` and converged rule unchanged (#735 contract preserved).

Rose verdict: PASS WITH NOTES — targeted tests green; full suite not re-run on this agent.