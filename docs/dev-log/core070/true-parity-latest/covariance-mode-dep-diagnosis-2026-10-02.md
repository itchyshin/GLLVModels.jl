# Covariance-mode DEP failures in the Frozen R CI log: diagnosis (2026-10-02)

## Summary

The five failed checks printed at tools/core070_covariance_mode_fits.jl:373 are not a Julia/R disagreement and not a regression. They come from the default-control baseline build that test/parity/test_covariance_modes_required.jl runs in a separate Julia process (added by #614, commit 2304b1da1, 2026-09-30). That process uses R's default optimizer controls, and the retained contract (docs/dev-log/core070/covariance-mode-fits-contract.md, rows 11, 14 and 17) already records that the default-control R fits of the three DEP cases fail the R gradient check, and for ANIMAL-DEP and KERNEL-DEP also the structured covariance check. The gating run (tight controls) passes all 25 checks in all seven cases in the same job log. Class: (b), with a cosmetic side effect of #614. No gate was changed.

## Evidence

Job 110671286854 (Frozen R 0.7.0 family smoke), log lines in the baseline subprocess, started by the line "building a default-control baseline in a separate process":

- FIT-MODE-ORD-DEP: r_grad = 2.737e-4. One failed key: r_gradient (limit 1e-4).
- FIT-MODE-ANIMAL-DEP and FIT-MODE-KERNEL-DEP: r_grad = 2.477e-4. Two failed keys each: r_gradient and structured_source_covariance. Every other key, including likelihood, beta, native_health, native_objective_at_r_coordinates and structured_residual_variance, is true.
- That is 1 + 2 + 2 = 5 failures, matching the "22 pass, 2 fail" lines for each DEP testset.
- ANIMAL-DEP and KERNEL-DEP print identical r_grad because the two fixtures give the same R problem.

The retained default-control receipt from 2026-08-31 (docs/dev-log/core070/covariance-mode-fits-evidence.json) holds gradient_max 2.737107e-4 and 2.477076e-4 for the same cases, to six digits the same numbers as CI. The contract table next to it already lists exactly these failures. So the CI values reproduce the retained evidence; nothing drifted.

Later in the same log, the tight-control run (the one the required test asserts) prints r_grad = 5.09e-5 for both DEP cases, with r_gradient, structured_source_covariance and baseline_data_map_unchanged all true. The required run passes.

## Mechanism

The native fit converges to a gradient norm of 1e-7 or better. R with default controls stops at a gradient of about 2.5e-4. The check structured_source_covariance compares the native covariance with R's at atol = rtol = 1e-5, so an R fit that has not reached the optimum fails it; the covariance is only approximately at the optimum. Both failures are therefore the same R early-stopping symptom, once as a direct gradient check and once as a downstream covariance difference. The native objective evaluated at R's coordinates still matches R's objective to 1e-6 (native_objective_at_r_coordinates is true), which is consistent with a likelihood surface that is flat along the stopping direction rather than a mismatch in the model.

This is the same pattern the maintainer ruled "record, not gate" for NB2 (R gradient near 5e-5 to 6e-4). Here the tight-control run already handles it by tightening R's controls, so no rule change is needed for the gate.

## Since when

The tool and its default-control failures date from a20b89d57 (2026-08-31), where they were retained evidence only. They first appear in CI logs from 2304b1da1 (#614, 2026-09-30), which makes CI run the tool with default controls to produce a baseline. 5712e35d5 (2026-10-01) contains it. #635 (6aa88ec54, optional baseline) is not the cause: it only made the baseline optional for the tight run. The step is advisory (continue-on-error on the job, and the subprocess is started with ignorestatus), so the job concludes success.

## Classification

- (a) Real Julia/R disagreement: no. Likelihood, beta, residual variance and objective-at-R-coordinates all agree, and the tight run agrees on the covariance.
- (b) Check too strict at default R controls: yes, as designed and already documented.
- (c) Fixture or receipt mismatch: no.

## Proposal for the maintainer (not applied)

The only defect is log noise: the baseline subprocess prints "Test Failed" blocks and a red testset summary that look like a regression to a reader. Two options, neither changes a gate in the required run:

1. In test/parity/test_covariance_modes_required.jl, pass an environment variable (for example CORE070_BASELINE_BUILD=1) to the subprocess, and in tools/core070_covariance_mode_fits.jl skip the final assertion testset when it is set, printing one line "default-control baseline build: gradient failures expected, see covariance-mode-fits-contract.md". Retained-evidence runs by hand keep asserting as before.
2. Leave it and add a comment in the workflow step. Cheapest, but the noise stays.

I recommend option 1. It touches the parity test and tool, which the lane brief did not authorise me to change, so I have left it for a decision.

## Not done

I did not run R or Julia locally. The diagnosis rests on the CI log, the retained receipt and the source. Reproducing the default-control numbers locally would need the lane-local P1 or P0 R library.
