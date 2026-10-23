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

## Change applied (option 1, at the coordinator's request)

The noise is removed without touching any check, tolerance or key.

- test/parity/test_covariance_modes_required.jl starts the default-control baseline subprocess with CORE070_BASELINE_BUILD=1 (via addenv, so the variable exists in that child only). Before the tight-control run it errors if the variable is set in its own environment, so a leak is a hard failure rather than silently skipped assertions.
- tools/core070_covariance_mode_fits.jl reads the variable. It errors if the variable is set together with the tight-control policy. When it is set under default controls, the final assertion testset is skipped and one line is printed naming the cases and keys expected to fail (r_gradient for the three DEP cases; structured_source_covariance also for ANIMAL-DEP and KERNEL-DEP) with a pointer to covariance-mode-fits-contract.md. The CORE070_COVARIANCE_MODE_FITS_PASS line is not printed in that mode.
- With the variable unset, which covers the tight gate run and hand-run retained-evidence runs, the tool asserts exactly as before.

How it was checked: the tool needs the pinned R oracle and the parity Julia environment, which are not instantiated on this Mac, so the full tool was not run. Instead I ran the modified guard and the modified assertion block verbatim in a stub with one failing check, over the four combinations. Variable unset (default or tight): the assertion fails as before. Variable set, default policy: assertions skipped and the expectation line printed. Variable set, tight policy: error "must not be set for a tight-control run". The CI Frozen R job will be the end-to-end check on the next run.

## Not done

I did not run R or Julia locally. The diagnosis rests on the CI log, the retained receipt and the source. Reproducing the default-control numbers locally would need the lane-local P1 or P0 R library.
