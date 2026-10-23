# F-ISDM-1FO-PREDICT

Verdict: VERIFIED

HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. The earlier open status was superseded by the later pass recorded below: `test/parity/isdm_cases.jl` passed 41/41 and 219/219. The GATES.md checkbox is ticked. The scoreboard is unchanged. No row was admitted. C2 is not claimed MET.

## Contract

`docs/dev-log/core070/true-parity-latest/GATES.md` names this gate as R's public door, non-spatial, Laplace, first order plus `predict`. Those words are the signed D-295 scope and the scope sentence of `docs/design/isdm-port-spec.md`. Section 3.4 of that spec names the paired cases and the bounds. The file that runs them is `test/parity/isdm_cases.jl`, and its header gives the command `julia --project=. test/parity/isdm_cases.jl`.

The public-door cell is `isdm_case("predict")`. It has 120 rows. The family is `isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))`. The formula is `value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 1, unique = FALSE)`. There is no spatial term. The likelihood call is `isdm_marginal_loglik_laplace`. The fit is `fit_isdm_gllvm`. `predict` is compared on the link and response scales, in sample, with random effects removed, and on the training rows after the offset is set to zero.

The bounds in that cell, as the test writes them from section 3.4, are:

- absolute `1e-6` for the Laplace marginal at R's fitted coefficients against R's logLik, and for the recorded R objective at Julia's recorded estimate against Julia's marginal there
- absolute `1e-4` and, on this cell, relative `1e-4` for `b_fix` by name
- absolute `1e-4` for the linear predictor
- relative `1e-4` for `Lambda Lambda'`
- absolute `1e-4` for each of the five `predict` vectors

The R numbers are the recorded P1 fixture `test/fixtures/isdm/r_values_p1.toml`. That file pins gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9` (0.7.1) and was written 2026-09-27. This run did not call R.

The approved execution contract (`docs/dev-log/plans/2026-10-19-p1-approved-execution-contract.md`) asks for F-capability coverage and says to stop before an unsigned admission. This note records the measurement. It does not add a scoreboard id.

## Run

Local, 2026-10-20. `JULIA_NUM_THREADS=1`. `OPENBLAS_NUM_THREADS=1`. Julia 1.10.0. The gap script observed `Threads.nthreads() == 1` and `BLAS.get_num_threads() == 1`. Totoro was not used. The full suite was not run.

```
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/.juliaup/bin/julia --project=. test/parity/isdm_cases.jl
```

Exit code 0. Wall clock 37.4 seconds, including startup.

```
Test Summary:        | Pass  Total  Time
P1-ISDM-ADMISSION-20 |   41     41  4.6s
Test Summary:        | Pass  Total   Time
iSDM P1 paired twins |  219    219  10.1s
```

The same predict cell was then evaluated once more so the gaps could be printed. The in-sample link gap matched the first fresh fit to the printed digits.

## Predict-cell gaps

Fresh Julia fit against the recorded R fixture. Every gap is inside the bound above.

| Quantity | Bound | Measured gap |
| --- | --- | --- |
| Laplace marginal at R's `(b_fix, Lambda)` vs R logLik | abs 1e-6 | 9.691802915767767e-12 |
| Recorded R objective at Julia's recorded estimate vs Julia marginal | abs 1e-6 | 1.2363443602225743e-11 |
| Recorded Julia estimate vs its recorded logLik | abs 1e-9 in the test | 0.0 |
| Fresh `b_fix` vs R, maximum absolute | abs 1e-4 | 2.629068578896554e-6 |
| Fresh `b_fix` vs R, maximum relative | rel 1e-4 | 1.5498765494519738e-5 |
| Fresh linear predictor vs R, maximum absolute | abs 1e-4 | 3.2001637834677688e-6 |
| Fresh `Lambda Lambda'` vs R, maximum absolute | rel 1e-4 in the test | 8.873099306971888e-7 |
| Fresh `Lambda Lambda'` vs R, maximum relative | rel 1e-4 | 6.499328861488664e-6 |
| `predict` link, in sample, 120 rows | abs 1e-4 | 3.2001637834677688e-6 |
| `predict` response, in sample, 120 rows | abs 1e-4 | 6.595181629709401e-6 |
| `predict` link, `re_form = :zero` | abs 1e-4 | 3.474671117165151e-6 |
| `predict` link, newdata, offset zeroed | abs 1e-4 | 3.2001637834677688e-6 |
| `predict` response, newdata, offset zeroed | abs 1e-4 | 4.3967877527695265e-6 |

`ft.converged` was true and every cell mode converged. `predict` returned the keys `(:cell_id, :trait, :isdm_source, :est)`. R's recorded convergence on this cell is 0, with `pdHess` true.

The stored receipt `docs/dev-log/core070/true-parity-latest/receipts/julia-twins/isdm/predict.json` already records the two logLik gaps and the `b_fix` absolute gap above. Its own text says that receipt is a fit-level comparison and does not restate `predict`. The five `predict` gaps in the table were computed in this run.

## Limits

The port-spec header still says draft for maintainer review. This note does not sign that draft. The paired test that section 3.4 names passed on this HEAD.

Intervals are outside this gate. Spatial iSDM is outside this cell. The other three paired cases in the same file (`ms3`, `srcform_pois`, `srcform_mixed`) and the 20 admission predicates also passed in the 219 and 41 counts. Those counts are file results. The table above is the public-door cell this gate names.

The thirteen paused rows were not refit. Issue 986 was not edited. Tolerances were not changed. Weights were not regenerated. Nothing was committed, pushed, or merged.
