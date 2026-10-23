# F-PHYLO-LATENT-1FO

Verdict: VERIFIED

HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. The GATES.md checkbox for F-PHYLO-LATENT-1FO is ticked after the A15 run below. The scoreboard is unchanged. No row was admitted. C2 is not claimed MET.

## Contract

`docs/dev-log/core070/true-parity-latest/GATES.md` names this gate as bare `phylo_latent()`, first-order paired, carrying A14/A15 from the 0.7.0 scoreboard. The first-order row is A14. Its two paired cases are `STRUCT-PHY-TREE-RR` and `STRUCT-PHY-DENSE-RR`. The same replay also checks that the tree and dense routes agree, and that in-keyword `Ainv = inv(C)` follows the dense route. Those checks live in `test/test_phylo_latent_paired_p1.jl`. The tolerances live in `tools/phylo_latent/compare_phylo_latent_p1.jl` as `PL_P1_TOL`:

- logLik relative `1e-6`
- cross objective absolute `1e-8`
- estimates relative `1e-4` (`Sigma_phy`, beta, `sigma_eps^2`)
- log-det absolute `1e-8`
- cross gradient absolute `1e-4`

Tree-versus-dense agreement uses relative `1e-4`, the bound already written in that test. The fixture is `docs/dev-log/core070/phylo-latent-p1/a14-fixture.json`, SHA-256 `9a4c2e1f87a3fbb519400fff8a99fdfb6d66b249d473b0e25833b93b78e7e13c`. R is the SHA-256-pinned receipt from gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9` (0.7.1), DLL `cba0f54d5492f0c6d0c5474c19281e3b58d5e2b0709e536684619558aaf55b8d`. This run did not call R.

The approved execution contract asks for F-capability coverage and says to stop before an unsigned admission. This note records the A14 measurement. It does not add a scoreboard id.

A15 (`COV-PHYLO-LATENT-RSZ`) is the realistic-size receipt named in the same parenthesis. That A14 run did not refit it. The A15 section below records the later execution of the existing A15 testset. `extract_phylo_signal` is a different row and was not measured.

## Run

Local, 2026-10-20. `JULIA_NUM_THREADS=1`. `OPENBLAS_NUM_THREADS=1`. Julia 1.10.0. The process observed `Threads.nthreads() == 1` and `BLAS.get_num_threads() == 1`. Totoro was not used. The full suite was not run.

The A14 testsets in `test/test_phylo_latent_paired_p1.jl` were executed from this HEAD. The A15 testset in that file was left unexecuted in that A14 run. The repo file was not edited.

```
Test Summary:                         | Pass  Total  Time
F-PHYLO-LATENT-1FO A14 written replay |   78     78  0.5s
```

Exit code 0. The Julia process took 14.7 seconds, including startup. The tree fit took 1.60 seconds and includes the first call into the fitter. The dense fit took 0.023 seconds. The `Ainv` fit took 0.043 seconds. Wall time is not a pass bar.

The fresh tree and dense log-likelihoods matched the stored Julia receipts at the printed digits. Tree Julia logLik `-8.68192008542988` against R `-8.681920085429528`. Dense Julia logLik `-8.681920118992728` against R `-8.68192011899238`.

## A14 gaps

Fresh Julia fit against the recorded R receipt. Every gap is inside the bound above.

| Case | Quantity | Bound | Measured gap |
| --- | --- | --- | --- |
| tree | logLik relative | rel 1e-6 | 4.0511620786687847e-14 |
| tree | Julia objective at R's coefficients, absolute | abs 1e-8 | 1.0658141036401503e-14 |
| tree | R objective at Julia's coefficients, absolute | abs 1e-8 | 7.105427357601002e-15 |
| tree | R gradient at Julia's coefficients | abs 1e-4 | 9.327435694752052e-6 |
| tree | `Sigma_phy` relative | rel 1e-4 | 2.07777058552435e-7 |
| tree | beta relative | rel 1e-4 | 2.1376952318575102e-7 |
| tree | `sigma_eps^2` relative | rel 1e-4 | 2.5637392569962937e-8 |
| tree | log-det absolute | abs 1e-8 | 0.0 |
| tree | `n_aug` | 14 and 14 | match |
| tree | fresh `converged` | written test requires true | true |
| tree | fresh gradient norm at own optimum | reported with `converged`; the 1e-4 bar is the cross gradient above | 9.334047298553624e-6 |
| dense | logLik relative | rel 1e-6 | 4.0102412340883996e-14 |
| dense | Julia objective at R's coefficients, absolute | abs 1e-8 | 1.0658141036401503e-14 |
| dense | R objective at Julia's coefficients, absolute | abs 1e-8 | 3.552713678800501e-15 |
| dense | R gradient at Julia's coefficients | abs 1e-4 | 9.323844090827108e-6 |
| dense | `Sigma_phy` relative | rel 1e-4 | 2.115067508326458e-7 |
| dense | beta relative | rel 1e-4 | 2.0817980036931874e-7 |
| dense | `sigma_eps^2` relative | rel 1e-4 | 2.6873110798521528e-8 |
| dense | log-det absolute | abs 1e-8 | 0.0 |
| dense | `n_aug` | 8 and 8 | match |
| dense | fresh `converged` | written test requires true | true |
| dense | fresh gradient norm at own optimum | reported with `converged`; the 1e-4 bar is the cross gradient above | 9.330527119805625e-6 |
| Ainv | logLik relative vs dense R | rel 1e-6 | 4.092082891926938e-14 |
| Ainv | Julia objective at R's dense coefficients | abs 1e-8 | 1.0658141036401503e-14 |
| Ainv | log-det absolute | abs 1e-8 | 8.881784197001252e-16 |
| Ainv | `Sigma_phy` relative | rel 1e-4 | 2.113791232016148e-7 |
| Ainv | fresh `converged` and `n_aug` | written test: true, and 8 | true, 8 |
| Ainv | fresh gradient norm at own optimum | reported with `converged` | 9.334340646782625e-6 |
| routes, R | tree vs dense logLik relative | rel 1e-4 | 3.8658328564571656e-9 |
| routes, R | tree vs dense `Sigma_phy` relative | rel 1e-4 | 6.989053737848104e-9 |
| routes, Julia | tree vs dense logLik relative | rel 1e-4 | 3.865832447248721e-9 |
| routes, Julia | tree vs dense `Sigma_phy` relative | rel 1e-4 | 1.0718746832981231e-8 |

The written test also requires Julia's finite-difference gradient at R's coefficients to be at most `1e-4`. That assertion is inside the 78 passes. This note does not print that finite-difference value on its own line.

## Limits

The shared F-gate line still says `CHECK: none yet` and `EVIDENCE: pending`. This note does not change that line. A15 stays the stored realistic-size receipt, including its recorded stationarity gap. The original Julia receipts were written on Julia 1.10.12. This replay is Julia 1.10.0, and the fresh A14 log-likelihoods matched those receipts.

The thirteen paused rows were not refit. Issue 986 was not edited. Tolerances were not changed. Weights were not regenerated. Nothing was committed, pushed, or merged.

## A15 run

Local, 2026-10-20. `JULIA_NUM_THREADS=1`. `OPENBLAS_NUM_THREADS=1`. Julia 1.10.0. The process observed `Threads.nthreads() == 1` and `BLAS.get_num_threads() == 1`. Totoro was not used. The full suite was not run. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`.

The existing A15 testset in `test/test_phylo_latent_paired_p1.jl` was executed from this HEAD. The A14 testsets in that file were not run again. A14 remains the 78/78 pass recorded above. The repo file was not edited. Its SHA-256 before and after this run is `770eb85e8aacff5bbec1e461979a79e9bb0ac00fe9bed35cc1ea284aaed7607e`.

Command, from this worktree:

```
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/z3437171/.juliaup/bin/julia --project=. --color=no
```

The process evaluated the helpers in that file, then only this existing testset:

```julia
@testset "A15 COV-PHYLO-LATENT-RSZ" begin
    _plp1_case("cov_phylo_latent_rsz", "a15"; refit = true, stationarity_gap = true)
end
```

That call is the A15 check already in the file, with `refit = true`. Print lines lived only in that process, so the gaps below could be copied. The file on disk, its assertions, and its tolerances are unchanged.

```
Test Summary:            | Pass  Total   Time
A15 COV-PHYLO-LATENT-RSZ |   35     35  23.6s
```

Exit code 0. The Julia process took 39.7 seconds, including startup. The live fit took 19.4 seconds. Wall time is not a pass bar. The summary shows 35 passes and 35 total, so there were no skips and no failures.

## A15 gaps

The written A15 bar is the one in `_plp1_case` when `stationarity_gap` is true: `PL_P1_TOL` for logLik, the cross objective, `Sigma_phy`, `sigma_eps^2`, and log-det, plus `_PLP1_A15_BOUNDS` for the recorded stationarity gap, plus the live-refit assertions (`converged`, `stopping_reason === :converged`, `gradient_norm <= 1e-5`, beta absolute `<= 1e-4`). Every measured gap below is inside that bar.

| Quantity | Bound | Measured gap |
| --- | --- | --- |
| log-det absolute | abs 1e-8 | 0.0 |
| `n_aug` | 198 and 198 | match |
| Julia objective at R's coefficients, absolute | abs 1e-8 | 3.728928277269006e-11 |
| R objective at Julia's coefficients, absolute | abs 1e-8 | 1.7280399333685637e-11 |
| stored logLik relative | rel 1e-6 | 2.833931244154946e-13 |
| R gradient at the stored Julia coefficients | abs 1e-3 | 0.00020297691399518103 |
| Julia finite-difference gradient at R's coefficients | abs 1e-2 | 0.003972639056410717 |
| stored `converged` | false, `gradient_not_converged` | false, gradient_not_converged |
| stored gradient norm | abs 1e-3 | 0.00020178603297874473 |
| stored beta absolute | abs 1e-4 | 1.6353235340549332e-5 |
| stored `sigma_eps^2` relative | rel 1e-4 | 4.22124956235751e-7 |
| stored `Sigma_phy` relative | rel 1e-4 | 1.1196690997163215e-6 |
| live `converged` | true, `:converged` | true, converged |
| live gradient norm | abs 1e-5 | 6.157966023169731e-6 |
| live logLik relative | rel 1e-6 | 3.0720456238336163e-13 (live logLik `-7371.771401584334`) |
| live beta absolute | abs 1e-4 | 2.113210968657242e-6 |
| live `sigma_eps^2` relative | rel 1e-4 | 4.0045278305939204e-7 |
| live `Sigma_phy` relative | rel 1e-4 | 1.6500371338193223e-6 |

The recorded receipt still sits above the 1e-4 cross-gradient figure (R gradient 2.03e-4, Julia finite-difference gradient 3.97e-3). The existing test records that gap with the looser bounds in the table. Those bounds were already in the file.

The GATES.md checkbox for F-PHYLO-LATENT-1FO is ticked. A14 remains the recorded 78/78 pass, and A15 passed 35/35 inside the written bar. The shared F-gate line still says `CHECK: none yet` and `EVIDENCE: pending`. The scoreboard, receipts, and the thirteen paused rows were left untouched. Issue 986 was left unedited. F-TEMPORAL-1FO, F-ORDINAL-LOGIT-1FO, and F-ZI-1FO were left open. Weights were left as stored. Nothing was committed, pushed, or merged.
