# data/RD-FUNGI-BINOMIAL misses

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Diagnosis only. `data/RD-FUNGI-BINOMIAL` is not admitted. The tolerances 0.0001, 0.0001, and 1e-06 stay as written in the receipt.

Labels: beta **REAL_DIFFERENCE**. LLt **REAL_DIFFERENCE**. Linear predictor **REAL_DIFFERENCE**.

The receipt `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-FUNGI-BINOMIAL.json:9` fails the row because beta abs 0.000123933 > 0.0001, LLt abs 0.000474438 > 0.0001, and the training linear predictor abs 0.000328279 > 1e-06. The unrounded maxima below are the values `maxabs` recomputed from the stored series. Log-likelihood is inside its bar: R `-5792.18625245028`, Julia `-5792.186252388768`, absolute difference `6.151276465971023e-08`, tolerance `1e-06` (`RD-FUNGI-BINOMIAL.json:172-183`).

## beta: REAL_DIFFERENCE

`maxabs` in `tools/true_parity/campaign/write_receipts.py:88` subtracts the stored coefficient vectors element by element. `cases_for` at `tools/true_parity/campaign/write_receipts.py:225-234` checks that R's names are the 59 trait intercepts in `trait_levels` order followed by `TEMPR`, `PRECIP`, and `logAREA`, then compares `R["beta"]` with `J["beta"]` in that order.

R stores `b_fix` at `tools/true_parity/campaign/run_R.R:133`. Julia stores `vcat(fit.β, fit.γ)` at `tools/true_parity/campaign/run_J.jl:215`. Both raw files list traits `sp01` through `sp59`.

One of the 62 coefficients is outside 0.0001. It is `traitsp54`:

| Side | `traitsp54` |
| --- | ---: |
| R | -5.55145011088071 |
| Julia | -5.551574043405938 |

`abs(-5.55145011088071 - (-5.551574043405938)) = 0.000123932525228021`.

The next intercepts are inside 0.0001 (`traitsp45` `6.987141826026e-05`, `traitsp47` `5.192225520921e-05`). The three slopes are inside 0.0001: `TEMPR` `4.842831192e-06`, `PRECIP` `3.6123207217e-07`, `logAREA` `1.984689871878e-06`. The median absolute coefficient difference is `2.6290901365966235e-06`.

## LLt: REAL_DIFFERENCE

R writes `extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none")$Sigma` at `tools/true_parity/campaign/run_R.R:143-145`. The same lines record `LLt_from_Lambda_B_max_abs_diff`. The raw file `fungi_R.json.gz` stores that check as 0, so the compared R matrix is `Lambda_B Lambda_B'`. Its trait names are `sp01` through `sp59`.

Julia writes `rows(fit.Λ * fit.Λ')` at `tools/true_parity/campaign/run_J.jl:208`. `cases_for` at `tools/true_parity/campaign/write_receipts.py:237` passes the two matrices through with no sign flip and no trait reorder. `orient` at `tools/true_parity/campaign/write_receipts.py:142-146` is used for temporal loadings only. The tolerance text at `tools/true_parity/campaign/write_receipts.py:155` is 1e-4 absolute per element of `Lambda Lambda'`, the rotation-and-sign invariant.

The maximum is the `sp54` diagonal:

| Side | Entry (`sp54`, `sp54`) |
| --- | ---: |
| R | 7.30290967368964 |
| Julia | 7.303384111216356 |

`abs(7.30290967368964 - 7.303384111216356) = 0.00047443752671583184`.

Both stored matrices are symmetric. Twenty-one unique pairs, 39 stored entries counting both off-diagonal copies, exceed 0.0001. Each of those pairs touches `sp54`, `sp47`, or `sp45`. The species order matches the beta order above.

## Linear predictor: REAL_DIFFERENCE

R stores `as.numeric(fit$report$eta)` at `tools/true_parity/campaign/run_R.R:137`. The long data are built at `tools/true_parity/campaign/run_R.R:67-70`: each site is repeated once per trait, so trait is fastest inside the site. Julia builds `Y` as traits by sites at `tools/true_parity/campaign/run_J.jl:199` and stores `vec(predict(fit, Y, X; type = :link))` at `tools/true_parity/campaign/run_J.jl:223`. That `predict` method is `src/postfit.jl:3084-3090`: `η = β + Xγ + Λẑ` on the training rows. Column-major `vec` of a traits-by-sites matrix is the same site order, trait fastest.

The maximum is flattened index 7900, site position 133 and trait `sp54`:

| Side | Training link predictor, site 133, `sp54` |
| --- | ---: |
| R | -11.8103781114479 |
| Julia | -11.810706390460352 |

`abs(-11.8103781114479 - (-11.810706390460352)) = 0.00032827901245191526`.

The tolerance is `1e-06`. 16012 of 17700 entries exceed it. The median absolute difference is `5.707953576239433e-06`. The largest entry, and 191 of the 300 `sp54` rows above `0.0001`, sit on `sp54`. Subtracting the `sp54` intercept gap from those rows leaves a remaining absolute gap up to `0.00020434648722389426`, so the linear-predictor gap includes the latent contribution as well as the intercept.

## What the capture shows

The 2026-10-20 capture in `docs/dev-log/2026-10-20-p1-c4-spider-beetle-fungi-capture.md:49` rebuilt the returned Julia objective for this cell. Reconstruction absolute error was 0. The maximum absolute gradient was `7.5812079072257177e-06`. The captured log-likelihood was `-5792.1862523887676`, beside the receipt Julia log-likelihood `-5792.186252388768`. That capture checks Julia's own returned point. It does not match Julia to R.

R returned convergence 0, `pdHess` true, and message `relative convergence (4)`, with its own maximum absolute gradient `0.00386035726088885` (`RD-FUNGI-BINOMIAL.json:140-144`). The receipt states that R and Julia condition numbers sit in different parameter coordinates (`RD-FUNGI-BINOMIAL.json:124`). R's engine message names `sp54` (prevalence 0.0433, max loading 2.62) and calls the loading a runaway (`RD-FUNGI-BINOMIAL.json:59-60`). R still returned the beta vector, the `Lambda Lambda'` matrix, and the training linear predictor that the checker compared. The failing checks are those three cross-engine values.

`data/RD-FUNGI-BINOMIAL` is not admitted.
