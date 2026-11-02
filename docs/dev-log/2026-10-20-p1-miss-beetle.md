# data/RD-BEETLE-NB2 is not admitted

Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`, HEAD `5cfc6754212588222779ccfd66ad7000375cf8d5`. This note only diagnoses the existing receipt. It does not change a tolerance, a scoreboard, or the receipt, and it does not admit the row. The flat-dispersion / loading-degeneracy / negative-curvature split was not run.

`data/RD-BEETLE-NB2` fails and is not admitted. Receipt: `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-BEETLE-NB2.json`.

Labels, one per miss:

1. R `pdHess` false: **R_SIDE_BLOCKER**
2. logLik absolute difference `9.080691597773694e-06` > `1e-06`: **REAL_DIFFERENCE**
3. beta absolute difference `0.00028714120128903176` > `0.0001`: **REAL_DIFFERENCE**
4. LLt absolute difference `0.004632934295180746` > `0.0001`: **REAL_DIFFERENCE**
5. linear predictor absolute difference `0.0008694545642065066` > `1e-06`: **REAL_DIFFERENCE**

The four numeric misses are aligned quantities that differ by those amounts. Trait order matches, `Lambda Lambda'` is already the rotation and sign invariant, and a fresh read of the committed raw files reproduces the receipt maxima.

## R pdHess false alone keeps the row unbound

Yes. If the four numeric gaps were later brought inside their tolerances and `pdHess` stayed false, the row would still not bind.

`pass_rule` sets `R_pdHess_true` only when `R["pdHess"] is True` (`tools/true_parity/campaign/write_receipts.py:307`). A campaign row binds only when every leg is true (`write_receipts.py:570`). `R_pdHess_true` is outside `OPEN_LEGS` (`write_receipts.py:321`), so `numbers_ok` is false as well (`write_receipts.py:334-336`). The C4 rule text requires R convergence 0 with a positive-definite Hessian (`write_receipts.py:408-411`). With `pdHess` false, the verdict stays `FAIL` (`write_receipts.py:432`) and the row status starts with "does not bind: R's Hessian is not positive definite (pdHess is false)" (`write_receipts.py:326`).

The receipt is already in that state. `pass_rule.legs.R_pdHess_true` is false (`RD-BEETLE-NB2.json:15`) and `engines.R.pdHess` is false (`RD-BEETLE-NB2.json:140`). R recorded `pdHess` with `isTRUE(sdr$pdHess)` (`tools/true_parity/campaign/run_R.R:125`). The optimizer code is 0 and the message is "relative convergence (4)" (`run_R.R:119`, receipt `:138-139`). The recorded max absolute gradient is `0.00295313866816116` (`run_R.R:121`, receipt `:141`). `cond(H)` is `339301874.918461` (receipt `:143`). The R log warns that `sqrt(diag(sdr$cov.fixed))` produced NaNs (receipt `:57`).

This receipt has no N9 gradient leg. `convergence_parity` returns nothing unless both raw files carry a finite max absolute gradient (`write_receipts.py:296-297`). The campaign Julia file has no `max_abs_gradient`, so the R gradient is recorded and is not the leg that unbinds the row. `pdHess` false is sufficient on its own.

## logLik: REAL_DIFFERENCE

R marginal logLik `-8933.78120307063` (`run_R.R:120`, `logLik = -opt$objective`). Julia `-8933.781193989938` (`tools/true_parity/campaign/run_J.jl:207`). The checker difference is `9.080691597773694e-06`, tolerance `1e-06` (`write_receipts.py:152` and `:216`; receipt `:243-251`). Julia is higher by that amount. The same subtraction on `raw/beetle_R.json.gz` and `raw/beetle_J.toml.gz` matches the receipt.

Today's capture rebuilt the returned Julia objective with absolute error 0 and max absolute gradient `6.5472829954215804e-06`. Its reported logLik is `-8933.781193989882` (`docs/dev-log/2026-10-20-p1-c4-spider-beetle-fungi-capture.md:35`), which differs from the campaign Julia value by `5.6e-11`. The gap versus R remains `9.080691597773694e-06`. That capture did not run the pathology split, and it does not admit the row.

## beta: REAL_DIFFERENCE

Both sides have 71 coefficients. The 68 trait levels are equal, and the checker requires the R names to be `trait` plus each species, then `pH`, `Moist`, `Org` (`write_receipts.py:225-234`). The max absolute difference is index 18, `traitsp19`: R `-2.53416191256899`, Julia `-2.534449053770279`, difference `0.00028714120128903176` > `0.0001` (`write_receipts.py:153`; receipt `:402-408`).

Six of 71 coefficients exceed `0.0001`: `traitsp19` `0.00028714120128903176`, `traitsp28` `0.00027374250880995987`, `traitsp27` `0.00013221685420639062`, `traitsp51` `0.00012735576256472925`, `traitsp37` `0.00011983707446816538`, and `traitsp13` `0.0001001448629223134`. The covariate slopes stay inside `0.0001`. The largest is `Moist`: R `-0.457846914797519`, Julia `-0.45786341262715297`, difference `1.6497829633976835e-05`.

## LLt: REAL_DIFFERENCE

R stores `extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none")$Sigma` (`run_R.R:143-145`). The same file records `LLt_from_Lambda_B_max_abs_diff = 0`, so that matrix matches `Lambda_B %*% t(Lambda_B)`. Julia stores `rows(fit.Λ * fit.Λ')` (`run_J.jl:208`). The tolerance rule calls this the rotation and sign invariant (`write_receipts.py:155`). R `LLt_trait_names`, R `trait_levels`, and Julia `trait_levels` are the same 68 names.

The max absolute element is row 18, column 36, `sp19` by `sp37`, and the transpose element is the same pair: R `-59.276468357273`, Julia `-59.28110129156818`, difference `0.004632934295180746` > `0.0001` (receipt `:412-419`). 1188 of 4624 elements exceed `0.0001`. The relative Frobenius difference is `2.312787562271615e-05`. The largest absolute entry on the R matrix is `247.404382336875`.

## linear predictor: REAL_DIFFERENCE

R stores `as.numeric(fit$report$eta)` (`run_R.R:137`). Julia stores `vec(fit.β .+ O .+ fit.Λ * Z')` after `getLV(fit, Y, X; rotate = false)` (`run_J.jl:218-221`). Both vectors have length 5916, which is 68 species by 87 sites. The max absolute difference is index 1650: R `-9.7874270571224`, Julia `-9.788296511686607`, difference `0.0008694545642065066` > `1e-06` (`write_receipts.py:177` and `:249`; receipt `:422-430`). In Julia's column-major species-by-site layout, index 1650 is trait 18 (`sp19`) at site 24, the same species as the worst intercept. 5763 of 5916 entries exceed `1e-06`. The median absolute difference is `2.1176743367989914e-05`.

## Recorded dispersion sizes

The Julia warning names groups 13, 28, 30, and 37 above the `1e6` size boundary (receipt `:58`, pointing at `grouped_dispersion.jl:925`). Those are the true entries in `dispersion_boundary`. This row does not score dispersion. The raw sizes are still far apart: `sp37` R `42418940.3542573` and Julia `9.671911174468858e+74`; `sp28` R `275771814.196211` and Julia `367141440460.5911`; `sp30` R `812477424.817218` and Julia `10685386230.563404`; `sp13` R `172590.377130604` and Julia `1.0158948369117058e+21`. The largest scored LLt gap is the `sp19` by `sp37` entry above. Those sizes were not put through the pathology split, so this note leaves the cause of `pdHess` false unassigned.

Both engines read data sha256 `1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`. The row is not admitted.
