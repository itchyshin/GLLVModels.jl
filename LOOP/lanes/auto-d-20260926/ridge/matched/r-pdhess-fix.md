# Proposed gllvmTMB fix: test the penalised Hessian under the loading ridge

Status: APPLIED on gllvmTMB #1324 at 215f9544f (2026-09-29). Matched K = 3 cell after the fix: bic_sites 9/10 (was 4/10), bic 4/10 (was 3/10), 0 rejected; `R_pdhessfix_n120_p20_K3.csv`.

## Why

`select_lv()` rejects a fit when `fit$sd_report$pdHess` is FALSE. Under `aghq_ridge` the penalty is added
in R, outside the TMB template, so `sdreport()` tests the unpenalised Hessian at the penalised optimum.
At all six flagged fits checked, the penalised Hessian is positive definite (smallest eigenvalue 0.088 to
0.245) and the penalised gradient is 1e-4 to 3e-4. #1092 fixed the same defect for the gradient.

## Change (R/select-lv.R)

Add a helper and use it at both places that set `pdh` (currently lines 852 and 920 on #1324):

```r
## pdHess of the objective the optimiser minimised. Without a ridge this is
## sdreport's own flag. Under the loading ridge (applied in R, outside the TMB
## template) sdreport tests the UNPENALISED Hessian at the penalised optimum,
## which can be indefinite at a proper penalised optimum (same defect as
## #1092 for the gradient), so add 1/tau^2 on the ridge block and test that.
.select_lv_pd_hessian <- function(fit) {
  if (is.null(fit$sd_report)) return(NA)
  if (isTRUE(fit$sd_report$pdHess)) return(TRUE)
  tau <- fit$aghq$ridge_tau %||% Inf
  obj <- fit$tmb_obj
  par <- fit$opt$par
  if (is.null(obj) || !.gllvmTMB_loading_ridge_applies(tau, names(par))) return(FALSE)
  H <- tryCatch(stats::optimHess(par, obj$fn, obj$gr), error = function(e) NULL)
  obj$fn(par)  # leave the tape at the fitted parameters
  if (is.null(H) || any(!is.finite(H))) return(FALSE)
  li <- .gllvmTMB_ridge_block_index(names(par))
  diag(H)[li] <- diag(H)[li] + 1 / tau^2
  min(eigen((H + t(H)) / 2, symmetric = TRUE, only.values = TRUE)$values) > 0
}
```

```r
pdh <- .select_lv_pd_hessian(fit_try)   # line 852
pdh <- .select_lv_pd_hessian(res$fit)   # line 920
```

Record in the table which Hessian was tested (e.g. a `message` "unpenalised Hessian indefinite; penalised
Hessian positive definite") so the change is visible to users.

## Tests (tests/testthat/test-select-lv*.R)

1. On one matched dataset (n = 120, p = 20, K = 3, rep 3; `Y_n120_p20_K3_r3.csv`), `select_lv(...,
   binary_ridge = 2)` accepts d = 3 (status "ok"), and `pd_hessian` is TRUE for d = 3.
2. `binary_ridge = Inf` unchanged: an unpenalised fit with `sdreport$pdHess = FALSE` is still rejected.
3. After the fix, re-run `r_on_julia_data.R` on the four matched cells and compare with Julia
   (expected: n = 120, p = 20, K = 3, `bic_sites` moves from 4/10 towards Julia's 9/10).

## Cost

One numerical Hessian (2 × npar gradient calls) per ridge fit whose sdreport flag is FALSE only.
