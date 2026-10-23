## Is R's pd_hessian = FALSE under the ridge a false alarm? Compare the unpenalised Hessian
## (what TMB::sdreport checks) with the penalised one (+ 1/tau^2 on theta_rr_B) at the same optimum.
suppressMessages(devtools::load_all("~/local-scratch/lanes/gllvmTMB-auto-d-20260926", quiet = TRUE))
for (r in c(3, 5, 6)) {
  Y <- as.matrix(read.csv(sprintf("/Users/z3437171/local-scratch/lanes/GLLVM.jl-auto-d-20260926/LOOP/lanes/auto-d-20260926/ridge/matched/Y_n120_p20_K3_r%d.csv", r), header = FALSE))
  Y <- matrix(as.integer(Y == TRUE | Y == "true"), nrow(Y), ncol(Y))
  df <- data.frame(unit = factor(rep(seq_len(ncol(Y)), each = nrow(Y))),
                   trait = factor(rep(seq_len(nrow(Y)), ncol(Y))), value = as.vector(Y))
  sel <- suppressMessages(suppressWarnings(select_lv(
    value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
    data = df, unit = "unit", trait = "trait", family = stats::binomial(),
    d_max = 4, criterion = "bic_sites", binary_ridge = 2)))
  tb <- sel$table
  for (k in tb$d) {
    f <- sel$fits[[as.character(k)]]; if (is.null(f)) next
    tau <- f$aghq$ridge_tau %||% Inf; par <- f$opt$par
    H <- optimHess(par, f$tmb_obj$fn, f$tmb_obj$gr)
    li <- which(names(par) == "theta_rr_B")
    Hp <- H; if (is.finite(tau)) diag(Hp)[li] <- diag(Hp)[li] + 1 / tau^2
    gp <- .gllvmTMB_penalised_gradient(f$tmb_obj, par, tau)
    cat(sprintf("rep %d d %d status %-11s pdHess %-5s | min eig unpen %9.3g  pen %9.3g | max|grad| unpen %8.2g pen %8.2g\n",
      r, k, tb$status[tb$d == k], tb$pd_hessian[tb$d == k],
      min(eigen(H, symmetric = TRUE, only.values = TRUE)$values),
      min(eigen(Hp, symmetric = TRUE, only.values = TRUE)$values),
      max(abs(f$tmb_obj$gr(par))), max(abs(gp))))
  }
}
