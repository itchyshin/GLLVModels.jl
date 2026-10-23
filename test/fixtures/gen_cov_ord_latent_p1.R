## Regenerates cov_ord_latent_p1.toml and cov_ord_latent_p1_data.csv.
## NOT run by CI or by any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1").
## Needs gllvmTMB installed at that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_cov_ord_latent_p1.R
##
## Twins of the covariance rows COV-ORD-LATENT-BARE, COV-ORD-LATENT-DEFAULT and
## COV-ORD-LATENT-COMMON. The R cases at P1 (tools/core070_covariance_batch.R) parse exactly these
## three formulas; here each one is FITTED, Gaussian, unit = "site", on one simulated data set
## (n = 60 sites, p = 4 traits, one observation per trait and site):
##   bare     value ~ 0 + trait + latent(0 + trait | site, unique = FALSE)
##            Sigma_B = Lambda Lambda'                       (rank 1), sigma_eps free
##   default  value ~ 0 + trait + latent(0 + trait | site)
##            Sigma_B = Lambda Lambda' + diag(psi_1..psi_p)  (auto per-trait unique diag)
##   common   value ~ 0 + trait + latent(0 + trait | site, common = TRUE)
##            Sigma_B = Lambda Lambda' + psi I_p             (one shared unique variance)
## vec(Y) ~ N(trait means, I_n (x) Sigma_B + sigma_eps^2 I). With a site-level unique diagonal
## (default, common) gllvmTMB auto-suppresses sigma_eps and FIXES it at max(0.001 sd(y), 1e-6)
## (the map of log_sigma_eps is NA); the bare fit estimates it. Both facts are recorded and
## asserted below. theta_rr_B is the packed rank-1 loading column; theta_diag_B is a log SD.
##
## The data are written to CSV first and the fits are made on the data READ BACK from the CSV,
## so R and Julia see identical doubles. Tight nlminb control (rel.tol, sing.tol, x.tol 1e-12, as
## gen_animal_scalar_p1.R); every fit must converge (code 0) with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")
mat <- function(m) paste0("[", paste(apply(m, 1, vec), collapse = ", "), "]")

## ---- simulated data ----
set.seed(20261005L)
n <- 60L; p <- 4L
lam <- c(0.9, -0.6, 0.7, 0.4); psi <- c(0.5, 0.3, 0.6, 0.4); mu <- c(0.3, -0.2, 0.5, 0.1)
z <- rnorm(n)
Y <- outer(lam, z) + matrix(rnorm(p * n), p) * sqrt(psi) + mu   # p x n
tr <- paste0("t", seq_len(p))
d0 <- data.frame(site = rep(seq_len(n), each = p), trait = rep(tr, n), value = as.vector(Y))
write.csv(d0, "cov_ord_latent_p1_data.csv", row.names = FALSE)

## ---- read back (the fitted doubles are the CSV doubles) ----
d <- read.csv("cov_ord_latent_p1_data.csv")
stopifnot(max(abs(d$value - d0$value)) < 1e-12)   # write.csv keeps 15 significant digits
d$site <- factor(d$site, levels = seq_len(n))
d$trait <- factor(d$trait, levels = tr)

ctl <- gllvmTMBcontrol(optArgs = list(control = list(rel.tol = 1e-12, sing.tol = 1e-12,
                                                       x.tol = 1e-12, eval.max = 2000L,
                                                       iter.max = 1500L)))
cases <- list(
  bare = list(id = "COV-ORD-LATENT-BARE", term = "latent(0 + trait | site, unique = FALSE)",
              names = c(rep("b_fix", p), "log_sigma_eps", rep("theta_rr_B", p)), n_diag = 0L,
              sigma_fixed = FALSE),
  default = list(id = "COV-ORD-LATENT-DEFAULT", term = "latent(0 + trait | site)",
                 names = c(rep("b_fix", p), rep("theta_rr_B", p), rep("theta_diag_B", p)), n_diag = p,
                 sigma_fixed = TRUE),
  common = list(id = "COV-ORD-LATENT-COMMON", term = "latent(0 + trait | site, common = TRUE)",
                names = c(rep("b_fix", p), rep("theta_rr_B", p), "theta_diag_B"), n_diag = 1L,
                sigma_fixed = TRUE))
sigma_rule <- max(0.001 * sd(d$value), 1e-6)

out <- list()
for (k in names(cases)) {
  cs <- cases[[k]]
  fml <- as.formula(paste("value ~ 0 + trait +", cs$term))
  f <- suppressMessages(suppressWarnings(gllvmTMB(fml, data = d, unit = "site", control = ctl)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  stopifnot(identical(names(f$opt$par), cs$names))
  pl <- f$tmb_obj$env$parList(f$opt$par)
  grad_max <- max(abs(f$tmb_obj$gr(f$opt$par)))
  sigma_eps <- exp(as.numeric(pl$log_sigma_eps))
  if (cs$sigma_fixed) {
    stopifnot(all(is.na(f$tmb_obj$env$map$log_sigma_eps)), abs(sigma_eps - sigma_rule) < 1e-15)
  } else {
    stopifnot(is.null(f$tmb_obj$env$map$log_sigma_eps))
  }
  lambda <- as.numeric(pl$theta_rr_B)
  stopifnot(length(lambda) == p)
  stopifnot(max(abs(as.numeric(f$report$Lambda_B) - lambda)) < 1e-12)   # Lambda_B is theta_rr_B as a column
  diag_logsd <- as.numeric(pl$theta_diag_B)
  free_diag <- if (cs$n_diag == 0L) numeric(0) else unique(diag_logsd)
  if (cs$n_diag == 0L) {
    stopifnot(all(is.na(f$tmb_obj$env$map$theta_diag_B)))
    psi_var <- rep(0, p)
  } else {
    stopifnot(length(free_diag) == cs$n_diag)
    psi_var <- exp(2 * diag_logsd)
  }
  if (identical(k, "common")) stopifnot(length(unique(psi_var)) == 1L)
  Sigma_B <- tcrossprod(lambda) + diag(psi_var)
  ## R's own accessor for the unit-level trait covariance: it reports the same matrix
  S_acc <- unname(extract_Sigma(f, level = "unit", part = "total")$Sigma)
  acc_diff <- max(abs(S_acc - Sigma_B))
  ll <- as.numeric(logLik(f))
  stopifnot(abs(ll + f$opt$objective) < 1e-8)
  ## Likelihood identity away from the optimum: R's objective at a fixed perturbed parameter
  ## vector (Gaussian, so TMB's Laplace marginal is exact). The Julia twin evaluates its own
  ## objective at the same point.
  probe <- as.numeric(f$opt$par) + 0.05 * sin(seq_along(f$opt$par))
  probe_ll <- -as.numeric(f$tmb_obj$fn(probe))
  stopifnot(is.finite(probe_ll), probe_ll < ll)
  out[[k]] <- list(cs = cs, f = f, ll = ll, beta = as.numeric(pl$b_fix), lambda = lambda,
                   free_diag = free_diag, psi_var = psi_var, sigma_eps = sigma_eps,
                   Sigma_B = Sigma_B, acc_diff = acc_diff, grad_max = grad_max,
                   df = attr(logLik(f), "df"), probe = probe, probe_ll = probe_ll)
}
## The bare model (free residual) and the common model (shared unique variance, fixed tiny
## residual) differ only in where the one shared diagonal variance sits; recorded, not asserted.
bare_common_ll_diff <- out$common$ll - out$bare$ll

con <- file("cov_ord_latent_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for covariance rows COV-ORD-LATENT-BARE / -DEFAULT / -COMMON (gllvmTMB P1).")
w("# Generated once by gen_cov_ord_latent_p1.R, which also writes the CSV (sha256 recorded).")
w("# Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("data_file = \"cov_ord_latent_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("cov_ord_latent_p1_data.csv"))
w("p = %d", p)
w("n_site = %d", n)
w("trait_names = %s", svec(tr))
w("# max(0.001 * sd(value), 1e-6): the residual SD gllvmTMB fixes when a site-level diag is present")
w("sigma_eps_rule = %s", fmt(sigma_rule))
w("# logLik(common) - logLik(bare): recorded only")
w("common_minus_bare_loglik = %s", fmt(bare_common_ll_diff))
for (k in names(out)) {
  o <- out[[k]]
  w("")
  w("[%s]", k)
  w("case = \"%s\"", o$cs$id)
  w("formula = \"value ~ 0 + trait + %s\"", o$cs$term)
  w("converged = true")
  w("pd_hessian = true")
  w("r_gradient_max = %s", fmt(o$grad_max))
  w("r_df = %d", as.integer(o$df))
  w("par_names = %s", svec(o$cs$names))
  w("sigma_eps_fixed = %s", if (o$cs$sigma_fixed) "true" else "false")
  w("loglik = %s", fmt(o$ll))
  w("# b_fix: trait intercepts, in trait order")
  w("beta = %s", vec(o$beta))
  w("# theta_rr_B: the rank-1 loading column (Lambda_B)")
  w("lambda = %s", vec(o$lambda))
  w("# free theta_diag_B entries (log SD of the unique diagonal); empty for bare")
  w("diag_logsd = %s", if (length(o$free_diag)) vec(o$free_diag) else "[]")
  w("# exp(log_sigma_eps): the residual SD (estimated for bare, fixed for default and common)")
  w("sigma_eps = %s", fmt(o$sigma_eps))
  w("# Lambda Lambda' + diag(exp(2 theta_diag_B)): the site-level trait covariance, row by row")
  w("trait_covariance = %s", mat(o$Sigma_B))
  w("# opt$par + 0.05 sin(1:k), in par_names order, and -obj$fn at that point")
  w("probe_par = %s", vec(o$probe))
  w("probe_loglik = %s", fmt(o$probe_ll))
  w("# max |extract_Sigma(fit, level = 'unit')$Sigma - trait_covariance|")
  w("extract_sigma_diff = %s", fmt(o$acc_diff))
}
close(con)
for (k in names(out)) {
  o <- out[[k]]
  cat(sprintf("%-8s loglik %.10f sigma_eps %.8g grad %.2e acc_diff %.2e df %d\n", k, o$ll, o$sigma_eps,
              o$grad_max, o$acc_diff, as.integer(o$df)))
}
cat(sprintf("common - bare logLik %.3e\n", bare_common_ll_diff))
