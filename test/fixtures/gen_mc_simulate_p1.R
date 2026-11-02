## Regenerates mc_simulate_p1.toml. NOT run by CI or by any Julia test: provenance for how the
## fixture was produced, against gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9
## (version 0.7.1, "P1"). Needs gllvmTMB installed at that commit; set GLLVM_P1_RLIB to its library.
## Run from test/fixtures/:  OPENBLAS_NUM_THREADS=1 Rscript gen_mc_simulate_p1.R
##
## Monte-Carlo moment twins for two simulation rows (maintainer ruling 2026-10-05, vault D-319, item N4):
##   [unit_trait]        postfit/POSTFIT-SURFACE-simulate_unit_trait
##   [simulate_default]  postfit-policy/POST-SIMULATE-DEFAULT (simulate() with its default arguments)
##
## MONTE-CARLO TOLERANCE RULE (stated 2026-10-05, before any run; the same text is in
## test/test_mc_simulate_p1.jl and in every receipt):
##   * Each engine produces B independent replicates (B fixed below, never changed after a run).
##     From every replicate it computes the same K moment statistics (listed per section).
##   * For moment k, let m_R, m_J be the replicate means and s_R, s_J the replicate standard
##     deviations (n - 1 denominator) over the B_k replicate values (B_k = B, or B / 2 for a
##     statistic built from a pair of replicates). The tolerance is
##         tol_k = z * sqrt(s_R^2 / B_k + s_J^2 / B_k),   z = qnorm(1 - alpha / (2 K)), alpha = 0.01,
##     a two-sided Bonferroni bound over the K moments of the section (familywise alpha 0.01).
##   * The row passes when |m_R - m_J| <= tol_k for every k. No other tolerance is used, and the
##     rule is not loosened after a run: a failure is reported as a failure.
##   * Discrimination control: an alternative that differs in the property the row names must FAIL
##     the same rule on at least one moment, or the twin does not bind. unit_trait: Julia with psi_B
##     doubled. simulate_default: R simulate(..., condition_on_RE = TRUE) (draws given the fitted
##     latent modes) against the Julia default.
##   * Seeds: R uses the seeds written below; Julia uses Random.Xoshiro streams (test file).
##     The engines' random streams are independent, so this is a distributional comparison.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

B <- 400L
ALPHA <- 0.01
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
ivec <- function(x) paste0("[", paste(as.integer(x), collapse = ", "), "]")
upper <- function(S) S[upper.tri(S, diag = TRUE)]   # column-major upper triangle incl. diagonal
summ <- function(M) list(mean = colMeans(M), sd = apply(M, 2, stats::sd), n = nrow(M))

## ---------------------------------------------------------------- [unit_trait]
## Moments per replicate (T = 4 traits, U units, O observations per unit):
##   trait means (T); pooled within-unit covariance, divisor U (O - 1) (T(T+1)/2, upper triangle);
##   covariance of the unit means, divisor U - 1 (T(T+1)/2). K = 4 + 10 + 10 = 24.
U <- 20L; O <- 3L; Tn <- 4L
Lambda_B <- matrix(c(0.8, 0.5, -0.3, 0.2, 0.1, -0.4, 0.6, 0.5), Tn, 2)
Lambda_W <- matrix(c(0.4, -0.2, 0.3, 0.1), Tn, 1)
alpha <- c(0.5, -0.3, 0.2, 0.8)
psi_B <- rep(0.3, Tn); psi_W <- rep(0.3, Tn); sigma2_eps <- 0.4
ut_moments <- function(value) {   # value: long data ordered unit, observation, trait
  Y <- matrix(value, nrow = Tn)   # T x (U O), column = (unit, observation), unit slowest
  unit <- rep(seq_len(U), each = O)
  ybar <- sapply(seq_len(U), function(u) rowMeans(Y[, unit == u, drop = FALSE]))   # T x U
  R <- Y - ybar[, unit]
  c(rowMeans(Y), upper(R %*% t(R) / (U * (O - 1L))), upper(stats::cov(t(ybar))))
}
ut_rows <- t(vapply(seq_len(B), function(b) {
  sim <- simulate_unit_trait(n_units = U, n_obs_per_unit = O, n_traits = Tn, alpha = alpha,
                             Lambda_B = Lambda_B, Lambda_W = Lambda_W, psi_B = psi_B, psi_W = psi_W,
                             sigma2_eps = sigma2_eps, seed = b)
  d <- sim$data
  stopifnot(identical(as.integer(d$trait), rep(seq_len(Tn), U * O)))
  ut_moments(d$value)
}, numeric(24)))
ut <- summ(ut_rows)

## ---------------------------------------------------------- [simulate_default]
## Poisson fit, value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE), on counts
## drawn once below (seed 20261005, p = 5 traits, n = 60 units). simulate(fit, nsim = B, seed)
## with every other argument at its default (condition_on_RE = FALSE: latent scores redrawn).
## Moments per simulated data set Y (p x n): trait means (p); covariance across units, divisor
## n - 1 (p(p+1)/2); and, per pair of data sets (2j - 1, 2j), the within-cell variance
## mean_s (Y1[t, s] - Y2[t, s])^2 / 2 for each trait (p; B / 2 pair replicates). K = 5 + 15 + 5 = 25.
set.seed(20261005)
p <- 5L; n <- 60L
lam <- c(0.9, 0.6, -0.5, 0.4, 0.7); beta <- c(0.8, 0.2, 1.1, -0.2, 0.5)
z <- stats::rnorm(n)
Ycount <- matrix(stats::rpois(p * n, exp(beta + outer(lam, z))), p, n)   # p x n
df <- data.frame(unit = factor(rep(seq_len(n), each = p)), trait = factor(rep(paste0("t", 1:p), n), levels = paste0("t", 1:p)),
                 value = as.vector(Ycount))
fit <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
  data = df, unit = "unit", trait = "trait", family = poisson())))
stopifnot(fit$opt$convergence == 0L, isTRUE(fit$sd_report$pdHess))
stopifnot(identical(as.integer(fit$data$trait), rep(seq_len(p), n)), identical(as.integer(fit$data$unit), rep(seq_len(n), each = p)))
sd_moments <- function(S) {   # S: (p n) x B, rows in df order (trait fastest)
  m1 <- t(apply(S, 2, function(y) { Y <- matrix(y, p, n); c(rowMeans(Y), upper(stats::cov(t(Y)))) }))
  pairs <- seq(1L, ncol(S), by = 2L)
  m2 <- t(vapply(pairs, function(j) rowMeans((matrix(S[, j], p, n) - matrix(S[, j + 1L], p, n))^2) / 2, numeric(p)))
  list(m1 = summ(m1), m2 = summ(m2))
}
SEED_DEFAULT <- 77L; SEED_CONDITIONAL <- 78L
sd_def <- sd_moments(simulate(fit, nsim = B, seed = SEED_DEFAULT))
sd_cond <- sd_moments(simulate(fit, nsim = B, seed = SEED_CONDITIONAL, condition_on_RE = TRUE))
stopifnot(identical(formals(gllvmTMB:::simulate.gllvmTMB_multi)$condition_on_RE, FALSE))

con <- file("mc_simulate_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Monte-Carlo moment twins (D-319 item N4) for simulate_unit_trait and POST-SIMULATE-DEFAULT (gllvmTMB P1).")
w("# Generated once by gen_mc_simulate_p1.R. Do not hand-edit; regenerate from the script.")
w("# The Monte-Carlo tolerance rule is stated in gen_mc_simulate_p1.R and test/test_mc_simulate_p1.jl.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("B = %d", B)
w("alpha = %s", fmt(ALPHA))
w("")
w("[unit_trait]")
w("n_units = %d", U); w("n_obs_per_unit = %d", O); w("n_traits = %d", Tn)
w("Lambda_B = %s  # column-major %d x 2", vec(Lambda_B), Tn)
w("Lambda_W = %s  # %d x 1", vec(Lambda_W), Tn)
w("alpha_t = %s", vec(alpha)); w("psi_B = %s", vec(psi_B)); w("psi_W = %s", vec(psi_W))
w("sigma2_eps = %s", fmt(sigma2_eps))
w("r_seeds = \"1:%d, one simulate_unit_trait(seed = b) call per replicate\"", B)
w("K = 24")
w("z = %s", fmt(stats::qnorm(1 - ALPHA / (2 * 24))))
w("r_mean = %s", vec(ut$mean)); w("r_sd = %s", vec(ut$sd)); w("r_n = %d", ut$n)
w("")
w("[simulate_default]")
w("p = %d", p); w("n = %d", n)
w("y = %s  # p x n counts, column-major (trait fastest)", ivec(Ycount))
w("loglik = %s", fmt(as.numeric(logLik(fit))))
w("converged = true"); w("pd_hessian = true")
w("r_seed_default = %d", SEED_DEFAULT); w("r_seed_conditional = %d", SEED_CONDITIONAL)
w("K = 25")
w("z = %s", fmt(stats::qnorm(1 - ALPHA / (2 * 25))))
w("r_mean_m1 = %s", vec(sd_def$m1$mean)); w("r_sd_m1 = %s", vec(sd_def$m1$sd)); w("r_n_m1 = %d", sd_def$m1$n)
w("r_mean_m2 = %s", vec(sd_def$m2$mean)); w("r_sd_m2 = %s", vec(sd_def$m2$sd)); w("r_n_m2 = %d", sd_def$m2$n)
w("cond_mean_m1 = %s", vec(sd_cond$m1$mean)); w("cond_sd_m1 = %s", vec(sd_cond$m1$sd))
w("cond_mean_m2 = %s", vec(sd_cond$m2$mean)); w("cond_sd_m2 = %s", vec(sd_cond$m2$sd))
close(con)
cat("wrote mc_simulate_p1.toml; logLik", as.numeric(logLik(fit)), "\n")
