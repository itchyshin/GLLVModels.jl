## Regenerates ns_gauss_p1_data.csv, ns_lv_p1_data.csv and ns_numeric_p1.toml. NOT run
## by CI or by any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9
## (version 0.7.1, "P1"). Needs gllvmTMB installed at that exact commit in a
## lane-local library (git worktree at the pin, then
## `R CMD INSTALL --library=<lib> <dir>`); set GLLVM_P1_RLIB to that library.
## Run from test/fixtures/:  Rscript gen_namespace_numeric_p1.R
##
## Two Gaussian models, both Sigma = Lambda Lambda^T + sigma_eps^2 I (the structure of
## GLLVModels.jl's fit_gllvm(Y; family = Normal(), K = d)), p = 6 traits, n = 200 units:
##   main: value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)
##         recorded: vcov() (fixed-effect block), extract_loadings(rotate = "none"),
##         extract_rotated_loadings_table(method = "varimax", raw and standardized)
##   lv  : same plus latent(..., lv = ~ x1 + x2), predictor-informed latent scores
##         recorded: extract_lv_effects(type = "trait_effect" and "axis_effect")
##   two : two-level Gaussian, p = 5 traits, 120 units x 4 observations, per tier
##         latent(d = 1) + unique: value ~ 0 + trait + latent(0 + trait | unit, d = 1) +
##         unique(0 + trait | unit) + latent(0 + trait | obs, d = 1) + unique(0 + trait | obs)
##         i.e. GLLVModels.jl's fit_twolevel_gaussian(y, individual; K_B = 1, K_W = 1).
##         recorded: extract_communality (unit, unit_obs), extract_Sigma_B / extract_Sigma_W
## Every fit must converge with a positive-definite Hessian (asserted below).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 6L; n <- 200L
tr <- paste0("t", seq_len(p))
Lam <- matrix(c(1.0, 0, 0.6, 0.8, -0.5, 0.7,
                0, 0.9, -0.7, 0.5, 0.8, -0.4), p, 2)
mu <- c(0.5, -0.3, 0.2, 0.8, -0.6, 0.1)
long <- function(Y, extra = NULL) {
  df <- data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                   trait = factor(rep(tr, n), levels = tr),
                   value = as.vector(t(Y)))
  if (!is.null(extra)) df <- cbind(df, extra[as.integer(df$unit), , drop = FALSE])
  df
}

## ---- main dataset (seed 20261001) ----
set.seed(20261001L)
Z <- matrix(rnorm(n * 2), n, 2)
Y <- sweep(Z %*% t(Lam), 2, mu, "+") + matrix(rnorm(n * p, sd = 0.5), n, p)
df1 <- long(Y)
write.csv(df1, "ns_gauss_p1_data.csv", row.names = FALSE)

## ---- predictor-informed dataset (seed 20261002) ----
set.seed(20261002L)
X <- cbind(x1 = rnorm(n), x2 = rnorm(n))
alpha <- matrix(c(0.8, -0.5, 0.3, 0.7), 2, 2)       # q_lv x K
Z2 <- X %*% alpha + matrix(rnorm(n * 2), n, 2)
Y2 <- sweep(Z2 %*% t(Lam), 2, mu, "+") + matrix(rnorm(n * p, sd = 0.5), n, p)
df2 <- long(Y2, X)
write.csv(df2, "ns_lv_p1_data.csv", row.names = FALSE)

fit1 <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
                 data = df1, unit = "unit", trait = "trait")
fit2 <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 2, lv = ~ x1 + x2, unique = FALSE),
                 data = df2, unit = "unit", trait = "trait")
## ---- two-level dataset (seed 20261003) ----
set.seed(20261003L)
p3 <- 5L; L3 <- 120L; m3 <- 4L
tr3 <- paste0("t", seq_len(p3))
LB <- matrix(c(1.0, 0.7, -0.5, 0.6, 0.3), p3, 1); LW <- matrix(c(0.8, -0.6, 0.4, 0.5, -0.7), p3, 1)
sB <- c(0.3, 0.4, 0.25, 0.35, 0.3); sW <- c(0.4, 0.3, 0.5, 0.35, 0.45)
rows <- vector("list", L3 * m3); k <- 0L
for (i in seq_len(L3)) {
  zb <- rnorm(1); ub <- rnorm(p3, sd = sqrt(sB))
  for (r in seq_len(m3)) {
    k <- k + 1L; zw <- rnorm(1); uw <- rnorm(p3, sd = sqrt(sW))
    y <- c(0.2, -0.1, 0.4, 0, -0.3) + LB[, 1] * zb + ub + LW[, 1] * zw + uw
    rows[[k]] <- data.frame(unit = i, obs = k, trait = tr3, value = y)
  }
}
df3 <- do.call(rbind, rows)
df3$unit <- factor(df3$unit); df3$obs <- factor(df3$obs); df3$trait <- factor(df3$trait, levels = tr3)
write.csv(df3, "ns_twolevel_p1_data.csv", row.names = FALSE)
fit3 <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 1) + unique(0 + trait | unit) +
    latent(0 + trait | obs, d = 1) + unique(0 + trait | obs),
  data = df3, unit = "unit", unit_obs = "obs", trait = "trait")))

## Reproducer-only fit (not bound): same data as main, latent + unique at one tier.
fit1u <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit),
  data = df1, unit = "unit", trait = "trait")))
for (f in list(fit1, fit2, fit3, fit1u)) stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))

V1 <- vcov(fit1)
L1 <- suppressMessages(extract_loadings(fit1, level = "unit", rotate = "none"))
tab_raw <- suppressMessages(extract_rotated_loadings_table(fit1, level = "unit", method = "varimax",
                                                           loading_scale = "raw"))
tab_std <- suppressMessages(extract_rotated_loadings_table(fit1, level = "unit", method = "varimax",
                                                           loading_scale = "standardized"))
for (tb in list(tab_raw, tab_std))
  stopifnot(identical(as.character(tb$trait), rep(tr, 2)), identical(tb$axis, rep(c("LV1", "LV2"), each = p)))
ef_trait <- extract_lv_effects(fit2, level = "unit", type = "trait_effect")
ef_axis  <- extract_lv_effects(fit2, level = "unit", type = "axis_effect")
stopifnot(identical(unname(as.character(ef_trait$trait)), rep(tr, 2)),
          identical(unname(as.character(ef_trait$predictor)), rep(c("x1", "x2"), each = p)))

cm_B <- extract_communality(fit3, level = "unit")
cm_W <- extract_communality(fit3, level = "unit_obs")
sg_B <- suppressMessages(extract_Sigma_B(fit3))$Sigma_B
sg_W <- suppressMessages(extract_Sigma_W(fit3))$Sigma_W
stopifnot(identical(names(cm_B), tr3), nrow(sg_B) == p3, nrow(sg_W) == p3)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) {
  out <- system2("shasum", c("-a", "256", f), stdout = TRUE); strsplit(out, " ")[[1]][1]
}
con <- file("ns_numeric_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the namespace numeric rows vcov.gllvmTMB_multi, extract_loadings and")
w("# extract_lv_effects (gllvmTMB P1). Generated once by gen_namespace_numeric_p1.R from the two")
w("# CSVs (sha256 guarded below). Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("trait_names = [%s]", paste0("\"", tr, "\"", collapse = ", "))
w("")
w("[main]")
w("# value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE); seed 20261001")
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit1))))
w("beta = %s", vec(coef(fit1)))
w("# vcov(fit): fixed-effect (trait mean) covariance, row-major 6 x 6")
w("vcov = %s", vec(t(V1)))
w("# extract_loadings(fit, level = \"unit\", rotate = \"none\"): 6 x 2, row-major")
w("loadings = %s", vec(t(L1)))
w("# extract_rotated_loadings_table(fit, level = \"unit\", method = \"varimax\", order_axes = TRUE,")
w("#   sign_anchor = \"auto\"): loading column in table order (axis-major: 6 traits for LV1, then LV2)")
w("rot_raw_loading = %s", vec(tab_raw$loading))
w("rot_raw_axis_variance = %s", vec(tab_raw$axis_variance[c(1, p + 1)]))
w("rot_raw_axis_share = %s", vec(tab_raw$axis_share[c(1, p + 1)]))
w("rot_std_loading = %s", vec(tab_std$loading))
w("")
w("[lv]")
w("# ... + latent(0 + trait | unit, d = 2, lv = ~ x1 + x2, unique = FALSE); seed 20261002")
w("data_file = \"ns_lv_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_lv_p1_data.csv"))
w("predictors = [\"x1\", \"x2\"]")
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit2))))
w("# extract_lv_effects(type = \"trait_effect\"): B_lv = Lambda alpha', 6 x 2 (trait x predictor), row-major")
B <- matrix(ef_trait$estimate, nrow = p, ncol = 2)
w("trait_effect = %s", vec(t(B)))
w("# extract_lv_effects(type = \"axis_effect\"): alpha, 2 x 2 (predictor x axis), row-major")
A <- matrix(ef_axis$estimate, nrow = 2, ncol = 2)
w("axis_effect = %s", vec(t(A)))
w("")
w("[two]")
w("# two-level model above; seed 20261003; obs ids are 1..480 in order, unit = individual")
w("data_file = \"ns_twolevel_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_twolevel_p1_data.csv"))
w("p = %d", p3); w("n_unit = %d", L3); w("n_obs = %d", L3 * m3)
w("trait_names = [%s]", paste0("\"", tr3, "\"", collapse = ", "))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit3))))
w("# extract_communality(fit, level = \"unit\") and level = \"unit_obs\"")
w("communality_unit = %s", vec(cm_B))
w("communality_unit_obs = %s", vec(cm_W))
w("# extract_Sigma_B(fit)$Sigma_B and extract_Sigma_W(fit)$Sigma_W, 5 x 5, row-major")
w("sigma_unit = %s", vec(t(sg_B)))
w("sigma_unit_obs = %s", vec(t(sg_W)))
w("")
w("[gap_unique]")
w("# NOT a bound twin; reproducer data for repro_namespace_twin_gaps_p1.jl. Same data as [main],")
w("# value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit).")
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit1u))))
w("communality_unit = %s", vec(extract_communality(fit1u, level = "unit")))
w("sigma_eps = %s", fmt(fit1u$report$sigma_eps))
close(con)
print(cm_B); print(cm_W)
print(tab_raw); print(tab_std); print(V1); print(L1); print(B); print(A)
