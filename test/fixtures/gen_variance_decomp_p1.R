## Regenerates variance_decomp_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_variance_decomp_p1.R
##
## Rebuilds the two datasets of gen_namespace_numeric_p1.R in memory (same seeds and code), checks
## that written to CSV they are byte-identical to the tracked CSVs (sha256 guarded), fits on the
## in-memory data (as gen_namespace_numeric_p1.R does; the two-level fit from the CSV-read data
## reports nlminb code 1, "false convergence (8)", gradient ~2e-3, same log-likelihood, so the
## in-memory fit is the one whose convergence is clean) and records
##   unique : value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit)
##            on ns_gauss_p1_data.csv (p = 6, 200 units, one observation per cell).
##            Recorded: extract_proportions(fit) rows with component "shared_unit".
##   two    : the two-level fit of gen_namespace_numeric_p1.R (p = 5, 120 units x 4 observations),
##            value ~ 0 + trait + latent(0 + trait | unit, d = 1) + unique(0 + trait | unit) +
##              latent(0 + trait | obs, d = 1) + unique(0 + trait | obs)
##            Recorded: extract_proportions(fit) (long format: component, variance, proportion, in
##            R's row order) and extract_residual_split(fit) (sigma2_d, sigma2_e, sigma2_total).
## Every fit must converge with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")

## ---- datasets (copied from gen_namespace_numeric_p1.R) ----
csv_same <- function(df, f) {
  tmp <- tempfile(fileext = ".csv"); write.csv(df, tmp, row.names = FALSE)
  stopifnot(identical(shaw(tmp), shaw(f)))
}
p <- 6L; n <- 200L
tr1 <- paste0("t", seq_len(p))
Lam <- matrix(c(1.0, 0, 0.6, 0.8, -0.5, 0.7,
                0, 0.9, -0.7, 0.5, 0.8, -0.4), p, 2)
mu <- c(0.5, -0.3, 0.2, 0.8, -0.6, 0.1)
set.seed(20261001L)
Z <- matrix(rnorm(n * 2), n, 2)
Y <- sweep(Z %*% t(Lam), 2, mu, "+") + matrix(rnorm(n * p, sd = 0.5), n, p)
d1 <- data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                 trait = factor(rep(tr1, n), levels = tr1), value = as.vector(t(Y)))
csv_same(d1, "ns_gauss_p1_data.csv")
set.seed(20261003L)   # the two-level dataset is seeded afresh in gen_namespace_numeric_p1.R
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
d3 <- do.call(rbind, rows)
d3$unit <- factor(d3$unit); d3$obs <- factor(d3$obs); d3$trait <- factor(d3$trait, levels = tr3)
csv_same(d3, "ns_twolevel_p1_data.csv")

## ---- unique: one tier, latent + unique ----
f1 <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit),
  data = d1, unit = "unit", trait = "trait")))
stopifnot(f1$opt$convergence == 0L, isTRUE(f1$sd_report$pdHess))
p1 <- extract_proportions(f1)
stopifnot(identical(levels(p1$component), c("shared_unit", "unique_unit")),
          identical(as.character(p1$trait[p1$component == "shared_unit"]), tr1))
sh1 <- p1$proportion[p1$component == "shared_unit"]

## ---- two: two-level ----
f3 <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 1) + unique(0 + trait | unit) +
    latent(0 + trait | obs, d = 1) + unique(0 + trait | obs),
  data = d3, unit = "unit", unit_obs = "obs", trait = "trait")))
stopifnot(f3$opt$convergence == 0L, isTRUE(f3$sd_report$pdHess))
p3 <- extract_proportions(f3)
comp <- c("shared_unit", "unique_unit", "shared_unit_obs", "unique_unit_obs")
stopifnot(identical(levels(p3$component), comp), nrow(p3) == 4L * 5L,
          identical(as.character(p3$component), rep(comp, each = 5L)),
          identical(as.character(p3$trait), rep(tr3, 4L)))
rs <- extract_residual_split(f3)
stopifnot(identical(as.character(rs$trait), tr3))

con <- file("variance_decomp_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for postfit row POSTFIT-SURFACE-extract_proportions and namespace rows")
w("# extract_proportions / extract_residual_split (gllvmTMB P1). Generated once by")
w("# gen_variance_decomp_p1.R from two tracked CSVs (sha256 guarded). Do not hand-edit;")
w("# regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[unique]")
w("# value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit); data of ns_numeric_p1.toml [main]")
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("p = 6")
w("n_unit = 200")
w("trait_names = %s", svec(tr1))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(f1))))
w("# extract_proportions(fit): proportion column of the rows with component = \"shared_unit\"")
w("shared_unit_proportion = %s", vec(sh1))
w("")
w("[two]")
w("# two-level model above; data of ns_numeric_p1.toml [two] (seed 20261003)")
w("data_file = \"ns_twolevel_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_twolevel_p1_data.csv"))
w("p = 5")
w("n_unit = 120")
w("n_obs = 480")
w("trait_names = %s", svec(tr3))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(f3))))
w("# extract_proportions(fit), long format: components in R's order, 5 traits each (component-major)")
w("components = %s", svec(comp))
w("variance = %s", vec(p3$variance))
w("proportion = %s", vec(p3$proportion))
w("# extract_residual_split(fit), one value per trait")
w("sigma2_d = %s", vec(rs$sigma2_d))
w("sigma2_e = %s", vec(rs$sigma2_e))
w("sigma2_total = %s", vec(rs$sigma2_total))
close(con)
print(p3); print(rs); print(sh1)
