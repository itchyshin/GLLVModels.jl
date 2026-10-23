#!/usr/bin/env Rscript
# gllvm-parity-tag: P1
#
# Generates test/fixtures/gllvmtmb_anova_fixture.json, the R oracle for
# test/test_model_comparison.jl. The test reads a TOML copy
# (gllvmtmb_anova_fixture.toml), produced from this JSON by
#   julia --project=test test/fixtures/convert_gllvmtmb_anova_fixture_to_toml.jl
# which checks every value round-trips exactly; the JSON itself is not
# committed. Oracle for
# test/test_model_comparison.jl (twin of gllvmTMB's AIC.gllvmTMB_multi,
# BIC.gllvmTMB_multi, anova.gllvmTMB_multi at pin
# 9539352f66f2db2cc26b1c393e67212a359b60c9, "P1").
#
# Fits three nested Gaussian long-format gllvmTMB models that differ ONLY in
# the latent rank d = 1, 2, 3 (a pure rank-step sequence, the case
# anova.gllvmTMB_multi's boundary/chi-bar-square branch is for), records
# logLik, its df/nobs attributes, AIC, BIC, and the full anova() comparison
# table (both test = "chibar" and test = "chisq"), and writes them to JSON.
#
# Usage (run once, from a checkout of gllvmTMB at the P1 pin, with a
# gllvmTMB build installed into a library that also has its ordinary
# dependencies -- assertthat, TMB, etc. -- available; set GLLVMTMB_RLIB to a
# library holding just gllvmTMB itself and it is PREPENDED to the existing
# .libPaths(), never replacing it, so those dependencies keep resolving):
#   GLLVMTMB_RLIB=/path/to/temp/lib Rscript test/fixtures/generate_gllvmtmb_anova_fixture.R
#
# The committed TOML fixture's sha256 is asserted by test/test_model_comparison.jl
# so a silent drift (re-running this script with a different gllvmTMB build,
# a different R version, or a different seed) is caught rather than
# committed unnoticed.

extra_lib <- Sys.getenv("GLLVMTMB_RLIB", unset = "")
if (nzchar(extra_lib)) {
  .libPaths(c(extra_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(gllvmTMB)
  library(jsonlite)
})

set.seed(20260927L)
n_site  <- 25L
p_trait <- 4L
d_max   <- 3L

beta   <- c(-0.4, 0.3, 0.6, -0.1)
Lambda <- matrix(c( 0.9, -0.5,  0.3,
                    -0.6,  0.7,  0.2,
                     0.4,  0.4, -0.6,
                     0.2, -0.3,  0.5),
                  nrow = p_trait, ncol = d_max, byrow = TRUE)
u <- matrix(rnorm(n_site * d_max), nrow = n_site, ncol = d_max)

site  <- factor(rep(paste0("s", seq_len(n_site)), each = p_trait),
                levels = paste0("s", seq_len(n_site)))
trait <- factor(rep(paste0("t", seq_len(p_trait)), n_site),
                levels = paste0("t", seq_len(p_trait)))

eta <- numeric(n_site * p_trait)
for (i in seq_len(n_site)) {
  for (t in seq_len(p_trait)) {
    idx <- (i - 1L) * p_trait + t
    eta[idx] <- beta[t] + sum(Lambda[t, ] * u[i, ])
  }
}
y <- eta + rnorm(length(eta), sd = 0.25)
dat <- data.frame(site = site, trait = trait, y = y)

ctrl <- gllvmTMBcontrol(n_init = 1, init_jitter = 0, se = FALSE, aghq_ridge = Inf)

fit_at_d <- function(d) {
  fml <- as.formula(sprintf("y ~ 0 + trait + latent(0 + trait | site, d = %d, unique = FALSE)", d))
  gllvmTMB(fml, data = dat, family = gaussian(), control = ctrl,
           unit = "site", silent = TRUE, estimator = "ml")
}

fits <- lapply(seq_len(d_max), fit_at_d)

per_model <- lapply(fits, function(f) {
  ll <- stats::logLik(f)
  list(
    d       = f$d_B,
    npar    = attr(ll, "df"),
    nobs    = stats::nobs(f),
    loglik  = as.numeric(ll),
    aic     = stats::AIC(f),
    bic     = stats::BIC(f)
  )
})

anova_to_list <- function(tab) {
  df <- as.data.frame(tab)
  list(
    model    = df$model,
    d        = df$d,
    npar     = df$npar,
    logLik   = df$logLik,
    deviance = df$deviance,
    df       = df$df,
    LRT      = df$LRT,
    test     = df$test,
    pvalue   = df$p.value
  )
}

anova_chibar <- anova_to_list(anova(fits[[1]], fits[[2]], fits[[3]], test = "chibar"))
anova_chisq  <- anova_to_list(anova(fits[[1]], fits[[2]], fits[[3]], test = "chisq"))

## Wide (p_trait x n_site) response matrix, species-major -- the same Y
## layout fit_gaussian_gllvm(Y; K) expects on the Julia side -- so the
## end-to-end twin test can refit the IDENTICAL data instead of only
## checking the table-construction arithmetic against recorded numbers.
Y_wide <- matrix(NA_real_, nrow = p_trait, ncol = n_site)
for (i in seq_len(n_site)) {
  for (t in seq_len(p_trait)) {
    Y_wide[t, i] <- dat$y[dat$site == levels(site)[i] & dat$trait == levels(trait)[t]]
  }
}

out <- list(
  meta = list(
    gllvmtmb_pin = "9539352f66f2db2cc26b1c393e67212a359b60c9",
    r_version = R.version.string,
    gllvmtmb_version = as.character(utils::packageVersion("gllvmTMB")),
    seed = 20260927L,
    n_site = n_site,
    p_trait = p_trait,
    d_values = seq_len(d_max)
  ),
  Y_wide = Y_wide,
  per_model = per_model,
  anova_chibar = anova_chibar,
  anova_chisq = anova_chisq
)

this_file <- sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
out_dir <- if (length(this_file)) dirname(this_file) else "."
out_path <- file.path(out_dir, "gllvmtmb_anova_fixture.json")

write_json(out, out_path, auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote", out_path, "\n")
sha_tool <- if (requireNamespace("digest", quietly = TRUE)) {
  digest::digest(file = out_path, algo = "sha256")
} else {
  # No `digest` dependency assumed: shell out to the standard sha256 tool
  # (present on both the macOS and Linux CI images) as a fallback.
  toupper(strsplit(system2("shasum", c("-a", "256", out_path), stdout = TRUE), " ")[[1]][1])
}
cat("sha256:", sha_tool, "\n")
