## Regenerates ns_beta_p1_data.csv, ns_nb2_p1_data.csv and ns_numeric_p1_b.toml. NOT run
## by CI or by any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9
## (version 0.7.1, "P1"). Needs gllvmTMB installed at that exact commit in a
## lane-local library; set GLLVM_P1_RLIB to that library. Needs ns_gauss_p1_data.csv from
## gen_namespace_numeric_p1.R (read, not regenerated).
## Run from test/fixtures/:  Rscript gen_namespace_numeric_p1_b.R
##
##   tidy : tidy(fit, effects = "fixed") on the [main] fit of ns_numeric_p1.toml
##          (value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE), refitted
##          here from ns_gauss_p1_data.csv): estimate and std.error.
##   beta : Beta() family, p = 6, n = 200, latent(d = 1, unique = FALSE), seed 20261101.
##   nb2  : nbinom2() family, same design and seed stream (data drawn after the Beta data).
##          recorded: logLik, trait intercepts, Lambda Lambda^T, per-trait dispersion.
## Every fit must converge with a positive-definite Hessian (asserted below).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 6L; n <- 200L; tr <- paste0("t", seq_len(p))
long <- function(Y) data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                               trait = factor(rep(tr, n), levels = tr), value = as.vector(t(Y)))

## ---- tidy: refit the [main] model ----
df1 <- read.csv("ns_gauss_p1_data.csv")
df1$unit <- factor(df1$unit, levels = seq_len(n)); df1$trait <- factor(df1$trait, levels = tr)
fit1 <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
                 data = df1, unit = "unit", trait = "trait")
stopifnot(fit1$opt$convergence == 0L, isTRUE(fit1$sd_report$pdHess))
td <- tidy(fit1, effects = "fixed")
stopifnot(nrow(td) == p, identical(names(td)[1:3], c("term", "estimate", "std.error")))

## ---- Beta and NB2 data ----
lam <- c(0.9, -0.6, 0.7, 0.5, -0.8, 0.4); mu <- c(0.2, -0.3, 0.4, 0, 0.3, -0.2)
set.seed(20261101L)
z <- rnorm(n)
eta <- outer(z, lam) + matrix(mu, n, p, byrow = TRUE)
phi <- 8
Yb <- matrix(rbeta(n * p, plogis(eta) * phi, (1 - plogis(eta)) * phi), n, p)
Yn <- matrix(rnbinom(n * p, mu = exp(eta + 1), size = 3), n, p)
dfb <- long(Yb); dfn <- long(Yn)
write.csv(dfb, "ns_beta_p1_data.csv", row.names = FALSE)
write.csv(dfn, "ns_nb2_p1_data.csv", row.names = FALSE)
fb <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
               data = dfb, unit = "unit", trait = "trait", family = Beta())
fn <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
               data = dfn, unit = "unit", trait = "trait", family = nbinom2())
for (f in list(fb, fn)) stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
Lb <- suppressMessages(extract_loadings(fb, level = "unit", rotate = "none"))
Ln <- suppressMessages(extract_loadings(fn, level = "unit", rotate = "none"))
phib <- as.numeric(fb$report$phi_beta); phin <- as.numeric(exp(fn$opt$par[names(fn$opt$par) == "log_phi_nbinom2"]))
stopifnot(length(phin) == p)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("ns_numeric_p1_b.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the namespace rows tidy.gllvmTMB_multi, Beta and nbinom2 (gllvmTMB P1).")
w("# Generated once by gen_namespace_numeric_p1_b.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("trait_names = [%s]", paste0("\"", tr, "\"", collapse = ", "))
w("")
w("[tidy]")
w("# tidy(fit, effects = \"fixed\") on the ns_numeric_p1.toml [main] fit; data ns_gauss_p1_data.csv")
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit1))))
w("terms = [%s]", paste0("\"", td$term, "\"", collapse = ", "))
w("estimate = %s", vec(td$estimate))
w("std_error = %s", vec(td$std.error))
w("link = [%s]", paste0("\"", td$link, "\"", collapse = ", "))
for (s in list(list("beta", "ns_beta_p1_data.csv", fb, Lb, phib, "Beta()"),
               list("nb2", "ns_nb2_p1_data.csv", fn, Ln, phin, "nbinom2()"))) {
  w("")
  w("[%s]", s[[1]])
  w("# value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE), family = %s; seed 20261101", s[[6]])
  w("data_file = \"%s\"", s[[2]])
  w("data_sha256 = \"%s\"", shaw(s[[2]]))
  w("converged = true")
  w("pd_hessian = true")
  w("loglik = %s", fmt(as.numeric(logLik(s[[3]]))))
  w("beta = %s", vec(coef(s[[3]])))
  w("# Lambda Lambda^T from extract_loadings(rotate = \"none\"), 6 x 6 row-major (sign of a K = 1 axis is not identified)")
  w("lambda_lambdat = %s", vec(t(tcrossprod(s[[4]]))))
  w("# per-trait dispersion phi (Beta: fit$report$phi_beta; NB2: exp(log_phi_nbinom2)), var = mu + mu^2/phi for NB2")
  w("phi = %s", vec(s[[5]]))
}
close(con)
print(td); print(logLik(fb)); print(logLik(fn)); print(phib); print(phin)
