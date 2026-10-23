## Regenerates confint_inspect_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_confint_inspect_p1.R
##
## Twin of the namespace row export/confint_inspect. It reuses the [main] dataset of the namespace
## numeric twins (ns_gauss_p1_data.csv, written by gen_namespace_numeric_p1.R; sha256 checked against
## ns_numeric_p1.toml) and refits the same model:
##   value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE), Gaussian, p = 6, n = 200
## (GLLVModels.jl's fit_gllvm(Y; family = Normal(), K = 2)). It then calls the public
## confint_inspect(fit, parm, level = 0.95, ystep = 0.02) on four direct profile targets of
## different kinds and records $bounds (estimate, profile bounds, Wald bounds, natural scale) and
## $diagnostics:
##   sigma_eps            log_sigma_eps, transformation exp
##   b_fix[1]             trait-1 mean, linear_predictor
##   Lambda_B_packed[2]   theta_rr_B[2] = Lambda[2,2] (diagonal), lambda_packed (identity)
##   Lambda_B_packed[3]   theta_rr_B[3] = Lambda[2,1] (off-diagonal), lambda_packed (identity)
## The packed order (the d diagonal entries, then the strict lower triangle column by column) is
## asserted below against extract_loadings(rotate = "none").
##
## Why ystep = 0.02 and not the default 0.5: confint_inspect() reads the profile bound off the
## TMB::tmbprofile() grid by linear interpolation (.profile_bounds()), so with the default grid the
## bound carries an interpolation error (measured here: 1.7e-6 for sigma_eps, 2.9e-6 for b_fix[1],
## 2.2e-5 for Lambda[2,2]). ystep is a public argument; 0.02 shrinks that error ~600-fold, so the
## recorded bound is the profile crossing itself rather than an artefact of the grid. The default
## (ystep = 0.5) bounds are recorded too, for disclosure; they are not compared.
## The fit must converge (code 0) with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")

p <- 6L; n <- 200L
tr <- paste0("t", seq_len(p))
data_file <- "ns_gauss_p1_data.csv"
ns_toml <- readLines("ns_numeric_p1.toml")
main_sha <- sub('^data_sha256 = "([0-9a-f]+)"$', "\\1",
                grep('^data_sha256 = ', ns_toml, value = TRUE)[1])
stopifnot(identical(shaw(data_file), main_sha))
df1 <- read.csv(data_file)
df1$unit <- factor(df1$unit, levels = seq_len(n))
df1$trait <- factor(df1$trait, levels = tr)
fit1 <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
                 data = df1, unit = "unit", trait = "trait")
stopifnot(fit1$opt$convergence == 0L, isTRUE(fit1$sd_report$pdHess))

## packed order check: theta_rr_B[2] = Lambda[2,2], theta_rr_B[3] = Lambda[2,1]
L1 <- suppressMessages(extract_loadings(fit1, level = "unit", rotate = "none"))
th <- unname(fit1$opt$par[names(fit1$opt$par) == "theta_rr_B"])
stopifnot(length(th) == 11L, th[1] == L1[1, 1], th[2] == L1[2, 2], th[3] == L1[2, 1])

targets <- c("sigma_eps", "b_fix[1]", "Lambda_B_packed[2]", "Lambda_B_packed[3]")
keys <- c("sigma_eps", "b_fix_1", "lambda_packed_2", "lambda_packed_3")
pt <- profile_targets(fit1, ready_only = FALSE)
YSTEP <- 0.02
res <- lapply(targets, function(pm) {
  stopifnot(isTRUE(pt$profile_ready[pt$parm == pm]))
  fine <- confint_inspect(fit1, parm = pm, level = 0.95, ystep = YSTEP)
  dflt <- confint_inspect(fit1, parm = pm, level = 0.95)
  b <- fine$bounds
  stopifnot(nrow(b) == 1L, all(is.finite(unlist(b[, c("lower_natural", "upper_natural",
                                                      "wald_lower_natural", "wald_upper_natural")]))))
  list(pm = pm, b = b, diag = fine$diagnostics, n_curve = nrow(fine$curve),
       transformation = pt$transformation[pt$parm == pm], dflt = dflt$bounds)
})

con <- file("confint_inspect_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for namespace row export/confint_inspect (gllvmTMB P1). Generated once by")
w("# gen_confint_inspect_p1.R from ns_gauss_p1_data.csv (sha256 recorded). Do not hand-edit;")
w("# regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[main]")
w("formula = \"value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)\"")
w("data_file = \"%s\"", data_file)
w("data_sha256 = \"%s\"", shaw(data_file))
w("p = %d", p); w("n_unit = %d", n)
w("trait_names = %s", svec(tr))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit1))))
w("level = 0.95")
w("ystep = %s", fmt(YSTEP))
w("targets = %s", svec(targets))
for (i in seq_along(res)) {
  r <- res[[i]]; b <- r$b
  w("")
  w("[inspect.%s]", keys[i])
  w("# confint_inspect(fit, parm = \"%s\", level = 0.95, ystep = %s)$bounds, natural scale", r$pm, fmt(YSTEP))
  w("parm = \"%s\"", r$pm)
  w("transformation = \"%s\"", r$transformation)
  w("estimate = %s", fmt(b$estimate_natural))
  w("profile = %s", vec(c(b$lower_natural, b$upper_natural)))
  w("wald = %s", vec(c(b$wald_lower_natural, b$wald_upper_natural)))
  w("wald_profile_disagree = [%s, %s]", tolower(b$wald_profile_disagree_lower),
    tolower(b$wald_profile_disagree_upper))
  w("diagnostics = %s", svec(r$diag))
  w("n_curve = %d", r$n_curve)
  w("# NOT compared (disclosure): profile bounds at the default ystep = 0.5")
  w("profile_default_ystep = %s", vec(c(r$dflt$lower_natural, r$dflt$upper_natural)))
}
close(con)
for (r in res) { print(r$b, digits = 12); print(r$diag) }
