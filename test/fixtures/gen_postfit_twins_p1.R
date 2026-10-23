## Regenerates postfit_twins_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_postfit_twins_p1.R
##
## Reads the tracked ns_gauss_p1_data.csv (sha256 guarded, written by
## gen_namespace_numeric_p1.R) and refits the same model:
##   value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)
## Recorded: coef(), tidy()$estimate (fixed effects) and deviance(). The trait means are
## 0.5, -0.3, 0.2, 0.8, -0.6, 0.1 (uncentred), so unlike the row-centred batch fixture of the
## postfit rows these values are far from zero and a constant or zero implementation fails.
## The fit must converge with a positive-definite Hessian and reproduce the log-likelihood
## recorded in ns_numeric_p1.toml (asserted below).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
tr <- paste0("t", 1:6)
df1 <- read.csv("ns_gauss_p1_data.csv")
df1$unit <- factor(df1$unit, levels = sort(unique(df1$unit)))
df1$trait <- factor(df1$trait, levels = tr)
fit1 <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
  data = df1, unit = "unit", trait = "trait")))
stopifnot(fit1$opt$convergence == 0L, isTRUE(fit1$sd_report$pdHess))
prev <- grep("^loglik", readLines("ns_numeric_p1.toml"), value = TRUE)[1]
stopifnot(abs(as.numeric(logLik(fit1)) - as.numeric(sub(".*= ", "", prev))) < 1e-6)
td <- tidy(fit1)
stopifnot(identical(as.character(td$term), paste0("trait", tr)))
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
con <- file("postfit_twins_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for postfit rows tidy.gllvmTMB_multi, POST-COEF-NAMED and POST-DEVIANCE")
w("# (gllvmTMB P1). Generated once by gen_postfit_twins_p1.R from ns_gauss_p1_data.csv")
w("# (sha256 guarded). Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[main]")
w("# value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE); seed 20261001 (data of ns_numeric_p1.toml [main])")
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("converged = true")
w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(fit1))))
w("coef_names = [%s]", paste0("\"", names(coef(fit1)), "\"", collapse = ", "))
w("coef = %s", vec(coef(fit1)))
w("tidy_term = [%s]", paste0("\"", td$term, "\"", collapse = ", "))
w("tidy_estimate = %s", vec(td$estimate))
w("deviance = %s", fmt(as.numeric(deviance(fit1))))
close(con)
print(coef(fit1)); print(td); print(deviance(fit1))
