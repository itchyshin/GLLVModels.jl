## Regenerates predict_missing_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_predict_missing_p1.R
##
## Reads the tracked ns_gauss_p1_data.csv (sha256 guarded, written by gen_namespace_numeric_p1.R),
## sets a fixed set of cells to NA (the mask is defined here and recorded in the fixture; Julia
## reads it back, it does not recompute it) and fits
##   value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)
## with missing = miss_control(response = "include"), so the NA cells stay in the likelihood as
## masked cells. Recorded: predict_missing(fit)$est (link and response scale) at every masked
## cell, with the unit and trait index of each cell, and the log-likelihood of the masked fit.
## The fit must converge with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
tr <- paste0("t", 1:6)
df <- read.csv("ns_gauss_p1_data.csv")
df$unit <- factor(df$unit, levels = sort(unique(df$unit)))
df$trait <- factor(df$trait, levels = tr)
ui <- as.integer(df$unit); ti <- as.integer(df$trait)
## traits 2 and 5 masked in every unit u with u %% 10 == 3; trait 1 masked in every unit u with u %% 7 == 4
masked <- (ui %% 10L == 3L & ti %in% c(2L, 5L)) | (ui %% 7L == 4L & ti == 1L)
df$value[masked] <- NA
fit <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
  data = df, unit = "unit", trait = "trait",
  missing = miss_control(response = "include"))))
stopifnot(fit$opt$convergence == 0L, isTRUE(fit$sd_report$pdHess))
pl <- predict_missing(fit, type = "link")
pr <- predict_missing(fit, type = "response")
stopifnot(nrow(pl) == sum(masked), nrow(pr) == nrow(pl), identical(pl$model_row, pr$model_row),
          identical(pl$model_row, which(masked)))
## the pre-existing (no-mask) fit does not enter; this is the masked fit's own log-likelihood
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
ivec <- function(x) paste0("[", paste(as.integer(x), collapse = ", "), "]")
con <- file("predict_missing_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for postfit row POSTFIT-SURFACE-predict_missing (gllvmTMB P1). Generated once by")
w("# gen_predict_missing_p1.R from ns_gauss_p1_data.csv (sha256 guarded). Do not hand-edit;")
w("# regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[main]")
w("# value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE), miss_control(response = \"include\")")
w("# data of ns_numeric_p1.toml [main] (seed 20261001) with the cells below set to NA")
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("converged = true")
w("pd_hessian = true")
w("n_masked = %d", sum(masked))
w("loglik = %s", fmt(as.numeric(logLik(fit))))
w("# masked cells in predict_missing() row order (model_row ascending = unit, then trait)")
w("masked_unit = %s", ivec(as.character(pl$unit)))
w("masked_trait = %s", ivec(match(as.character(pl$trait), tr)))
w("est_link = %s", vec(pl$est))
w("est_response = %s", vec(pr$est))
close(con)
print(head(pl)); print(summary(pl$est)); print(logLik(fit))
