## Regenerates select_lv_p1_data.csv and select_lv_p1.toml. NOT run by CI or by
## any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9
## (version 0.7.1, "P1"). Needs gllvmTMB installed at that exact commit in a
## lane-local library (git worktree at the pin, then
## `R CMD INSTALL --library=<lib> <dir>`); set GLLVM_P1_RLIB to that library.
## Run from test/fixtures/:  Rscript gen_select_lv_p1.R
##
## Model: Gaussian, value ~ 0 + trait + latent(0 + trait | unit, d = k,
## unique = FALSE), i.e. Sigma = Lambda Lambda^T + sigma_eps^2 I, the structure
## of GLLVModels.jl's fit_gllvm(Y; family = Normal(), K = k). select_lv() at
## P1, criterion = "bic" (R's default), d_max = 3. True rank 2 (p = 6,
## n = 150); every candidate rank converges with a positive-definite Hessian
## (asserted below), so the sweep stays away from the non-PD-Hessian fence.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

seed <- 20260930L
set.seed(seed)
p <- 6L; n <- 150L
Lam <- matrix(c(1.0, 0, 0.6, 0.8, -0.5, 0.7,
                0, 0.9, -0.7, 0.5, 0.8, -0.4), p, 2)
mu <- c(0.5, -0.3, 0.2, 0.8, -0.6, 0.1)
Z <- matrix(rnorm(n * 2), n, 2)
Y <- sweep(Z %*% t(Lam), 2, mu, "+") + matrix(rnorm(n * p, sd = 0.5), n, p)
tr <- paste0("t", seq_len(p))
df <- data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                 trait = factor(rep(tr, n), levels = tr),
                 value = as.vector(t(Y)))
write.csv(df, "select_lv_p1_data.csv", row.names = FALSE)

sel <- select_lv(
  value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
  data = df, unit = "unit", trait = "trait", d_max = 3L, criterion = "bic")
tb <- sel$table
stopifnot(all(tb$converged), all(tb$pd_hessian), all(is.na(tb$error)),
          nrow(tb) == 3L)
## nobs behind R's BIC penalty
nobs_r <- attr(logLik(sel$fits[["1"]]), "nobs")

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(x, fmt, ""), collapse = ", "), "]")
con <- file("select_lv_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for select_lv() (R/select-lv.R at gllvmTMB P1). Generated once by")
w("# gen_select_lv_p1.R from select_lv_p1_data.csv (sha256 guarded below). Do not")
w("# hand-edit the [r_reference] block; regenerate it from the script.")
w("data_file = \"select_lv_p1_data.csv\"")
w("data_sha256 = \"%s\"", digest_sha <- {
  out <- system2("shasum", c("-a", "256", "select_lv_p1_data.csv"), stdout = TRUE)
  strsplit(out, " ")[[1]][1] })
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("seed = %d", seed)
w("p = %d", p); w("n_unit = %d", n)
w("trait_names = [%s]", paste0("\"", tr, "\"", collapse = ", "))
w("true_rank = 2")
w("")
w("[r_reference]")
w("# select_lv(value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),")
w("#           data, unit = \"unit\", trait = \"trait\", d_max = 3, criterion = \"bic\")")
w("d = [1, 2, 3]")
w("npar = [%s]", paste(tb$npar, collapse = ", "))
w("nobs = %d", nobs_r)
w("loglik = %s", vec(tb$logLik))
w("aic = %s", vec(tb$aic))
w("bic = %s", vec(tb$bic))
w("aicc = %s", vec(tb$aicc))
w("converged = [%s]", paste(tolower(tb$converged), collapse = ", "))
w("pd_hessian = [%s]", paste(tolower(tb$pd_hessian), collapse = ", "))
w("selected_d = %d", sel$selected_d)
w("criterion = \"%s\"", sel$criterion)
close(con)
print(tb[, c("d","npar","logLik","aic","bic","aicc","converged","pd_hessian")])
cat("selected_d", sel$selected_d, "\n")
