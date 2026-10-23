## Regenerates predict_offset_twin_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it. Reads data_twins_pois_p1_data.csv
## (written by gen_data_twins_p1.R; read, not regenerated).
## Run from test/fixtures/:  Rscript gen_predict_offset_twin_p1.R
##
## Twin for the core070 row data/DATA-OFF-TRAIN-STORED: a fit keeps the offset it was trained with,
## and the training-row prediction uses it. R side: the Poisson exposure fit of gen_data_twins_p1.R
## (value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE)), then
##   * the stored training offset, .gllvmTMB_offset_vec(fit) (= fit$tmb_data$offset_vec), and
##   * predict(fit, type = "link")$est, the training-row linear predictor at the latent modes, which
##     includes the stored offset (methods-gllvmTMB.R: report$eta).
## The fit must converge with a positive-definite Hessian (asserted), and the fixed-effects-only
## training prediction minus the stored offset must equal the trait intercepts (asserted), so the
## recorded prediction carries the stored offset.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 6L; n <- 150L; tr <- paste0("t", seq_len(p))
pf <- "data_twins_pois_p1_data.csv"
d1 <- read.csv(pf)
d1$unit <- factor(d1$unit, levels = seq_len(n)); d1$trait <- factor(d1$trait, levels = tr)
stopifnot(nrow(d1) == n * p, identical(as.integer(d1$unit), rep(seq_len(n), each = p)),
          identical(as.character(d1$trait), rep(tr, n)))

f <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE),
  data = d1, unit = "unit", trait = "trait", family = poisson())))
stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))

off <- gllvmTMB:::.gllvmTMB_offset_vec(f)
stopifnot(length(off) == n * p, isTRUE(all.equal(off, log(d1$e), tolerance = 0)))
pr <- predict(f, type = "link")
stopifnot(nrow(pr) == n * p, identical(as.integer(pr$unit), as.integer(d1$unit)),
          identical(as.character(pr$trait), as.character(d1$trait)))
eta <- as.numeric(pr$est)
## the stored offset is in the prediction: the fixed-effects-only training prediction
## (re_form = ~0, X b + .gllvmTMB_offset_vec(fit)) minus the stored offset is the trait intercept
fixed <- as.numeric(predict(f, type = "link", re_form = ~0)$est)
stopifnot(isTRUE(all.equal(fixed - off, as.numeric(coef(f))[as.integer(d1$trait)])),
          isTRUE(all.equal(eta, as.numeric(f$report$eta))), max(abs(off)) > 0.5)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("predict_offset_twin_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 data/DATA-OFF-TRAIN-STORED (stored training offset, training-row predict), gllvmTMB P1.")
w("# Generated once by gen_predict_offset_twin_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("")
w("[pois_exposure_stored]")
w("# Poisson, value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE); data in long order (unit-major, trait within unit)")
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
w("converged = true"); w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(f))))
w("# .gllvmTMB_offset_vec(fit): the stored training offset, one per long row")
w("offset_vec = %s", vec(off))
w("# predict(fit, type = \"link\")$est: training-row linear predictor at the latent modes, offset included")
w("eta_link = %s", vec(eta))
close(con)
print(as.numeric(logLik(f))); print(range(eta))
