## Regenerates off_predict_twin_p1.toml and off_predict_twin_p1_data.csv. NOT run by CI or by any
## Julia test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it. Reads data_twins_pois_p1_data.csv
## (written by gen_data_twins_p1.R; read, not regenerated).
## Run from test/fixtures/:  Rscript gen_off_predict_twin_p1.R
##
## Twin for the core070 row data/DATA-OFF-PREDICT (maintainer ruling 2026-10-05, vault D-319,
## option (b)): a prediction on the training units with a NEW offset. R re-evaluates the stored
## offset expression on `newdata` and keeps the training units' latent modes, so on the training
## units predict(fit, newdata = nd, type = "link")$est = beta + log(e_new) + Lambda z_train. The
## Julia side is predict(fit, Y; type = :link, offset = log.(E_new), modes = :training).
## R side: the Poisson exposure fit of gen_data_twins_p1.R
## (value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE)), then
## newdata = the training rows with e replaced by e_new (runif 0.5 to 3, seed 20261006, written to
## off_predict_twin_p1_data.csv and read back). Asserted: the fit converges with a positive-definite
## Hessian; the newdata rows come back in training order; the new prediction minus log(e_new)
## equals the training prediction minus log(e) (the training modes are kept); e_new moves the
## predictor by more than 0.5 somewhere (the comparison discriminates).
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

set.seed(20261006L)
nf <- "off_predict_twin_p1_data.csv"
write.csv(data.frame(unit = as.integer(d1$unit), trait = as.character(d1$trait),
                     e_new = runif(n * p, 0.5, 3)), nf, row.names = FALSE)
en <- read.csv(nf)                                                  # use the read-back doubles
stopifnot(identical(en$unit, as.integer(d1$unit)), identical(en$trait, as.character(d1$trait)))
nd <- d1; nd$e <- en$e_new

pt <- predict(f, type = "link")
pn <- suppressMessages(predict(f, newdata = nd, type = "link"))
stopifnot(nrow(pn) == n * p, identical(as.integer(pn$unit), as.integer(d1$unit)),
          identical(as.character(pn$trait), as.character(d1$trait)))
eta_train <- as.numeric(pt$est); eta_new <- as.numeric(pn$est)
## the training modes are kept: only the offset changes
stopifnot(max(abs((eta_new - log(nd$e)) - (eta_train - log(d1$e)))) < 1e-12,
          max(abs(eta_new - eta_train)) > 0.5)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("off_predict_twin_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 data/DATA-OFF-PREDICT (training units, new offset, training modes kept), gllvmTMB P1.")
w("# Generated once by gen_off_predict_twin_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("")
w("[pois_exposure_new_offset]")
w("# Poisson, value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE); data in long order (unit-major, trait within unit)")
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
w("new_offset_file = \"%s\"", nf); w("new_offset_sha256 = \"%s\"", shaw(nf))
w("new_offset_column = \"e_new\"")
w("converged = true"); w("pd_hessian = true")
w("loglik = %s", fmt(as.numeric(logLik(f))))
w("# predict(fit, type = \"link\")$est: training-row linear predictor at the latent modes, training offset log(e)")
w("eta_link_train = %s", vec(eta_train))
w("# predict(fit, newdata = training rows with e = e_new, type = \"link\")$est: training modes, new offset log(e_new)")
w("eta_link_new_offset = %s", vec(eta_new))
close(con)
print(as.numeric(logLik(f))); print(range(eta_new - eta_train))
