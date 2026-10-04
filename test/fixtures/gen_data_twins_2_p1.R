## Regenerates data_twins_2_p1.toml and data_twins_2_mi_p1_data.csv. NOT run by CI or by any Julia
## test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_data_twins_2_p1.R
##
## Fit-level twins for the core070 rows data/DATA-MISS-MODEL (miss_control(predictor = "model")),
## data/DATA-MISS-BOTH (miss_control("include", "model")) and the namespace row
## S3method/imputed,gllvmTMB. The Julia side is test_data_twins_2_p1.jl.
##
## One Gaussian design, p = 4 traits, n = 120 units, seed 20261004: one unit-level predictor x with
## covariate model x ~ N(mu_x + gamma z, sigma_x^2), a shared slope on x, one latent factor and a
## common residual SD:
##   value ~ 0 + trait + mi(x) + latent(0 + trait | unit, d = 1, unique = FALSE),
##   impute = list(x = x ~ z), family = gaussian().
## x is set to NA at 18 units (all trait rows of a unit).
##   miss_model : every response observed; missing = miss_control(predictor = "model").
##   miss_both  : additionally response cells set to NA (column value_na): 35 drawn at random, one
##                trait of a unit whose x is missing, and every response of units 5 (x observed) and
##                17 (x missing); missing = miss_control(response = "include", predictor = "model").
##                A unit with no response contributes only its covariate density (unit 5) or
##                nothing (unit 17), and imputed() at unit 17 is the covariate-model mean.
## Both fits must converge with a positive-definite Hessian (asserted). For miss_both, the
## response = "drop" fit removes the two units with no response altogether (asserted equal to the
## include fit of the data without them), so it differs from "include"; it is recorded as
## loglik_drop for reference.
## imputed(fit) (rows = "missing") is recorded for both fits: the missing units and their
## conditional modes; the standard errors are recorded for reference only.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 4L; n <- 120L; tr <- paste0("t", seq_len(p))
set.seed(20261004L)
z <- rnorm(n)
x <- 0.3 + 0.7 * z + rnorm(n, sd = 0.6)
lam <- c(0.9, 0.6, -0.5, 0.4); a <- c(0.5, -0.2, 0.1, 0.3); bx <- 0.8
eta <- rnorm(n)
Y <- outer(a, rep(1, n)) + bx * matrix(x, p, n, byrow = TRUE) + lam %o% eta +
  matrix(rnorm(p * n, sd = 0.5), p, n)                              # p x n
miss_x <- sort(sample(n, 18L))
Yna <- Y
na_cells <- sort(sample(p * n, 35L))
Yna[na_cells] <- NA
Yna[1L, miss_x[1L]] <- NA                                           # a unit missing x and one response
Yna[, c(5L, 17L)] <- NA                                             # units with no response at all
stopifnot(!(5L %in% miss_x), 17L %in% miss_x, all(rowSums(!is.na(Yna)) >= 1L),
          sum(colSums(!is.na(Yna)) == 0L) == 2L)
xna <- x; xna[miss_x] <- NA
d <- data.frame(unit = rep(seq_len(n), each = p), trait = rep(tr, n),
                value = as.vector(Y), value_na = as.vector(Yna),
                x = rep(xna, each = p), z = rep(z, each = p))
pf <- "data_twins_2_mi_p1_data.csv"
write.csv(d, pf, row.names = FALSE)
d <- read.csv(pf)                                                   # fit the read-back doubles
d$unit <- factor(d$unit, levels = seq_len(n)); d$trait <- factor(d$trait, levels = tr)
stopifnot(sum(is.na(d$value_na)) == sum(is.na(Yna)), sum(is.na(d$x)) == p * length(miss_x))

fit <- function(resp, missing) {
  f <- suppressMessages(suppressWarnings(gllvmTMB(
    as.formula(paste(resp, "~ 0 + trait + mi(x) + latent(0 + trait | unit, d = 1, unique = FALSE)")),
    data = d, unit = "unit", trait = "trait", family = gaussian(),
    impute = list(x = x ~ z), missing = missing)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
f_model <- fit("value", miss_control(predictor = "model"))
f_both <- fit("value_na", miss_control(response = "include", predictor = "model"))
f_drop <- fit("value_na", miss_control(response = "drop", predictor = "model"))
## response = "drop" removes the rows of a unit with no response, so unit 5's covariate density
## leaves the drop likelihood while "include" keeps it: the two differ here (recorded, loglik_drop).
## The drop fit equals the include fit of the data without units 5 and 17 (asserted).
d_no <- droplevels(d[!(d$unit %in% c(5L, 17L)), ])
f_inc_no <- suppressMessages(suppressWarnings(gllvmTMB(
  value_na ~ 0 + trait + mi(x) + latent(0 + trait | unit, d = 1, unique = FALSE),
  data = d_no, unit = "unit", trait = "trait", family = gaussian(),
  impute = list(x = x ~ z), missing = miss_control(response = "include", predictor = "model"))))
stopifnot(abs(as.numeric(logLik(f_inc_no)) - as.numeric(logLik(f_drop))) < 1e-6,
          abs(as.numeric(logLik(f_both)) - as.numeric(logLik(f_drop))) > 1e-3)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("data_twins_2_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 data/DATA-MISS-MODEL, data/DATA-MISS-BOTH and namespace S3method/imputed,gllvmTMB, gllvmTMB P1.")
w("# Generated once by gen_data_twins_2_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
w("# the units whose x is NA (1-based)")
w("missing_x_units = [%s]", paste(miss_x, collapse = ", "))
sec <- function(name, comment, f) {
  pl <- f$tmb_obj$env$parList(f$opt$par)
  L <- matrix(f$report$Lambda_B, nrow = p)
  im <- imputed(f)
  stopifnot(identical(as.integer(im$level_id), miss_x), all(!im$observed),
            isTRUE(all.equal(im$estimate, as.numeric(pl$x_mis))))
  w(""); w("[%s]", name); w("# %s", comment)
  w("converged = true"); w("pd_hessian = true")
  w("loglik = %s", fmt(as.numeric(logLik(f))))
  w("# b_fix: trait intercepts t1..t4, then the slope on x")
  w("intercepts = %s", vec(pl$b_fix[seq_len(p)]))
  w("b_x = %s", fmt(pl$b_fix[p + 1L]))
  w("# covariate model x ~ N(mu_x + gamma z, sigma_x^2): beta_mi = (mu_x, gamma), log_sigma_mi")
  w("mu_x = %s", fmt(pl$beta_mi[1L])); w("gamma_z = %s", fmt(pl$beta_mi[2L]))
  w("sigma_x = %s", fmt(exp(pl$log_sigma_mi)))
  w("sigma_eps = %s", fmt(exp(pl$log_sigma_eps)))
  w("# Lambda Lambda', column-major p x p")
  w("LLt = %s", vec(L %*% t(L)))
  w("# imputed(fit): level_id of each missing unit, its conditional mode, and (reference only) its SE")
  w("imputed_level_id = [%s]", paste(as.integer(im$level_id), collapse = ", "))
  w("imputed_estimate = %s", vec(im$estimate))
  w("imputed_std_error = %s", vec(im$std_error))
}
sec("miss_model", "value ~ 0 + trait + mi(x) + latent(d = 1, unique = FALSE), impute x ~ z, miss_control(predictor = 'model'); every response observed", f_model)
sec("miss_both", "value_na ~ 0 + trait + mi(x) + latent(d = 1, unique = FALSE), impute x ~ z, miss_control(response = 'include', predictor = 'model'); units 5 and 17 have no response", f_both)
w("nobs = %d", as.integer(stats::nobs(f_both)))
w("# the units with no observed response (1-based)")
w("no_response_units = [5, 17]")
w("# response = 'drop' on the same data: rows of units 5 and 17 removed, so unit 5's covariate density is not in it;")
w("# equal to the include fit without those units (asserted). Recorded for reference")
w("loglik_drop = %s", fmt(as.numeric(logLik(f_drop))))
close(con)
print(c(model = as.numeric(logLik(f_model)), both = as.numeric(logLik(f_both)),
        drop = as.numeric(logLik(f_drop))))
