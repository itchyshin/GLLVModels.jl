## Regenerates weights_twins_p1.toml and weights_twins_p1_data.csv. NOT run by CI or by any Julia
## test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_weights_twins_p1.R
##
## Fit-level twins for the 13 core070 rows data/DATA-W-*. Their R batch cases replay the internal
## shape helper normalise_weights() (R/weights-shape.R) to an identical() expectation, with no fit
## number. Here each weight shape is used in a real Poisson fit, one fit per row, and the Julia side
## (test/test_weights_twins_p1.jl) fits the same data with fit_gllvm(Y; family = Poisson(), K = 1,
## weights = ...). A weight multiplies its row's NLL in gllvmTMB (src/gllvmTMB.cpp, weights_i).
##
## One data set: p = 6 traits, n = 120 units, seed 20261006, one latent factor, Poisson counts:
##   y_ts ~ Poisson(exp(mu_t + lambda_t z_s)).
## A second response column y_na has 14 cells set to NA (fixed cells, listed in the TOML) for the
## mask rows. Per-cell and per-unit weight columns are drawn once and written to the CSV.
##
## Routes, as gllvmTMB exposes them at P1:
##   long        gllvmTMB(value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
##               data = long, weights = <length nrow(long) vector>)
##   wide_matrix gllvmTMB_wide() takes Y units x traits and weights scalar / length-n / n x p, but
##               hardcodes latent(0 + trait | site, d) with latent()'s default unique = TRUE (a
##               per-trait unique variance the Julia Poisson route does not carry). So this script
##               runs gllvmTMB_wide()'s own steps verbatim -- normalise_weights(weights,
##               "wide_matrix", ...) and the same pivot and NA-cell drop -- and calls gllvmTMB() with
##               unique = FALSE. That substitution is the only difference from gllvmTMB_wide().
##   wide_df     gllvmTMB(traits(t1, ..., t6) ~ 1 + latent(1 | unit, d = 1, unique = FALSE),
##               data = wide, weights = <length-n vector>), the public traits() route.
## Mask rows: drop = default missing handling (NA cells removed); include =
## missing = miss_control(response = "include") (NA cells kept, masked out of the likelihood; the
## wide_matrix normaliser zeroes their weights).
## Every fit must converge with a positive-definite Hessian (asserted).
## gllvmTMB's logLik() aborts for non-unit weights ("undefined for this non-unit weighted
## objective"), so the compared number is the maximised weighted objective -fit$opt$objective, the
## quantity Julia's fit.loglik holds for a weighted fit. For the unweighted fit it equals logLik()
## (asserted), and both are recorded.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
normalise_weights <- getFromNamespace("normalise_weights", "gllvmTMB")

p <- 6L; n <- 120L; tr <- paste0("t", seq_len(p))
lam <- c(0.6, -0.4, 0.5, 0.35, -0.5, 0.3); mu <- c(0.8, 0.4, 0.6, 1.0, 0.5, 0.7)
set.seed(20261006L)
z <- rnorm(n)
Y <- matrix(rpois(n * p, exp(outer(z, lam) + matrix(mu, n, p, byrow = TRUE))), n, p)   # n x p
na_cells <- cbind(unit = c(3, 9, 14, 22, 30, 37, 45, 51, 60, 68, 77, 85, 99, 110),
                  trait = c(1, 4, 2, 6, 3, 5, 1, 2, 4, 6, 3, 5, 2, 1))
Yna <- Y; Yna[na_cells] <- NA
w_int  <- matrix(sample(1:3, n * p, replace = TRUE), n, p)                 # long, integer
w_frac <- matrix(round(runif(n * p, 0.25, 2), 3), n, p)                    # long, fractional
w_zero <- matrix(1, n, p); w_zero[sample(n * p, 15)] <- 0                   # long, some zeros
w_cell <- matrix(round(runif(n * p, 0.5, 2), 3), n, p)                     # wide matrix cells
w_unit <- round(runif(n, 0.5, 2), 3)                                       # one per unit

long <- data.frame(unit = rep(seq_len(n), each = p), trait = rep(tr, n),
                   y = as.vector(t(Y)), y_na = as.vector(t(Yna)),
                   w_int = as.vector(t(w_int)), w_frac = as.vector(t(w_frac)),
                   w_zero = as.vector(t(w_zero)), w_cell = as.vector(t(w_cell)),
                   w_unit = rep(w_unit, each = p))
pf <- "weights_twins_p1_data.csv"
write.csv(long, pf, row.names = FALSE)
long <- read.csv(pf)                                                     # fit the read-back doubles
long$unit <- factor(long$unit, levels = seq_len(n)); long$trait <- factor(long$trait, levels = tr)
## read-back matrices (n x p), so every route sees the same doubles
back <- function(col) { m <- matrix(NA_real_, n, p); m[cbind(as.integer(long$unit), as.integer(long$trait))] <- long[[col]]; m }
Y <- back("y"); Yna <- back("y_na"); W_cell <- back("w_cell")
w_unit <- long$w_unit[seq(1, n * p, by = p)]
colnames(Y) <- colnames(Yna) <- colnames(W_cell) <- tr

chk <- function(f) { stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess)); f }
quiet <- function(expr) suppressMessages(suppressWarnings(expr))
fml_long <- value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE)

fit_long <- function(col, w) {
  d <- long; d$value <- d[[col]]
  chk(quiet(gllvmTMB(fml_long, data = d, unit = "unit", trait = "trait", family = poisson(), weights = w)))
}
## gllvmTMB_wide()'s steps, with unique = FALSE (see header)
fit_wide_matrix <- function(Ym, weights, include = FALSE) {
  rownames(Ym) <- paste0("site", seq_len(n))
  drop <- !include
  w_long <- normalise_weights(weights = weights, response_shape = "wide_matrix",
                              n_obs = if (drop) sum(!is.na(Ym)) else n * p,
                              n_units = n, n_traits = p, na_mask = is.na(Ym), drop_masked = drop)
  ld <- data.frame(site = factor(rep(rownames(Ym), p), levels = rownames(Ym)),
                   species = factor(rep(colnames(Ym), each = n), levels = colnames(Ym)),
                   value = as.numeric(Ym))
  ld$trait <- ld$species
  if (drop && anyNA(ld$value)) ld <- ld[!is.na(ld$value), , drop = FALSE]
  args <- list(value ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE),
               data = ld, unit = "site", family = poisson(), weights = w_long)
  if (include) args$missing <- miss_control(response = "include")
  list(f = chk(quiet(do.call(gllvmTMB, args))), w_long = w_long)
}
fit_wide_df <- function(Ym, wu, include = FALSE) {
  wide <- data.frame(unit = factor(seq_len(n)), Ym)
  args <- list(traits(t1, t2, t3, t4, t5, t6) ~ 1 + latent(1 | unit, d = 1, unique = FALSE),
               data = wide, unit = "unit", family = poisson(), weights = wu)
  if (include) args$missing <- miss_control(response = "include")
  chk(quiet(do.call(gllvmTMB, args)))
}

res <- list()
res$null      <- list(f = fit_long("y", NULL))
res$long      <- list(f = fit_long("y", long$w_int))
res$fractional <- list(f = fit_long("y", long$w_frac))
res$zero      <- list(f = fit_long("y", long$w_zero))
res$matrix_scalar <- fit_wide_matrix(Y, 2)
res$matrix_unit   <- fit_wide_matrix(Y, w_unit)
res$matrix_cells  <- fit_wide_matrix(Y, W_cell)
Wna <- W_cell; Wna[is.na(Yna)] <- NA                                     # NA exactly where Y is NA
res$matrix_mask_drop    <- fit_wide_matrix(Yna, Wna)
res$matrix_mask_include <- fit_wide_matrix(Yna, Wna, include = TRUE)
res$matrix_mask_scalar  <- fit_wide_matrix(Yna, 2)
res$df_unit         <- list(f = fit_wide_df(Y, w_unit))
res$df_mask_drop    <- list(f = fit_wide_df(Yna, w_unit))
res$df_mask_include <- list(f = fit_wide_df(Yna, w_unit, include = TRUE))
stopifnot(abs(as.numeric(logLik(res$null$f)) + res$null$f$opt$objective) < 1e-8)
## the include normaliser zeroes the masked weights; the drop normaliser shrinks the vector
stopifnot(length(res$matrix_mask_drop$w_long) == n * p - nrow(na_cells),
          length(res$matrix_mask_include$w_long) == n * p,
          all(res$matrix_mask_include$w_long[is.na(as.numeric(Yna))] == 0),
          length(res$matrix_mask_scalar$w_long) == n * p - nrow(na_cells))

LL <- function(f) { L <- suppressMessages(extract_loadings(f, level = "unit", rotate = "none")); as.numeric(t(tcrossprod(L))) }
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
desc <- list(
  null = c("data/DATA-W-NULL", "long", "y", "none", "weights = NULL (unweighted)"),
  long = c("data/DATA-W-LONG", "long", "y", "w_int", "long API, integer weights 1..3, one per row (length nrow(data))"),
  fractional = c("data/DATA-W-FRACTIONAL", "long", "y", "w_frac", "long API, fractional weights in (0.25, 2)"),
  zero = c("data/DATA-W-ZERO", "long", "y", "w_zero", "long API, weight 0 on 15 rows, 1 elsewhere"),
  matrix_scalar = c("data/DATA-W-MATRIX-SCALAR", "wide_matrix", "y", "scalar 2", "wide matrix route, scalar weight 2"),
  matrix_unit = c("data/DATA-W-MATRIX-UNIT", "wide_matrix", "y", "w_unit", "wide matrix route, one weight per unit (length nrow(Y))"),
  matrix_cells = c("data/DATA-W-MATRIX-CELLS", "wide_matrix", "y", "w_cell", "wide matrix route, n x p per-cell weight matrix"),
  matrix_mask_drop = c("data/DATA-W-MATRIX-MASK-DROP", "wide_matrix", "y_na", "w_cell", "wide matrix route, per-cell weights NA where Y is NA, NA cells dropped"),
  matrix_mask_include = c("data/DATA-W-MATRIX-MASK-INCLUDE", "wide_matrix", "y_na", "w_cell", "wide matrix route, per-cell weights NA where Y is NA, response = 'include' (weights zeroed there)"),
  matrix_mask_scalar = c("data/DATA-W-MATRIX-MASK-SCALAR", "wide_matrix", "y_na", "scalar 2", "wide matrix route, scalar weight 2 with NA cells dropped"),
  df_unit = c("data/DATA-W-DF-UNIT", "wide_df", "y", "w_unit", "traits() route, one weight per unit"),
  df_mask_drop = c("data/DATA-W-DF-MASK-DROP", "wide_df", "y_na", "w_unit", "traits() route, one weight per unit, NA cells dropped"),
  df_mask_include = c("data/DATA-W-DF-MASK-INCLUDE", "wide_df", "y_na", "w_unit", "traits() route, one weight per unit, response = 'include'"))
con <- file("weights_twins_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the 13 core070 rows data/DATA-W-* (observation weights on a Poisson fit), gllvmTMB P1.")
w("# Generated once by gen_weights_twins_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
w("na_units = %s", vec(na_cells[, "unit"])); w("na_traits = %s", vec(na_cells[, "trait"]))
for (nm in names(res)) {
  f <- res[[nm]]$f; d <- desc[[nm]]
  w(""); w("[%s]", nm); w("description = \"%s\"", d[5])
  w("source_id = \"%s\"", d[1]); w("route = \"%s\"", d[2])
  w("response_column = \"%s\"", d[3]); w("weights = \"%s\"", d[4])
  w("converged = true"); w("pd_hessian = true")
  w("# maximised weighted objective, -fit$opt$objective (logLik() for the unweighted fit)")
  w("objective = %s", fmt(-f$opt$objective))
  if (nm == "null") w("loglik = %s", fmt(as.numeric(logLik(f))))
  w("beta = %s", vec(coef(f)))
  w("lambda_lambdat = %s", vec(LL(f)))
}
close(con)
for (nm in names(res)) cat(sprintf("%-20s %.6f\n", nm, -res[[nm]]$f$opt$objective))
