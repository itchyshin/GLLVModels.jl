## Regenerates data_twins_p1.toml and the three data_twins_*_p1_data.csv files. NOT run by CI or by
## any Julia test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it. Also reads ns_gauss_p1_data.csv
## (written by gen_namespace_numeric_p1.R; read, not regenerated).
## Run from test/fixtures/:  Rscript gen_data_twins_p1.R
##
## Fit-level twins for the core070 `data` rows (offset and missing-response handling). R's own
## offset(), miss_control() and fitted() are exercised through gllvmTMB(); the Julia side is
## test_data_twins_p1.jl. Every R fit must converge with a positive-definite Hessian (asserted).
##
##   pois : Poisson, p = 6, n = 150, latent(d = 1, unique = FALSE), seed 20261002.
##          none / scalar offset(log(2)) / exposure offset(log(e)), all on the complete data;
##          drop and include on the same data with 12 response cells set to NA (column value_na).
##   nb2  : nbinom2(), p = 6, n = 150, exposure offset(log(e)); recorded beta, LL', per-trait phi.
##   nb1  : nbinom1(), same design; recorded beta, LL', per-trait phi.
##   gauss_zero : the ns_gauss_p1_data.csv rank-2 Gaussian fit refitted with offset(0). R refuses a
##          NONZERO offset on a Gaussian trait (recorded as a refusal, not compared).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")


## Optional provenance check: with GLLVM_P1_SRC pointing at a checkout of gllvmTMB at the pin, the
## functions this fixture exercises must deparse identically to the installed ones.
src <- Sys.getenv("GLLVM_P1_SRC", "")
if (nzchar(src)) {
  env <- new.env(); for (f in list.files(file.path(src, "R"), "[.]R$", full.names = TRUE)) {
    for (ex in parse(f)) if (is.call(ex) && identical(ex[[1L]], as.name("<-")) && is.name(ex[[2L]]))
      assign(as.character(ex[[2L]]), tryCatch(eval(ex[[3L]], env), error = function(e) NULL), envir = env) }
  ns <- asNamespace("gllvmTMB")
  for (fn in c("gllvmTMB", "miss_control", "gll_prepare_offset", ".gllvmTMB_offset_vec", "extract_loadings"))
    stopifnot(is.function(env[[fn]]), identical(deparse(get(fn, ns)), deparse(env[[fn]])))
  cat("P1 deparse check OK\n")
}

p <- 6L; n <- 150L; tr <- paste0("t", seq_len(p))
long <- function(...) {
  d <- data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                  trait = factor(rep(tr, n), levels = tr), ...)
  d
}
lam <- c(0.9, -0.6, 0.7, 0.5, -0.8, 0.4); mu <- c(0.8, 0.2, -0.3, 0.5, 0.1, 0.4)
set.seed(20261002L)
z <- rnorm(n)
eta <- outer(z, lam) + matrix(mu, n, p, byrow = TRUE)            # n x p
ex <- function() matrix(runif(n * p, 0.5, 3), n, p)               # exposure, varies by cell
e1 <- ex(); e2 <- ex(); e3 <- ex()
Y1 <- matrix(rpois(n * p, exp(eta + log(e1))), n, p)
Y2 <- matrix(rnbinom(n * p, mu = exp(eta + log(e2)), size = 3), n, p)
m3 <- exp(eta + log(e3))
Y3 <- matrix(rnbinom(n * p, mu = m3, size = m3 / 1.5), n, p)       # linear NB1: var = mu (1 + 1.5)
na_cells <- sort(sample(n * p, 12L))
Y1na <- Y1; Y1na[na_cells] <- NA
d1 <- long(value = as.vector(t(Y1)), value_na = as.vector(t(Y1na)), e = as.vector(t(e1)))
d2 <- long(value = as.vector(t(Y2)), e = as.vector(t(e2)))
d3 <- long(value = as.vector(t(Y3)), e = as.vector(t(e3)))
write.csv(d1, "data_twins_pois_p1_data.csv", row.names = FALSE)
write.csv(d2, "data_twins_nb2_p1_data.csv", row.names = FALSE)
write.csv(d3, "data_twins_nb1_p1_data.csv", row.names = FALSE)
## read back what Julia will read, so R fits the file bytes, not the in-memory doubles
d1 <- read.csv("data_twins_pois_p1_data.csv"); d2 <- read.csv("data_twins_nb2_p1_data.csv")
d3 <- read.csv("data_twins_nb1_p1_data.csv")
for (nm in c("d1", "d2", "d3")) { d <- get(nm); d$unit <- factor(d$unit, levels = seq_len(n)); d$trait <- factor(d$trait, levels = tr); assign(nm, d) }

fit <- function(rhs, resp, d, fam, ...) {
  f <- suppressMessages(suppressWarnings(gllvmTMB(
    as.formula(paste(resp, "~ 0 + trait", rhs, "+ latent(0 + trait | unit, d = 1, unique = FALSE)")),
    data = d, unit = "unit", trait = "trait", family = fam, ...)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
f_none  <- fit("", "value", d1, poisson())
f_scal  <- fit("+ offset(log(2))", "value", d1, poisson())
f_exp   <- fit("+ offset(log(e))", "value", d1, poisson())
f_drop  <- fit("", "value_na", d1, poisson())
f_incl  <- fit("", "value_na", d1, poisson(), missing = miss_control(response = "include"))
f_nb2   <- fit("+ offset(log(e))", "value", d2, nbinom2())
f_nb1   <- fit("+ offset(log(e))", "value", d3, nbinom1())
## the exposure offset moves the optimum: the offset-free logLik must differ from the exposure one
stopifnot(abs(as.numeric(logLik(f_none)) - as.numeric(logLik(f_exp))) > 1)
## R documents that include reaches the drop optimum
stopifnot(abs(as.numeric(logLik(f_drop)) - as.numeric(logLik(f_incl))) < 1e-6)
stopifnot(sum(is.na(d1$value_na)) == 12L)

dg <- read.csv("ns_gauss_p1_data.csv"); dg$unit <- factor(dg$unit, levels = sort(unique(dg$unit))); dg$trait <- factor(dg$trait, levels = tr)
g2 <- function(rhs) gllvmTMB(as.formula(paste("value ~ 0 + trait", rhs, "+ latent(0 + trait | unit, d = 2, unique = FALSE)")),
                             data = dg, unit = "unit", trait = "trait")
f_g0 <- suppressMessages(suppressWarnings(g2("+ offset(0)")))
stopifnot(f_g0$opt$convergence == 0L, isTRUE(f_g0$sd_report$pdHess))
f_gplain <- suppressMessages(suppressWarnings(g2("")))
stopifnot(abs(as.numeric(logLik(f_g0)) - as.numeric(logLik(f_gplain))) < 1e-8)
nonzero_refusal <- tryCatch({ suppressMessages(suppressWarnings(g2("+ offset(0.5)"))); "NOT REFUSED" },
                            error = function(e) conditionMessage(e))
stopifnot(grepl("offsets are supported for count families", nonzero_refusal))

LL <- function(f) { L <- suppressMessages(extract_loadings(f, level = "unit", rotate = "none")); as.numeric(t(tcrossprod(L))) }
phi2 <- function(f, nm) { x <- f$opt$par[names(f$opt$par) == nm]; as.numeric(exp(x)) }
phi_nb2 <- phi2(f_nb2, "log_phi_nbinom2"); phi_nb1 <- as.numeric(f_nb1$report$phi_nbinom1)
stopifnot(length(phi_nb2) == p, length(phi_nb1) == p)

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("data_twins_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the core070 data rows (offset, missing-response handling), gllvmTMB P1.")
w("# Generated once by gen_data_twins_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n); w("n_unit_gauss = %d", nlevels(dg$unit))
w("trait_names = [%s]", paste0("\"", tr, "\"", collapse = ", "))
sec <- function(name, desc, file, f, extra = function() invisible()) {
  w(""); w("[%s]", name); w("# %s", desc)
  w("data_file = \"%s\"", file); w("data_sha256 = \"%s\"", shaw(file))
  w("converged = true"); w("pd_hessian = true")
  w("loglik = %s", fmt(as.numeric(logLik(f))))
  w("beta = %s", vec(coef(f)))
  w("# Lambda Lambda^T from extract_loadings(rotate = \"none\"), p x p row-major (the sign of a K = 1 axis is not identified)")
  w("lambda_lambdat = %s", vec(LL(f)))
  extra()
}
pf <- "data_twins_pois_p1_data.csv"
sec("pois_none", "Poisson, no offset: value ~ 0 + trait + latent(d = 1, unique = FALSE); seed 20261002", pf, f_none)
sec("pois_scalar", "Poisson, offset(log(2)) (a scalar offset); same data", pf, f_scal)
sec("pois_exposure", "Poisson, offset(log(e)) with e = column e (varies by cell); same data", pf, f_exp)
sec("pois_na_drop", "Poisson on column value_na (12 NA cells), default missing = miss_control() (response = drop)", pf, f_drop,
    function() w("n_na = 12"))
sec("pois_na_include", "Poisson on column value_na (12 NA cells), missing = miss_control(response = \"include\")", pf, f_incl,
    function() w("n_na = 12"))
sec("nb2_exposure", "nbinom2(), offset(log(e)); per-trait phi = exp(log_phi_nbinom2)", "data_twins_nb2_p1_data.csv", f_nb2,
    function() w("phi = %s", vec(phi_nb2)))
sec("nb1_exposure", "nbinom1(), offset(log(e)); per-trait phi = fit$report$phi_nbinom1", "data_twins_nb1_p1_data.csv", f_nb1,
    function() w("phi = %s", vec(phi_nb1)))
sec("gauss_zero", "Gaussian rank-2 (latent d = 2) on ns_gauss_p1_data.csv with offset(0); equals the plain fit (asserted)",
    "ns_gauss_p1_data.csv", f_g0, function() {
      w("loglik_plain = %s", fmt(as.numeric(logLik(f_gplain))))
      w("nonzero_offset_refusal = \"%s\"", gsub("\"", "'", gsub("\\s+", " ", sub("\n.*", "", nonzero_refusal))))
    })
close(con)
for (f in list(f_none, f_scal, f_exp, f_drop, f_incl, f_nb2, f_nb1, f_g0)) print(as.numeric(logLik(f)))
print(phi_nb2); print(phi_nb1)
