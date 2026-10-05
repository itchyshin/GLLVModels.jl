## Regenerates off_all_count_twin_p1.toml and off_all_count_twin_p1_data.csv. NOT run by CI or by any
## Julia test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_off_all_count_twin_p1.R
##
## Fit-level twin for the core070 row data/DATA-OFF-ALL-COUNT. The R batch case prepares one exposure
## offset for three count families (family ids 5, 10, 11: nbinom2(), truncated_poisson(),
## truncated_nbinom2()) and replays to c(1, 2, 4) with no fit number. Here each of the three families
## is fitted with an exposure offset(log(e)); the Julia side is test_off_all_count_twin_p1.jl.
## Julia has no fit that mixes families with an offset, so the three families are three single-family
## fits, not one mixed-family model.
##
## One design per family, p = 6 traits, n = 150 units, seed 20261005, one latent factor:
##   value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE)
##   nb2   : nbinom2(), size 3, per-trait phi.
##   tpois : truncated_poisson(), y >= 1 (zero-truncated draw by rejection).
##   tnb2  : truncated_nbinom2(), size 2, per-trait phi, y >= 1.
## The exposure e varies by cell (runif 0.5 to 3) and is drawn separately for each family.
## Every fit must converge with a positive-definite Hessian, and every phi must sit well inside
## (1e-3, 1e3) so no trait is at the Poisson limit (asserted). That the offset is not absorbed (the
## fit without it differs) is checked on the Julia side, in the twin test. (R's own no-offset
## truncated_nbinom2 fit of these data stops at "false convergence (8)", so no R value is recorded
## for it.)
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 6L; n <- 150L; tr <- paste0("t", seq_len(p))
lam <- c(0.6, -0.4, 0.5, 0.35, -0.5, 0.3); mu <- c(0.8, 0.4, 0.6, 1.0, 0.5, 0.7)
set.seed(20261005L)
z <- rnorm(n)
eta <- outer(z, lam) + matrix(mu, n, p, byrow = TRUE)              # n x p
ex <- function() matrix(runif(n * p, 0.5, 3), n, p)
## zero-truncated draw by rejection, cell by cell
rtrunc <- function(m, draw) {
  y <- draw(m)
  while (any(k <- y == 0L)) y[k] <- draw(m[k])
  y
}
e1 <- ex(); e2 <- ex(); e3 <- ex()
m1 <- exp(eta + log(e1)); m2 <- exp(eta + log(e2)); m3 <- exp(eta + log(e3))
Y1 <- matrix(rnbinom(n * p, mu = m1, size = 3), n, p)
Y2 <- matrix(rtrunc(as.vector(m2), function(m) rpois(length(m), m)), n, p)
Y3 <- matrix(rtrunc(as.vector(m3), function(m) rnbinom(length(m), mu = m, size = 2)), n, p)
stopifnot(all(Y2 >= 1L), all(Y3 >= 1L))
d <- data.frame(unit = rep(seq_len(n), each = p), trait = rep(tr, n),
                y_nb2 = as.vector(t(Y1)), e_nb2 = as.vector(t(e1)),
                y_tpois = as.vector(t(Y2)), e_tpois = as.vector(t(e2)),
                y_tnb2 = as.vector(t(Y3)), e_tnb2 = as.vector(t(e3)))
pf <- "off_all_count_twin_p1_data.csv"
write.csv(d, pf, row.names = FALSE)
d <- read.csv(pf)                                                   # fit the read-back doubles
d$unit <- factor(d$unit, levels = seq_len(n)); d$trait <- factor(d$trait, levels = tr)

fit <- function(y, e, fam) {
  d$value <- d[[y]]; d$e <- d[[e]]
  f <- suppressMessages(suppressWarnings(gllvmTMB(
    value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE),
    data = d, unit = "unit", trait = "trait", family = fam)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
LL <- function(f) { L <- suppressMessages(extract_loadings(f, level = "unit", rotate = "none")); as.numeric(t(tcrossprod(L))) }
phi_of <- function(f, nm) { x <- f$opt$par[names(f$opt$par) == nm]; stopifnot(length(x) == p); as.numeric(exp(x)) }

fams <- list(
  nb2   = list(y = "y_nb2",   e = "e_nb2",   fam = nbinom2(),           fid = 5L,  phi = "log_phi_nbinom2",
               desc = "nbinom2(), offset(log(e_nb2)); per-trait phi = exp(log_phi_nbinom2); size 3 in the draw"),
  tpois = list(y = "y_tpois", e = "e_tpois", fam = truncated_poisson(), fid = 10L, phi = NULL,
               desc = "truncated_poisson(), offset(log(e_tpois)); y >= 1 (zero-truncated draw by rejection)"),
  tnb2  = list(y = "y_tnb2",  e = "e_tnb2",  fam = truncated_nbinom2(), fid = 11L, phi = "log_phi_truncnb2",
               desc = "truncated_nbinom2(), offset(log(e_tnb2)); per-trait phi = exp(log_phi_truncnb2); size 2 in the draw, y >= 1"))
res <- lapply(fams, function(a) {
  f <- fit(a$y, a$e, a$fam)
  ph <- if (is.null(a$phi)) NULL else phi_of(f, a$phi)
  if (!is.null(ph)) stopifnot(all(ph > 1e-3), all(ph < 1e3))
  list(f = f, phi = ph)
})

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("off_all_count_twin_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 data/DATA-OFF-ALL-COUNT (exposure offset on nbinom2, truncated_poisson, truncated_nbinom2), gllvmTMB P1.")
w("# Generated once by gen_off_all_count_twin_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
for (nm in names(fams)) {
  a <- fams[[nm]]; r <- res[[nm]]
  w(""); w("[%s]", nm); w("# %s", a$desc)
  w("family_id = %d", a$fid)
  w("response_column = \"%s\"", a$y); w("exposure_column = \"%s\"", a$e)
  w("converged = true"); w("pd_hessian = true")
  w("loglik = %s", fmt(as.numeric(logLik(r$f))))
  w("beta = %s", vec(coef(r$f)))
  w("# Lambda Lambda^T from extract_loadings(rotate = \"none\"), p x p row-major (the sign of a K = 1 axis is not identified)")
  w("lambda_lambdat = %s", vec(LL(r$f)))
  if (!is.null(r$phi)) w("phi = %s", vec(r$phi))
}
close(con)
for (nm in names(res)) cat(nm, as.numeric(logLik(res[[nm]]$f)), res[[nm]]$phi, "\n")
