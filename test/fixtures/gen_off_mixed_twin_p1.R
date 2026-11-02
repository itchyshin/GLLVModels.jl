## Regenerates off_mixed_twin_p1.toml and off_mixed_twin_p1_data.csv. NOT run by CI or by any Julia
## test -- provenance for how the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at that
## exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_off_mixed_twin_p1.R
##
## Fit-level twin for the core070 row data/DATA-OFF-MIXED: the R batch case
## gll_prepare_offset(quote(c(1,0,4)), c(2L,0L,5L), ...) admits, in one mixed-family fit with family
## ids 2, 0, 5 (poisson, gaussian, nbinom2), a zero offset on the non-count (gaussian) row next to
## nonzero offsets on the count rows, and replays to c(1, 0, 4) with no fit number. Here that is one
## real mixed-family fit with an exposure offset(log(e)) that is zero on the gaussian trait; the Julia
## side is test_off_mixed_twin_p1.jl (fit_mixed_gllvm with offset = log.(E)).
##
## p = 6 traits, n = 150 units, seed 20261006, one latent factor:
##   value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE)
##   family = list(t1 = poisson(), t2 = gaussian(), t3 = nbinom2(), t4 = poisson(), t5 = nbinom2(),
##                 t6 = poisson()), attr(, "family_var") = "trait".
## One gaussian trait only: gllvmTMB shares one log_sigma_eps across all gaussian traits of a
## mixed fit, while Julia's fit_mixed_gllvm gives each Normal trait its own sigma; with a single
## gaussian trait the two parameterisations coincide. nbinom2 phi is per trait on both sides.
## The exposure e varies by cell (runif 0.5 to 3) on the count traits and is exactly 1 on the
## gaussian trait, so log(e) is exactly 0 there (asserted). The R fit must converge with a
## positive-definite Hessian (asserted), and every phi must sit inside (1e-3, 1e3) (asserted).
## The same model without the offset is also fitted and recorded (loglik_no_offset); it must
## converge, with a positive-definite Hessian, at a logLik lower by more than 1 (asserted).
## Negative control: the same fit with a nonzero offset (0.5) on the gaussian trait must be refused by
## R's row-wise offset gate (asserted); the first line of R's message is recorded in the fixture.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

p <- 6L; n <- 150L; tr <- paste0("t", seq_len(p))
famname <- c("poisson", "gaussian", "nbinom2", "poisson", "nbinom2", "poisson")
fid <- c(poisson = 2L, gaussian = 0L, nbinom2 = 5L)[famname]
lam <- c(0.6, -0.5, 0.5, 0.4, 0.7, -0.45); mu <- c(0.8, 1.0, 0.6, 1.1, -0.5, 0.7)
set.seed(20261006L)
z <- rnorm(n)
eta <- outer(z, lam) + matrix(mu, n, p, byrow = TRUE)              # n x p
E <- matrix(runif(n * p, 0.5, 3), n, p)
E[, famname == "gaussian"] <- 1                                    # zero offset on the gaussian trait
Y <- matrix(NA_real_, n, p)
for (t in seq_len(p)) {
  m <- eta[, t] + log(E[, t])
  Y[, t] <- switch(famname[t],
                   poisson  = rpois(n, exp(m)),
                   nbinom2  = rnbinom(n, mu = exp(m), size = 3),
                   gaussian = m + rnorm(n, sd = 0.6))
}
d <- data.frame(unit = rep(seq_len(n), each = p), trait = rep(tr, n),
                value = as.vector(t(Y)), e = as.vector(t(E)))
pf <- "off_mixed_twin_p1_data.csv"
write.csv(d, pf, row.names = FALSE)
d <- read.csv(pf)                                                   # fit the read-back doubles
d$unit <- factor(d$unit, levels = seq_len(n)); d$trait <- factor(d$trait, levels = tr)
stopifnot(all(log(d$e[d$trait == "t2"]) == 0), all(d$e[d$trait != "t2"] != 1))

fams <- list(t1 = poisson(), t2 = gaussian(), t3 = nbinom2(), t4 = poisson(), t5 = nbinom2(),
             t6 = poisson())
attr(fams, "family_var") <- "trait"
fitm <- function(fml) {
  f <- suppressMessages(suppressWarnings(gllvmTMB(fml, data = d, unit = "unit", trait = "trait",
                                                   family = fams)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
f <- fitm(value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE))
f0 <- fitm(value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE))
## Negative control: the same fit with a NONZERO offset on the gaussian trait (log(e2) = 0.5 there)
## must be refused by gll_prepare_offset (R/offset.R at P1), the refusal half of the row-wise rule.
d$e2 <- ifelse(d$trait == "t2", exp(0.5), d$e)
options(cli.width = 1000)                                            # keep the cli header on one line
nonzero_refusal <- tryCatch({
  suppressMessages(suppressWarnings(gllvmTMB(
    value ~ 0 + trait + offset(log(e2)) + latent(0 + trait | unit, d = 1, unique = FALSE),
    data = d, unit = "unit", trait = "trait", family = fams)))
  "NOT REFUSED"
}, error = function(e) conditionMessage(e))
stopifnot(grepl("offsets are supported for count families", nonzero_refusal),
          grepl("t2", nonzero_refusal), grepl("gaussian", nonzero_refusal))
ll <- as.numeric(logLik(f)); ll0 <- as.numeric(logLik(f0))
stopifnot(ll - ll0 > 1)
stopifnot(sum(names(f$opt$par) == "log_sigma_eps") == 1L)
phi <- as.numeric(exp(f$opt$par[names(f$opt$par) == "log_phi_nbinom2"]))
stopifnot(length(phi) == 2L, all(phi > 1e-3), all(phi < 1e3))
sigma <- as.numeric(exp(f$opt$par[names(f$opt$par) == "log_sigma_eps"]))
L <- suppressMessages(extract_loadings(f, level = "unit", rotate = "none"))
LL <- as.numeric(t(tcrossprod(L)))

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", x, "\"", collapse = ", "), "]")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("off_mixed_twin_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 data/DATA-OFF-MIXED (one poisson/gaussian/nbinom2 fit, exposure offset zero on the gaussian trait), gllvmTMB P1.")
w("# Generated once by gen_off_mixed_twin_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("p = %d", p); w("n_unit = %d", n)
w("data_file = \"%s\"", pf); w("data_sha256 = \"%s\"", shaw(pf))
w("response_column = \"value\""); w("exposure_column = \"e\"")
w("families = %s", svec(famname))
w("family_ids = %s", paste0("[", paste(fid, collapse = ", "), "]"))
w("converged = true"); w("pd_hessian = true")
w("loglik = %s", fmt(ll))
w("beta = %s", vec(coef(f)))
w("# Lambda Lambda^T from extract_loadings(rotate = \"none\"), p x p row-major (the sign of a K = 1 axis is not identified)")
w("lambda_lambdat = %s", vec(LL))
w("# nbinom2 per-trait phi = exp(log_phi_nbinom2), traits t3 and t5 in order")
w("phi = %s", vec(phi))
w("# gaussian residual sd = exp(log_sigma_eps), trait t2")
w("sigma = %s", fmt(sigma))
w("# the same model without the offset (converged, positive-definite Hessian)")
w("loglik_no_offset = %s", fmt(ll0))
w("# R refuses the same fit with a nonzero offset on the gaussian trait t2 (first line of the message)")
w("nonzero_gaussian_offset_refusal = \"%s\"", gsub("\"", "'", gsub("`", "", gsub("\\s+", " ", sub("\n.*", "", nonzero_refusal)))))
close(con)
cat("loglik", ll, "no offset", ll0, "phi", phi, "sigma", sigma, "\n")
