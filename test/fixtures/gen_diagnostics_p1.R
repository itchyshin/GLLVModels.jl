## Regenerates diagnostics_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_diagnostics_p1.R
##
## Twin fixture for sanity_multi() and compare_loadings().
##   sanity_multi : reads the tracked ns_gauss_p1_data.csv (sha256 guarded, written by
##                  gen_namespace_numeric_p1.R; p = 6, 200 units, one observation per cell) and fits
##                    value ~ 0 + trait + latent(0 + trait | unit, d = D, unique = FALSE)
##                  for D = 1 and D = 2 (D = 2 is the model of gen_postfit_twins_p1.R). Recorded per
##                  fit: the returned flags of sanity_multi(fit) and its printed report lines. Every
##                  fit must converge with a positive-definite Hessian.
##   compare_loadings : two input pairs, recorded with the full output (R, Lambda_a_rot, frobenius,
##                  cor_per_factor) and the inputs themselves at 17 significant digits.
##                    fit_vs_truth : Lambda_a = the D = 2 fit's Lambda_B, Lambda_b = the loadings the
##                                   data were simulated from (gen_namespace_numeric_p1.R), the use
##                                   R's documentation gives for this function.
##                    reflected    : p = 8, d = 3; Lambda_b random, Lambda_a = Lambda_b Q' + noise with
##                                   Q a random orthogonal matrix with det(Q) = -1, so the optimal
##                                   transform is a reflection (seed 20261004).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
mat <- function(m) paste0("[", paste(apply(m, 1, vec), collapse = ", "), "]")   # array of rows
svec <- function(x) paste0("[", paste0("\"", gsub("\"", "\\\\\"", x), "\"", collapse = ", "), "]")

tr <- paste0("t", 1:6)
df1 <- read.csv("ns_gauss_p1_data.csv")
df1$unit <- factor(df1$unit, levels = sort(unique(df1$unit)))
df1$trait <- factor(df1$trait, levels = tr)
fit_d <- function(D) {
  f <- suppressMessages(suppressWarnings(gllvmTMB(
    as.formula(sprintf("value ~ 0 + trait + latent(0 + trait | unit, d = %d, unique = FALSE)", D)),
    data = df1, unit = "unit", trait = "trait")))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
fits <- list(d1 = fit_d(1L), d2 = fit_d(2L))
san <- lapply(fits, function(f) {
  printed <- capture.output(fl <- sanity_multi(f))
  stopifnot(identical(names(fl), c("converged", "max_gradient", "sdreport_ok", "pd_hessian",
                                   "max_se", "rr_B_min_loading")))
  list(flags = fl, printed = printed)
})

## gen_namespace_numeric_p1.R simulated ns_gauss_p1_data.csv from these loadings
Lam <- matrix(c(1.0, 0, 0.6, 0.8, -0.5, 0.7,
                0, 0.9, -0.7, 0.5, 0.8, -0.4), 6, 2)
cl1_a <- fits$d2$report$Lambda_B[, 1:2, drop = FALSE]
cl1 <- compare_loadings(cl1_a, Lam)
set.seed(20261004L)
Lb <- matrix(rnorm(24), 8, 3)
Q <- qr.Q(qr(matrix(rnorm(9), 3, 3)))
if (det(Q) > 0) Q[, 1] <- -Q[, 1]
stopifnot(det(Q) < 0)
La <- Lb %*% t(Q) + matrix(rnorm(24, sd = 0.05), 8, 3)
cl2 <- compare_loadings(La, Lb)
stopifnot(det(cl2$R) < 0)

con <- file("diagnostics_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for postfit rows POSTFIT-SURFACE-sanity_multi and POSTFIT-SURFACE-compare_loadings")
w("# and namespace row export/sanity_multi (gllvmTMB P1). Generated once by gen_diagnostics_p1.R from")
w("# ns_gauss_p1_data.csv (sha256 guarded). Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("data_file = \"ns_gauss_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("ns_gauss_p1_data.csv"))
w("p = 6")
w("n_unit = 200")
w("trait_names = %s", svec(tr))
for (nm in names(fits)) {
  f <- fits[[nm]]; fl <- san[[nm]]$flags
  w("")
  w("[sanity.%s]", nm)
  w("# value ~ 0 + trait + latent(0 + trait | unit, d = %d, unique = FALSE)", f$d_B)
  w("d = %d", f$d_B)
  w("converged = true")
  w("pd_hessian = true")
  w("loglik = %s", fmt(as.numeric(logLik(f))))
  w("# sanity_multi(fit): returned flags, in R's order")
  w("flag_names = %s", svec(names(fl)))
  w("flag_converged = %s", tolower(as.character(fl$converged)))
  w("flag_sdreport_ok = %s", tolower(as.character(fl$sdreport_ok)))
  w("flag_pd_hessian = %s", tolower(as.character(fl$pd_hessian)))
  w("# max |gradient| of R's objective at nlminb's optimum: an optimiser artefact, recorded, not compared")
  w("max_gradient = %s", fmt(fl$max_gradient))
  w("max_se = %s", fmt(fl$max_se))
  w("rr_B_min_loading = %s", fmt(fl$rr_B_min_loading))
  w("# the per-trait fixed-effect SEs and Lambda_B behind max_se and rr_B_min_loading (context)")
  w("b_fix_se = %s", vec(gllvmTMB:::.gllvmTMB_b_fix_se(f)))
  w("Lambda_B = %s", mat(f$report$Lambda_B))
  w("# capture.output(sanity_multi(fit))")
  w("printed = %s", svec(san[[nm]]$printed))
}
cw <- function(tag, a, b, r) {
  w("")
  w("[compare_loadings.%s]", tag)
  w("Lambda_a = %s", mat(a))
  w("Lambda_b = %s", mat(b))
  w("R = %s", mat(r$R))
  w("Lambda_a_rot = %s", mat(r$Lambda_a_rot))
  w("frobenius = %s", fmt(r$frobenius))
  w("cor_per_factor = %s", vec(r$cor_per_factor))
}
cw("fit_vs_truth", cl1_a, Lam, cl1)
cw("reflected", La, Lb, cl2)
close(con)
str(san); str(cl1); str(cl2)
