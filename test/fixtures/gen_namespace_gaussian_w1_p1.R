## Regenerates ns_gauss_w1_p1.toml. NOT run by CI or by any Julia test -- provenance for how
## the fixture was produced, against gllvmTMB pinned at commit
## 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs gllvmTMB installed at
## that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_namespace_gaussian_w1_p1.R
##
## Twins of three namespace rows on ONE shared Gaussian dataset, ns_gauss_p1_data.csv (written by
## gen_namespace_numeric_p1.R; p = 6 traits, n = 200 units, one observation per trait and unit;
## its sha256 is recorded below and checked by the Julia test):
##   wide    : export(gllvmTMB_wide). gllvmTMB_wide(Y, d = 2) on the wide 200 x 6 matrix. The
##             wrapper builds value ~ 0 + trait + latent(0 + trait | site, d = 2), and latent()
##             carries a per-trait unique variance by default (unique = TRUE), so the model is
##             Sigma = Lambda Lambda' + diag(sd_B^2) + sigma_eps^2 I with sigma_eps mapped off
##             (fixed by R's Q7 rule, max(1e-3 * sd(y), 1e-6) over all responses, recorded below).
##   ordiplot: S3method(ordiplot, gllvmTMB_multi). ordiplot(fit) on the main fit of
##             gen_namespace_numeric_p1.R (latent(d = 2, unique = FALSE)); the invisible return
##             value list(scores, loadings) is recorded (plot sent to a null device).
##   flag    : export(flag_unreliable_loadings). A confirmatory fit, latent(d = 2, unique = FALSE)
##             with lambda_constraint = list(unit = M), M pinning Lambda[1, 2] = 0 (the engine's
##             structural zero, stated explicitly) and Lambda[2, 1] = 0 (trait 2 loads on axis 2
##             only, as in the simulating loadings). flag_unreliable_loadings() is called with its
##             default null region c(-0.1, 0.1) and with c(-0.5, 0.5); the second makes some
##             intervals overlap the region, so the flag column is not constant.
## Every fit uses tight nlminb tolerances (as gen_animal_scalar_p1.R) and must converge (code 0)
## with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")
rowmajor <- function(m) vec(t(as.matrix(m)))

p <- 6L; n <- 200L
tr <- paste0("t", seq_len(p))
data_file <- "ns_gauss_p1_data.csv"
df <- read.csv(data_file)
df$unit <- factor(df$unit, levels = seq_len(n))
df$trait <- factor(df$trait, levels = tr)
stopifnot(nrow(df) == n * p, !anyNA(df$unit), !anyNA(df$trait))
Y <- matrix(NA_real_, n, p, dimnames = list(paste0("site", seq_len(n)), tr))
Y[cbind(as.integer(df$unit), as.integer(df$trait))] <- df$value
stopifnot(!anyNA(Y))

ctl <- gllvmTMBcontrol(optArgs = list(control = list(rel.tol = 1e-12, sing.tol = 1e-12,
                                                       x.tol = 1e-12, eval.max = 2000L,
                                                       iter.max = 1500L)))
check_fit <- function(f) {
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  max(abs(f$tmb_obj$gr(f$opt$par)))
}

## ---- wide: gllvmTMB_wide(Y, d = 2) ----
fw <- suppressMessages(suppressWarnings(gllvmTMB_wide(Y, d = 2, control = ctl)))
gw <- check_fit(fw)
stopifnot(identical(unique(names(fw$opt$par)), c("b_fix", "theta_rr_B", "theta_diag_B")))
plw <- fw$tmb_obj$env$parList(fw$opt$par)
w_sigma_eps <- exp(as.numeric(plw$log_sigma_eps))          # mapped off: not in opt$par
stopifnot(abs(w_sigma_eps - as.numeric(fw$report$sigma_eps)) < 1e-15,
          abs(w_sigma_eps - max(1e-3 * stats::sd(df$value), 1e-6)) < 1e-15)   # R/fit-multi.R Q7
w_beta <- as.numeric(plw$b_fix)
w_L <- as.matrix(fw$report$Lambda_B)
w_sdB <- as.numeric(fw$report$sd_B)
w_ll <- as.numeric(logLik(fw))
stopifnot(abs(w_ll + fw$opt$objective) < 1e-8, nrow(w_L) == p, ncol(w_L) == 2L)

## ---- ordiplot: ordiplot(fit) on the main latent(d = 2, unique = FALSE) fit ----
f1 <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
  data = df, unit = "unit", trait = "trait", control = ctl)))
g1 <- check_fit(f1)
grDevices::pdf(NULL)
o <- suppressMessages(ordiplot(f1))
grDevices::dev.off()
stopifnot(identical(names(o), c("scores", "loadings")), identical(dim(o$scores), c(n, 2L)),
          identical(dim(o$loadings), c(p, 2L)), identical(rownames(o$loadings), tr),
          identical(rownames(o$scores), as.character(seq_len(n))))
o_ll <- as.numeric(logLik(f1))

## ---- flag: flag_unreliable_loadings() on a confirmatory fit ----
M <- matrix(NA_real_, p, 2L, dimnames = list(tr, c("LV1", "LV2")))
M[1, 2] <- 0; M[2, 1] <- 0
fc <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE),
  data = df, unit = "unit", trait = "trait", lambda_constraint = list(unit = M),
  control = ctl)))
gc <- check_fit(fc)
c_ll <- as.numeric(logLik(fc))
c_sigma_eps <- as.numeric(fc$report$sigma_eps)
fl_default <- flag_unreliable_loadings(fc)
fl_wide <- flag_unreliable_loadings(fc, null_region = c(-0.5, 0.5))
for (fl in list(fl_default, fl_wide)) {
  stopifnot(identical(as.character(fl$trait), rep(tr, 2)),
            identical(as.character(fl$axis), rep(c("LV1", "LV2"), each = p)),
            all(fl$method == "wald"), all(fl$loading_scale == "raw"),
            all(fl$ci_status == "ok"), all(fl$pd_hessian))
}
stopifnot(identical(fl_default[, c("estimate", "se", "lower", "upper")],
                    fl_wide[, c("estimate", "se", "lower", "upper")]))
flagstr <- function(x) svec(ifelse(is.na(x), "NA", ifelse(x, "TRUE", "FALSE")))

con <- file("ns_gauss_w1_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the namespace rows export/gllvmTMB_wide, S3method/ordiplot,gllvmTMB_multi and")
w("# export/flag_unreliable_loadings (gllvmTMB P1). Generated once by gen_namespace_gaussian_w1_p1.R")
w("# from ns_gauss_p1_data.csv (sha256 recorded). Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("data_file = \"%s\"", data_file)
w("data_sha256 = \"%s\"", shaw(data_file))
w("p = %d", p)
w("n = %d", n)
w("trait_names = %s", svec(tr))
w("")
w("[wide]")
w("call = \"gllvmTMB_wide(Y, d = 2)\"")
w("# formula built by the wrapper: value ~ 0 + trait + latent(0 + trait | site, d = 2), unique = TRUE")
w("converged = true")
w("pd_hessian = true")
w("r_gradient_max = %s", fmt(gw))
w("loglik = %s", fmt(w_ll))
w("# b_fix: trait intercepts, in trait order")
w("beta = %s", vec(w_beta))
w("# report$Lambda_B, 6 x 2, row-major")
w("Lambda = %s", rowmajor(w_L))
w("# report$sd_B: per-trait unique SD (the latent() default unique = TRUE)")
w("sd_B = %s", vec(w_sdB))
w("# exp(log_sigma_eps): mapped off by gllvmTMB's Q7 rule, max(1e-3 * sd(y), 1e-6) over all responses; not estimated")
w("sigma_eps_fixed = %s", fmt(w_sigma_eps))
w("")
w("[ordiplot]")
w("call = \"ordiplot(fit)\"")
w("formula = \"value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)\"")
w("converged = true")
w("pd_hessian = true")
w("r_gradient_max = %s", fmt(g1))
w("loglik = %s", fmt(o_ll))
w("# ordiplot(fit)$scores, 200 x 2, row-major (rotate = \"none\", the default)")
w("scores = %s", rowmajor(o$scores))
w("# ordiplot(fit)$loadings, 6 x 2, row-major")
w("loadings = %s", rowmajor(o$loadings))
w("")
w("[flag]")
w("formula = \"value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)\"")
w("# lambda_constraint = list(unit = M); M is NA except M[1, 2] = 0 and M[2, 1] = 0")
w("pins = [[1, 2], [2, 1]]")
w("converged = true")
w("pd_hessian = true")
w("r_gradient_max = %s", fmt(gc))
w("loglik = %s", fmt(c_ll))
w("sigma_eps = %s", fmt(c_sigma_eps))
w("# loading_ci() columns as returned inside flag_unreliable_loadings(fit); rows in R order")
w("# (trait fastest, axis slowest): t1..t6 on LV1, then t1..t6 on LV2. method wald, raw scale.")
w("estimate = %s", vec(fl_default$estimate))
w("se = %s", vec(fl_default$se))
w("lower = %s", vec(fl_default$lower))
w("upper = %s", vec(fl_default$upper))
w("pinned = [%s]", paste(ifelse(fl_default$pinned, "true", "false"), collapse = ", "))
w("# flag_unreliable_loadings(fit): default null_region = c(-0.1, 0.1)")
w("null_region_default = %s", vec(c(-0.1, 0.1)))
w("unreliable_default = %s", flagstr(fl_default$unreliable))
w("# flag_unreliable_loadings(fit, null_region = c(-0.5, 0.5))")
w("null_region_wide = %s", vec(c(-0.5, 0.5)))
w("unreliable_wide = %s", flagstr(fl_wide$unreliable))
close(con)
cat(sprintf("wide ll %.10f grad %.2e | ordiplot ll %.10f grad %.2e | flag ll %.10f grad %.2e\n",
            w_ll, gw, o_ll, g1, c_ll, gc))
print(table(fl_wide$unreliable, useNA = "ifany"))
