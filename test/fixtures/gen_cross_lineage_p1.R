## Regenerates cross_lineage_p1.toml, cross_lineage_p1_data.csv and cross_lineage_p1_K.csv.
## NOT run by CI or by any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1").
## Needs gllvmTMB installed at that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_cross_lineage_p1.R
##
## Twin of the namespace rows export/extract_Gamma and export/extract_coevolution_modules. One
## Gaussian fit of a cross-lineage kernel tier:
##   value ~ 0 + trait + kernel_latent(species, K = K, d = 3, name = "cross", unique = FALSE),
##   unit = "site", trait = "trait"
## with K = make_cross_kernel(A_H, A_P, W, rho = 0.6) over 8 host and 8 partner species, 4 sites
## per species, and 4 traits (h_size, h_defence, p_size, p_attack), one observation per trait and
## site (complete data). The model is
##   vec(Y) ~ N(trait means, (P K_eff P') (x) Lambda Lambda' + sigma_eps^2 I),
## with P the site -> species incidence, Lambda 4 x 3 and K_eff = K + 1e-8 I (the jitter gllvmTMB
## adds; it is measured below from tmb_data$Ainv_phy_rr).
##
## Recorded (R's own functions, the estimands of the two rows):
##   extract_Gamma(fit, level = "cross", row_traits, col_traits): the host-by-partner block, and a
##     second call with reordered names that mixes the lineages, to pin orientation and order;
##   extract_coevolution_modules(fit, level = "cross", host traits, partner traits): R, the
##     singular values, squared shares and both axis tables; and the same call with n_modules = 1.
## d = 3 with two traits per lineage is chosen so that the module decomposition is not degenerate:
## with d = 2 both singular values are 1 by construction (Lambda_H is then square), whereas with
## d = 3 the first singular value is 1 by construction (two planes in R^3 share a line) and the
## second is a genuine canonical correlation.
##
## The data and K are written to CSV first and the fit is made on the data READ BACK from the
## CSVs, so R and Julia see identical doubles. The fit uses tight nlminb tolerances; it must
## converge (code 0) with a positive-definite Hessian.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")

## ---- simulated cross-lineage data ----
set.seed(20261005L)
nH <- 8L; nP <- 8L; m <- 4L; d_lv <- 3L
corr1d <- function(n, ell) outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / ell))
hs <- paste0("H", seq_len(nH)); ps <- paste0("P", seq_len(nP))
A_H <- corr1d(nH, 2); dimnames(A_H) <- list(hs, hs)
A_P <- corr1d(nP, 3); dimnames(A_P) <- list(ps, ps)
W <- exp(-abs(outer(seq(-1, 1, length.out = nH), seq(-1, 1, length.out = nP), "-")) / 0.35)
dimnames(W) <- list(hs, ps)
rho <- 0.6
K <- make_cross_kernel(A_H, A_P, W, rho = rho)
sp <- rownames(K); ns <- length(sp)
stopifnot(identical(sp, c(hs, ps)))
tr <- c("h_size", "h_defence", "p_size", "p_attack")
host <- tr[1:2]; partner <- tr[3:4]
Lam <- matrix(c(1.0,  0.0, 0.0,
                0.5,  0.8, 0.0,
                0.6, -0.4, 0.7,
               -0.3,  0.6, 0.5), 4, 3, byrow = TRUE)
G <- t(chol(K + diag(1e-10, ns))) %*% matrix(rnorm(ns * d_lv), ns, d_lv)   # ns x d, cols iid N(0, K)
mu <- c(0.2, -0.15, 0.1, -0.05)
site_sp <- rep(seq_len(ns), each = m); n_site <- ns * m
Y <- t(G[site_sp, , drop = FALSE] %*% t(Lam)) + mu + matrix(rnorm(4 * n_site, sd = 0.4), 4)
d0 <- data.frame(site = rep(seq_len(n_site), each = 4), trait = rep(tr, n_site),
                 species = rep(sp[site_sp], each = 4), value = as.vector(Y))
write.csv(d0, "cross_lineage_p1_data.csv", row.names = FALSE)
write.csv(as.data.frame(unclass(K)[, , drop = FALSE]), "cross_lineage_p1_K.csv", row.names = FALSE)

## ---- read back (the fitted doubles are the CSV doubles) ----
d <- read.csv("cross_lineage_p1_data.csv")
K_csv <- as.matrix(read.csv("cross_lineage_p1_K.csv", check.names = FALSE))
dimnames(K_csv) <- list(sp, sp)
stopifnot(identical(colnames(read.csv("cross_lineage_p1_K.csv", check.names = FALSE)), sp),
          max(abs(K_csv - unclass(K)[, ])) < 1e-14)   # write.csv keeps 15 digits; K_csv is the fitted K
attr(K_csv, "gllvmTMB_cross_kernel") <- attr(K, "gllvmTMB_cross_kernel")   # keep R's rho record
d$site <- factor(d$site, levels = seq_len(n_site))
d$trait <- factor(d$trait, levels = tr)
d$species <- factor(d$species, levels = sp)

ctl <- gllvmTMBcontrol(optArgs = list(control = list(rel.tol = 1e-12, sing.tol = 1e-12,
                                                       x.tol = 1e-12, eval.max = 4000L,
                                                       iter.max = 3000L)))
f <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait + kernel_latent(species, K = K_csv, d = 3, name = "cross", unique = FALSE),
  data = d, unit = "site", trait = "trait", family = gaussian(), control = ctl)))
stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
stopifnot(identical(names(f$opt$par), c(rep("b_fix", 4L), "log_sigma_eps", rep("theta_rr_phy", 9L))))
grad_max <- max(abs(f$tmb_obj$gr(f$opt$par)))
pl <- f$tmb_obj$env$parList(f$opt$par)
td <- f$tmb_data
K_eff <- solve(as.matrix(td$Ainv_phy_rr))
jitter <- mean(diag(K_eff - K_csv))
stopifnot(abs(jitter - 1e-8) < 1e-12, max(abs(K_eff - K_csv - diag(jitter, ns))) < 1e-12)
ll <- as.numeric(logLik(f))
stopifnot(abs(ll + f$opt$objective) < 1e-8)
beta <- as.numeric(pl$b_fix)
sigma_eps <- exp(as.numeric(pl$log_sigma_eps))
Sig <- suppressMessages(extract_Sigma(f, level = "cross", part = "shared", link_residual = "none"))$Sigma
stopifnot(identical(rownames(Sig), tr), identical(colnames(Sig), tr))

## ---- the two estimands ----
Gam <- extract_Gamma(f, level = "cross", row_traits = host, col_traits = partner)
stopifnot(identical(dimnames(Gam), list(host, partner)), max(abs(Gam - Sig[host, partner])) == 0)
row_perm <- c("h_defence", "h_size"); col_perm <- c("p_attack", "h_size", "p_size")
Gam_perm <- extract_Gamma(f, level = "cross", row_traits = row_perm, col_traits = col_perm)
stopifnot(identical(dimnames(Gam_perm), list(row_perm, col_perm)))
mo <- extract_coevolution_modules(f, level = "cross", row_traits = host, col_traits = partner)
stopifnot(identical(dimnames(mo$R), list(host, partner)),
          identical(mo$modules$module, c("module_1", "module_2")),
          identical(mo$modules$component, c("cross", "cross")),
          identical(names(mo$row_axes), c("component", "side", "module", "trait", "loading")),
          identical(mo$row_axes$trait, rep(host, 2L)), identical(mo$col_axes$trait, rep(partner, 2L)),
          identical(mo$row_axes$module, rep(c("module_1", "module_2"), each = 2L)),
          identical(mo$row_axes$side, rep("row", 4L)), identical(mo$col_axes$side, rep("column", 4L)))
mo1 <- extract_coevolution_modules(f, level = "cross", row_traits = host, col_traits = partner,
                                   n_modules = 1)
stopifnot(nrow(mo1$modules) == 1L, nrow(mo1$row_axes) == 2L, max(abs(mo1$R - mo$R)) == 0)
sv <- mo$modules$singular_value
stopifnot(abs(sv[1] - 1) < 1e-6, sv[2] > 0.05, sv[2] < 0.95)   # first is 1 by construction (d = 3)

con <- file("cross_lineage_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for namespace rows export/extract_Gamma and export/extract_coevolution_modules")
w("# (gllvmTMB P1). Generated once by gen_cross_lineage_p1.R, which also writes the two CSVs")
w("# (sha256 recorded). Do not hand-edit; regenerate from the script. Matrices are written")
w("# column-major (R's as.vector), with their row and column trait names alongside.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[fit]")
w("# Gaussian, unit = \"site\", trait = \"trait\"; K = make_cross_kernel(A_H, A_P, W, rho = 0.6)")
w("formula = \"value ~ 0 + trait + kernel_latent(species, K = K, d = 3, name = \\\"cross\\\", unique = FALSE)\"")
w("level = \"cross\"")
w("d = %d", d_lv)
w("rho = %s", fmt(rho))
w("data_file = \"cross_lineage_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("cross_lineage_p1_data.csv"))
w("K_file = \"cross_lineage_p1_K.csv\"")
w("K_sha256 = \"%s\"", shaw("cross_lineage_p1_K.csv"))
w("n_host = %d", nH)
w("n_partner = %d", nP)
w("n_site = %d", n_site)
w("species = %s", svec(sp))
w("trait_names = %s", svec(tr))
w("# gllvmTMB fits K + jitter * I (measured from tmb_data$Ainv_phy_rr)")
w("K_jitter = %s", fmt(1e-8))
w("converged = true")
w("pd_hessian = true")
w("r_gradient_max = %s", fmt(grad_max))
w("loglik = %s", fmt(ll))
w("# b_fix: trait intercepts, in trait order")
w("beta = %s", vec(beta))
w("# exp(log_sigma_eps): the residual SD")
w("sigma_eps = %s", fmt(sigma_eps))
w("# extract_Sigma(fit, level = \"cross\", part = \"shared\", link_residual = \"none\")$Sigma, 4 x 4")
w("Sigma_shared = %s", vec(Sig))
w("")
w("[gamma]")
w("# extract_Gamma(fit, level = \"cross\", row_traits = host, col_traits = partner), 2 x 2")
w("row_traits = %s", svec(host))
w("col_traits = %s", svec(partner))
w("Gamma = %s", vec(Gam))
w("# extract_Gamma(fit, level = \"cross\", row_traits = row_perm, col_traits = col_perm), 2 x 3")
w("row_perm = %s", svec(row_perm))
w("col_perm = %s", svec(col_perm))
w("Gamma_perm = %s", vec(Gam_perm))
w("")
w("[modules]")
w("# extract_coevolution_modules(fit, level = \"cross\", row_traits = host, col_traits = partner)")
w("row_traits = %s", svec(host))
w("col_traits = %s", svec(partner))
w("# $R, 2 x 2 (rows host, columns partner)")
w("R = %s", vec(mo$R))
w("module = %s", svec(mo$modules$module))
w("singular_value = %s", vec(mo$modules$singular_value))
w("squared_share = %s", vec(mo$modules$squared_share))
w("# $row_axes$loading and $col_axes$loading (trait fastest, module slowest, as R's tables)")
w("row_axes_trait = %s", svec(mo$row_axes$trait))
w("row_axes_module = %s", svec(mo$row_axes$module))
w("row_axes_loading = %s", vec(mo$row_axes$loading))
w("col_axes_trait = %s", svec(mo$col_axes$trait))
w("col_axes_module = %s", svec(mo$col_axes$module))
w("col_axes_loading = %s", vec(mo$col_axes$loading))
w("# the same call with n_modules = 1")
w("n1_singular_value = %s", vec(mo1$modules$singular_value))
w("n1_squared_share = %s", vec(mo1$modules$squared_share))
w("n1_row_axes_loading = %s", vec(mo1$row_axes$loading))
w("n1_col_axes_loading = %s", vec(mo1$col_axes$loading))
close(con)
cat(sprintf("loglik %.10f  sigma_eps %.8f  grad %.2e\n", ll, sigma_eps, grad_max))
print(Gam); print(Gam_perm); print(mo$modules); print(mo$row_axes); print(mo$col_axes)
