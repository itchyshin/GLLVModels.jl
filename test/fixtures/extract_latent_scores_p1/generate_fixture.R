# Generates every fixture file in this directory (Gaussian, Poisson, NB2 --
# all rank K=2). Requires gllvmTMB installed at pin P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, version 0.7.1) to a library on
# .libPaths(); the exact install used to produce the committed fixtures was:
#
#   git -C <path-to-gllvmTMB> worktree add --detach <scratch>/gllvmTMB-p1 \
#     9539352f66f2db2cc26b1c393e67212a359b60c9
#   R CMD INSTALL --library=<scratch>/Rlib <scratch>/gllvmTMB-p1
#
# Re-running this script end to end with R's RNG unchanged (R >= 3.6,
# default "Rejection" sampler) and the .libPaths() line pointing at that
# library reproduces the committed CSV/txt files exactly (`set.seed()` below
# is the only source of randomness). Adjust .libPaths() to wherever gllvmTMB
# 0.7.1 is installed locally before running.
.libPaths(c("/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/Rlib", .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
cat("gllvmTMB version:", as.character(packageVersion("gllvmTMB")), "\n")

set.seed(20260927)

n_sites <- 15L
p <- 6L
K <- 2L

## ---- shared latent draws (used to generate BOTH families' data) ----------
Lambda_true <- matrix(c(
  0.9, 0.6,
  -0.4, 0.5,
  0.3, -0.6,
  0.8, 0.2,
  -0.5, 0.4,
  0.6, -0.3
), nrow = p, ncol = K, byrow = TRUE)
alpha_true <- c(1.0, -0.5, 0.3, 0.8, -0.2, 0.5)
Z_true <- matrix(rnorm(n_sites * K), nrow = n_sites, ncol = K)

eta <- matrix(alpha_true, nrow = n_sites, ncol = p, byrow = TRUE) + Z_true %*% t(Lambda_true)
# eta is n_sites x p

make_long <- function(value_mat) {
  data.frame(
    site  = factor(rep(seq_len(n_sites), times = p), levels = seq_len(n_sites)),
    trait = factor(rep(paste0("trait_", seq_len(p)), each = n_sites),
                    levels = paste0("trait_", seq_len(p))),
    value = as.vector(value_mat)
  )
}

## =========================== GAUSSIAN ======================================
sigma_eps_true <- 0.5
Y_gauss <- eta + matrix(rnorm(n_sites * p, sd = sigma_eps_true), nrow = n_sites, ncol = p)
df_gauss <- make_long(Y_gauss)

fit_gauss <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_gauss,
  family = gaussian(),
  unit = "site"
)
cat("Gaussian fit class:", paste(class(fit_gauss), collapse = ","), "\n")
cat("Gaussian pdHess:", isTRUE(fit_gauss$sd_report$pdHess), "\n")

z_gauss_unit <- extract_latent_scores(fit_gauss, level = "unit")
z_gauss_unit_obs <- extract_latent_scores(fit_gauss, level = "unit_obs")
stopifnot(is.null(z_gauss_unit_obs))
cat("Gaussian z_hat dim:", paste(dim(z_gauss_unit), collapse = "x"), "\n")

## fixed-effect / loading extraction: probe likely accessors
Lambda_gauss_hat <- fit_gauss$report$Lambda_B
beta_gauss_hat <- {
  par <- fit_gauss$tmb_obj$env$last.par.best
  as.numeric(par[names(par) == "b_fix"])
}
cat("Gaussian beta_hat:", paste(round(beta_gauss_hat, 4), collapse = ", "), "\n")
cat("Gaussian Lambda_hat dim:", paste(dim(Lambda_gauss_hat), collapse = "x"), "\n")
sigma_eps_hat <- fit_gauss$report$sigma_eps
if (is.null(sigma_eps_hat)) sigma_eps_hat <- fit_gauss$report$phi
cat("Gaussian sigma_eps_hat:", sigma_eps_hat, "\n")

loglik_gauss <- as.numeric(stats::logLik(fit_gauss))
cat("Gaussian logLik:", loglik_gauss, "\n")

## =========================== POISSON =======================================
mu_pois <- exp(eta)
Y_pois <- matrix(rpois(n_sites * p, lambda = as.vector(mu_pois)), nrow = n_sites, ncol = p)
df_pois <- make_long(Y_pois)

fit_pois <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_pois,
  family = poisson(),
  unit = "site"
)
cat("Poisson fit class:", paste(class(fit_pois), collapse = ","), "\n")
cat("Poisson pdHess:", isTRUE(fit_pois$sd_report$pdHess), "\n")

z_pois_unit <- extract_latent_scores(fit_pois, level = "unit")
z_pois_unit_obs <- extract_latent_scores(fit_pois, level = "unit_obs")
stopifnot(is.null(z_pois_unit_obs))
cat("Poisson z_hat dim:", paste(dim(z_pois_unit), collapse = "x"), "\n")

Lambda_pois_hat <- fit_pois$report$Lambda_B
beta_pois_hat <- {
  par <- fit_pois$tmb_obj$env$last.par.best
  as.numeric(par[names(par) == "b_fix"])
}
cat("Poisson beta_hat:", paste(round(beta_pois_hat, 4), collapse = ", "), "\n")
loglik_pois <- as.numeric(stats::logLik(fit_pois))
cat("Poisson logLik:", loglik_pois, "\n")

## =========================== NB2 (per-trait dispersion) =====================
## Own (larger) n_sites: per-trait dispersion + rank-2 loadings + per-trait
## intercept is a lot of parameters for n_sites=15 (NB2 pdHess was FALSE
## there); a bigger sample keeps the fit well-posed while remaining cheap.
n_sites_nb2 <- 60L
r_true <- c(4, 6, 8, 5, 7, 10)  # per-trait true NB2 size (moderate overdispersion)
Z_true_nb2 <- matrix(rnorm(n_sites_nb2 * K), nrow = n_sites_nb2, ncol = K)
eta_nb2 <- matrix(alpha_true, nrow = n_sites_nb2, ncol = p, byrow = TRUE) +
  Z_true_nb2 %*% t(Lambda_true)
mu_nb2 <- exp(eta_nb2)
Y_nb2 <- matrix(
  rnbinom(n_sites_nb2 * p, size = rep(r_true, each = n_sites_nb2), mu = as.vector(mu_nb2)),
  nrow = n_sites_nb2, ncol = p
)
df_nb2 <- data.frame(
  site  = factor(rep(seq_len(n_sites_nb2), times = p), levels = seq_len(n_sites_nb2)),
  trait = factor(rep(paste0("trait_", seq_len(p)), each = n_sites_nb2),
                  levels = paste0("trait_", seq_len(p))),
  value = as.vector(Y_nb2)
)

fit_nb2 <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_nb2,
  family = nbinom2(),
  unit = "site"
)
cat("NB2 fit class:", paste(class(fit_nb2), collapse = ","), "\n")
cat("NB2 pdHess:", isTRUE(fit_nb2$sd_report$pdHess), "\n")

z_nb2_unit <- extract_latent_scores(fit_nb2, level = "unit")
z_nb2_unit_obs <- extract_latent_scores(fit_nb2, level = "unit_obs")
stopifnot(is.null(z_nb2_unit_obs))
cat("NB2 z_hat dim:", paste(dim(z_nb2_unit), collapse = "x"), "\n")

Lambda_nb2_hat <- fit_nb2$report$Lambda_B
beta_nb2_hat <- {
  par <- fit_nb2$tmb_obj$env$last.par.best
  as.numeric(par[names(par) == "b_fix"])
}
phi_nb2_hat <- fit_nb2$report$phi_nbinom2
cat("NB2 beta_hat:", paste(round(beta_nb2_hat, 4), collapse = ", "), "\n")
cat("NB2 phi_hat:", paste(round(phi_nb2_hat, 4), collapse = ", "), "\n")
loglik_nb2 <- as.numeric(stats::logLik(fit_nb2))
cat("NB2 logLik:", loglik_nb2, "\n")

## default-method error check
err <- tryCatch(extract_latent_scores(data.frame(x = 1)), error = function(e) e)
cat("Default method error message:", conditionMessage(err), "\n")

## =========================== WRITE FIXTURE =================================
out <- list(
  n_sites = n_sites, p = p, K = K,
  gaussian = list(
    Y = Y_gauss,                      # n_sites x p (rows=sites, cols=traits)
    z_hat_unit = z_gauss_unit,         # n_sites x K
    Lambda_hat = Lambda_gauss_hat,     # p x K
    beta_hat = beta_gauss_hat,         # length p
    sigma_eps_hat = sigma_eps_hat,
    loglik = loglik_gauss,
    Lambda_true = Lambda_true, alpha_true = alpha_true, Z_true = Z_true,
    sigma_eps_true = sigma_eps_true
  ),
  poisson = list(
    Y = Y_pois,
    z_hat_unit = z_pois_unit,
    Lambda_hat = Lambda_pois_hat,
    beta_hat = beta_pois_hat,
    loglik = loglik_pois,
    Lambda_true = Lambda_true, alpha_true = alpha_true, Z_true = Z_true
  ),
  nb2 = list(
    Y = Y_nb2,
    n_sites = n_sites_nb2,
    z_hat_unit = z_nb2_unit,
    Lambda_hat = Lambda_nb2_hat,
    beta_hat = beta_nb2_hat,
    phi_hat = phi_nb2_hat,
    loglik = loglik_nb2,
    Lambda_true = Lambda_true, alpha_true = alpha_true, Z_true = Z_true_nb2,
    r_true = r_true
  )
)

dir.create("/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture", showWarnings = FALSE)
saveRDS(out, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/extract_latent_scores_p1.rds")

## Also dump as plain-text JSON-ish (no jsonlite dependency assumed) via dput-friendly CSVs
write.csv(Y_gauss, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Y_gauss.csv", row.names = FALSE)
write.csv(Y_pois, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Y_pois.csv", row.names = FALSE)
write.csv(z_gauss_unit, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/z_gauss_unit.csv", row.names = FALSE)
write.csv(z_pois_unit, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/z_pois_unit.csv", row.names = FALSE)
write.csv(Lambda_gauss_hat, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Lambda_gauss_hat.csv", row.names = FALSE)
write.csv(Lambda_pois_hat, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Lambda_pois_hat.csv", row.names = FALSE)
writeLines(as.character(beta_gauss_hat), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/beta_gauss_hat.txt")
writeLines(as.character(beta_pois_hat), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/beta_pois_hat.txt")
writeLines(as.character(loglik_gauss), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/loglik_gauss.txt")
writeLines(as.character(loglik_pois), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/loglik_pois.txt")
writeLines(as.character(sigma_eps_hat), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/sigma_eps_hat.txt")

write.csv(Y_nb2, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Y_nb2.csv", row.names = FALSE)
write.csv(z_nb2_unit, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/z_nb2_unit.csv", row.names = FALSE)
write.csv(Lambda_nb2_hat, "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/Lambda_nb2_hat.csv", row.names = FALSE)
writeLines(as.character(beta_nb2_hat), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/beta_nb2_hat.txt")
writeLines(as.character(phi_nb2_hat), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/phi_nb2_hat.txt")
writeLines(as.character(loglik_nb2), "/Users/z3437171/local-scratch/gllvmTMB-p1-scratch/fixture/loglik_nb2.txt")

cat("DONE\n")
