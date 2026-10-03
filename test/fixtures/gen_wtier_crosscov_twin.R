## Provenance for the R-twin value pinned in test/test_wtier_crosscov.jl (issue #135).
## NOT run by CI or by any Julia test. Produced with installed gllvmTMB 0.7.1
## (twin origin/main at 8010cfd4c, 2026-10-02). Run: Rscript gen_wtier_crosscov_twin.R
##
## One observation per (site, trait) and exactly one site_species per site, which is
## the layout of GLLVModels.jl's wide p x n matrix. The C++ W tier adds
## sum_k Lambda_W[t, k] * z_W[k, ss] with z_W shared by all traits of one
## site_species, so the marginal per-site covariance is
##   Lambda_B Lambda_B' + Lambda_W Lambda_W' + sigma_eps^2 I.
## The script evaluates the TMB objective (Laplace, exact for Gaussian) at a fixed
## parameter point and checks it against a dense multivariate-normal NLL in R.
suppressMessages(library(gllvmTMB))
p <- 3L; n <- 8L
yw <- outer(1:p, 1:n, function(t, s) sin(0.9 * t + 1.7 * s) + 0.3 * cos(2.1 * t * s))
dat <- data.frame(
  site    = factor(rep(sprintf("s%02d", 1:n), each = p)),
  species = factor(rep("sp1", n * p)),
  trait   = factor(rep(sprintf("t%d", 1:p), times = n)),
  value   = as.vector(yw)
)
dat$site_species <- factor(paste(dat$site, dat$species, sep = "_"))
fit <- suppressMessages(suppressWarnings(gllvmTMB(
  value ~ 0 + trait +
    latent(0 + trait | site, d = 1, unique = FALSE) +
    latent(0 + trait | site_species, d = 2, unique = FALSE),
  data = dat, unit = "site")))
obj <- fit$tmb_obj
par <- obj$par
alpha <- c(0.2, -0.1, 0.05)               # per-trait intercepts (b_fix)
th_B  <- c(0.8, -0.4, 0.5)                # Lambda_B, 3 x 1
th_W  <- c(0.6, 0.45, 0.3, -0.35, 0.25)   # Lambda_W, 3 x 2: diag first, then strict lower by column
sig   <- 0.6
par[names(par) == "b_fix"] <- alpha
par[names(par) == "theta_rr_B"] <- th_B
par[names(par) == "theta_rr_W"] <- th_W
par[names(par) == "log_sigma_eps"] <- log(sig)
nll <- obj$fn(par)
rep <- obj$report()
LB <- rep$Lambda_B; LW <- rep$Lambda_W
S <- LB %*% t(LB) + LW %*% t(LW) + diag(sig^2, p)
r <- yw - alpha
ll <- 0
for (s in 1:n) ll <- ll - 0.5 * (p * log(2 * pi) + as.numeric(determinant(S)$modulus) +
                                  sum(r[, s] * solve(S, r[, s])))
stopifnot(abs(nll + ll) < 1e-10)
cat(sprintf("R_TMB_NLL        = %.17g\n", nll))
cat(sprintf("R_dense_full_NLL = %.17g\n", -ll))
print(LB); print(LW)
