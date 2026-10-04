## Regenerates animal_scalar_p1.toml, animal_scalar_p1_data.csv and animal_scalar_p1_A.csv.
## NOT run by CI or by any Julia test -- provenance for how the fixture was produced, against
## gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1").
## Needs gllvmTMB installed at that exact commit in a lane-local library; set GLLVM_P1_RLIB to it.
## Run from test/fixtures/:  Rscript gen_animal_scalar_p1.R
##
## Twin of the namespace row export/animal_scalar. One Gaussian fit:
##   value ~ 0 + trait + animal_scalar(id, A = A),  unit = "site", cluster = "id"
## on a simulated pedigree (12 founders + 48 offspring, A = pedigree_to_A(ped)), p = 3 traits,
## two records (sites) per individual, one observation per trait and site. The model is
##   vec(Y) ~ N(X beta, sigma2_a (P A_eff P') (x) I_p + sigma_eps^2 I),
## with P the site -> individual incidence and A_eff = A + 1e-8 I (the jitter gllvmTMB adds; it is
## measured below from tmb_data$Cphy_inv). One shared additive variance sigma2_a, one residual SD.
##
## The data and A are written to CSV first and the fit is made on the data READ BACK from the CSVs,
## so R and Julia see identical doubles. The pedigree = route is fitted too (R-side check only): it
## carries A without the 1e-8 jitter, so its log-likelihood differs from the A = route by ~2e-7;
## the difference is recorded and must stay below 1e-6. The fit uses tight nlminb tolerances (rel.tol, sing.tol and
## x.tol 1e-12; the Core070 tight-control covariance runner sets the first two) so the recorded
## optimum is sharp (max |gradient| ~2e-6 against ~3e-5 with default control, same objective); it
## must converge (code 0) with a positive-definite Hessian. rel.tol 1e-12 alone stops with
## "singular convergence (7)" at the same objective, so it is not used.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
shaw <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", as.character(x), "\"", collapse = ", "), "]")

## ---- simulated pedigree and data ----
set.seed(20261004L)
nf <- 12L; no <- 48L; p <- 3L; m <- 2L
ids <- sprintf("a%02d", seq_len(nf + no))
## two generations of offspring: the first 24 offspring have founder parents, the last 24 have
## parents drawn from the founders and the first offspring generation (sires odd, dams even ids).
sire <- dam <- rep(NA_character_, nf + no)
for (k in seq_len(no)) {
  i <- nf + k
  pool <- if (k <= 24L) ids[seq_len(nf)] else ids[seq_len(nf + 24L)]
  pool_s <- pool[seq(1L, length(pool), by = 2L)]; pool_d <- pool[seq(2L, length(pool), by = 2L)]
  sire[i] <- sample(pool_s, 1L); dam[i] <- sample(pool_d, 1L)
}
ped <- data.frame(id = ids, sire = sire, dam = dam)
A <- pedigree_to_A(ped)
stopifnot(identical(rownames(A), ids), isSymmetric(A))
n_id <- nrow(A); n_site <- n_id * m
sig2a <- 0.6; sig_eps <- 0.7; mu <- c(0.3, -0.2, 0.5)
G <- t(chol(A)) %*% matrix(rnorm(n_id * p), n_id, p) * sqrt(sig2a)   # n_id x p, cols iid N(0, sig2a A)
site_id <- rep(seq_len(n_id), each = m)
Y <- t(G[site_id, , drop = FALSE]) + mu + matrix(rnorm(p * n_site, sd = sig_eps), p)   # p x n_site
tr <- paste0("t", seq_len(p))
d0 <- data.frame(site = rep(seq_len(n_site), each = p), trait = rep(tr, n_site),
                 id = rep(site_id, each = p), value = as.vector(Y))
write.csv(d0, "animal_scalar_p1_data.csv", row.names = FALSE)
Adf <- as.data.frame(unname(A)); write.csv(Adf, "animal_scalar_p1_A.csv", row.names = FALSE)

## ---- read back (the fitted doubles are the CSV doubles) ----
d <- read.csv("animal_scalar_p1_data.csv")
A_csv <- as.matrix(read.csv("animal_scalar_p1_A.csv")); dimnames(A_csv) <- list(ids, ids)
stopifnot(max(abs(A_csv - A)) == 0)
d$site <- factor(d$site, levels = seq_len(n_site))
d$trait <- factor(d$trait, levels = tr)
d$id <- factor(ids[d$id], levels = ids)

ctl <- gllvmTMBcontrol(optArgs = list(control = list(rel.tol = 1e-12, sing.tol = 1e-12,
                                                       x.tol = 1e-12, eval.max = 2000L,
                                                       iter.max = 1500L)))
fit_one <- function(term) {
  fml <- as.formula(paste("value ~ 0 + trait +", term))
  suppressMessages(suppressWarnings(gllvmTMB(fml, data = d, unit = "site", cluster = "id",
                                             control = ctl)))
}
f <- fit_one("animal_scalar(id, A = A_csv)")
stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
stopifnot(identical(names(f$opt$par), c("b_fix", "b_fix", "b_fix", "log_sigma_eps", "loglambda_phy")))
grad_max <- max(abs(f$tmb_obj$gr(f$opt$par)))
pl <- f$tmb_obj$env$parList(f$opt$par)
td <- f$tmb_data
A_eff <- solve(as.matrix(td$Cphy_inv))
jitter <- mean(diag(A_eff - A))
stopifnot(abs(jitter - 1e-8) < 1e-12, max(abs(A_eff - A - diag(jitter, n_id))) < 1e-12)
stopifnot(identical(as.integer(td$species_id) + 1L, rep(site_id, each = p)))   # site -> individual map

## pedigree route: same model up to the jitter (R-side check only); it carries A with no jitter
fp <- fit_one("animal_scalar(id, pedigree = ped)")
stopifnot(max(abs(solve(as.matrix(fp$tmb_data$Cphy_inv)) - A)) < 1e-12)
ped_ll_diff <- as.numeric(logLik(fp)) - as.numeric(logLik(f))
stopifnot(fp$opt$convergence == 0L, isTRUE(fp$sd_report$pdHess), abs(ped_ll_diff) < 1e-6)

beta <- as.numeric(pl$b_fix)
sigma2_a <- exp(as.numeric(pl$loglambda_phy))   # loglambda_phy is a log VARIANCE
sigma_eps <- exp(as.numeric(pl$log_sigma_eps))
ll <- as.numeric(logLik(f))
stopifnot(abs(ll + f$opt$objective) < 1e-8)

con <- file("animal_scalar_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for namespace row export/animal_scalar (gllvmTMB P1). Generated once by")
w("# gen_animal_scalar_p1.R, which also writes the two CSVs (sha256 recorded). Do not hand-edit;")
w("# regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("")
w("[scalar]")
w("# value ~ 0 + trait + animal_scalar(id, A = A), unit = \"site\", cluster = \"id\", Gaussian")
w("formula = \"value ~ 0 + trait + animal_scalar(id, A = A)\"")
w("data_file = \"animal_scalar_p1_data.csv\"")
w("data_sha256 = \"%s\"", shaw("animal_scalar_p1_data.csv"))
w("A_file = \"animal_scalar_p1_A.csv\"")
w("A_sha256 = \"%s\"", shaw("animal_scalar_p1_A.csv"))
w("p = %d", p)
w("n_id = %d", n_id)
w("n_site = %d", n_site)
w("trait_names = %s", svec(tr))
w("# gllvmTMB fits A + jitter * I (measured from tmb_data$Cphy_inv)")
w("A_jitter = %s", fmt(1e-8))
w("converged = true")
w("pd_hessian = true")
w("r_gradient_max = %s", fmt(grad_max))
w("# logLik(pedigree = ped route, A without jitter) - loglik below")
w("pedigree_route_loglik_diff = %s", fmt(ped_ll_diff))
w("loglik = %s", fmt(ll))
w("# b_fix: trait intercepts, in trait order")
w("beta = %s", vec(beta))
w("# exp(loglambda_phy): the one shared additive (animal) variance")
w("sigma2_a = %s", fmt(sigma2_a))
w("# exp(log_sigma_eps): the residual SD")
w("sigma_eps = %s", fmt(sigma_eps))
close(con)
cat(sprintf("loglik %.10f  sigma2_a %.8f  sigma_eps %.8f  grad %.2e\n", ll, sigma2_a, sigma_eps, grad_max))
print(beta)
