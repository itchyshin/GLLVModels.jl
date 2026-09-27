## Regenerates zi_{poisson,nbinom2,binomial}_p1_data.csv and the [<family>.r_reference]
## and [<family>.r_at_julia] blocks of zi_p1.toml. NOT run by CI or by any Julia
## test -- it is provenance for how the fixture was produced, against gllvmTMB
## pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"
## in the true-parity export ledger).
##
## Requires gllvmTMB installed at that exact commit, e.g. into a scratch library:
##   git -C <gllvmTMB clone> worktree add --detach <dir> 9539352f66f2db2cc26b1c393e67212a359b60c9
##   R CMD INSTALL --library=<lib> <dir>
##   R_LIBS=<lib> Rscript gen_zi_p1.R                  # phase 1: data + R fits
##   R_LIBS=<lib> Rscript gen_zi_p1.R julia_params.txt # phase 2: also R objective at Julia's optimum
## Run from test/fixtures/. Phase 2 reads Julia's fitted parameters, one line per
## vector: "<family> <name> v1 v2 ...", names beta, theta_rr_B (Julia's
## pack_lambda(Λ), the same diagonal-then-strict-lower layout as TMB's
## theta_rr_B), logit_zi, log_phi (zi_nbinom2 only), loglik. The Julia side
## writes that file with test/fixtures/gen_zi_p1_julia_params.jl.
##
## Do not hand-edit the R blocks of zi_p1.toml; regenerate them from this output.
library(gllvmTMB)

args <- commandArgs(trailingOnly = TRUE)
julia_params <- if (length(args) >= 1L) args[1L] else NULL

cat("# gllvmTMB version:", as.character(packageVersion("gllvmTMB")), "\n")
cat("#", R.version.string, "\n")

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(x, fmt, character(1)), collapse = ", "), "]")

p <- 3L
lambda_true <- c(0.6, -0.5, 0.4)

simulate_family <- function(family, seed, n) {
  set.seed(seed)
  u <- stats::rnorm(n)
  out <- data.frame(site = rep(seq_len(n), p), trait = rep(seq_len(p), each = n))
  if (family == "zi_poisson") {
    beta <- c(1.2, 0.9, 1.4); zi <- c(0.25, 0.15, 0.30)
    eta <- outer(u, lambda_true) + matrix(beta, n, p, byrow = TRUE)
    z <- matrix(stats::rbinom(n * p, 1L, 1 - rep(zi, each = n)), n, p)
    y <- matrix(stats::rpois(n * p, exp(eta)), n, p) * z
    out$y <- as.vector(y)
  } else if (family == "zi_nbinom2") {
    beta <- c(1.3, 1.0, 1.5); zi <- c(0.20, 0.25, 0.15); phi <- c(1.5, 2, 1)
    eta <- outer(u, lambda_true) + matrix(beta, n, p, byrow = TRUE)
    z <- matrix(stats::rbinom(n * p, 1L, 1 - rep(zi, each = n)), n, p)
    y <- matrix(stats::rnbinom(n * p, mu = exp(eta), size = rep(phi, each = n)), n, p) * z
    out$y <- as.vector(y)
  } else {
    beta <- c(0.2, -0.3, 0.5); zi <- c(0.25, 0.20, 0.30)
    lam <- c(0.8, -0.6, 0.7)
    eta <- outer(u, lam) + matrix(beta, n, p, byrow = TRUE)
    N <- matrix(sample(3:8, n * p, replace = TRUE), n, p)
    z <- matrix(stats::rbinom(n * p, 1L, 1 - rep(zi, each = n)), n, p)
    y <- matrix(stats::rbinom(n * p, as.vector(N), stats::plogis(as.vector(eta))), n, p) * z
    out$y <- as.vector(y)
    out$trials <- as.vector(N)
  }
  out
}

fit_family <- function(family, dat) {
  d <- dat
  d$site <- factor(d$site, levels = sort(unique(d$site)))
  d$trait <- factor(d$trait, levels = seq_len(p))
  fam <- switch(family, zi_poisson = zi_poisson(), zi_nbinom2 = zi_nbinom2(),
                zi_binomial = zi_binomial())
  if (family == "zi_binomial") {
    d$fail <- d$trials - d$y
    gllvmTMB(cbind(y, fail) ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE),
             data = d, family = fam, unit = "site",
             control = gllvmTMBcontrol(se = FALSE))
  } else {
    gllvmTMB(y ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE),
             data = d, family = fam, unit = "site",
             control = gllvmTMBcontrol(se = FALSE))
  }
}

read_julia <- function(path) {
  if (is.null(path)) return(NULL)
  out <- list()
  for (ln in readLines(path)) {
    if (!nzchar(ln)) next
    tok <- strsplit(ln, " +")[[1]]
    out[[tok[1]]][[tok[2]]] <- as.numeric(tok[-(1:2)])
  }
  out
}
jl <- read_julia(julia_params)

cfg <- list(zi_poisson = c(seed = 20260927L, n = 120L),
            zi_nbinom2 = c(seed = 20260928L, n = 200L),
            zi_binomial = c(seed = 20260929L, n = 120L))

for (family in names(cfg)) {
  dat <- simulate_family(family, cfg[[family]][["seed"]], cfg[[family]][["n"]])
  csv <- paste0(family, "_p1_data.csv")
  ## Written BEFORE fitting so the dataset is reproducible independent of the fit.
  write.csv(dat, csv, row.names = FALSE)
  fit <- fit_family(family, dat)
  fid <- c(zi_poisson = 17L, zi_nbinom2 = 18L, zi_binomial = 19L)[[family]]
  stopifnot(identical(fit$opt$convergence, 0L), fit$tmb_data$family_id_vec[1] == fid)

  par <- fit$opt$par
  pn <- names(par)
  ll <- as.numeric(logLik(fit))
  ## Recompute the marginal at the optimum through the TMB objective: this is the
  ## function Julia's cross-objective evaluates, and must equal logLik().
  ll_fn <- -as.numeric(fit$tmb_obj$fn(par))
  cat("\n[", family, ".r_reference]\n", sep = "")
  cat("convergence = ", fit$opt$convergence, "\n", sep = "")
  cat("loglik = ", fmt(ll), "\n", sep = "")
  cat("loglik_fn_at_opt = ", fmt(ll_fn), "\n", sep = "")
  cat("beta = ", vec(par[pn == "b_fix"]), "\n", sep = "")
  cat("theta_rr_B = ", vec(par[pn == "theta_rr_B"]), "\n", sep = "")
  cat("Lambda_B = ", vec(as.numeric(fit$report$Lambda_B)), "\n", sep = "")
  cat("logit_zi = ", vec(par[pn == "logit_zi"]), "\n", sep = "")
  cat("zi = ", vec(as.numeric(fit$report$zi)), "\n", sep = "")
  if (family == "zi_nbinom2")
    cat("phi = ", vec(as.numeric(fit$report$phi_nbinom2)), "\n", sep = "")

  if (!is.null(jl) && !is.null(jl[[family]])) {
    j <- jl[[family]]
    par_j <- par
    par_j[pn == "b_fix"] <- j$beta
    par_j[pn == "theta_rr_B"] <- j$theta_rr_B
    par_j[pn == "logit_zi"] <- j$logit_zi
    if (family == "zi_nbinom2") par_j[pn == "log_phi_nbinom2"] <- j$log_phi
    ll_rj <- -as.numeric(fit$tmb_obj$fn(par_j))
    cat("\n[", family, ".r_at_julia]\n", sep = "")
    cat("julia_loglik = ", fmt(j$loglik), "\n", sep = "")
    cat("r_objective_at_julia_optimum = ", fmt(ll_rj), "\n", sep = "")
  }
}
cat("\n# DONE\n")
