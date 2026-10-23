## Regenerates the CSVs and aghq_p1.toml in this directory. NOT run by CI or by any
## Julia test -- provenance for how the fixture was produced, against gllvmTMB pinned
## at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1"). Needs
## gllvmTMB installed at that exact commit in a lane-local library (git worktree at
## the pin, `R CMD INSTALL --library=<lib> <dir>`); set GLLVM_P1_RLIB to that library.
## Run from this directory:  GLLVM_P1_RLIB=<lib> Rscript gen_aghq_p1.R
##
## Model (all cases): value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
## i.e. eta_tj = beta_j + lambda_j z_t, z_t ~ N(0, 1): the structure of GLLVModels.jl's
## fit_<family>_gllvm(Y; K = 1). Gaussian adds one residual sd (unique = FALSE).
## Control mirrors tools/core070_aghq_public_policy_bind.R: n_init = 1, init_jitter = 0,
## se = FALSE, aghq_ridge = Inf (no loading penalty), aghq as named per case; the
## DEFAULT-OFF case uses gllvmTMBcontrol() defaults (aghq not named, no ridge named).
##
## Data are SIMULATED WITH A REAL LATENT FACTOR. The policy bind's toy data (eta ~
## N(0, 0.3^2) independent across traits, no factor structure) make the binomial AGHQ
## adaptation stall on both engines (R "stalled", Julia no_merit_descent; reported in
## itchyshin/GLLVModels.jl#586). Here every case must converge on R's own AGHQ verdict
## (fit$aghq$converged; Laplace fits: fit$opt$convergence == 0): asserted below.
## Binomial data use 10 trials per cell; with Bernoulli cells and p = 5 R's AGHQ
## adaptation does not reach its relative-gradient tolerance at any n tried (10 of 12
## trial fits stalled with n up to 500), so the fixture uses 10 trials to be well conditioned.
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
## The P1 library must be the one in use (a user-library gllvmTMB of the same version
## number would otherwise be picked up silently).
stopifnot(nzchar(rlib), startsWith(normalizePath(find.package("gllvmTMB")), normalizePath(rlib)))

## Compound Poisson-Gamma (Tweedie, 1 < power < 2) draws with mean mu, dispersion phi:
## N ~ Poisson(mu^(2-p) / (phi (2-p))), y = sum of N Gamma(shape (2-p)/(p-1), scale phi (p-1) mu^(p-1)).
rtweedie_cpg <- function(mu, phi, pw) {
  lam <- mu^(2 - pw) / (phi * (2 - pw)); shp <- (2 - pw) / (pw - 1); sc <- phi * (pw - 1) * mu^(pw - 1)
  N <- rpois(length(mu), lam); out <- numeric(length(mu))
  for (i in which(N > 0)) out[i] <- rgamma(1, shape = N[i] * shp, scale = sc[i])
  out
}
make_data <- function(fam, p, n, seed, lam_scale, size = 10L) {
  set.seed(seed)
  mu <- if (fam %in% c("poisson", "nb2", "tweedie")) seq(0.1, 0.9, length.out = p) else seq(-0.4, 0.6, length.out = p)
  lam <- lam_scale * rep(c(0.9, -0.7, 0.6, -0.5, 0.8), length.out = p)
  z <- rnorm(n)
  eta <- outer(z, lam) + matrix(mu, n, p, byrow = TRUE)
  Y <- switch(fam,
    poisson  = matrix(rpois(n * p, exp(eta)), n, p),
    nb2      = matrix(rnbinom(n * p, mu = exp(eta), size = rep(c(2, 3, 4, 2.5, 5), length.out = p)[col(eta)]), n, p),
    binomial = matrix(rbinom(n * p, size, plogis(eta)), n, p),
    gaussian = eta + matrix(rnorm(n * p, sd = 0.5), n, p),
    delta_gamma = {  # one shared eta drives both parts (gllvmTMB fid 13): P(y > 0) = plogis(eta), y | y > 0 ~ Gamma(mean exp(eta), CV phi_t)
      phi_t <- rep(c(0.6, 0.8, 0.5, 0.7, 0.9), length.out = p)[col(eta)]
      pos <- runif(n * p) < plogis(as.vector(eta))
      matrix(ifelse(pos, rgamma(n * p, shape = 1 / as.vector(phi_t)^2, scale = exp(as.vector(eta)) * as.vector(phi_t)^2), 0), n, p)
    },
    tweedie  = {  # per-trait dispersion and per-trait power, as gllvmTMB's tweedie() estimates them
      phi_t <- rep(c(1.2, 0.8, 1.5, 1.0, 1.3), length.out = p); pw_t <- rep(c(1.4, 1.6, 1.5, 1.3, 1.55), length.out = p)
      matrix(rtweedie_cpg(as.vector(exp(eta)), rep(phi_t, each = n), rep(pw_t, each = n)), n, p)
    },
    ordinal  = {  # probit threshold model: 4 ordered categories, y* = eta + N(0, 1)
      ystar <- eta + matrix(rnorm(n * p), n, p)
      Yo <- matrix(findInterval(ystar, c(-0.5, 0.5, 1.5)) + 1L, n, p)
      stopifnot(all(apply(Yo, 2, function(v) length(unique(v)) == 4L)))  # C = 4 on every trait
      Yo
    })
  tr <- paste0("t", seq_len(p))
  df <- data.frame(unit = factor(rep(seq_len(n), each = p), levels = seq_len(n)),
                   trait = factor(rep(tr, n), levels = tr), value = as.vector(t(Y)))
  if (fam == "binomial") df$trials <- size
  df$fail <- if (fam == "binomial") df$trials - df$value else NA
  list(df = df, p = p, n = n, seed = seed, lam_scale = lam_scale, fam = fam, trait_names = tr)
}

## Seed search: for each dataset the first seed (base + 0, 1, 2, ...) at which EVERY case on
## that dataset converges in R, and which Julia does not reject (julia_skip below), is kept.
## R-rejected seeds are recorded in the fixture as rejected_seeds, Julia-rejected ones as
## julia_rejected_seeds. GLLVM_AGHQ_SKIP_SEEDS="name:seed,name:seed" adds Julia rejections.
specs <- list(
  poisson_p5   = list(fam = "poisson",  p = 5L,  n = 150L, lam = 1.0),
  nb2_p5       = list(fam = "nb2",      p = 5L,  n = 150L, lam = 1.0),
  ordinal_p5   = list(fam = "ordinal",  p = 5L,  n = 150L, lam = 1.0),
  tweedie_p5   = list(fam = "tweedie",  p = 5L,  n = 150L, lam = 1.0),
  delta_gamma_p5 = list(fam = "delta_gamma", p = 5L, n = 150L, lam = 1.0),
  gaussian_p5  = list(fam = "gaussian", p = 5L,  n = 150L, lam = 1.0),
  binomial_p5  = list(fam = "binomial", p = 5L,  n = 100L, lam = 1.0),
  binomial_p19 = list(fam = "binomial", p = 19L, n = 100L, lam = 1.0),
  binomial_p20 = list(fam = "binomial", p = 20L, n = 100L, lam = 1.0))
seed_base <- c(poisson_p5 = 20260100L, nb2_p5 = 20260600L, ordinal_p5 = 20260700L, tweedie_p5 = 20260800L, delta_gamma_p5 = 20260900L, gaussian_p5 = 20260200L, binomial_p5 = 20260300L,
               binomial_p19 = 20260400L, binomial_p20 = 20260500L)
## Seeds at which R converged but the Julia fit did NOT (test/test_aghq_p1_twin.jl guards
## this; found by running the Julia fits on each candidate dataset). Julia's AGHQ adaptation
## stopped (reason no_merit_descent, logLik 2.8e-5 to 2.2e-4 ABOVE R's, parameters ~1e-3 off) on these Poisson datasets;
## both engines accept steps on the re-centred objective but certify on the frozen-node gradient, so R's optimum is a fixed point, not a minimum
## (see docs/dev-log/core070/true-parity-latest/aghq-fixed-point-note-2026-10-01.md),
## so they are not used. Recorded in the fixture as julia_rejected_seeds. Extend this list
## (never the test tolerances) if a regenerated dataset fails the Julia convergence guard.
julia_skip <- c("poisson_p5:20260103", "poisson_p5:20260107", "poisson_p5:20260110", "poisson_p5:20260111")
skip <- c(julia_skip, strsplit(Sys.getenv("GLLVM_AGHQ_SKIP_SEEDS", ""), ",")[[1]])

fam_obj <- function(f) switch(f, poisson = poisson(), gaussian = gaussian(), binomial = binomial(), nb2 = nbinom2(), ordinal = ordinal_probit(), tweedie = tweedie(), delta_gamma = delta_gamma())
fit_case <- function(ds, aghq, default_control = FALSE) {
  d <- ds$df
  ctrl <- if (default_control) gllvmTMBcontrol(n_init = 1L, init_jitter = 0, se = FALSE)
          else gllvmTMBcontrol(n_init = 1L, init_jitter = 0, se = FALSE, aghq = aghq, aghq_ridge = Inf)
  form <- if (ds$fam == "binomial")
    cbind(value, fail) ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE)
  else value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE)
  fit <- suppressWarnings(gllvmTMB(form, data = d, unit = "unit", family = fam_obj(ds$fam), control = ctrl))
  used <- isTRUE(fit$aghq$used)
  conv <- if (used) isTRUE(fit$aghq$converged) else identical(fit$opt$convergence, 0L)
  if (!conv) {
    message("NOT CONVERGED: ", ds$fam, " p=", ds$p, " seed=", ds$seed, " aghq=", format(aghq), " | ",
            as.character(fit$aghq$stop_reason), " | opt.convergence=", fit$opt$convergence)
    stop("a case did not converge; choose different simulated data (do not loosen the check)")
  }
  ll <- -unname(fit$opt$objective)
  stopifnot(isTRUE(all.equal(ll, as.numeric(logLik(fit)), tolerance = 1e-12)))
  par <- fit$opt$par
  list(used = used, k = if (used) as.integer(fit$aghq$k) else 0L, converged = conv, loglik = ll,
       beta = unname(par[names(par) == "b_fix"]), lambda = as.numeric(fit$report$Lambda_B),
       sigma_eps = if (ds$fam == "gaussian") as.numeric(fit$report$sigma_eps)[1] else NA_real_,
       phi = if (ds$fam == "nb2") unname(exp(par[names(par) == "log_phi_nbinom2"])) else if (ds$fam == "tweedie") unname(exp(par[names(par) == "log_phi_tweedie"])) else if (ds$fam == "delta_gamma") unname(exp(par[names(par) == "log_phi_gamma_delta"])) else NULL,
       power = if (ds$fam == "tweedie") unname(1 + plogis(par[names(par) == "logit_p_tweedie"])) else NULL,
       log_incr = if (ds$fam == "ordinal") unname(par[names(par) == "ordinal_log_increments"]) else NULL,
       reason = as.character(fit$aghq$reason), grad_rel = if (used) fit$aghq$grad_rel else NA_real_)
}

cases <- list(
  list(id = "AGHQ-AUTO-K-POISSON",   ds = "poisson_p5",   aghq = "auto", expect_used = TRUE,  expect_k = 5L),
  list(id = "AGHQ-AUTO-K-BINOMIAL",  ds = "binomial_p5",  aghq = "auto", expect_used = TRUE,  expect_k = 5L),
  list(id = "AGHQ-AUTO-K-NB2",       ds = "nb2_p5",       aghq = "auto", expect_used = TRUE,  expect_k = 5L),
  list(id = "AGHQ-AUTO-K-ORDINAL",   ds = "ordinal_p5",   aghq = "auto", expect_used = TRUE,  expect_k = 9L),
  list(id = "AGHQ-AUTO-K-TWEEDIE",   ds = "tweedie_p5",   aghq = "auto", expect_used = TRUE,  expect_k = 9L),
  list(id = "AGHQ-AUTO-K-DELTA",     ds = "delta_gamma_p5", aghq = "auto", expect_used = TRUE, expect_k = 5L),
  list(id = "AGHQ-AUTO-K-GAUSSIAN",  ds = "gaussian_p5",  aghq = "auto", expect_used = TRUE,  expect_k = 5L),
  list(id = "AGHQ-DEFAULT-OFF",      ds = "poisson_p5",   aghq = NA,     expect_used = FALSE, expect_k = 0L, default_control = TRUE),
  list(id = "AGHQ-POLICY-OFF",       ds = "binomial_p5",  aghq = FALSE,  expect_used = FALSE, expect_k = 0L),
  list(id = "AGHQ-POLICY-EXPLICIT",  ds = "binomial_p5",  aghq = 3L,     expect_used = TRUE,  expect_k = 3L),
  list(id = "AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF", ds = "binomial_p20", aghq = 9L, expect_used = TRUE, expect_k = 9L),
  list(id = "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF",    ds = "binomial_p20", aghq = "auto", expect_used = FALSE, expect_k = 0L),
  list(id = "AGHQ-POLICY-TRAITS19",  ds = "binomial_p19", aghq = "auto", expect_used = TRUE,  expect_k = 5L))

run_case <- function(cs, ds) {
  t0 <- Sys.time()
  r <- fit_case(ds, cs$aghq, isTRUE(cs$default_control))
  if (!identical(r$used, cs$expect_used) || !identical(r$k, cs$expect_k)) stop("unexpected aghq decision for ", cs$id)
  if (identical(cs$id, "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF") && !grepl("cutoff", r$reason, fixed = TRUE))
    stop("decline reason does not name the cutoff")
  r$seconds <- as.numeric(Sys.time() - t0, units = "secs")
  r
}
datasets <- list(); res <- list(); rejected <- list(); jrejected <- list()
for (nm in names(specs)) {
  sp <- specs[[nm]]; off <- 0L; rejected[[nm]] <- integer(); jrejected[[nm]] <- integer()
  repeat {
    seed <- seed_base[[nm]] + off
    if (sprintf("%s:%d", nm, seed) %in% skip) { jrejected[[nm]] <- c(jrejected[[nm]], seed); off <- off + 1L; next }
    ds <- make_data(sp$fam, sp$p, sp$n, seed, sp$lam)
    out <- tryCatch({
      rs <- list(); for (cs in cases) if (identical(cs$ds, nm)) rs[[cs$id]] <- run_case(cs, ds); rs
    }, error = function(e) NULL)
    if (!is.null(out)) { datasets[[nm]] <- ds; res <- c(res, out); break }
    rejected[[nm]] <- c(rejected[[nm]], seed); off <- off + 1L
    if (off > 60L) stop("no converging seed for ", nm)
  }
  cat(sprintf("%-14s seed %d (R-rejected: %s; Julia-rejected: %s)\n", nm, datasets[[nm]]$seed,
              paste(rejected[[nm]], collapse = " "), paste(jrejected[[nm]], collapse = " ")))
}
for (nm in names(datasets)) {
  d <- datasets[[nm]]$df
  write.csv(d[, intersect(c("unit", "trait", "value", "trials"), names(d))], sprintf("aghq_%s.csv", nm), row.names = FALSE)
}
for (id in names(res)) cat(sprintf("%-40s used=%s k=%d ll=%.10f %.1fs\n", id, res[[id]]$used, res[[id]]$k, res[[id]]$loglik, res[[id]]$seconds))

fmt <- function(x) sprintf("%.17g", x)
vec <- function(x) paste0("[", paste(vapply(x, fmt, ""), collapse = ", "), "]")
shaf <- function(f) strsplit(system2("shasum", c("-a", "256", f), stdout = TRUE), " ")[[1]][1]
con <- file("aghq_p1.toml", "w"); w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for the aghq policy rows (gllvmTMB 0.7.1 at P1). Generated once by")
w("# gen_aghq_p1.R from the CSVs in this directory (sha256 guarded). Do not hand-edit")
w("# the [case.*.r] blocks; regenerate them from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  w("")
  w("[dataset.%s]", nm)
  w("file = \"aghq_%s.csv\"", nm)
  w("sha256 = \"%s\"", shaf(sprintf("aghq_%s.csv", nm)))
  w("family = \"%s\"", d$fam)
  w("seed = %d", d$seed); w("p = %d", d$p); w("n_unit = %d", d$n)
  w("rejected_seeds = [%s]", paste(rejected[[nm]], collapse = ", "))
  w("julia_rejected_seeds = [%s]", paste(jrejected[[nm]], collapse = ", "))
  w("trait_names = [%s]", paste0("\"", d$trait_names, "\"", collapse = ", "))
}
for (cs in cases) {
  r <- res[[cs$id]]
  w("")
  w("[case.%s]", cs$id)
  w("dataset = \"%s\"", cs$ds)
  w("aghq_request = \"%s\"", if (isTRUE(cs$default_control)) "default" else if (is.character(cs$aghq)) cs$aghq else if (isFALSE(cs$aghq)) "false" else as.character(cs$aghq))
  w("")
  w("[case.%s.r]", cs$id)
  w("used = %s", tolower(r$used)); w("k = %d", r$k); w("converged = %s", tolower(r$converged))
  w("loglik = %s", fmt(r$loglik))
  w("beta = %s", vec(r$beta)); w("lambda = %s", vec(r$lambda))
  if (!is.na(r$sigma_eps)) w("sigma_eps = %s", fmt(r$sigma_eps))
  if (!is.null(r$phi)) w("phi = %s", vec(r$phi))
  if (!is.null(r$power)) w("power = %s", vec(r$power))
  if (!is.null(r$log_incr)) w("log_incr = %s", vec(r$log_incr))
  w("reason = \"%s\"", gsub("\"", "'", r$reason))
}
close(con)
