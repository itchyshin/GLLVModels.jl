#!/usr/bin/env Rscript
# True-parity campaign (clauses C3 and C4), data generator. Signed itchyshin/GLLVModels.jl#684 item 4.
#
# Writes ONE literal data file per cell under <outdir>/data (default ./data) and a
# <cell>.meta.json beside it (seed, dimensions, source, sha256 of the CSV). Both engines then read
# those exact bytes, so a paired number cannot drift with either engine's RNG. Synthetic cells are
# seeded here (R's RNG); real-data cells are loaded BY NAME from the installed package at run time
# and are never copied into the repository (GPL data in an MIT repo, plan section 1.3): the repository
# tracks this loader, the dataset sha256 recorded in each receipt, and the receipts only.
#
# The seven SYNTHETIC files this generator wrote for the campaign run are committed (gzip) in
# tools/true_parity/campaign/data/ with their sha256 in data_sha256.json: another host's R random
# streams give different bytes (a regeneration on the Mac differed), so verify the pin against those files.
# Usage: Rscript gen_data.R <cell> [outdir]
#   cell in gaussian poisson nb2 binomial ordinal temporal isdm   (C3, synthetic, n = 500)
#           crabs spider beetle fungi urban                       (C4, real data)
# Env for the reduced-size development runs only (never used for a receipt): CAMPAIGN_SMALL=1.
#
# Real-data rules, fixed BEFORE any campaign run:
#   crabs   MASS::crabs, 200 x 5, traits log(FL, RW, CL, CW, BD); sex x species group (4 levels).
#   spider  gllvm::eSpider. Its covariates are measured at 28 of the 100 sites only (the `nonNA`
#           vector); the plan's "100 sites x 12 species with covariates" is not available. Rule: the
#           28 sites with covariates, all 12 species, 3 covariates ConWate, BareSand, CovMoss, scaled.
#   beetle  gllvm::beetle, 87 sites, species with a positive total (68), covariates pH, Moist, Org, scaled.
#   fungi   gllvm::fungi presence/absence. Species with prevalence in [0.05, 0.60] over all 1666 sites,
#           most prevalent first, at most 60 (59 qualify); 300 sites by a seeded draw (seed 20261004);
#           covariates TEMPR, PRECIP, log.AREA, scaled.
#   urban   the maintainer's own binary model matrix (path in env URBMAP_ROOT), items as in
#           tools/wedge_a_acc_urbanisation_scout.R. Never copied, never tracked.
args <- commandArgs(trailingOnly = TRUE)
cell <- args[1]
outdir <- if (length(args) >= 2) args[2] else "."
dd <- file.path(outdir, "data"); dir.create(dd, recursive = TRUE, showWarnings = FALSE)
SMALL <- nzchar(Sys.getenv("CAMPAIGN_SMALL"))
sha <- function(f) sub(" .*$", "", system2(if (Sys.info()[["sysname"]] == "Darwin") "shasum" else "sha256sum",
                                          if (Sys.info()[["sysname"]] == "Darwin") c("-a", "256", shQuote(f)) else shQuote(f), stdout = TRUE))
lower_tri_loadings <- function(p, K, sd) { L <- matrix(rnorm(p * K, 0, sd), p, K); L[upper.tri(L)] <- 0; L }
long_basic <- function(Y) {  # Y is n x p
  n <- nrow(Y); p <- ncol(Y)
  data.frame(site = rep(seq_len(n), each = p), trait = rep(sprintf("t%02d", seq_len(p)), times = n),
             value = as.vector(t(Y)))
}
meta <- list(cell = cell, small = SMALL)
if (cell %in% c("gaussian", "poisson", "nb2", "binomial", "ordinal")) {
  seeds <- c(gaussian = 20261011L, poisson = 20261012L, nb2 = 20261013L, binomial = 20261014L, ordinal = 20261001L)
  p <- if (SMALL) 6L else 20L; n <- if (SMALL) 80L else 500L; K <- 2L
  set.seed(seeds[[cell]])
  meta$seed <- seeds[[cell]]; meta$p <- p; meta$n <- n; meta$K <- K
  if (cell == "gaussian") {
    alpha <- rnorm(p, 0, 0.5); L <- lower_tri_loadings(p, K, 0.6); Z <- matrix(rnorm(n * K), n, K)
    Y <- matrix(alpha, n, p, byrow = TRUE) + Z %*% t(L) + matrix(rnorm(n * p, 0, 0.5), n, p)
    meta$dgp <- "value = alpha_t + Z Lambda' + eps; alpha ~ N(0, .5), Lambda lower-triangular N(0, .6), sigma_eps = 0.5"
  } else if (cell == "poisson") {
    beta <- log(2 + 3 * runif(p)); L <- lower_tri_loadings(p, K, 0.35); Z <- matrix(rnorm(n * K), n, K)
    eta <- matrix(beta, n, p, byrow = TRUE) + Z %*% t(L)
    Y <- matrix(rpois(n * p, exp(pmin(pmax(eta, -8), 8))), n, p)
    meta$dgp <- "Poisson, log link, beta_t = log(2 + 3 U), Lambda lower-triangular N(0, .35)"
  } else if (cell == "nb2") {
    beta <- log(2 + 3 * runif(p)); L <- lower_tri_loadings(p, K, 0.35); Z <- matrix(rnorm(n * K), n, K)
    r <- 3 + 2 * runif(p); eta <- matrix(beta, n, p, byrow = TRUE) + Z %*% t(L)
    Y <- matrix(rnbinom(n * p, mu = exp(pmin(pmax(eta, -8), 8)), size = rep(r, each = n)), n, p)
    meta$dgp <- "NB2, log link, per-trait size r_t = 3 + 2 U, beta_t = log(2 + 3 U), Lambda lower-triangular N(0, .35)"
  } else if (cell == "binomial") {
    beta <- 0.4 * rnorm(p); L <- lower_tri_loadings(p, K, 0.35); Z <- matrix(rnorm(n * K), n, K)
    eta <- matrix(beta, n, p, byrow = TRUE) + Z %*% t(L)
    Y <- matrix(rbinom(n * p, 1, plogis(pmin(pmax(eta, -8), 8))), n, p)
    meta$dgp <- "Bernoulli logit, beta_t ~ N(0, .4), Lambda lower-triangular N(0, .35)"
  } else {
    alpha <- rnorm(p, 0, 0.4); L <- lower_tri_loadings(p, K, 0.8); taus <- c(0, 0.7, 1.4)
    Z <- matrix(rnorm(n * K), n, K); eta <- matrix(alpha, n, p, byrow = TRUE) + Z %*% t(L)
    ys <- eta + matrix(rlogis(n * p), n, p)
    Y <- 1L + (ys > taus[1]) + (ys > taus[2]) + (ys > taus[3])
    meta$dgp <- "ordinal logit, 4 categories, thresholds 0, .7, 1.4 on the latent scale, alpha_t ~ N(0, .4), Lambda lower-triangular N(0, .8)"
  }
  d <- long_basic(Y); d$value <- if (cell %in% c("poisson", "nb2", "binomial", "ordinal")) as.integer(d$value) else d$value
  f <- file.path(dd, paste0(cell, ".csv"))
} else if (cell == "temporal") {
  set.seed(20261002L); p <- if (SMALL) 6L else 20L; ns <- if (SMALL) 6L else 25L; nt <- if (SMALL) 8L else 20L
  beta <- rnorm(p, 0, 0.5); L <- matrix(rnorm(p, 0, 0.7), p, 1); sigma <- 0.6; phi <- 0.6
  meta$seed <- 20261002L; meta$p <- p; meta$n_series <- ns; meta$n_time <- nt; meta$K <- 1L
  meta$dgp <- "Gaussian, AR(1) latent (phi = 0.6), rank 1 (the only rank P1 admits for temporal), sigma_eps = 0.6"
  rows <- vector("list", ns)
  for (g in seq_len(ns)) {
    z <- numeric(nt); z[1] <- rnorm(1); for (k in 2:nt) z[k] <- phi * z[k - 1] + sqrt(1 - phi^2) * rnorm(1)
    dd1 <- expand.grid(trait = sprintf("t%02d", seq_len(p)), occasion = seq_len(nt), KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    ti <- as.integer(sub("t", "", dd1$trait))
    dd1$value <- beta[ti] + L[ti, 1] * z[dd1$occasion] + sigma * rnorm(nrow(dd1))
    dd1$series <- sprintf("g%02d", g)
    rows[[g]] <- dd1[, c("series", "occasion", "trait", "value")]
  }
  d <- do.call(rbind, rows); f <- file.path(dd, "temporal.csv")
} else if (cell == "isdm") {
  set.seed(20261003L); p <- if (SMALL) 6L else 20L; n <- if (SMALL) 80L else 500L; K <- 2L
  meta$seed <- 20261003L; meta$p <- p; meta$n <- n; meta$K <- K
  meta$dgp <- "two sources: gbif Poisson count (support 1.5) and survey presence/absence cloglog (support 0.9); env slope per species; rank-2 latent"
  sp <- sprintf("sp%02d", seq_len(p)); cells <- sprintf("c%03d", seq_len(n))
  env <- as.numeric(scale(runif(n))); alpha <- rnorm(p, -0.2, 0.3); b <- rnorm(p, 0, 0.3)
  L <- lower_tri_loadings(p, K, 0.5); Z <- matrix(rnorm(n * K), n, K); U <- Z %*% t(L)
  mk <- function(src, kind, support) {
    d <- expand.grid(cell_id = cells, trait = sp, stringsAsFactors = FALSE)
    ci <- match(d$cell_id, cells); si <- match(d$trait, sp)
    eta <- alpha[si] + b[si] * env[ci] + U[cbind(ci, si)] + if (kind == "count") 0.3 else 0
    d$isdm_source <- src; d$support <- support; d$env <- env[ci]
    d$value <- if (kind == "count") rpois(nrow(d), support * exp(eta)) else rbinom(nrow(d), 1, -expm1(-support * exp(eta)))
    d
  }
  d <- rbind(mk("gbif", "count", 1.5), mk("survey", "pa", 0.9))
  d$log_support <- log(d$support); d$src_gbif <- as.integer(d$isdm_source == "gbif")
  f <- file.path(dd, "isdm.csv")
} else if (cell == "crabs") {
  suppressPackageStartupMessages(library(MASS)); data(crabs, package = "MASS")
  meta$source <- sprintf("MASS::crabs (MASS %s), GPL-2 or GPL-3; loaded by name, not tracked", as.character(packageVersion("MASS")))
  d <- data.frame(specimen = seq_len(nrow(crabs)), sp = as.character(crabs$sp), sex = as.character(crabs$sex),
                  FL = log(crabs$FL), RW = log(crabs$RW), CL = log(crabs$CL), CW = log(crabs$CW), BD = log(crabs$BD))
  d$grp <- paste0(d$sex, ".", d$sp)
  f <- file.path(dd, "crabs.csv")
} else if (cell == "spider") {
  suppressPackageStartupMessages(library(gllvm)); data(eSpider, package = "gllvm")
  meta$source <- sprintf("gllvm::eSpider (gllvm %s), GPL-2; loaded by name, not tracked", as.character(packageVersion("gllvm")))
  keep <- sort(as.integer(eSpider$nonNA)); Y <- as.matrix(eSpider$abund)[keep, ]
  stopifnot(!anyNA(eSpider$X[keep, c("ConWate", "BareSand", "CovMoss")]))
  Xs <- scale(eSpider$X[keep, c("ConWate", "BareSand", "CovMoss")]); colnames(Xs) <- c("ConWate", "BareSand", "CovMoss")
  z <- colSums(Y) == 0; cat("spider species absent from the 28 sites (dropped):", sum(z), "\n"); Y <- Y[, !z, drop = FALSE]
  colnames(Y) <- sprintf("sp%02d", seq_len(ncol(Y)))
  d <- data.frame(site = seq_len(nrow(Y)), Xs, Y, check.names = FALSE); f <- file.path(dd, "spider_wide.csv")
  meta$rule <- "28 sites with covariates (eSpider$nonNA), all species with positive total, covariates ConWate BareSand CovMoss scaled"
} else if (cell == "beetle") {
  suppressPackageStartupMessages(library(gllvm)); data(beetle, package = "gllvm")
  meta$source <- sprintf("gllvm::beetle (gllvm %s), GPL-2; loaded by name, not tracked", as.character(packageVersion("gllvm")))
  Y <- as.matrix(beetle$Y); keep <- colSums(Y) > 0; cat("beetle species kept:", sum(keep), "of", ncol(Y), "\n")
  Y <- Y[, keep]; colnames(Y) <- sprintf("sp%02d", seq_len(ncol(Y)))
  Xs <- scale(beetle$X[, c("pH", "Moist", "Org")]); colnames(Xs) <- c("pH", "Moist", "Org")
  d <- data.frame(site = seq_len(nrow(Y)), Xs, Y, check.names = FALSE); f <- file.path(dd, "beetle_wide.csv")
} else if (cell == "fungi") {
  suppressPackageStartupMessages(library(gllvm)); data(fungi, package = "gllvm")
  meta$source <- sprintf("gllvm::fungi (gllvm %s), GPL-2; loaded by name, not tracked", as.character(packageVersion("gllvm")))
  Y <- as.matrix(fungi$Y); X <- fungi$X
  prev <- colMeans(Y > 0); ok <- which(prev >= 0.05 & prev <= 0.60)
  sel <- ok[order(-prev[ok])]; sel <- sel[seq_len(min(60L, length(sel)))]
  set.seed(20261004L); sites <- sort(sample(nrow(Y), 300L)); meta$seed <- 20261004L
  Yn <- (Y[sites, sel] > 0) * 1L; z <- colSums(Yn) == 0
  cat("fungi species selected:", length(sel), " absent from the 300 sites (dropped):", sum(z), "\n")
  Yn <- Yn[, !z, drop = FALSE]; sel <- sel[!z]; colnames(Yn) <- sprintf("sp%02d", seq_len(ncol(Yn)))
  Xs <- scale(X[sites, c("TEMPR", "PRECIP", "log.AREA")]); colnames(Xs) <- c("TEMPR", "PRECIP", "logAREA")
  d <- data.frame(site = seq_along(sites), Xs, Yn, check.names = FALSE); f <- file.path(dd, "fungi_wide.csv")
  meta$rule <- "species with prevalence in [.05,.60] (59 qualify), 300 sites by seeded draw, covariates TEMPR PRECIP log.AREA scaled"
} else if (cell == "urban") {
  root <- Sys.getenv("URBMAP_ROOT", "/Users/z3437171/Dropbox/Github Local/urbanisation_map")
  rds <- file.path(root, "data/processed/model_matrix_primary.rds"); stopifnot(file.exists(rds))
  Mp <- readRDS(rds); cols_pri <- setdiff(names(Mp), c("review_id", "level_individual"))
  mpf <- file.path(root, "outputs/tables/main_pruning.csv")
  if (file.exists(mpf)) { mp <- read.csv(mpf, stringsAsFactors = FALSE); seven <- mp$indicator[mp$consensus_pruned]
    items <- setdiff(cols_pri, c(seven, "level_ecosystem")) } else items <- setdiff(cols_pri, "level_ecosystem")
  meta$source <- "the maintainer's own urbanisation_map model_matrix_primary.rds; local file, not tracked, redistribution unconfirmed"
  meta$source_rds_sha256 <- sha(rds)
  d <- data.frame(review = seq_len(nrow(Mp))); for (i in seq_along(items)) d[[sprintf("it%02d", i)]] <- as.integer(Mp[[items[i]]])
  meta$item_names_hash_note <- "item columns it01..itNN follow the order of `items` (names not copied)"
  f <- file.path(dd, "urban_wide.csv")
} else stop("unknown cell ", cell)
if (SMALL) f <- sub("\\.csv$", "_small.csv", f)
write.csv(d, f, row.names = FALSE)
meta$file <- basename(f); meta$rows <- nrow(d); meta$cols <- ncol(d); meta$csv_sha256 <- sha(f)
meta$R_version <- R.version.string
jsonlite::write_json(meta, sub("\\.csv$", ".meta.json", f), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("WROTE", f, nrow(d), "rows sha256", meta$csv_sha256, "\n")
