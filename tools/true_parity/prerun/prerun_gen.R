#!/usr/bin/env Rscript
# Pre-run data generation (campaign plan 4.3). Writes CSVs under ./data (never tracked).
# Usage: Rscript prerun_gen.R <cell>   cell in ordinal|temporal|isdm|beetle|fungi
.libPaths(c("/home/snakagaw/R/lib", .libPaths()))
cell <- commandArgs(trailingOnly = TRUE)[1]
dir.create("data", showWarnings = FALSE)
sha <- function(f) sub(" .*$", "", system2("sha256sum", shQuote(f), stdout = TRUE))
lower_tri_loadings <- function(p, K, sd) { L <- matrix(rnorm(p * K, 0, sd), p, K); L[upper.tri(L)] <- 0; L }
if (cell == "ordinal") {
  set.seed(20261001L); p <- 20L; n <- 500L; K <- 2L
  alpha <- rnorm(p, 0, 0.4); L <- lower_tri_loadings(p, K, 0.8); taus <- c(0, 0.7, 1.4)
  Z <- matrix(rnorm(n * K), n, K)
  eta <- matrix(alpha, n, p, byrow = TRUE) + Z %*% t(L)
  ystar <- eta + matrix(rlogis(n * p), n, p)
  Y <- 1L + (ystar > taus[1]) + (ystar > taus[2]) + (ystar > taus[3])
  d <- data.frame(site = rep(seq_len(n), each = p), trait = rep(sprintf("t%02d", seq_len(p)), times = n),
                  value = as.integer(as.vector(t(Y))))
  f <- "data/ordinal_p20_n500_K2.csv"
} else if (cell == "temporal") {
  set.seed(20261002L); p <- 20L; ns <- 25L; nt <- 20L; K <- 1L   # d = 1: the only rank admitted at P1 (both engines)
  beta <- rnorm(p, 0, 0.5); L <- matrix(rnorm(p, 0, 0.7), p, 1); sigma <- 0.6; phi <- 0.6
  rows <- vector("list", ns)
  for (g in seq_len(ns)) {
    z <- numeric(nt); z[1] <- rnorm(1); for (k in 2:nt) z[k] <- phi * z[k - 1] + sqrt(1 - phi^2) * rnorm(1)
    dd <- expand.grid(trait = sprintf("t%02d", seq_len(p)), occasion = seq_len(nt), KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    ti <- as.integer(sub("t", "", dd$trait))
    dd$value <- beta[ti] + L[ti, 1] * z[dd$occasion] + sigma * rnorm(nrow(dd))
    dd$series <- sprintf("g%02d", g)
    rows[[g]] <- dd[, c("series", "occasion", "trait", "value")]
  }
  d <- do.call(rbind, rows); f <- "data/temporal_s25_t20_p20_d1.csv"
} else if (cell == "isdm") {
  set.seed(20261003L); p <- 20L; n <- 500L; K <- 2L
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
  f <- "data/isdm_c500_sp20_s2_K2.csv"
} else if (cell == "beetle") {
  data(beetle, package = "gllvm"); Y <- as.matrix(beetle$Y); X <- beetle$X
  keep <- colSums(Y) > 0; cat("beetle species kept:", sum(keep), "of", ncol(Y), "\n")
  Y <- Y[, keep]; colnames(Y) <- sprintf("sp%02d", seq_len(ncol(Y)))
  Xs <- scale(X[, c("pH", "Moist", "Org")]); colnames(Xs) <- c("pH", "Moist", "Org")
  d <- data.frame(site = seq_len(nrow(Y)), Xs, Y, check.names = FALSE); f <- "data/beetle_wide.csv"
} else if (cell == "fungi") {
  data(fungi, package = "gllvm"); Y <- as.matrix(fungi$Y); X <- fungi$X
  prev <- colMeans(Y > 0); ok <- which(prev >= 0.05 & prev <= 0.60)
  sel <- ok[order(-prev[ok])]; sel <- sel[seq_len(min(60L, length(sel)))]   # plan rule says 60; only 59 species qualify at 5-60 percent prevalence
  set.seed(20261004L); sites <- sort(sample(nrow(Y), 300L))
  Yn <- Y[sites, sel]; Yn <- (Yn > 0) * 1L; z <- colSums(Yn) == 0; cat("species absent from the 300 sites (dropped):", sum(z), "\n"); Yn <- Yn[, !z, drop = FALSE]; sel <- sel[!z]; colnames(Yn) <- sprintf("sp%02d", seq_len(ncol(Yn)))
  Xs <- scale(X[sites, c("TEMPR", "PRECIP", "log.AREA")]); colnames(Xs) <- c("TEMPR", "PRECIP", "logAREA")
  cat("fungi subsample: prevalence range of chosen species", range(prev[sel]), "\n")
  d <- data.frame(site = seq_along(sites), Xs, Yn, check.names = FALSE); f <- "data/fungi_wide.csv"
  writeLines(c(paste(sel, collapse = ","), paste(sites, collapse = ",")), "data/fungi_subsample_rule_indices.txt")
}
write.csv(d, f, row.names = FALSE)
cat("WROTE", f, nrow(d), "rows sha256", sha(f), "\n")
