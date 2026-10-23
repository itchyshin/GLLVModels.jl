# Generating script for the iSDM unit-level unique-variance twins
# (R's default `latent(..., unique = TRUE)`, TMB parameter `theta_diag_B`).
#
# Pin: gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9 (P1), installed into a
# temporary library from a detached worktree; R reads nothing else.
#
#   git -C <gllvmTMB> worktree add --detach <scratch>/gllvmTMB-p1 9539352f6
#   R CMD INSTALL --library=<scratch>/Rlib <scratch>/gllvmTMB-p1
#   GLLVMTMB_P1_LIB=<scratch>/Rlib GLLVMTMB_P1_SRC=<scratch>/gllvmTMB-p1 \
#     Rscript test/fixtures/isdm/export_psi_fixtures.R <stage>
#
# Stages (run in this order, from the repository root):
#   fixtures  write isdm_psi4.csv ONCE (four traits, so the one-factor model
#             with a per-trait unique variance is identified: p >= 2K + 1)
#   fits      read the CSVs back, fit each case in R with the default
#             `unique = TRUE`, write r_values_psi_p1.toml
#   xobj      read julia_estimates_psi_p1.toml (written by
#             export_julia_psi_estimates.jl) and evaluate R's own objective at
#             Julia's estimate; appends to r_values_psi_p1.toml
#
# Cases: `psi4` (the new identified fixture) and the default-formula fits of the
# two latent fixtures of export_p1_fixtures.R (`predict_default`,
# `ms3_default`), which have two traits, so theta_diag_B is not identified and
# runs toward the boundary; those are recorded as a documented case.

p1_lib <- Sys.getenv("GLLVMTMB_P1_LIB")
p1_src <- Sys.getenv("GLLVMTMB_P1_SRC")
if (!nzchar(p1_lib) || !nzchar(p1_src)) {
  stop("Set GLLVMTMB_P1_LIB and GLLVMTMB_P1_SRC.")
}
.libPaths(c(p1_lib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
P1_SHA <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
src_sha <- system2("git", c("-C", shQuote(p1_src), "rev-parse", "HEAD"), stdout = TRUE)
if (!identical(src_sha, P1_SHA)) stop("GLLVMTMB_P1_SRC is not at P1: ", src_sha)

stage <- commandArgs(trailingOnly = TRUE)[1]
out_dir <- file.path("test", "fixtures", "isdm")
stopifnot(dir.exists(out_dir))

## ---------------------------------------------------------------------------
## The identified fixture: 4 traits x 80 cells x 3 sources (two count arms
## with their own reporting rates, one detection arm), one latent factor plus
## a per-trait unit-level unique effect shared by the trait's rows in a cell.
## Seed: the first of 20260927-20260930 whose default R fit put every unique
## SD in the interior at n_cell = 80 (seeds 20260927 and 20260928 put one
## trait's SD at the boundary, a sampling outcome; recorded so the choice is
## visible).
## ---------------------------------------------------------------------------
.isdm_psi_fixture <- function(seed = 20260929L, n_cell = 80L) {
  set.seed(seed)
  cells <- paste0("c", seq_len(n_cell))
  traits <- c("sp1", "sp2", "sp3", "sp4")
  env <- as.numeric(scale(stats::runif(n_cell)))
  alpha <- c(-0.2, 0.1, 0.3, -0.1)
  beta <- c(0.4, -0.3, 0.2, 0.1)
  lambda <- c(0.8, 0.6, -0.5, 0.4)
  psi_sd <- c(0.5, 0.4, 0.6, 0.3)
  u <- stats::rnorm(n_cell)
  s <- matrix(stats::rnorm(length(traits) * n_cell, sd = rep(psi_sd, n_cell)),
              nrow = length(traits))
  rate <- c(gbif = 0.2, inat = -0.3, survey = 0)
  support <- c(gbif = 1.5, inat = 1.0, survey = 0.9)
  make_source <- function(src) {
    d <- expand.grid(cell_id = cells, trait = traits, stringsAsFactors = FALSE)
    ci <- match(d$cell_id, cells); ti <- match(d$trait, traits)
    eta <- alpha[ti] + beta[ti] * env[ci] + lambda[ti] * u[ci] + s[cbind(ti, ci)] + rate[[src]]
    d$isdm_source <- src
    d$support <- support[[src]]
    d$value <- if (src == "survey") stats::rbinom(nrow(d), 1L, -expm1(-d$support * exp(eta)))
               else stats::rpois(nrow(d), d$support * exp(eta))
    d
  }
  dat <- rbind(make_source("gbif"), make_source("inat"), make_source("survey"))
  dat$env <- env[match(dat$cell_id, cells)]
  dat$log_support <- log(dat$support)
  dat$src_gbif <- as.integer(dat$isdm_source == "gbif")
  dat$src_inat <- as.integer(dat$isdm_source == "inat")
  dat
}

fixture_files <- c(psi4 = "isdm_psi4.csv", predict_default = "isdm_predict.csv",
                   ms3_default = "isdm_ms3.csv")

read_fixture <- function(case) {
  d <- utils::read.csv(file.path(out_dir, fixture_files[[case]]), stringsAsFactors = FALSE)
  for (v in intersect(c("trait", "cell_id", "src"), names(d))) d[[v]] <- factor(d[[v]])
  lv <- switch(case,
    psi4 = c("gbif", "inat", "survey"),
    predict_default = c("gbif", "survey"),
    ms3_default = c("gbif", "literature", "survey"))
  d$isdm_source <- factor(d$isdm_source, levels = lv)
  if (case == "ms3_default") d$src <- factor(d$src, levels = lv)
  d
}

## The R calls. Every latent() term uses R's default `unique = TRUE`.
cases <- list(
  psi4 = list(
    family = function() isdm_sources(gbif = poisson(), inat = poisson(),
                                     survey = binomial(link = "cloglog")),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + trait:src_inat +
      offset(log_support) + latent(0 + trait | cell_id, d = 1),
    predict = TRUE),
  predict_default = list(
    family = function() isdm_sources(gbif = poisson(), survey = binomial(link = "cloglog")),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) +
      latent(0 + trait | cell_id, d = 1),
    predict = FALSE),
  ms3_default = list(
    family = function() isdm_sources(gbif = poisson(), literature = poisson(),
                                     survey = binomial(link = "cloglog")),
    formula = value ~ 0 + trait + trait:env + trait:src + offset(log_support) +
      latent(0 + trait | cell_id, d = 1),
    predict = FALSE)
)

fit_case <- function(case) {
  d <- read_fixture(case)
  fit <- suppressWarnings(suppressMessages(gllvmTMB(
    cases[[case]]$formula, data = d, trait = "trait", unit = "cell_id",
    family = cases[[case]]$family(), silent = TRUE)))
  list(fit = fit, dat = d)
}

fmt_num <- function(x) {
  vapply(x, function(v) {
    if (is.na(v)) "nan" else if (is.infinite(v)) (if (v > 0) "inf" else "-inf")
    else sprintf("%.17g", v)
  }, character(1))
}
fmt_str <- function(x) paste0('"', gsub('"', '\\\\"', x), '"')
kv_num <- function(k, v) paste0(k, " = ", fmt_num(v))
kv_str <- function(k, v) paste0(k, " = ", fmt_str(v))
kv_arr <- function(k, v) paste0(k, " = [", paste(fmt_num(v), collapse = ", "), "]")
kv_sarr <- function(k, v) paste0(k, " = [", paste(fmt_str(v), collapse = ", "), "]")
sha256_file <- function(path) sub(" .*$", "", system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE))

if (identical(stage, "fixtures")) {
  path <- file.path(out_dir, fixture_files[["psi4"]])
  if (file.exists(path)) stop("Refusing to overwrite ", path, ": fixtures are exported once.")
  utils::write.csv(.isdm_psi_fixture(), path, row.names = FALSE)
  cat("psi4", nrow(utils::read.csv(path)), sha256_file(path), "\n")
} else if (identical(stage, "fits")) {
  lines <- c(
    "# Recorded R values for the iSDM unique-variance twins (latent(..., unique = TRUE)).",
    "# Written by test/fixtures/isdm/export_psi_fixtures.R (stages fits, xobj). Do not edit.",
    kv_str("gllvmtmb_sha", P1_SHA),
    kv_str("gllvmtmb_version", as.character(packageVersion("gllvmTMB"))),
    kv_str("tmb_version", as.character(packageVersion("TMB"))),
    kv_str("r_version", R.version.string),
    kv_str("written", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")))
  for (case in names(cases)) {
    t0 <- proc.time()[["elapsed"]]
    fx <- fit_case(case)
    wall <- proc.time()[["elapsed"]] - t0
    fit <- fx$fit; d <- fx$dat
    par <- fit$opt$par
    L <- as.matrix(fit$report$Lambda_B)
    lines <- c(lines, "", paste0("[cases.", case, "]"),
      kv_str("fixture", fixture_files[[case]]),
      kv_str("fixture_sha256", sha256_file(file.path(out_dir, fixture_files[[case]]))),
      kv_num("fixture_rows", nrow(d)),
      kv_str("formula", paste(deparse(cases[[case]]$formula, width.cutoff = 500L), collapse = " ")),
      kv_num("loglik", -fit$opt$objective),
      kv_num("convergence", fit$opt$convergence),
      kv_str("message", fit$opt$message),
      kv_num("iterations", fit$opt$iterations),
      kv_str("pdHess", as.character(isTRUE(fit$sd_report$pdHess))),
      kv_num("max_abs_gradient", max(abs(fit$tmb_obj$gr(par)))),
      kv_num("wall_seconds", wall),
      kv_sarr("par_names", names(par)),
      kv_sarr("random_names", unique(names(fit$tmb_obj$env$par)[fit$tmb_obj$env$random])),
      kv_arr("diag_B_skip", fit$tmb_data$diag_B_skip),
      kv_sarr("b_fix_names", fit$X_fix_names),
      kv_arr("b_fix", par[names(par) == "b_fix"]),
      kv_arr("theta_rr_B", par[names(par) == "theta_rr_B"]),
      kv_arr("theta_diag_B", par[names(par) == "theta_diag_B"]),
      kv_arr("sd_B", fit$report$sd_B),
      kv_num("K", ncol(L)),
      kv_arr("Lambda_B_colmajor", as.numeric(L)),
      kv_arr("LLt_colmajor", as.numeric(L %*% t(L))),
      kv_arr("eta", as.numeric(fit$report$eta)))
    pol <- stats::nlminb(par, fit$tmb_obj$fn, fit$tmb_obj$gr,
                         control = list(rel.tol = 1e-14, x.tol = 1e-12,
                                        iter.max = 2000L, eval.max = 4000L))
    lines <- c(lines,
      kv_num("polished_loglik", -pol$objective),
      kv_num("polished_convergence", pol$convergence),
      kv_str("polished_message", pol$message),
      kv_num("polished_max_abs_gradient", max(abs(fit$tmb_obj$gr(pol$par)))),
      kv_arr("polished_b_fix", pol$par[names(pol$par) == "b_fix"]),
      kv_arr("polished_theta_rr_B", pol$par[names(pol$par) == "theta_rr_B"]),
      kv_arr("polished_theta_diag_B", pol$par[names(pol$par) == "theta_diag_B"]))
    if (isTRUE(cases[[case]]$predict)) {
      nd0 <- d; nd0$log_support <- 0
      ## Unseen units: the first 12 rows moved to a unit absent at fit time.
      ndu <- d; ndu$cell_id <- as.character(ndu$cell_id); ndu$cell_id[1:12] <- "c_new"
      lines <- c(lines,
        kv_sarr("predict_columns", names(predict(fit))),
        kv_arr("predict_link", predict(fit)$est),
        kv_arr("predict_response", predict(fit, type = "response")$est),
        kv_arr("predict_link_zero_re", suppressMessages(predict(fit, re_form = ~0))$est),
        kv_arr("predict_newdata_offset0_link",
               suppressMessages(predict(fit, newdata = nd0, type = "link"))$est),
        kv_arr("predict_newdata_offset0_response",
               suppressMessages(predict(fit, newdata = nd0, type = "response"))$est),
        kv_arr("predict_newdata_unseen_link",
               suppressMessages(predict(fit, newdata = ndu, type = "link"))$est))
    }
    cat(case, "loglik", -fit$opt$objective, "conv", fit$opt$convergence,
        "sd_B", fit$report$sd_B, "\n")
  }
  writeLines(lines, file.path(out_dir, "r_values_psi_p1.toml"))
} else if (identical(stage, "xobj")) {
  est_path <- file.path(out_dir, "julia_estimates_psi_p1.toml")
  stopifnot(file.exists(est_path))
  est <- readLines(est_path)
  get_arr <- function(section, key) {
    i0 <- which(est == paste0("[", section, "]"))
    stopifnot(length(i0) == 1L)
    nxt <- which(grepl("^\\[", est) & seq_along(est) > i0)
    i1 <- if (length(nxt)) min(nxt) - 1L else length(est)
    ln <- grep(paste0("^", key, " = "), est[i0:i1], value = TRUE)
    stopifnot(length(ln) == 1L)
    trimws(strsplit(sub("^[^=]*= \\[(.*)\\]$", "\\1", ln), ",")[[1]])
  }
  lines <- readLines(file.path(out_dir, "r_values_psi_p1.toml"))
  add <- c("", "[xobj]", "# R's objective (TMB Laplace, sign flipped to a log-likelihood) at Julia's estimate")
  for (case in names(cases)) {
    sec <- paste0("julia.", case)
    jn <- gsub('"', "", get_arr(sec, "b_fix_names"))
    jb <- as.numeric(get_arr(sec, "b_fix"))
    fit <- fit_case(case)$fit
    rn <- fit$X_fix_names
    if (!setequal(jn, rn)) stop("b_fix names do not pair for ", case)
    par <- fit$opt$par
    stopifnot(all(names(par) %in% c("b_fix", "theta_rr_B", "theta_diag_B")))
    par[names(par) == "b_fix"] <- jb[match(rn, jn)]
    par[names(par) == "theta_rr_B"] <- as.numeric(get_arr(sec, "theta_rr_B"))
    par[names(par) == "theta_diag_B"] <- as.numeric(get_arr(sec, "theta_diag_B"))
    add <- c(add, kv_num(case, -fit$tmb_obj$fn(par)))
    cat(case, "R loglik at Julia estimate", -fit$tmb_obj$fn(par), "\n")
  }
  add <- c(add, kv_str("julia_estimates_sha256", sha256_file(est_path)))
  keep <- if (any(lines == "[xobj]")) which(lines == "[xobj]") - 2L else length(lines)
  writeLines(c(lines[seq_len(keep)], add), file.path(out_dir, "r_values_psi_p1.toml"))
} else {
  stop("Unknown stage: ", stage)
}
