#!/usr/bin/env Rscript
# True-parity campaign (C3, C4), R side. Signed itchyshin/GLLVModels.jl#684 item 4.
# Fits one cell with the private P1 gllvmTMB (9539352f6, 0.7.1) and writes the RAW outputs to
# <out>/<cell>_R.json. Nothing here classifies, compares or signs anything; tools/true_parity/campaign/
# write_receipts.py reads this file and run_J.jl's TOML and writes the receipts.
#
# Usage: Rscript run_R.R <cell>      (env: GLLVMTMB_P1_LIB, GLLVMTMB_P1_SRC, CAMPAIGN_DATA, CAMPAIGN_OUT,
#                                     CAMPAIGN_SMALL=1 for the development runs, never a receipt)
# The P1 guard (below) refuses to run unless
#   (a) the loaded gllvmTMB is the library under GLLVMTMB_P1_LIB and reports version 0.7.1,
#   (b) every P1 source file listed in p1_source_sha256.json hashes to its recorded P1 value in
#       GLLVMTMB_P1_SRC (the recorded values are `git show 9539352f6:<file> | sha256`),
#   (c) every R function this script calls from the library deparses identically to its definition
#       in those P1 source files (installed byte-compiled function vs parse(text) of the P1 file).
args <- commandArgs(trailingOnly = TRUE); cell <- args[1]
lib <- Sys.getenv("GLLVMTMB_P1_LIB"); src <- Sys.getenv("GLLVMTMB_P1_SRC")
data_dir <- Sys.getenv("CAMPAIGN_DATA", "data"); out_dir <- Sys.getenv("CAMPAIGN_OUT", "out")
SMALL <- nzchar(Sys.getenv("CAMPAIGN_SMALL")); sfx <- if (SMALL) "_small" else ""
if (!nzchar(lib) || !nzchar(src)) stop("Set GLLVMTMB_P1_LIB and GLLVMTMB_P1_SRC.")
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({ library(gllvmTMB); library(jsonlite) })
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
script_dir <- (function() { a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)]); if (length(f)) dirname(normalizePath(f)) else "." })()

## ---- P1 guard --------------------------------------------------------------------------------
sha256 <- function(f) { if (Sys.info()[["sysname"]] == "Darwin") sub(" .*$", "", system2("shasum", c("-a", "256", shQuote(f)), stdout = TRUE))
                        else sub(" .*$", "", system2("sha256sum", shQuote(f), stdout = TRUE)) }
man <- fromJSON(file.path(script_dir, "p1_source_sha256.json"))
loaded <- normalizePath(find.package("gllvmTMB"))
if (!identical(loaded, normalizePath(file.path(lib, "gllvmTMB")))) stop("gllvmTMB loaded from ", loaded, ", not the P1 library")
if (!identical(as.character(packageVersion("gllvmTMB")), "0.7.1")) stop("gllvmTMB is not 0.7.1")
src_hash <- list()
for (f in names(man$files)) {
  got <- sha256(file.path(src, f)); src_hash[[f]] <- got
  if (!identical(got, man$files[[f]])) stop("P1 source file ", f, " sha256 ", got, " != recorded P1 value ", man$files[[f]])
}
options(keep.source = FALSE)
# (function name, defining file, how to fetch the installed object)
checks <- list(
  list("gllvmTMB", "R/gllvmTMB.R", function() gllvmTMB::gllvmTMB),
  list("gllvmTMBcontrol", "R/gllvmTMB.R", function() gllvmTMB::gllvmTMBcontrol),
  list("nbinom2", "R/families.R", function() gllvmTMB::nbinom2),
  list("ordinal_logit", "R/families.R", function() gllvmTMB::ordinal_logit),
  list("isdm_sources", "R/isdm-sources.R", function() gllvmTMB::isdm_sources),
  list("extract_Sigma", "R/extract-sigma.R", function() gllvmTMB::extract_Sigma),
  list("extract_cutpoints", "R/extract-cutpoints.R", function() gllvmTMB::extract_cutpoints),
  list("predict.gllvmTMB_multi", "R/methods-gllvmTMB.R", function() utils::getS3method("predict", "gllvmTMB_multi")),
  list("extract_temporal", "R/temporal.R", function() gllvmTMB::extract_temporal),
  list("temporal_latent", "R/temporal.R", function() gllvmTMB::temporal_latent)
)
deparse_check <- lapply(checks, function(ch) {
  ex <- parse(file.path(src, ch[[2]]), keep.source = FALSE)
  hit <- NULL
  for (e in ex) if (is.call(e) && (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
                      identical(as.character(e[[2]]), ch[[1]])) hit <- e[[3]]
  if (is.null(hit)) stop("P1 source ", ch[[2]], " has no top-level definition of ", ch[[1]])
  fs <- eval(hit, baseenv()); fl <- ch[[3]]()
  same <- identical(deparse(fs, width.cutoff = 500L), deparse(fl, width.cutoff = 500L))
  if (!same) stop("deparse of ", ch[[1]], " differs from its P1 definition in ", ch[[2]])
  list(name = ch[[1]], file = ch[[2]], identical_deparse = same)
})

## ---- cell definitions --------------------------------------------------------------------------
rd <- function(f) read.csv(file.path(data_dir, paste0(f, sfx, ".csv")), stringsAsFactors = FALSE)
data_sha <- NULL
sdhash <- function(f) sha256(file.path(data_dir, paste0(f, sfx, ".csv")))
long_from_wide <- function(w, id, covs, sp_prefix) {
  sp <- setdiff(names(w), c(id, covs)); p <- length(sp)
  d <- data.frame(site = factor(rep(w[[id]], each = p)), trait = factor(rep(sp, times = nrow(w)), levels = sp),
                  value = as.vector(t(as.matrix(w[, sp]))))
  for (cv in covs) d[[cv]] <- rep(w[[cv]], each = p)
  d
}
kind <- "latent"; formula <- NULL; unit <- "site"; fam <- NULL; d <- NULL; fam_label <- NULL
if (cell %in% c("gaussian", "poisson", "nb2", "binomial", "ordinal")) {
  d <- rd(cell); d$site <- factor(d$site); d$trait <- factor(d$trait); data_sha <- sdhash(cell)
  formula <- value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE)
  fam <- switch(cell, gaussian = stats::gaussian(), poisson = stats::poisson(), nb2 = gllvmTMB::nbinom2(),
                binomial = stats::binomial(), ordinal = gllvmTMB::ordinal_logit())
} else if (cell == "temporal") {
  d <- rd("temporal"); d$series <- factor(d$series); d$trait <- factor(d$trait); data_sha <- sdhash("temporal")
  formula <- value ~ 0 + trait + temporal_latent(0 + trait | series, time = occasion, d = 1, structure = "ar1", unique = FALSE)
  unit <- "series"; fam <- stats::gaussian(); kind <- "temporal"
} else if (cell == "isdm") {
  d <- rd("isdm"); d$cell_id <- factor(d$cell_id); d$trait <- factor(d$trait); data_sha <- sdhash("isdm")
  d$isdm_source <- factor(d$isdm_source, levels = c("gbif", "survey"))
  formula <- value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 2, unique = FALSE)
  unit <- "cell_id"; fam <- gllvmTMB::isdm_sources(gbif = poisson(), survey = binomial(link = "cloglog")); kind <- "isdm"
} else if (cell == "crabs") {
  w <- rd("crabs"); data_sha <- sdhash("crabs"); trs <- c("FL", "RW", "CL", "CW", "BD"); p <- 5L
  d <- data.frame(specimen = factor(rep(w$specimen, each = p)), trait = factor(rep(trs, times = nrow(w)), levels = trs),
                  grp = factor(rep(w$grp, each = p)), value = as.vector(t(as.matrix(w[, trs]))))
  formula <- value ~ 0 + trait:grp + latent(0 + trait | specimen, d = 1, unique = FALSE); unit <- "specimen"; fam <- stats::gaussian()
} else if (cell %in% c("spider", "beetle", "fungi")) {
  covs <- switch(cell, spider = c("ConWate", "BareSand", "CovMoss"), beetle = c("pH", "Moist", "Org"), fungi = c("TEMPR", "PRECIP", "logAREA"))
  w <- rd(paste0(cell, "_wide")); data_sha <- sdhash(paste0(cell, "_wide")); d <- long_from_wide(w, "site", covs)
  # shared (not per-species) slopes: the only slope structure Julia's NB2 / binomial routes fit with per-species dispersion
  formula <- as.formula(paste("value ~ 0 + trait +", paste(covs, collapse = " + "), "+ latent(0 + trait | site, d = 2, unique = FALSE)"))
  fam <- if (cell == "fungi") stats::binomial() else gllvmTMB::nbinom2()
} else if (cell == "urban") {
  w <- rd("urban_wide"); data_sha <- sdhash("urban_wide"); sp <- setdiff(names(w), "review"); p <- length(sp)
  d <- data.frame(review = factor(rep(w$review, each = p)), trait = factor(rep(sp, times = nrow(w)), levels = sp),
                  value = as.vector(t(as.matrix(w[, sp]))))
  formula <- value ~ 0 + trait + latent(0 + trait | review, d = 2, unique = FALSE); unit <- "review"; fam <- stats::binomial(link = "probit")
} else stop("unknown cell ", cell)

## ---- fit ---------------------------------------------------------------------------------------
t0 <- Sys.time()
fit <- suppressMessages(gllvmTMB(formula, data = d, unit = unit, trait = "trait", family = fam, silent = TRUE,
                control = gllvmTMBcontrol(n_init = 1L, se = FALSE)))
wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
opt <- fit$opt; pl <- fit$tmb_obj$env$parList(opt$par)
out <- list(engine = "R gllvmTMB", cell = cell, small = SMALL, pin = "P1", reference_commit = man$reference_commit,
            gllvmTMB_version = as.character(packageVersion("gllvmTMB")), gllvmTMB_loaded_from = loaded,
            TMB_version = as.character(packageVersion("TMB")), R_version = R.version.string,
            host = Sys.info()[["nodename"]], finished_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
            data_sha256 = data_sha, formula = paste(deparse(formula, width.cutoff = 500L), collapse = " "),
            p1_source_sha256 = src_hash, deparse_check = deparse_check,
            wall_fit_sec = wall_fit, convergence = opt$convergence, message = opt$message, iterations = opt$iterations,
            logLik = -opt$objective, n_par = length(opt$par),
            max_abs_gradient = max(abs(fit$tmb_obj$gr(opt$par))))
t1 <- Sys.time()
sdr <- tryCatch(TMB::sdreport(fit$tmb_obj, par.fixed = opt$par, getJointPrecision = FALSE), error = function(e) e)
if (inherits(sdr, "error")) { out$sdreport_error <- conditionMessage(sdr); out$pdHess <- FALSE } else {
  out$pdHess <- isTRUE(sdr$pdHess)
  se_all <- tryCatch(sqrt(diag(sdr$cov.fixed)), error = function(e) rep(NA_real_, length(opt$par)))
  nm <- names(sdr$par.fixed)
  out$cond_H <- tryCatch(kappa(solve(sdr$cov.fixed), exact = FALSE), error = function(e) NA_real_)
  out$par_names <- nm
}
out$wall_sdreport_sec <- as.numeric(Sys.time() - t1, units = "secs")
b_idx <- if (!is.null(out$par_names)) which(out$par_names == "b_fix") else integer(0)
out$beta <- as.numeric(pl$b_fix)
out$beta_names <- if (!is.null(fit$X_fix_names)) as.character(fit$X_fix_names) else NULL
out$beta_se <- if (length(b_idx) == length(out$beta)) as.numeric(se_all[b_idx]) else NULL
out$eta_n <- length(fit$report$eta)
if (cell %in% c("crabs", "urban", "spider", "beetle", "fungi")) out$eta <- as.numeric(fit$report$eta)
out$trait_levels <- levels(d$trait)
if (kind == "latent") {
  # part = "shared", link_residual = "none" is Lambda Lambda' exactly; the default link_residual = "auto" adds a link-scale
  # residual variance to the diagonal for non-Gaussian families (measured: max difference 0.41 on a Poisson fit), which is not
  # what the Julia loadings estimate.
  S <- suppressMessages(extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none"))$Sigma
  Lb <- as.matrix(fit$report$Lambda_B); out$LLt_from_Lambda_B_max_abs_diff <- max(abs(Lb %*% t(Lb) - S))
  out$LLt <- unname(as.matrix(S)); out$LLt_trait_names <- colnames(S)
}
if (cell == "nb2") out$dispersion_phi <- as.numeric(fit$report$phi_nbinom2)
if (cell %in% c("spider", "beetle")) out$dispersion_phi <- as.numeric(fit$report$phi_nbinom2)
if (cell == "ordinal") {
  ct <- suppressMessages(extract_cutpoints(fit, quiet = TRUE))
  out$cutpoints <- list(trait = as.character(ct$trait), index = as.integer(ct$cutpoint_index), tau = as.numeric(ct$tau_estimate))
}
if (kind == "temporal") {
  et <- suppressMessages(extract_temporal(fit))
  out$temporal_phi <- as.numeric(et$time$value); out$temporal_phi_parameter <- as.character(et$time$parameter)
  out$temporal_loadings <- as.numeric(et$loadings[, 1]); out$temporal_loadings_traits <- rownames(et$loadings)
  L <- as.matrix(fit$report$Lambda_temporal); out$LLt <- unname(L %*% t(L))
}
if (kind == "isdm") {
  L <- as.matrix(fit$report$Lambda_B); out$LLt <- unname(L %*% t(L))
  nd0 <- d; nd0$log_support <- 0
  pr <- suppressMessages(predict(fit)); out$predict_columns <- names(pr)
  out$predict_link <- as.numeric(pr$est)
  out$predict_response <- as.numeric(suppressMessages(predict(fit, type = "response"))$est)
  out$predict_newdata_offset0_link <- as.numeric(suppressMessages(predict(fit, newdata = nd0, type = "link"))$est)
  out$predict_newdata_offset0_response <- as.numeric(suppressMessages(predict(fit, newdata = nd0, type = "response"))$est)
  out$predict_cell_id <- as.character(pr$cell_id); out$predict_trait <- as.character(pr$trait); out$predict_source <- as.character(pr$isdm_source)
}
write_json(out, file.path(out_dir, paste0(cell, sfx, "_R.json")), auto_unbox = TRUE, digits = NA, pretty = FALSE, null = "null", na = "null")
cat("DONE", cell, "logLik", out$logLik, "conv", out$convergence, "pdHess", out$pdHess, "wall_fit", round(wall_fit, 1), "\n")
