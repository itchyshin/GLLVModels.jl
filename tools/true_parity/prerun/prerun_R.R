#!/usr/bin/env Rscript
# Pre-run R side (campaign plan 4.3). Usage: Rscript prerun_R.R <cell>
.libPaths(c(Sys.getenv("GLLVMTMB_P1_LIB"), "/home/snakagaw/R/lib", .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
cell <- commandArgs(trailingOnly = TRUE)[1]
TAG <- Sys.getenv("PRERUN_TAG", cell); NSUB <- as.integer(Sys.getenv("PRERUN_NSUB", "0")); PSUB <- as.integer(Sys.getenv("PRERUN_PSUB", "0"))  # diagnostics only
dir.create("out", showWarnings = FALSE)
outf <- file.path("out", paste0(TAG, "_R_summary.txt"))
sink_lines <- character(); emit <- function(...) { l <- sprintf(...); sink_lines <<- c(sink_lines, l); writeLines(sink_lines, outf) }
emit("gllvmTMB_version=%s", as.character(packageVersion("gllvmTMB")))
emit("cell=%s tag=%s nsub=%d psub=%d", cell, TAG, NSUB, PSUB)
d <- NULL; fam <- NULL; formula <- NULL; unit <- NULL
rd <- function(f) read.csv(f, stringsAsFactors = FALSE)
if (cell == "ordinal") {
  d <- rd("data/ordinal_p20_n500_K2.csv"); d$site <- factor(d$site); d$trait <- factor(d$trait)
  formula <- value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE); unit <- "site"; fam <- ordinal_logit(); trait <- "trait"
} else if (cell == "temporal") {
  d <- rd("data/temporal_s25_t20_p20_d1.csv"); if (NSUB > 0) d <- d[as.integer(sub("g", "", d$series)) <= NSUB, ]; if (PSUB > 0) d <- d[d$occasion <= PSUB, ]; d$series <- factor(d$series); d$trait <- factor(d$trait)
  formula <- value ~ 0 + trait + temporal_latent(0 + trait | series, time = occasion, d = 1, structure = "ar1", unique = FALSE)
  unit <- "series"; fam <- gaussian(); trait <- "trait"
} else if (cell == "isdm") {
  d <- rd("data/isdm_c500_sp20_s2_K2.csv"); d$cell_id <- factor(d$cell_id); d$trait <- factor(d$trait)
  d$isdm_source <- factor(d$isdm_source, levels = c("gbif", "survey"))
  formula <- value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 2, unique = FALSE)
  unit <- "cell_id"; fam <- isdm_sources(gbif = poisson(), survey = binomial(link = "cloglog")); trait <- "trait"
} else if (cell %in% c("beetle", "fungi")) {
  w <- rd(paste0("data/", cell, "_wide.csv")); covs <- if (cell == "beetle") c("pH", "Moist", "Org") else c("TEMPR", "PRECIP", "logAREA")
  if (NSUB > 0) w <- w[seq_len(NSUB), ]; sp <- setdiff(names(w), c("site", covs)); if (PSUB > 0) sp <- sp[seq_len(PSUB)]; p <- length(sp)
  d <- data.frame(site = factor(rep(w$site, each = p)), trait = factor(rep(sp, times = nrow(w)), levels = sp),
                  value = as.vector(t(as.matrix(w[, sp]))))
  for (cv in covs) d[[cv]] <- rep(w[[cv]], each = p)
  # shared (not per-species) slopes: the only slope structure Julia's NB2/binomial formula routes fit with per-species dispersion
  formula <- as.formula(paste("value ~ 0 + trait +", paste(covs, collapse = " + "), "+ latent(0 + trait | site, d = 2, unique = FALSE)"))
  unit <- "site"; trait <- "trait"
  fam <- if (cell == "beetle") gllvmTMB::nbinom2() else stats::binomial()
}
emit("formula=%s", paste(deparse(formula, width.cutoff = 500L), collapse = " "))
t0 <- Sys.time()
fit <- gllvmTMB(formula, data = d, unit = unit, trait = trait, family = fam, silent = TRUE,
                control = gllvmTMBcontrol(n_init = 1L, se = FALSE))
wall <- as.numeric(Sys.time() - t0, units = "secs")
emit("wall_fit_sec=%.3f", wall)
emit("logLik=%.10f", -fit$opt$objective)
emit("convergence=%s", fit$opt$convergence)
emit("message=%s", fit$opt$message)
emit("max_abs_gradient=%.3e", max(abs(fit$tmb_obj$gr(fit$opt$par))))
emit("n_par=%d", length(fit$opt$par))
t1 <- Sys.time()
sdr <- tryCatch(TMB::sdreport(fit$tmb_obj, par.fixed = fit$opt$par, getJointPrecision = FALSE), error = function(e) e)
if (inherits(sdr, "error")) { emit("sdreport_error=%s", conditionMessage(sdr)) } else {
  emit("pdHess=%s", isTRUE(sdr$pdHess))
  condH <- tryCatch(kappa(solve(sdr$cov.fixed), exact = FALSE), error = function(e) NA_real_)
  emit("cond_H=%.6g", condH)
}
emit("wall_sdreport_sec=%.3f", as.numeric(Sys.time() - t1, units = "secs"))
emit("DONE")
