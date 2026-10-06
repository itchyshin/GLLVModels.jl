#!/usr/bin/env Rscript
# P1 public-door half of the first-seven behavioural checkpoint.
# Usage: GLLVM_P1_RLIB=... GLLVMTMB_DIR=... Rscript tools/first_seven_behaviour_R.R OUT.tsv
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: first_seven_behaviour_R.R OUT.tsv", call. = FALSE)
out <- args[[1L]]
lib <- Sys.getenv("GLLVM_P1_RLIB")
if (!nzchar(lib)) stop("GLLVM_P1_RLIB must name the frozen P1 R library", call. = FALSE)
if (Sys.getenv("GLLVM_PARITY_ORACLE_BUILD") != "totoro") stop("GLLVM_PARITY_ORACLE_BUILD must be totoro", call. = FALSE)
if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1" || Sys.getenv("OMP_NUM_THREADS") != "1")
  stop("set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before running fits", call. = FALSE)
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(utils::packageVersion("gllvmTMB")) == "0.7.1")
if (normalizePath(find.package("gllvmTMB")) != normalizePath(file.path(lib, "gllvmTMB")))
  stop("gllvmTMB was not loaded from GLLVM_P1_RLIB", call. = FALSE)

source(file.path(normalizePath("."), "tools/core070_source_pin.R"))
sp <- core070_source_pin(normalizePath("."), lib, "P1", "9539352f66f2db2cc26b1c393e67212a359b60c9")

capture <- function(id, expr, call=paste(deparse(substitute(expr), width.cutoff=500L), collapse=" ")) {
  msg <- ""
  errclass <- ""
  value <- tryCatch(withCallingHandlers(force(expr), warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) { msg <<- conditionMessage(e); errclass <<- paste(class(e), collapse="/"); NULL })
  if (is.null(value)) return(data.frame(case_id=id, outcome="ERROR", class=errclass, actual="", message=msg, call=call))
  actual <- if (id == "CORE070-FIRST7-CHECK-AUTO-RESIDUAL") {
    if (identical(value$status, "ok")) "coherent" else "not-coherent"
  } else "returned"
  data.frame(case_id=id, outcome="RETURN", class=paste(class(value), collapse="/"),
             actual=actual, message=if (id == "CORE070-FIRST7-CHECK-AUTO-RESIDUAL") value$status else "", call=call)
}

# Full trait x source x unit panel, shared conceptually with the Julia runner.
d <- expand.grid(cell_id=factor(c("u1", "u2")), isdm_source=c("count", "detect"),
                trait=factor(c("a", "b")), KEEP.OUT.ATTRS=FALSE)
d$value <- ifelse(d$isdm_source == "count", 1 + seq_len(nrow(d)) %% 4,
                  seq_len(nrow(d)) %% 2)
d$log_support <- rep(seq(0.05, 0.4, length.out=nrow(d)), 1)
stopifnot(nrow(d) == 8L, all(table(interaction(d$trait, d$isdm_source), d$cell_id) == 1L))
f <- value ~ 0 + trait + offset(log_support) + latent(0 + trait | cell_id, d=1, unique=FALSE)
fam <- function() isdm_sources(count=poisson(), detect=binomial(link="cloglog"))

records <- list()
records[[1]] <- capture("CORE070-FIRST7-CHECK-AUTO-RESIDUAL", {
  gd <- expand.grid(cell_id=factor(paste0("u", 1:12)), trait=factor(c("a", "b")),
                    KEEP.OUT.ATTRS=FALSE)
  gd$value <- sin(seq_len(nrow(gd)) / 3)
  gf <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | cell_id, d=1, unique=FALSE),
                 data=gd, family=gaussian(), num.lv=1L, verbose=FALSE)
  check_auto_residual(gf)
})
records[[2]] <- capture("CORE070-FIRST7-ISDM-COUNT", {
  # Existing public declaration with no detection-law arm; fit must reject it.
  gllvmTMB(f, data=d, family=isdm_sources(count=poisson(), detect=poisson()), num.lv=1L)
})
records[[3]] <- capture("CORE070-FIRST7-ISDM-EXTRA-SOURCE", {
  data_with_unknown_source <- d; data_with_unknown_source$isdm_source[1] <- "unknown"
  stopifnot(nrow(data_with_unknown_source) == nrow(d), sum(data_with_unknown_source$isdm_source == "unknown") == 1L)
  gllvmTMB(f, data=data_with_unknown_source, family=fam(), num.lv=1L)
})
records[[4]] <- capture("CORE070-FIRST7-ISDM-MISSING-IN-TRAIT", {
  data_missing_source_in_trait <- subset(d, !(trait == "a" & isdm_source == "detect"))
  stopifnot(nrow(d) - nrow(data_missing_source_in_trait) == 2L, !any(data_missing_source_in_trait$trait == "a" & data_missing_source_in_trait$isdm_source == "detect"))
  gllvmTMB(f, data=data_missing_source_in_trait, family=fam(), num.lv=1L)
})
records[[5]] <- capture("CORE070-FIRST7-ISDM-MISSING-SOURCE", {
  data_missing_declared_source <- subset(d, isdm_source == "count")
  stopifnot(nrow(d) - nrow(data_missing_declared_source) == 4L, identical(unique(as.character(data_missing_declared_source$isdm_source)), "count"))
  gllvmTMB(f, data=data_missing_declared_source, family=fam(), num.lv=1L)
})
logit_source <- isdm_source(binomial("logit"), observation=~x)
stopifnot(inherits(logit_source, "gllvmTMB_isdm_source"))
wrapper_refusal <- tryCatch({
  isdm_sources(gbif=poisson(), survey=isdm_source(binomial("logit"), observation=~access))
  "ACCEPTED"
}, error=function(e) paste("REFUSED: isdm_sources:", conditionMessage(e)))
stopifnot(startsWith(wrapper_refusal, "REFUSED:"))
valid_cloglog <- isdm_sources(gbif=poisson(), survey=binomial(link="cloglog"))
stopifnot(!is.null(valid_cloglog))
records[[6]] <- data.frame(case_id="CORE070-FIRST7-ISDM-WRAPPER-LAW", outcome="ERROR",
  class="collector_refusal", actual="refused", message=wrapper_refusal,
  call="isdm_sources(gbif=poisson(), survey=isdm_source(binomial(\"logit\"), observation=~access))")
# Discriminating diagnostic control: ordinal probit must warn in R and be
# flagged by Julia. Store this in the primary row's message, not as a new
# scoreboard case.
od <- expand.grid(cell_id=factor(paste0("u", 1:12)), trait=factor(c("a", "b")), KEEP.OUT.ATTRS=FALSE)
od$value <- 1L + (seq_len(nrow(od)) %% 3L)
of <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | cell_id, d=1, unique=FALSE),
               data=od, family=ordinal_probit(), num.lv=1L, verbose=FALSE)
oc <- suppressWarnings(check_auto_residual(of))
stopifnot(identical(oc$status, "warn"))
records[[1]]$message <- paste0("ordinal-probit control status=", oc$status)
res <- do.call(rbind, records)
res$engine <- "R"
res$pin <- "P1"
res$package_version <- as.character(utils::packageVersion("gllvmTMB"))
res$r_version <- as.character(getRversion())
res$host <- as.character(Sys.info()[["nodename"]])
res$openblas_threads <- Sys.getenv("OPENBLAS_NUM_THREADS")
res$omp_threads <- Sys.getenv("OMP_NUM_THREADS")
res$oracle_build <- Sys.getenv("GLLVM_PARITY_ORACLE_BUILD")
res$source_marker_sha256 <- sp$marker_sha256
res$source_tree_sha256 <- sp$source_tree_sha256
res$installed_tree_sha256 <- sp$installed_tree_sha256
res$namespace_sha256 <- sp$namespace_sha256
res$package_version <- sp$version
fixture_path <- tempfile()
fixture_rows <- paste(d$trait, d$isdm_source, as.character(d$cell_id), sprintf("%.8f", d$value),
                      sprintf("%.8f", d$log_support), sep="\t")
writeLines(fixture_rows, fixture_path, useBytes=TRUE)
res$fixture_sha256 <- core070_sha256_file(fixture_path); unlink(fixture_path)
res$runner_sha256 <- core070_sha256_file("tools/first_seven_behaviour_R.R")
utils::write.table(res, out, sep="\t", row.names=FALSE, quote=TRUE, na="")
cat("CORE070_FIRST7_R_RAW_WRITTEN\n")
