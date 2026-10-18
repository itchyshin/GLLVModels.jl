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
  } else if (id == "CORE070-FIRST7-ISDM-COUNT") {
    laws <- vapply(value$family_input, function(f) paste(f$family, f$link, sep=":"), character(1))
    if (length(laws) == 2L && all(laws == "poisson:log")) "all-count-nonmixed" else "wrong-count-route"
  } else "returned"
  data.frame(case_id=id, outcome="RETURN", class=paste(class(value), collapse="/"),
             actual=actual, message=if (id == "CORE070-FIRST7-CHECK-AUTO-RESIDUAL") value$status else "", call=call)
}

diagnostic_fixture <- function() {
  n <- 12L
  trait <- factor(rep(c("a", "b"), each=n), levels=c("a", "b"))
  cell_id <- factor(rep(paste0("u", seq_len(n)), times=2L),
                    levels=paste0("u", seq_len(n)))
  value <- sin(seq_len(2L*n) / 3)
  Y <- rbind(value[trait == "a"], value[trait == "b"])
  X <- matrix(1, nrow=n, ncol=1L)
  ordinal_values <- 1L + (seq_len(n) %% 3L)
  ordinal <- rbind(ordinal_values, ordinal_values)
  data <- data.frame(cell_id=cell_id, trait=trait, value=value)
  list(data=data, Y=Y, X=X, ordinal=ordinal)
}

# Full trait x source x unit panel, shared with the Julia runner.
panel <- function() {
  d <- expand.grid(cell_id=factor(c("u1", "u2")), isdm_source=c("count", "detect"),
                   trait=factor(c("a", "b")), KEEP.OUT.ATTRS=FALSE, stringsAsFactors=FALSE)
  d$value <- ifelse(d$isdm_source == "count", 1 + seq_len(nrow(d)) %% 4,
                    seq_len(nrow(d)) %% 2)
  d$log_support <- seq(0.05, 0.4, length.out=nrow(d))
  stopifnot(nrow(d) == 8L, all(table(interaction(d$trait, d$isdm_source), d$cell_id) == 1L))
  d
}

panel_input_rows <- function(data) {
  paste(data$trait, data$isdm_source, as.character(data$cell_id),
        sprintf("%.8f", data$value), sprintf("%.8f", data$log_support), sep="\t")
}

diagnostic_input_rows <- function(fixture) {
  c(paste0("gaussian_Y\t", c("a", "b"), "\t",
           apply(fixture$Y, 1L, function(x) paste(sprintf("%.8f", x), collapse="\t"))),
    paste0("gaussian_X\t", paste(sprintf("%.8f", fixture$X[,1L]), collapse="\t")),
    paste0("ordinal_Y\t", c("a", "b"), "\t",
           apply(fixture$ordinal, 1L, function(x) paste(as.integer(x), collapse="\t"))))
}

case_input_sha256 <- function(case_id, model, rows) {
  payload <- paste(c("CORE070-FIRST7-INPUT-v1", paste("case", case_id, sep="\t"),
                     paste("model", model, sep="\t"), rows), collapse="\n")
  path <- tempfile()
  on.exit(unlink(path), add=TRUE)
  writeBin(charToRaw(enc2utf8(payload)), path)
  core070_sha256_file(path)
}

case_fixture_hashes <- function(d, diagnostic, unknown, missing_trait, missing_source) {
  ids <- c("CORE070-FIRST7-CHECK-AUTO-RESIDUAL", "CORE070-FIRST7-ISDM-COUNT",
           "CORE070-FIRST7-ISDM-EXTRA-SOURCE", "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT",
           "CORE070-FIRST7-ISDM-MISSING-SOURCE", "CORE070-FIRST7-ISDM-WRAPPER-LAW")
  specs <- c("gaussian:0+trait+latent(0+trait|unit,d=1,unique=false);trait_intercepts",
             "isdm:count=poisson-log,detect=poisson-log",
             "isdm:count=poisson-log,detect=binomial-cloglog",
             "isdm:count=poisson-log,detect=binomial-cloglog",
             "isdm:count=poisson-log,detect=binomial-cloglog",
             "isdm_sources:gbif=poisson-log,survey=wrapped-binomial-logit-observation-access")
  rows <- list(diagnostic_input_rows(diagnostic), panel_input_rows(d),
               panel_input_rows(unknown), panel_input_rows(missing_trait),
               panel_input_rows(missing_source),
               c("standalone=binomial-logit-observation-x",
                 "wrapped=survey:binomial-logit-observation-access",
                 "control=survey:binomial-cloglog"))
  stats::setNames(Map(case_input_sha256, ids, specs, rows) |> unlist(), ids)
}

d <- panel()
diagnostic <- diagnostic_fixture()
unknown <- d; unknown$isdm_source[1L] <- "unknown"
missing_trait <- subset(d, !(trait == "a" & isdm_source == "detect"))
missing_source <- subset(d, isdm_source == "count")
input_hashes <- case_fixture_hashes(d, diagnostic, unknown, missing_trait, missing_source)
f <- value ~ 0 + trait + offset(log_support) + latent(0 + trait | cell_id, d=1, unique=FALSE)
fam <- function() isdm_sources(count=poisson(), detect=binomial(link="cloglog"))

# Public positive control: both source laws must be admitted together.
base_fit <- gllvmTMB(f, data=d, unit="cell_id", trait="trait", family=fam(), silent=TRUE)
base_laws <- vapply(base_fit$family_input, function(law) paste(law$family, law$link, sep=":"), character(1))
stopifnot(setequal(base_laws, c("poisson:log", "binomial:cloglog")))
records <- list()
records[[1]] <- capture("CORE070-FIRST7-CHECK-AUTO-RESIDUAL", {
  gf <- gllvmTMB(value ~ 0 + trait + latent(0 + trait | cell_id, d=1, unique=FALSE),
                 data=diagnostic$data, unit="cell_id", trait="trait", family=gaussian(), silent=TRUE)
  check_auto_residual(gf)
})
records[[2]] <- capture("CORE070-FIRST7-ISDM-COUNT", {
  # Existing public declaration with no detection-law arm; fit must reject it.
  gllvmTMB(f, data=d, unit="cell_id", trait="trait", family=isdm_sources(count=poisson(), detect=poisson()))
})
records[[3]] <- capture("CORE070-FIRST7-ISDM-EXTRA-SOURCE", {
  stopifnot(nrow(unknown) == nrow(d), sum(unknown$isdm_source == "unknown") == 1L)
  gllvmTMB(f, data=unknown, unit="cell_id", trait="trait", family=fam())
})
records[[4]] <- capture("CORE070-FIRST7-ISDM-MISSING-IN-TRAIT", {
  stopifnot(nrow(d) - nrow(missing_trait) == 2L,
            !any(missing_trait$trait == "a" & missing_trait$isdm_source == "detect"))
  gllvmTMB(f, data=missing_trait, unit="cell_id", trait="trait", family=fam())
})
records[[5]] <- capture("CORE070-FIRST7-ISDM-MISSING-SOURCE", {
  stopifnot(nrow(d) - nrow(missing_source) == 4L,
            identical(unique(as.character(missing_source$isdm_source)), "count"))
  gllvmTMB(f, data=missing_source, unit="cell_id", trait="trait", family=fam())
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
od <- diagnostic$data
od$value <- as.vector(t(diagnostic$ordinal))
of <- gllvmTMB(value ~ 0 + latent(0 + trait | cell_id, d=1, unique=FALSE),
               data=od, unit="cell_id", trait="trait", family=ordinal_probit(), silent=TRUE)
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
res$public_positive_control <- "mixed-source-accepted"
res$source_marker_sha256 <- sp$marker_sha256
res$source_tree_sha256 <- sp$source_tree_sha256
res$installed_tree_sha256 <- sp$installed_tree_sha256
res$namespace_sha256 <- sp$namespace_sha256
res$package_version <- sp$version
res$fixture_sha256 <- unname(input_hashes[res$case_id])
res$runner_sha256 <- core070_sha256_file("tools/first_seven_behaviour_R.R")
utils::write.table(res, out, sep="\t", row.names=FALSE, quote=TRUE, qmethod="double", na="")
cat("CORE070_FIRST7_R_RAW_WRITTEN\n")
