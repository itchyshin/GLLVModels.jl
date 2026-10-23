# Generating script for the iSDM admission-twin fixtures and recorded R values.
#
# Pin: gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9 (P1), installed into a
# temporary library (see export_p1_fixtures.R for the install recipe).
#
#   GLLVMTMB_P1_LIB=<scratch>/Rlib GLLVMTMB_P1_SRC=<scratch>/gllvmTMB-p1 \
#     Rscript test/fixtures/isdm/export_admission_twins_p1.R <stage>
#
# Stages (from the repository root):
#   fixtures  write the adm_*.csv fixtures ONCE (never regenerated)
#   fits      fit each case in R through the public door; write
#             r_values_admission_p1.toml (logLik, b_fix, eta, a polished optimum,
#             and the admission path each case exercises)
#   xobj      read julia_estimates_admission_p1.toml (written by
#             export_julia_admission_estimates.jl) and append R's own objective
#             at Julia's estimate
#
# Each case is built to reach one positive admission that the four fits of
# export_p1_fixtures.R do not reach (docs/dev-log/core070/true-parity-latest/
# audit-isdm-546-2026-10-01.md). Both engines read the SAME CSV bytes.

p1_lib <- Sys.getenv("GLLVMTMB_P1_LIB"); p1_src <- Sys.getenv("GLLVMTMB_P1_SRC")
if (!nzchar(p1_lib) || !nzchar(p1_src)) stop("Set GLLVMTMB_P1_LIB and GLLVMTMB_P1_SRC.")
.libPaths(c(p1_lib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
P1_SHA <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
src_sha <- system2("git", c("-C", shQuote(p1_src), "rev-parse", "HEAD"), stdout = TRUE)
if (!identical(src_sha, P1_SHA)) stop("GLLVMTMB_P1_SRC is not at P1: ", src_sha)
stage <- commandArgs(trailingOnly = TRUE)[1]
out_dir <- file.path("test", "fixtures", "isdm")
stopifnot(dir.exists(out_dir))

## One simulator for every case: two traits, cells c1..cN, one long block per
## source. `kinds` maps source -> "count" (Poisson, intensity * support) or
## "pa" (Bernoulli-cloglog, 1 - exp(-support * intensity)).
mk_data <- function(seed, n_cell, kinds, support = NULL, latent_sd = 0,
                    access = FALSE) {
  set.seed(seed)
  cells <- paste0("c", seq_len(n_cell)); traits <- c("sp1", "sp2")
  env <- as.numeric(scale(runif(n_cell)))
  alpha <- c(-0.1, 0.2); beta <- c(0.4, -0.3); lam <- c(0.9, 0.6)
  u <- rnorm(n_cell, sd = latent_sd)
  if (is.null(support)) support <- setNames(ifelse(kinds == "count", 1.5, 0.9), names(kinds))
  blocks <- lapply(names(kinds), function(s) {
    d <- expand.grid(cell_id = cells, trait = traits, stringsAsFactors = FALSE)
    ci <- match(d$cell_id, cells); ti <- match(d$trait, traits)
    d$isdm_source <- s; d$env <- env[ci]; d$support <- support[[s]]
    d$access <- rnorm(nrow(d))
    eta <- alpha[ti] + beta[ti] * d$env + lam[ti] * u[ci] +
      if (access && identical(s, "gbif")) 0.25 + 0.5 * d$access else 0
    d$value <- if (kinds[[s]] == "count") rpois(nrow(d), d$support * exp(eta))
               else rbinom(nrow(d), 1, -expm1(-d$support * exp(eta)))
    d
  })
  d <- do.call(rbind, blocks)
  d$log_support <- log(d$support); d$access2 <- 2 * d$access
  d$src_gbif <- as.integer(d$isdm_source == "gbif")
  d
}

gens <- list(
  ## ISDM-ALIASED: gbif's observation formula carries access and an exact
  ## multiple of it; the QR rank rule keeps the first and drops the copy.
  adm_aliased = function() mk_data(101L, 50L, c(gbif = "count", survey = "pa"), access = TRUE),
  ## ISDM-ALIGN: survey is declared FIRST but the data's factor levels (and
  ## Julia's sorted levels) put gbif first.
  adm_align = function() mk_data(102L, 40L, c(gbif = "count", survey = "pa")),
  ## ISDM-NO-OFFSET: no offset() term at all.
  adm_nooffset = function() mk_data(103L, 40L, c(gbif = "count", survey = "pa"),
                                    support = c(gbif = 1, survey = 1)),
  ## ISDM-ZERO-ORDINARY: an all-count declaration (the mixed contract is NOT
  ## admitted, so the cloglog exception is off) with an identically zero offset.
  adm_zeroord = function() mk_data(104L, 40L, c(gbif = "count", inat = "count"),
                                   support = c(gbif = 1, inat = 1)),
  ## ISDM-UNBALANCED: latent field + 12 rows removed; every trait still
  ## carries every source, but the cell x trait x source grid is incomplete.
  adm_unbalanced = function() {
    d <- mk_data(105L, 40L, c(gbif = "count", survey = "pa"), latent_sd = 0.8)
    set.seed(205L)
    drop <- sample(seq_len(nrow(d)), 12L)
    stopifnot(all(table(d$trait[-drop], d$isdm_source[-drop]) > 0))
    d[-drop, ]
  },
  ## ISDM-MASKED-ARM reproducer: NA responses inside both arms.
  adm_maskedna = function() {
    d <- mk_data(106L, 40L, c(gbif = "count", survey = "pa"))
    set.seed(206L)
    na_rows <- c(sample(which(d$isdm_source == "gbif"), 6L), sample(which(d$isdm_source == "survey"), 4L))
    d$value[na_rows] <- NA
    d
  }
)
fixture_files <- setNames(paste0(names(gens), ".csv"), names(gens))

read_fixture <- function(case) {
  d <- utils::read.csv(file.path(out_dir, fixture_files[[case]]), stringsAsFactors = FALSE)
  d$trait <- factor(d$trait); d$cell_id <- factor(d$cell_id)
  lv <- if (case == "adm_zeroord") c("gbif", "inat") else c("gbif", "survey")
  d$isdm_source <- factor(d$isdm_source, levels = lv)
  d
}

cl <- function() binomial(link = "cloglog")
cases <- list(
  adm_aliased = list(
    family = function() isdm_sources(
      gbif = isdm_source(poisson(), observation = ~ access + access2), survey = cl()),
    formula = value ~ 0 + trait + trait:env + offset(log_support)),
  adm_align = list(
    family = function() isdm_sources(survey = cl(), gbif = poisson()),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support)),
  adm_nooffset = list(
    family = function() isdm_sources(gbif = poisson(), survey = cl()),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif),
  adm_zeroord = list(
    family = function() isdm_sources(gbif = poisson(), inat = poisson()),
    formula = value ~ 0 + trait + trait:env + offset(log_support)),
  adm_unbalanced = list(
    family = function() isdm_sources(gbif = poisson(), survey = cl()),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) +
      latent(0 + trait | cell_id, d = 1, unique = FALSE)),
  adm_maskedna = list(
    family = function() isdm_sources(gbif = poisson(), survey = cl()),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support))
)
bound_cases <- c("adm_aliased", "adm_align", "adm_nooffset", "adm_zeroord", "adm_unbalanced")

fit_case <- function(case) {
  d <- read_fixture(case); msgs <- character()
  fit <- withCallingHandlers(
    gllvmTMB(cases[[case]]$formula, data = d, trait = "trait", unit = "cell_id",
             family = cases[[case]]$family(), silent = TRUE),
    message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") })
  list(fit = fit, dat = d, msgs = msgs)
}

fmt_num <- function(x) vapply(x, function(v) {
  if (is.na(v)) "nan" else if (is.infinite(v)) (if (v > 0) "inf" else "-inf") else sprintf("%.17g", v)
}, character(1))
fmt_str <- function(x) paste0('"', gsub('"', '\\\\"', gsub("\n", " ", x)), '"')
kv_num <- function(k, v) paste0(k, " = ", fmt_num(v))
kv_str <- function(k, v) paste0(k, " = ", fmt_str(v))
kv_arr <- function(k, v) paste0(k, " = [", paste(fmt_num(v), collapse = ", "), "]")
kv_sarr <- function(k, v) paste0(k, " = [", paste(fmt_str(v), collapse = ", "), "]")
sha256_file <- function(path) sub(" .*$", "", system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE))

if (identical(stage, "fixtures")) {
  for (case in names(gens)) {
    path <- file.path(out_dir, fixture_files[[case]])
    if (file.exists(path)) stop("Refusing to overwrite ", path, ": fixtures are exported once.")
    utils::write.csv(gens[[case]](), path, row.names = FALSE)
    cat(case, nrow(utils::read.csv(path)), sha256_file(path), "\n")
  }
} else if (identical(stage, "fits")) {
  lines <- c("# Recorded R values for the iSDM admission twins. Written by",
    "# test/fixtures/isdm/export_admission_twins_p1.R (stages fits, xobj). Do not edit.",
    kv_str("gllvmtmb_sha", P1_SHA),
    kv_str("gllvmtmb_version", as.character(packageVersion("gllvmTMB"))),
    kv_str("tmb_version", as.character(packageVersion("TMB"))),
    kv_str("r_version", R.version.string),
    kv_str("written", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")))
  for (case in bound_cases) {
    fx <- fit_case(case); fit <- fx$fit; d <- fx$dat
    path <- file.path(out_dir, fixture_files[[case]])
    b_fix <- fit$opt$par[names(fit$opt$par) == "b_fix"]
    lines <- c(lines, "", paste0("[cases.", case, "]"),
      kv_str("fixture", fixture_files[[case]]), kv_str("fixture_sha256", sha256_file(path)),
      kv_num("fixture_rows", nrow(d)),
      kv_str("formula", paste(deparse(cases[[case]]$formula, width.cutoff = 500L), collapse = " ")),
      kv_sarr("declared_sources", names(cases[[case]]$family())),
      kv_sarr("factor_levels", levels(d$isdm_source)),
      kv_sarr("messages", fx$msgs),
      kv_num("loglik", -fit$opt$objective), kv_num("convergence", fit$opt$convergence),
      kv_str("message", fit$opt$message),
      kv_num("max_abs_gradient", max(abs(fit$tmb_obj$gr(fit$opt$par)))),
      kv_sarr("b_fix_names", fit$X_fix_names), kv_arr("b_fix", b_fix))
    if (!is.null(fit$report$Lambda_B) && length(fit$report$Lambda_B)) {
      L <- as.matrix(fit$report$Lambda_B)
      lines <- c(lines, kv_num("K", ncol(L)), kv_arr("Lambda_B_colmajor", as.numeric(L)))
    } else lines <- c(lines, kv_num("K", 0))
    lines <- c(lines, kv_arr("eta", as.numeric(fit$report$eta)))
    pol <- stats::nlminb(fit$opt$par, fit$tmb_obj$fn, fit$tmb_obj$gr,
                         control = list(rel.tol = 1e-14, x.tol = 1e-12, iter.max = 2000L, eval.max = 4000L))
    lines <- c(lines, kv_num("polished_loglik", -pol$objective),
      kv_num("polished_convergence", pol$convergence),
      kv_num("polished_max_abs_gradient", max(abs(fit$tmb_obj$gr(pol$par)))),
      kv_arr("polished_b_fix", pol$par[names(pol$par) == "b_fix"]))
    cat(case, "loglik", -fit$opt$objective, "conv", fit$opt$convergence, "grad",
        max(abs(fit$tmb_obj$gr(fit$opt$par))), "\n  cols:", fit$X_fix_names, "\n  msgs:", fx$msgs, "\n")
  }
  ## Reproducer: R's door with NA responses inside both arms (ISDM-MASKED-ARM).
  fx <- fit_case("adm_maskedna"); fit <- fx$fit
  lines <- c(lines, "", "[reproducers.adm_maskedna]",
    kv_str("fixture", fixture_files[["adm_maskedna"]]),
    kv_str("fixture_sha256", sha256_file(file.path(out_dir, fixture_files[["adm_maskedna"]]))),
    kv_num("rows", nrow(fx$dat)), kv_num("na_rows", sum(is.na(fx$dat$value))),
    kv_num("loglik", -fit$opt$objective), kv_num("convergence", fit$opt$convergence),
    kv_sarr("b_fix_names", fit$X_fix_names),
    kv_arr("b_fix", fit$opt$par[names(fit$opt$par) == "b_fix"]))
  ## Reproducer: R's door with a logit-binomial source (ISDM-WRAPPER-LAW).
  wr <- tryCatch({ isdm_sources(gbif = poisson(), survey = isdm_source(binomial("logit"), observation = ~ access)); "ACCEPTED" },
                 error = function(e) paste("REFUSED:", conditionMessage(e)))
  lines <- c(lines, "", "[reproducers.wrapper_logit]", kv_str("isdm_sources_result", wr),
             kv_str("isdm_source_alone_class", class(isdm_source(binomial("logit"), observation = ~ x))[1]))
  cat("wrapper:", wr, "\n")
  writeLines(lines, file.path(out_dir, "r_values_admission_p1.toml"))
} else if (identical(stage, "xobj")) {
  est <- readLines(file.path(out_dir, "julia_estimates_admission_p1.toml"))
  get_arr <- function(section, key) {
    i0 <- which(est == paste0("[", section, "]")); stopifnot(length(i0) == 1L)
    nxt <- which(grepl("^\\[", est) & seq_along(est) > i0)
    i1 <- if (length(nxt)) min(nxt) - 1L else length(est)
    ln <- grep(paste0("^", key, " = "), est[i0:i1], value = TRUE); stopifnot(length(ln) == 1L)
    trimws(strsplit(sub("^[^=]*= \\[(.*)\\]$", "\\1", ln), ",")[[1]])
  }
  lines <- readLines(file.path(out_dir, "r_values_admission_p1.toml"))
  cut <- which(lines == "[xobj]"); if (length(cut)) lines <- lines[seq_len(cut - 2L)]
  add <- c("", "[xobj]", "# R's objective (TMB Laplace, -loglik sign flipped) at Julia's estimate")
  for (case in bound_cases) {
    sec <- paste0("julia.", case)
    jn <- gsub('"', "", get_arr(sec, "b_fix_names")); jb <- as.numeric(get_arr(sec, "b_fix"))
    fit <- fit_case(case)$fit; rn <- fit$X_fix_names
    if (!setequal(jn, rn)) stop("b_fix names do not pair for ", case)
    par <- fit$opt$par; par[names(par) == "b_fix"] <- jb[match(rn, jn)]
    if (any(names(par) == "theta_rr_B")) par[names(par) == "theta_rr_B"] <- as.numeric(get_arr(sec, "theta_rr_B"))
    stopifnot(all(names(par) %in% c("b_fix", "theta_rr_B")))
    add <- c(add, kv_num(case, -fit$tmb_obj$fn(par)))
    cat(case, "R loglik at Julia estimate", -fit$tmb_obj$fn(par), "\n")
  }
  add <- c(add, kv_str("julia_estimates_sha256", sha256_file(file.path(out_dir, "julia_estimates_admission_p1.toml"))))
  writeLines(c(lines, add), file.path(out_dir, "r_values_admission_p1.toml"))
} else stop("Unknown stage: ", stage)
