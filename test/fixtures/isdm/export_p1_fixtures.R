# Generating script for the iSDM P1 twin fixtures and recorded R values.
#
# Pin: gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9 (P1), installed into a
# temporary library from a detached worktree; R reads nothing else.
#
#   git -C <gllvmTMB> worktree add --detach <scratch>/gllvmTMB-p1 9539352f6
#   R CMD INSTALL --library=<scratch>/Rlib <scratch>/gllvmTMB-p1
#   GLLVMTMB_P1_LIB=<scratch>/Rlib GLLVMTMB_P1_SRC=<scratch>/gllvmTMB-p1 \
#     Rscript test/fixtures/isdm/export_p1_fixtures.R <stage>
#
# Stages (run in this order, from the repository root):
#   fixtures  write the four CSV fixtures ONCE, from the R test files' own
#             generators (set.seed as in the R tests); never regenerated
#   fits      read the CSVs back, fit each case in R, write r_values_p1.toml
#   grid      value / score / observed weight of gll_dbinom_cloglog on a
#             24-point eta grid via a scalar TMB MakeADFun; writes
#             cloglog_grid_p1.csv
#   xobj      read julia_estimates_p1.toml (written by the Julia side) and
#             evaluate R's own objective at Julia's estimate; appends to
#             r_values_p1.toml
#   admission replay the 20 CORE070-ISDM-*-PAIRED-CONTROL predicates of
#             docs/dev-log/core070/isdm-batch-contract.json at P1; writes
#             admission_p1.toml
#
# Both engines read the SAME CSV bytes, so the paired numbers cannot drift
# with R's RNG. The Julia side checks each CSV's sha256 before use.

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
## Fixture generators, copied verbatim from the R test files at P1.
## ---------------------------------------------------------------------------

## tests/testthat/test-isdm-predict.R:6-34
.isdm_predict_fixture <- function() {
  set.seed(7)
  n_cell <- 30L
  cells <- paste0("c", seq_len(n_cell))
  species <- c("sp1", "sp2")
  x <- as.numeric(scale(runif(n_cell)))
  alpha <- c(-0.1, 0.2); beta <- c(0.4, -0.3)
  u_cell <- rnorm(n_cell, sd = 0.8)
  lam_tr <- c(0.9, 0.6)
  mk <- function(src, kind, support) {
    d <- expand.grid(cell_id = cells, trait = species, stringsAsFactors = FALSE)
    ci <- match(d$cell_id, cells); si <- match(d$trait, species)
    eta <- alpha[si] + x[ci] * beta[si] + u_cell[ci] * lam_tr[si]
    d$isdm_source <- src
    d$support <- support
    d$value <- if (kind == "count") rpois(nrow(d), support * exp(eta))
               else rbinom(nrow(d), 1, -expm1(-support * exp(eta)))
    d
  }
  dat <- rbind(mk("gbif", "count", 1.5), mk("survey", "pa", 0.9))
  dat$trait <- factor(dat$trait)
  dat$cell_id <- factor(dat$cell_id)
  dat$isdm_source <- factor(dat$isdm_source, levels = c("gbif", "survey"))
  dat$log_support <- log(dat$support)
  dat$env <- x[match(as.character(dat$cell_id), cells)]
  dat$src_gbif <- as.integer(dat$isdm_source == "gbif")
  dat
}

## tests/testthat/test-isdm-multisource.R:4-31
.ms_fixture <- function(sources = c(gbif = "count", literature = "count",
                                    survey = "pa"),
                        n_cell = 30L, seed = 7L) {
  set.seed(seed)
  cells <- paste0("c", seq_len(n_cell))
  species <- c("sp1", "sp2")
  x <- as.numeric(scale(runif(n_cell)))
  alpha <- c(-0.1, 0.2); beta <- c(0.4, -0.3)
  out <- do.call(rbind, lapply(names(sources), function(src) {
    d <- expand.grid(cell_id = cells, trait = species,
                     stringsAsFactors = FALSE)
    ci <- match(d$cell_id, cells); si <- match(d$trait, species)
    eta <- alpha[si] + x[ci] * beta[si]
    d$isdm_source <- src
    d$support <- if (sources[[src]] == "count") 1.5 else 0.9
    d$value <- if (sources[[src]] == "count") {
      rpois(nrow(d), d$support * exp(eta))
    } else {
      rbinom(nrow(d), 1, -expm1(-d$support * exp(eta)))
    }
    d
  }))
  out$trait <- factor(out$trait)
  out$cell_id <- factor(out$cell_id)
  out$log_support <- log(out$support)
  out$env <- x[match(as.character(out$cell_id), cells)]
  out
}

## tests/testthat/test-isdm-source-formula.R:23-58
.isdm_source_recovery_fixture <- function(seed = 41L, n_cell = 120L) {
  set.seed(seed)
  cells <- paste0("c", seq_len(n_cell))
  traits <- c("sp1", "sp2")
  env <- as.numeric(scale(stats::runif(n_cell)))
  make_source <- function(source) {
    d <- expand.grid(cell_id = cells, trait = traits, stringsAsFactors = FALSE)
    cell_id <- match(d$cell_id, cells)
    trait_id <- match(d$trait, traits)
    d$isdm_source <- source
    d$env <- env[cell_id]
    d$support <- 1.5
    d$access <- stats::rnorm(nrow(d))
    d$observer <- NA_character_
    d$method <- NA_character_
    eta <- c(-0.2, 0.15)[trait_id] + c(0.3, -0.2)[trait_id] * d$env
    if (identical(source, "gbif")) eta <- eta + 0.25 + 0.5 * d$access
    if (identical(source, "survey")) {
      d$observer <- sample(c("o1", "o2"), nrow(d), replace = TRUE)
      d$method <- sample(c("walk", "point"), nrow(d), replace = TRUE)
      eta <- eta + ifelse(d$observer == "o2", 0.2, 0) +
        ifelse(d$method == "point", -0.15, 0)
    }
    d$value <- stats::rpois(nrow(d), d$support * exp(eta))
    d
  }
  dat <- rbind(make_source("gbif"), make_source("inat"), make_source("survey"))
  dat$trait <- factor(dat$trait)
  dat$cell_id <- factor(dat$cell_id)
  dat$isdm_source <- factor(dat$isdm_source,
                            levels = c("gbif", "inat", "survey"))
  dat$observer <- factor(dat$observer)
  dat$method <- factor(dat$method)
  dat$log_support <- log(dat$support)
  dat
}

## tests/testthat/test-isdm-source-formula.R:171-181 (the mixed-law relabelling)
.isdm_source_mixed_fixture <- function() {
  dat <- .isdm_source_recovery_fixture(n_cell = 60L)
  survey_rows <- dat$isdm_source == "survey"
  dat$observer[!survey_rows] <- "o1"
  dat$method[!survey_rows] <- "walk"
  dat$observer <- factor(dat$observer)
  dat$method <- factor(dat$method)
  set.seed(92)
  dat$value[survey_rows] <- stats::rbinom(sum(survey_rows), size = 1L, prob = 0.35)
  dat
}

fixture_files <- c(
  predict = "isdm_predict.csv",
  ms3 = "isdm_ms3.csv",
  srcform_pois = "isdm_srcform_pois.csv",
  srcform_mixed = "isdm_srcform_mixed.csv"
)

## Read a fixture back with the factor coding the R tests use.
read_fixture <- function(case) {
  d <- utils::read.csv(file.path(out_dir, fixture_files[[case]]),
                       stringsAsFactors = FALSE)
  for (v in intersect(c("trait", "cell_id", "observer", "method", "src"), names(d))) {
    d[[v]] <- factor(d[[v]])
  }
  lv <- switch(case,
    predict = c("gbif", "survey"),
    ms3 = c("gbif", "literature", "survey"),
    c("gbif", "inat", "survey"))
  d$isdm_source <- factor(d$isdm_source, levels = lv)
  if (case == "ms3") d$src <- factor(d$src, levels = lv)
  d
}

## Case definitions: the R test calls. Latent fits use `unique = FALSE`, the
## model the port spec describes (section 1.2): R's `latent()` default is
## `unique = TRUE` (R/brms-sugar.R:607), which adds a per-trait unit-level
## diagonal (`theta_diag_B`) that the spec does not port. The default-formula
## fit is recorded alongside as a finding (`default_unique_*` keys).
cases <- list(
  predict = list(
    family = function() isdm_sources(gbif = poisson(), survey = binomial(link = "cloglog")),
    formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) +
      latent(0 + trait | cell_id, d = 1, unique = FALSE),
    default_formula = value ~ 0 + trait + trait:env + trait:src_gbif + offset(log_support) +
      latent(0 + trait | cell_id, d = 1),
    predict = TRUE
  ),
  ms3 = list(
    family = function() isdm_sources(gbif = poisson(), literature = poisson(),
                                     survey = binomial(link = "cloglog")),
    formula = value ~ 0 + trait + trait:env + trait:src + offset(log_support) +
      latent(0 + trait | cell_id, d = 1, unique = FALSE),
    default_formula = value ~ 0 + trait + trait:env + trait:src + offset(log_support) +
      latent(0 + trait | cell_id, d = 1),
    predict = TRUE
  ),
  srcform_pois = list(
    family = function() isdm_sources(
      gbif = isdm_source(poisson(), observation = ~ access),
      inat = poisson(),
      survey = isdm_source(poisson(), observation = ~ observer + method)),
    formula = value ~ 0 + trait + trait:env + offset(log_support),
    default_formula = NULL,
    predict = FALSE
  ),
  srcform_mixed = list(
    family = function() isdm_sources(
      gbif = isdm_source(poisson(), observation = ~ access),
      inat = poisson(),
      survey = isdm_source(binomial(link = "cloglog"),
                           observation = ~ observer + method)),
    formula = value ~ 0 + trait + trait:env + offset(log_support),
    default_formula = NULL,
    predict = FALSE
  )
)

fit_case <- function(case, formula = cases[[case]]$formula) {
  d <- read_fixture(case)
  fit <- suppressMessages(gllvmTMB(
    formula, data = d, trait = "trait", unit = "cell_id",
    family = cases[[case]]$family(), silent = TRUE
  ))
  list(fit = fit, dat = d)
}

## ---------------------------------------------------------------------------
## Small TOML writer (numbers with 17 significant digits).
## ---------------------------------------------------------------------------
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

sha256_file <- function(path) {
  out <- system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE)
  sub(" .*$", "", out)
}

if (identical(stage, "fixtures")) {
  gens <- list(
    predict = .isdm_predict_fixture,
    ms3 = function() {
      d <- .ms_fixture()
      ## test-isdm-multisource.R:61-62, the per-source reporting-rate factor
      d$src <- factor(d$isdm_source, levels = c("gbif", "literature", "survey"))
      d
    },
    srcform_pois = function() .isdm_source_recovery_fixture(),
    srcform_mixed = .isdm_source_mixed_fixture
  )
  for (case in names(gens)) {
    path <- file.path(out_dir, fixture_files[[case]])
    if (file.exists(path)) stop("Refusing to overwrite ", path, ": fixtures are exported once.")
    utils::write.csv(gens[[case]](), path, row.names = FALSE)
    cat(case, nrow(utils::read.csv(path)), sha256_file(path), "\n")
  }
} else if (identical(stage, "fits")) {
  lines <- c(
    "# Recorded R values for the iSDM P1 twins. Written by",
    "# test/fixtures/isdm/export_p1_fixtures.R (stages fits, xobj). Do not edit.",
    kv_str("gllvmtmb_sha", P1_SHA),
    kv_str("gllvmtmb_version", as.character(packageVersion("gllvmTMB"))),
    kv_str("tmb_version", as.character(packageVersion("TMB"))),
    kv_str("r_version", R.version.string),
    kv_str("host", Sys.info()[["nodename"]]),
    kv_str("written", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
    "",
    "[r_file_sha256]"
  )
  for (f in c("R/isdm-sources.R", "R/fit-multi.R", "R/offset.R",
              "R/methods-gllvmTMB.R", "src/gllvmTMB.cpp", "src/gllvmTMB_cloglog.h")) {
    lines <- c(lines, kv_str(fmt_str(f), sha256_file(file.path(p1_src, f))))
  }
  for (case in names(cases)) {
    t0 <- proc.time()[["elapsed"]]
    fx <- fit_case(case)
    wall <- proc.time()[["elapsed"]] - t0
    fit <- fx$fit; d <- fx$dat
    path <- file.path(out_dir, fixture_files[[case]])
    b_fix <- fit$opt$par[names(fit$opt$par) == "b_fix"]
    lines <- c(lines, "", paste0("[cases.", case, "]"),
      kv_str("fixture", fixture_files[[case]]),
      kv_str("fixture_sha256", sha256_file(path)),
      kv_num("fixture_rows", nrow(d)),
      kv_str("formula", paste(deparse(cases[[case]]$formula, width.cutoff = 500L), collapse = " ")),
      kv_num("loglik", -fit$opt$objective),
      kv_num("convergence", fit$opt$convergence),
      kv_str("message", fit$opt$message),
      kv_num("iterations", fit$opt$iterations),
      kv_str("pdHess", as.character(isTRUE(fit$sd_report$pdHess))),
      kv_num("max_abs_gradient", max(abs(fit$tmb_obj$gr(fit$opt$par)))),
      kv_num("wall_seconds", wall),
      kv_sarr("par_names", names(fit$tmb_obj$par)),
      kv_sarr("b_fix_names", fit$X_fix_names),
      kv_arr("b_fix", b_fix))
    if (!is.null(fit$report$Lambda_B) && length(fit$report$Lambda_B)) {
      L <- as.matrix(fit$report$Lambda_B)
      lines <- c(lines, kv_num("K", ncol(L)), kv_arr("Lambda_B_colmajor", as.numeric(L)),
                 kv_arr("LLt_colmajor", as.numeric(L %*% t(L))))
    } else {
      lines <- c(lines, kv_num("K", 0))
    }
    lines <- c(lines, kv_arr("eta", as.numeric(fit$report$eta)))
    ## A polished R optimum: nlminb restarted from the door's optimum on the
    ## SAME TMB objective with tight tolerances. Diagnostic only: it separates
    ## optimiser stopping noise on the R side from a Julia-side difference.
    pol <- stats::nlminb(fit$opt$par, fit$tmb_obj$fn, fit$tmb_obj$gr,
                         control = list(rel.tol = 1e-14, x.tol = 1e-12,
                                        iter.max = 2000L, eval.max = 4000L))
    lines <- c(lines,
      kv_num("polished_loglik", -pol$objective),
      kv_num("polished_convergence", pol$convergence),
      kv_num("polished_max_abs_gradient", max(abs(fit$tmb_obj$gr(pol$par)))),
      kv_arr("polished_b_fix", pol$par[names(pol$par) == "b_fix"]),
      kv_arr("polished_theta_rr_B", pol$par[names(pol$par) == "theta_rr_B"]))
    if (isTRUE(cases[[case]]$predict)) {
      nd0 <- d; nd0$log_support <- 0
      pr <- predict(fit)
      lines <- c(lines,
        kv_sarr("predict_columns", names(pr)),
        kv_arr("predict_link", pr$est),
        kv_arr("predict_response", predict(fit, type = "response")$est),
        kv_arr("predict_link_zero_re", suppressMessages(predict(fit, re_form = ~0))$est),
        kv_arr("predict_newdata_offset0_link",
               suppressMessages(predict(fit, newdata = nd0, type = "link"))$est),
        kv_arr("predict_newdata_offset0_response",
               suppressMessages(predict(fit, newdata = nd0, type = "response"))$est))
    }
    if (!is.null(cases[[case]]$default_formula)) {
      fd <- fit_case(case, cases[[case]]$default_formula)$fit
      lines <- c(lines,
        kv_num("default_unique_loglik", -fd$opt$objective),
        kv_num("default_unique_convergence", fd$opt$convergence),
        kv_arr("default_unique_theta_diag_B", fd$opt$par[names(fd$opt$par) == "theta_diag_B"]))
    }
    cat(case, "loglik", -fit$opt$objective, "conv", fit$opt$convergence, "\n")
  }
  writeLines(lines, file.path(out_dir, "r_values_p1.toml"))
} else if (identical(stage, "grid")) {
  tpl_dir <- tempfile("isdm_cloglog_")
  dir.create(tpl_dir)
  file.copy(file.path(p1_src, "src", "gllvmTMB_cloglog.h"), tpl_dir)
  cpp <- file.path(tpl_dir, "isdm_cloglog_grid.cpp")
  writeLines(c(
    "#include <TMB.hpp>",
    "#include \"gllvmTMB_cloglog.h\"",
    "template<class Type>",
    "Type objective_function<Type>::operator() ()",
    "{",
    "  DATA_SCALAR(y);",
    "  PARAMETER(eta);",
    "  return -gll_dbinom_cloglog(y, Type(1.0), eta);",
    "}"), cpp)
  owd <- setwd(tpl_dir)
  TMB::compile("isdm_cloglog_grid.cpp")
  dyn.load(TMB::dynlib("isdm_cloglog_grid"))
  setwd(owd)
  eta_grid <- c(-40, -30, -25, -20.5, -20, -19.5, -15, -10, -5, -2, -1, -0.5,
                0, 0.5, 1, 2, 3, 5, 10, 40, 100, 699.5, 700, 720)
  stopifnot(length(eta_grid) == 24L)
  rows <- list()
  for (y in c(0, 1)) for (e in eta_grid) {
    obj <- TMB::MakeADFun(list(y = y), list(eta = e), DLL = "isdm_cloglog_grid", silent = TRUE)
    rows[[length(rows) + 1L]] <- data.frame(
      y = y, eta = e,
      value = -obj$fn(e),
      score = -as.numeric(obj$gr(e)),
      weight = as.numeric(obj$he(e)))
  }
  grid <- do.call(rbind, rows)
  path <- file.path(out_dir, "cloglog_grid_p1.csv")
  ## 17 significant digits so the recorded doubles round-trip exactly.
  g <- grid
  for (v in c("eta", "value", "score", "weight")) g[[v]] <- sprintf("%.17g", g[[v]])
  utils::write.csv(g, path, row.names = FALSE, quote = FALSE)
  cat("grid rows", nrow(grid), sha256_file(path), "\n")
} else if (identical(stage, "xobj")) {
  est_path <- file.path(out_dir, "julia_estimates_p1.toml")
  stopifnot(file.exists(est_path))
  est <- readLines(est_path)
  get_arr <- function(section, key) {
    i0 <- which(est == paste0("[", section, "]"))
    stopifnot(length(i0) == 1L)
    nxt <- which(grepl("^\\[", est) & seq_along(est) > i0)
    i1 <- if (length(nxt)) min(nxt) - 1L else length(est)
    ln <- grep(paste0("^", key, " = "), est[i0:i1], value = TRUE)
    stopifnot(length(ln) == 1L)
    body <- sub("^[^=]*= \\[(.*)\\]$", "\\1", ln)
    parts <- trimws(strsplit(body, ",")[[1]])
    parts
  }
  lines <- readLines(file.path(out_dir, "r_values_p1.toml"))
  add <- c("", "[xobj]", "# R's objective (TMB Laplace, -loglik sign flipped) at Julia's estimate")
  for (case in names(cases)) {
    sec <- paste0("julia.", case)
    jn <- gsub('"', "", get_arr(sec, "b_fix_names"))
    jb <- as.numeric(get_arr(sec, "b_fix"))
    fx <- fit_case(case)
    fit <- fx$fit
    rn <- fit$X_fix_names
    if (!setequal(jn, rn)) stop("b_fix names do not pair for ", case)
    par <- fit$opt$par
    par[names(par) == "b_fix"] <- jb[match(rn, jn)]
    if (any(names(par) == "theta_rr_B")) {
      par[names(par) == "theta_rr_B"] <- as.numeric(get_arr(sec, "theta_rr_B"))
    }
    stopifnot(all(names(par) %in% c("b_fix", "theta_rr_B")))
    add <- c(add, kv_num(case, -fit$tmb_obj$fn(par)))
    cat(case, "R loglik at Julia estimate", -fit$tmb_obj$fn(par), "\n")
  }
  add <- c(add, kv_str("julia_estimates_sha256", sha256_file(est_path)))
  writeLines(c(lines[seq_len(if (any(lines == "[xobj]")) which(lines == "[xobj]") - 2L else length(lines))], add),
             file.path(out_dir, "r_values_p1.toml"))
} else if (identical(stage, "admission")) {
  contract <- jsonlite::fromJSON(file.path("docs", "dev-log", "core070", "isdm-batch-contract.json"),
                                 simplifyVector = FALSE)
  env <- new.env(parent = asNamespace("gllvmTMB"))
  sys.source(file.path("test", "parity", "fixtures", "core070_isdm_admission.R"), envir = env)
  lines <- c("# R replay at P1 of the 20 iSDM admission predicates. Written by",
             "# test/fixtures/isdm/export_p1_fixtures.R (stage admission). Do not edit.",
             kv_str("gllvmtmb_sha", P1_SHA),
             kv_str("contract_sha256", sha256_file(file.path("docs", "dev-log", "core070", "isdm-batch-contract.json"))),
             kv_str("fixture_sha256", sha256_file(file.path("test", "parity", "fixtures", "core070_isdm_admission.R"))),
             "", "[cases]")
  for (cs in contract$cases) {
    val <- tryCatch(as.character(isTRUE(eval(parse(text = cs$expression), env))),
                    error = function(e) paste0("ERROR: ", conditionMessage(e)))
    lines <- c(lines, kv_str(fmt_str(cs$admission_case_id), val))
    cat(cs$admission_case_id, val, "\n")
  }
  writeLines(lines, file.path(out_dir, "admission_p1.toml"))
} else {
  stop("Unknown stage: ", stage)
}
