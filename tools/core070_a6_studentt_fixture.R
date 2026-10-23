# A6 Student-t interior-nu paired fixture -- R stage (maintainer round2-3 #11).
#
# Fits the frozen gllvmTMB 0.7.0 `student()` family (families.R:367-381) on
# the SAME response matrix the Julia stage generated at a moderate INTERIOR
# nu (the panel's finding-4 fixture pushed estimated nu to the flat
# Gaussian-limit boundary on pure-Gaussian data; this fixture is
# deliberately genuinely heavy-tailed so both engines are expected to land
# INTERIOR).
#
# REVISION (post-Totoro run suite-run-01/a6-run): the original single
# df=NULL comparison came back HONESTLY FAILED with a structural finding,
# not a numeric near-miss -- R's student(df=NULL) estimates ONE degrees of
# freedom PER TRAIT (documented behaviour, R/gllvmTMB.R:167-168: "The
# student() family fits one log-sigma and one log(df-1) per trait";
# confirmed in the per-trait `dispersion_trait_map`/pin machinery,
# R/fit-multi.R:5317-5349), while GLLVModels.jl's default `disp_group = :shared`
# fits ONE shared degrees of freedom across all traits. Two genuinely
# different models were being compared numerically -- the same
# dispersion-structure class as the NB2 benchmark's
# shared-r-vs-per-trait-phi lesson.
#
# This script now fits BOTH cases in one invocation, using the frozen
# `student()` constructor's documented `df` argument
# (`student(link = "identity", df = <n>)` PINS every trait's degrees of
# freedom at exactly `<n>` -- R/fit-multi.R:5325-5342 -- so a single scalar
# is model-matched to a single scalar `nu` fixed on the Julia side, unlike
# the free case):
#
#   "fixed"  (PRIMARY, GATING on the Julia side): student(df = FIXTURE_NU) --
#            matched models.
#   "free"   (SECONDARY, NON-GATING on the Julia side): student(df = NULL) --
#            R's documented per-trait default; recorded as a structural
#            finding only, never numerically gated.
#
# House convention: argv 2, base-R only (no jsonlite dependency) -- a flat
# `<case>_<key><TAB>[index<TAB>]value` table, sprintf("%.17g", .) for every
# numeric, mirroring tools/core070_student_warmstart_readback.R and
# tools/core070_aghq_frozen_reference.R. Never `library(gllvmTMB)` at parse
# time inside a `Filter`/`substitute` trick -- this script does two genuine
# live fits, not a frozen-helper extraction.
#
# argv[1]: path to the Y matrix (p rows x n columns, whitespace-delimited,
#          written by the Julia stage's generation pass -- `read.table`).
# argv[2]: output path for this script's TSV readback (both cases).
#
# K (latent dimension) and FIXTURE_NU (the pinned df for the "fixed" case)
# are FIXED design constants of this specific fixture, not runtime
# arguments -- matching the argv-2 house convention, where only data/output
# paths are positional and fixture design constants are frozen in both
# sibling scripts and the contract JSON.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
data_path <- args[[1L]]
output_path <- args[[2L]]
stopifnot(file.exists(data_path))

K <- 1L
FIXTURE_NU <- 6.0

Y <- as.matrix(read.table(data_path, header = FALSE))
p <- nrow(Y); n <- ncol(Y)
stopifnot(p >= 2L, n >= 2L)

library(gllvmTMB)

# Pin switch (D-294/D-295): GLLVM_PARITY_PIN unset or "P0" runs as before; "P1"
# requires the installed library's CORE070_SOURCE_PIN.toml marker to match the
# P1 entry of tools/core070_oracle_pins.toml (tools/core070_source_pin.R) and
# records it beside the readback as <output>.source-pin.tsv. Any other value
# stops. Nothing is written when a check fails.
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P1")))
if (!parity_pin %in% c("P0", "P1")) stop("GLLVM_PARITY_PIN must be P0 or P1, got '", parity_pin, "'")
expected_reference <- if (identical(parity_pin, "P1"))
  "9539352f66f2db2cc26b1c393e67212a359b60c9" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
root <- normalizePath(".")
source(file.path(root, "tools/core070_source_pin.R"))
source_pin <- core070_source_pin(root, dirname(find.package("gllvmTMB")), parity_pin, expected_reference)

trait_levels <- paste0("t", seq_len(p))
df_data <- data.frame(
  site  = factor(rep(seq_len(n), each = p)),
  trait = factor(rep(trait_levels, times = n), levels = trait_levels),
  value = as.vector(Y)
)

# Fit one student() case and return its readback rows, prefixed
# `<case_prefix>_<key>`. Shared by both the "fixed" (df = FIXTURE_NU) and
# "free" (df = NULL) cases below -- identical extraction code path either
# way; only the family object differs.
fit_one_case <- function(case_prefix, student_family) {
  fit_warnings <- character()
  fit <- withCallingHandlers(
    gllvmTMB(
      value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
      data = df_data, unit = "site", trait = "trait",
      family = student_family,
      control = gllvmTMBcontrol(se = FALSE)
    ),
    warning = function(w) {
      fit_warnings <<- c(fit_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  opt <- fit$opt
  obj <- fit$tmb_obj
  pars <- opt$par
  gradient <- as.numeric(obj$gr(pars))
  report <- obj$report(obj$env$last.par)
  par_list <- obj$env$parList(x = pars, par = obj$env$last.par)

  # Inline health check, mirroring test/parity/r_health.R's
  # core070_tweedie_health() shape (optimizer code + finiteness + a
  # scaled-gradient bound) but specialised to the student() family's own
  # boundary: `log_df_student` running large positive is the nu -> infinity
  # Gaussian-limit boundary this fixture is designed to sit AWAY from
  # (finding 4, docs/dev-log/core070/parity-panel-2026-09-01.md).
  nu_boundary_max <- 1e6
  g_tol <- 1e-3
  optimizer_ok <- isTRUE(as.integer(opt$convergence) == 0L)
  finite_objective <- length(opt$objective) == 1L && is.finite(opt$objective)
  finite_parameters <- is.numeric(pars) && length(pars) > 0L && all(is.finite(pars))
  finite_gradient <- is.numeric(gradient) && length(gradient) == length(pars) && all(is.finite(gradient))
  scaled_gradient <- if (finite_gradient && finite_objective) {
    max(abs(gradient)) / max(1, abs(opt$objective))
  } else Inf
  sigma_student <- as.numeric(report$sigma_student)
  df_student <- as.numeric(report$df_student)
  nu_at_boundary <- length(df_student) > 0L && any(is.finite(df_student) & df_student > nu_boundary_max)
  healthy <- optimizer_ok && finite_objective && finite_parameters && finite_gradient &&
    scaled_gradient <= g_tol && !nu_at_boundary && all(is.finite(sigma_student)) && all(sigma_student > 0)

  rows <- character(0)
  push <- function(key, value) {
    rows[[length(rows) + 1L]] <<- paste(paste0(case_prefix, "_", key), sprintf("%.17g", value), sep = "\t")
  }
  push_vec <- function(key, values) {
    for (i in seq_along(values)) {
      rows[[length(rows) + 1L]] <<- paste(paste0(case_prefix, "_", key), i, sprintf("%.17g", values[[i]]), sep = "\t")
    }
  }

  push("p", p)
  push("K", K)
  push("n", n)
  push("loglik", -opt$objective)   # opt$objective is the NEGATIVE log-likelihood
  push("optimizer_convergence_code", opt$convergence)
  push("healthy", as.integer(healthy))
  push("nu_at_boundary", as.integer(nu_at_boundary))
  push("scaled_gradient", scaled_gradient)
  push_vec("beta", as.numeric(par_list$b_fix))
  push_vec("sigma_student", sigma_student)
  push_vec("df_student", df_student)
  push_vec("loading", as.numeric(as.matrix(report$Lambda_B)))
  rows[[length(rows) + 1L]] <- paste(paste0(case_prefix, "_n_warnings"), length(fit_warnings), sep = "\t")
  rows
}

fixed_rows <- fit_one_case("fixed", student(link = "identity", df = FIXTURE_NU))
free_rows  <- fit_one_case("free",  student(link = "identity"))

writeLines(c(fixed_rows, free_rows), output_path)
if (!is.null(source_pin)) {
  writeLines(c(paste("parity_pin", parity_pin, sep = "\t"),
               paste("gllvmTMB_version", as.character(utils::packageVersion("gllvmTMB")), sep = "\t"),
               paste("r_version", R.version.string, sep = "\t"),
               paste(names(source_pin), unlist(source_pin, use.names = FALSE), sep = "\t")),
             paste0(output_path, ".source-pin.tsv"))
}
cat("CORE070_A6_STUDENTT_R_FIT_PASS", output_path, "\n")
