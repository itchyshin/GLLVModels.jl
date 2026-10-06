# Retained evidence runner for the "namespace-2" manifest-area batch (9
# EXECUTABLE_NOW cases + 4 negative controls; see
# docs/dev-log/core070/namespace-2-batch-contract.json for the full case
# list, the 47 NEEDS_NEW_JULIA_SURFACE deferrals, and the 34
# REUSED_OR_RECLASSIFY rows carried over verbatim from the manifest draft's
# own `reclassify` proposals).
#
# Design follows tools/core070_postfit_policy_batch.R's repair pattern: THIS
# R process (the one with the frozen gllvmTMB library loaded) does 100% of
# the live R-side computation itself and hands the Julia child a plain JSON
# oracle file; the Julia child (tools/core070_namespace_2_batch.jl) runs zero
# R code and carries no RCall dependency at all.
#
# Two cases (gllvm_julia_setup / gllvm_julia_fit bridge admission) are
# deliberately scoped to avoid embedding a live JuliaCall session inside this
# R process -- see the contract's `rationale_notes` for why (the same class
# of live-cross-language-embed fragility that forced the postfit-policy
# repair). Two more cases (lognormal / truncated_poisson family-bridge) are
# pure R-side admission-gate rejection checks and touch neither a live model
# fit nor the Julia child at all.
#
# argv (matches the contract's runner.outer_argv and the --self-test local
# smoke, which never reaches this file):
#   Rscript --vanilla tools/core070_namespace_2_batch.R <frozen-library> <destination>
#
# Pin switch (D-294/D-295): GLLVM_PARITY_PIN unset or "P0" keeps the frozen P0
# contract and the P0 readback tree, unchanged. "P1" reads
# docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json
# (tools/core070_namespace_2_p1_contract.py: cases verbatim, source pins at P1),
# reads the pinned R sources with `git -C $GLLVMTMB_DIR show <P1>:<path>`, and
# requires the installed library's NAMESPACE hash and version to match
# tools/core070_oracle_pins.toml [P1]:
#   GLLVM_PARITY_PIN=P1 GLLVMTMB_DIR=<clone> Rscript --vanilla tools/core070_namespace_2_batch.R <lib> <dest>
#
# <frozen-library> is an R library directory containing an installed
# gllvmTMB built from the contract's pinned reference_commit (P0 b4d5fee... or
# P1 9539352f..., per GLLVM_PARITY_PIN) -- the FROZEN, INSTALLED library,
# matching masks_known.R's arg 1, not a source tree.
# <destination> must not already exist; it is created and holds
# r-oracle.json, julia-results.json, julia-stdout.log, julia-stderr.log,
# results.tsv, diagnostics.log, and receipt.json.

args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
frozen_library <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
stopifnot(!dir.exists(output_dir))
dir.create(output_dir, recursive = TRUE)

sha256_file <- function(path) {
  command <- if (nzchar(Sys.which("sha256sum"))) "sha256sum" else "shasum"
  argv <- if (identical(command, "sha256sum")) path else c("-a", "256", path)
  line <- system2(command, argv, stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(line, "status")), length(line) >= 1L)
  sub("[[:space:]].*$", "", line[[1L]])
}

.libPaths(c(frozen_library, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package("gllvmTMB")) ==
          normalizePath(file.path(frozen_library, "gllvmTMB")))

root <- normalizePath(".")
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P0")))
if (!parity_pin %in% c("P0", "P1")) stop("GLLVM_PARITY_PIN must be P0 or P1, got '", parity_pin, "'")
expected_reference <- if (identical(parity_pin, "P1"))
  "9539352f66f2db2cc26b1c393e67212a359b60c9" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
contract_path <- file.path(root, if (identical(parity_pin, "P1"))
  "docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json" else
  "docs/dev-log/core070/namespace-2-batch-contract.json")
contract <- jsonlite::read_json(contract_path, simplifyVector = FALSE)
contract_sha256 <- sha256_file(contract_path)

stopifnot(identical(contract$reference_commit, expected_reference),
          identical(contract$status, "FROZEN_NAMESPACE_2_BATCH_CONTRACT"),
          length(contract$cases) == contract$expected_case_count,
          contract$expected_case_count == 9L,
          length(contract$negative_controls) >= 2L)

# --- installed library at the pin (P1) ------------------------------------
# The library's NAMESPACE hash and version must match tools/core070_oracle_pins.toml.
# A CORE070_SOURCE_PIN.toml marker, when present, is checked too (tools/core070_source_pin.R).
library_pin <- NULL
if (identical(parity_pin, "P1")) {
  source(file.path(root, "tools/core070_source_pin.R"))
  oracle_pin <- core070_read_flat_toml(file.path(root, "tools/core070_oracle_pins.toml"), "P1")
  stopifnot(identical(oracle_pin$reference_commit, expected_reference))
  lib_ns <- file.path(frozen_library, "gllvmTMB", "NAMESPACE")
  library_pin <- list(
    namespace_sha256 = sha256_file(lib_ns),
    version = as.character(utils::packageVersion("gllvmTMB")),
    marker = file.exists(file.path(frozen_library, "gllvmTMB", "CORE070_SOURCE_PIN.toml"))
  )
  stopifnot(identical(library_pin$namespace_sha256, oracle_pin$namespace_sha256),
            identical(library_pin$version, oracle_pin$version))
  if (isTRUE(library_pin$marker)) {
    library_pin$source_pin <- core070_source_pin(root, frozen_library, parity_pin, expected_reference)
  }
}

# --- validate pinned R source ------------------------------------------
# P0: the pre-extracted readback tree under .unlazy/. source_access = "git-show"
# (P1): every byte comes from `git -C $GLLVMTMB_DIR show <commit>:<path>` into a
# fresh temp dir; the gllvmTMB clone is never checked out or edited.
gllvmtmb_resolved <- NA_character_
if (identical(contract$source_access, "git-show")) {
  gdir <- Sys.getenv("GLLVMTMB_DIR", unset = "")
  if (!nzchar(gdir)) stop("FATAL: GLLVMTMB_DIR must name a local gllvmTMB clone for pin ", parity_pin)
  gllvmtmb_resolved <- system2("git", c("-C", shQuote(gdir), "rev-parse",
                                        shQuote(paste0(expected_reference, "^{commit}"))),
                               stdout = TRUE)
  stopifnot(identical(gllvmtmb_resolved, expected_reference))
  source_root <- tempfile("core070-namespace-2-src-")
  dir.create(source_root)
  for (rel in names(contract$source_pins)) {
    dest <- file.path(source_root, rel)
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    status <- system2("git", c("-C", shQuote(gdir), "show",
                               shQuote(paste0(expected_reference, ":", rel))), stdout = dest)
    stopifnot(identical(status, 0L))
  }
} else {
  source_root <- file.path(root, ".unlazy/core070-aghq/oracle-source/readback")
}
for (rel in names(contract$source_pins)) {
  path <- file.path(source_root, rel)
  stopifnot(file.exists(path))
  digest <- sha256_file(path)
  stopifnot(identical(digest, contract$source_pins[[rel]]))
}

# ---------------------------------------------------------------------------
# 0. Pure-R admission-gate mapper, sourced directly from the pinned
#    R/julia-bridge.R (no gllvmTMB package build required for this half --
#    it works even without the frozen library, see the batch's local smoke).
#    Used for the dispatch-key checks and the two rejection-path cases.
# ---------------------------------------------------------------------------
.ns2_bridge_env <- new.env()
sys.source(file.path(source_root, "R/julia-bridge.R"), envir = .ns2_bridge_env)
`%||%` <- function(a, b) if (is.null(a)) b else a
assign("%||%", get("%||%"), envir = .ns2_bridge_env)

.ns2_family_key <- function(family_string) {
  tryCatch(list(ok = TRUE, key = .ns2_bridge_env$.gllvm_julia_family_scalar(family_string), error = ""),
            error = function(e) list(ok = FALSE, key = NA_character_, error = conditionMessage(e)))
}

# ---------------------------------------------------------------------------
# 1. Gaussian fixture (same convention as tools/core070_postfit_policy_batch.R;
#    a fresh, independently-seeded draw).
# ---------------------------------------------------------------------------
set.seed(42)
p <- 5L; K <- 2L; n <- 80L
Lambda_true <- matrix(c(
  0.8,  0.0,
  0.5,  0.6,
  0.3, -0.4,
 -0.2,  0.5,
  0.1,  0.3
), nrow = p, ncol = K, byrow = TRUE)
sigma_true <- 0.7
eta_g <- matrix(rnorm(K * n), nrow = K, ncol = n)
Y_g <- Lambda_true %*% eta_g + sigma_true * matrix(rnorm(p * n), nrow = p, ncol = n)
Y_g <- Y_g - rowMeans(Y_g)

trait_names_g <- paste0("t", seq_len(p))
df_long_g <- data.frame(
  site  = factor(rep(seq_len(n), each = p)),
  trait = factor(rep(trait_names_g, times = n), levels = trait_names_g),
  value = as.vector(Y_g)
)

fit_g <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_long_g, unit = "site", trait = "trait", family = gaussian(),
  control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
)
stopifnot("gllvmTMB_multi" %in% class(fit_g))
oracle_g <- list(
  coef = as.numeric(coef(fit_g)),
  loglik = as.numeric(logLik(fit_g)),
  # Loadings comparand (namespace export/gllvmTMB option (a), signed 2026-10-05 under D-319): the
  # contract names loadings as a comparand. Lambda Lambda' is rotation and sign invariant; with
  # unique = FALSE the unit-level Sigma is the latent part only. p x p, column-major.
  LLt = as.numeric(extract_Sigma(fit_g, level = "unit")$Sigma)
)

# --- gllvmTMB_wide consistency (pure R-internal, no Julia involved) --------
# Batch-spec repair (2026-09-01): gllvmTMB_wide constructs
# `latent(0 + trait | site, d = K)` at the DEFAULT `unique`, whereas oracle_g
# above is fit with `unique = FALSE` — a smaller model (15 vs 19 df on this
# fixture; measured logLik gap 6.38). Comparing the wide wrapper against the
# unique=FALSE oracle was a wrong-model comparison in the original batch spec.
# The correct reference is a long-formula fit with the same default `unique`;
# measured agreement on this fixture is ~1e-8 (two independent optimizations),
# so the check tolerance is 1e-6 — calibrated at authoring time, not a
# widening of any previously accepted contract.
fit_g_default_unique <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K),
  data = df_long_g, unit = "site", trait = "trait", family = gaussian(),
  control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
)
Y_wide <- t(Y_g)                      # n_sites x n_species, matching gllvmTMB_wide's convention
colnames(Y_wide) <- trait_names_g
fit_wide <- gllvmTMB_wide(Y_wide, d = K, family = gaussian(),
                           control = gllvmTMBcontrol(n_init = 1L, se = FALSE))
loglik_wide <- as.numeric(logLik(fit_wide))
wide_delta <- abs(loglik_wide - as.numeric(logLik(fit_g_default_unique)))

# ---------------------------------------------------------------------------
# 2. Negative-binomial fixture (shared by nbinom1 and nbinom2 fits).
# ---------------------------------------------------------------------------
set.seed(4201)
p_nb <- 5L; K_nb <- 1L; n_nb <- 60L
Lambda_nb <- matrix(c(0.6, 0.4, -0.3, 0.5, 0.2), nrow = p_nb, ncol = K_nb)
eta_nb <- matrix(rnorm(K_nb * n_nb), nrow = K_nb, ncol = n_nb)
trait_int_nb <- c(0.5, 0.2, -0.1, 0.3, 0.0)
mu_lin <- Lambda_nb %*% eta_nb + trait_int_nb
mu_nb <- exp(mu_lin)
Y_nb <- matrix(rnbinom(p_nb * n_nb, mu = as.vector(mu_nb), size = 4), nrow = p_nb, ncol = n_nb)

trait_names_nb <- paste0("t", seq_len(p_nb))
df_long_nb <- data.frame(
  site  = factor(rep(seq_len(n_nb), each = p_nb)),
  trait = factor(rep(trait_names_nb, times = n_nb), levels = trait_names_nb),
  count = as.vector(Y_nb)
)

fit_nb1 <- gllvmTMB(
  count ~ 0 + trait + latent(0 + trait | site, d = K_nb, unique = FALSE),
  data = df_long_nb, unit = "site", trait = "trait", family = nbinom1(),
  control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
)
fit_nb2 <- gllvmTMB(
  count ~ 0 + trait + latent(0 + trait | site, d = K_nb, unique = FALSE),
  data = df_long_nb, unit = "site", trait = "trait", family = nbinom2(),
  control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
)
oracle_nb1 <- list(loglik = as.numeric(logLik(fit_nb1)))
oracle_nb2 <- list(loglik = as.numeric(logLik(fit_nb2)))

# ---------------------------------------------------------------------------
# 3. Ordinal (probit) fixture.
# ---------------------------------------------------------------------------
set.seed(4301)
p_ord <- 4L; K_ord <- 1L; n_ord <- 60L; n_cat <- 3L
Lambda_ord <- matrix(c(0.7, -0.5, 0.4, 0.3), nrow = p_ord, ncol = K_ord)
eta_ord <- matrix(rnorm(K_ord * n_ord), nrow = K_ord, ncol = n_ord)
lin_ord <- Lambda_ord %*% eta_ord
cuts <- c(-0.4, 0.4)
prob_to_cat <- function(x) {
  p_star <- pnorm(c(cuts, Inf) - x) - pnorm(c(-Inf, cuts) - x)
  sample.int(n_cat, size = 1, prob = pmax(p_star, 1e-8))
}
Y_ord <- matrix(vapply(as.vector(lin_ord), prob_to_cat, integer(1)), nrow = p_ord, ncol = n_ord)

trait_names_ord <- paste0("t", seq_len(p_ord))
df_long_ord <- data.frame(
  site  = factor(rep(seq_len(n_ord), each = p_ord)),
  trait = factor(rep(trait_names_ord, times = n_ord), levels = trait_names_ord),
  cat   = factor(as.vector(Y_ord), levels = seq_len(n_cat), ordered = TRUE)
)

fit_ord <- gllvmTMB(
  cat ~ 0 + trait + latent(0 + trait | site, d = K_ord, unique = FALSE),
  data = df_long_ord, unit = "site", trait = "trait", family = ordinal_probit(),
  control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
)
oracle_ord <- list(loglik = as.numeric(logLik(fit_ord)))

# ---------------------------------------------------------------------------
# 4. Admission-gate dispatch-key results (pure R, no live fit).
# ---------------------------------------------------------------------------
gate_gaussian <- .ns2_family_key("gaussian")
gate_nb1      <- .ns2_family_key("nbinom1")
gate_nb2      <- .ns2_family_key("nbinom2")
gate_ordprob  <- .ns2_family_key("ordinal_probit")
gate_lognorm  <- .ns2_family_key("lognormal")
gate_truncpois <- .ns2_family_key("truncated_poisson")

# gllvm_julia_setup precondition check (no live JuliaCall invocation).
julia_bin_r <- Sys.which("julia")
setup_precondition_r <- nzchar(julia_bin_r)

# ---------------------------------------------------------------------------
# 5. Write the oracle JSON (all fixtures + R-side values) for the Julia child.
# ---------------------------------------------------------------------------
oracle_path <- file.path(output_dir, "r-oracle.json")
jsonlite::write_json(
  list(
    schema = "core070-namespace-2-r-oracle/v1",
    gaussian = list(p = p, n = n, K = K, y = as.numeric(Y_g),
                     coef = oracle_g$coef, loglik = oracle_g$loglik, LLt = oracle_g$LLt),
    gllvmtmb_wide = list(loglik_wide = loglik_wide, loglik_long = oracle_g$loglik,
                          delta = wide_delta),
    nb = list(p = p_nb, n = n_nb, K = K_nb, y = as.numeric(Y_nb),
              loglik_nbinom1 = oracle_nb1$loglik, loglik_nbinom2 = oracle_nb2$loglik),
    ordinal = list(p = p_ord, n = n_ord, K = K_ord, n_categories = n_cat,
                    y = as.integer(Y_ord), loglik = oracle_ord$loglik),
    gate = list(
      gaussian = gate_gaussian, nbinom1 = gate_nb1, nbinom2 = gate_nb2,
      ordinal_probit = gate_ordprob, lognormal = gate_lognorm,
      truncated_poisson = gate_truncpois
    ),
    setup_precondition_r = setup_precondition_r,
    julia_bin_r = julia_bin_r
  ),
  oracle_path, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = 17
)

# --- invoke the Julia child ---------------------------------------------
julia_out <- file.path(output_dir, "julia-results.json")
julia_env <- c(CORE070_NAMESPACE_2_R_ORACLE = oracle_path)
julia_bin <- Sys.which("julia")
stopifnot(nzchar(julia_bin))
old_env <- Sys.getenv(names(julia_env), unset = NA, names = TRUE)
do.call(Sys.setenv, as.list(julia_env))
t0 <- Sys.time()
julia_status <- tryCatch(
  system2(julia_bin, c("--project=.", "tools/core070_namespace_2_batch.jl", julia_out),
          stdout = file.path(output_dir, "julia-stdout.log"),
          stderr = file.path(output_dir, "julia-stderr.log")),
  finally = {
    unset <- names(old_env)[is.na(old_env)]
    if (length(unset)) Sys.unsetenv(unset)
    set <- old_env[!is.na(old_env)]
    if (length(set)) do.call(Sys.setenv, as.list(set))
  }
)
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

# --- read back and evaluate ----------------------------------------------
julia_report <- if (file.exists(julia_out)) {
  jsonlite::read_json(julia_out, simplifyVector = FALSE)
} else {
  NULL
}

contract_case_ids <- vapply(contract$cases, `[[`, "", "case_id")
contract_neg_ids <- vapply(contract$negative_controls, `[[`, "", "control_id")

# gllvmTMB_wide + the two rejection-path cases are decided entirely R-side
# (never touch julia_report); merge them into the same PASS/FAIL vocabulary
# the Julia child uses for the other 6 cases before folding into one receipt.
r_only_cases <- list(
  "CORE070-NAMESPACE2-GLLVMTMB-WIDE-NATIVE-FIT" = list(
    pass = isTRUE(wide_delta <= 1e-6), delta = wide_delta),
  "CORE070-NAMESPACE2-LOGNORMAL-FAMILY-BRIDGE" = list(
    pass = isTRUE(!gate_lognorm$ok && grepl("GJL-GATE-FAMILY", gate_lognorm$error, fixed = TRUE) &&
                   grepl("lognormal", gate_lognorm$error, fixed = TRUE)),
    error = gate_lognorm$error),
  "CORE070-NAMESPACE2-TRUNCATED-POISSON-FAMILY-BRIDGE" = list(
    pass = isTRUE(!gate_truncpois$ok && grepl("GJL-GATE-FAMILY", gate_truncpois$error, fixed = TRUE) &&
                   grepl("truncated_poisson", gate_truncpois$error, fixed = TRUE)),
    error = gate_truncpois$error)
)

results_ok <- !is.null(julia_report) &&
  identical(julia_status, 0L) &&
  identical(julia_report$status, "PASS") &&
  isTRUE(julia_report$all_positive_pass) &&
  isTRUE(julia_report$negative_controls_behaved_as_expected) &&
  isTRUE(julia_report$all_checks) &&
  all(vapply(julia_report$cases, function(x) isTRUE(x$pass), logical(1))) &&
  all(vapply(r_only_cases, function(x) isTRUE(x$pass), logical(1)))

all_case_ids_seen <- if (!is.null(julia_report)) union(names(julia_report$cases), names(r_only_cases)) else names(r_only_cases)
missing_case_ids <- setdiff(contract_case_ids, all_case_ids_seen)
extra_case_ids <- setdiff(all_case_ids_seen, contract_case_ids)
results_ok <- results_ok && length(missing_case_ids) == 0 && length(extra_case_ids) == 0

raw_lines <- vapply(contract_case_ids, function(id) {
  if (id %in% names(r_only_cases)) {
    pass <- isTRUE(r_only_cases[[id]]$pass)
  } else if (!is.null(julia_report) && id %in% names(julia_report$cases)) {
    pass <- isTRUE(julia_report$cases[[id]]$pass)
  } else {
    pass <- FALSE
  }
  paste(id, if (pass) "PASS" else "FAIL", "positive", sep = "\t")
}, character(1))
neg_lines <- if (!is.null(julia_report)) {
  vapply(contract_neg_ids, function(id) {
    behaved <- isTRUE(julia_report$negative_controls[[id]]$behaved)
    paste(id, if (behaved) "PASS" else "FAIL", "negative_control", sep = "\t")
  }, character(1))
} else character(0)
raw_path <- file.path(output_dir, "results.tsv")
writeLines(c(raw_lines, neg_lines), raw_path)

diag_lines <- character(0)
if (!identical(julia_status, 0L)) diag_lines <- c(diag_lines, paste("julia_exit_code", julia_status))
if (is.null(julia_report)) diag_lines <- c(diag_lines, "julia-results.json was not written")
if (length(missing_case_ids)) diag_lines <- c(diag_lines, paste("missing_case_ids:", paste(missing_case_ids, collapse = ", ")))
if (length(extra_case_ids)) diag_lines <- c(diag_lines, paste("extra_case_ids:", paste(extra_case_ids, collapse = ", ")))
diag_path <- file.path(output_dir, "diagnostics.log")
writeLines(diag_lines, diag_path)

receipt <- list(
  status = if (isTRUE(results_ok)) "PASS" else "FAIL",
  scope = "CORE070_NAMESPACE_2_BATCH",
  reference_commit = contract$reference_commit,
  contract_sha256 = contract_sha256,
  source_pins = contract$source_pins,
  source_unchanged = TRUE,
  case_count = contract$expected_case_count,
  negative_control_count = length(contract$negative_controls),
  expected_case_ids = contract_case_ids,
  negative_control_case_ids = contract_neg_ids,
  julia_exit_code = julia_status,
  julia_elapsed_seconds = elapsed,
  julia_results_sha256 = if (file.exists(julia_out)) sha256_file(julia_out) else NA_character_,
  raw_sha256 = sha256_file(raw_path),
  diagnostics_sha256 = sha256_file(diag_path),
  r_version = R.version.string,
  gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
  frozen_library = frozen_library,
  pin = parity_pin,
  source_access = contract$source_access %||% "readback",
  gllvmtmb_resolved_commit = gllvmtmb_resolved,
  library_pin = library_pin
)
receipt_path <- file.path(output_dir, "receipt.json")
jsonlite::write_json(receipt, receipt_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

cat("CORE070_NAMESPACE_2_BATCH_", receipt$status, "\n", sep = "")
quit(status = if (isTRUE(results_ok)) 0L else 1L)
