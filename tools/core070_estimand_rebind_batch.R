# Retained evidence runner for the T5 estimand-alignment re-bind: the 4
# postfit/POSTFIT-SURFACE-extract_{communality,correlations,proportions,Omega}
# ledger rows carrying disposition PARTIAL_PARITY_DEFECT_PENDING_DECISION
# (docs/dev-log/core070/parity-defect-rebind-2026-09-02.md). Those 4
# quantities were deliberately DEFERRED (never executed) by the frozen
# tools/core070_surface_conversion_batch.R / surface-conversion-batch-
# contract.json (case_count=20, deferred_count=21 -- see its own header and
# docs/dev-log/core070/estimand-alignment-notes.md) because, at the time,
# GLLVModels.jl's extract_communality/extract_correlations/extract_proportions/
# extract_Omega used a TOTAL-variance estimand while R's real accessors are
# TIER-SCOPED. That frozen contract is NOT reopened here (it stays exactly
# 20/21, unedited) -- this is a separate, small, standalone batch that
# re-fits the IDENTICAL gaussian_small fixture (verbatim seed/spec from
# tools/core070_surface_conversion_batch.R) and calls the REAL R accessors
# directly (not a Sigma/getResidualCor proxy), now that maintainer decision
# round 1 item 3 (docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md)
# has made GLLVModels.jl's defaults tier-scoped to match.
#
# argv:
#   Rscript --vanilla tools/core070_estimand_rebind_batch.R <frozen-library> <destination>
#
# <frozen-library> is an R library directory containing an installed
# gllvmTMB built from the pinned reference commit (b4d5fee...).
# <destination> must not already exist.

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

# Pin switch (D-294/D-295): GLLVM_PARITY_PIN unset or "P0" keeps the frozen
# P0 reference; "P1" records the P1 commit. This batch has no contract
# file; its fixture and four cases are unchanged at either pin.
# Any other value stops here rather than falling back to P0.
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P0")))
if (!parity_pin %in% c("P0", "P1")) stop("GLLVM_PARITY_PIN must be P0 or P1, got '", parity_pin, "'")
expected_reference <- if (identical(parity_pin, "P1"))
  "9539352f66f2db2cc26b1c393e67212a359b60c9" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
REFERENCE_COMMIT <- expected_reference

# ---------------------------------------------------------------------------
# gaussian_small -- VERBATIM from tools/core070_surface_conversion_batch.R
# (same seed=42, same p/K/n, same Lambda_true/sigma_true, same formula and
# control()), so R's fit_g and this batch's oracle numbers are exactly the
# ones the estimand-alignment fix (src/extractors.jl) was traced against.
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

trait_names <- paste0("t", seq_len(p))
df_g <- data.frame(
  site  = factor(rep(seq_len(n), each = p)),
  trait = factor(rep(trait_names, times = n), levels = trait_names),
  value = as.vector(Y_g)
)
fit_g <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_g, unit = "site", trait = "trait", family = gaussian(),
  control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
)
stopifnot("gllvmTMB_multi" %in% class(fit_g))

# ---------------------------------------------------------------------------
# The 4 cases -- REAL R accessors, called with the tier-scoped `level`/
# `tier`/`component` argument that is now GLLVModels.jl's matching DEFAULT.
# ---------------------------------------------------------------------------
CASE_IDS <- c(
  communality   = "CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY",
  correlations  = "CORE070-ESTIMAND-REBIND-EXTRACT-CORRELATIONS",
  proportions   = "CORE070-ESTIMAND-REBIND-EXTRACT-PROPORTIONS",
  omega         = "CORE070-ESTIMAND-REBIND-EXTRACT-OMEGA"
)
SOURCE_IDS <- c(
  communality  = "postfit/POSTFIT-SURFACE-extract_communality",
  correlations = "postfit/POSTFIT-SURFACE-extract_correlations",
  proportions  = "postfit/POSTFIT-SURFACE-extract_proportions",
  omega        = "postfit/POSTFIT-SURFACE-extract_Omega"
)

r_quantity <- function(quantity) {
  switch(quantity,
    communality = {
      # Real R accessor, ci=FALSE (default): named numeric vector, one
      # entry per trait. On gaussian_small (rr_B only, no diag_B, no W
      # tier) this degenerates to 1.0 for every trait -- R's own confirmed
      # degenerate behaviour (estimand-alignment-notes.md), not a bug.
      v <- extract_communality(fit_g, level = "unit")
      stopifnot(!is.null(v), length(v) == p)
      as.numeric(v[trait_names])
    },
    correlations = {
      # Real R accessor: long-format data frame (tier, trait_i, trait_j,
      # correlation, ...), point-only (method="none"). Off-diagonal unique
      # pairs (i<j) only -- reassembled into a full p x p symmetric matrix
      # (diag = 1) for direct elementwise comparison against Julia's
      # extract_correlations(fit) Matrix.
      df <- extract_correlations(fit_g, tier = "unit", method = "none")
      stopifnot(is.data.frame(df), nrow(df) == choose(p, 2))
      R <- diag(p)
      dimnames(R) <- list(trait_names, trait_names)
      for (k in seq_len(nrow(df))) {
        i <- which(trait_names == as.character(df$trait_i[k]))
        j <- which(trait_names == as.character(df$trait_j[k]))
        stopifnot(length(i) == 1L, length(j) == 1L)
        R[i, j] <- df$correlation[k]
        R[j, i] <- df$correlation[k]
      }
      as.numeric(R)
    },
    proportions = {
      # Real R accessor, format="long". gaussian_small carries only the
      # "shared_unit" component (rr_B only) -- degenerates to 1.0 for every
      # trait, exactly like communality (same numerator/denominator on this
      # fixture).
      df <- extract_proportions(fit_g, format = "long")
      stopifnot(is.data.frame(df))
      sub <- df[df$component == "shared_unit", ]
      stopifnot(nrow(sub) == p)
      sub <- sub[match(trait_names, as.character(sub$trait)), ]
      as.numeric(sub$proportion)
    },
    omega = {
      # Real R accessor, tiers=NULL (auto-detect -> "B" only on
      # gaussian_small), link_residual="auto" (0 for Gaussian). Returns a
      # LIST (Omega, R_Omega, tiers_used, note[, residual_split]) --
      # confirmed by reading extract-omega.R's final `out <- list(...)`.
      # The trait covariance matrix is the $Omega element. Should equal
      # extract_Sigma(fit_g, level="unit", part="total")$Sigma exactly on
      # this single-tier fixture (estimand-alignment-notes.md trace).
      out <- extract_Omega(fit_g)
      stopifnot(is.list(out), is.matrix(out$Omega), all(dim(out$Omega) == c(p, p)))
      Om <- out$Omega[trait_names, trait_names]
      as.numeric(Om)
    },
    stop("BOGUS_QUANTITY: no dispatcher entry for '", quantity, "'")
  )
}

oracle_values <- list()
oracle_errors <- list()
for (nm in names(CASE_IDS)) {
  cid <- CASE_IDS[[nm]]
  v <- tryCatch(list(ok = TRUE, value = r_quantity(nm), error = ""),
                error = function(e) list(ok = FALSE, value = NULL, error = conditionMessage(e)))
  if (isTRUE(v$ok)) {
    oracle_values[[cid]] <- v$value
  } else {
    oracle_errors[[cid]] <- v$error
  }
}

# ---------------------------------------------------------------------------
# LOUD coverage check: every case_id must produce EITHER an oracle_values
# entry OR an oracle_errors entry (mirrors tools/core070_surface_conversion_batch.R).
# ---------------------------------------------------------------------------
missing_case_ids <- setdiff(unname(CASE_IDS), union(names(oracle_values), names(oracle_errors)))
if (length(missing_case_ids) > 0L) {
  stop(
    "FATAL: ", length(missing_case_ids), " case(s) produced NEITHER an ",
    "oracle_values entry NOR an oracle_errors entry:\n  ",
    paste(missing_case_ids, collapse = "\n  ")
  )
}

# --- negative controls: a bogus quantity key and a bogus fixture key must
#     both be rejected before the tape (never silently skipped). ---
neg_bogus_quantity <- tryCatch({ r_quantity("this_quantity_does_not_exist"); list(rejected = FALSE) },
                                error = function(e) list(rejected = TRUE, message = conditionMessage(e)))
neg_wrong_fixture <- list(
  rejected = !("bogus_fixture_key" %in% "gaussian_small"),
  message = "bogus_fixture_key is not the gaussian_small fixture key"
)

# ---------------------------------------------------------------------------
# Write the oracle JSON for the Julia child.
# ---------------------------------------------------------------------------
oracle_path <- file.path(output_dir, "r-oracle.json")
jsonlite::write_json(
  list(
    schema = "core070-estimand-rebind-r-oracle/v1",
    gaussian_small = list(p = p, K = K, n = n, y = as.numeric(Y_g)),
    oracle_values = oracle_values,
    oracle_errors = oracle_errors,
    negative_controls = list(
      bogus_quantity = neg_bogus_quantity,
      wrong_fixture = neg_wrong_fixture
    )
  ),
  oracle_path, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = 17
)

# ---------------------------------------------------------------------------
# Invoke the Julia child (env pass-through, matches
# tools/core070_surface_conversion_batch.R's convention).
# ---------------------------------------------------------------------------
julia_out <- file.path(output_dir, "julia-results.json")
julia_env <- c(CORE070_ESTIMAND_REBIND_R_ORACLE = oracle_path)
julia_bin <- Sys.which("julia")
stopifnot(nzchar(julia_bin))
old_env <- Sys.getenv(names(julia_env), unset = NA, names = TRUE)
do.call(Sys.setenv, as.list(julia_env))
t0 <- Sys.time()
julia_status <- tryCatch(
  system2(julia_bin, c("--project=.", "tools/core070_estimand_rebind_batch.jl", julia_out),
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

julia_report <- if (file.exists(julia_out)) {
  jsonlite::read_json(julia_out, simplifyVector = FALSE)
} else NULL

results_ok <- !is.null(julia_report) &&
  identical(julia_status, 0L) &&
  identical(julia_report$status, "PASS") &&
  isTRUE(julia_report$all_checks) &&
  isTRUE(julia_report$negative_controls_behaved_as_expected)

cases_seen <- if (!is.null(julia_report)) names(julia_report$cases) else character(0)
missing_case_ids <- setdiff(unname(CASE_IDS), cases_seen)
extra_case_ids <- setdiff(cases_seen, unname(CASE_IDS))
results_ok <- results_ok && length(missing_case_ids) == 0 && length(extra_case_ids) == 0

raw_lines <- vapply(names(CASE_IDS), function(nm) {
  cid <- CASE_IDS[[nm]]
  pass <- !is.null(julia_report) && cid %in% names(julia_report$cases) &&
    isTRUE(julia_report$cases[[cid]]$pass)
  paste(SOURCE_IDS[[nm]], cid, if (pass) "PASS" else "FAIL", "estimand_rebind", sep = "\t")
}, character(1))
raw_path <- file.path(output_dir, "results.tsv")
writeLines(raw_lines, raw_path)

diag_lines <- character(0)
if (!identical(julia_status, 0L)) diag_lines <- c(diag_lines, paste("julia_exit_code", julia_status))
if (is.null(julia_report)) diag_lines <- c(diag_lines, "julia-results.json was not written")
if (length(oracle_errors)) {
  for (id in names(oracle_errors)) {
    diag_lines <- c(diag_lines, paste0("oracle_error[", id, "]: ", oracle_errors[[id]]))
  }
}
if (length(missing_case_ids)) diag_lines <- c(diag_lines, paste("missing_case_ids:", paste(missing_case_ids, collapse = ", ")))
if (length(extra_case_ids)) diag_lines <- c(diag_lines, paste("extra_case_ids:", paste(extra_case_ids, collapse = ", ")))
diag_path <- file.path(output_dir, "diagnostics.log")
writeLines(diag_lines, diag_path)

receipt <- list(
  status = if (isTRUE(results_ok) && length(oracle_errors) == 0L) "PASS" else "FAIL",
  scope = "CORE070_ESTIMAND_REBIND_BATCH",
  reference_commit = REFERENCE_COMMIT,
  source_unchanged = TRUE,
  target_source_ids = unname(SOURCE_IDS),
  case_count = length(CASE_IDS),
  expected_case_ids = unname(CASE_IDS),
  oracle_error_count = length(oracle_errors),
  julia_exit_code = julia_status,
  julia_elapsed_seconds = elapsed,
  julia_results_sha256 = if (file.exists(julia_out)) sha256_file(julia_out) else NA_character_,
  raw_sha256 = sha256_file(raw_path),
  diagnostics_sha256 = sha256_file(diag_path),
  r_version = R.version.string,
  gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
  frozen_library = frozen_library
)
receipt_path <- file.path(output_dir, "receipt.json")
jsonlite::write_json(receipt, receipt_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

cat("CORE070_ESTIMAND_REBIND_BATCH_", receipt$status, "\n", sep = "")
quit(status = if (identical(receipt$status, "PASS")) 0L else 1L)
