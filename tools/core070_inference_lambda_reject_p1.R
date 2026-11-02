# One live R observation on the P1 oracle for CI-ROUTE-006 and CI-ROUTE-007 (W2-1, group G4).
#
# The route probe (tools/core070_inference_routes_p1.R) stubs .confint_lambda, so for
# confint(parm = "Lambda", method = "fisher-z" | "bogus") its record is a dispatch label, not
# the refusal an R user sees. This script fits a small confirmatory Gaussian model on the
# installed P1 gllvmTMB and makes the real calls:
#
#   method = "fisher-z", "bogus"   the refusals (class and full message are recorded)
#   method = "wald"                the valid-method control on the same fit: it must return an
#                                  interval (a confirmatory fit, because P1 refuses per-entry
#                                  Wald on Lambda for an unconstrained fit)
#
# Fixture: the data of tools/core070_inference_remainder_batch.R (set.seed(42), p = 5, K = 2,
# n = 80), with one loading pinned to 0 so that Lambda is identified.
#
#   Rscript --vanilla tools/core070_inference_lambda_reject_p1.R <frozen-library> <destination>
#
# <destination> must not exist. GLLVM_PARITY_PIN must be P1 (the default here).

args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
frozen_library <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
stopifnot(!dir.exists(output_dir))

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
stopifnot(normalizePath(find.package("gllvmTMB")) == normalizePath(file.path(frozen_library, "gllvmTMB")))

root <- normalizePath(".")
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P1")))
if (!identical(parity_pin, "P1")) stop("this observation is defined at P1 only, got '", parity_pin, "'")
expected_reference <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
source(file.path(root, "tools/core070_source_pin.R"))
source_pin <- core070_source_pin(root, frozen_library, parity_pin, expected_reference)
dir.create(output_dir, recursive = TRUE)

set.seed(42)
p <- 5L; K <- 2L; n <- 80L
Lambda_true <- matrix(c(0.8, 0.0, 0.5, 0.6, 0.3, -0.4, -0.2, 0.5, 0.1, 0.3), nrow = p, ncol = K, byrow = TRUE)
eta_g <- matrix(rnorm(K * n), nrow = K, ncol = n)
Y_g <- Lambda_true %*% eta_g + 0.7 * matrix(rnorm(p * n), nrow = p, ncol = n)
Y_g <- Y_g - rowMeans(Y_g)
trait_names <- paste0("t", seq_len(p))
df_long <- data.frame(site = factor(rep(seq_len(n), each = p)),
                      trait = factor(rep(trait_names, times = n), levels = trait_names),
                      value = as.vector(Y_g))
M <- matrix(NA_real_, p, K); M[1, 2] <- 0
fit <- suppressWarnings(gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
  data = df_long, unit = "site", trait = "trait", family = gaussian(),
  lambda_constraint = list(unit = M),
  control = gllvmTMBcontrol(n_init = 1L, se = TRUE)))
stopifnot("gllvmTMB_multi" %in% class(fit))

call_confint <- function(method) {
  tryCatch({
    x <- suppressWarnings(confint(fit, parm = "Lambda", level = 0.95, method = method))
    list(method = method, raised = FALSE, error_class = "", message = "", n_rows = nrow(x),
         finite = all(is.finite(c(x$lower, x$upper)), na.rm = TRUE), columns = names(x))
  }, error = function(e) list(method = method, raised = TRUE, error_class = paste(class(e), collapse = ";"),
                              message = conditionMessage(e), n_rows = 0L, finite = FALSE, columns = list()))
}
calls <- lapply(c("fisher-z", "bogus", "wald"), call_confint)
names(calls) <- vapply(calls, `[[`, "", "method")

oracle_path <- file.path(output_dir, "r-oracle.json")
jsonlite::write_json(list(
  schema = "core070-inference-lambda-reject-r-oracle/v1",
  r_call = "confint(fit, parm = \"Lambda\", level = 0.95, method = <m>)",
  fixture = list(p = p, n = n, K = K, seed = 42L, lambda_constraint = "Lambda[1, 2] pinned to 0 (unit tier)"),
  calls = calls),
  oracle_path, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = 17)

ok <- isTRUE(calls[["fisher-z"]]$raised) && isTRUE(calls[["bogus"]]$raised) &&
  !isTRUE(calls[["wald"]]$raised) && isTRUE(calls[["wald"]]$finite)
receipt <- list(
  status = if (ok) "PASS" else "FAIL",
  scope = "CORE070_INFERENCE_LAMBDA_REJECT_P1",
  reference_commit = expected_reference,
  oracle_sha256 = sha256_file(oracle_path),
  r_version = R.version.string,
  gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
  frozen_library = frozen_library,
  source_pin = source_pin)
jsonlite::write_json(receipt, file.path(output_dir, "receipt.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("CORE070_INFERENCE_LAMBDA_REJECT_P1_", receipt$status, "\n", sep = "")
quit(status = if (ok) 0L else 1L)
