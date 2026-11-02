# DRAFT, DIAGNOSTIC ONLY. Not a receipt and not evidence for any row status.
#
# POST-COEF-EMPTY ruling packet (docs/dev-log/plans/2026-11-02-c2-post-coef-empty-ruling-packet.md).
# Fits real gllvmTMB models through the public gllvmTMB() entry point at the P1 pin and records the
# length and names of coef(). This replaces the synthetic list(X_fix_names = character(0)) object the
# committed batch uses (tools/core070_postfit_policy_batch.R:175-184).
#
#   Rscript --vanilla tools/core070_coef_empty_pair_draft.R <pinned-R-source-root> <out.json>
#
# It needs an installed gllvmTMB built from that same source tree. It refuses to run when
# R/vcov-coef.R in <pinned-R-source-root> is not the blob at the pin.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
src <- normalizePath(args[[1]], mustWork = TRUE)
out <- args[[2]]
PIN_COMMIT <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
PIN_BLOB_VCOV_COEF <- "a707d537e209fe2963dd09a688fe51a611ca3de2"  # git rev-parse PIN:R/vcov-coef.R
blob <- trimws(system2("git", c("hash-object", file.path(src, "R/vcov-coef.R")), stdout = TRUE))
if (!identical(blob, PIN_BLOB_VCOV_COEF)) stop("R/vcov-coef.R is not the blob at the pin: ", blob)

suppressMessages(library(gllvmTMB))
set.seed(42)
p <- 5L; K <- 1L; n <- 80L
lam <- c(0.8, 0.5, 0.3, -0.2, 0.1)
Y <- outer(lam, rnorm(n)) + 0.7 * matrix(rnorm(p * n), p, n)
Y <- Y - rowMeans(Y)
tn <- paste0("t", seq_len(p))
x_site <- rnorm(n)
df <- data.frame(site = factor(rep(seq_len(n), each = p)),
                 trait = factor(rep(tn, times = n), levels = tn),
                 x = rep(x_site, each = p),
                 value = as.vector(Y))

fit_one <- function(label, julia_formula, fixed) {
  f <- stats::as.formula(paste0("value ~ ", fixed, " + latent(0 + trait | site, d = ", K, ", unique = FALSE)"))
  fit <- gllvmTMB(f, data = df, unit = "site", trait = "trait", family = gaussian(),
                  control = gllvmTMBcontrol(n_init = 1L, se = TRUE))
  cc <- coef(fit)
  list(label = label, r_formula = fixed, julia_formula = julia_formula,
       class = class(fit)[[1L]],
       x_fix_names_is_null = is.null(fit$X_fix_names),
       coef_length = length(cc),
       coef_has_names_attribute = !is.null(names(cc)),
       coef_names = unname(names(cc)),
       coef_identical_to_named_empty = identical(cc, stats::setNames(numeric(0), character(0))))
}
res <- list(
  fit_one("no fixed effects", "y ~ 0", "0"),
  fit_one("trait intercepts", "y ~ 1", "0 + trait"),
  fit_one("trait intercepts plus one shared covariate", "y ~ 1 + x", "0 + trait + x")
)
jsonlite::write_json(list(
  schema = "core070-coef-empty-pair-draft-r-diagnostic/v1",
  binding = FALSE,
  note = "DIAGNOSTIC, NON-BINDING. Not a receipt. Public gllvmTMB() fits at the pin, coef() length and names only.",
  reference_commit = PIN_COMMIT,
  vcov_coef_blob = blob,
  gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
  p = p, n = n, K = K,
  pairs = res), out, auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("WROTE", out, "\n")
