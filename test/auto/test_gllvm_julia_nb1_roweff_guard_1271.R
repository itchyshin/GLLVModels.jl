# Behavioural check for .check_nb1_row_eff (refs #1271).
# Usage: Rscript --vanilla test_gllvm_julia_nb1_roweff_guard_1271.R /path/to/gllvmtmb_julia.R
args <- commandArgs(trailingOnly = TRUE)
scaffold <- if (length(args) >= 1L) args[[1L]] else {
  file.path(dirname(sys.frame(1)$ofile), "..", "..", "r", "gllvmtmb_julia.R")
}
if (!file.exists(scaffold)) {
  stop("scaffold not found: ", scaffold, call. = FALSE)
}
src <- readLines(scaffold)
cut <- grep("^library\\(JuliaConnectoR\\)$", src, perl = FALSE)
if (length(cut) != 1L) {
  stop("expected exactly one library(JuliaConnectoR) line, got ", length(cut), call. = FALSE)
}
eval(parse(text = src[seq_len(cut - 1L)]))
if (!is.function(.check_nb1_row_eff)) {
  stop(".check_nb1_row_eff was not defined from the scaffold prefix", call. = FALSE)
}
expect_stop <- function(family, row.eff, must_include) {
  e <- tryCatch(.check_nb1_row_eff(family, row.eff), error = identity)
  if (!inherits(e, "error")) {
    stop(sprintf("expected an error for family=%s row.eff=%s", family, row.eff), call. = FALSE)
  }
  msg <- conditionMessage(e)
  for (s in must_include) {
    if (!grepl(s, msg, fixed = TRUE)) {
      stop(sprintf("message missing '%s': %s", s, msg), call. = FALSE)
    }
  }
  invisible(TRUE)
}
expect_stop("negative.binomial1", "fixed",
            c("negative.binomial1", "fixed", "none", "negative.binomial"))
expect_stop("negative.binomial1", "random",
            c("negative.binomial1", "random", "none", "negative.binomial"))
.check_nb1_row_eff("negative.binomial1", "none")
.check_nb1_row_eff("negative.binomial", "fixed")
.check_nb1_row_eff("negative.binomial", "random")
.check_nb1_row_eff("poisson", "fixed")
cat("OK\n")
