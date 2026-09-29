#!/bin/bash
# Gate: gllvmTMB select_lv guard tests + the two existing select_lv test files pass on the R lane branch.
set -e
cd /Users/z3437171/local-scratch/lanes/gllvmTMB-auto-d-20260926
Rscript -e '
suppressMessages(devtools::load_all(quiet = TRUE))
fs <- c("tests/testthat/test-latent-auto.R", "tests/testthat/test-brms-sugar.R", "tests/testthat/test-latent-unique-rename.R", "tests/testthat/test-select-lv-guard.R", "tests/testthat/test-select-lv-anova.R", "tests/testthat/test-example-model-selection-rank.R")
res <- do.call(rbind, lapply(fs, function(f) as.data.frame(testthat::test_file(f, reporter = "silent"))))
bad <- sum(res$failed) + sum(res$error)
cat("passed", sum(res$nb) - sum(res$failed), "failed", sum(res$failed), "errors", sum(res$error), "\n")
if (bad == 0) cat("R-TESTS-PASS\n") else quit(status = 1)
' 2>/dev/null | tail -2
