#!/usr/bin/env Rscript
# gllvm-parity-tag: P1
#
# R side of the behavioural twins for two C1 rows (itchyshin/GLLVModels.jl#684 item 2):
#   model-comparison/print.anova.gllvmTMB_multi   (R/aghq-report.R)
#   latent-scores/extract_latent_scores.default   (R/extract-latent-scores.R)
#   model-comparison/update.gllvmTMB_multi        (R/methods-gllvmTMB.R)
# Records what gllvmTMB at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1) actually
# produces: the printed table of anova() on three nested Gaussian fits, the condition
# raised by extract_latent_scores() on four objects it has no method for, and what update()
# does on a temporal fit (replay, data override, an unnamed override) and on an ordinary fit
# that keeps no call. Nothing is typed by
# hand: the Julia side and the receipt writer (tools/true_parity_julia_receipts.jl) read these
# raw files and derive every label from them.
#
# Usage (from the repository root; run once, the outputs are tracked):
#   GLLVM_P1_RLIB=<lane-local library holding gllvmTMB 0.7.1 built from the pin> \
#   GLLVMTMB_CLONE=<path of a gllvmTMB clone that has the pin> \
#     Rscript tools/core070_c1_behaviour_p1.R
# Writes test/fixtures/c1_behaviour_p1/{r_c1_behaviour.toml, r_anova_print.txt}.
#
# Provenance check (the convention of the other P1 twin generators). The installed functions
# are the ones that run, so before recording anything this script reads the SAME functions
# from `git show 9539352f6:R/<file>` of the clone (read only, never checked out), parses them
# with keep.source = FALSE, and requires deparse() of the installed function to equal deparse()
# of the pinned source for every function named in `PINNED_FUNCTIONS`. The result, with the
# sha256 of each source file read, is written into the fixture, so a library built from any
# other commit makes this script stop instead of recording the wrong engine.
#
# The anova data is the simulation of test/fixtures/generate_gllvmtmb_anova_fixture.R (same
# seed, same recipe). The script records the three log-likelihoods; the Julia side requires
# them to equal test/fixtures/gllvmtmb_anova_fixture.toml, which shows the data is the same.

rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
clone <- Sys.getenv("GLLVMTMB_CLONE", path.expand("~/Dropbox/Github Local/gllvmTMB"))
P1 <- "9539352f66f2db2cc26b1c393e67212a359b60c9"

suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

out_dir <- "test/fixtures/c1_behaviour_p1"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- provenance: installed function == pinned source ---------------------------------------
PINNED_FUNCTIONS <- list(
  list(file = "R/aghq-report.R", names = c("anova.gllvmTMB_multi", "print.anova.gllvmTMB_multi",
                                          ".gllvmTMB_anova_global_check", ".gllvmTMB_anova_classify_step")),
  list(file = "R/chibar.R", names = "chibar2_pvalue"),
  list(file = "R/extract-latent-scores.R", names = c("extract_latent_scores", "extract_latent_scores.default")),
  list(file = "R/methods-gllvmTMB.R", names = "update.gllvmTMB_multi")
)
sha256_text <- function(lines) {
  tmp <- tempfile(); on.exit(unlink(tmp))
  writeLines(lines, tmp, useBytes = TRUE)
  unname(sub(" .*", "", system2("shasum", c("-a", "256", shQuote(tmp)), stdout = TRUE)))
}
plain <- function(f) { attr(f, "srcref") <- NULL; deparse(f, width.cutoff = 500L) }
provenance <- list()
for (spec in PINNED_FUNCTIONS) {
  src <- suppressWarnings(system2("git", c("-C", shQuote(clone), "show", shQuote(paste0(P1, ":", spec$file))),
                                  stdout = TRUE, stderr = FALSE))
  if (!length(src) || !is.null(attr(src, "status"))) stop("cannot read ", spec$file, " at ", P1, " from ", clone)
  exprs <- parse(text = src, keep.source = FALSE)
  for (nm in spec$names) {
    hit <- Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) && identical(as.character(e[[2]]), nm), as.list(exprs))
    if (length(hit) != 1L) stop(nm, ": expected one definition in ", spec$file, " at the pin, found ", length(hit))
    pinned <- eval(hit[[1]][[3]], baseenv())
    installed <- get(nm, envir = asNamespace("gllvmTMB"))
    same <- identical(plain(pinned), plain(installed))
    if (!same) stop("installed ", nm, " differs from ", spec$file, " at ", P1, "; this library is not the pin")
    provenance[[length(provenance) + 1L]] <- list(file = spec$file, name = nm, deparse_identical = same,
                                                   source_sha256 = sha256_text(src), source_lines = length(src))
  }
}

# --- the anova print -----------------------------------------------------------------------
set.seed(20260927L)
n_site <- 25L; p_trait <- 4L; d_max <- 3L
beta <- c(-0.4, 0.3, 0.6, -0.1)
Lambda <- matrix(c( 0.9, -0.5,  0.3,
                   -0.6,  0.7,  0.2,
                    0.4,  0.4, -0.6,
                    0.2, -0.3,  0.5), nrow = p_trait, ncol = d_max, byrow = TRUE)
u <- matrix(rnorm(n_site * d_max), nrow = n_site, ncol = d_max)
site  <- factor(rep(paste0("s", seq_len(n_site)), each = p_trait), levels = paste0("s", seq_len(n_site)))
trait <- factor(rep(paste0("t", seq_len(p_trait)), n_site), levels = paste0("t", seq_len(p_trait)))
eta <- numeric(n_site * p_trait)
for (i in seq_len(n_site)) for (t in seq_len(p_trait)) {
  eta[(i - 1L) * p_trait + t] <- beta[t] + sum(Lambda[t, ] * u[i, ])
}
y <- eta + rnorm(length(eta), sd = 0.25)
dat <- data.frame(site = site, trait = trait, y = y)
ctrl <- gllvmTMBcontrol(n_init = 1, init_jitter = 0, se = FALSE, aghq_ridge = Inf)
fit_at_d <- function(d) {
  fml <- as.formula(sprintf("y ~ 0 + trait + latent(0 + trait | site, d = %d, unique = FALSE)", d))
  gllvmTMB(fml, data = dat, family = gaussian(), control = ctrl, unit = "site", silent = TRUE, estimator = "ml")
}
fits <- lapply(seq_len(d_max), fit_at_d)
loglik <- vapply(fits, function(f) as.numeric(stats::logLik(f)), numeric(1))
tab <- anova(fits[[1]], fits[[2]], fits[[3]], test = "chibar")
stopifnot(inherits(tab, "anova.gllvmTMB_multi"))
old_width <- options(width = 200L)   # keep the 9-column table on one block; recorded in the fixture
printed <- capture.output(print(tab))
options(old_width)
writeLines(printed, file.path(out_dir, "r_anova_print.txt"), useBytes = TRUE)

# --- extract_latent_scores on objects with no method ---------------------------------------
objects <- list(
  list(label = "character", x = "a string"),
  list(label = "integer",   x = 1:3),
  list(label = "list",      x = list(a = 1)),
  list(label = "NULL",      x = NULL)
)
observe_refusal <- function(x) {
  warns <- character()
  res <- withCallingHandlers(
    tryCatch(list(raised = FALSE, value = extract_latent_scores(x, level = "unit")),
             error = function(e) list(raised = TRUE, cond = e)),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  if (res$raised) {
    list(raised = TRUE, condition_classes = class(res$cond),
         message = cli::ansi_strip(conditionMessage(res$cond)), returned_class = "", warnings = warns)
  } else {
    list(raised = FALSE, condition_classes = character(), message = "",
         returned_class = paste(class(res$value), collapse = "/"), warnings = warns)
  }
}
refusals <- lapply(objects, function(o) c(list(object_class = class(o$x)[1]), observe_refusal(o$x)))

# --- update() on a temporal fit, and on an ordinary fit that keeps no call -------------------
set.seed(20261002L)
udat <- expand.grid(series = paste0("s", 1:4), occasion = 1:5, trait = paste0("t", 1:3),
                    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
udat$value <- rnorm(nrow(udat))
uctrl <- gllvmTMBcontrol(se = FALSE)
tfit <- suppressWarnings(gllvmTMB(value ~ 0 + trait + temporal_indep(0 + trait | series, time = occasion),
                                  data = udat, unit = "series", family = gaussian(), control = uctrl, silent = TRUE))
stopifnot(isTRUE(tfit$temporal$active), tfit$opt$convergence == 0L)
uchanged <- udat; uchanged$value <- uchanged$value + 0.01 * seq_len(nrow(uchanged))
replay <- suppressWarnings(update(tfit))
override <- suppressWarnings(update(tfit, data = uchanged))
observe_error <- function(expr) {
  res <- tryCatch(list(raised = FALSE, value = expr), error = function(e) list(raised = TRUE, cond = e))
  if (res$raised) list(raised = TRUE, condition_classes = class(res$cond), message = cli::ansi_strip(conditionMessage(res$cond)),
                       returned_class = "")
  else list(raised = FALSE, condition_classes = character(), message = "", returned_class = paste(class(res$value), collapse = "/"))
}
unnamed <- observe_error(suppressWarnings(update(tfit, uchanged)))
# An ordinary (non-temporal) fit: one latent factor over units = series x occasion, no saved call.
udat$site <- factor(paste(udat$series, udat$occasion, sep = "_"))
ofit <- suppressWarnings(gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = 1), data = udat, unit = "site",
                                  family = gaussian(), control = uctrl, silent = TRUE))
stopifnot(!isTRUE(ofit$temporal$active), is.null(ofit$call))
nocall <- observe_error(suppressWarnings(update(ofit)))
upd <- list(
  data = list(series = udat$series, occasion = udat$occasion, trait = udat$trait, value = udat$value),
  changed_value = uchanged$value,
  original = list(loglik = as.numeric(logLik(tfit)), response = tfit$data$value, temporal_active = isTRUE(tfit$temporal$active)),
  replay = list(class = class(replay), loglik = as.numeric(logLik(replay)), response = replay$data$value,
                temporal_active = isTRUE(replay$temporal$active)),
  override = list(class = class(override), loglik = as.numeric(logLik(override)), response = override$data$value,
                  temporal_active = isTRUE(override$temporal$active)),
  unnamed = unnamed, nocall = nocall)

# --- write the TOML ------------------------------------------------------------------------
tq <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE); s <- gsub("\"", "\\\"", s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE); s <- gsub("\t", "\\t", s, fixed = TRUE)
  paste0("\"", s, "\"")
}
tarr <- function(v) paste0("[", paste(vapply(v, tq, ""), collapse = ", "), "]")
con <- file(file.path(out_dir, "r_c1_behaviour.toml"), "w", encoding = "UTF-8")
w <- function(...) writeLines(sprintf(...), con)
w("# Raw R observations for the C1 behavioural twins (generated by tools/core070_c1_behaviour_p1.R).")
w("# Do not hand-edit: regenerate from the script. The labels compared with Julia are derived from")
w("# these records and from r_anova_print.txt by test/fixtures/c1_behaviour_p1/helpers.jl.")
w("gllvmtmb_commit = %s", tq(P1))
w("gllvmtmb_version = %s", tq(as.character(packageVersion("gllvmTMB"))))
w("r_version = %s", tq(R.version.string))
w("seed = %d", 20260927L)
w("print_width_option = %d", 200L)
w("anova_print_file = \"r_anova_print.txt\"")
w("anova_print_sha256 = %s", tq(sha256_text(printed)))
w("loglik = [%s]", paste(sprintf("%.17g", loglik), collapse = ", "))
for (p in provenance) {
  w("")
  w("[[provenance]]")
  w("file = %s", tq(p$file)); w("name = %s", tq(p$name))
  w("deparse_identical = %s", if (p$deparse_identical) "true" else "false")
  w("source_sha256 = %s", tq(p$source_sha256)); w("source_lines = %d", p$source_lines)
}
for (r in refusals) {
  w("")
  w("[[refusal]]")
  w("object_class = %s", tq(r$object_class))
  w("raised = %s", if (r$raised) "true" else "false")
  w("condition_classes = %s", tarr(r$condition_classes))
  w("message = %s", tq(r$message))
  w("returned_class = %s", tq(r$returned_class))
  w("warnings = %s", tarr(r$warnings))
}
tnums <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ", "), "]")
w("")
w("[update]")
w("seed = %d", 20261002L)
w("formula = %s", tq("value ~ 0 + trait + temporal_indep(0 + trait | series, time = occasion)"))
w("series = %s", tarr(as.character(upd$data$series)))
w("occasion = %s", tnums(upd$data$occasion))
w("trait = %s", tarr(as.character(upd$data$trait)))
w("value = %s", tnums(upd$data$value))
w("changed_value = %s", tnums(upd$changed_value))
for (nm in c("original", "replay", "override")) {
  x <- upd[[nm]]
  w("")
  w("[update.%s]", nm)
  if (!is.null(x$class)) w("class = %s", tarr(x$class))
  w("loglik = %.17g", x$loglik)
  w("response = %s", tnums(x$response))
  w("temporal_active = %s", if (x$temporal_active) "true" else "false")
}
for (nm in c("unnamed", "nocall")) {
  x <- upd[[nm]]
  w("")
  w("[update.%s]", nm)
  w("raised = %s", if (x$raised) "true" else "false")
  w("condition_classes = %s", tarr(x$condition_classes))
  w("message = %s", tq(x$message))
  w("returned_class = %s", tq(x$returned_class))
}
close(con)
cat("wrote", out_dir, "\n")
