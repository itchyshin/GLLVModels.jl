#!/usr/bin/env Rscript
## R side of the P1 grouping-level paired Gaussian receipts.
##
## Usage: Rscript tools/core070_grouping_p1_r.R <oracle_build_dir> <out.json>
##
## Loads gllvmTMB from the read-only P1 oracle library (<oracle_build_dir>/library),
## refuses to run unless <oracle_build_dir>/build.json names the P1 commit, and
## refuses any fixture CSV whose sha256 differs from fixtures/manifest.json.
## For each grouping level it
##   (a) measures name parity by CALLING gllvmTMB(): the paired fit passes the
##       keyword (a successful fit is the positive result), and a negative control
##       calls gllvmTMB() with a misspelt keyword and records the error;
##   (b) fits the diagonal Gaussian model by ML and writes logLik, fixed effects,
##       the level's trait covariance (extract_Sigma, part = "total") and sigma_eps.
## The gllvmTMB clone is never touched; nothing here writes outside <out.json>.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
build_dir <- normalizePath(args[[1]], mustWork = TRUE)
out_path <- args[[2]]
P1 <- "9539352f66f2db2cc26b1c393e67212a359b60c9"

build <- jsonlite::fromJSON(file.path(build_dir, "build.json"))
if (!identical(build$reference_commit, P1) || !identical(as.integer(build$exit_code), 0L)) {
  stop("oracle build.json does not record a clean P1 build: ", build$reference_commit)
}
lib <- file.path(build_dir, "library")
library("gllvmTMB", lib.loc = lib, character.only = TRUE)
loaded_path <- normalizePath(find.package("gllvmTMB"))
if (!identical(loaded_path, normalizePath(file.path(lib, "gllvmTMB")))) {
  stop("gllvmTMB loaded from ", loaded_path, ", not the oracle library")
}

fixture_dir <- "docs/dev-log/core070/true-parity-latest/receipts/grouping/fixtures"
manifest <- jsonlite::fromJSON(file.path(fixture_dir, "manifest.json"))
sha256 <- function(p) {
  unname(sub(" .*", "", system2("shasum", c("-a", "256", shQuote(p)), stdout = TRUE)))
}

levels <- list(
  unit = list(column = "unit", formula = value ~ 0 + trait + indep(0 + trait | unit),
              args = list(unit = "unit")),
  unit_obs = list(column = "unit_obs", formula = value ~ 0 + trait + indep(0 + trait | unit_obs),
                  args = list(unit = "unit", unit_obs = "unit_obs")),
  cluster = list(column = "cluster_id", formula = value ~ 0 + trait + indep(0 + trait | cluster_id),
                 args = list(unit = "unit", cluster = "cluster_id")),
  cluster2 = list(column = "cluster2_id", formula = value ~ 0 + trait + indep(0 + trait | cluster2_id),
                  args = list(unit = "unit", cluster2 = "cluster2_id"))
)

base_call <- function(spec, data) {
  c(list(formula = spec$formula, data = data, family = gaussian(), trait = "trait", REML = FALSE,
         control = gllvmTMBcontrol(n_init = 1L, se = FALSE)), spec$args)
}

results <- list()
for (lvl in names(levels)) {
  spec <- levels[[lvl]]
  csv <- file.path(fixture_dir, paste0(lvl, ".csv"))
  got <- sha256(csv)
  if (!identical(got, manifest$files[[paste0(lvl, ".csv")]])) stop(csv, " sha256 mismatch: ", got)
  data <- utils::read.csv(csv, stringsAsFactors = FALSE)
  data$trait <- factor(data$trait, levels = c("trait_1", "trait_2"))
  for (col in c("unit", "unit_obs", "cluster_id", "cluster2_id")) data[[col]] <- factor(data[[col]])

  ## (a) negative control: a misspelt keyword must be rejected by the call itself.
  bogus <- paste0(lvl, "_zz_not_a_keyword")
  neg_args <- base_call(spec, data)
  neg_args[[bogus]] <- spec$column
  neg <- tryCatch({ do.call(gllvmTMB, neg_args); NULL }, error = function(e) conditionMessage(e))

  ## (b) the paired fit, passing the keyword under test.
  t0 <- Sys.time()
  fit <- tryCatch(do.call(gllvmTMB, base_call(spec, data)), error = function(e) e)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(fit, "error")) {
    results[[lvl]] <- list(keyword = lvl, keyword_accepted = FALSE, fit_error = conditionMessage(fit),
                           negative_control = list(keyword = bogus, rejected = !is.null(neg), message = neg))
    next
  }
  ## Membership control: move observation 1 (both trait rows) into the group of
  ## the first later observation (row order) whose group differs, and refit; a
  ## keyword that were accepted but ignored would leave logLik unchanged. For
  ## unit_obs that group lies in the same unit, so nesting is preserved.
  moved <- data
  g <- as.character(data[[spec$column]])
  target <- g[which(g != g[data$obs_row == 1][1])[1]]
  moved[[spec$column]][moved$obs_row == 1] <- target
  moved_fit <- tryCatch(do.call(gllvmTMB, base_call(spec, moved)), error = function(e) e)
  membership <- if (inherits(moved_fit, "error")) list(error = conditionMessage(moved_fit)) else
    list(moved_obs_row = 1L, to_group = target, logLik = as.numeric(stats::logLik(moved_fit)))
  S <- extract_Sigma(fit, level = lvl, part = "total")$Sigma
  opt <- fit$opt
  results[[lvl]] <- list(
    keyword = lvl, keyword_accepted = TRUE,
    call = paste(deparse(spec$formula), paste(names(spec$args), unlist(spec$args), sep = " = ", collapse = ", "),
                 sep = "; "),
    negative_control = list(keyword = bogus, rejected = !is.null(neg), message = neg),
    membership_control = membership,
    fixture = csv, fixture_sha256 = got,
    n_rows = nrow(data), n_groups = nlevels(data[[spec$column]]),
    fit_seconds = secs,
    convergence = opt$convergence, optimizer_message = opt$message,
    max_abs_gradient = max(abs(fit$tmb_obj$gr(opt$par))),
    logLik = as.numeric(stats::logLik(fit)),
    beta = unname(as.numeric(fit$tmb_obj$env$parList(opt$par)$b_fix)),
    Sigma_diag = unname(diag(S)),
    Sigma_offdiag = unname(S[1, 2]),
    sigma_eps = as.numeric(fit$report$sigma_eps)
  )
}

out <- list(engine = "R gllvmTMB", pin = "P1", reference_commit = P1,
            gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
            loaded_path = loaded_path, oracle_build_json_sha256 = sha256(file.path(build_dir, "build.json")),
            oracle_installed_tree_sha256 = build$installed_tree_sha256,
            R_version = R.version.string, levels = results)
jsonlite::write_json(out, out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")
cat("wrote", out_path, "\n")
