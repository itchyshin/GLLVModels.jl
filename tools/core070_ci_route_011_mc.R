# CI-ROUTE-011 Monte-Carlo replicates, R side (ruling N4, vault D-319).
#
# The rule (receipts/inference/ci-route-011-mc/rule.json) was committed before any run.
# This script builds the twolevel_small fixture exactly as tools/core070_surface_conversion_batch.R
# does, then for each seed s runs set.seed(s); confint(fit_tl, parm = "icc", method = "bootstrap",
# level = 0.95, nsim = 200) and records the endpoints, the point estimate, the structural check and
# the wall time. It writes Y_tl and `individual` too, so the Julia side fits the same data.
#
#   Rscript --vanilla tools/core070_ci_route_011_mc.R <frozen-library> <destination> <seed> [<seed> ...]
# <destination> must not exist.

args <- commandArgs(TRUE)
stopifnot(length(args) >= 3L)
frozen_library <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
seeds <- as.integer(args[-(1:2)])
stopifnot(!dir.exists(output_dir), all(!is.na(seeds)))
dir.create(output_dir, recursive = TRUE)

.libPaths(c(frozen_library, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(normalizePath(find.package("gllvmTMB")) == normalizePath(file.path(frozen_library, "gllvmTMB")))
root <- normalizePath(".")
source(file.path(root, "tools/core070_source_pin.R"))
p1 <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
source_pin <- core070_source_pin(root, frozen_library, "P1", p1)

# ---- twolevel_small, verbatim from tools/core070_surface_conversion_batch.R ----
set.seed(1)
p_tl <- 4L; n_ind <- 30L; reps <- 4L
n_tl <- n_ind * reps
Lambda_B_true <- matrix(c(0.7, 0.4, -0.3, 0.5), ncol = 1L)
Lambda_W_true <- matrix(c(0.3, 0.5, 0.4, -0.2), ncol = 1L)
individual <- rep(seq_len(n_ind), each = reps)
b_true <- matrix(rnorm(n_ind), nrow = 1L)
w_true <- matrix(rnorm(n_tl), nrow = 1L)
Y_tl <- Lambda_B_true %*% b_true[, individual, drop = FALSE] +
        Lambda_W_true %*% w_true +
        0.3 * matrix(rnorm(p_tl * n_tl), nrow = p_tl)
Y_tl <- Y_tl - rowMeans(Y_tl)
tl_trait_names <- paste0("t", seq_len(p_tl))
df_tl <- data.frame(
  site         = factor(rep(individual, each = p_tl)),
  site_species = factor(rep(seq_len(n_tl), each = p_tl)),
  trait        = factor(rep(tl_trait_names, times = n_tl), levels = tl_trait_names),
  value        = as.vector(Y_tl)
)
fit_tl <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | site, d = 1) +
    latent(0 + trait | site_species, d = 1),
  data = df_tl, unit = "site", unit_obs = "site_species", trait = "trait",
  family = gaussian(),
  control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
)
stopifnot("gllvmTMB_multi" %in% class(fit_tl))
point <- as.numeric(extract_repeatability(fit_tl, method = "wald")$R)

reps_out <- list()
for (s in seeds) {
  t0 <- proc.time()[["elapsed"]]
  set.seed(s)
  v <- tryCatch({
    ci <- confint(fit_tl, parm = "icc", method = "bootstrap", level = 0.95, nsim = 200)
    lower <- as.numeric(ci[, 1L]); upper <- as.numeric(ci[, 2L])
    list(ok = TRUE, lower = lower, upper = upper,
         finite = all(is.finite(lower)) && all(is.finite(upper)),
         ordered = all(lower <= upper), brackets_point = all(lower <= point & point <= upper))
  }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  v$seed <- s
  v$elapsed_seconds <- proc.time()[["elapsed"]] - t0
  reps_out[[length(reps_out) + 1L]] <- v
  message(sprintf("seed %d: %.1f s ok=%s", s, v$elapsed_seconds, v$ok))
}

write_json(list(schema = "core070-ci-route-011-mc-r/v1", reference_commit = p1, source_pin = source_pin,
                gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
                r_version = R.version.string, point = point, seeds = seeds, replicates = reps_out,
                fixture = list(p = p_tl, n_individual = n_ind, reps_per_individual = reps,
                               individual = individual, y = as.numeric(Y_tl))),
           file.path(output_dir, "r-mc.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("CI_ROUTE_011_MC_R_DONE\n")
