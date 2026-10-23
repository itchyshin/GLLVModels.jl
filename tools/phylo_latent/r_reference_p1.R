#!/usr/bin/env Rscript
# R side of the phylo_latent twin receipts at gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). No R package source is edited:
# the package is loaded from a private library built with R CMD INSTALL from
# a detached worktree at P1.
#
# Two modes.
#   generate FIXTURE DATA_JSON
#       FIXTURE is a14 (shared by both A14 cases) or a15. Draw the fixture (tree, dense vcv, response) and write it. The
#       JSON is the literal fixture both engines read; R re-reads it before
#       fitting, so neither side fits in-memory values the other never saw.
#   fit PRIVATE_LIBRARY CASE DATA_JSON JULIA_JSON OUTPUT_JSON
#       Fit the P1 R model on DATA_JSON, record R's own optimum, and evaluate
#       R's objective at the Julia optimum read from JULIA_JSON (the cross
#       objective). JULIA_JSON carries theta in R's coordinate order.
#
# CASE is one of struct_phy_tree_rr, struct_phy_dense_rr (A14) or
# cov_phylo_latent_rsz (A15).
args <- commandArgs(trailingOnly = TRUE)
source_pin <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
cases <- c("struct_phy_tree_rr", "struct_phy_dense_rr", "cov_phylo_latent_rsz")
sha <- function(path) unname(tools::sha256sum(path))[[1L]]
yhash <- function(x) digest::digest(writeBin(as.double(c(x)), raw(), size = 8L,
                                          endian = "little"),
                                    algo = "sha256", serialize = FALSE)
rows <- function(x) unname(lapply(seq_len(nrow(x)), function(i) unname(as.numeric(x[i, ]))))
as_matrix <- function(x) {
  if (is.matrix(x)) return(unname(x))
  do.call(rbind, lapply(x, as.numeric))
}

a14_newick <- "(((s1:2,s2:2):1,(s3:1,s4:1):2):1,((s5:1.5,s6:1.5):1,(s7:1,s8:1):1.5):1.5);"

generate <- function(case, output) {
  if (file.exists(output)) stop("use a fresh output path; fixtures are immutable")
  if (case == "a14") {
    # One shared A14 draw: both A14 cases fit the same response.
    seed <- 20260927L
    newick <- a14_newick
    tree <- ape::read.tree(text = newick)
    traits <- c("a", "b", "c"); reps <- 2L; K <- 1L
    beta <- c(.4, -.2, .3); loading <- matrix(c(.9, .5, -.6), 3L, 1L); sd_eps <- .25
  } else {
    seed <- 20260928L
    set.seed(seed)
    tree0 <- ape::rcoal(100L)
    tree0$tip.label <- paste0("sp", seq_len(100L))
    newick <- ape::write.tree(tree0, digits = 17L)
    tree <- ape::read.tree(text = newick)
    traits <- sprintf("t%02d", seq_len(20L)); reps <- 5L; K <- 2L
    beta <- round(stats::rnorm(20L, sd = .5), 3)
    loading <- matrix(round(stats::rnorm(20L * K, sd = .6), 3), 20L, K)
    loading[1L, 2L] <- 0
    sd_eps <- .5
  }
  set.seed(seed + 1L)
  tips <- tree$tip.label
  C <- ape::vcv(tree, corr = TRUE)[tips, tips]
  p <- length(tips); n_traits <- length(traits)
  L <- t(chol(C))
  g <- sapply(seq_len(K), function(k) as.numeric(L %*% stats::rnorm(p)))
  g <- matrix(g, p, K)
  obs_species <- rep(seq_len(p), each = reps)
  m <- length(obs_species)
  mu <- matrix(beta, n_traits, m) + loading %*% t(g[obs_species, , drop = FALSE])
  Y <- mu + matrix(stats::rnorm(n_traits * m, sd = sd_eps), n_traits, m)
  fixture <- list(schema_version = "phylo-latent-p1-fixture-1", fixture = case,
    source_pin = source_pin, seed = seed, newick = newick, tip_labels = tips,
    trait_names = traits, rank = K, replicates = reps,
    observation_species = tips[obs_species],
    vcv_corr = rows(C), Y_traits_by_observations = rows(Y),
    data_sha256 = yhash(Y), data_hash_encoding = "Float64 little-endian column-major",
    generating_truth = list(beta = beta, loading = rows(loading), sd_eps = sd_eps),
    claim = "functional fixture for paired evaluation; not recovery evidence")
  jsonlite::write_json(fixture, output, auto_unbox = TRUE, pretty = TRUE,
                       digits = 17L, null = "null", na = "null")
  back <- jsonlite::fromJSON(output, simplifyVector = FALSE)
  stopifnot(identical(as_matrix(back$Y_traits_by_observations), unname(Y)),
            identical(as_matrix(back$vcv_corr), unname(C)))
  cat(sha(output), "\n")
}

fit_case <- function(lib, case, data_path, julia_path, output) {
  if (file.exists(output)) stop("use a fresh output path; receipts are immutable")
  if ("gllvmTMB" %in% loadedNamespaces()) stop("start a fresh R process")
  lib <- normalizePath(lib, mustWork = TRUE)
  library("gllvmTMB", lib.loc = lib, character.only = TRUE)
  package_path <- normalizePath(find.package("gllvmTMB"), mustWork = TRUE)
  dll_path <- normalizePath(getLoadedDLLs()[["gllvmTMB"]][["path"]], mustWork = TRUE)
  stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1",
            startsWith(package_path, paste0(lib, "/")),
            startsWith(dll_path, paste0(lib, "/")))
  fx <- jsonlite::fromJSON(data_path, simplifyVector = FALSE)
  stopifnot(identical(fx$fixture, if (case == "cov_phylo_latent_rsz") "a15" else "a14"))
  Y <- as_matrix(fx$Y_traits_by_observations)
  stopifnot(identical(yhash(Y), fx$data_sha256))
  tips <- unlist(fx$tip_labels); traits <- unlist(fx$trait_names)
  obs_species <- unlist(fx$observation_species)
  K <- as.integer(fx$rank)
  tree <- ape::read.tree(text = fx$newick)
  C <- as_matrix(fx$vcv_corr); dimnames(C) <- list(tips, tips)
  n_traits <- nrow(Y); m <- ncol(Y)
  df <- data.frame(
    observation = rep(seq_len(m), each = n_traits),
    species = factor(rep(obs_species, each = n_traits), levels = tips),
    trait = factor(rep(traits, times = m), levels = traits),
    value = as.numeric(Y))
  route <- if (case == "struct_phy_dense_rr") "vcv" else "tree"
  formula <- if (route == "vcv") {
    value ~ 0 + trait + phylo_latent(species, d = K, vcv = C)
  } else {
    value ~ 0 + trait + phylo_latent(species, d = K, tree = tree)
  }
  control <- gllvmTMBcontrol(n_init = 1L, optimizer = "nlminb",
    optArgs = list(control = list(iter.max = 1000L, eval.max = 2000L, rel.tol = 1e-12)))
  warnings <- list()
  started <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(gllvmTMB(formula, data = df, trait = "trait",
      unit = "species", cluster = "species", family = gaussian(),
      REML = FALSE, control = control),
    warning = function(w) {
      warnings[[length(warnings) + 1L]] <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    })
  elapsed <- proc.time()[["elapsed"]] - started
  obj <- fit$tmb_obj
  par_names <- names(fit$opt$par)
  expected <- c(rep("b_fix", n_traits), "log_sigma_eps",
                rep("theta_rr_phy", n_traits * K - K * (K - 1L) / 2L))
  stopifnot(identical(par_names, expected), identical(names(obj$par), expected))
  own <- unname(fit$opt$par)
  own_nll <- obj$fn(own)
  rep_own <- obj$report(obj$env$last.par)  # full vector at the own optimum
  own_gr <- as.numeric(obj$gr(own))
  Ainv <- fit$tmb_data$Ainv_phy_rr
  trip <- Matrix::summary(methods::as(Ainv, "generalMatrix"))
  cond_h <- tryCatch({
    fit_se <- gllvmTMB::standard_errors(fit)
    sd <- fit_se$sd_report
    list(pd_hessian = isTRUE(sd$pdHess),
         condition_number = kappa(sd$cov.fixed, exact = TRUE))
  }, error = function(e) list(error = conditionMessage(e)))
  jl <- jsonlite::fromJSON(julia_path, simplifyVector = FALSE)
  stopifnot(identical(jl$case, case), identical(jl$data_sha256, fx$data_sha256))
  jtheta <- as.numeric(unlist(jl$theta_r_order))
  stopifnot(length(jtheta) == length(own))
  cross_nll <- obj$fn(jtheta)
  cross_gr <- as.numeric(obj$gr(jtheta))
  stopifnot(is.finite(own_nll), is.finite(cross_nll))
  receipt <- list(schema_version = "phylo-latent-p1-r-receipt-1", case = case,
    status = "recorded", source_pin = source_pin,
    package_version = as.character(packageVersion("gllvmTMB")),
    package_path = package_path, dll_path = dll_path, dll_sha256 = sha(dll_path),
    data_file_sha256 = sha(data_path), data_sha256 = fx$data_sha256,
    julia_receipt_sha256 = sha(julia_path), route = route,
    formula = paste(deparse(formula), collapse = " "),
    call_arguments = "trait = 'trait', unit = 'species', cluster = 'species', family = gaussian(), REML = FALSE, nlminb rel.tol = 1e-12",
    parameter_names = par_names, theta_hat = own, objective = own_nll,
    optimizer_objective = fit$opt$objective, loglik = as.numeric(stats::logLik(fit)),
    gradient = own_gr, gradient_max_abs = max(abs(own_gr)),
    convergence = fit$opt$convergence, message = fit$opt$message,
    iterations = fit$opt$iterations, evaluations = fit$opt$evaluations,
    Sigma_phy = rows(rep_own$Sigma_phy), Lambda_phy = rows(rep_own$Lambda_phy),
    log_det_A_phy_rr = fit$tmb_data$log_det_A_phy_rr,
    n_aug_phy = fit$tmb_data$n_aug_phy,
    species_aug_id_zero_based = as.integer(fit$tmb_data$species_aug_id),
    Ainv_triplets = list(i = trip$i, j = trip$j, x = trip$x),
    Ainv_node_labels = rownames(Ainv),
    hessian = cond_h,
    cross = list(julia_theta_r_order = jtheta, r_objective_at_julia_theta = cross_nll,
                 r_gradient_at_julia_theta = cross_gr,
                 r_gradient_max_abs_at_julia_theta = max(abs(cross_gr))),
    elapsed_seconds = elapsed, warnings = warnings,
    session_info = paste(capture.output(sessionInfo()), collapse = "\n"),
    qualified = FALSE)
  jsonlite::write_json(receipt, output, auto_unbox = TRUE, pretty = TRUE,
                       digits = 17L, null = "null", na = "null")
  cat(sprintf("R %s nll=%.15g conv=%d cross=%.15g elapsed=%.2fs\n", case, own_nll,
              fit$opt$convergence, cross_nll, elapsed))
  cat(sha(output), "\n")
}

if (length(args) >= 1L && args[[1L]] == "generate") {
  if (length(args) != 3L || !(args[[2L]] %in% c("a14", "a15"))) stop("usage: generate a14|a15 DATA_JSON")
  generate(args[[2L]], args[[3L]])
} else if (length(args) >= 1L && args[[1L]] == "fit") {
  if (length(args) != 6L || !(args[[3L]] %in% cases))
    stop("usage: fit PRIVATE_LIBRARY CASE DATA_JSON JULIA_JSON OUTPUT_JSON")
  fit_case(args[[2L]], args[[3L]], args[[4L]], args[[5L]], args[[6L]])
} else {
  stop("usage: r_reference_p1.R generate|fit ...")
}
