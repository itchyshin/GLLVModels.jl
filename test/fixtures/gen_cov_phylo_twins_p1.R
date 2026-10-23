## Regenerates cov_phylo_twins_p1.toml. NOT run by CI or by any Julia test -- provenance for how the
## fixture was produced, against gllvmTMB pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9
## (version 0.7.1, "P1"). Needs gllvmTMB installed at that exact commit in a lane-local library; set
## GLLVM_P1_RLIB to it. Run from test/fixtures/:  Rscript gen_cov_phylo_twins_p1.R
##
## Fit-level twins for three core070 covariance rows whose P1 batch case is an R-only formula-grammar
## check (parse_multi_formula(desugar_brms_sugar(f))$covstructs) with no fit number:
##   covariance/COV-PHYLO-DEP            phylo_dep(0 + trait | species)
##                                       -> phylo_rr(d = n_traits, .dep = TRUE), the phylo_latent(d = T)
##                                          engine path (R/brms-sugar.R rewrite table at P1)
##   covariance/COV-PHYLO-A-ALIAS        phylo_latent(species, A = A) -> phylo_rr(vcv = A)
##   covariance/COV-PHYLO-FOLDED-UNIQUE  phylo_latent(species, unique = TRUE)
##                                       -> phylo_rr(d = 1) + phylo_rr(.phylo_unique, .auto_unique):
##                                          the folded per-trait phylogenetic unique companion, which
##                                          populates the phylo_diag block of src/gllvmTMB.cpp
##                                          (g_phy_diag[, t] ~ N(0, A), same Ainv_phy_rr as phylo_rr,
##                                          scaled by exp(log_sd_phy_diag[t])).
## Here each is one real Gaussian fit on one shared fixture; the Julia side is
## test/test_cov_phylo_twins_p1.jl (fit_phylo_latent_gllvm with d = n_traits, A = A, unique = true).
##
## Fixture: a coalescent tree of 120 tips (ape::rcoal, seed 20261005), 4 traits, 2 replicate
## observations per species, drawn from
##   y[t, o] = b[t] + Lambda[t] g[species(o)] + s[t] u_t[species(o)] + eps,
##   g, u_t ~ N(0, A) independently, eps ~ N(0, 0.4^2),
## with A the correlation-form (unit-height) tip covariance. The tree, A (ape::vcv(corr = TRUE),
## tip-label order) and the response are rounded to 17 significant digits before R fits them, and the
## tree and response are written to the TOML at that precision, so both engines fit the same doubles.
## A itself is not stored (120^2 values); its sum, Frobenius norm and first row are, and the Julia
## test rebuilds A from the Newick string and checks it against them.
##
## Every R fit must converge (nlminb code 0) with a positive-definite Hessian (asserted). The A =
## alias fit and the same model spelled vcv = A must give the identical objective (asserted): the
## alias is a rewrite, not a different model. R's parameter vector at its optimum is recorded by
## name so the Julia test can evaluate Julia's objective at R's parameters (a fixed-parameter
## likelihood check, independent of either optimiser).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")

n_sp <- 120L; n_tr <- 4L; reps <- 2L
set.seed(20261005L)
tree0 <- ape::rcoal(n_sp)
tree0$tip.label <- sprintf("sp%02d", seq_len(n_sp))
newick <- ape::write.tree(tree0, digits = 17L)
tree <- ape::read.tree(text = newick)
tips <- tree$tip.label
A <- ape::vcv(tree, corr = TRUE)[tips, tips]
traits <- sprintf("t%d", seq_len(n_tr))
beta <- c(0.4, -0.2, 0.3, 0.1)
lam <- c(0.8, 0.5, -0.6, 0.4)
s_u <- c(0.7, 0.6, 0.8, 0.6)
sd_eps <- 0.4
L <- t(chol(A))
g <- as.numeric(L %*% rnorm(n_sp))
U <- sapply(seq_len(n_tr), function(t) as.numeric(L %*% rnorm(n_sp)))  # n_sp x n_tr
obs_sp <- rep(seq_len(n_sp), each = reps)
m <- length(obs_sp)
Y <- matrix(beta, n_tr, m) + outer(lam, g[obs_sp]) + t(U[obs_sp, ] %*% diag(s_u)) +
  matrix(rnorm(n_tr * m, sd = sd_eps), n_tr, m)
fmt <- function(x) sprintf("%.17g", x)
Y <- matrix(as.numeric(fmt(Y)), n_tr, m)                            # fit the written doubles
A_w <- matrix(as.numeric(fmt(A)), n_sp, n_sp, dimnames = list(tips, tips))
stopifnot(isSymmetric(A_w))

df <- data.frame(observation = rep(seq_len(m), each = n_tr),
                 species = factor(rep(tips[obs_sp], each = n_tr), levels = tips),
                 trait = factor(rep(traits, times = m), levels = traits),
                 value = as.numeric(Y))
control <- gllvmTMBcontrol(n_init = 1L, optimizer = "nlminb",
  optArgs = list(control = list(iter.max = 1000L, eval.max = 2000L, rel.tol = 1e-10)))
fitp <- function(fml) {
  f <- suppressMessages(suppressWarnings(gllvmTMB(fml, data = df, trait = "trait",
    unit = "species", cluster = "species", family = gaussian(), REML = FALSE, control = control)))
  stopifnot(f$opt$convergence == 0L, isTRUE(f$sd_report$pdHess))
  f
}
f_dep   <- fitp(value ~ 0 + trait + phylo_dep(0 + trait | species, tree = tree))
f_alias <- fitp(value ~ 0 + trait + phylo_latent(species, A = A_w))
f_vcv   <- fitp(value ~ 0 + trait + phylo_latent(species, vcv = A_w))
f_uniq  <- fitp(value ~ 0 + trait + phylo_latent(species, unique = TRUE, tree = tree))
stopifnot(identical(f_alias$opt$objective, f_vcv$opt$objective))

info <- function(f) {
  rep <- f$tmb_obj$report(f$tmb_obj$env$last.par.best)
  gr <- as.numeric(f$tmb_obj$gr(f$opt$par))
  list(loglik = as.numeric(logLik(f)), objective = f$tmb_obj$fn(f$opt$par),
       par = unname(f$opt$par), par_names = names(f$opt$par), grad = max(abs(gr)),
       Lambda = rep$Lambda_phy, Sigma_rr = rep$Sigma_phy,
       sd_diag = if (!is.null(rep$sd_phy_diag)) as.numeric(rep$sd_phy_diag) else NULL,
       sigma_eps = exp(f$opt$par[["log_sigma_eps"]]))
}
I_dep <- info(f_dep); I_alias <- info(f_alias); I_uniq <- info(f_uniq)
stopifnot(identical(I_dep$par_names, c(rep("b_fix", n_tr), "log_sigma_eps", rep("theta_rr_phy", 10L))),
          identical(I_alias$par_names, c(rep("b_fix", n_tr), "log_sigma_eps", rep("theta_rr_phy", n_tr))),
          identical(I_uniq$par_names, c(rep("b_fix", n_tr), "log_sigma_eps", rep("theta_rr_phy", n_tr),
                                        rep("log_sd_phy_diag", n_tr))),
          length(I_uniq$sd_diag) == n_tr)
## Total phylogenetic trait covariance of the folded fit: Lambda Lambda^T + diag(sd_phy_diag^2).
Sigma_uniq_total <- tcrossprod(I_uniq$Lambda) + diag(I_uniq$sd_diag^2)

vec <- function(x) paste0("[", paste(vapply(as.numeric(x), fmt, ""), collapse = ", "), "]")
svec <- function(x) paste0("[", paste0("\"", x, "\"", collapse = ", "), "]")
con <- file("cov_phylo_twins_p1.toml", "w")
w <- function(...) writeLines(sprintf(...), con)
w("# Twin fixture for core070 covariance/COV-PHYLO-DEP, COV-PHYLO-A-ALIAS and COV-PHYLO-FOLDED-UNIQUE, gllvmTMB P1.")
w("# Generated once by gen_cov_phylo_twins_p1.R. Do not hand-edit; regenerate from the script.")
w("gllvmtmb_commit = \"9539352f66f2db2cc26b1c393e67212a359b60c9\"")
w("gllvmtmb_version = \"%s\"", as.character(packageVersion("gllvmTMB")))
w("r_version = \"%s\"", R.version.string)
w("seed = 20261005")
w("n_species = %d", n_sp); w("n_traits = %d", n_tr); w("replicates = %d", reps)
w("newick = \"%s\"", newick)
w("tip_labels = %s", svec(tips))
w("trait_names = %s", svec(traits))
w("observation_species = %s", svec(tips[obs_sp]))
w("# A = ape::vcv(tree, corr = TRUE) in tip_labels order (the values R fitted, after the 17-digit round trip).")
w("# Not stored in full (n_species^2 values): the Julia test rebuilds A from newick and checks it against these.")
w("A_sum = %s", fmt(sum(A_w)))
w("A_frobenius = %s", fmt(sqrt(sum(A_w^2))))
w("A_first_row = %s", vec(A_w[1, ]))
w("# response, traits x observations, row-major (row t = trait t)")
w("Y = %s", vec(t(Y)))
w("call_arguments = \"trait = 'trait', unit = 'species', cluster = 'species', family = gaussian(), REML = FALSE, nlminb rel.tol = 1e-10, n_init = 1\"")
w("converged = true"); w("pd_hessian = true")
blk <- function(name, formula, I, extra = NULL) {
  w(""); w("[%s]", name)
  w("formula = \"%s\"", formula)
  w("loglik = %s", fmt(I$loglik))
  w("objective = %s", fmt(I$objective))
  w("max_abs_gradient = %s", fmt(I$grad))
  w("par_names = %s", svec(I$par_names))
  w("par = %s", vec(I$par))
  w("beta = %s", vec(I$par[I$par_names == "b_fix"]))
  w("sigma_eps = %s", fmt(I$sigma_eps))
  w("# Lambda_phy Lambda_phy^T (report Sigma_phy), traits x traits, row-major")
  w("Sigma_rr = %s", vec(t(I$Sigma_rr)))
  if (!is.null(extra)) extra()
}
blk("dep", "value ~ 0 + trait + phylo_dep(0 + trait | species, tree = tree)", I_dep)
blk("alias", "value ~ 0 + trait + phylo_latent(species, A = A)", I_alias, function() {
  w("# the same model spelled vcv = A gives the identical nlminb objective (asserted)")
  w("objective_vcv_spelling = %s", fmt(f_vcv$opt$objective))
})
blk("unique", "value ~ 0 + trait + phylo_latent(species, unique = TRUE, tree = tree)", I_uniq, function() {
  w("sd_phy_diag = %s", vec(I_uniq$sd_diag))
  w("# Lambda Lambda^T + diag(sd_phy_diag^2), traits x traits, row-major")
  w("Sigma_total = %s", vec(t(Sigma_uniq_total)))
})
close(con)
cat("dep", I_dep$loglik, I_dep$grad, "alias", I_alias$loglik, I_alias$grad,
    "unique", I_uniq$loglik, I_uniq$grad, "sd_diag", I_uniq$sd_diag, "\n")
