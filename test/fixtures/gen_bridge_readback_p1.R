## Records bridge_readback_p1.toml: a live run of the gllvmTMB P1 Julia bridge (R ->
## JuliaCall -> this checkout's GLLVModels) on the namespace twin data ns_gauss_p1_data.csv.
## NOT run by CI or by any Julia test -- provenance for the tracked TOML. Needs:
##   * gllvmTMB installed at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (0.7.1, "P1") in a
##     lane-local library, path in GLLVM_P1_RLIB (with the oracle build.json next to it, see
##     GLLVM_P1_BUILD_JSON);
##   * the R package JuliaCall, and a Julia environment (GLLVM_BRIDGE_JLENV) that develops this
##     checkout and also holds RCall; start with JULIA_PROJECT set to that environment so
##     JuliaCall and GLLVModels load from one resolved manifest;
##   * JULIA_HOME pointing at the Julia bin directory.
## Run from test/fixtures/:  Rscript gen_bridge_readback_p1.R
##
## What is recorded, for one Gaussian fit of
##   value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)   (p = 6, n = 200)
##   1. the outputs of the R S3 methods coef, fitted, logLik, predict, residuals and summary on
##      the gllvmTMB_julia object returned by gllvmTMB(..., engine = "julia");
##   2. the same quantities computed directly in Julia, in the same JuliaCall session, by the
##      native GLLVModels accessors on a fit made with the call bridge_fit makes for this row
##      (alpha = row means of Y; fit_gaussian_gllvm(Y .- alpha; K = 2), src/bridge.jl);
##   3. gllvm_julia_fit(Y, "gaussian", num.lv = 2) called directly (the export), and the
##      equivalent native R fit gllvmTMB(..., engine = "tmb") on the same data.
## The bridge does not return the Julia fit object, so "the same fit" in (2) is a refit with the
## identical call and inputs (Y read back from the bridge object's own bridge_input$y).
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB))
stopifnot(as.character(packageVersion("gllvmTMB")) == "0.7.1")
build_json <- Sys.getenv("GLLVM_P1_BUILD_JSON", file.path(dirname(rlib), "build.json"))
build <- jsonlite::fromJSON(build_json)
stopifnot(identical(build$reference_commit, "9539352f66f2db2cc26b1c393e67212a359b60c9"))
jlenv <- Sys.getenv("GLLVM_BRIDGE_JLENV")
stopifnot(nzchar(jlenv), identical(normalizePath(Sys.getenv("JULIA_PROJECT")), normalizePath(jlenv)))
julia_home <- Sys.getenv("JULIA_HOME")
stopifnot(nzchar(julia_home))

data_file <- "ns_gauss_p1_data.csv"
data_sha <- unname(tools::sha256sum(data_file))
stopifnot(identical(data_sha, "e714fa10411f6a66150c534506dec87cf1c748fed4c55ae13942828f1fca0326"))
df <- read.csv(data_file)
df$unit <- factor(df$unit, levels = sort(unique(df$unit)))
df$trait <- factor(df$trait, levels = paste0("t", 1:6))
form <- value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE)

## ---- 1. bridge object through the public formula route --------------------------------------
gllvm_julia_setup(jl_path = jlenv, julia_home = julia_home)
J <- function(x) JuliaCall::julia_eval(x)
gllvmodels_src <- J("pathof(GLLVModels)")
fit_j <- gllvmTMB(form, data = df, unit = "unit", trait = "trait", engine = "julia")
stopifnot(inherits(fit_j, "gllvmTMB_julia"), isTRUE(fit_j$converged))
Y <- fit_j$bridge_input$y
stopifnot(identical(dim(Y), c(6L, 200L)))

cf <- coef(fit_j)
ll <- logLik(fit_j)
fit_resp <- fitted(fit_j)                      # type = "response"
fit_link <- fitted(fit_j, type = "link")
pr_link <- predict(fit_j)                      # type = "link", long frame trait/unit/est
pr_resp <- predict(fit_j, type = "response")
rs_resp <- residuals(fit_j)                    # type = "response"
rs_pear <- residuals(fit_j, type = "pearson")
sm <- summary(fit_j)
stopifnot(identical(pr_link$trait, rep(rownames(Y), ncol(Y))),
          identical(as.character(pr_link$unit), rep(colnames(Y), each = nrow(Y))))

## ---- 2. the same quantities directly in Julia (same session, same Y) -------------------------
JuliaCall::julia_assign("Yb", Y)
JuliaCall::julia_command(paste0(
  "let; Y = Matrix{Float64}(Yb); p, n = size(Y);",
  " alpha = vec(GLLVModels.Statistics.mean(Y; dims = 2)); Yc = Y .- alpha;",
  " f = GLLVModels.fit_gaussian_gllvm(Yc; K = 2);",
  " mu = alpha .+ GLLVModels.predict(f, Yc; type = :response);",
  " eta = alpha .+ GLLVModels.predict(f, Yc; type = :link);",
  " k = p + GLLVModels.StatsAPI.dof(f); ll = GLLVModels.StatsAPI.loglikelihood(f);",
  " global JD = Dict{String,Any}(",
  "  \"converged\" => f.converged,",
  "  \"alpha\" => alpha,",
  "  \"loadings\" => Matrix{Float64}(GLLVModels.getLoadings(f; rotate = true)),",
  "  \"loglik\" => ll, \"df\" => k, \"nobs\" => p * n,",
  "  \"aic\" => 2k - 2ll, \"bic\" => k * log(p * n) - 2ll,",
  "  \"fitted_response\" => mu, \"fitted_link\" => eta,",
  "  \"resid_response\" => Y .- mu,",
  "  \"resid_pearson\" => GLLVModels.residuals(f, Yc; type = :pearson),",
  "  \"Sigma\" => Matrix{Float64}(GLLVModels.sigma_y_site(f)),",
  "  \"correlation\" => Matrix{Float64}(GLLVModels.correlation(f)),",
  "  \"communality\" => Vector{Float64}(GLLVModels.communality(f)),",
  "  \"sigma_eps\" => f.pars.σ_eps); nothing; end"))
JD <- JuliaCall::julia_eval("JD")
stopifnot(isTRUE(JD$converged))

## ---- 3. the gllvm_julia_fit export directly, and the native R engine on the same data -------
fit_x <- gllvm_julia_fit(Y, family = "gaussian", num.lv = 2L)
stopifnot(inherits(fit_x, "gllvmTMB_julia"), isTRUE(fit_x$converged))
fit_t <- gllvmTMB(form, data = df, unit = "unit", trait = "trait")
stopifnot(fit_t$opt$convergence == 0L, isTRUE(fit_t$sd_report$pdHess))
tmb_beta <- unname(fit_t$opt$par[names(fit_t$opt$par) == "b_fix"])
tmb_Sigma <- unname(as.matrix(extract_Sigma(fit_t, level = "unit")$Sigma))

num <- function(x) unname(as.numeric(x))
mat <- function(x) { x <- as.matrix(x); list(nrow = nrow(x), ncol = ncol(x), colmajor = num(x)) }
out <- list(
  schema = "bridge-readback-p1/v1",
  gllvmtmb_commit = build$reference_commit,
  gllvmtmb_version = as.character(packageVersion("gllvmTMB")),
  gllvmtmb_library = find.package("gllvmTMB"),
  oracle_build_installed_tree_sha256 = build$installed_tree_sha256,
  r_version = R.version.string,
  juliacall_version = as.character(packageVersion("JuliaCall")),
  julia_version = J("string(VERSION)"),
  gllvmodels_source = gllvmodels_src,
  julia_threads = J("Threads.nthreads()"),
  blas_threads = J("GLLVModels.LinearAlgebra.BLAS.get_num_threads()"),
  formula = paste(deparse(form), collapse = " "),
  data_file = data_file, data_sha256 = data_sha,
  trait_names = rownames(Y), n_unit = ncol(Y),
  y = mat(Y),
  bridge = list(
    model = fit_j$model, converged = fit_j$converged,
    coef_alpha = num(cf$alpha), coef_loadings = mat(cf$loadings),
    logLik = num(ll), logLik_df = num(attr(ll, "df")), logLik_nobs = num(attr(ll, "nobs")),
    fitted_response = mat(fit_resp), fitted_link = mat(fit_link),
    predict_link_est = num(pr_link$est), predict_response_est = num(pr_resp$est),
    residuals_response = mat(rs_resp), residuals_pearson = mat(rs_pear),
    summary_logLik = num(sm$header$logLik), summary_AIC = num(sm$header$AIC),
    summary_BIC = num(sm$header$BIC), summary_df = num(sm$header$df),
    summary_nobs = num(sm$header$nobs),
    summary_coef_alpha = num(sm$coefficients$alpha),
    summary_coef_loadings = mat(sm$coefficients$loadings),
    summary_Sigma = mat(sm$covariance$Sigma),
    summary_correlation = mat(sm$covariance$correlation),
    summary_communality = num(sm$covariance$communality)),
  julia_direct = list(
    converged = JD$converged, alpha = num(JD$alpha), loadings = mat(JD$loadings),
    loglik = num(JD$loglik), df = num(JD$df), nobs = num(JD$nobs),
    aic = num(JD$aic), bic = num(JD$bic),
    fitted_response = mat(JD$fitted_response), fitted_link = mat(JD$fitted_link),
    resid_response = mat(JD$resid_response), resid_pearson = mat(JD$resid_pearson),
    Sigma = mat(JD$Sigma), correlation = mat(JD$correlation),
    communality = num(JD$communality), sigma_eps = num(JD$sigma_eps)),
  gllvm_julia_fit = list(
    converged = fit_x$converged, model = fit_x$model,
    loglik = num(fit_x$loglik), df = num(fit_x$df), alpha = num(fit_x$alpha),
    loadings = mat(fit_x$loadings), Sigma = mat(fit_x$Sigma), sigma_eps = num(fit_x$sigma_eps)),
  tmb = list(
    convergence = fit_t$opt$convergence, pd_hessian = isTRUE(fit_t$sd_report$pdHess),
    logLik = num(logLik(fit_t)), df = num(attr(logLik(fit_t), "df")),
    beta = tmb_beta, Sigma_latent = mat(tmb_Sigma),
    sigma_eps = num(exp(fit_t$opt$par[["log_sigma_eps"]]))))
## TOML (the P1 twin CI job runs each test in the package root environment, which has the TOML
## stdlib but no JSON reader). Doubles are written with 17 significant digits (exact round trip);
## a matrix is a sub-table {nrow, ncol, colmajor}.
toml_value <- function(x) {
  one <- function(v) {
    if (is.logical(v)) return(if (isTRUE(v)) "true" else "false")
    if (is.character(v)) return(paste0("\"", gsub("\"", "\\\\\"", gsub("\\\\", "\\\\\\\\", v)), "\""))
    if (is.integer(v)) return(as.character(v))
    stopifnot(is.double(v), is.finite(v))
    s <- sprintf("%.17g", v)
    if (!grepl("[.eE]", s)) s <- paste0(s, ".0")
    s
  }
  if (length(x) == 1L && is.null(names(x)) && !is.list(x)) return(one(x))
  paste0("[", paste(vapply(x, one, ""), collapse = ", "), "]")
}
is_mat <- function(x) is.list(x) && identical(names(x), c("nrow", "ncol", "colmajor"))
toml_table <- function(tab, prefix) {
  scal <- names(tab)[!vapply(tab, is.list, TRUE)]
  subs <- names(tab)[vapply(tab, is.list, TRUE)]
  lines <- character()
  if (nzchar(prefix)) lines <- c(lines, "", paste0("[", prefix, "]"))
  for (k in scal) lines <- c(lines, paste0(k, " = ", toml_value(tab[[k]])))
  for (k in subs) {
    key <- if (nzchar(prefix)) paste0(prefix, ".", k) else k
    sub <- tab[[k]]
    if (is_mat(sub)) sub <- list(nrow = as.integer(sub$nrow), ncol = as.integer(sub$ncol), colmajor = sub$colmajor)
    lines <- c(lines, toml_table(sub, key))
  }
  lines
}
writeLines(c("# Live P1 bridge readback, recorded by gen_bridge_readback_p1.R. Do not hand-edit; regenerate.",
             toml_table(out, "")), "bridge_readback_p1.toml")
cat("BRIDGE_READBACK_P1_RECORDED\n")
