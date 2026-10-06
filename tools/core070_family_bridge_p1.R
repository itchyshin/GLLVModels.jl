# P1 twin of the public R bridge batch for the family rows (tools/core070_bridge_models.R is
# the P0 batch), plus the FAMILY-BETA-ALIAS adapter case. Run at gllvmTMB pin P1.
#
#   Rscript --vanilla tools/core070_family_bridge_p1.R <runs-dir> <destination>
#   Rscript --vanilla tools/core070_family_bridge_p1.R --family11-boundary <destination>
#
# The second form recreates the registered seed-58 truncated-NB2 response without fitting,
# probes both public R bridge routes, and writes r-public-bridge.json. It requires the same
# P1 library / JuliaCall environment plus GLLVM_PARITY_GLVMODELS_COMMIT identifying the
# exact GLLVModels run commit. The destination must not exist.
#
# <runs-dir> holds the runparity output directories of the SAME clean commit
# (runparity-poisson/, runparity-beta/, runparity-nb2/). <destination> must not exist.
# Environment:
#   GLLVM_P1_RLIB       R library holding gllvmTMB built from the P1 commit (with its
#                       CORE070_SOURCE_PIN.toml marker, tools/core070_build_oracle.py)
#   GLLVM_BRIDGE_JLENV  Julia project that develops this checkout and holds RCall
#                       (test/parity works); JULIA_PROJECT must name the same directory
#   JULIA_HOME          Julia bin directory
#   GLLVM_PARITY_PIN    must be P1 (or unset)
#
# What is measured, per family case (02 Poisson, 07 Beta, 05 NB2):
#   * data: the Y the same-run native cell fitted (poisson-fixture.toml, beta-fixture.toml
#     from the run directory; NB2 from test/parity/fixtures/nb2_smoke_data.toml), with its
#     sha256 checked. NB2 uses the stored smoke draw, not the frozen bridge contract's
#     original data (decision 2026-09-28: on the original data both engines put two
#     traits at the Poisson boundary); the receipt names the deviation.
#   * native: a fresh default GLLVModels fit in the same Julia session, with fit health
#     (tools/core070_bridge_models.jl, unchanged from P0).
#   * the two public bridge routes: gllvm_julia_fit(Y, family, num.lv = K) and
#     gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
#     reversed long data, engine = "julia").
#   * the R-vs-Julia number: each route's logLik against the same-run P1 R fit of the
#     native cell (<family>-health.toml r_loglik, its R convergence code reverified).
#     This replaces the P0 batch's retained P0 R logLik, which is not on any host.
# BETA-ALIAS: the alias descriptor structure(list(family = "beta", link = "logit"),
#   class = "family") on the Beta data, through (a) native R gllvmTMB (TMB engine),
#   against the same call with gllvmTMB::Beta(); (b) both public bridge routes, against
#   the native Julia fit; and (c) R-vs-Julia: the alias R fit's logLik against the
#   native Julia logLik.
# Nothing here decides a verdict; tools/core070_family_p1_receipts.py does, from the
# results JSON this script writes.
args <- commandArgs(trailingOnly = TRUE)

# Narrow, no-fit probe for FAMILY-11's public R bridge boundary. This is
# intentionally separate from the frozen three-case numeric bridge batch.
if (length(args) == 2L && identical(args[[1]], "--family11-boundary")) {
  output_dir <- args[[2]]
  if (file.exists(output_dir)) stop("destination must not exist: ", output_dir)
  root <- normalizePath(".")
  P1 <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
  parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P1")))
  if (!identical(parity_pin, "P1")) stop("this boundary probe runs at P1 only")
  rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
  if (!nzchar(rlib)) stop("GLLVM_P1_RLIB is required")
  rlib <- normalizePath(rlib, mustWork = TRUE)
  .libPaths(c(rlib, .libPaths()))
  suppressPackageStartupMessages(library(gllvmTMB, lib.loc = rlib))
  stopifnot(normalizePath(find.package("gllvmTMB")) == normalizePath(file.path(rlib, "gllvmTMB")),
            as.character(packageVersion("gllvmTMB")) == "0.7.1")
  source(file.path(root, "tools/core070_source_pin.R"))
  source_pin <- core070_source_pin(root, rlib, "P1", P1)
  jlenv <- Sys.getenv("GLLVM_BRIDGE_JLENV")
  stopifnot(nzchar(jlenv), identical(normalizePath(Sys.getenv("JULIA_PROJECT")), normalizePath(jlenv)))
  julia_home <- Sys.getenv("JULIA_HOME")
  stopifnot(nzchar(julia_home))
  engine_commit <- Sys.getenv("GLLVM_PARITY_GLVMODELS_COMMIT", "")
  if (!grepl("^4b78fa012", engine_commit)) stop("GLLVM_PARITY_GLVMODELS_COMMIT must identify 4b78fa012")

  # Recreate the registered seed-58 response in Julia without invoking a
  # likelihood or optimiser. These statements mirror the fixture recipe in
  # test/parity/test_truncated_nbinom2_parity.jl.
  gllvm_julia_setup(jl_path = jlenv, julia_home = julia_home)
  JuliaCall::julia_command(paste0(
    "using Random, Distributions; ",
    "function core070_family11_fixture(); Random.seed!(58); p,K,n=5,1,120; ",
    "β=log.([4.0,5.0,3.5,4.5,4.0]); Λ=0.2 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3][:,1:K]; ",
    "Z=randn(K,n); η=β .+ Λ*Z; Y=Matrix{Int}(undef,p,n); ",
    "for t in 1:p,s in 1:n; μ=exp(clamp(η[t,s],-3.0,3.5)); ",
    "while true; v=rand(Distributions.NegativeBinomial(4.0,4.0/(4.0+μ))); ",
    "if v>=1; Y[t,s]=v; break; end; end; end; Y; end"))
  Y <- JuliaCall::julia_eval("core070_family11_fixture()")
  Y <- matrix(as.integer(Y), nrow = 5L, ncol = 120L)
  sha256_values <- function(values) {
    path <- tempfile(); on.exit(unlink(path), add = TRUE)
    writeBin(as.double(values), path, size = 8L, endian = "little")
    unname(tools::sha256sum(path))
  }
  sha256_file <- function(path) unname(tools::sha256sum(path))
  data_sha256 <- sha256_values(as.vector(Y))
  expected_sha <- "ecbcf9f501c7e618131f2c3f1f0d213bb0e92364a72c0519095c52ef30930948"
  if (!identical(data_sha256, expected_sha)) stop("seed-58 fixture hash mismatch: ", data_sha256)
  dimnames(Y) <- list(sprintf("trait%02d", seq_len(5L)), sprintf("site%03d", seq_len(120L)))
  d <- expand.grid(trait = rownames(Y), site = colnames(Y))
  d$value <- as.vector(Y)
  d$trait <- factor(d$trait, levels = rownames(Y)); d$site <- factor(d$site, levels = colnames(Y))
  fam <- gllvmTMB::truncated_nbinom2()
  capture_boundary <- function(expr, call) {
    warnings <- character()
    value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
    }), error = function(e) e)
    list(call = call, refused = inherits(value, "error"),
         error_class = if (inherits(value, "error")) class(value) else character(),
         message = if (inherits(value, "error")) conditionMessage(value) else "",
         warnings = warnings)
  }
  calls <- list(
    capture_boundary(gllvmTMB::gllvm_julia_fit(Y, family = fam, num.lv = 1L),
                     "gllvm_julia_fit(Y, family = gllvmTMB::truncated_nbinom2(), num.lv = 1)"),
    capture_boundary(gllvmTMB::gllvmTMB(
      value ~ 0 + trait + latent(0 + trait | site, d = 1L, unique = FALSE),
      data = d[nrow(d):1L, ], unit = "site", trait = "trait", family = fam, engine = "julia"),
      "gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE), data = reversed long, family = truncated_nbinom2(), engine = 'julia')")
  )
  names(calls) <- c("matrix", "formula")
  dir.create(output_dir, recursive = TRUE)
  result <- list(schema = "core070-family11-r-boundary/v1",
    case_id = "CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE", pin = "P1", reference_commit = P1,
    gllvmtmb_version = as.character(packageVersion("gllvmTMB")), source_pin = source_pin,
    namespace_sha256 = unname(tools::sha256sum(file.path(rlib, "gllvmTMB", "NAMESPACE"))),
    fixture = list(recipe = "test/parity/test_truncated_nbinom2_parity.jl seed=58", p = 5L, n = 120L,
                   K = 1L, data_sha256 = data_sha256),
    routes = calls, capture = list(r_version = R.version.string, julia_version = JuliaCall::julia_eval("string(VERSION)"),
      glvmodels_path = JuliaCall::julia_eval("pathof(GLLVModels)"),
      glvmodels_commit = engine_commit))
  out <- file.path(output_dir, "r-public-bridge.json")
  jsonlite::write_json(result, out, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
  cat("CORE070_FAMILY11_BOUNDARY_WRITTEN", sha256_file(out), "\n")
  quit(save = "no", status = 0L)
}

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
runs_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
if (file.exists(output_dir)) stop("destination must not exist: ", output_dir)

P1 <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P1")))
if (!identical(parity_pin, "P1")) stop("this batch runs at P1 only")
rlib <- Sys.getenv("GLLVM_P1_RLIB", "")
if (!nzchar(rlib)) stop("GLLVM_P1_RLIB is required")
rlib <- normalizePath(rlib, mustWork = TRUE)
.libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(gllvmTMB, lib.loc = rlib))
stopifnot(normalizePath(find.package("gllvmTMB")) == normalizePath(file.path(rlib, "gllvmTMB")),
          as.character(packageVersion("gllvmTMB")) == "0.7.1")
root <- normalizePath(".")
source(file.path(root, "tools/core070_source_pin.R"))
source_pin <- core070_source_pin(root, rlib, "P1", P1)
jlenv <- Sys.getenv("GLLVM_BRIDGE_JLENV")
stopifnot(nzchar(jlenv), identical(normalizePath(Sys.getenv("JULIA_PROJECT")), normalizePath(jlenv)))
julia_home <- Sys.getenv("JULIA_HOME")
stopifnot(nzchar(julia_home))
sha256_file <- function(path) unname(tools::sha256sum(path))

contract_path <- "docs/dev-log/core070/public-bridge-required-cases.json"
contract <- jsonlite::read_json(contract_path, simplifyVector = FALSE)
stopifnot(identical(contract$reference_commit, "b4d5fee64def88bc768dda1f1f77c29b295edd86"))
case_of <- function(fam) Filter(function(x) identical(x$family, fam), contract$cases)[[1]]

dir.create(output_dir, recursive = TRUE)
gllvm_julia_setup(jl_path = jlenv, julia_home = julia_home)
JuliaCall::julia_command(sprintf('include("%s")', file.path(root, "tools/core070_bridge_models.jl")))
JuliaCall::julia_command(paste0(
  "using TOML; function core070_bridge_p1_data(path, key); d = TOML.parsefile(path);",
  " Y = Float64.(reshape(d[key], d[\"p\"], d[\"n\"]));",
  " (Y = Y, p = d[\"p\"], n = d[\"n\"], K = d[\"K\"],",
  "  data_sha256 = bytes2hex(SHA.sha256(reinterpret(UInt8, vec(Y))))); end"))
JuliaCall::julia_command(paste0(
  "function core070_bridge_p1_health(path); h = TOML.parsefile(path);",
  " (r_loglik = Float64(h[\"r_loglik\"]), r_code = Float64(get(h, \"r_code\", NaN)),",
  "  r_converged = string(get(h, \"r_converged\", \"absent\")), r_gradient_max = Float64(h[\"r_gradient_max\"]),",
  "  policy = string(h[\"policy\"]), data_sha256 = string(h[\"data_sha256\"]),",
  "  native_loglik = Float64(h[\"native_loglik\"])); end"))
J <- function(x) JuliaCall::julia_eval(x)

capture <- function(expr) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) structure(list(error = conditionMessage(e)), class = "bridge_error"))
  list(value = value, warnings = warnings, error = if (inherits(value, "bridge_error")) value$error else NULL)
}
num <- function(x) unname(as.numeric(x))
long_data <- function(Y) {
  d <- expand.grid(trait = rownames(Y), site = colnames(Y))
  d$value <- as.vector(Y)
  d$trait <- factor(d$trait, levels = rownames(Y)); d$site <- factor(d$site, levels = colnames(Y))
  d
}
route_record <- function(fit) {
  if (!is.null(fit$error)) return(list(error = fit$error, warnings = fit$warnings))
  f <- fit$value
  L <- as.matrix(f$loadings)
  list(class = class(f), converged = isTRUE(f$converged), loglik = num(logLik(f)), df = num(f$df),
       n_traits = num(f$n_traits), n_units = num(f$n_units), d = num(f$d), alpha = num(f$alpha),
       shared_covariance = num(tcrossprod(L)), dispersion = num(f$dispersion),
       family = family_label(f$family), warnings = fit$warnings)
}
family_label <- function(x) tryCatch(as.character(jsonlite::toJSON(unclass(x), auto_unbox = TRUE, force = TRUE)),
                                     error = function(e) NA_character_)
native_record <- function(n) {
  if (!is.null(n$error)) return(list(error = n$error))
  v <- n$value
  list(loglik = num(v$loglik), converged = isTRUE(v$converged), alpha = num(v$alpha),
       shared_covariance = num(v$shared_covariance), dispersion = num(v$dispersion), df = num(v$df),
       gradient_max = num(v$gradient_max), fd_stability = num(v$fd_stability),
       objective_delta = num(v$objective_delta), hessian = v$hessian, data_sha256 = v$data_sha256)
}
health_record <- function(path) {
  h <- JuliaCall::julia_call("core070_bridge_p1_health", path)
  list(file = basename(path), sha256 = sha256_file(path), r_loglik = num(h$r_loglik),
       r_code = if (is.nan(h$r_code)) NA else num(h$r_code), r_converged = h$r_converged,
       r_gradient_max = num(h$r_gradient_max), policy = h$policy, data_sha256 = h$data_sha256,
       native_loglik = num(h$native_loglik))
}

cases <- list(
  list(id = "CORE070-FAMILY-02-LOG-PUBLIC-R-BRIDGE", family = "poisson", fam = poisson(), fam_call = "poisson()",
       data = file.path(runs_dir, "runparity-poisson", "poisson-fixture.toml"), key = "Y_column_major",
       health = file.path(runs_dir, "runparity-poisson", "poisson-health.toml"),
       expected_sha = case_of("poisson")$data_sha256, deviation = NULL),
  list(id = "CORE070-FAMILY-07-LOGIT-PUBLIC-R-BRIDGE", family = "beta", fam = gllvmTMB::Beta(), fam_call = "gllvmTMB::Beta()",
       data = file.path(runs_dir, "runparity-beta", "beta-fixture.toml"), key = "Y_column_major",
       health = file.path(runs_dir, "runparity-beta", "beta-health.toml"),
       expected_sha = case_of("beta")$data_sha256, deviation = NULL),
  list(id = "CORE070-FAMILY-05-LOG-PUBLIC-R-BRIDGE", family = "nb2", fam = gllvmTMB::nbinom2(), fam_call = "gllvmTMB::nbinom2()",
       data = file.path(root, "test/parity/fixtures/nb2_smoke_data.toml"), key = "Y_column_major",
       health = file.path(runs_dir, "runparity-nb2", "nb2-health.toml"),
       expected_sha = "2bf2d819802a66e9836600caefec6e50802cac047db5d1a14611b4152ff1837c",
       deviation = paste0("data: the stored NB2 smoke draw (test/parity/fixtures/nb2_smoke_data.toml, sha256 ",
                          "2bf2d819...), the data the NATIVE-06-NB2 cell fits since decision 2026-09-28; the ",
                          "frozen bridge contract names the original draw (", case_of("nb2")$data_sha256, ")")))

results <- list()
fixtures <- list()
for (cs in cases) {
  d <- JuliaCall::julia_call("core070_bridge_p1_data", cs$data, cs$key)
  K <- as.integer(d$K); p <- as.integer(d$p); n <- as.integer(d$n)
  Y <- matrix(as.numeric(d$Y), p, n)
  dimnames(Y) <- list(sprintf("trait%02d", seq_len(p)), sprintf("site%03d", seq_len(n)))
  data <- long_data(Y)
  fixtures[[cs$family]] <- list(Y = Y, data = data, K = K)
  native <- capture(JuliaCall::julia_call("core070_bridge_native", Y, cs$family, K))
  matrix_fit <- capture(gllvmTMB::gllvm_julia_fit(Y, family = cs$fam, num.lv = K, ci_method = "none"))
  formula_fit <- capture(gllvmTMB::gllvmTMB(
    value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
    data = data[nrow(data):1, ], unit = "site", trait = "trait", family = cs$fam,
    engine = "julia", ci_method = "none"))
  results[[cs$id]] <- list(
    case_id = cs$id, family = cs$family, p = p, n = n, K = K,
    data_file = sub(paste0("^", root, "/"), "", sub(paste0("^", runs_dir, "/"), "<runs>/", cs$data)),
    data_file_sha256 = sha256_file(cs$data), data_sha256 = d$data_sha256, expected_data_sha256 = cs$expected_sha,
    deviation = cs$deviation, same_run_r_fit = health_record(cs$health),
    native = native_record(native), matrix = route_record(matrix_fit), formula = route_record(formula_fit),
    r_calls = c(sprintf("gllvm_julia_fit(Y, family = %s, num.lv = %d, ci_method = \"none\")", cs$fam_call, K),
                sprintf("gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = %d, unique = FALSE), data = reversed long, unit = \"site\", trait = \"trait\", family = %s, engine = \"julia\", ci_method = \"none\")", K, cs$fam_call)))
  cat("FAMILY_BRIDGE_P1_CASE", cs$id, "\n")
}

## ---- BETA-ALIAS adapter case ------------------------------------------------------------
alias <- structure(list(family = "beta", link = "logit"), class = "family")
fb <- fixtures[["beta"]]
form <- value ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE)
r_fit <- function(fam) capture(gllvmTMB::gllvmTMB(form, data = fb$data, unit = "site", trait = "trait", family = fam))
r_record <- function(fit) {
  if (!is.null(fit$error)) return(list(error = fit$error, warnings = fit$warnings))
  f <- fit$value
  list(convergence = num(f$opt$convergence), message = f$opt$message, loglik = num(logLik(f)),
       df = num(attr(logLik(f), "df")), objective = num(f$opt$objective), par = num(f$opt$par),
       par_names = names(f$opt$par), pd_hessian = isTRUE(f$sd_report$pdHess),
       family = family_label(f$family), warnings = fit$warnings)
}
r_canonical <- r_fit(gllvmTMB::Beta())
r_alias <- r_fit(alias)
alias_native <- capture(JuliaCall::julia_call("core070_bridge_native", fb$Y, "beta", 1L))
alias_matrix <- capture(gllvmTMB::gllvm_julia_fit(fb$Y, family = alias, num.lv = 1L, ci_method = "none"))
alias_formula <- capture(gllvmTMB::gllvmTMB(form, data = fb$data[nrow(fb$data):1, ], unit = "site",
                                            trait = "trait", family = alias, engine = "julia", ci_method = "none"))
results[["CORE070-FAMILY-BETA-ALIAS-COMPATIBILITY-ADAPTER"]] <- list(
  case_id = "CORE070-FAMILY-BETA-ALIAS-COMPATIBILITY-ADAPTER", family = "beta", p = nrow(fb$Y), n = ncol(fb$Y), K = 1L,
  descriptor = "structure(list(family = \"beta\", link = \"logit\"), class = \"family\")",
  data_sha256 = results[["CORE070-FAMILY-07-LOGIT-PUBLIC-R-BRIDGE"]]$data_sha256,
  r_canonical = r_record(r_canonical), r_alias = r_record(r_alias),
  native = native_record(alias_native), matrix = route_record(alias_matrix), formula = route_record(alias_formula),
  r_calls = c("gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE), data, unit = \"site\", trait = \"trait\", family = <alias>) (TMB engine), and the same call with family = gllvmTMB::Beta()",
              "gllvm_julia_fit(Y, family = <alias>, num.lv = 1, ci_method = \"none\")",
              "gllvmTMB(<same formula>, data = reversed long, family = <alias>, engine = \"julia\", ci_method = \"none\")"))
cat("FAMILY_BRIDGE_P1_CASE CORE070-FAMILY-BETA-ALIAS-COMPATIBILITY-ADAPTER\n")

out <- list(
  schema = "core070-family-bridge-p1/v1", reference_commit = P1,
  gllvmtmb_version = as.character(packageVersion("gllvmTMB")), source_pin = source_pin,
  r_version = R.version.string, tmb_version = as.character(packageVersion("TMB")),
  matrix_version = as.character(packageVersion("Matrix")),
  juliacall_version = as.character(packageVersion("JuliaCall")), julia_version = J("string(VERSION)"),
  gllvmodels_source = J("pathof(GLLVModels)"), julia_threads = J("Threads.nthreads()"),
  blas_threads = J("GLLVModels.LinearAlgebra.BLAS.get_num_threads()"),
  bridge_contract = contract_path, bridge_contract_sha256 = sha256_file(contract_path),
  cases = unname(results))
jsonlite::write_json(out, file.path(output_dir, "results.json"), auto_unbox = TRUE, pretty = TRUE,
                     digits = NA, null = "null", na = "null")
cat("CORE070_FAMILY_BRIDGE_P1_WRITTEN", sha256_file(file.path(output_dir, "results.json")), "\n")
