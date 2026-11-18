# gllvmtmb_julia.R — a gllvmTMB-style R front end to GLLVModels.jl via JuliaConnectoR.
#
# This is the "engine = julia" bridge: it lets R `gllvm`/`gllvmTMB` users drive the
# fast Julia engine with a call that *looks like* gllvm::gllvm(...) — same family
# strings, `num.lv`, `row.eff`, `disp.formula` — and get back a list in gllvm
# parameter conventions. It mirrors the drmTMB <-> DRM.jl pattern.
#
# It BUILDS ON the low-level accessor wrappers in r/gllvmjl.R (coef table, getLV,
# loadings, predict, residuals). Those wrappers call GLLVModels.jl's per-family fitters
# directly with `K =` (the real Julia keyword); this file adds the gllvmTMB-flavoured
# *front door* + the parameterization conversions documented in
# docs/src/gllvmtmb-parity.md ("R bridge: parameterization map").
#
# STATUS: SCAFFOLD — transport smoke-tested in a live R + JuliaConnectoR session
# on 2026-06-14. Julia startup, `GLLVM`/`Distributions` loading, family marker
# construction, and already-converted field access now work for a small Poisson
# LA check. Full numerical parity with R `{gllvm}` is still open; see
# r/README_bridge.md and r/parity_check.R for the current diagnostic row.
#
# ---------------------------------------------------------------------------------
# IMPORTANT API NOTES (verified against src/ at authoring time)
# ---------------------------------------------------------------------------------
# * Orientation. gllvm/gllvmTMB take y as n x p (SITES in rows, SPECIES in columns).
#   GLLVModels.jl takes Y as p x n (SPECIES in rows, SITES in columns). We TRANSPOSE on
#   the way in, and the returned loadings (p x K) / scores (n x K) are already in
#   gllvm's (species, site) orientation after transpose handling.
# * The unified `fit_gllvm(Y; family, K, ...)` only covers the plain families
#   (Normal/Binomial/Poisson/NegativeBinomial/Beta/Ordinal/Gamma/Exponential) and
#   takes `K` (not `num_lv`). It has NO `row_eff` / `disp_group` / `pervar` kwargs.
#   Those route through DEDICATED fitters:
#     - per-species / grouped dispersion -> fit_<fam>_gllvm_grouped(Y; K, group)
#     - per-species Gaussian variance     -> fit_gaussian_pervar_gllvm(Y; K)
#     - fixed row effects                 -> fit_roweffect_gllvm(Y; family, K)
#     - random row effects                -> fit_row_random_gllvm(Y; family, K)
#   So this bridge dispatches to the right Julia fitter for each option combo,
#   rather than pretending one signature does it all.
# * Dispersion field names on the returned Julia fit objects:
#     NBFit$r, NB1Fit$phi, BetaFit$phi, GammaFit$alpha, TweedieFit$phi & $p,
#     GaussianPerVarFit$<phi-squared> (per-species variances).
#     Grouped variants are VECTORS: NBGroupedFit$r_group, BetaGroupedFit$phi,
#     GammaGroupedFit$alpha, NB1GroupedFit$phi, TweedieGroupedFit$phi.
#   JuliaConnectoR may return Unicode fields either as JuliaProxy objects or as
#   already-converted R values; `.jl_value()` handles both. Scalar β / φ / α
#   access is smoke-tested. GaussianPerVarFit's Unicode variance vector (`φ²`)
#   still needs a dedicated live pervar check.
# ---------------------------------------------------------------------------------

library(JuliaConnectoR)

# Reuse the low-level wrappers + the .gllvm() module handle if gllvmjl.R is present.
if (!exists(".gllvm", mode = "function")) {
  .this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
  .sib <- file.path(.this_dir, "gllvmjl.R")
  if (file.exists(.sib)) source(.sib)
}

# Fallback module handle if gllvmjl.R was not sourced (keeps this file usable alone).
if (!exists(".gllvm_env")) .gllvm_env <- new.env(parent = emptyenv())
if (!exists(".jl_string", mode = "function")) {
  .jl_string <- function(x) {
    x <- normalizePath(x, winslash = "/", mustWork = TRUE)
    paste0("\"", gsub('(["\\\\])', "\\\\\\1", x), "\"")
  }
}
if (!exists("gllvm_jl_init", mode = "function")) {
  gllvm_jl_init <- function(jl_path = Sys.getenv("GLLVM_JL_PATH", "")) {
    if (!identical(jl_path, "")) {
      jl_path <- normalizePath(jl_path, winslash = "/", mustWork = TRUE)
      juliaEval(sprintf("import Pkg; Pkg.activate(%s)", .jl_string(jl_path)))
      .gllvm_env$jl_path <- jl_path
    }
    module_name <- juliaEval(paste(
      'if Base.find_package("GLLVModels") !== nothing',
      '  "GLLVModels"',
      'elseif Base.find_package("GLLVM") !== nothing',
      '  "GLLVM"',
      'else',
      '  error("Install GLLVModels.jl in the active Julia environment.")',
      'end',
      sep = "\n"
    ))
    .gllvm_env$GLLVM <- juliaImport(module_name)
    juliaEval("using Distributions")
    invisible(TRUE)
  }
}
if (!exists(".gllvm", mode = "function")) {
  .gllvm <- function() { if (is.null(.gllvm_env$GLLVM)) gllvm_jl_init(); .gllvm_env$GLLVM }
}
if (!exists(".jl_value", mode = "function")) {
  .jl_value <- function(x) {
    if (inherits(x, "JuliaProxy")) juliaGet(x) else x
  }
}

# ---------------------------------------------------------------------------------
# Family + link mapping (gllvm string -> GLLVModels.jl marker / fitter metadata).
# ---------------------------------------------------------------------------------
# `julia_family` is the Distributions.jl marker constructor name to use with the
# unified fit_gllvm dispatch (where supported); `disp` names the dispersion concept
# and how to convert it back to gllvm's convention (see .convert_dispersion below).
# `int` flags integer-valued responses (Poisson/NB/Binomial/Ordinal).
.family_map <- list(
  gaussian            = list(julia = "Normal",           disp = "gaussian_sd", int = FALSE),
  poisson             = list(julia = "Poisson",          disp = "none",        int = TRUE),
  `negative.binomial` = list(julia = "NegativeBinomial", disp = "nb2",         int = TRUE),
  negative.binomial1  = list(julia = "NegativeBinomial", disp = "nb1",         int = TRUE), # NB1: own fitter
  binomial            = list(julia = "Binomial",         disp = "none",        int = TRUE),
  beta                = list(julia = "Beta",             disp = "beta",        int = FALSE),
  Gamma               = list(julia = "Gamma",            disp = "gamma",       int = FALSE),
  gamma               = list(julia = "Gamma",            disp = "gamma",       int = FALSE),
  exponential         = list(julia = "Exponential",      disp = "none",        int = FALSE),
  ordinal             = list(julia = "Ordinal",          disp = "none",        int = TRUE),
  tweedie             = list(julia = "Tweedie",          disp = "tweedie",     int = FALSE)
)

# gllvm link string -> Julia Link constructor name.
.link_map <- list(
  log     = "LogLink",
  logit   = "LogitLink",
  probit  = "ProbitLink",
  cloglog = "CLogLogLink",
  identity= "IdentityLink"
)

# Sensible default link per family (matches gllvm/GLLVModels.jl defaults).
.default_link <- function(family) {
  switch(family,
    gaussian = "identity",
    poisson = , `negative.binomial` = , negative.binomial1 = , tweedie = ,
    Gamma = , gamma = , exponential = "log",
    binomial = "logit",
    beta = "logit",
    ordinal = "logit",
    "log")
}

# ---------------------------------------------------------------------------------
# Dispersion conversion: Julia (engine) -> gllvm (R) convention.
# Source of truth: docs/src/gllvmtmb-parity.md, table "R bridge: parameterization map".
# Returns a list(name=<gllvm name>, value=<numeric, possibly vector>).
# ---------------------------------------------------------------------------------
.convert_dispersion <- function(disp_kind, julia_fit, grouped) {
  switch(disp_kind,
    # NB2: gllvm phi (Var = mu + mu^2 * phi); Julia r (size, Var = mu + mu^2/r).
    # Bridge rule: phi = 1 / r  (also applies to ZINB / Hurdle-NB / grouped-NB).
    nb2 = {
      r <- if (grouped) as.numeric(.jl_value(julia_fit$r_group)) else as.numeric(.jl_value(julia_fit$r))
      list(name = "phi", value = 1 / r)                       # r = 1/phi  <=>  phi = 1/r
    },
    # NB1: gllvm phi (Var = mu + mu*phi); Julia phi (Var = mu*(1+phi)). Identity map.
    nb1 = {
      v <- if (grouped) as.numeric(.jl_value(julia_fit[["φ"]])) else as.numeric(.jl_value(julia_fit[["φ"]]))
      list(name = "phi", value = v)                           # identity (maps 1:1)
    },
    # Gamma: gllvm phi == SHAPE (Var = mu^2/phi); Julia alpha == shape (Var = mu^2/alpha).
    # Relabel only, no inversion.
    gamma = {
      a <- if (grouped) as.numeric(.jl_value(julia_fit[["α"]])) else as.numeric(.jl_value(julia_fit[["α"]]))
      list(name = "phi", value = a)                           # relabel alpha -> phi
    },
    # Beta: precision phi identical in both. Identity.
    beta = {
      v <- if (grouped) as.numeric(.jl_value(julia_fit[["φ"]])) else as.numeric(.jl_value(julia_fit[["φ"]]))
      list(name = "phi", value = v)                           # identity
    },
    # Tweedie: power and phi both identity; we also surface the power.
    tweedie = {
      list(name = "phi", value = as.numeric(.jl_value(julia_fit[["φ"]])),
           power = as.numeric(.jl_value(julia_fit$p)))         # identity (set p_init=1.1 to match gllvm path)
    },
    # Gaussian per-species SD: gllvm reports per-species phi_j (SD). GLLVModels.jl pervar
    # fit stores per-species VARIANCES; gllvm convention is SD, so sqrt them.
    gaussian_sd = {
      ## VERIFY: field name for the per-species variance vector on GaussianPerVarFit
      ## (Julia source names it with a Unicode "phi-squared"). Access + sqrt -> SD.
      v2 <- tryCatch(as.numeric(.jl_value(julia_fit[["φ²"]])), error = function(e) NA_real_)
      list(name = "phi", value = sqrt(v2))                    # gllvm reports SD
    },
    # No dispersion parameter (Poisson, Binomial, Exponential, Ordinal).
    none = list(name = NA_character_, value = NA_real_),
    list(name = NA_character_, value = NA_real_)
  )
}

# ---------------------------------------------------------------------------------
# The gllvmTMB-style front door.
# ---------------------------------------------------------------------------------

#' Fit a GLLVM in Julia with a gllvmTMB-style call.
#'
#' @param y          n x p response matrix (SITES in rows, SPECIES in columns) —
#'                   the gllvm orientation. Transposed to p x n for GLLVModels.jl.
#' @param X          (reserved) site covariate matrix/data.frame. NOT YET wired
#'                   through this legacy direct `gllvm_julia()` scaffold —
#'                   passing a non-NULL X errors. The current `gllvmTMB(...,
#'                   engine = "julia")` bridge has its own capability ledger and
#'                   admits a tested subset of `X` models through `GLLVModels.bridge_fit`.
#' @param family     gllvm family string; one of names(.family_map).
#' @param num.lv     number of latent variables (gllvm name; -> Julia `K`).
#' @param row.eff    "none" (default), "fixed", or "random".
#' @param disp.formula  NULL  -> per-species dispersion (gllvm default; routes via a
#'                            grouped fitter with group = 1:p), or
#'                      ~1   -> shared scalar dispersion (the plain fitter).
#'                   Only meaningful for families with a dispersion parameter.
#' @param link       link string (see .link_map); default per family.
#' @param method     "LA" (Laplace, default) or "VA" (variational). gllvm's default
#'                   is "VA"; GLLVModels.jl's default path is Laplace. Pin to match.
#' @param N          n x p trial-count matrix for binomial (default Bernoulli).
#' @param p_init     Tweedie only: optimiser start for the power (set 1.1 to match gllvm).
#' @param ...        forwarded to the underlying Julia fitter (e.g. g_tol, iterations).
#'
#' @return a list mirroring a gllvm fit object's key components:
#'   $coefficients (intercepts beta, gllvm "Intercept"/"theta" scale),
#'   $loadings (p x K), $lvs (n x K site scores), $logLik, $df, $num.lv, $family,
#'   $link, $method, $row.eff, $dispersion (list: name in gllvm convention + value),
#'   $params (raw, engine-scale), $jl_fit (opaque Julia ref for the accessors),
#'   $y (p x n, as passed to Julia). The dispersion is ALREADY converted to the
#'   gllvm convention (e.g. NB phi = 1/r) per docs/src/gllvmtmb-parity.md.
gllvm_julia <- function(y, X = NULL, family = "negative.binomial", num.lv = 2L,
                        row.eff = c("none", "fixed", "random"),
                        disp.formula = NULL, link = NULL, method = c("LA", "VA"),
                        N = NULL, p_init = NULL, ...) {
  G <- .gllvm()
  row.eff <- match.arg(row.eff)
  method  <- toupper(match.arg(method))
  if (!family %in% names(.family_map))
    stop(sprintf("family '%s' not supported by the bridge (supported: %s)",
                 family, paste(names(.family_map), collapse = ", ")))
  fmap <- .family_map[[family]]
  if (family == "negative.binomial1" && row.eff != "none")
    stop("family 'negative.binomial1' (NB1) with row.eff = '", row.eff, "' is not supported: ",
         "GLLVModels.jl has no NB1 row-effect fitter, and this bridge would fit NB2 instead. ",
         "Use row.eff = 'none' with NB1, or family = 'negative.binomial' to keep the row effect.",
         call. = FALSE)
  link <- if (is.null(link)) .default_link(family) else link
  if (!link %in% names(.link_map))
    stop(sprintf("link '%s' not supported (supported: %s)",
                 link, paste(names(.link_map), collapse = ", ")))

  if (!is.null(X))
    stop("X (site covariates) is not yet wired through this bridge scaffold; ",
         "use GLLVModels.jl's fit_gllvm_cov / @formula directly, or extend gllvm_julia. ",
         "See r/README_bridge.md 'known-unsupported combos'.")

  K <- as.integer(num.lv)
  if (is.na(K) || K < 1L) stop("num.lv must be a positive integer")

  # Orientation + storage mode. gllvm y is n x p; GLLVModels.jl wants p x n.
  Y <- t(as.matrix(y))                                    # now p x n
  storage.mode(Y) <- if (isTRUE(fmap$int)) "integer" else "double"
  Nt <- if (!is.null(N)) { Nm <- t(as.matrix(N)); storage.mode(Nm) <- "integer"; Nm } else NULL

  # Dispersion structure: NULL => per-species (route via grouped fitter, group=1:p);
  # ~1 => shared scalar (plain fitter). Only relevant for the dispersion families.
  has_disp <- fmap$disp %in% c("nb2", "nb1", "gamma", "beta", "tweedie", "gaussian_sd")
  per_species <- has_disp && is.null(disp.formula)        # gllvm's per-species default
  shared      <- has_disp && (!per_species)

  jl_family <- .jl_family_marker(G, fmap$julia, family)
  jl_link   <- G[[.link_map[[link]]]]()

  # ---- VA vs LA guardrails (the available VA fitters) ----
  va_ok <- family %in% c("poisson", "negative.binomial", "binomial", "beta", "Gamma", "gamma")
  if (method == "VA" && !va_ok)
    stop(sprintf("method='VA' is not available for family '%s' in GLLVModels.jl ", family),
         "(VA fitters exist for poisson / negative.binomial / binomial / beta / gamma). ",
         "Use method='LA'.")
  if (method == "VA" && (per_species || row.eff != "none"))
    stop("method='VA' is only wired here for the plain (shared-dispersion, no-row-effect) ",
         "fitters; use method='LA' for grouped dispersion or row effects.")

  # ---- dispatch to the right Julia fitter ----
  fit <- .dispatch_fit(G, family, fmap, Y, K, jl_family, jl_link, method,
                       Nt, row.eff, per_species, shared, p_init, ...)

  grouped <- per_species  # per-species routed through the grouped fitter
  disp <- .convert_dispersion(fmap$disp, fit, grouped)

  # Loadings (p x K), scores (n x K), intercepts, logLik via the engine accessors.
  loadings <- tryCatch(as.matrix(G$getLoadings(fit, rotate = TRUE)),
                       error = function(e) NULL)
  lvs <- tryCatch(.bridge_getLV(G, fit, Y, Nt), error = function(e) NULL)
  beta <- tryCatch(as.numeric(.jl_value(fit$β)), error = function(e) NA_real_) # fit.β intercepts
  ll  <- tryCatch(as.numeric(.jl_value(fit$loglik)), error = function(e) NA_real_)

  structure(list(
    call        = match.call(),
    family      = family,
    link        = link,
    method      = method,
    row.eff     = row.eff,
    num.lv      = K,
    coefficients= beta,           # species intercepts (engine scale; identity for log/logit etc.)
    loadings    = loadings,       # p x K
    lvs         = lvs,            # n x K site scores
    logLik      = ll,
    dispersion  = disp,           # gllvm convention (e.g. NB phi = 1/r)
    disp.structure = if (!has_disp) "none" else if (per_species) "per-species" else "shared",
    jl_fit      = fit,            # opaque Julia ref — pass to gllvm_coeftable(), predict(), ...
    y           = Y               # p x n (as sent to Julia)
  ), class = "gllvm_julia")
}

# Build the Distributions.jl family marker. Most are zero-arg constructors; the
# bridge's NB2 fitter estimates r internally, so a marker with a placeholder is fine.
.jl_family_marker <- function(G, julia_name, family) {
  expr <- sprintf("Distributions.%s()", julia_name)
  tryCatch(
    juliaEval(expr),
    error = function(e) {
      stop(sprintf("Julia constructor %s failed for family '%s': %s",
                   expr, family, conditionMessage(e)), call. = FALSE)
    }
  )
}

# Accessor for site scores: signature differs across fit types (some take N).
.bridge_getLV <- function(G, fit, Y, Nt) {
  if (!is.null(Nt)) {
    as.matrix(G$getLV(fit, Y, N = Nt, rotate = TRUE))
  } else {
    as.matrix(G$getLV(fit, Y, rotate = TRUE))
  }
}

# ---------------------------------------------------------------------------------
# Fitter dispatch. Encodes the routing rules from docs/src/gllvmtmb-parity.md:
#   - row.eff fixed/random  -> fit_roweffect_gllvm / fit_row_random_gllvm
#   - per-species dispersion -> fit_<fam>_gllvm_grouped(Y; K, group = 1:p)
#   - shared / plain         -> fit_<fam>_gllvm (or fit_*_gllvm_va for method=VA)
#   - gaussian per-species   -> fit_gaussian_pervar_gllvm
# ---------------------------------------------------------------------------------
.dispatch_fit <- function(G, family, fmap, Y, K, jl_family, jl_link, method,
                          Nt, row.eff, per_species, shared, p_init, ...) {
  p <- nrow(Y)

  # ---- row effects take priority (fixed/random) ----
  if (row.eff == "fixed") {
    # fit_roweffect_gllvm(Y; family, K, link, N)
    if (is.null(Nt)) return(G$fit_roweffect_gllvm(Y, family = jl_family, K = K, link = jl_link, ...))
    return(G$fit_roweffect_gllvm(Y, family = jl_family, K = K, link = jl_link, N = Nt, ...))
  }
  if (row.eff == "random") {
    # fit_row_random_gllvm(Y; family, K, link, N)
    if (is.null(Nt)) return(G$fit_row_random_gllvm(Y, family = jl_family, K = K, link = jl_link, ...))
    return(G$fit_row_random_gllvm(Y, family = jl_family, K = K, link = jl_link, N = Nt, ...))
  }

  # ---- Gaussian special cases ----
  if (family == "gaussian") {
    if (per_species) return(G$fit_gaussian_pervar_gllvm(Y, K = K, ...))   # per-species variances
    return(G$fit_gaussian_gllvm(Y, K = K, ...))                          # shared sigma (profiled)
  }

  # ---- Tweedie (no unified dispatch; its own fitter) ----
  if (family == "tweedie") {
    pinit <- if (is.null(p_init)) 1.5 else as.numeric(p_init)
    if (per_species)
      return(G$fit_tweedie_gllvm_grouped(Y, K = K, group = .group1p(G, p), link = jl_link, p_init = pinit, ...))
    return(G$fit_tweedie_gllvm(Y, K = K, link = jl_link, p_init = pinit, ...))
  }

  # ---- NB1 (own fitter; not in unified dispatch) ----
  if (family == "negative.binomial1") {
    if (per_species)
      return(G$fit_nb1_gllvm_grouped(Y, K = K, group = .group1p(G, p), link = jl_link, ...))
    return(G$fit_nb1_gllvm(Y, K = K, link = jl_link, ...))
  }

  # ---- ordinal (own fitter; no dispersion) ----
  if (family == "ordinal") return(G$fit_ordinal_gllvm(Y, K = K, link = jl_link, ...))

  # ---- per-species dispersion families -> grouped fitters with group = 1:p ----
  if (per_species) {
    grouped_fitter <- switch(family,
      `negative.binomial` = "fit_nb_gllvm_grouped",
      beta                = "fit_beta_gllvm_grouped",
      Gamma = , gamma     = "fit_gamma_gllvm_grouped",
      stop(sprintf("per-species dispersion not wired for family '%s'", family)))
    return(G[[grouped_fitter]](Y, K = K, group = .group1p(G, p), link = jl_link, ...))
  }

  # ---- shared / plain families: VA or LA ----
  if (method == "VA") {
    va_fitter <- switch(family,
      poisson             = "fit_poisson_gllvm_va",
      `negative.binomial` = "fit_nb_gllvm_va",
      binomial            = "fit_binomial_gllvm_va",
      beta                = "fit_beta_gllvm_va",
      Gamma = , gamma     = "fit_gamma_gllvm_va",
      stop(sprintf("no VA fitter for family '%s'", family)))
    fn <- G[[va_fitter]]
    if (family == "binomial" && !is.null(Nt)) return(fn(Y, K = K, N = Nt, link = jl_link, ...))
    return(fn(Y, K = K, link = jl_link, ...))
  }

  # LA (Laplace) plain fitters — go through the unified fit_gllvm where possible.
  if (family == "binomial" && !is.null(Nt))
    return(G$fit_gllvm(Y, family = jl_family, K = K, link = jl_link, N = Nt, ...))
  G$fit_gllvm(Y, family = jl_family, K = K, link = jl_link, ...)
}

# group = 1:p as a Julia integer vector (each species its own dispersion group).
.group1p <- function(G, p) juliaCall("collect", juliaCall(":", 1L, as.integer(p)))

# ---------------------------------------------------------------------------------
# Convenience pass-throughs that reuse the low-level accessors (orientation-aware).
# These accept a gllvm_julia object (or a raw Julia fit) and the original n x p y.
# ---------------------------------------------------------------------------------

#' Tidy Wald coefficient table for a gllvm_julia fit.
gllvm_julia_coeftable <- function(fit) {
  G <- .gllvm()
  jf <- if (inherits(fit, "gllvm_julia")) fit$jl_fit else fit
  Y  <- if (inherits(fit, "gllvm_julia")) fit$y else stop("pass a gllvm_julia object")
  ct <- G$coef_table(jf, Y)
  data.frame(
    term      = as.character(.jl_value(ct$term)),
    estimate  = as.numeric(.jl_value(ct$estimate)),
    std.error = as.numeric(.jl_value(ct$std_error)),
    z         = as.numeric(.jl_value(ct$z)),
    p.value   = as.numeric(.jl_value(ct$pvalue)),
    lower     = as.numeric(.jl_value(ct$lower)),
    upper     = as.numeric(.jl_value(ct$upper)),
    stringsAsFactors = FALSE
  )
}

#' Predicted means (type="response") or linear predictor (type="link"), returned
#' n x p (gllvm orientation).
gllvm_julia_predict <- function(fit, type = "response") {
  G <- .gllvm()
  jf <- if (inherits(fit, "gllvm_julia")) fit$jl_fit else fit
  Y  <- fit$y
  pr <- G$predict(jf, Y, type = juliaCall("Symbol", type))
  t(as.matrix(pr))                                        # back to n x p
}

# Pretty printer.
print.gllvm_julia <- function(x, ...) {
  cat(sprintf("GLLVModels.jl fit (via JuliaConnectoR) — family=%s, link=%s, method=%s\n",
              x$family, x$link, x$method))
  cat(sprintf("  num.lv = %d, row.eff = %s, dispersion structure = %s\n",
              x$num.lv, x$row.eff, x$disp.structure))
  cat(sprintf("  logLik = %s\n", format(x$logLik, digits = 8)))
  if (!is.na(x$dispersion$name)) {
    v <- x$dispersion$value
    vs <- if (length(v) > 6) sprintf("[%s, ...] (length %d)",
              paste(format(head(v, 3), digits = 4), collapse = ", "), length(v))
          else paste(format(v, digits = 4), collapse = ", ")
    cat(sprintf("  dispersion (%s, gllvm convention) = %s\n", x$dispersion$name, vs))
  }
  invisible(x)
}
