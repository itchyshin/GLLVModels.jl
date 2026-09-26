using TOML
using SHA

parity_loadings_p5k2() = [
    0.8   0.0
    0.5   0.6
    0.3  -0.4
   -0.2   0.5
    0.1   0.3
]

# Zero-truncated NB2 interior/boundary fixtures for the `truncated_nbinom2` second-
# order cell and the per-trait Wald CI test. Pure Julia (TOML + SHA, no RCall) so
# test/test_second_order_truncnb2_ci.jl can include it directly in the default suite,
# same reason r_lib.jl above stays RCall-free.
include(joinpath(@__DIR__, "truncnb2_fixtures.jl"))

function parity_site_design(x::AbstractVector{<:Real}, p::Integer)
    n = length(x)
    X = zeros(Float64, p, n, 1)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = Float64(x[s])
    end
    return X
end

# ---------------------------------------------------------------------------
# Local samplers (ported from the same test files; no Distributions dep for
# the ones the original tests avoided it for -- kept identical so the DGP
# reproduces bit-for-bit under the same seed).
# ---------------------------------------------------------------------------
function _rand_poisson(λ::Float64)
    λ = clamp(λ, 0.0, 1e6)
    L = exp(-λ)
    k = 0
    prod = 1.0
    while true
        k += 1
        prod *= rand()
        prod <= L && return k - 1
    end
end

function _rand_beta_jonk(a::Float64, b::Float64)
    a = max(a, 1e-12)
    b = max(b, 1e-12)
    while true
        u = rand()
        v = rand()
        x = u^(1 / a)
        y = v^(1 / b)
        s = x + y
        if s <= 1.0 && s > 0.0
            return x / s
        end
    end
end

function _rand_gamma_ms(shape::Float64, scale::Float64)
    if shape < 1.0
        return _rand_gamma_ms(shape + 1.0, scale) * rand()^(1.0 / shape)
    end
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        z = randn()
        v = (1.0 + c * z)^3
        v <= 0 && continue
        u = rand()
        z2 = z * z
        u < 1.0 - 0.0331 * z2 * z2 && return d * v * scale
        logu = log(u)
        logu < 0.5 * z2 + d * (1.0 - v + log(v)) && return d * v * scale
    end
end

_rand_nb2_ms(μ::Float64, r::Float64) = _rand_poisson(_rand_gamma_ms(r, μ / r))

# ===========================================================================
# R-side (se=TRUE) fit helpers -- generalized twins of
# fit_gllvmtmb_parity_loglik / _x / _species_x (test/parity/parity_helpers.jl)
# with se=TRUE and sd_report extraction (mirrors se-prerun-01's r_fit.R).
# ===========================================================================
include(joinpath(@__DIR__, "r_lib.jl"))

function _require_gllvmtmb!()
    lib = second_order_r_lib()
    if lib === nothing
        R"""
        suppressMessages(library(gllvmTMB))
        """
        return nothing
    end
    @rput lib
    R"""
    local({
      expected <- normalizePath(file.path(lib, "gllvmTMB"), mustWork = TRUE)
      loaded_from <- function() normalizePath(getNamespaceInfo("gllvmTMB", "path"), mustWork = TRUE)
      if ("gllvmTMB" %in% loadedNamespaces() && loaded_from() != expected) {
        stop("gllvmTMB is already loaded from ", loaded_from(),
             ", not GLLVM_PARITY_R_LIBS (", lib, "); start a fresh process")
      }
      .libPaths(c(lib, .libPaths()))
      suppressMessages(library(gllvmTMB, lib.loc = lib))
      if (loaded_from() != expected) {
        stop("gllvmTMB loaded from ", loaded_from(), ", not GLLVM_PARITY_R_LIBS (", lib, ")")
      }
    })
    """
    return nothing
end

# no-X
function r_fit_se(y::AbstractMatrix, K::Integer; family::Symbol,
        N::Union{Nothing,AbstractMatrix} = nothing, binomial_link::Symbol = :logit)
    p, n = size(y)
    trials = N === nothing ? ones(Float64, size(y)) : Matrix{Float64}(N)
    trials_provided = N !== nothing
    fam = String(family)
    blink = String(binomial_link)
    _require_gllvmtmb!()
    @rput y K p n fam trials blink trials_provided
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- switch(fam,
        gaussian     = stats::gaussian(),
        binomial     = stats::binomial(link = blink),
        poisson      = stats::poisson(),
        gamma        = stats::Gamma(link = "log"),
        negbinomial  = gllvmTMB::nbinom2(),
        nb1          = gllvmTMB::nbinom1(),
        beta         = gllvmTMB::Beta(),
        betabinomial = gllvmTMB::betabinomial(),
        lognormal = gllvmTMB::lognormal(),
        truncated_poisson = gllvmTMB::truncated_poisson(),
        truncated_nbinom2 = gllvmTMB::truncated_nbinom2(),
        stop(sprintf("unknown family: %s", fam))
    )
    weights_vec <- if (identical(fam, "betabinomial") || (identical(fam, "binomial") && trials_provided)) as.vector(trials) else NULL
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        weights = weights_vec,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); se_raw <- numeric(0); cv <- matrix(numeric(0),0,0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        se_raw <- sqrt(diag(cv))
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# small helpers so the NamedTuple above stays readable (rcopy needs care with
# possibly-empty R vectors)
has_sd_names() = rcopy(Vector{String}, R"nm")
has_sd_pf() = rcopy(Vector{Float64}, R"as.numeric(pf)")
has_sd_cv() = rcopy(Matrix{Float64}, R"matrix(as.numeric(cv), nrow=nrow(cv))")

# Delta-lognormal / Delta-Gamma (twin-identity shared η; test/parity/parity_helpers.jl)
function r_fit_se_delta(y::AbstractMatrix, K::Integer; family::Symbol)
    family in (:delta_lognormal, :delta_gamma) ||
        throw(ArgumentError("r_fit_se_delta: family must be :delta_lognormal or :delta_gamma; got :$family"))
    p, n = size(y)
    fam = String(family)
    _require_gllvmtmb!()
    @rput y K p n fam
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- switch(fam,
        delta_lognormal = gllvmTMB::delta_lognormal(),
        delta_gamma     = gllvmTMB::delta_gamma(),
        stop(sprintf("unknown family: %s", fam))
    )
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); cv <- matrix(numeric(0), 0, 0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# Extract sd_report fixed block after a gllvmTMB fit (or adapter rebuild).
function _run_r_tweedie_sd_extract!()
    R"""
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); cv <- matrix(numeric(0), 0, 0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return nothing
end

# Tweedie fixed-power (parity with Julia `fit_tweedie_gllvm_grouped(...; power=p0)`).
# Estimated shared power: `r_fit_se_tweedie_shared` (adapter + sdreport).
# Species estimated power stays out until its own SO cell.
function r_fit_se_tweedie(y::AbstractMatrix, K::Integer; p_fixed::Real)
    (1.0 < p_fixed < 2.0) || throw(ArgumentError(
        "r_fit_se_tweedie: p_fixed must be in (1, 2); got $p_fixed"))
    p, n = size(y)
    pv = Float64(p_fixed)
    _require_gllvmtmb!()
    @rput y K p n pv
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- gllvmTMB::tweedie(link = "log", p = pv)
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    """
    _run_r_tweedie_sd_extract!()
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# Estimated shared Tweedie power — reference constraint adapter + TMB sdreport.
# Julia-side tools only (same adapter as `fit_gllvmtmb_parity_tweedie`); no R engine change.
function r_fit_se_tweedie_shared(y::AbstractMatrix, K::Integer)
    p, n = size(y)
    _require_gllvmtmb!()
    @rput y K p n
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- gllvmTMB::tweedie(link = "log")
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    if (!identical(fit_r$estimator, "ML") || isTRUE(fit_r$aghq$used) ||
        !identical(fit_r$random, "z_B") || !all(fit_r$tmb_data$family_id == 6L)) {
        stop("shared Tweedie power adapter requires an ordinary ML, all-Tweedie, z_B-only Laplace fit", call. = FALSE)
    }
    shared_map <- fit_r$tmb_map
    shared_params <- fit_r$tmb_params
    fitted_params <- fit_r$tmb_obj$env$parList(fit_r$opt$par)
    for (nm in intersect(names(shared_params), names(fitted_params))) {
        shared_params[[nm]] <- fitted_params[[nm]]
    }
    length(shared_params$logit_p_tweedie) == p ||
        stop("shared Tweedie adapter expected one power entry per trait", call. = FALSE)
    shared_params$logit_p_tweedie[] <- mean(fitted_params$logit_p_tweedie)
    shared_map$logit_p_tweedie <- factor(rep(1L, p))
    obj_shared <- TMB::MakeADFun(
        data = fit_r$tmb_data, parameters = shared_params, map = shared_map,
        random = fit_r$random, DLL = "gllvmTMB", silent = TRUE
    )
    sum(grepl("^logit_p_tweedie", names(obj_shared$par))) == 1L ||
        stop("shared Tweedie adapter did not produce exactly one free power coordinate", call. = FALSE)
    opt_shared <- nlminb(start = obj_shared$par, objective = obj_shared$fn,
                         gradient = obj_shared$gr)
    fit_r$opt <- opt_shared
    fit_r$tmb_obj <- obj_shared
    fit_r$tmb_params <- shared_params
    fit_r$tmb_map <- shared_map
    obj_shared$fn(opt_shared$par)
    fit_r$report <- obj_shared$report(obj_shared$env$last.par.best)
    p_report <- as.numeric(fit_r$report$p_tweedie)
    if (!all(is.finite(p_report)) || length(unique(round(p_report, 12))) != 1L) {
        stop("shared Tweedie adapter report does not carry one common power", call. = FALSE)
    }
    fit_r$sd_report <- TMB::sdreport(obj_shared, par.fixed = opt_shared$par, getJointPrecision = FALSE)
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- -as.numeric(fit_r$opt$objective)
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    """
    _run_r_tweedie_sd_extract!()
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
        reference_constraint_adapter = true,
        n_power_free = 1,
    )
end

# Per-trait estimated Tweedie power — public frozen-R `tweedie()` with se=TRUE.
function r_fit_se_tweedie_species(y::AbstractMatrix, K::Integer)
    p, n = size(y)
    _require_gllvmtmb!()
    @rput y K p n
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- gllvmTMB::tweedie(link = "log")
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    """
    _run_r_tweedie_sd_extract!()
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
        reference_constraint_adapter = false,
        n_power_free = p,
    )
end

# Ordinal per-trait cutpoints (probit); twin matches test_ordinal_probit_parity.jl.
function r_fit_se_ordinal_probit(y::AbstractMatrix{<:Integer}, K::Integer)
    p, n = size(y)
    _require_gllvmtmb!()
    @rput y K p n
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait",
        family = ordinal_probit(),
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); cv <- matrix(numeric(0), 0, 0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# Student-t fixed df (identity link); per-trait σ on R side — pair β[] with species σ on Julia.
function r_fit_se_student(y::AbstractMatrix, K::Integer; df_fixed::Real)
    df_fixed > 1 || throw(ArgumentError("df_fixed must be > 1"))
    p, n = size(y)
    dfv = Float64(df_fixed)
    _require_gllvmtmb!()
    @rput y K p n dfv
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fam_obj <- gllvmTMB::student(link = "identity", df = dfv)
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); cv <- matrix(numeric(0), 0, 0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# Multinomial FE softmax (no LV); y is length-n category codes 1..ncat.
function r_fit_se_multinomial(y::AbstractVector{<:Integer}, ncat::Integer)
    ncat >= 3 || throw(ArgumentError("multinomial needs ncat ≥ 3"))
    n = length(y)
    yv = collect(Int, y)
    _require_gllvmtmb!()
    @rput yv ncat n
    R"""
    lev <- as.character(seq_len(ncat))
    df_long <- data.frame(
        unit  = factor(seq_len(n)),
        trait = factor(rep("t1", n)),
        value = factor(as.character(yv), levels = lev)
    )
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait,
        data = df_long, unit = "unit", trait = "trait",
        family = multinomial(),
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); cv <- matrix(numeric(0), 0, 0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# shared site-X
function r_fit_se_x(y::AbstractMatrix, x_site::AbstractVector{<:Real}, K::Integer;
        family::Symbol, N::Union{Nothing,AbstractMatrix} = nothing,
        binomial_link::Symbol = :logit)
    p, n = size(y)
    trials = N === nothing ? ones(Float64, size(y)) : Matrix{Float64}(N)
    trials_provided = N !== nothing
    fam = String(family)
    blink = String(binomial_link)
    x = collect(Float64, x_site)
    _require_gllvmtmb!()
    @rput y K p n fam x trials blink trials_provided
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y),
        x     = rep(as.numeric(x), each = p)
    )
    fam_obj <- switch(fam,
        gaussian     = stats::gaussian(),
        binomial     = stats::binomial(link = blink),
        poisson      = stats::poisson(),
        gamma        = stats::Gamma(link = "log"),
        negbinomial  = gllvmTMB::nbinom2(),
        nb1          = gllvmTMB::nbinom1(),
        beta         = gllvmTMB::Beta(),
        betabinomial = gllvmTMB::betabinomial(),
        stop(sprintf("unknown family: %s", fam))
    )
    weights_vec <- if (identical(fam, "betabinomial") || (identical(fam, "binomial") && trials_provided)) as.vector(trials) else NULL
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + x + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        weights = weights_vec,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); se_raw <- numeric(0); cv <- matrix(numeric(0),0,0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        se_raw <- sqrt(diag(cv))
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# species-specific X ((0+trait):x)
function r_fit_se_species_x(y::AbstractMatrix, x_site::AbstractVector{<:Real}, K::Integer;
        family::Symbol)
    p, n = size(y)
    fam = String(family)
    x = collect(Float64, x_site)
    _require_gllvmtmb!()
    @rput y K p n fam x
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y),
        x     = rep(as.numeric(x), each = p)
    )
    fam_obj <- switch(fam,
        poisson  = stats::poisson(),
        binomial = stats::binomial(link = "logit"),
        stop(sprintf("unknown family: %s", fam))
    )
    t0 <- Sys.time()
    fit_r <- gllvmTMB(
        value ~ 0 + trait + (0 + trait):x + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
    )
    wall_fit <- as.numeric(Sys.time() - t0, units = "secs")
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    has_sd  <- !is.null(fit_r$sd_report)
    nm <- character(0); pf <- numeric(0); se_raw <- numeric(0); cv <- matrix(numeric(0),0,0)
    pdh <- NA; rcond <- NA_real_
    if (has_sd) {
        sdr <- fit_r$sd_report
        pf  <- sdr$par.fixed
        cv  <- sdr$cov.fixed
        se_raw <- sqrt(diag(cv))
        nm  <- names(pf)
        pdh <- isTRUE(sdr$pdHess)
        rcond <- tryCatch(kappa(cv), error = function(e) NA_real_)
    }
    """
    return (
        logLik = rcopy(Float64, R"r_logL"),
        objective = rcopy(Float64, R"r_obj"),
        converged = rcopy(Bool, R"r_conv"),
        has_sd = rcopy(Bool, R"has_sd"),
        names = has_sd_names(),
        par_fixed = has_sd_pf(),
        cov_fixed = has_sd_cv(),
        pd_hessian = rcopy(Any, R"pdh"),
        r_condition_number = rcopy(Any, R"rcond"),
        wall_fit = rcopy(Float64, R"wall_fit"),
    )
end

# ===========================================================================
# JSON writer (hand-rolled -- no JSON dependency assumed in this Project.toml)
# ===========================================================================
function _jnum(x::Real)
    xf = Float64(x)
    isnan(xf) && return "null"
    isinf(xf) && return xf > 0 ? "1e309" : "-1e309"
    return repr(xf)
end
_jnum(::Nothing) = "null"
_jbool(b::Bool) = b ? "true" : "false"
_jstr(s) = "\"" * replace(String(s), "\"" => "\\\"") * "\""

function write_json(path::AbstractString, d::AbstractDict)
    open(path, "w") do io
        println(io, "{")
        ks = collect(keys(d))
        for (i, k) in enumerate(ks)
            v = d[k]
            print(io, "  ", _jstr(k), ": ")
            _write_json_value(io, v)
            println(io, i == length(ks) ? "" : ",")
        end
        println(io, "}")
    end
end

function _write_json_value(io, v)
    if v isa AbstractString || v isa Symbol
        print(io, _jstr(v))
    elseif v isa Bool
        print(io, _jbool(v))
    elseif v isa Integer
        print(io, string(v))
    elseif v isa Real
        print(io, _jnum(v))
    elseif v isa Nothing || v isa Missing
        print(io, "null")
    elseif v isa AbstractVector
        print(io, "[")
        for (i, x) in enumerate(v)
            _write_json_value(io, x)
            i < length(v) && print(io, ", ")
        end
        print(io, "]")
    elseif v isa AbstractDict
        print(io, "{")
        dk = collect(keys(v))
        for (i, k) in enumerate(dk)
            print(io, _jstr(k), ": ")
            _write_json_value(io, v[k])
            i < length(dk) && print(io, ", ")
        end
        print(io, "}")
    else
        print(io, _jstr(string(v)))
    end
end
