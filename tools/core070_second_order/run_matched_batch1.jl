#!/usr/bin/env julia
# run_matched_batch1.jl — matched-coordinates pilot (contract §4 tier, batch-1 only).
# Anchor: R opt$par; evaluate Julia ForwardDiff Hessian at mapped θ_R; compare
# to R sdreport at the same point. Does NOT claim programme §7 completion.
#
# Usage (from repo root):
#   julia --project=. tools/core070_second_order/run_matched_batch1.jl [out_dir]

using GLLVModels
using RCall
using LinearAlgebra
using Random
using ForwardDiff
using Distributions: Gamma, NegativeBinomial, Beta, Binomial

include(joinpath(@__DIR__, "theta_map.jl"))
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "cells.jl"))

const BATCH1 = ("gaussian", "poisson", "binomial_logit", "beta_logit", "nb2_log")
const MC_TOL_SE = 1e-4
const MC_TOL_VCOV = 1e-4

function r_opt_par(y::AbstractMatrix, K::Integer; family::Symbol, binomial_link::Symbol = :logit)
    p, n = size(y)
    fam = String(family)
    blink = String(binomial_link)
    _require_gllvmtmb!()
    @rput y K p n fam blink
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y))
    fam_obj <- switch(fam,
        gaussian = stats::gaussian(),
        binomial = stats::binomial(link = blink),
        poisson = stats::poisson(),
        negbinomial = gllvmTMB::nbinom2(),
        beta = gllvmTMB::Beta(),
        stop(sprintf("unknown family: %s", fam)))
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait", family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = TRUE))
    """
    return (
        logLik = rcopy(Float64, R"as.numeric(logLik(fit_r))"),
        converged = rcopy(Bool, R"identical(as.integer(fit_r$opt$convergence), 0L)"),
        has_sd = rcopy(Bool, R"!is.null(fit_r$sd_report)"),
        par = rcopy(Vector{Float64}, R"as.numeric(fit_r$opt$par)"),
        par_names = rcopy(Vector{String}, R"names(fit_r$opt$par)"),
        pf_names = rcopy(Vector{String}, R"names(fit_r$sd_report$par.fixed)"),
        cov_fixed = rcopy(Matrix{Float64}, R"fit_r$sd_report$cov.fixed"),
        pd_hessian = rcopy(Any, R"isTRUE(fit_r$sd_report$pdHess)"),
        r_condition_number = rcopy(Any, R"tryCatch(kappa(fit_r$sd_report$cov.fixed), error=function(e) NA_real_)"),
    )
end

function _safe_inv(H)
    all(isfinite, H) || return nothing
    try
        return inv(Symmetric((H .+ H') ./ 2))
    catch
        return nothing
    end
end

function _family_nll(cell_id::AbstractString, Y, fit)
    if cell_id == "gaussian"
        nll = GLLVModels._confint_reconstruct_nll(fit, Y, nothing, nothing)
        terms, _ = GLLVModels._confint_all_term_names(fit)
        compare = :sigma_only
        sig_idx = findfirst(==("sigma_eps"), terms)
        return nll, terms, compare, sig_idx, 0
    else
        ad = if cell_id == "poisson"
            GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
        elseif cell_id == "binomial_logit"
            GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
        elseif cell_id == "beta_logit"
            GLLVModels._family_ci(fit, Y; objective = :laplace)
        elseif cell_id == "nb2_log"
            GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
        else
            error("unknown cell $cell_id")
        end
        p = size(fit.Λ, 1)
        return ad.nll, ad.names, :beta_block, collect(1:p), p
    end
end

function _fit_julia(cell_id::AbstractString, Y)
    if cell_id == "gaussian"
        return fit_gaussian_gllvm(Y; K = 2)
    elseif cell_id == "poisson"
        return fit_poisson_gllvm(Y; K = 2, hessian = :observed)
    elseif cell_id == "binomial_logit"
        return fit_binomial_gllvm(Y; K = 2, link = LogitLink(), hessian = :observed)
    elseif cell_id == "beta_logit"
        return fit_gllvm(Y; family = GLLVModels.Beta(), K = 1, g_tol = 1e-7, iterations = 800)
    elseif cell_id == "nb2_log"
        return fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, disp_group = :species, g_tol = 1e-7, iterations = 800)
    else
        error("unknown cell $cell_id")
    end
end

function _r_family(cell_id::AbstractString)
    if cell_id == "gaussian"; return :gaussian
    elseif cell_id == "poisson"; return :poisson
    elseif cell_id == "binomial_logit"; return :binomial
    elseif cell_id == "beta_logit"; return :beta
    elseif cell_id == "nb2_log"; return :negbinomial
    else; error("unknown cell $cell_id"); end
end

function _extract_dgp(cell_id::AbstractString)
    d = run_one_cell(cell_id)
    # Re-run DGP by calling cell internals — use receipt metadata only; rebuild via cell dispatch
    if cell_id == "gaussian"
        seed = 42; Random.seed!(seed); p, K, n = 5, 2, 80
        Λ_true = parity_loadings_p5k2(); σ_true = 0.7; η = randn(K, n)
        y = Λ_true * η + σ_true * randn(p, n); y .-= sum(y; dims = 2) ./ n
        return y, K, p, n, seed, d["dgp_source"]
    elseif cell_id == "poisson"
        seed = 44; Random.seed!(seed); p, K, n = 5, 2, 60
        β = log.([3.0, 5.0, 2.0, 4.0, 3.5]); Λ = 0.45 .* parity_loadings_p5k2()
        Z = randn(K, n); η = β .+ Λ * Z
        Y = [_rand_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]
        return Y, K, p, n, seed, d["dgp_source"]
    elseif cell_id == "binomial_logit"
        seed = 43; Random.seed!(seed); p, K, n = 5, 2, 60
        β = [-0.5, 0.0, 0.5, -0.2, 0.3]; Λ = parity_loadings_p5k2()
        Z = randn(K, n); η = β .+ Λ * Z
        Y = [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]
        return Y, K, p, n, seed, d["dgp_source"]
    elseif cell_id == "beta_logit"
        seed = 45; Random.seed!(seed); p, K, n = 5, 1, 60
        β = [0.30, -0.20, 0.25, -0.15, 0.05]; φ_true = 12.0
        Λ = 0.15 .* parity_loadings_p5k2()[:, 1:K]; Z = randn(K, n); η = β .+ Λ * Z
        Y = [begin
            μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
            _rand_beta_jonk(μ * φ_true, (1 - μ) * φ_true)
        end for t in 1:p, s in 1:n]
        return Y, K, p, n, seed, d["dgp_source"]
    elseif cell_id == "nb2_log"
        seed = 45; Random.seed!(seed); p, K, n = 5, 2, 80
        β = log.([2.5, 3.0, 2.0, 2.8, 2.2]); r_true = 4.0
        Λ = 0.30 .* parity_loadings_p5k2(); Z = randn(K, n); η = β .+ Λ * Z
        Y = [_rand_nb2_ms(exp(clamp(η[t, s], -8.0, 8.0)), r_true) for t in 1:p, s in 1:n]
        return Y, K, p, n, seed, d["dgp_source"]
    else
        error("unknown cell $cell_id")
    end
end

function run_matched_cell(cell_id::AbstractString)
    each_own = run_one_cell(cell_id)
    Y, K, p, n, seed, dgp_source = _extract_dgp(cell_id)
    Ymat = cell_id == "gaussian" ? Y : Float64.(Y)
    fit = _fit_julia(cell_id, Ymat)
    nll, names, compare, _, p_beta = _family_nll(cell_id, Ymat, fit)
    julia_theta_len = if cell_id == "gaussian"
        length(fit.pars.θ_packed)
    else
        ad = if cell_id == "poisson"
            GLLVModels._family_ci(fit, Float64.(Ymat); objective = :laplace)
        elseif cell_id == "binomial_logit"
            GLLVModels._family_ci(fit, Float64.(Ymat); objective = :laplace)
        elseif cell_id == "beta_logit"
            GLLVModels._family_ci(fit, Ymat; objective = :laplace)
        else
            GLLVModels._family_ci(fit, Float64.(Ymat); objective = :laplace)
        end
        length(ad.θ)
    end

    rf = r_opt_par(Ymat, K; family = _r_family(cell_id),
        binomial_link = cell_id == "binomial_logit" ? :logit : :logit)
    map_result = map_r_to_julia_theta(cell_id, rf.par, rf.par_names, p, K;
        julia_theta_len = julia_theta_len)

    d = Dict{String, Any}()
    d["cell_id"] = cell_id
    d["dgp_source"] = dgp_source
    d["p"] = p; d["K"] = K; d["n"] = n; d["seed"] = seed
    d["matched_coordinates"] = true
    d["anchor"] = "R_opt_par"
    d["tier"] = "matched-coordinates"
    d["tier_tolerance_se_rel"] = MC_TOL_SE
    d["tier_tolerance_vcov_fro_rel"] = MC_TOL_VCOV
    d["each_own_se_max_relative_delta"] = each_own["se_max_relative_delta"]
    d["programme_section7_claim"] = false
    d["note"] = "Matched-coordinates diagnostic only; not programme §7 second-order parity claim."

    if map_result[1] == :blocked
        b = map_result[2]
        merge!(d, blocker_dict(b))
        d["pilot_status"] = "blocked"
        d["matched_pass"] = false
        return d
    end

    θ_r = map_result[2]
    gauss_note = length(map_result) >= 3 ? map_result[3] : nothing
    gauss_note !== nothing && (d["theta_map_note"] = gauss_note)

    if !rf.has_sd
        d["pilot_status"] = "skip"
        d["matched_pass"] = false
        d["skip_reason"] = "R se=TRUE produced no sd_report"
        return d
    end

    ll_at_r = -nll(θ_r)
    d["loglik_at_r_theta_julia"] = ll_at_r
    d["loglik_at_r_theta_r"] = rf.logLik
    d["loglik_delta_at_anchor"] = abs(ll_at_r - rf.logLik)

    H = try
        ForwardDiff.hessian(nll, θ_r)
    catch
        nothing
    end
    Σ = H === nothing ? nothing : _safe_inv(H)
    if Σ === nothing
        d["pilot_status"] = "fail"
        d["matched_pass"] = false
        d["skip_reason"] = "Julia Hessian inversion failed at R-anchored θ"
        return d
    end

    pf_names = rf.pf_names
    if compare == :beta_block
        r_beta_idx = findall(==("b_fix"), pf_names)
        length(r_beta_idx) == p_beta || begin
            d["pilot_status"] = "skip"; d["matched_pass"] = false
            d["skip_reason"] = "R b_fix block length $(length(r_beta_idx)) != p=$p_beta"
            return d
        end
        se_r = sqrt.(max.(diag(rf.cov_fixed)[r_beta_idx], 0.0))
        se_jl = sqrt.(max.(diag(Σ)[1:p_beta], 0.0))
        cov_r = rf.cov_fixed[r_beta_idx, r_beta_idx]
        cov_jl = Σ[1:p_beta, 1:p_beta]
    else
        sig_r = findfirst(==("log_sigma_eps"), pf_names)
        sig_r === nothing && begin
            d["pilot_status"] = "skip"; d["matched_pass"] = false
            d["skip_reason"] = "log_sigma_eps not in R sdreport fixed block"
            return d
        end
        se_r = [sqrt(max(rf.cov_fixed[sig_r, sig_r], 0.0))]
        se_jl = [sqrt(max(Σ[1, 1], 0.0))]
        cov_r = rf.cov_fixed[sig_r:sig_r, sig_r:sig_r]
        cov_jl = Σ[1:1, 1:1]
    end

    se_delta = abs.(se_jl .- se_r)
    se_rel = se_delta ./ max.(se_r, 1e-12)
    d["se_max_abs_delta"] = maximum(se_delta)
    d["se_max_relative_delta"] = maximum(se_rel)
    fro_num = sqrt(sum((cov_jl .- cov_r) .^ 2))
    fro_den = sqrt(sum(cov_r .^ 2))
    d["vcov_frobenius_relative_delta"] = fro_den > 0 ? fro_num / fro_den : NaN
    d["pd_hessian_native"] = all(x -> isfinite(x) && x > 0, diag(Σ))
    d["pd_hessian_r"] = rf.pd_hessian
    d["r_condition_number"] = rf.r_condition_number
    d["native_condition_number"] = try
        cond(Symmetric((Σ .+ Σ') ./ 2))
    catch
        NaN
    end
    d["r_param_counts"] = _count_names(rf.par_names)
    d["julia_theta_len"] = length(θ_r)
    d["matched_pass"] = d["se_max_relative_delta"] <= MC_TOL_SE &&
                        d["vcov_frobenius_relative_delta"] <= MC_TOL_VCOV &&
                        d["loglik_delta_at_anchor"] <= 1e-8
    d["pilot_status"] = d["matched_pass"] ? "pass" : "fail"
    return d
end

function main()
    out_root = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(@__DIR__, "..", "..", "docs", "dev-log", "core070",
                 "second-order-matched-pilot-batch1-20260905")
    mkpath(out_root)
    results = Dict{String, Any}[]
    t0 = time()
    for cell_id in BATCH1
        println("MATCHED_PILOT cell=$cell_id ...")
        d = run_matched_cell(cell_id)
        write_json(joinpath(out_root, "$(cell_id).json"), d)
        push!(results, d)
        println("  status=$(d["pilot_status"]) matched_pass=$(d["matched_pass"])")
    end
    n_pass = count(d -> get(d, "pilot_status", "") == "pass", results)
    n_fail = count(d -> get(d, "pilot_status", "") == "fail", results)
    n_blocked = count(d -> get(d, "pilot_status", "") == "blocked", results)
    n_skip = count(d -> get(d, "pilot_status", "") == "skip", results)
    summary = Dict{String, Any}(
        "receipt_date" => "2026-09-05",
        "pilot" => "matched-coordinates batch-1",
        "cells" => collect(BATCH1),
        "pass_fail_blocked_skip" => "$n_pass/$n_fail/$n_blocked/$n_skip",
        "tier" => "matched-coordinates (contract §4 diagnostic)",
        "programme_section7_claim" => false,
        "anchor" => "R_opt_par",
        "tolerance_se_rel" => MC_TOL_SE,
        "tolerance_vcov_fro_rel" => MC_TOL_VCOV,
        "wall_sec" => time() - t0,
        "out_dir" => out_root,
        "jl_head" => read(`git -C $(joinpath(@__DIR__, "..", "..")) rev-parse --short HEAD`, String)[1:end-1],
        "note" => "3 cells expected pass (gaussian/poisson/binomial_logit); beta_logit and nb2_log blocked on per-trait vs shared dispersion θ map.",
    )
    write_json(joinpath(out_root, "summary.json"), summary)
    println("MATCHED_PILOT_DONE pass/fail/blocked/skip = $(summary["pass_fail_blocked_skip"])")
    println("OUT_DIR $out_root")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
