#!/usr/bin/env julia
# core070_realistic_size_cell.jl -- T4 realistic-size grid, Julia side.
# One cell = one (family, p, n, K, seed) combination. Self-contained: builds
# its own DGP (no dependency on test/parity fixtures, which are toy-sized),
# fits, computes second-order quantities (SE, fixed-effect vcov block, Wald
# endpoints, cond(H)) via the SAME technique as
# docs/dev-log/core070/second-order-prerun-2026-09-02.md (family-generic
# confint() for the per-term SE/pd_hessian/boundary_terms, plus a private-API
# rebuild of the marginal NLL for the full covariance block and cond(H)).
#
# Usage: julia --project=. core070_realistic_size_cell.jl <family> <p> <n> <K> <seed>
#   family in {gaussian, poisson, nb2, binomial}
#
# Writes to ./out/<family>_p<p>_n<n>_K<K>_julia_summary.txt (+ _terms.csv,
# _vcov_beta.csv). Also emits the shared CSV Y matrix to ./data/ so the
# paired R run (Totoro background, or a future Nibi run) reads
# byte-identical data.
using GLLVModels
using Random
using DelimitedFiles
using LinearAlgebra
using ForwardDiff

length(ARGS) >= 5 || error("usage: core070_realistic_size_cell.jl <family> <p> <n> <K> <seed>")
fam  = ARGS[1]
p    = parse(Int, ARGS[2])
n    = parse(Int, ARGS[3])
K    = parse(Int, ARGS[4])
seed = parse(Int, ARGS[5])

mkpath("out"); mkpath("data")
tag = "$(fam)_p$(p)_n$(n)_K$(K)"
datapath = joinpath("data", "$(tag).csv")

function write_mat(path, M)
    open(path, "w") do io
        for i in 1:size(M, 1)
            println(io, join(M[i, :], ","))
        end
    end
end

function write_terms(path, term, estimate, se, lower, upper)
    open(path, "w") do io
        println(io, "term,estimate,se,lower,upper")
        for i in eachindex(term)
            println(io, "$(term[i]),$(estimate[i]),$(se[i]),$(lower[i]),$(upper[i])")
        end
    end
end

# ---------------------------------------------------------------------
# DGP: p traits x n sites, K latent factors. Loadings + coefficients are
# deterministic functions of (p, K, seed) -- reproducible, not ported from
# any toy fixture (those are p<=5; this grid needs p in {20,50}).
# ---------------------------------------------------------------------
Random.seed!(seed)
Λ_true = 0.35 .* randn(p, K)
β_log  = log.(2.0 .+ 3.0 .* rand(p))   # trait intercepts, log-scale mean 2-5
Z      = randn(K, n)
η      = β_log .+ Λ_true * Z

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
function _rand_gamma(shape::Float64, scale::Float64)
    if shape < 1.0
        return _rand_gamma(shape + 1.0, scale) * rand()^(1.0 / shape)
    end
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn()
        v = (1.0 + c * x)^3
        v <= 0 && continue
        u = rand()
        x2 = x * x
        u < 1.0 - 0.0331 * x2 * x2 && return d * v * scale
        logu = log(u)
        logu < 0.5 * x2 + d * (1.0 - v + log(v)) && return d * v * scale
    end
end
_rand_nb2(μ::Float64, r::Float64) = _rand_poisson(_rand_gamma(r, μ / r))

summary_lines = String[]
push!(summary_lines, "family=$fam"); push!(summary_lines, "p=$p n=$n K=$K seed=$seed")

if fam == "gaussian"
    σ_true = 0.7
    Y = Λ_true * Z .+ σ_true .* randn(p, n)
    Y .-= sum(Y; dims = 2) ./ n
    write_mat(datapath, Y)
    if length(ARGS) >= 6 && ARGS[6] == "data-only"
        println("DATA-ONLY $tag written to $datapath")
        exit(0)
    end

    # R pairs per-trait intercept SEs (`0 + trait`); Julia's default X=nothing is
    # zero-mean. On row-centred Y the intercept path is logLik-identical but
    # exposes beta[1:p] in confint (see test_fixed_effects.jl).
    X = zeros(p, n, p)
    for t in 1:p
        X[t, :, t] .= 1.0
    end

    t0 = time(); fit = fit_gaussian_gllvm(Y; K = K, X = X); wall_fit = time() - t0
    push!(summary_lines, "converged=$(fit.converged)")
    push!(summary_lines, "logLik=$(fit.logLik)")
    push!(summary_lines, "wall_fit_sec=$(wall_fit)")

    t1 = time(); ci = confint(fit, Y; X = X); wall_ci = time() - t1
    push!(summary_lines, "pd_hessian=$(ci.pd_hessian)")
    push!(summary_lines, "wall_confint_sec=$(wall_ci)")
    boundary_terms = hasproperty(ci, :boundary_terms) ? ci.boundary_terms : String[]
    push!(summary_lines, "dispersion_boundary=NA (Gaussian has no grouped dispersion)")
    push!(summary_lines, "boundary_terms=$(join(boundary_terms, ';'))")
    write_terms(joinpath("out", "$(tag)_julia_terms.csv"), ci.term, ci.estimate, ci.se, ci.lower, ci.upper)

    θ̂ = fit.pars.θ_packed
    nll = GLLVModels._confint_reconstruct_nll(fit, Y, X, nothing)
    H = ForwardDiff.hessian(nll, θ̂)
    Hs = Symmetric((H .+ H') ./ 2)
    push!(summary_lines, "cond_H=$(cond(Hs))")
    Σ = try
        inv(Hs)
    catch e
        push!(summary_lines, "vcov_inversion_error=$(sprint(showerror, e))")
        fill(NaN, size(H))
    end
    write_mat(joinpath("out", "$(tag)_julia_vcov_full.csv"), Σ)
    write_mat(joinpath("out", "$(tag)_julia_vcov_beta.csv"), Σ[1:p, 1:p])

elseif fam == "poisson"
    Y = [_rand_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]
    write_mat(datapath, Y)
    if length(ARGS) >= 6 && ARGS[6] == "data-only"
        println("DATA-ONLY $tag written to $datapath")
        exit(0)
    end
    Yi = Y

    t0 = time(); fit = fit_poisson_gllvm(Yi; K = K, hessian = :observed); wall_fit = time() - t0
    push!(summary_lines, "converged=$(fit.converged)")
    push!(summary_lines, "logLik=$(fit.loglik)")
    push!(summary_lines, "wall_fit_sec=$(wall_fit)")

    t1 = time(); ci = confint(fit, Yi); wall_ci = time() - t1
    push!(summary_lines, "pd_hessian=$(ci.pd_hessian)")
    push!(summary_lines, "wall_confint_sec=$(wall_ci)")
    boundary_terms = hasproperty(ci, :boundary_terms) ? ci.boundary_terms : String[]
    push!(summary_lines, "dispersion_boundary=NA (Poisson has no dispersion parameter)")
    push!(summary_lines, "boundary_terms=$(join(boundary_terms, ';'))")
    write_terms(joinpath("out", "$(tag)_julia_terms.csv"), ci.term, ci.estimate, ci.se, ci.lower, ci.upper)

    ad = GLLVModels._family_ci(fit, Yi; objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Hs = Symmetric((H .+ H') ./ 2)
    push!(summary_lines, "cond_H=$(cond(Hs))")
    Σ = try
        inv(Hs)
    catch e
        push!(summary_lines, "vcov_inversion_error=$(sprint(showerror, e))")
        fill(NaN, size(H))
    end
    write_mat(joinpath("out", "$(tag)_julia_vcov_beta.csv"), Σ[1:p, 1:p])

elseif fam == "binomial"
    # Bernoulli trials (N=1), logit link — matches gllvmTMB's `stats::binomial()`
    # default on a 0/1 response, and fit_binomial_gllvm's default N = fill(1,p,n).
    # The shared β_log/η above is a count-family rate-scale intercept
    # (log(2..5) ≈ 0.69..1.61) and is NOT a sensible logit-scale intercept: it
    # puts baseline prevalence at 0.66-0.83 rather than centered near 0.5,
    # which (measured) drove one trait into quasi-complete separation. Build a
    # zero-centered logit intercept instead, reusing the same Λ_true/Z latent
    # draw (RNG state is deterministic given `seed`, so this stays reproducible).
    β_bin = 0.4 .* randn(p)
    η_bin = β_bin .+ Λ_true * Z
    P = 1.0 ./ (1.0 .+ exp.(-clamp.(η_bin, -8.0, 8.0)))
    Y = Int.(rand(p, n) .< P)
    write_mat(datapath, Y)
    if length(ARGS) >= 6 && ARGS[6] == "data-only"
        println("DATA-ONLY $tag written to $datapath")
        exit(0)
    end
    Yi = Y

    t0 = time(); fit = fit_binomial_gllvm(Yi; K = K); wall_fit = time() - t0
    push!(summary_lines, "converged=$(fit.converged)")
    push!(summary_lines, "logLik=$(fit.loglik)")
    push!(summary_lines, "wall_fit_sec=$(wall_fit)")

    t1 = time(); ci = confint(fit, Yi); wall_ci = time() - t1
    push!(summary_lines, "pd_hessian=$(ci.pd_hessian)")
    push!(summary_lines, "wall_confint_sec=$(wall_ci)")
    boundary_terms = hasproperty(ci, :boundary_terms) ? ci.boundary_terms : String[]
    push!(summary_lines, "dispersion_boundary=NA (Binomial has no dispersion parameter)")
    push!(summary_lines, "boundary_terms=$(join(boundary_terms, ';'))")
    write_terms(joinpath("out", "$(tag)_julia_terms.csv"), ci.term, ci.estimate, ci.se, ci.lower, ci.upper)

    ad = GLLVModels._family_ci(fit, Yi; objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Hs = Symmetric((H .+ H') ./ 2)
    push!(summary_lines, "cond_H=$(cond(Hs))")
    Σ = try
        inv(Hs)
    catch e
        push!(summary_lines, "vcov_inversion_error=$(sprint(showerror, e))")
        fill(NaN, size(H))
    end
    write_mat(joinpath("out", "$(tag)_julia_vcov_beta.csv"), Σ[1:p, 1:p])

elseif fam == "nb2"
    r_true = 3.0 .+ 2.0 .* rand(p)   # per-trait dispersion (grouped by species)
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -8.0, 8.0))
        Y[t, s] = _rand_nb2(μ, r_true[t])
    end
    write_mat(datapath, Y)
    if length(ARGS) >= 6 && ARGS[6] == "data-only"
        println("DATA-ONLY $tag written to $datapath")
        exit(0)
    end
    Yi = Y

    t0 = time()
    fit = fit_gllvm(Yi; family = GLLVModels.NegativeBinomial(), K = K, g_tol = 1e-7, iterations = 800)
    wall_fit = time() - t0
    push!(summary_lines, "converged=$(fit.converged)")
    push!(summary_lines, "logLik=$(fit.loglik)")
    push!(summary_lines, "hessian_selector=$(fit.hessian) (grouped default)")
    push!(summary_lines, "wall_fit_sec=$(wall_fit)")
    push!(summary_lines, "dispersion_boundary=$(fit.dispersion_boundary)")

    t1 = time(); ci = confint(fit, Yi); wall_ci = time() - t1
    push!(summary_lines, "pd_hessian=$(ci.pd_hessian)")
    push!(summary_lines, "wall_confint_sec=$(wall_ci)")
    boundary_terms = hasproperty(ci, :boundary_terms) ? ci.boundary_terms : String[]
    push!(summary_lines, "boundary_terms=$(join(boundary_terms, ';'))")
    write_terms(joinpath("out", "$(tag)_julia_terms.csv"), ci.term, ci.estimate, ci.se, ci.lower, ci.upper)

    ad = GLLVModels._family_ci(fit, Yi; objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Hs = Symmetric((H .+ H') ./ 2)
    push!(summary_lines, "cond_H=$(cond(Hs))")
    Σ = try
        inv(Hs)
    catch e
        push!(summary_lines, "vcov_inversion_error=$(sprint(showerror, e))")
        fill(NaN, size(H))
    end
    write_mat(joinpath("out", "$(tag)_julia_vcov_beta.csv"), Σ[1:p, 1:p])
else
    error("unknown family $fam (expected gaussian|poisson|nb2|binomial)")
end

open(joinpath("out", "$(tag)_julia_summary.txt"), "w") do io
    for l in summary_lines; println(io, l); end
end
println("DONE $tag")
for l in summary_lines; println(l); end
