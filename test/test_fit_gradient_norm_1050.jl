using GLLVModels, Test, Random, Distributions

# Issue 1050: BinomialFit and the four ordinal fit types store `gradient_norm`, the max-abs
# gradient of the optimiser objective at the returned optimum (Optim.g_residual), as the
# Gaussian-sources, temporal, precision and grouped non-Gaussian fit types already do.
# Informational only: `converged`, `_fit_verdict`, defaults and tolerances are unchanged.
# Assertions are relations (finite, non-negative, small where converged, agrees with an
# independent central-difference gradient), not pinned values, because Optim's
# finite-difference L-BFGS path is not bit-reproducible across BLAS builds.

const _GN_GTOL = 1e-5    # the fitters' default g_tol

# Central-difference max-abs gradient of `nll` at `θ`.
function _gn_fd_maxgrad(nll, θ; h = 1e-5)
    g = map(eachindex(θ)) do i
        e = zeros(length(θ)); e[i] = h
        (nll(θ .+ e) - nll(θ .- e)) / (2h)
    end
    return maximum(abs, g)
end

function _gn_binomial_data(seed; p = 4, n = 300)
    rng = MersenneTwister(seed)
    Λ = [0.9, -0.6, 0.5, 0.8][1:p]
    z = randn(rng, n)
    η = 0.2 .+ Λ * z'
    return Float64.(rand(rng, p, n) .< 1 ./ (1 .+ exp.(-η)))
end

function _gn_ordinal_data(seed; p = 5, n = 300, C = 3)
    rng = MersenneTwister(seed)
    τ = [-0.6, 0.7]
    Λ = [0.9, -0.6, 0.5, 0.8, 0.7][1:p]
    η = Λ * randn(rng, n)'
    Y = Matrix{Int}(undef, p, n)
    for i in 1:n, t in 1:p
        pr = [GLLVModels._ord_prob(c, η[t, i], τ) for c in 1:C]
        Y[t, i] = rand(rng, Categorical(pr))
    end
    return Y
end

function _gn_check_common(fit; fd = nothing)
    gn = fit.gradient_norm
    @test hasfield(typeof(fit), :gradient_norm)
    @test gn isa Float64
    @test isfinite(gn)
    @test gn >= 0
    @test fit.converged
    @test gn <= 10 * _GN_GTOL
    fd === nothing || @test abs(gn - fd) <= 1e-4
end

@testset "gradient_norm on five fit types (#1050)" begin
    @testset "BinomialFit (Laplace, no X_lv)" begin
        Y = _gn_binomial_data(1050)
        p, n = size(Y); K = 1
        f = fit_gllvm(Y; family = Binomial(), K = K)
        @test f isa BinomialFit
        nll(θ) = -GLLVModels.binomial_marginal_loglik_laplace(Y, ones(Int, p, n),
                    GLLVModels.unpack_lambda(θ[(p + 1):end], p, K), θ[1:p], f.link;
                    hessian = f.hessian)
        θ = vcat(f.β, GLLVModels.pack_lambda(f.Λ))
        _gn_check_common(f; fd = _gn_fd_maxgrad(nll, θ))
    end

    @testset "BinomialFit (Laplace, X_lv route)" begin
        Y = _gn_binomial_data(1051)
        n = size(Y, 2)
        X_lv = reshape(randn(MersenneTwister(7), n), n, 1)
        f = fit_binomial_gllvm(Y; K = 1, X_lv = X_lv)
        @test f isa BinomialFit
        _gn_check_common(f)
    end

    @testset "BinomialFit (variational route)" begin
        Y = _gn_binomial_data(1052)
        f = GLLVModels.fit_binomial_gllvm_va(Int.(Y); K = 1)
        @test f isa BinomialFit
        @test isfinite(f.gradient_norm) && f.gradient_norm >= 0
        f.converged && @test f.gradient_norm <= 10 * _GN_GTOL
    end

    @testset "OrdinalFit (shared cutpoints)" begin
        Y = _gn_ordinal_data(1053)
        p, n = size(Y); K = 1; C = 3
        f = fit_ordinal_gllvm(Y; K = K)
        @test f isa OrdinalFit
        ψ = [f.τ[1]; log.(diff(f.τ))]
        nll(θ) = -GLLVModels.ordinal_marginal_loglik_laplace(Y,
                    GLLVModels.unpack_lambda(θ[1:(end - C + 1)], p, K),
                    GLLVModels._unpack_cutpoints(θ[(end - C + 2):end]))
        θ = vcat(GLLVModels.pack_lambda(f.Λ), ψ)
        _gn_check_common(f; fd = _gn_fd_maxgrad(nll, θ))
    end

    @testset "OrdinalPerTraitFit (what fit_gllvm returns)" begin
        Y = _gn_ordinal_data(1054)
        p, n = size(Y); K = 1
        f = fit_gllvm(Y; family = Ordinal(), K = K)
        @test f isa OrdinalPerTraitFit
        rr = GLLVModels.rr_theta_len(p, K)
        nll(θ) = -GLLVModels.ordinal_marginal_loglik_laplace_pertrait(Y,
                    GLLVModels.unpack_lambda(θ[(p + 1):(p + rr)], p, K), θ[1:p],
                    GLLVModels._unpack_cutpoints_pertrait(θ[(p + rr + 1):end], f.C), f.C)
        θ = vcat(f.β, GLLVModels.pack_lambda(f.Λ), GLLVModels._ord_psi_from_tau(f.τ, f.C))
        _gn_check_common(f; fd = _gn_fd_maxgrad(nll, θ))
    end

    @testset "OrdinalPerTraitCovFit" begin
        Y = _gn_ordinal_data(1055)
        p, n = size(Y)
        X = reshape(randn(MersenneTwister(8), p * n), p, n, 1)
        f = fit_ordinal_gllvm_pertrait_cov(Y; X = X, K = 1)
        @test f isa OrdinalPerTraitCovFit
        _gn_check_common(f)
    end

    @testset "OrdinalPerTraitAGHQFit" begin
        Y = _gn_ordinal_data(1056)
        # aghq = 1 is the Laplace rule: the AGHQ type wraps the Laplace fit and must
        # carry its gradient_norm over.
        lap = fit_ordinal_gllvm_pertrait(Y; K = 1)
        f = GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y; K = 1, aghq = 1)
        @test f isa GLLVModels.OrdinalPerTraitAGHQFit
        @test f.integration.actual === :laplace
        _gn_check_common(f)
        @test f.gradient_norm ≈ lap.gradient_norm atol = 1e-6
        # A genuine AGHQ fit does not record the AGHQ outer gradient: NaN, by design.
        g = GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y; K = 1, aghq = 3, iterations = 2)
        @test g.integration.actual === :aghq
        @test isnan(g.gradient_norm)
    end

    @testset "BinomialFit AGHQ: re-wrap keeps the field, genuine AGHQ is NaN" begin
        Y = _gn_binomial_data(1057)
        N = fill(1, size(Y)...)
        g = fit_binomial_gllvm(Y; K = 1, N = N, iterations = 2, aghq = 3)
        @test g.integration.actual === :aghq
        @test isnan(g.gradient_norm)
        base = fit_binomial_gllvm(Y; K = 1, N = N)
        w = GLLVModels._binomial_with_integration(base, g.integration)
        @test w.gradient_norm === base.gradient_norm
    end

    @testset "old positional constructors default gradient_norm to NaN" begin
        old = GLLVModels.BinomialFit([0.0], zeros(1, 1), GLLVModels.LogitLink(), -1.0, true, 3)
        @test isnan(old.gradient_norm)
        o12 = GLLVModels.BinomialFit([0.0], zeros(1, 1), GLLVModels.LogitLink(), -1.0, true, 3,
                                     nothing, Float64[], :observed, nothing, nothing, Inf)
        @test isnan(o12.gradient_norm)
        oo = OrdinalFit(zeros(1, 1), [0.0], 2, GLLVModels.LogitLink(), -1.0, true, 3)
        @test isnan(oo.gradient_norm)
    end
end
