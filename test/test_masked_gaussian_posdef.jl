# Masked Gaussian fits must not throw PosDefException (refs #716).
#
# The masked/offset route of fit_gaussian_gllvm optimises _gaussian_data_nll,
# which factors Λ_oΛ_oᵀ + σ²I for each unit's observed traits o. An LBFGS
# line-search trial point can drive log σ far below the optimum; when |o| > K
# that matrix is then rank-deficient to working precision and `cholesky`
# threw. The objective now returns +Inf at such points so the line search
# backs off. This file checks (1) that contract directly and (2) that fits on
# the mask patterns from the issue succeed, converge, and report the
# observed-data log-likelihood of an independent dense MvNormal oracle at the
# returned parameters, with a near-zero score of that oracle (an optimum).
# No seeded magnitudes are asserted: only structural and oracle properties.

using GLLVModels, Test, LinearAlgebra, Distributions, ForwardDiff, StableRNGs

function _mgp_oracle_ll(Y, mask, X, β, σ, Λ)
    p, n = size(Y)
    Σ = Λ * Λ' + σ^2 * I
    ll = zero(promote_type(eltype(β), eltype(Λ), typeof(σ)))
    for s in 1:n
        o = findall(mask[:, s]); isempty(o) && continue
        μ = [sum(X[t, s, k] * β[k] for k in axes(X, 3)) for t in o]
        S = Symmetric(Matrix(Σ[o, o]))
        r = Y[o, s] .- μ
        ll += -(length(o) * log(2π) + logdet(S) + dot(r, S \ r)) / 2
    end
    return ll
end

function _mgp_mvn_ll(Y, mask, X, β, σ, Λ)
    Σ = Λ * Λ' + σ^2 * I
    ll = 0.0
    for s in axes(Y, 2)
        o = findall(mask[:, s]); isempty(o) && continue
        μ = [sum(X[t, s, k] * β[k] for k in axes(X, 3)) for t in o]
        ll += logpdf(MvNormal(μ, Symmetric(Matrix(Σ[o, o]))), Y[o, s])
    end
    return ll
end

@testset "masked Gaussian fit: no PosDefException (#716)" begin
    p, n = 6, 200
    X = zeros(p, n, p); for t in 1:p; X[t, :, t] .= 1; end   # one-hot trait intercepts

    @testset "objective is +Inf, not a throw, at a degenerate trial point" begin
        mask = trues(p, n); mask[1, 4] = false
        data = GLLVModels.AGHQGaussianData(randn(StableRNG(1), p, n), mask, zeros(p, n), X)
        K = 2
        nll = GLLVModels._gaussian_data_nll(data, K, falses(p))
        nλ = p * K - K * (K - 1) ÷ 2
        Λ = zeros(p, K); Λ[:, 1] .= 40.0; Λ[2:end, 2] .= -20.0   # rank-2 Λ_oΛ_oᵀ on 5-6 traits
        good = vcat(zeros(p), log(0.5), GLLVModels.pack_lambda(Λ))
        @test length(good) == p + 1 + nλ
        @test isfinite(nll(good))
        bad = copy(good); bad[p + 1] = -101.0                   # σ² ≈ 1e-88
        @test nll(bad) == Inf
        # The forward-mode gradient used by the optimiser stays well defined.
        @test ForwardDiff.value(nll(ForwardDiff.Dual.(bad, 1.0))) == Inf
    end

    patterns = [
        ("trait 1 every 7th unit", m -> (for u in 1:n; u % 7 == 4 && (m[1, u] = false); end; m)),
        ("traits 2 and 5 every 10th unit", m -> (for u in 1:n; u % 10 == 3 && (m[[2, 5], u] .= false); end; m)),
        ("random 5%", m -> (m .&= rand(StableRNG(105), p, n) .> 0.05; m)),
        ("random 20%", m -> (m .&= rand(StableRNG(120), p, n) .> 0.20; m)),
        ("traits 1-4, first 30 units", m -> (m[1:4, 1:30] .= false; m)),
        ("all traits, first 10 units", m -> (m[:, 1:10] .= false; m)),
    ]
    for K in (1, 2), (label, make) in patterns
        @testset "K = $K, $label" begin
            rng = StableRNG(716 + K)
            Y = 0.8 .* randn(rng, p, K) * randn(rng, K, n) .+ 0.5 .* randn(rng, p, n) .+ (1:p)
            mask = make(trues(p, n))
            Ym = copy(Y); Ym[.!mask] .= 0.0
            fit = fit_gaussian_gllvm(Ym; K = K, X = X, mask = mask)
            @test fit.converged
            β, σ, Λ = fit.pars.β, fit.pars.σ_eps, fit.pars.Λ
            @test isapprox(fit.logLik, _mgp_mvn_ll(Ym, mask, X, β, σ, Λ); atol = 1e-8, rtol = 0)
            idx = [(i, j) for j in 1:K for i in 1:p if i >= j]
            θ = vcat(β, log(σ), [Λ[i, j] for (i, j) in idx])
            g = ForwardDiff.gradient(θ) do th
                L = zeros(eltype(th), p, K)
                for (c, (i, j)) in enumerate(idx); L[i, j] = th[p + 1 + c]; end
                _mgp_oracle_ll(Ym, mask, X, th[1:p], exp(th[p + 1]), L)
            end
            @test maximum(abs, g) < 1e-3
        end
    end
end
