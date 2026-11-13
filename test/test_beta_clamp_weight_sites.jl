using GLLVModels, Test, Random
const GM = GLLVModels

# The Beta fits clamp y to [1e-12, 1 - 1e-12] in the density and the score (`_clamp_y`, the
# counterpart of gllvmTMB's y_safe). The observed Laplace log-determinant weight read the raw y,
# so a response of 1e-70 gave a different log-marginal than 1e-12 and, below about 1e-60, a
# non-positive-definite site precision and logLik = -Inf. With the clamp at the weight too, any
# response at or below 1e-12 is the same cell as 1e-12 (and any at or above 1 - 1e-12 is the same
# cell as 1 - 1e-12).

# Deterministic table, every response well inside (0, 1).
function _bcw_table()
    rng = MersenneTwister(20261113); p, n = 6, 40
    z = randn(rng, n); λ = 0.5 .+ 0.4 .* rand(rng, p); b = 0.3 .* randn(rng, p)
    Y = [clamp(1 / (1 + exp(-(b[j] + λ[j] * z[i] + 0.6 * randn(rng)))), 0.02, 0.98) for j in 1:p, i in 1:n]
    return Y, λ, b
end
_bcw_with(Y, v) = (Y2 = copy(Y); for c in (5, 17, 29); Y2[c] = v; end; Y2)

@testset "Beta observed weight uses the same y clamp as the density and score" begin
    Y, λ, b = _bcw_table()
    Λ = reshape(λ, :, 1); φ = 10.0
    f = GM.Beta(φ, 1.0); link = GM.LogitLink()
    for η in (-1.0, 0.3, 2.0)
        μ = 1 / (1 + exp(-η)); me = μ * (1 - μ)
        w(y) = GM._beta_grouped_laplace_weight(:observed, f, μ, me, y, link, η)
        @test w(1e-70) == w(1e-12)
        @test w(1e-300) == w(1e-12)
        @test w(1 - 1e-17) == w(1 - 1e-12)          # 1 - 1e-17 is 1.0 in Float64
        @test w(0.4) == GM._beta_grouped_laplace_weight(:observed, f, μ, me, 0.4, link, η)
    end
    # Shared-precision and per-species Laplace marginals at fixed parameters.
    ref  = GM.beta_marginal_loglik_laplace(_bcw_with(Y, 1e-12), Λ, b, φ)
    refg = GM.beta_grouped_marginal_loglik_laplace(_bcw_with(Y, 1e-12), Λ, b, fill(φ, 6))
    for v in (1e-13, 1e-20, 1e-50, 1e-70, 1e-100, 1e-300)
        @test isfinite(GM.beta_marginal_loglik_laplace(_bcw_with(Y, v), Λ, b, φ))
        @test GM.beta_marginal_loglik_laplace(_bcw_with(Y, v), Λ, b, φ) == ref
        @test GM.beta_grouped_marginal_loglik_laplace(_bcw_with(Y, v), Λ, b, fill(φ, 6)) == refg
    end
    refhi = GM.beta_marginal_loglik_laplace(_bcw_with(Y, 1 - 1e-12), Λ, b, φ)
    for v in (1 - 1e-13, 1 - 1e-16)
        @test GM.beta_marginal_loglik_laplace(_bcw_with(Y, v), Λ, b, φ) == refhi
    end
end

@testset "Beta analytic gradient and fits are unchanged below the y clamp" begin
    Y, λ, b = _bcw_table()
    Λ = reshape(λ, :, 1); φ = 10.0; p = 6
    θ0 = vcat(b, λ, log(φ))
    obj(θ, Yv) = GM.beta_marginal_loglik_laplace(Yv, reshape(θ[(p + 1):(2p)], p, 1), θ[1:p], exp(θ[2p + 1]))
    function fdgrad(Yv)
        g = zeros(length(θ0))
        for k in eachindex(θ0)
            e = zeros(length(θ0)); e[k] = 1e-6
            g[k] = (obj(θ0 .+ e, Yv) - obj(θ0 .- e, Yv)) / 2e-6
        end
        g
    end
    gref = GM.beta_laplace_grad(_bcw_with(Y, 1e-12), Λ, b, φ)
    for v in (1e-70, 1e-300)
        Y2 = _bcw_with(Y, v)
        @test GM.beta_laplace_grad(Y2, Λ, b, φ) == gref
        @test maximum(abs, GM.beta_laplace_grad(Y2, Λ, b, φ) .- fdgrad(Y2)) < 1e-4
    end
    # Fits: a response of 1e-300 is the cell 1e-12, on the shared and the default (per-species) routes.
    for fit in (Y -> GM.fit_beta_gllvm(Y; K = 1), Y -> fit_gllvm(Y; family = GM.Beta(), K = 1))
        f12 = fit(_bcw_with(Y, 1e-12)); f300 = fit(_bcw_with(Y, 1e-300))
        @test isfinite(f300.loglik)
        @test f300.converged == f12.converged
        @test isapprox(f300.loglik, f12.loglik; atol = 1e-10)
    end
end
