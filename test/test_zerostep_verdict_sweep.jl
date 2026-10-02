using GLLVModels, Test, Random, LinearAlgebra, Distributions, Optim

# #505 (the remaining #485 class): fitters that trusted `Optim.converged(res)` after a
# zero-length line-search step. Each family gets its own verdict (per the #502 decision):
# `converged` additionally requires `gres <= max(g_tol, g_tol * |nll|)`, the rule already
# used by `_tweedie_verdict`, `_beta_grouped_g_met` and `_nb1_grouped_g_met`. Estimates
# are untouched; only the `converged` flag changes.
#
# Platform robustness (as in test_fit_verdict_gradient.jl): live-RNG fixtures are never
# asserted to land on one outcome; relations ("converged => small independent gradient")
# are checked instead. The phylo reproduction uses a literal bad START, which is
# deterministic (no RNG-dependent optimiser path).

struct _ZS505FakeResult
    gres::Float64
    nll::Float64
end
Optim.g_residual(r::_ZS505FakeResult) = r.gres
Optim.minimum(r::_ZS505FakeResult) = r.nll

_zs505_bnw(l, bl) = length(l) == 1 ? l[1] * ":" * string(bl) :
    "(" * _zs505_bnw(l[1:cld(length(l), 2)], bl) * "," *
          _zs505_bnw(l[(cld(length(l), 2) + 1):end], bl) * "):" * string(bl)
_zs505_tree(p) = _zs505_bnw(["t$i" for i in 1:p], 0.1) * ";"

# Independent central-difference gradient of the phylo negll in (log σ²_phy, log σ²_eps).
function _zs505_phylo_maxgrad(phy, y, fit, profile_mu)
    p = phy.n_leaves
    function f(θ)
        st = GLLVModels.build_node_perspecies(phy, fill(exp(θ[1] / 2), p), exp(θ[2]))
        μ = profile_mu ? GLLVModels._phylo_profile_mu(st, y) : fit.μ
        return GLLVModels._phylo_negll(st, y, μ)
    end
    θ = [log(fit.σ²_phy), log(fit.σ²_eps)]
    return maximum(1:2) do j
        e = zeros(2); e[j] = 1e-5
        abs(f(θ .+ e) - f(θ .- e)) / 2e-5
    end
end

@testset "#505 phylo Gaussian: a start the gradient cannot leave is not convergence" begin
    @testset "unit contract (platform-free)" begin
        g_tol = 1e-5
        @test !GLLVModels._phylo_g_met(_ZS505FakeResult(7e9, 12.0), g_tol)
        @test !GLLVModels._phylo_g_met(_ZS505FakeResult(NaN, 12.0), g_tol)
        @test GLLVModels._phylo_g_met(_ZS505FakeResult(1e-7, 12.0), g_tol)
        @test GLLVModels._phylo_g_met(_ZS505FakeResult(5e-5, 12.0), g_tol)  # < g_tol*|nll|
    end

    @testset "cliff start: old code said converged = true at a huge/NaN gradient" begin
        for (p, seed, pm) in ((8, 1, true), (16, 3, true), (16, 4, false), (12, 8, false))
            phy = GLLVModels.augmented_phy(_zs505_tree(p))
            y = randn(MersenneTwister(seed), p)
            fit = GLLVModels.fit_phylo_gaussian(phy, y; profile_mu = pm,
                logσ²phy0 = 30.0, logσ²eps0 = -30.0, μ0 = 0.0)
            @test !fit.converged
        end
    end

    @testset "relation: converged => stationary, over starts and seeds" begin
        for (p, seed, pm) in ((8, 1, true), (16, 3, true), (16, 4, false), (12, 8, false)),
            (a, b) in ((0.0, 0.0), (20.0, 20.0), (-20.0, -20.0), (-30.0, 30.0), (45.0, 45.0))
            phy = GLLVModels.augmented_phy(_zs505_tree(p))
            y = randn(MersenneTwister(seed), p)
            fit = GLLVModels.fit_phylo_gaussian(phy, y; profile_mu = pm,
                logσ²phy0 = a, logσ²eps0 = b, μ0 = 0.0)
            fit.converged && @test _zs505_phylo_maxgrad(phy, y, fit, pm) < 1e-2
        end
    end

    @testset "default start still converges" begin
        phy = GLLVModels.augmented_phy(_zs505_tree(16))
        y = randn(MersenneTwister(5), 16)
        @test GLLVModels.fit_phylo_gaussian(phy, y).converged
    end
end

function _zs505_nb1_fixture(seed; p = 4, n = 40, K = 1)
    rng = MersenneTwister(seed)
    β = 0.5 .+ rand(rng, p)
    Λ = 0.7 .* randn(rng, p, K)
    for j in 1:K
        Λ[j, j] = abs(Λ[j, j]) + 0.2
    end
    Z = randn(rng, K, n)
    return [rand(rng, NegativeBinomial(exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30)),
                                       0.5)) for t in 1:p, i in 1:n]
end

@testset "#505 scalar NB1 (fit_nb1_gllvm): zero-length step is not convergence" begin
    @testset "unit contract (platform-free)" begin
        g_tol = 1e-5
        @test !GLLVModels._nb1_grouped_g_met(_ZS505FakeResult(3.85, 100.0), g_tol)
        @test GLLVModels._nb1_grouped_g_met(_ZS505FakeResult(1e-7, 100.0), g_tol)
    end

    @testset "relation: converged => small independent gradient" begin
        p, K = 4, 1
        rr = GLLVModels.rr_theta_len(p, K)
        for seed in 201:204
            Y = _zs505_nb1_fixture(seed; p = p, K = K)
            fit = GLLVModels.fit_nb1_gllvm(Y; K = K)
            if fit.converged
                θ = vcat(fit.β, GLLVModels.pack_lambda(fit.Λ), log(fit.φ))
                f(θ) = -GLLVModels.nb1_marginal_loglik_laplace(Y,
                    GLLVModels.unpack_lambda(θ[(p + 1):(p + rr)], p, K), θ[1:p], exp(θ[end]))
                mg = maximum(eachindex(θ)) do j
                    e = zeros(length(θ)); e[j] = 1e-5
                    abs(f(θ .+ e) - f(θ .- e)) / 2e-5
                end
                @test mg < 1e-2
            end
        end
    end

    @testset "a genuinely stationary fit stays converged" begin
        Y = _zs505_nb1_fixture(202)
        fit = GLLVModels.fit_nb1_gllvm(Y; K = 1)
        @test isfinite(fit.loglik)
        @test fit.converged
    end
end
