using GLLVModels, Test, Random, Distributions, LinearAlgebra

# Opt-in loading ridge for the Laplace Binomial fitter (vault D-293): binary
# data get a loading-ridge sweep when choosing the latent dimension K. This
# gates `loading_ridge` on `fit_binomial_gllvm` / `_fit_binomial_gllvm_laplace`
# (src/families/binomial.jl) and its AGHQ guard (src/families/aghq_binomial_fit.jl).

@testset "Binomial loading ridge" begin

    @testset "loading_ridge = Inf is a no-op (bit-identical)" begin
        Random.seed!(11)
        p, n, K = 6, 300, 1
        link = LogitLink()
        Λtrue = 1.0 .* randn(p, K)
        βtrue = 0.3 .* randn(p)
        Z = randn(K, n)
        η = βtrue .+ Λtrue * Z
        P = 1 ./ (1 .+ exp.(-η))
        Y = Int.(rand(p, n) .< P)

        fit_default = fit_binomial_gllvm(Y; K = K, link = link)
        fit_inf     = fit_binomial_gllvm(Y; K = K, link = link, loading_ridge = Inf)

        @test fit_inf.loglik === fit_default.loglik
        @test fit_inf.β == fit_default.β
        @test fit_inf.Λ == fit_default.Λ
        @test fit_inf.converged == fit_default.converged
        @test fit_inf.loading_ridge == Inf
        @test fit_default.loading_ridge == Inf   # omitting the kwarg ⇒ same default
    end

    @testset "penalised analytic gradient matches finite differences" begin
        # Same construction as the `ag` closure in `_fit_binomial_gllvm_laplace`
        # (binomial.jl): `-binomial_laplace_grad` plus the ridge gradient
        # `pack_lambda(Λ) / τ²` on the Λ block, checked against a central
        # finite difference of the penalised objective `negll(θ) + 0.5‖Λ‖²/τ²`.
        Random.seed!(202)
        p, K, n = 6, 2, 40
        β = 0.3 .* randn(p)
        Λ = 0.5 .* randn(p, K)
        N = fill(4, p, n)
        Y = [rand(0:N[t, s]) for t in 1:p, s in 1:n]
        τ = 1.5
        rr = GLLVModels.rr_theta_len(p, K)
        θ = vcat(β, GLLVModels.pack_lambda(Λ))

        function negll_pen(θv)
            b = θv[1:p]
            L = GLLVModels.unpack_lambda(θv[(p + 1):(p + rr)], p, K)
            v = -GLLVModels.binomial_marginal_loglik_laplace(Y, N, L, b, LogitLink();
                                                              maxiter = 200, tol = 1e-12)
            return v + 0.5 * sum(abs2, L) / τ^2
        end

        h = 1e-6
        g_fd = similar(θ)
        for i in eachindex(θ)
            θp = copy(θ); θp[i] += h
            θm = copy(θ); θm[i] -= h
            g_fd[i] = (negll_pen(θp) - negll_pen(θm)) / (2h)
        end

        g_an = -GLLVModels.binomial_laplace_grad(Y, N, Λ, β)
        g_an[(p + 1):(p + rr)] .+= GLLVModels.pack_lambda(Λ) ./ τ^2

        relerr = norm(g_an - g_fd) / norm(g_fd)
        @test relerr < 1e-5
    end

    @testset "ridge controls a separation-prone Bernoulli runaway" begin
        # seed=30, p=10, n=60, K=2, loadings 0.8·randn: the unpenalised fit's
        # max loading row norm exceeds 10 (a Laplace-saturation runaway, per
        # scratchpad/probe_ridge_seed30.jl — measured max_row_norm ≈ 91.3,
        # fit0.converged == false, the expected shape of this pathology).
        Random.seed!(30)
        p, n, K = 10, 60, 2
        Λtrue = 0.8 .* randn(p, K)
        βtrue = 0.3 .* randn(p)
        Z = randn(K, n)
        η = βtrue .+ Λtrue * Z
        P = 1 ./ (1 .+ exp.(-η))
        Y = Int.(rand(p, n) .< P)

        fit0 = fit_binomial_gllvm(Y; K = K, iterations = 300)
        rownorm(Λ) = maximum(sqrt.(sum(abs2, Λ; dims = 2)))
        @test rownorm(fit0.Λ) > 10   # confirms the runaway premise

        fitr = fit_binomial_gllvm(Y; K = K, iterations = 300, loading_ridge = 2.0)
        @test rownorm(fitr.Λ) < 10
        @test fitr.loading_ridge == 2.0

        # `loglik` must be the UNPENALISED Laplace marginal at the ridge
        # optimum — recompute independently via the negll path, not the
        # stored value.
        recomputed = GLLVModels.binomial_marginal_loglik_laplace(
            Y, fill(1, p, n), fitr.Λ, fitr.β, GLLVModels.LogitLink())
        @test isapprox(recomputed, fitr.loglik; atol = 1e-8, rtol = 1e-10)

        # A penalised optimum cannot beat the unpenalised maximum when the
        # latter converged; skip this check for a genuine runaway (as here).
        if fit0.converged
            @test fitr.loglik <= fit0.loglik + 1e-6
        end
    end

    @testset "AGHQ + finite loading_ridge throws" begin
        Random.seed!(3)
        p, n, K = 6, 40, 1
        Y = rand(0:1, p, n)
        for aghqval in (true, 1, :auto, 5)
            @test_throws ArgumentError fit_binomial_gllvm(Y; K = K, aghq = aghqval,
                                                           loading_ridge = 2.0, iterations = 2)
        end
        # aghq off (the default) is unaffected
        fit = fit_binomial_gllvm(Y; K = K, aghq = false, loading_ridge = 2.0, iterations = 5)
        @test fit.loading_ridge == 2.0
    end

    @testset "loading_ridge must be positive" begin
        Random.seed!(4)
        Y = rand(0:1, 4, 20)
        @test_throws ArgumentError fit_binomial_gllvm(Y; K = 1, loading_ridge = 0.0)
        @test_throws ArgumentError fit_binomial_gllvm(Y; K = 1, loading_ridge = -1.0)
    end
end
