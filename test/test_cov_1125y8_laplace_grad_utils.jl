# Lane cursor-cloud-1125y8: laplace_grad.jl _pois_logpmf and _optimize_with_analytic.
using Test, GLLVModels, Optim, ForwardDiff, Random, Distributions

const G = GLLVModels

@testset "cov 1125y8 laplace grad utilities" begin
    @testset "_pois_logpmf" begin
        @test G._pois_logpmf(2.5, 3) ≈ logpdf(Poisson(2.5), 3)
        @test G._pois_logpmf(1.0, 0) ≈ -1.0
        μ = 4.2
        g = ForwardDiff.derivative(μ -> G._pois_logpmf(μ, 2), μ)
        @test g ≈ 2 / μ - 1 atol = 1e-12
        @test all(isfinite, [G._pois_logpmf(0.1 + i / 10, i) for i in 0:5])
    end

    @testset "_optimize_with_analytic FD fallback" begin
        θ0 = [1.0, 2.0]
        target = [0.3, -0.7]
        negll(θ) = sum(abs2, θ .- target)
        bad_grad(θ) = fill(NaN, length(θ))
        good_grad(θ) = 2 .* (θ .- target)
        ls = Optim.LBFGS()
        opts = Optim.Options(g_tol = 1e-12, iterations = 20)
        res = G._optimize_with_analytic(negll, bad_grad, θ0, ls, opts)
        @test Optim.converged(res)
        @test Optim.minimizer(res) ≈ target atol = 1e-5
        res2 = G._optimize_with_analytic(negll, good_grad, θ0, ls, opts)
        @test Optim.minimizer(res2) ≈ target atol = 1e-8
    end

    @testset "_bfgs_continuation short-circuit" begin
        run(m, θ) = Optim.optimize(x -> sum(x .^ 2), θ, m, Optim.Options(iterations = 1))
        res = Optim.optimize(x -> sum(x .^ 2), [3.0], Optim.LBFGS(),
            Optim.Options(iterations = 5, g_tol = 1e-12))
        θ, f, c, it = G._bfgs_continuation(run, res)
        @test c == Optim.converged(res)
        @test θ == Optim.minimizer(res)
        rs = Optim.optimize(x -> G._NLL_SENTINEL, [1.0], Optim.LBFGS())
        @test G._bfgs_continuation(run, rs)[2] == G._NLL_SENTINEL
    end
end