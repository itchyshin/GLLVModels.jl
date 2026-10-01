using GLLVModels, Test, TOML, Optim
const GM = GLLVModels

# Poisson-limit ridge (#615). Toward r = ∞ the NB2 likelihood in log r flattens but
# keeps rising, so L-BFGS could crawl along that ridge until the iteration cap and
# report converged = false. `_nb_poisson_ridge_polish` refits such a fit with the
# groups at r > 1e3 fixed at r = 1e10. The toy checks need no platform-recorded
# values; the fit check asserts the verdict and that the reported loglik is the
# likelihood at the reported estimates.

@testset "NB2 Poisson-limit ridge polish (#615)" begin
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = 1e-8, iterations = 500)
    short = Optim.Options(g_tol = 1e-8, iterations = 3)

    @testset "toy ridge rising to the limit" begin
        # θ[2] plays log r: the objective keeps falling as θ[2] → ∞.
        f(θ) = (θ[1] - 1)^2 + exp(-θ[2])
        res = Optim.optimize(f, [0.0, 8.0], ls, short; autodiff = :finite)
        @test !Optim.converged(res)
        @test exp(Optim.minimizer(res)[2]) > 1e3
        θ, nll, conv, iters = GM._nb_poisson_ridge_polish(f, res, ls, opts, 2)
        @test conv
        @test θ[2] == log(1e10)
        @test θ[1] ≈ 1 atol = 1e-6
        @test nll <= Optim.minimum(res)
        @test nll ≈ f(θ)
        @test iters > Optim.iterations(res)
    end

    @testset "finite optimum: the stalled point is kept" begin
        # The optimum sits near log r = 9 (r ≈ 8e3), so fixing r = 1e10 is worse. Not a
        # quadratic, which L-BFGS would solve within the three allowed iterations.
        f(θ) = (θ[1] - 1)^2 + (θ[2] - 9)^4 + exp(θ[1] * θ[2] / 20)
        res = Optim.optimize(f, [0.0, 7.5], ls, short; autodiff = :finite)
        @test !Optim.converged(res)
        θ, nll, conv, iters = GM._nb_poisson_ridge_polish(f, res, ls, opts, 2)
        @test θ == Optim.minimizer(res)
        @test nll == Optim.minimum(res)
        @test !conv
        @test iters == Optim.iterations(res)
    end

    @testset "a converged fit is returned unchanged" begin
        f(θ) = (θ[1] - 1)^2 + exp(-θ[2])
        res = Optim.optimize(f, [0.0, 8.0], ls, Optim.Options(g_tol = 1e-3); autodiff = :finite)
        @test Optim.converged(res)
        θ, nll, conv, iters = GM._nb_poisson_ridge_polish(f, res, ls, opts, 2)
        @test θ == Optim.minimizer(res)
        @test nll == Optim.minimum(res)
        @test conv
    end

    @testset "per-species fit that stalled on the ridge" begin
        fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb_grouped_poisson_ridge.toml"))
        p, n, K = fixture["p"], fixture["n"], fixture["K"]
        Y = reshape(Int64.(fixture["Y_column_major"]), p, n)
        fit = GM.fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p))
        @test fit.converged
        @test isfinite(fit.loglik)
        rvec = fit.r_group[fit.group]
        @test fit.loglik ≈ GM.nb_grouped_marginal_loglik_laplace(Y, fit.Λ, fit.β, rvec;
                                                             maxiter = 100, tol = 1e-9) rtol = 1e-10
    end
end
