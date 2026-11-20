# SHINICHI-1753-1118 converged rule, Gaussian path: first match wins in the order
# not_converged > boundary > degenerate > converged.
using GLLVModels, Optim, Test, Random, StableRNGs, Distributions

function _heywood_fixture_1120()
    rng = StableRNG(1284)
    p, n = 9, 276
    λ = [0.15, -0.1, -0.08, 0.1, -0.14, 0.02, -0.04, 0.0, 0.19]
    η = randn(rng, 1, n)
    Y = λ * η .+ randn(rng, p, n)
    Y[8, :] .= 90 .* Y[8, :] .+ 250
    return Y
end

@testset "convergence_status order (#1120)" begin
    Y = _heywood_fixture_1120()
    fit = redirect_stderr(devnull) do
        fit_gllvm(Y; family = Normal(), K = 1)
    end
    @test convergence_status(fit) === :boundary
    # Optimiser did not finish AND a parameter is at a boundary: not_converged wins.
    # Fake an optimiser that stopped without converging, keeping the Heywood pars.
    res_bad = Optim.optimize(x -> sum(abs2, x), [1.0], Optim.NelderMead(),
                             Optim.Options(iterations = 1))
    @test !Optim.converged(res_bad)
    fit_nc = GllvmFit(fit.model, fit.pars, fit.logLik, fit.n_iter, false, res_bad,
                      fit.cputime, fit.integration, fit.optimizer_info)
    @test !isempty(boundary_params(fit_nc))
    @test convergence_status(fit_nc) === :not_converged
end

@testset "convergence_status :degenerate (#1120)" begin
    # Rule: singular / non-PD Hessian or non-finite SE -> degenerate (after boundary).
    H_sing = [1.0 1.0; 1.0 1.0]
    _, se_s, pd_s = GLLVModels._gaussian_wald_covariance_from_hessian(H_sing, 2)
    @test GLLVModels._wald_degenerate(pd_s, se_s)
    @test GLLVModels._wald_degenerate(true, [0.1, NaN])
    @test GLLVModels._wald_degenerate(true, [0.1, Inf])
    @test !GLLVModels._wald_degenerate(true, [0.1, 0.2])

    # Well-conditioned fit with the data supplied: Hessian is PD, status stays :converged.
    rng = StableRNG(11)
    p, n = 6, 150
    Λ = [0.8 0.5 0.3 0.6 0.4 0.7]'
    Y = Λ * randn(rng, 1, n) .+ randn(rng, p, n)
    fit = fit_gllvm(Y; family = Normal(), K = 1)
    @test convergence_status(fit) === :converged
    @test convergence_status(fit; y = Y) === :converged

    # A boundary fit is never read as degenerate, even when y is supplied.
    Yh = _heywood_fixture_1120()
    fith = redirect_stderr(devnull) do
        fit_gllvm(Yh; family = Normal(), K = 1)
    end
    @test convergence_status(fith; y = Yh) === :boundary
end

@testset "max gradient at the reported optimum (#1120)" begin
    # Well-conditioned Gaussian fixture: the fit's own objective has a near-zero gradient at the
    # reported optimum, relative to max(1, |nll|) (unit-free, as the converged rule requires).
    rng = StableRNG(21)
    p, n = 6, 300
    Λ = [0.8 0.5 0.3 0.6 0.4 0.7]'
    Y = Λ * randn(rng, 1, n) .+ randn(rng, p, n)
    fit = fit_gllvm(Y; family = Normal(), K = 1)
    @test convergence_status(fit; y = Y) === :converged
    nll = GLLVModels._confint_reconstruct_nll(fit, Y, nothing, nothing)
    θ̂ = fit.pars.θ_packed
    g = GLLVModels.ForwardDiff.gradient(nll, θ̂)
    @test all(isfinite, g)
    @test maximum(abs, g) / max(1.0, abs(nll(θ̂))) <= 1e-3
end
