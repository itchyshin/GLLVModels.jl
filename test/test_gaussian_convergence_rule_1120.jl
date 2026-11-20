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
