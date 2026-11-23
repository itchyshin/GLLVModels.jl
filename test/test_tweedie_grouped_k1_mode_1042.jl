using GLLVModels, Test, DelimitedFiles, LinearAlgebra

# refs #1042. `fit_gllvm(Y; family = TweedieED(1.0, 1.5), K = 1, disp_group = :species)` on
# an ordinary simulated Tweedie table (p = 5, n = 100, power 1.5, from gllvmTMB's parity
# fixtures) returned loglik = -Inf, converged = false. The grouped per-site mode search was
# pure Fisher scoring; where y >> mu the Fisher weight under-states the curvature by more
# than a factor of 2, so its unchecked small-step branch limit-cycled (|step| ~ 3e-3 for
# 2,000 iterations) and the site was reported as not converged -> -Inf at the start point.
# The step weight is now max(observed, Fisher), as in the scalar Tweedie search (#623).

const _G1042 = GLLVModels

@testset "Tweedie grouped K = 1 mode search (#1042)" begin
    Y = readdlm(joinpath(@__DIR__, "fixtures", "tweedie_k1_1042.csv"), ',', Float64)
    p, n = size(Y)
    @test (p, n) == (5, 100)

    # The empirical-log warm start fit_tweedie_gllvm_grouped uses (K = 1).
    Zemp = log.(Y .+ _G1042._tweedie_log_offset(Y, nothing))
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β0)
    Λ0 = reshape(F.U[:, 1] .* (F.S[1] / sqrt(n)), p, 1)
    fams = [_G1042.TweedieED(1.0, 1.5) for _ in 1:p]
    N = ones(Int, p, n)

    # Every site's mode search must converge, in particular sites 40 and 91 (pre-fix:
    # limit cycle), and the converged z must be a stationary point: Λ's - z = 0.
    for i in 1:n
        z, ok = _G1042._tweedie_grouped_mode(fams, Y[:, i], N[:, i], Λ0, β0, _G1042.LogLink();
                                             maxiter = 100, tol = 1e-9)
        @test ok
        η = _G1042._clamp_eta.(β0 .+ Λ0 * z)
        μ = _G1042._clamp_mu.(fams, _G1042.linkinv.(Ref(_G1042.LogLink()), η))
        me = _G1042.mu_eta.(Ref(_G1042.LogLink()), η)
        s = _G1042._glm_score.(fams, μ, N[:, i], me, Y[:, i])
        @test norm(Λ0' * s .- z, Inf) < 1e-6
    end

    for hs in (:observed, :fisher)
        @test isfinite(_G1042.tweedie_grouped_marginal_loglik_laplace(
            Y, Λ0, β0, ones(p), 1.5; hessian = hs))
    end

    # The issue's own call (all three power settings) now fits.
    for kw in ((power_group = :species,), NamedTuple(), (power = 1.5,))
        f = fit_gllvm(Y; family = _G1042.TweedieED(1.0, 1.5), K = 1,
                      disp_group = :species, kw...)
        @test isfinite(f.loglik)
    end
end
