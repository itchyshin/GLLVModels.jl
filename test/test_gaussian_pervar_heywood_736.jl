# Per-variable Gaussian fit near a Heywood boundary reports a converged fit (refs #736).
#
# With `pervar = true` the default route is EM, which converges only linearly as a unique
# variance tends to 0 and so exhausted its iteration cap with `converged = false`, short of the
# optimum. The EM result is now polished by L-BFGS on the log-variance scale, where the gradient
# vanishes at the boundary. No seeded magnitudes are asserted: only structural properties and
# that the polish never loses likelihood against the EM point.

using GLLVModels, Test, Random, Distributions, LinearAlgebra, StableRNGs

@testset "pervar Heywood boundary converges (#736)" begin
    rng = StableRNG(736)
    p, n, K = 6, 77, 1
    λ = [0.9, 0.5, 1.0, 0.7, 0.6, 0.4]
    ψtrue = [0.3, 0.9, 1e-6, 0.3, 0.4, 0.6]   # trait 3 is a Heywood case
    η = randn(rng, 1, n)
    Y = λ * η .+ sqrt.(ψtrue) .* randn(rng, p, n)

    fit = fit_gaussian_pervar_gllvm(Y; K = K)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(>(0), fit.ψ²)
    @test minimum(fit.ψ²) < 1e-2          # the boundary trait is recovered

    # The polish must be at least as good as the EM point it started from.
    Yc = Y .- sum(Y, dims = 2) ./ n
    Λ0, _ = ppca_init(Yc, K)
    col_var = vec(sum(abs2, Yc, dims = 2)) ./ (n - 1)
    _, _, ll_em, _, _ = em_fa(Yc, K; λ_init = Λ0, ψ_init = max.(0.5 .* col_var, 1e-3),
                              tol = 1e-8, max_iter = 2000)
    @test fit.loglik >= ll_em - 1e-8

    # Direct check: likelihood at the returned parameters matches the reported one.
    ll = gaussian_pervar_marginal_loglik(Yc, fit.Λ, fit.φ²)
    @test isapprox(ll, fit.loglik; atol = 1e-6)
end

# Real-data Heywood cases from #736. References: gllvmTMB `latent(..., unique = TRUE)` logLik
# quoted in the issue (urine, OBrienKaiser) and the direct ML factor-analysis value for
# USJudgeRatings K = 3 (12.275397; gllvmTMB 12.275310). Fixtures are the issue's prepared data
# (urine standardised, OBrienKaiser centred) and raw datasets::USJudgeRatings; the per-species
# intercepts are profiled, so centring does not change the logLik.
function _read_736_fixture(name)
    lines = readlines(joinpath(@__DIR__, "fixtures", "pervar_heywood_736_$name.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    m = length(split(lines[1], ",")) - 1
    [parse(Float64, r[j + 1]) for j in 1:m, r in rows]   # p × n
end

@testset "pervar Heywood real data reaches the reference optimum (#736)" begin
    cases = (("urine", 1, -495.581931), ("urine", 2, -464.752746),
             ("obrienkaiser", 2, -370.288228), ("usjudgeratings", 3, 12.275397))
    for (name, K, ref) in cases, method in (:em, :lbfgs)
        Y = _read_736_fixture(name)
        fit = fit_gllvm(Y; family = Normal(), K = K, pervar = true, method = method)
        @testset "$name K=$K method=$method" begin
            @test fit.converged
            @test fit.loglik >= ref - 1e-5
            Yc = Y .- sum(Y, dims = 2) ./ size(Y, 2)
            @test isapprox(gaussian_pervar_marginal_loglik(Yc, fit.Λ, fit.φ²), fit.loglik;
                           atol = 1e-6)
        end
    end
end
