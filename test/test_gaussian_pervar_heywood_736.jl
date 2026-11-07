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

@testset "pervar Heywood real data: default fits (#736)" begin
    cases = (("urine", 1, -495.581931), ("urine", 2, -464.752746),
             ("obrienkaiser", 2, -370.288228), ("usjudgeratings", 3, 12.275397))
    for (name, K, ref) in cases, method in (:em, :lbfgs)
        Y = _read_736_fixture(name)
        fit = fit_gllvm(Y; family = Normal(), K = K, pervar = true, method = method)
        @testset "$name K=$K method=$method" begin
            Yc = Y .- sum(Y, dims = 2) ./ size(Y, 2)
            @test isapprox(gaussian_pervar_marginal_loglik(Yc, fit.Λ, fit.φ²), fit.loglik;
                           atol = 1e-6)
            if name == "usjudgeratings"
                # The default fit stalls short of the optimum at this Heywood boundary;
                # the default path is deliberately unchanged (maintainer decision, 31 Oct).
                @test_broken fit.converged
                @test fit.loglik >= ref - 1e-2
            else
                @test fit.converged
                @test fit.loglik >= ref - 1e-5
            end
        end
    end
end

@testset "pervar opt-in start, multi-start and boundary indicator (#736)" begin
    Y = _read_736_fixture("urine")
    Yc = Y .- sum(Y, dims = 2) ./ size(Y, 2)
    p = size(Y, 1)
    for method in (:em, :lbfgs)
        base = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method)

        # n_starts = 1 is the default fit and draws nothing from rng.
        rng = StableRNG(1)
        state = copy(rng)
        one = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method,
                        n_starts = 1, rng = rng)
        @test one.loglik == base.loglik && one.Λ == base.Λ && one.ψ² == base.ψ²
        @test rng == state

        # The default fit does not depend on the global seed.
        Random.seed!(1); a = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method)
        Random.seed!(2); b = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method)
        @test a.loglik == b.loglik && a.Λ == b.Λ

        # A user start at the optimum (including a ψ² near zero) stays there, and the
        # reported likelihood is the exact value at the returned parameters.
        u = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method,
                      λ_init = base.Λ, ψ²_init = base.ψ²)
        @test u.converged
        @test isapprox(u.loglik, base.loglik; atol = 1e-5)
        @test isapprox(gaussian_pervar_marginal_loglik(Yc, u.Λ, u.φ²), u.loglik; atol = 1e-6)

        # Multi-start is reproducible from its rng and never worse than the first start.
        m1 = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method,
                       n_starts = 4, rng = StableRNG(7))
        m2 = fit_gllvm(Y; family = Normal(), K = 2, pervar = true, method = method,
                       n_starts = 4, rng = StableRNG(7))
        @test m1.loglik == m2.loglik && m1.Λ == m2.Λ
        @test m1.converged
        @test m1.loglik >= base.loglik - 1e-8
    end

    @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y; K = 2, λ_init = zeros(p, 3))
    @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y; K = 2, ψ²_init = zeros(p))
    @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y; K = 2, ψ²_init = ones(p - 1))
    @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y; K = 2, n_starts = 0)

    # Boundary indicator: urine trait 3 sits at ψ² ≈ 0; OBrienKaiser K = 2 is interior.
    @test pervar_boundary(fit_gllvm(Y; family = Normal(), K = 1, pervar = true)) == [3]
    obk = fit_gllvm(_read_736_fixture("obrienkaiser"); family = Normal(), K = 2, pervar = true)
    @test isempty(pervar_boundary(obk))
    @test pervar_boundary(obk; rtol = 1.0) == collect(1:size(obk.Λ, 1))
end
