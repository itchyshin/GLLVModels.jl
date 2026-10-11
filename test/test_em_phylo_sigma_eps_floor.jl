# #753: em_fit_phylo must record the data-relative σ_eps² clamp and must not
# report converged = true on that floor. Rank-1 repro from the issue;
# on origin/main e6cfcd4ee this returns σ_eps = sqrt(eps()), converged=true
# after a handful of iterations. The floor is `1e-8 * mean(y.^2)`, so the
# same flags must hold after rescaling y. SQUAREM's main path (no fallback)
# must record the same clamp.

using GLLVModels, Test, Random

function _rank1_em_phylo_repro()
    Random.seed!(2)
    p, n = 5, 30
    Σ = [0.6^abs(i - j) for i in 1:p, j in 1:p]
    λ = [1.0, 0.8, -0.5, 0.3, 0.9]
    y = λ * randn(n)'
    return y, Σ
end

function _assert_sigma_eps_floor(fit, y)
    @test fit.sigma_eps_boundary
    @test !fit.converged
    @test isfinite(fit.logLik)
end

@testset "em_fit_phylo σ_eps floor (#753)" begin
    y, Σ = _rank1_em_phylo_repro()

    @testset "plain EM, unit scale" begin
        fit = em_fit_phylo(y, 1, Σ; max_iter = 3000, assert_monotone = false)
        _assert_sigma_eps_floor(fit, y)
    end

    @testset "plain EM flags invariant to rescaling y" begin
        flags = Tuple{Bool,Bool}[]
        for c in (1.0, 1e-5, 1e5)
            yc = c .* y
            fit = em_fit_phylo(yc, 1, Σ; max_iter = 3000, assert_monotone = false)
            _assert_sigma_eps_floor(fit, yc)
            push!(flags, (fit.converged, fit.sigma_eps_boundary))
        end
        @test flags[1] == flags[2] == flags[3]
        @test flags[1] == (false, true)
    end

    @testset "SQUAREM main path records the same floor" begin
        flags = Tuple{Bool,Bool}[]
        for c in (1.0, 1e-5, 1e5)
            yc = c .* y
            fit = em_fit_phylo_squarem(yc, 1, Σ; max_iter = 3000,
                                       assert_monotone = false,
                                       safety_check = false)
            @test !fit.fallback_used
            _assert_sigma_eps_floor(fit, yc)
            push!(flags, (fit.converged, fit.sigma_eps_boundary))
        end
        @test flags[1] == flags[2] == flags[3]
        @test flags[1] == (false, true)
    end
end
