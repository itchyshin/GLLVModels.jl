using GLLVModels, Test, Random, Statistics

# #876: fit_relaxed_clock must not report converged = true when the free tau2 or
# sigma2_eps update ends clamped at its floor.

@testset "relaxed clock boundary flags (#876)" begin
    phy = edge_phy("((A:0.3,B:0.3):0.3,(C:0.3,D:0.3):0.3);")

    @testset "constant-rate data ends on the tau2 floor, not converged" begin
        y, _, _ = simulate_relaxed_bm(phy, fill(1.0, phy.n_edges), 0.05, 80;
                                      rng = MersenneTwister(1))
        fit = fit_relaxed_clock(phy, y; max_iter = 2000, tol = 1e-4)
        @test fit.tau2_boundary
        @test fit.τ² == 1e-4
        @test !fit.converged
        @test !fit.sigma_eps_boundary
        @test occursin("tau2_boundary=true", sprint(show, fit))
    end

    @testset "heterogeneous rates: converged, no boundary" begin
        σ²_true = [0.1, 4.0, 0.2, 6.0, 0.05, 3.0]
        @test length(σ²_true) == phy.n_edges
        y, _, _ = simulate_relaxed_bm(phy, σ²_true, 0.05, 400;
                                      rng = MersenneTwister(7))
        fit = fit_relaxed_clock(phy, y; max_iter = 2000, tol = 1e-4)
        @test !fit.tau2_boundary
        @test !fit.sigma_eps_boundary
        @test fit.converged
    end

    @testset "pinned parameters are not boundaries" begin
        y, _, _ = simulate_relaxed_bm(phy, fill(1.0, phy.n_edges), 0.05, 80;
                                      rng = MersenneTwister(1))
        fit = fit_relaxed_clock(phy, y; max_iter = 2000, tol = 1e-4,
                                fix_τ² = 1e-4, fix_σ²_eps = 1e-12)
        @test !fit.tau2_boundary
        @test !fit.sigma_eps_boundary
    end

    @testset "boundary flags do not depend on the units of y" begin
        for seed in 1:4
            y, _, _ = simulate_relaxed_bm(phy, fill(1.0, phy.n_edges), 0.05, 80;
                                          rng = MersenneTwister(seed))
            ref = fit_relaxed_clock(phy, y; max_iter = 2000, tol = 1e-4)
            for c in (1e5, 1e-5)
                f = fit_relaxed_clock(phy, c .* y; max_iter = 2000, tol = 1e-4)
                @test f.tau2_boundary == ref.tau2_boundary
                @test f.sigma_eps_boundary == ref.sigma_eps_boundary
            end
        end
        # Near-noiseless data: the residual sits at the data-relative floor at
        # any scale (the old absolute 1e-8 floor flipped under rescaling).
        y0, _, _ = simulate_relaxed_bm(phy, fill(1.0, phy.n_edges), 0.0, 80;
                                       rng = MersenneTwister(3))
        flags = [fit_relaxed_clock(phy, c .* y0; max_iter = 300, tol = 1e-4).sigma_eps_boundary
                 for c in (1.0, 1e5, 1e-5)]
        @test allequal(flags)
    end

    @testset "positional constructor keeps working" begin
        f = fit_relaxed_clock(phy, randn(MersenneTwister(2), 4, 20); max_iter = 5)
        g = RelaxedClockFit(f.σ²_e, f.logrates, f.μ, f.τ², f.σ²_eps, f.n_iter,
                            f.converged, f.obj_trace, f.μ_nodes)
        @test !g.tau2_boundary && !g.sigma_eps_boundary
    end
end
