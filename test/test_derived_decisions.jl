using GLLVModels, Test, Random, LinearAlgebra

# Maintainer-approved decisions on derived-quantity profile CIs
# (issue sweep 2, slice s2).
#
# #142: `profile_ci_derived(...; bounds = (lo, hi))` clamps to the natural
# support of the quantity ([0, 1] for communality / ICC / phylo signal,
# [-1, 1] for correlations). The returned interval must satisfy
# lower <= estimate <= upper, and interior results must not change.
#
# Expected values are computed independently of the code under test: the
# natural edges come from the definition of the quantity, the flat-profile
# case from an exact likelihood invariance checked below, and the interior
# values were computed on the unmodified tree (also pinned in
# test_derived_ci_sweep.jl).

@testset "#142 profile_ci_derived natural-bounds clamp" begin
    # Interior J1 fixture (same as test_derived_ci_sweep.jl).
    Random.seed!(11)
    p, K, n = 4, 1, 80
    Λ = reshape([0.8, 0.6, 0.5, -0.4], p, K)
    y = Λ * randn(K, n) + 0.5 * randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    spec = GLLVModels._derived_spec(fit)
    f_c1 = GLLVModels._make_communality_closure(spec, 1)
    lower_unmodified = 0.5976702559327752   # profile_ci_derived on the unmodified tree
    upper_unmodified = 0.7971364185161999

    @testset "#142 near-singular fit: upper bound above 1 is clamped to 1" begin
        # #670's fit2: trait 1 communality is 0.99967 and the raw profile
        # upper bound overshoots the support (1.0435).
        Random.seed!(11)
        y2 = [1.0; 1.0; 0.5; 0.2] * randn(1, 80) + 0.02 * randn(4, 80)
        fit2 = fit_gaussian_gllvm(y2; K = 1)
        spec2 = GLLVModels._derived_spec(fit2)
        Λ2, σ2 = fit2.pars.Λ, fit2.pars.σ_eps
        c2_hand = Λ2[1, 1]^2 / (Λ2[1, 1]^2 + σ2^2)
        ρ12_hand = Λ2[1, 1] * Λ2[2, 1] /
                   sqrt((Λ2[1, 1]^2 + σ2^2) * (Λ2[2, 1]^2 + σ2^2))

        f2_c1 = GLLVModels._make_communality_closure(spec2, 1)
        raw = GLLVModels.profile_ci_derived(fit2, f2_c1; y = y2)
        @test raw.upper > 1                       # the defect: outside [0, 1]
        ci = GLLVModels.profile_ci_derived(fit2, f2_c1; y = y2, bounds = (0, 1))
        @test ci.estimate ≈ c2_hand rtol = 1e-10
        @test ci.upper == 1.0
        @test ci.boundary
        @test ci.method === :profile
        @test 0 ≤ ci.lower ≤ ci.estimate ≤ ci.upper
        @test ci.lower == raw.lower               # the interior side is untouched

        f2_ρ = GLLVModels._make_correlation_closure(spec2, 1, 2)
        raw_ρ = GLLVModels.profile_ci_derived(fit2, f2_ρ; y = y2)
        @test raw_ρ.upper > 1
        cr = GLLVModels.profile_ci_derived(fit2, f2_ρ; y = y2, bounds = (-1, 1))
        @test cr.estimate ≈ ρ12_hand rtol = 1e-10
        @test cr.upper == 1.0
        @test cr.boundary
        @test -1 ≤ cr.lower ≤ cr.estimate ≤ cr.upper
    end

    @testset "#142 flat profile: a NaN bound at the edge becomes the edge" begin
        # On a has_diag fit the likelihood depends on σ²_B[t] and σ²_W[t]
        # only through their sum, so the share g = σ²_B/(σ²_B + σ²_W) has an
        # exactly flat profile over [0, 1]: the 95% CI is [0, 1].
        Random.seed!(7)
        p3, n3 = 4, 300
        Λ3 = reshape([0.9, 0.7, 0.5, -0.4], p3, 1)
        y3 = Λ3 * randn(1, n3) .+ [0.3, 0.5, 0.8, 0.4] .* randn(p3, n3)
        fd = fit_gaussian_gllvm(y3; K = 1, has_diag = true)
        sB, sW = fd.pars.σ²_B, fd.pars.σ²_W
        ll_fit = GLLVModels.gaussian_marginal_loglik(y3, fd.pars.Λ, fd.pars.σ_eps;
                                                     σ²_B = sB, σ²_W = sW)
        ll_moved = GLLVModels.gaussian_marginal_loglik(y3, fd.pars.Λ, fd.pars.σ_eps;
                                                       σ²_B = sB .+ 0.99 .* sW,
                                                       σ²_W = 0.01 .* sW)
        @test ll_moved ≈ ll_fit rtol = 1e-12     # flat profile, independently

        specd = GLLVModels._derived_spec(fd)
        f_share = θ -> begin
            u = GLLVModels._derived_unpack(θ, specd)
            u.σ²_B[3] / (u.σ²_B[3] + u.σ²_W[3])
        end
        # A short search (max_expand = 2) runs out before reaching the edge.
        raw = GLLVModels.profile_ci_derived(fd, f_share; y = y3, max_expand = 2)
        @test isnan(raw.lower) && isnan(raw.upper)
        ci = GLLVModels.profile_ci_derived(fd, f_share; y = y3, max_expand = 2,
                                           bounds = (0.0, 1.0))
        @test ci.lower == 0.0
        @test ci.upper == 1.0
        @test ci.boundary
        @test ci.method === :profile
        @test ci.lower ≤ ci.estimate ≤ ci.upper

        # The full search overshoots both edges; the clamp brings it back.
        ci_full = GLLVModels.profile_ci_derived(fd, f_share; y = y3, bounds = (0.0, 1.0))
        @test (ci_full.lower, ci_full.upper) == (0.0, 1.0)
        @test ci_full.boundary
    end

    @testset "#142 interior results are unchanged" begin
        r_none = GLLVModels.profile_ci_derived(fit, f_c1; y = y)
        r_b = GLLVModels.profile_ci_derived(fit, f_c1; y = y, bounds = (0, 1))
        @test r_b.lower == r_none.lower
        @test r_b.upper == r_none.upper
        @test r_b.estimate == r_none.estimate
        @test r_b.method === :profile
        @test !r_b.boundary
        @test r_b.lower ≈ lower_unmodified rtol = 1e-5
        @test r_b.upper ≈ upper_unmodified rtol = 1e-5
        # Without `bounds` the return shape is the same as before.
        @test keys(r_none) == (:lower, :upper, :estimate, :method)
    end

    @testset "#142 a NaN bound whose edge is outside the region is bisected" begin
        # max_expand = 1 leaves both sides NaN; the deviance at the edges 0
        # and 1 is far above the cutoff, so the bound lies between the
        # estimate and the edge and must agree with the full search.
        r1 = GLLVModels.profile_ci_derived(fit, f_c1; y = y, max_expand = 1)
        @test isnan(r1.lower) && isnan(r1.upper)
        ci = GLLVModels.profile_ci_derived(fit, f_c1; y = y, max_expand = 1,
                                           bounds = (0, 1))
        @test ci.lower ≈ lower_unmodified atol = 2e-4    # bisection tol_x = 1e-4
        @test ci.upper ≈ upper_unmodified atol = 2e-4
        @test !ci.boundary
        @test ci.method === :profile
        @test ci.lower ≤ ci.estimate ≤ ci.upper

        # The same rule in _profile_ci_bounded (post-processing of an existing
        # result) when it is handed a NaN bound.
        g_hat = f_c1(fit.pars.θ_packed)
        r_nan = (lower = NaN, upper = NaN, estimate = g_hat, method = :failed)
        rb = GLLVModels._profile_ci_bounded(fit, f_c1, r_nan; level = 0.95, y = y,
                                            X = nothing, Σ_phy = nothing,
                                            lo_bound = 0.0, hi_bound = 1.0)
        @test rb.lower ≈ lower_unmodified atol = 2e-4
        @test rb.upper ≈ upper_unmodified atol = 2e-4
        @test rb.method === :profile
        @test !rb.boundary
        @test rb.lower ≤ rb.estimate ≤ rb.upper

        # With estimated trait intercepts and X = nothing, the edge refits
        # must include the intercept mean, as the profile search does.
        yi = y .+ [1.0, 2.0, 3.0, 4.0]
        fi = fit_gllvm(yi; family = GLLVModels.Normal(), K = 1)
        fi_c1 = GLLVModels._make_communality_closure(GLLVModels._derived_spec(fi), 1)
        full = GLLVModels.profile_ci_derived(fi, fi_c1; y = yi)
        @test full.method === :profile
        ri = (lower = NaN, upper = full.upper, estimate = full.estimate, method = :partial)
        rbi = GLLVModels._profile_ci_bounded(fi, fi_c1, ri; level = 0.95, y = yi,
                                             X = nothing, Σ_phy = nothing,
                                             lo_bound = 0.0, hi_bound = 1.0)
        @test rbi.lower ≈ full.lower atol = 2e-4
        @test rbi.upper == full.upper
    end

    @testset "#142 invalid bounds are rejected" begin
        @test_throws ArgumentError GLLVModels.profile_ci_derived(fit, f_c1; y = y,
                                                                 bounds = (1, 0))
        # bounds that exclude the estimate (0.71) cannot give a valid interval
        @test_throws ArgumentError GLLVModels.profile_ci_derived(fit, f_c1; y = y,
                                                                 bounds = (0.8, 1.0))
        r_out = (lower = 0.1, upper = 0.9, estimate = 1.5, method = :profile)
        @test_throws ArgumentError GLLVModels._profile_ci_bounded(
            fit, f_c1, r_out; level = 0.95, y = y, X = nothing, Σ_phy = nothing,
            lo_bound = 0.0, hi_bound = 1.0)
    end
end
