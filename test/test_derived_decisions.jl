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

    @testset "#142 a NaN bound whose edge is outside the region stays NaN" begin
        # max_expand = 1 leaves both sides NaN. The full search above gives
        # [0.598, 0.797], so both edges 0 and 1 lie outside the confidence
        # region: the profile is not flat to the edge, and no bound may be
        # invented for a side whose search failed.
        r1 = GLLVModels.profile_ci_derived(fit, f_c1; y = y, max_expand = 1)
        @test isnan(r1.lower) && isnan(r1.upper)
        ci = GLLVModels.profile_ci_derived(fit, f_c1; y = y, max_expand = 1,
                                           bounds = (0, 1))
        @test isnan(ci.lower) && isnan(ci.upper)
        @test !ci.boundary
        @test ci.method === :failed

        # The same rule in _profile_ci_bounded (post-processing of an existing
        # result) when it is handed a NaN bound.
        g_hat = f_c1(fit.pars.θ_packed)
        r_nan = (lower = NaN, upper = NaN, estimate = g_hat, method = :failed)
        rb = GLLVModels._profile_ci_bounded(fit, f_c1, r_nan; level = 0.95, y = y,
                                            X = nothing, Σ_phy = nothing,
                                            lo_bound = 0.0, hi_bound = 1.0)
        @test isnan(rb.lower) && isnan(rb.upper)
        @test !rb.boundary
        @test rb.method === :failed

        # With estimated trait intercepts and X = nothing, the edge refit must
        # include the intercept mean, as the profile search does. An edge
        # placed halfway between the full-search lower bound and the estimate
        # lies inside the confidence region, so a NaN lower side resolves to
        # that edge. Without the intercepts the deviance there is huge and the
        # side stays NaN.
        yi = y .+ [1.0, 2.0, 3.0, 4.0]
        fi = fit_gllvm(yi; family = GLLVModels.Normal(), K = 1)
        fi_c1 = GLLVModels._make_communality_closure(GLLVModels._derived_spec(fi), 1)
        full = GLLVModels.profile_ci_derived(fi, fi_c1; y = yi)
        @test full.method === :profile
        edge_in = (full.lower + full.estimate) / 2
        ri = (lower = NaN, upper = full.upper, estimate = full.estimate, method = :partial)
        rbi = GLLVModels._profile_ci_bounded(fi, fi_c1, ri; level = 0.95, y = yi,
                                             X = nothing, Σ_phy = nothing,
                                             lo_bound = edge_in, hi_bound = 1.0)
        @test rbi.lower == edge_in
        @test rbi.boundary
        @test rbi.upper == full.upper
        @test rbi.method === :profile
    end

    @testset "#142 edge rules of _derived_bound_side (synthetic deviance)" begin
        # Pure-logic check of the per-side rule with a hand-made deviance.
        cutoff = 3.841458820694124            # qchisq(0.95, 1)
        no_call = c -> error("the deviance must not be evaluated here")
        side = GLLVModels._derived_bound_side
        # A finite bound inside the support is returned untouched.
        @test side(no_call, 0.3, 0.0, cutoff, true) == (0.3, false)
        @test side(no_call, 0.7, 1.0, cutoff, false) == (0.7, false)
        # A finite bound outside the support is set to the edge.
        @test side(no_call, -0.1, 0.0, cutoff, true) == (0.0, true)
        @test side(no_call, 1.2, 1.0, cutoff, false) == (1.0, true)
        # An infinite edge never changes the bound.
        r = side(no_call, NaN, Inf, cutoff, false)
        @test isnan(r[1]) && !r[2]
        # NaN bound, deviance at the edge below the cutoff: flat to the edge.
        @test side(c -> c == 0.0 ? 1.0 : NaN, NaN, 0.0, cutoff, true) == (0.0, true)
        # NaN bound, deviance at the edge above the cutoff: the crossing is
        # unknown, so the side stays NaN (refits fail everywhere but the edge).
        r = side(c -> c == 0.0 ? 10.0 : NaN, NaN, 0.0, cutoff, true)
        @test isnan(r[1]) && !r[2]
        # NaN bound, refit at the edge fails (NaN or Inf deviance): stays NaN.
        r = side(c -> NaN, NaN, 1.0, cutoff, false)
        @test isnan(r[1]) && !r[2]
        r = side(c -> Inf, NaN, 1.0, cutoff, false)
        @test isnan(r[1]) && !r[2]
    end

    @testset "#142 profile_ci_total_variance and profile_ci_phylo_signal use the natural bounds" begin
        # Phylogenetic signal near 1: a strong trait-level phylogenetic effect
        # and a small site-level variance. The raw profile upper bound
        # overshoots 1; the wrapper must return exactly 1 with boundary = true.
        rng = MersenneTwister(11)
        pp, Kp, np_ = 3, 1, 300
        Σ_phy = Matrix{Float64}(I, pp, pp)
        zp = randn(rng, Kp, np_)
        φ = randn(rng, pp)
        yp = reshape([0.1, 0.1, 0.1], pp, Kp) * zp .+ 0.05 .* randn(rng, pp, np_) .+
             3.0 .* φ
        fp = fit_gaussian_gllvm(yp; K = Kp, has_phy_unique = true, Σ_phy = Σ_phy)
        σ2phy = fp.pars.σ_phy[1]^2
        H2_hand = σ2phy / (σ2phy + fp.pars.Λ[1, 1]^2 + fp.pars.σ_eps^2)
        fH = GLLVModels._make_phylo_signal_closure(GLLVModels._derived_spec(fp), 1;
                                                   diag_Σphy = diag(Σ_phy))
        raw_H = GLLVModels.profile_ci_derived(fp, fH; y = yp, Σ_phy = Σ_phy)
        @test raw_H.upper > 1                     # the defect: outside [0, 1]
        ph = GLLVModels.profile_ci_phylo_signal(fp, 1; y = yp, Σ_phy = Σ_phy)
        @test ph.estimate ≈ H2_hand rtol = 1e-10
        @test ph.upper == 1.0
        @test ph.boundary
        @test ph.method === :profile
        @test ph.lower == raw_H.lower             # the interior side is untouched
        @test 0 ≤ ph.lower ≤ ph.estimate ≤ ph.upper

        # Total variance, interior: identical to the search without bounds.
        f_tv = GLLVModels._make_total_variance_closure(spec, 1)
        tv_hand = fit.pars.Λ[1, 1]^2 + fit.pars.σ_eps^2
        raw_tv = GLLVModels.profile_ci_derived(fit, f_tv; y = y)
        tv = GLLVModels.profile_ci_total_variance(fit, 1; y = y)
        @test tv.estimate ≈ tv_hand rtol = 1e-10
        @test (tv.lower, tv.upper) == (raw_tv.lower, raw_tv.upper)
        @test !tv.boundary
        @test tv.method === :profile
        @test 0 < tv.lower ≤ tv.estimate ≤ tv.upper

        # Total variance, truncated search (max_expand = 2): neither side
        # reaches the cutoff, and the lower edge 0 lies far outside the
        # confidence region found above, so both sides stay NaN.
        tv2 = GLLVModels.profile_ci_total_variance(fit, 1; y = y, max_expand = 2)
        @test isnan(tv2.lower) && isnan(tv2.upper)
        @test !tv2.boundary
        @test tv2.method === :failed
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
