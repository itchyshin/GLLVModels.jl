using GLLVModels, Test, Random, LinearAlgebra

# Derived-profile CI sweep (#137 constraint gate, #142 natural-boundary clamp).
# Pinned values were computed on the unmodified tree.
@testset "derived CI sweep" begin
    Random.seed!(11)
    p, K, n = 4, 1, 80
    Λ = reshape([0.8, 0.6, 0.5, -0.4], p, K)
    y = Λ * randn(K, n) + 0.5 * randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    spec = GLLVModels._derived_spec(fit)
    f_c1 = GLLVModels._make_communality_closure(spec, 1)
    g_hat = f_c1(fit.pars.θ_packed)

    @testset "#137 refit that misses g(θ)=c by > 0.05 is a failure" begin
        # A weak penalty leaves g at ≈0.711 against a target of 0.99.
        ll, ok, _, g_at = GLLVModels._derived_refit_with_fixed(
            fit, f_c1, 0.99, y, nothing, nothing; penalty_schedule = [1.0])
        @test abs(g_at - 0.99) > 0.05 || isnan(g_at)
        @test !ok
        @test isnan(ll)
    end

    @testset "#137 control: ordinary refits unchanged" begin
        @test g_hat ≈ 0.7101994105891059 rtol = 1e-6
        ll, ok, _, g_at = GLLVModels._derived_refit_with_fixed(
            fit, f_c1, g_hat - 0.05, y, nothing, nothing)
        @test ok
        @test ll ≈ -281.217871105688 rtol = 1e-6
        @test g_at ≈ 0.6602157808904908 rtol = 1e-5
        ci = GLLVModels.profile_ci_derived(fit, f_c1; y = y)
        @test ci.method === :profile
        @test ci.lower ≈ 0.5976702559327752 rtol = 1e-5
        @test ci.upper ≈ 0.7971364185161999 rtol = 1e-5
    end

    @testset "#142 communality/correlation profile CIs clamp to natural limits" begin
        # Near-singular one-factor data: communality of trait 1 is ≈ 1 and the
        # raw profile upper bound overshoots 1 (1.0575 before the clamp).
        Random.seed!(11)
        y2 = [1.0; 1.0; 0.5; 0.2] * randn(1, 80) + 0.02 * randn(4, 80)
        fit2 = fit_gaussian_gllvm(y2; K = 1)
        ci = GLLVModels.profile_ci_communality(fit2, 1; y = y2)
        @test ci.upper ≤ 0.999 + 1e-12      # R's q_hi_ceiling
        @test ci.boundary
        @test ci.lower ≥ 0.001
        @test ci.lower < ci.upper
        @test ci.method === :profile
        # correlation limits are ±0.999
        cr = GLLVModels.profile_ci_correlation(fit2, 1, 2; y = y2)
        @test -0.999 - 1e-12 ≤ cr.lower
        @test cr.upper ≤ 0.999 + 1e-12
    end

    @testset "#142 control: interior results equal profile_ci_derived" begin
        ci = GLLVModels.profile_ci_communality(fit, 1; y = y)
        @test !ci.boundary
        @test ci.lower ≈ 0.5976702559327752 rtol = 1e-5
        @test ci.upper ≈ 0.7971364185161999 rtol = 1e-5
        cr = GLLVModels.profile_ci_correlation(fit, 1, 2; y = y)
        @test !cr.boundary
        @test cr.lower ≈ 0.5218132123952197 rtol = 1e-5
        @test cr.upper ≈ 0.7335094545985572 rtol = 1e-5
    end
end
