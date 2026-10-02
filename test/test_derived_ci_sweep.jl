using GLLVModels, Test, Random, LinearAlgebra

# Derived-profile CI sweep (#137 constraint gate).
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
end
