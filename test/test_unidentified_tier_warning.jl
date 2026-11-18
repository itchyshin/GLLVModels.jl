using GLLVModels, Test, Random

# #792 (has_diag with one observation per unit) and #702 (K_W > 0): the unit and
# within-unit tiers are not identified, so the fitter warns once per session.
@testset "unidentified tier split warns (#792, #702)" begin
    Random.seed!(792)
    p, K, n = 4, 1, 60
    Λ = reshape([0.7, 0.5, 0.3, -0.2], p, K)
    y = Λ * randn(K, n) + 0.5 * randn(p, n)
    # On base this helper does not exist; skip the reset so the forcing test
    # fails on the missing warning (assertion), not UndefVarError.
    reset! = isdefined(GLLVModels, :_reset_unidentified_warnings!) ?
        GLLVModels._reset_unidentified_warnings! : () -> nothing

    @testset "has_diag = true warns about σ²_B / σ²_W (#792)" begin
        reset!()
        @test_logs (:warn, r"not identified") min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(y; K = K, has_diag = true)
    end

    @testset "K_W > 0 warns that tier-split quantities are not identified (#702)" begin
        reset!()
        @test_logs (:warn, r"K_W > 0.*not separately identified") min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(y; K = K, K_W = 1)
    end

    @testset "warns once per session, not on every refit" begin
        reset!()
        fit_gaussian_gllvm(y; K = K, has_diag = true)
        @test_logs min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(y; K = K, has_diag = true)
    end

    @testset "default fit (K_W = 0, has_diag = false) stays silent" begin
        reset!()
        @test_logs min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(y; K = K)
    end
end
