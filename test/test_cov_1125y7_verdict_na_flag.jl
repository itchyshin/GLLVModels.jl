using GLLVModels, Test, Optim

@testset "cov 1125y7 fit verdict, NA guard, flag unreliable rows" begin
    @testset "_fit_verdict sentinel screen" begin
        @test GLLVModels._fit_verdict(1.0e12, true, 0) == (-Inf, false, 0)
        @test GLLVModels._fit_verdict(1.0e11, true, 3) == (-Inf, false, 3)
        @test GLLVModels._fit_verdict(1.0e11 - 1, true, 2)[1] ≈ -(1.0e11 - 1)
        @test GLLVModels._fit_verdict(1.0e11 - 1, true, 2)[2]
        @test GLLVModels._fit_verdict(NaN, false, 1) == (-Inf, false, 1)
        @test GLLVModels._fit_verdict(Inf, true, 0) == (-Inf, false, 0)
        @test GLLVModels._fit_verdict(100.5, true, 7) == (-100.5, true, 7)
        @test !GLLVModels._nll_failed(-1e6)
        @test GLLVModels._nll_failed(1e11)
        @test GLLVModels._nll_failed(Inf)
    end

    @testset "_fit_verdict Optim wrapper" begin
        f(x) = (x[1] - 2)^2
        res = Optim.optimize(f, [0.0], Optim.NelderMead())
        ll, conv, it = GLLVModels._fit_verdict(res)
        @test isfinite(ll)
        @test conv
        @test it >= 0
    end

    @testset "_refuse_na_response" begin
        Ym = [1.0 2.0; missing 4.0]
        @test_throws ArgumentError GLLVModels._refuse_na_response("em_fa", Ym)
        Yn = [1.0 NaN; 3.0 4.0]
        @test_throws ArgumentError GLLVModels._refuse_na_response("fit_coevolution_gaussian", Yn; nan = true)
        Yi = [1.0 Inf; 3.0 4.0]
        @test_throws ArgumentError GLLVModels._refuse_na_response("em_fa", Yi; nan = true)
        Yok = [1.0 2.0; 3.0 4.0]
        @test GLLVModels._refuse_na_response("em_fa", Yok; nan = true) === nothing
    end

    @testset "flag_unreliable_loadings rows API" begin
        rows = [(estimate = 0.0, lower = -0.05, upper = 0.05, pinned = false),
                (estimate = 0.5, lower = 0.2, upper = 0.8, pinned = false),
                (estimate = 1.0, lower = 1.0, upper = 1.0, pinned = true)]
        out = GLLVModels.flag_unreliable_loadings(rows; null_region = (-0.1, 0.1))
        @test length(out) == 3
        @test out[1].unreliable == true
        @test out[2].unreliable == false
        @test out[3].unreliable === missing
        @test out[1].null_region_lo == -0.1
        narrow = GLLVModels.flag_unreliable_loadings(rows; null_region = (0.9, 1.1))
        @test narrow[1].unreliable == false
        @test_throws ArgumentError GLLVModels.flag_unreliable_loadings(
            [(estimate = 1.0, lower = 0.0, pinned = false)])
        @test_throws ArgumentError GLLVModels.flag_unreliable_loadings(rows; null_region = (1, 0))
        @test_throws ArgumentError GLLVModels.flag_unreliable_loadings(rows; null_region = (0,))
        na_row = [(estimate = 1.0, lower = NaN, upper = 0.2, pinned = false)]
        @test GLLVModels.flag_unreliable_loadings(na_row)[1].unreliable === missing
    end
end