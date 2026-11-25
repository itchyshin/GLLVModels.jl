using GLLVModels, Test

struct _Y7FakeBarFit
    ll::Float64
    Λ::Matrix{Float64}
    loading_ridge::Float64
end
GLLVModels._loglik(f::_Y7FakeBarFit) = f.ll

@testset "cov 1125y7 select_lv internal helpers" begin
    @testset "_lv_aicc" begin
        @test GLLVModels._lv_aicc(100.0, 5, 20) ≈ 100.0 + 2 * 5 * 6 / 14
        @test isnan(GLLVModels._lv_aicc(100.0, 5, 5))
        @test isnan(GLLVModels._lv_aicc(100.0, 10, 10))
    end

    @testset "_lv_warm_start" begin
        prev = (β = [1.0, 2.0], Λ = [0.5 0.0; 0.1 0.3; 0.0 0.2])
        @test GLLVModels._lv_warm_start(prev, 2) === nothing
        @test GLLVModels._lv_warm_start((β = [1.0],), 3) === nothing
        ws = GLLVModels._lv_warm_start(prev, 3)
        @test ws !== nothing
        @test ws.β_init == prev.β
        @test size(ws.Λ_init) == (3, 3)
        @test ws.Λ_init[:, 1:2] == prev.Λ
        @test ws.Λ_init[3, 3] == 0.1
        @test ws.Λ_init[1, 3] == 0.0
        @test GLLVModels._lv_warm_start(prev, 5) === nothing
    end

    @testset "_lv_bar_value" begin
        fit_ridge = _Y7FakeBarFit(-100.0, fill(2.0, 2, 2), 1.0)
        @test GLLVModels._lv_bar_value(fit_ridge) ≈ -100.0 - 0.5 * 16 / 1.0
        fit_plain = _Y7FakeBarFit(-50.0, zeros(2, 2), Inf)
        @test GLLVModels._lv_bar_value(fit_plain) ≈ -50.0
    end

    @testset "_lv_runaway" begin
        Λbig = zeros(3, 2)
        Λbig[1, :] .= 12.0
        fit = (Λ = Λbig, link = GLLVModels.LogitLink())
        msg = GLLVModels._lv_runaway(fit, GLLVModels.Binomial(); max_latent_sd = 4.0, ratio_max = 25.0)
        @test occursin("latent SD", msg)
        fit_g = (Λ = Λbig,)
        @test GLLVModels._lv_runaway(fit_g, GLLVModels.Normal(); max_latent_sd = 4.0, ratio_max = 25.0) == ""
        Λsep = [0.1 0.0; 8.0 0.0; 0.2 0.1]
        fit2 = (Λ = Λsep, link = GLLVModels.LogitLink())
        msg2 = GLLVModels._lv_runaway(fit2, GLLVModels.Binomial(); max_latent_sd = 20.0, ratio_max = 5.0)
        @test occursin("loading ratio", msg2) || occursin("latent SD", msg2)
        @test GLLVModels._lv_runaway((Λ = zeros(0, 0),), GLLVModels.Poisson(); max_latent_sd = 4.0, ratio_max = 25.0) == ""
    end

    @testset "_lv_format_column" begin
        xs = [1.23456, NaN, -0.5, Inf]
        txt = GLLVModels._lv_format_column(xs)
        @test length(txt) == 4
        @test occursin("NA", txt[2])
        @test occursin("NA", txt[4])
        @test txt[1] isa String
    end
end