using Test
using Random
using GLLVModels

# Issue #1432: ordinal fit accepts Float64 and Union{Missing,Int} Y, but post-fit used to
# require Matrix{Int}.
@testset "ordinal post-fit accepts the same Y eltypes as fit (#1432)" begin
    Random.seed!(2601)
    p, n = 4, 45
    Y = rand(1:3, p, n)
    Yf = float(Y)
    Ym = Matrix{Union{Missing, Int}}(Y)
    Ym[2, 3] = missing

    fit = fit_gllvm(Yf; family = Ordinal(), K = 1, iterations = 200)
    @test fit.converged
    @test isfinite(fit.loglik)
    predict(fit, Yf; type = :response)
    fitted(fit, Yf)
    residuals(fit, Yf)
    getLV(fit, Yf)
    predict_ref = predict(fit, Y; type = :class)
    @test predict(fit, Yf; type = :class) == predict_ref

    fitm = fit_gllvm(Ym; family = Ordinal(), K = 1, iterations = 200)
    predict(fitm, Ym; type = :class)
    fitted(fitm, Ym)
    residuals(fitm, Ym)
    getLV(fitm, Ym)

    Yh = copy(Yf)
    Yh[1, 2] = 1.7
    @test_throws ArgumentError predict(fit, Yh; type = :response)
    err = try
        predict(fit, Yh; type = :response)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Y[1, 2] = 1.7", err.msg)
    @test occursin("integer ordinal level", err.msg)
end