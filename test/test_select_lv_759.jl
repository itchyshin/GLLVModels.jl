# Issue #759: selection must not pick NaN criteria or non-converged accepted fits;
# failed K must not win; all-fail errors name select_lv.
using GLLVModels, Test

struct _Sel759Fit
    ll::Float64
    np::Int
    converged::Bool
    aic_val::Float64
    bic_val::Float64
    β::Vector{Float64}
    Λ::Matrix{Float64}
end
GLLVModels._loglik(f::_Sel759Fit) = f.ll
GLLVModels._nparams(f::_Sel759Fit) = f.np
GLLVModels.StatsAPI.aic(f::_Sel759Fit) = f.aic_val
GLLVModels.StatsAPI.bic(f::_Sel759Fit, ::AbstractMatrix; mask = nothing) = f.bic_val
GLLVModels.StatsAPI.bic(f::_Sel759Fit, n::Integer) = f.bic_val

function _sel759_fit(p, K; ll, converged, aic_val, bic_val)
    return _Sel759Fit(ll, 10K, converged, aic_val, bic_val, zeros(p), fill(0.5, p, K))
end

@testset "select_lv issue 759 — eligible selection only" begin
    Y = zeros(4, 20)

    @testset "NaN criterion is skipped when choosing best_k" begin
        tbl = Dict(
            1 => (ll = -500.0, aic = 200.0, bic = 210.0),
            2 => (ll = -400.0, aic = NaN, bic = NaN),
            3 => (ll = -390.0, aic = 50.0, bic = 55.0),
        )
        f = function (Y; family, K, kwargs...)
            row = tbl[K]
            return _sel759_fit(size(Y, 1), K; ll = row.ll, converged = true,
                               aic_val = row.aic, bic_val = row.bic)
        end
        sel = select_lv(Y; Kmax = 3, criterion = :aic, warm_start = false, _fitter = f)
        @test sel.K == [1, 2, 3]
        @test sel.best_k == 3
        @test isfinite(sel.aic[findfirst(==(3), sel.K)])
    end

    @testset "non-converged accepted fit cannot win best_k" begin
        tbl = Dict(
            1 => (ll = -500.0, aic = 200.0, conv = true),
            2 => (ll = -400.0, aic = 10.0, conv = false),
            3 => (ll = -395.0, aic = 80.0, conv = true),
        )
        f = function (Y; family, K, kwargs...)
            row = tbl[K]
            return _sel759_fit(size(Y, 1), K; ll = row.ll, converged = row.conv,
                               aic_val = row.aic, bic_val = row.aic + 5)
        end
        sel = select_lv(Y; Kmax = 3, criterion = :aic, warm_start = false, _fitter = f)
        @test 2 in sel.K
        @test sel.best_k == 3
    end

    @testset "every candidate failing names select_lv" begin
        f = function (Y; family, K, kwargs...)
            throw(ErrorException("stand-in failure at K=$K"))
        end
        err = nothing
        try
            select_lv(Y; Kmax = 2, _fitter = f)
        catch e
            err = e
        end
        @test err !== nothing
        @test occursin("select_lv", sprint(showerror, err))
    end
end
