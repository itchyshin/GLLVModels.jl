# Regression for #1108. On main these calls are a MethodError: the VA fitters
# are typed AbstractMatrix{<:Integer} or {<:Real}, so a Matrix{Union{Missing,T}}
# never dispatches. The maintainer decision (2026-11-10) is to derive the
# observation mask and drop those cells, with one @info (maxlog = 1).

using GLLVModels, Test, Random, Logging

const _VA1108 = (K = 1, iterations = 2, g_tol = 1e-3)

function _with_missing(Y)
    Ym = Matrix{Union{eltype(Y),Missing}}(Y)
    Ym[1, 1] = missing
    return Ym
end

function _drop_first(Y)
    msk = trues(size(Y))
    msk[1, 1] = false
    return msk
end

# A missing cell and an explicit mask that drops a different value in that
# cell must take the same fit: the cell is not in the ELBO or the warm start.
function _same_when_dropped(fitter, Y, Yalt; kwargs...)
    Ym = _with_missing(Y)
    msk = _drop_first(Y)
    fm = fitter(Ym; kwargs...)
    fk = fitter(Y; mask = msk, kwargs...)
    fa = fitter(Yalt; mask = msk, kwargs...)
    @test fm isa typeof(fk)
    @test isfinite(fm.loglik)
    @test fm.loglik ≈ fk.loglik atol = 1e-8
    @test fm.loglik ≈ fa.loglik atol = 1e-8
    return fm
end

@testset "Fix #1108: VA fitters derive a mask from missing in Y" begin
    Random.seed!(1108)

    @testset "Poisson derives the mask, says so once, and drops the cell" begin
        Y = rand(0:4, 4, 8)
        Yalt = copy(Y)
        Yalt[1, 1] = Y[1, 1] == 4 ? 0 : Y[1, 1] + 1
        Ym = _with_missing(Y)
        logger = TestLogger()
        fm = with_logger(logger) do
            fit_poisson_gllvm_va(Ym; _VA1108...)
        end
        infos = filter(r -> r.level == Logging.Info &&
                            occursin("derived an observation mask", r.message),
                       logger.logs)
        @test length(infos) == 1
        @test occursin("fit_poisson_gllvm_va", infos[1].message)
        @test isfinite(fm.loglik)

        msk = _drop_first(Y)
        fk = fit_poisson_gllvm_va(Y; mask = msk, _VA1108...)
        fa = fit_poisson_gllvm_va(Yalt; mask = msk, _VA1108...)
        @test fm.loglik ≈ fk.loglik atol = 1e-8
        @test fm.loglik ≈ fa.loglik atol = 1e-8
        @test fm.β ≈ fk.β atol = 1e-8

        # maxlog = 1 is honoured by ConsoleLogger (TestLogger records every event).
        io = IOBuffer()
        with_logger(ConsoleLogger(io, Logging.Info)) do
            fit_poisson_gllvm_va(Ym; _VA1108...)
            fit_poisson_gllvm_va(Ym; _VA1108...)
        end
        @test count("derived an observation mask", String(take!(io))) == 1

        # A Union eltype with no missing stored is the plain fit, and stays quiet.
        Yu = Matrix{Union{Int,Missing}}(Y)
        logger3 = TestLogger()
        fu = with_logger(logger3) do
            fit_poisson_gllvm_va(Yu; _VA1108...)
        end
        fp = fit_poisson_gllvm_va(Y; _VA1108...)
        @test fu.loglik ≈ fp.loglik atol = 1e-8
        @test fu.β ≈ fp.β atol = 1e-8
        @test !any(r -> occursin("derived an observation mask", r.message), logger3.logs)

        # Laplace Poisson already derived a mask; that control still fits.
        fl = fit_poisson_gllvm(Ym; K = 1, iterations = 2)
        @test isfinite(fl.loglik)
    end

    @testset "Binomial, NB, Gamma, Beta, Exponential, Delta-Gamma" begin
        # The one-time @info already fired above; keep the rest of the log quiet.
        with_logger(NullLogger()) do
        Yb = rand(0:1, 4, 6)
        Ybalt = copy(Yb)
        Ybalt[1, 1] = 1 - Yb[1, 1]
        _same_when_dropped(fit_binomial_gllvm_va, Yb, Ybalt; _VA1108...)

        Yn = rand(0:3, 4, 6)
        Ynalt = copy(Yn)
        Ynalt[1, 1] = Yn[1, 1] == 3 ? 0 : Yn[1, 1] + 1
        _same_when_dropped(fit_nb_gllvm_va, Yn, Ynalt; _VA1108...)

        Yg = rand(4, 6) .+ 0.3
        Ygalt = copy(Yg)
        Ygalt[1, 1] = Yg[1, 1] + 0.5
        _same_when_dropped(fit_gamma_gllvm_va, Yg, Ygalt; _VA1108...)
        _same_when_dropped(fit_exponential_gllvm_va, Yg, Ygalt; _VA1108...)

        Ye = rand(4, 6) .* 0.6 .+ 0.2
        Yealt = copy(Ye)
        Yealt[1, 1] = Ye[1, 1] > 0.5 ? 0.2 : 0.8
        _same_when_dropped(fit_beta_gllvm_va, Ye, Yealt; _VA1108...)

        Yd = rand(4, 6)
        Yd[Yd .< 0.4] .= 0.0
        Ydalt = copy(Yd)
        Ydalt[1, 1] = Yd[1, 1] == 0 ? 0.7 : 0.0
        _same_when_dropped(fit_delta_gamma_gllvm_va, Yd, Ydalt; _VA1108...)
        end
    end
end
