# All-zero count column warning on the Poisson and ZIP fitters.
# A species whose observed cells are all zero has no finite maximum-likelihood intercept; the
# fit used to report `converged = true` with no message. The warning is the only change: it
# names the column, and sparse columns (one or a few non-zero cells) stay silent.
using Test, GLLVModels, Distributions, Random, SparseArrays, Logging

_zc_pattern = r"species column\(s\) \[(\d+(, \d+)*)\] have no non-zero count in any observed cell"

# Warnings matching the all-zero-column text emitted while `f` runs (a TestLogger records every
# event, so `maxlog = 1` does not hide repeats here).
function _zc_warnings(f)
    logger = Test.TestLogger(min_level = Logging.Warn)
    res = Logging.with_logger(f, logger)
    msgs = [string(l.message) for l in logger.logs if l.level == Logging.Warn]
    return res, filter(m -> occursin(_zc_pattern, m), msgs)
end

function _zc_counts(; p = 6, n = 50, seed = 20261113)
    rng = Random.Xoshiro(seed)
    return [rand(rng, Poisson(exp(0.8 + 0.5 * sin(i + 2j)))) for i in 1:p, j in 1:n]
end

@testset "all-zero count column warning" begin
    Y0 = _zc_counts()
    Yz = copy(Y0); Yz[2, :] .= 0                    # species 2 all zero
    Y1 = copy(Yz); Y1[2, 7] = 1                     # one non-zero cell: sparse, not constant
    Y3 = copy(Yz); Y3[2, [3, 20, 41]] .= [1, 2, 1]  # three non-zero cells

    @testset "Poisson" begin
        _, w = _zc_warnings(() -> fit_poisson_gllvm(Yz; K = 1))
        @test length(w) == 1
        @test occursin("fit_poisson_gllvm: species column(s) [2]", only(w))
        # the same message through fit_gllvm, and with two all-zero columns
        _, w = _zc_warnings(() -> fit_gllvm(Yz; family = Poisson(), K = 1))
        @test length(w) == 1
        Y2z = copy(Yz); Y2z[5, :] .= 0
        _, w = _zc_warnings(() -> fit_poisson_gllvm(Y2z; K = 1))
        @test length(w) == 1 && occursin("[2, 5]", only(w))
        # n_init = 2 reaches the same fitter: warns at least once
        _, w = _zc_warnings(() -> fit_poisson_gllvm(Yz; K = 1, n_init = 2))
        @test length(w) >= 1
        # mask: all observed cells zero warns even if a masked cell is non-zero
        M = trues(size(Y1)); M[2, 7] = false
        _, w = _zc_warnings(() -> fit_poisson_gllvm(Y1; K = 1, mask = M))
        @test length(w) == 1 && occursin("[2]", only(w))
        # a column with no observed cell at all is not an all-zero column
        Mall = trues(size(Yz)); Mall[2, :] .= false
        _, w = _zc_warnings(() -> fit_poisson_gllvm(Yz; K = 1, mask = Mall))
        @test isempty(w)
    end

    @testset "ZIP" begin
        _, w = _zc_warnings(() -> fit_zip_gllvm(Yz; K = 1))
        @test length(w) == 1
        @test occursin("fit_zip_gllvm: species column(s) [2]", only(w))
        _, w = _zc_warnings(() -> fit_gllvm(Yz; family = ZIPoisson(), K = 1))
        @test length(w) == 1        # the Fisher restart and the nested Poisson fit stay quiet
    end

    @testset "sparse columns stay silent and fit as before" begin
        for Y in (Y0, Y1, Y3)
            _, w = _zc_warnings(() -> fit_poisson_gllvm(Y; K = 1))
            @test isempty(w)
            _, w = _zc_warnings(() -> fit_zip_gllvm(Y; K = 1))
            @test isempty(w)
        end
        # SparseMatrixCSC input: same fit as the dense matrix, no warning when a column has a
        # non-zero cell; the all-zero column is still found.
        S1 = sparse(Y1)
        fd, wd = _zc_warnings(() -> fit_poisson_gllvm(Y1; K = 1))
        fs, ws = _zc_warnings(() -> fit_poisson_gllvm(S1; K = 1))
        @test isempty(wd) && isempty(ws)
        @test fs.loglik == fd.loglik && fs.β == fd.β && fs.Λ == fd.Λ
        _, ws = _zc_warnings(() -> fit_poisson_gllvm(sparse(Yz); K = 1))
        @test length(ws) == 1
    end

    @testset "the warning changes no number" begin
        fa, _ = _zc_warnings(() -> fit_poisson_gllvm(Yz; K = 1))
        fb = fit_poisson_gllvm(Yz; K = 1, _warn_zero_columns = false)
        @test fa.loglik == fb.loglik && fa.β == fb.β && fa.Λ == fb.Λ && fa.converged == fb.converged
        za, _ = _zc_warnings(() -> fit_zip_gllvm(Yz; K = 1))
        zb = fit_zip_gllvm(Yz; K = 1, _warn_zero_columns = false)
        @test za.loglik == zb.loglik && za.βc == zb.βc && za.βz == zb.βz && za.Λc == zb.Λc
    end
end

# ---- once per public call, once per distinct (fitter, columns) per session ------------------
# `maxlog` is applied by the logger, so a TestLogger or a ConsoleLogger hides repeats inside
# one logger. `_ZcRecorder` ignores `maxlog` and keeps every record, which is the only honest
# way to count how many times a fit calls the warning.
struct _ZcRecorder <: Logging.AbstractLogger
    msgs::Vector{String}
end
_ZcRecorder() = _ZcRecorder(String[])
Logging.min_enabled_level(::_ZcRecorder) = Logging.Warn
Logging.shouldlog(::_ZcRecorder, args...) = true
Logging.handle_message(l::_ZcRecorder, level, msg, args...; kwargs...) =
    push!(l.msgs, string(msg))

function _zc_records(f)
    rec = _ZcRecorder()
    res = Logging.with_logger(f, rec)
    return res, filter(m -> occursin(_zc_pattern, m), rec.msgs)
end

# Messages seen by ONE maxlog-honouring logger across several fits (what a user's console sees).
function _zc_session(fs...)
    logger = Test.TestLogger(min_level = Logging.Warn)
    Logging.with_logger(logger) do
        foreach(f -> f(), fs)
    end
    return filter(m -> occursin(_zc_pattern, m),
                  [string(l.message) for l in logger.logs if l.level == Logging.Warn])
end

@testset "all-zero column warning: per distinct fitter and columns, once per public call" begin
    Y0 = _zc_counts()
    YA = copy(Y0); YA[2, :] .= 0                      # columns [2]
    YB = copy(Y0); YB[1, :] .= 0; YB[3, :] .= 0       # columns [1, 3]

    @testset "a second data set with other all-zero columns warns, naming its own columns" begin
        w = _zc_session(() -> fit_poisson_gllvm(YA; K = 1), () -> fit_poisson_gllvm(YB; K = 1))
        @test length(w) == 2
        @test occursin("species column(s) [2]", w[1])
        @test occursin("species column(s) [1, 3]", w[2])
        # the same through a ConsoleLogger, which also applies maxlog itself
        io = IOBuffer()
        Logging.with_logger(Logging.ConsoleLogger(io, Logging.Warn)) do
            fit_poisson_gllvm(YA; K = 1); fit_poisson_gllvm(YB; K = 1)
        end
        txt = String(take!(io))
        @test occursin("species column(s) [2]", txt) && occursin("species column(s) [1, 3]", txt)
    end

    @testset "Poisson and ZIP are separate messages" begin
        w = _zc_session(() -> fit_poisson_gllvm(YA; K = 1), () -> fit_zip_gllvm(YA; K = 1))
        @test length(w) == 2
        @test occursin("fit_poisson_gllvm", w[1]) && occursin("fit_zip_gllvm", w[2])
        w = _zc_session(() -> fit_zip_gllvm(YA; K = 1), () -> fit_zip_gllvm(YB; K = 1))
        @test length(w) == 2 && occursin("[2]", w[1]) && occursin("[1, 3]", w[2])
    end

    @testset "the same data fitted twice warns once in a session" begin
        w = _zc_session(() -> fit_poisson_gllvm(YA; K = 1), () -> fit_poisson_gllvm(YA; K = 1))
        @test length(w) == 1
        w = _zc_session(() -> fit_zip_gllvm(YA; K = 1), () -> fit_zip_gllvm(YA; K = 1))
        @test length(w) == 1
    end

    @testset "one public call emits exactly one record (maxlog ignored)" begin
        for (label, f) in (
                ("poisson", () -> fit_poisson_gllvm(YA; K = 1)),
                ("poisson n_init = 3", () -> fit_poisson_gllvm(YA; K = 1, n_init = 3)),
                ("poisson n_init = 3, K = 2", () -> fit_poisson_gllvm(YA; K = 2, n_init = 3)),
                ("fit_gllvm poisson n_init = 3",
                 () -> fit_gllvm(YA; family = Poisson(), K = 1, n_init = 3)),
                ("poisson aghq = 5", () -> fit_poisson_gllvm(YA; K = 1, aghq = 5)),
                ("zip", () -> fit_zip_gllvm(YA; K = 1)),
                ("zip fisher", () -> fit_zip_gllvm(YA; K = 1, hessian = :fisher)))
            _, w = _zc_records(f)
            @test (label, length(w)) == (label, 1)
        end
    end

    @testset "bootstrap refits do not repeat the warning" begin
        fit = fit_poisson_gllvm(YA; K = 1)
        _, w = _zc_records(() -> confint(fit, YA; method = :bootstrap, n_boot = 4, seed = 1))
        @test isempty(w)
        zfit = fit_zip_gllvm(YA; K = 1)
        _, w = _zc_records(() -> confint(zfit, YA; method = :bootstrap, n_boot = 4, seed = 1))
        @test isempty(w)
    end

    @testset "sparse columns stay silent in every one of these routes" begin
        Ys = copy(YA); Ys[2, 7] = 1
        for f in (() -> fit_poisson_gllvm(Ys; K = 1, n_init = 3),
                  () -> fit_poisson_gllvm(Ys; K = 1, aghq = 5),
                  () -> fit_zip_gllvm(Ys; K = 1),
                  () -> fit_gllvm(Ys; family = Poisson(), K = 1, n_init = 3))
            _, w = _zc_records(f)
            @test isempty(w)
        end
    end
end
