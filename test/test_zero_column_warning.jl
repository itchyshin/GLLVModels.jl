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
