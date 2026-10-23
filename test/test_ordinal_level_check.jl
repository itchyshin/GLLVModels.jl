using Test
using GLLVModels

# Observed ordinal levels must be integers >= 1. A level of 0 or below used to
# reach an `@inbounds` count loop (`counts[Int(Y[t, i])] += 1`) and write out of
# bounds. Every public ordinal fitter now rejects such data up front; masked
# cells keep an arbitrary placeholder and are not validated.
@testset "Ordinal fitters reject observed levels below 1" begin
    Y = [
        1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1
        1 2 3 3 1 2 3 2 1 2 3 1 1 2 3 2
        1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
    ]
    p, n = size(Y)
    X = reshape(repeat(collect(range(-1.0, 1.0; length = n))', p), p, n, 1)

    fitters = [
        ("fit_ordinal_gllvm", (Yv; kw...) -> fit_ordinal_gllvm(Yv; K = 1, kw...)),
        ("fit_ordinal_gllvm_pertrait",
         (Yv; kw...) -> fit_ordinal_gllvm_pertrait(Yv; K = 1, kw...)),
        ("fit_ordinal_gllvm_pertrait_cov",
         (Yv; kw...) -> fit_ordinal_gllvm_pertrait_cov(Yv; X = X, K = 1, kw...)),
    ]

    for (name, f) in fitters
        @testset "$name" begin
            for bad in (0, -1)
                Yb = copy(Y)
                Yb[2, 5] = bad
                @test_throws ArgumentError f(Yb)
                err = try
                    f(Yb); nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin(string(bad), err.msg)
                @test occursin("1..C", err.msg)
            end

            # A masked cell may hold an invalid placeholder: only observed
            # cells are validated.
            Ym = copy(Y)
            Ym[2, 5] = 0
            M = trues(p, n)
            M[2, 5] = false
            fm = f(Ym; mask = M)
            @test fm.converged
            @test isfinite(fm.loglik)

            # Valid data still fits.
            fv = f(Y)
            @test fv.converged
            @test isfinite(fv.loglik)
        end
    end
end
