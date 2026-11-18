using Test
using GLLVModels

# A Y with the wrong number of rows (traits) used to reach the @inbounds per-site mode
# search of the ordinal fits and read past the end of the column, ending in a BoundsError
# with a garbage index (or silently wrong values under --check-bounds=no). It is now a
# DimensionMismatch from getLV, and so from predict and residuals, which call getLV.
@testset "ordinal post-fit refuses a Y with the wrong number of rows" begin
    Y = [
        1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2
        1 2 3 3 1 2 3 2 1 2 3 1 1 2 3 2 3 1 2 3
        1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 2 1 3 3
        3 1 2 2 3 1 1 2 3 3 2 1 2 3 1 2 1 3 2 1
    ]
    p, n = size(Y)
    X = reshape(repeat(collect(range(-1.0, 1.0; length = n))', p), p, n, 1)
    Yw = Y[1:(end - 1), :]
    Xw = X[1:(end - 1), :, :]

    fits = [
        ("OrdinalFit", fit_ordinal_gllvm(Y; K = 1), (f, Yv) -> (getLV(f, Yv), predict(f, Yv), residuals(f, Yv))),
        ("OrdinalPerTraitFit", fit_ordinal_gllvm_pertrait(Y; K = 1), (f, Yv) -> (getLV(f, Yv), predict(f, Yv), residuals(f, Yv))),
    ]
    for (name, fit, _) in fits
        @testset "$name" begin
            @test_throws DimensionMismatch getLV(fit, Yw)
            @test_throws DimensionMismatch predict(fit, Yw)
            @test_throws DimensionMismatch residuals(fit, Yw)
            # the correct shape still works
            @test size(getLV(fit, Y)) == (n, 1)
        end
    end
    cfit = fit_ordinal_gllvm_pertrait_cov(Y; X = X, K = 1)
    @testset "OrdinalPerTraitCovFit" begin
        @test_throws DimensionMismatch getLV(cfit, Yw, Xw)
        @test size(getLV(cfit, Y, X)) == (n, 1)
    end
end
