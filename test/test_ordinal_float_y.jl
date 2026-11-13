using Test
using GLLVModels
using Random

# Ordinal fitters are typed on integer matrices, so a Float64 matrix holding the same codes
# (what readdlm or a CSV reader returns) ended in a MethodError naming an internal function
# (#1191). Integer-valued floats (3.0) are now accepted and give the same fit as the Int
# matrix; a non-integer float is an ArgumentError naming the first offending cell; an
# integer-valued code outside 1..C (0.0) keeps its own out-of-range error.
@testset "ordinal accepts integer-valued float codes (#1191)" begin
    Random.seed!(1)
    p, n = 4, 40
    Y = rand(1:3, p, n)
    Yf = Float64.(Y)
    X = reshape(repeat(collect(range(-1.0, 1.0; length = n))', p), p, n, 1)
    M = trues(p, n)
    M[2, 5] = false

    fitters = [
        ("fit_ordinal_gllvm", f -> fit_ordinal_gllvm(f; K = 1)),
        ("fit_ordinal_gllvm_pertrait", f -> fit_ordinal_gllvm_pertrait(f; K = 1)),
        ("fit_ordinal_gllvm_pertrait_cov", f -> fit_ordinal_gllvm_pertrait_cov(f; X = X, K = 1)),
        ("fit_gllvm Ordinal", f -> fit_gllvm(f; family = Ordinal(), K = 1)),
    ]
    for (name, fit) in fitters
        @testset "$name" begin
            fi = fit(Y)
            ff = fit(Yf)
            @test ff.loglik == fi.loglik
            @test ff.converged == fi.converged

            Yh = copy(Yf)
            Yh[2, 5] = 1.5
            err = try; fit(Yh); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("Y[2, 5] = 1.5", err.msg)
            @test occursin("integer ordinal level", err.msg)

            Yn = copy(Yf)
            Yn[3, 7] = NaN
            @test_throws ArgumentError fit(Yn)

            # out-of-range integer-valued code: the existing level error, not the float one
            Yz = copy(Yf)
            Yz[2, 5] = 0.0
            err = try; fit(Yz); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("not a valid ordinal level", err.msg)
        end
    end

    @testset "mask hides a non-integer placeholder" begin
        Ym = copy(Yf)
        Ym[2, 5] = 1.5
        fm = fit_ordinal_gllvm_pertrait(Ym; K = 1, mask = M)
        fi = fit_ordinal_gllvm_pertrait(Y; K = 1, mask = M)
        @test fm.loglik == fi.loglik
    end
end
