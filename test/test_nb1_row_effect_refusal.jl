using GLLVModels, Test, Random, Distributions

# NB1 with a fixed or random row effect is refused before any fitting (#1238). NB1 is a
# plain marker type, not a `Distributions.Distribution`, so the row-effect result structs
# cannot hold it; on main the fit ran to the end and then stopped with a MethodError.
# Structure, not timing: `iterations = 0` and a NaN cell would each make a fit do
# something other than throw an ArgumentError, so only a refusal made before the
# optimiser starts produces the ArgumentError asserted here.
@testset "NB1 with a row effect is refused up front (#1238)" begin
    rng = MersenneTwister(1238)
    p, n = 5, 12
    Y = rand(rng, Poisson(4.0), p, n)
    Ynan = float.(Y); Ynan[2, 3] = NaN
    nb1 = GLLVModels.NB1()

    function refusal(f)
        err = try f(); nothing catch e e end
        return err
    end
    function check_message(err, caller)
        @test err isa ArgumentError
        err isa ArgumentError || return
        msg = err.msg
        @test occursin(caller, msg)
        @test occursin("NB1()", msg)              # names the family
        @test occursin("row_eff", msg)            # names the argument
        @test occursin("Poisson()", msg) && occursin("NegativeBinomial()", msg)  # says what is supported
    end

    for (Yuse, tag) in ((Y, "counts"), (Ynan, "NaN cell"))
        @testset "direct fitters, $tag" begin
            check_message(refusal(() -> fit_roweffect_gllvm(Yuse; family = nb1, K = 1, iterations = 0)),
                          "fit_roweffect_gllvm")
            check_message(refusal(() -> fit_row_random_gllvm(Yuse; family = nb1, K = 1, iterations = 0)),
                          "fit_row_random_gllvm")
        end
        @testset "fit_gllvm, $tag" begin
            for re in (:fixed, :random)
                check_message(refusal(() -> fit_gllvm(Yuse; family = nb1, K = 1, row_eff = re,
                                                      iterations = 0)), "fit_gllvm")
            end
        end
    end

    @testset "formula front end" begin
        for re in (:fixed, :random)
            check_message(refusal(() -> gllvm(@formula(y ~ 1), Y, (site = 1:n,);
                                              family = nb1, K = 1, row_eff = re, iterations = 0)),
                          "fit_gllvm")
        end
    end

    @testset "NB1 without a row effect, and NB2 with one, are unchanged" begin
        @test refusal(() -> fit_roweffect_gllvm(Y; family = NegativeBinomial(), K = 1,
                                                iterations = 2)) === nothing
        @test refusal(() -> fit_row_random_gllvm(Y; family = NegativeBinomial(), K = 1,
                                                 iterations = 2)) === nothing
        @test refusal(() -> fit_gllvm(Y; family = nb1, K = 1, iterations = 2)) === nothing
    end
end
