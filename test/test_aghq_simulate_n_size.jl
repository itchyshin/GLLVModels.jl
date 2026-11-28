using GLLVModels, Test, Random, Distributions

# refs #1367: AGHQ binomial simulate uses DimensionMismatch for wrong-sized N (aligned with #1198).
@testset "AGHQ binomial simulate refuses wrong-sized N with DimensionMismatch" begin
    rng = MersenneTwister(3)
    p, n = 4, 80
    Y = [rand(rng, Binomial(3, 0.4)) for t in 1:p, s in 1:n]
    fb = fit_binomial_gllvm(Y; K = 1, N = fill(3, p, n), aghq = 3)
    err() = try
        simulate(fb, 10; N = fill(3, 2, 10))
        nothing
    catch e
        e
    end
    e = err()
    @test e isa DimensionMismatch
    @test occursin("N must be $(p)×10", sprint(showerror, e))
    @test simulate(fb, 10; N = fill(3, p, 10)) |> size == (p, 10)
end