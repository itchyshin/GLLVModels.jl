using GLLVModels, Test, Random, Distributions

# refs #1198: wrong-sized trials matrix N must DimensionMismatch before @inbounds indexing.
@testset "simulate refuses wrong-sized N (BinomialFit and GllvmCovFit)" begin
    rng = MersenneTwister(2)
    p, n = 5, 100
    Y = [rand(rng, Binomial(5, 0.3 + 0.1 * randn()^2 / 4)) for t in 1:p, s in 1:n]
    fb = fit_binomial_gllvm(Y; K = 1, N = fill(5, p, n))
    err(f) = try
        f()
        nothing
    catch e
        e
    end
  @testset "BinomialFit" begin
        e1 = err(() -> simulate(fb, 10; N = fill(5, 2, 10)))
        @test e1 isa DimensionMismatch && occursin("N must be $(p)×10", sprint(showerror, e1))
        e2 = err(() -> simulate(fb, 10; N = fill(5, p, 3)))
        @test e2 isa DimensionMismatch && occursin("N must be $(p)×10", sprint(showerror, e2))
        Ys = simulate(fb, 10; N = fill(5, p, 10))
        @test size(Ys) == (p, 10)
    end
    X = reshape(randn(rng, n), 1, n, 1) .* ones(p, 1, 1)
    fc = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1, N = fill(5, p, n))
    @testset "GllvmCovFit" begin
        e1 = err(() -> simulate(fc, zeros(p, 10, 1); N = fill(5, 2, 10)))
        @test e1 isa DimensionMismatch
        e2 = err(() -> simulate(fc, zeros(p, 10, 1); N = fill(5, p, 3)))
        @test e2 isa DimensionMismatch
        Ys = simulate(fc, X; N = fill(5, p, n))
        @test size(Ys) == (p, n)
    end
end