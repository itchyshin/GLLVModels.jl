using GLLVModels
using Test
using Random
using Distributions
using StableRNGs

@testset "cv_gllvm refuses unscorable families and offset (#773, #772)" begin
    rng = StableRNG(773)
    p, n = 4, 24
    μ = exp.(0.5 .+ 0.3 .* randn(rng, p, 1)) .* ones(1, n)
    Y = Float64.(rand.(rng, Poisson.(μ)))

    @testset "family without a CV density throws before any fit" begin
        err = try
            cv_gllvm(Y; k_folds = 3, family = NB1(), K = 1, rng = StableRNG(1))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("NB1", sprint(showerror, err))
        @test occursin("NegativeBinomial", sprint(showerror, err))
    end

    @testset "offset throws" begin
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 3, family = Poisson(), K = 1,
                                            offset = zeros(p, n), rng = StableRNG(1))
        @test_throws ArgumentError cv_gllvm(randn(rng, p, n); k_folds = 3, family = Normal(),
                                            K = 1, offset = zeros(p, n), rng = StableRNG(1))
    end

    @testset "supported families still run" begin
        Yb = Float64.(Y .> median(Y))
        Yp = rand(rng, p, n) .* 0.8 .+ 0.1
        Yg = Float64.(rand.(rng, Gamma.(3.0, μ ./ 3.0)))
        Ye = Float64.(rand.(rng, Exponential.(μ)))
        Ynb = Float64.(rand.(rng, NegativeBinomial.(5.0, 5.0 ./ (5.0 .+ μ))))
        cases = (Normal() => randn(rng, p, n), Poisson() => Y, Binomial() => Yb,
                 NegativeBinomial() => Ynb, Beta() => Yp, Gamma() => Yg, Exponential() => Ye)
        for (fam, Yf) in cases
            res = cv_gllvm(Yf; k_folds = 3, family = fam, K = 1, rng = StableRNG(2))
            @test res isa CVResult
            @test isfinite(res.mse)
        end
    end
end
