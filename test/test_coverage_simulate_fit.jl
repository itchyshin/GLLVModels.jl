using GLLVModels, Test, Random, Distributions, Statistics

@testset "simulate(fit) scalar and binomial routes" begin
  @testset "Exponential, Beta, NB2 (reproducible shape / type)" begin
    Random.seed!(88422)
    p, K, n = 4, 1, 60
    β = 0.2 .* randn(p)
    Λt = 0.45 .* randn(p, K)
    Z = randn(K, n)
    η = β .+ Λt * Z

    Ye = [rand(Exponential(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit_e = fit_exponential_gllvm(Ye; K = K, iterations = 80)
    if isfinite(fit_e.loglik)
      Ys = simulate(fit_e, 25; rng = MersenneTwister(11))
      @test size(Ys) == (p, 25) && eltype(Ys) == Float64 && all(Ys .> 0)
      @test simulate(fit_e, 20; rng = MersenneTwister(3)) ==
            simulate(fit_e, 20; rng = MersenneTwister(3))
    end

    Yβ = [rand(Beta(2.0, 5.0)) for t in 1:p, s in 1:n]
    fit_β = fit_beta_gllvm(Yβ; K = K)
    Yb = simulate(fit_β, 30; rng = MersenneTwister(5))
    @test size(Yb) == (p, 30) && all(0 .< Yb .< 1)

    r = 1.8
    Yn = [rand(NegativeBinomial(r, r / (r + exp(η[t, s])))) for t in 1:p, s in 1:n]
    fit_n = fit_nb_gllvm(Yn; K = K)
    Yn2 = simulate(fit_n, 35; rng = MersenneTwister(7))
    @test size(Yn2) == (p, 35) && eltype(Yn2) <: Integer && all(Yn2 .>= 0)
  end

  @testset "Binomial (Bernoulli default + explicit N)" begin
    Random.seed!(88423)
    p, K, n = 5, 1, 50
    η0 = 0.25 .* randn(p) .+ (0.5 .* randn(p, K)) * randn(K, n)
    μ = inv.(1 .+ exp.(-η0))
    Y = Int.(rand(p, n) .< μ)
    fit = fit_binomial_gllvm(Y; K = K)
    Y1 = simulate(fit, 20; rng = MersenneTwister(2))
    @test size(Y1) == (p, 20) && all(x -> x == 0 || x == 1, Y1)
    Nm = fill(4, p, 15)
    Y2 = simulate(fit, 15; N = Nm, rng = MersenneTwister(4))
    @test size(Y2) == (p, 15) && all(0 .≤ Y2 .≤ 4)
    @test_throws ArgumentError simulate(fit, 10; offset = zeros(p, 10))
  end

  @testset "ordinal simulation" begin
    rng = MersenneTwister(88424)
    p, K, n, C = 3, 1, 40, 4
    Λtrue = 0.5 .* randn(rng, p, K)
    τtrue = [-0.8, 0.1, 1.0]
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
      η = Λtrue * randn(rng, K)
      for t in 1:p
        u = rand(rng)
        cat = C
        cum = 0.0
        for c in 1:C
          Fhi = c == C ? 1.0 : inv(1 + exp(-(τtrue[c] - η[t])))
          Flo = c == 1 ? 0.0 : inv(1 + exp(-(τtrue[c - 1] - η[t])))
          cum += Fhi - Flo
          if u <= cum
            cat = c
            break
          end
        end
        Y[t, s] = cat
      end
    end
    fit = fit_ordinal_gllvm(Y; K = K, iterations = 200)
    Ys = simulate(fit, 12; rng = MersenneTwister(9))
    @test size(Ys) == (p, 12) && all(1 .≤ Ys .≤ C)
  end
end