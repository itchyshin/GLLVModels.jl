using GLLVModels, Test, Distributions, Random

@testset "fit_gllvm — unified family dispatch" begin
    Random.seed!(7)
    p, n, K = 5, 120, 2

    # Normal() dispatches to the Gaussian fitter with one intercept per trait
    Yg = 0.7 .* randn(p, K) * randn(K, n) .+ 0.5 .* randn(p, n)
    f1 = fit_gllvm(Yg; family = Normal(), K = K)
    f2 = fit_gaussian_gllvm(Yg; K = K, X = GLLVModels._trait_intercept_design(p, n))
    @test f1 isa GllvmFit
    @test f1.logLik ≈ f2.logLik

    # Binomial() dispatches to the binomial fitter (identical result)
    Λ = 1.0 .* randn(p, 1); β = 0.3 .* randn(p)
    η = β .+ Λ * randn(1, n)
    Yb = Int.(rand(p, n) .< 1 ./ (1 .+ exp.(-η)))
    b1 = fit_gllvm(Yb; family = Binomial(), K = 1)
    b2 = fit_binomial_gllvm(Yb; K = 1)
    @test b1 isa BinomialFit
    @test b1.loglik ≈ b2.loglik
    @test b1.link isa LogitLink            # default link

    # an explicit link flows through
    b3 = fit_gllvm(Yb; family = Binomial(), K = 1, link = ProbitLink())
    @test b3.link isa ProbitLink

    # Poisson() now dispatches to the Poisson fitter
    Yc = [rand(0:5) for _ in 1:p, _ in 1:n]
    @test fit_gllvm(Yc; family = Poisson(), K = 1) isa PoissonFit

    @test fit_gllvm(Yc; family = NegativeBinomial(), K = 1) isa NBGroupedFit

    # Beta() public default is per-species precision (proportions in (0,1))
    Yp = clamp.(rand(p, n), 1e-3, 1 - 1e-3)
    @test fit_gllvm(Yp; family = Beta(), K = 1) isa BetaGroupedFit

    # Gamma() now dispatches to fit_gamma_gllvm
    Ypos = abs.(randn(p, n)) .+ 0.1
    @test fit_gllvm(Ypos; family = Gamma(), K = 1) isa GammaFit

    # a still-unimplemented family → clear error
    @test_throws ArgumentError fit_gllvm(Yb; family = Geometric(), K = 1)
end
