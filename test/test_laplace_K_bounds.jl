using GLLVModels, Test, Random, Distributions, LinearAlgebra

# #1025: K > p on the Laplace-family routes reached the unchecked `@inbounds`
# writes in `pack_lambda` / `unpack_lambda` and segfaulted. K = 0 (a GLM fit)
# stays valid on these routes, so the bound is 0:p.
@testset "Laplace-family K bounds (#1025)" begin
    Random.seed!(1)
    p, n = 5, 60
    K = p + 3
    msg = "K must lie in 0:p (got K = $K, p = $p)"

    @test_throws ArgumentError(msg) GLLVModels.pack_lambda(zeros(p, K))
    @test_throws ArgumentError(msg) GLLVModels.unpack_lambda(zeros(GLLVModels.rr_theta_len(p, K)), p, K)
    @test_throws ArgumentError(msg) GLLVModels.init_theta_rr(p, K)

    Yc = Float64.(rand(0:6, p, n))
    Yb = Float64.(rand(0:1, p, n))
    @test_throws ArgumentError(msg) fit_gllvm(Yc; family = Poisson(), K = K)
    @test_throws ArgumentError(msg) fit_gllvm(Yc; family = NegativeBinomial(), K = K)
    @test_throws ArgumentError(msg) fit_gllvm(Yb; family = Binomial(), K = K)
    @test_throws ArgumentError(msg) fit_poisson_gllvm(Yc; K = K)
    @test_throws ArgumentError(msg) fit_nb_gllvm(Yc; K = K)
    @test_throws ArgumentError(msg) fit_binomial_gllvm(Yb; K = K)

    # K = p and K = 0 remain accepted by the packing primitives.
    Λ = randn(p, p)
    @test GLLVModels.unpack_lambda(GLLVModels.pack_lambda(Λ), p, p) ≈ LowerTriangular(Λ)
    @test length(GLLVModels.pack_lambda(zeros(p, 0))) == 0
end
