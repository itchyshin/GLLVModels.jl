# Lane 1125y5: phylo_binomial_xlv, phylo_nb_xlv, phylo_gamma_xlv small surfaces.
using Test, GLLVModels, Random, LinearAlgebra, Distributions
const G = GLLVModels

function _quad_phylo()
    phy = augmented_phy("((A:0.2,B:0.2):0.2,(C:0.2,D:0.2):0.2);")
    p, n = phy.n_leaves, 6
    X_lv = reshape(collect(range(-0.5, 0.5; length = n)), n, 1)
    return phy, p, n, X_lv
end

@testset "1125y5 phylo xLV marginal and fits" begin
    phy, p, n, X_lv = _quad_phylo()
    Random.seed!(1125)
    K = 1
    beta = fill(0.1, p)
    Lambda = reshape(range(0.1, 0.25; length = p), p, K)
    alpha = reshape([0.2], 1, K)

    # Binomial xLV
    N = fill(8, p, n)
    Yb = clamp.(rand(0:8, p, n), 0, N)
    llb = G._phylo_binomial_xlv_marginal_loglik(
        Yb, N, beta, Lambda, alpha, 0.3, phy, X_lv; maxiter = 40, tol = 1e-8)
    @test isfinite(llb) || llb == -Inf
    fitb = G._fit_phylo_binomial_xlv(Yb, N, phy; K = K, X_lv = X_lv, iterations = 12,
        newton_maxiter = 30)
    @test haskey(fitb, :loglik)
    @test haskey(fitb, :converged)
    @test_throws ArgumentError G._fit_phylo_binomial_xlv(Yb[1:2, :], N, phy; K = K, X_lv = X_lv)

    # NB xLV
    Yn = rand(0:5, p, n)
    lln = G._phylo_nb_xlv_marginal_loglik(
        Yn, beta, Lambda, alpha, 1.5, 0.35, phy, X_lv; maxiter = 40, tol = 1e-8)
    @test isfinite(lln) || lln == -Inf
    fitn = G._fit_phylo_nb_xlv(Yn, phy; K = K, X_lv = X_lv, iterations = 12, newton_maxiter = 30)
    @test fitn.r > 0 || !fitn.converged
    @test_throws ArgumentError G._fit_phylo_nb_xlv(Yn, phy; K = K, X_lv = X_lv[1:(end - 1), :])

    # Gamma xLV
    Yg = rand(Gamma(2.0, 1.0), p, n)
    llg = G._phylo_gamma_xlv_marginal_loglik(
        Yg, beta, Lambda, alpha, 2.0, 0.4, phy, X_lv; maxiter = 40, tol = 1e-8)
    @test isfinite(llg) || llg == -Inf
    fitg = G._fit_phylo_gamma_xlv(Yg, phy; K = K, X_lv = X_lv, iterations = 12, newton_maxiter = 30)
    @test haskey(fitg, :sigma2_phy)
    @test_throws ArgumentError G._fit_phylo_gamma_xlv(-abs.(Yg), phy; K = K, X_lv = X_lv)

    @test_throws ArgumentError G._phylo_binomial_xlv_marginal_loglik(
        Yb, fill(0, p, n), beta, Lambda, alpha, 0.3, phy, X_lv)
    @test G._phylo_binomial_xlv_marginal_loglik(
        Yb, N, beta, zeros(p, K), alpha, 0.0, phy, X_lv) == -Inf
end