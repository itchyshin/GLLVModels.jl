# P82 CONST-COLUMN (refs #1143): a Gamma or Beta species whose observed cells are all equal
# and that has a dispersion of its own is an ArgumentError naming the species. The
# likelihood of such a species is unbounded in its dispersion, so the fit used to run the
# dispersion to infinity and report a meaningless log-likelihood.
using Test, GLLVModels, Random, Distributions

@testset "constant Gamma or Beta species with its own dispersion is refused" begin
    thrown(f) = try; f(); nothing; catch e; e; end
    rng = MersenneTwister(11)
    p, n = 4, 30
    z = randn(rng, n)
    eta = [0.3 * j + 0.5 * z[i] for j in 1:p, i in 1:n]
    Yg = [rand(rng, Gamma(3.0, exp(eta[j, i]) / 3.0)) for j in 1:p, i in 1:n]
    Yb = [(m = 1 / (1 + exp(-eta[j, i])); rand(rng, Distributions.Beta(10m, 10(1 - m)))) for j in 1:p, i in 1:n]
    Yg[2, :] .= 2.5
    Yb[2, :] .= 0.5
    named(e, caller) = e isa ArgumentError && startswith(e.msg, "$caller: species 2 (row 2 of Y) is constant at") &&
                       occursin("own dispersion", e.msg)

    @testset "Beta, per-species dispersion" begin
        @test named(thrown(() -> fit_gllvm(Yb; family = GLLVModels.Beta(), K = 1)), "fit_beta_gllvm_grouped")
        @test named(thrown(() -> fit_beta_gllvm_grouped(Yb; K = 1)), "fit_beta_gllvm_grouped")
        X = reshape(randn(rng, p * n), p, n, 1)
        @test named(thrown(() -> fit_beta_gllvm_grouped_cov(Yb; X = X, K = 1)), "fit_beta_gllvm_grouped_cov")
    end
    @testset "Gamma, per-species dispersion" begin
        @test named(thrown(() -> fit_gllvm(Yg; family = Gamma(), K = 1, disp_group = :species)), "fit_gamma_gllvm_grouped")
        @test named(thrown(() -> fit_gamma_gllvm_grouped(Yg; K = 1)), "fit_gamma_gllvm_grouped")
        X = reshape(randn(rng, p * n), p, n, 1)
        @test named(thrown(() -> fit_gamma_gllvm_grouped_cov(Yg; X = X, K = 1)), "fit_gamma_gllvm_grouped_cov")
    end
    @testset "a masked-out varying cell does not hide a constant species; one observed cell is not constant" begin
        Ym = copy(Yg); Ym[2, 1] = 9.0                    # species 2 varies only in a cell the mask drops
        mask = trues(p, n); mask[2, 1] = false
        @test named(thrown(() -> fit_gamma_gllvm_grouped(Ym; K = 1, mask = mask)), "fit_gamma_gllvm_grouped")
        Ys = copy(Yg); Ys[2, 1] = 9.0                    # unmasked: species 2 is not constant, no refusal
        @test !(thrown(() -> fit_gamma_gllvm_grouped(Ys; K = 1, iterations = 3)) isa ArgumentError)
    end
    @testset "not refused: shared dispersion, shared group" begin
        # default Gamma has one shared shape; a constant species is bounded by the others
        @test isfinite(fit_gllvm(Yg; family = Gamma(), K = 1).loglik)
        # a constant species in a dispersion group with another species
        @test !(thrown(() -> fit_gamma_gllvm_grouped(Yg; K = 1, group = [1, 1, 2, 3], iterations = 3)) isa ArgumentError)
    end
end
