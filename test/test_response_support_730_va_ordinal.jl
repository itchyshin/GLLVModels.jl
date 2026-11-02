# #730 (part): the variational (VA) fitters reject out-of-support observed responses with
# an ArgumentError naming the fitter and the cell (Exponential VA used to return
# loglik = 3.7e12, converged = true for y = -1), and the ordinal marginals reject observed
# codes outside 1:C. Clean data still fits; masked ordinal cells stay legal.
using Test, GLLVModels, Random

@testset "#730 VA and ordinal response support" begin
    rng = MersenneTwister(1)
    p, n = 4, 40
    Yp = rand(rng, 0:6, p, n)
    Yg = exp.(randn(rng, p, n))
    Yb = 0.05 .+ 0.9 .* rand(rng, p, n)
    Yn = rand(rng, 0:1, p, n)
    thrown(f) = try; f(); nothing; catch e; e; end
    ok(e, name) = e isa ArgumentError && occursin(name, e.msg) && occursin("Y[1, 1]", e.msg)
    function with_cell(Y, v)
        Z = Matrix{Any}(Y); Z[1, 1] = v
        return identity.(Z)
    end

    @testset "VA fitters throw on out-of-support cells" begin
        @test ok(thrown(() -> fit_exponential_gllvm_va(with_cell(Yg, -1.0); K = 1)), "fit_exponential_gllvm_va")
        @test ok(thrown(() -> fit_exponential_gllvm_va(with_cell(Yg, 0.0); K = 1)), "fit_exponential_gllvm_va")
        @test ok(thrown(() -> fit_exponential_gllvm_va(with_cell(Yg, NaN); K = 1)), "fit_exponential_gllvm_va")
        @test ok(thrown(() -> fit_gamma_gllvm_va(with_cell(Yg, -1.0); K = 1)), "fit_gamma_gllvm_va")
        @test ok(thrown(() -> fit_gamma_gllvm_va(with_cell(Yg, 0.0); K = 1)), "fit_gamma_gllvm_va")
        @test ok(thrown(() -> fit_beta_gllvm_va(with_cell(Yb, 1.5); K = 1)), "fit_beta_gllvm_va")
        @test ok(thrown(() -> fit_beta_gllvm_va(with_cell(Yb, 0.0); K = 1)), "fit_beta_gllvm_va")
        @test ok(thrown(() -> fit_poisson_gllvm_va(with_cell(Yp, -2); K = 1)), "fit_poisson_gllvm_va")
        @test ok(thrown(() -> fit_nb_gllvm_va(with_cell(Yp, -3); K = 1)), "fit_nb_gllvm_va")
        @test ok(thrown(() -> fit_binomial_gllvm_va(with_cell(Yn, 2); K = 1)), "fit_binomial_gllvm_va")
        @test ok(thrown(() -> fit_binomial_gllvm_va(with_cell(Yn, -1); K = 1)), "fit_binomial_gllvm_va")
        N = fill(5, p, n); Y = rand(rng, 0:5, p, n); Y[1, 1] = 6
        @test ok(thrown(() -> fit_binomial_gllvm_va(Y; K = 1, N = N)), "fit_binomial_gllvm_va")
    end

    @testset "clean data still fit" begin
        for fit in (fit_exponential_gllvm_va(Yg; K = 1), fit_gamma_gllvm_va(Yg; K = 1),
                    fit_beta_gllvm_va(Yb; K = 1), fit_poisson_gllvm_va(Yp; K = 1),
                    fit_nb_gllvm_va(Yp; K = 1), fit_binomial_gllvm_va(Yn; K = 1))
            @test isfinite(fit.loglik)
        end
    end

    @testset "ordinal: out-of-range codes throw ArgumentError" begin
        C = 4; po, no = 5, 60
        Y1 = rand(rng, 1:C, po, no)
        Y0 = Y1 .- 1; c0 = findfirst(==(0), Y0)
        cell = "Y[$(c0[1]), $(c0[2])]"
        okc(e, name) = e isa ArgumentError && occursin(name, e.msg) && occursin(cell, e.msg)
        @test okc(thrown(() -> fit_ordinal_gllvm(Y0; K = 1)), "fit_ordinal_gllvm")
        @test okc(thrown(() -> fit_ordinal_gllvm_pertrait(Y0; K = 1)), "fit_ordinal_gllvm_pertrait")
        Λ = zeros(po, 1); τ = [-1.0, 0.0, 1.0]
        mll = GLLVModels.ordinal_marginal_loglik_laplace
        @test okc(thrown(() -> mll(Y0, Λ, τ)), "ordinal_marginal_loglik_laplace")
        Y9 = copy(Y1); Y9[1, 1] = 9
        @test ok(thrown(() -> mll(Y9, Λ, τ)), "ordinal_marginal_loglik_laplace")
        Y5 = copy(Y1); Y5[1, 1] = 5          # one above C = length(τ) + 1
        @test ok(thrown(() -> mll(Y5, Λ, τ)), "ordinal_marginal_loglik_laplace")
        @test isfinite(mll(Y1, Λ, τ))
        @test thrown(() -> GLLVModels.ordinal_marginal_loglik_laplace_pertrait(
            Y9, Λ, repeat(τ', po), fill(C, po))) isa ArgumentError
        mask = trues(po, no); mask[1, 1] = false
        @test isfinite(mll(Y9, Λ, τ; mask = mask))
        @test isfinite(fit_ordinal_gllvm(Y9; K = 1, mask = mask, iterations = 30).loglik)
        @test isfinite(fit_ordinal_gllvm_pertrait(Y1; K = 1, iterations = 30).loglik)
    end
end
