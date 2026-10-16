# #730: the Poisson / Binomial / Beta / Gamma Laplace fitters must reject out-of-support or
# non-finite OBSERVED responses with an ArgumentError that names the fitter and the cell,
# instead of returning a fit with loglik = -Inf. Masked and `missing` cells stay legal.
using Test, GLLVModels, Random

@testset "#730 response-domain validation" begin
    rng = MersenneTwister(1)
    p, n = 4, 40
    Yp = rand(rng, 0:6, p, n)
    Yg = exp.(randn(rng, p, n))
    Yb = 0.05 .+ 0.9 .* rand(rng, p, n)
    Yn = rand(rng, 0:1, p, n)

    function with_cell(Y, v)
        Z = Matrix{Any}(Y); Z[1, 1] = v
        return all(x -> x isa Real, Z) ? identity.(Z) : Z
    end
    thrown(f) = try; f(); nothing; catch e; e; end
    function msg_ok(e, name)
        e isa ArgumentError && occursin(name, e.msg) && occursin("Y[1, 1]", e.msg)
    end

    @testset "out-of-support cells throw ArgumentError" begin
        @test msg_ok(thrown(() -> fit_poisson_gllvm(with_cell(Yp, -2); K = 1)), "fit_poisson_gllvm")
        @test msg_ok(thrown(() -> fit_poisson_gllvm(with_cell(float.(Yp), 2.5); K = 1)), "fit_poisson_gllvm")
        @test msg_ok(thrown(() -> fit_poisson_gllvm(with_cell(float.(Yp), NaN); K = 1)), "fit_poisson_gllvm")
        @test msg_ok(thrown(() -> fit_binomial_gllvm(with_cell(Yn, 2); K = 1)), "fit_binomial_gllvm")
        @test msg_ok(thrown(() -> fit_binomial_gllvm(with_cell(Yn, -1); K = 1)), "fit_binomial_gllvm")
        @test msg_ok(thrown(() -> fit_gamma_gllvm(with_cell(Yg, -1.0); K = 1)), "fit_gamma_gllvm")
        @test msg_ok(thrown(() -> fit_gamma_gllvm(with_cell(Yg, 0.0); K = 1)), "fit_gamma_gllvm")
        @test msg_ok(thrown(() -> fit_gamma_gllvm(with_cell(Yg, NaN); K = 1)), "fit_gamma_gllvm")
        @test msg_ok(thrown(() -> fit_beta_gllvm(with_cell(Yb, 1.5); K = 1)), "fit_beta_gllvm")
        @test msg_ok(thrown(() -> fit_beta_gllvm(with_cell(Yb, 0.0); K = 1)), "fit_beta_gllvm")
        @test msg_ok(thrown(() -> fit_beta_gllvm(with_cell(Yb, NaN); K = 1)), "fit_beta_gllvm")
    end

    @testset "binomial y > N with explicit trials" begin
        N = fill(5, p, n)
        Y = rand(rng, 0:5, p, n); Y[1, 1] = 6
        @test msg_ok(thrown(() -> fit_binomial_gllvm(Y; K = 1, N = N)), "fit_binomial_gllvm")
        Y[1, 1] = 5          # y == N is legal
        @test thrown(() -> fit_binomial_gllvm(Y; K = 1, N = N)) === nothing
    end

    @testset "the aghq routes are guarded too" begin
        @test thrown(() -> fit_poisson_gllvm(with_cell(Yp, -2); K = 1, aghq = 5)) isa ArgumentError
        @test thrown(() -> fit_binomial_gllvm(with_cell(Yn, 2); K = 1, aghq = 5)) isa ArgumentError
    end

    @testset "masked and missing cells are not checked" begin
        mask = trues(p, n); mask[1, 1] = false
        for (f, Y, bad) in ((fit_poisson_gllvm, float.(Yp), NaN), (fit_poisson_gllvm, float.(Yp), -2.0),
                            (fit_binomial_gllvm, float.(Yn), NaN), (fit_gamma_gllvm, Yg, NaN),
                            (fit_gamma_gllvm, Yg, -1.0), (fit_beta_gllvm, Yb, NaN), (fit_beta_gllvm, Yb, 1.5))
            Z = copy(Y); Z[1, 1] = bad
            fit = f(Z; K = 1, mask = mask)
            @test isfinite(fit.loglik) && fit.converged
        end
        Zm = Matrix{Union{Missing, Float64}}(Yg); Zm[1, 1] = missing
        @test isfinite(fit_gamma_gllvm(Zm; K = 1).loglik)
        Zc = Matrix{Union{Missing, Int}}(Yp); Zc[2, 3] = missing
        @test isfinite(fit_poisson_gllvm(Zc; K = 1).loglik)
    end

    @testset "clean data fits as before" begin
        @test fit_poisson_gllvm(Yp; K = 1).converged
        @test fit_binomial_gllvm(Yn; K = 1).converged
        @test fit_gamma_gllvm(Yg; K = 1).converged
        @test fit_beta_gllvm(Yb; K = 1).converged
    end
end
