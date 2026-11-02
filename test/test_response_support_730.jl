# #730: the Poisson / Binomial / Beta / Gamma Laplace fitters (and the grouped-dispersion
# Beta / Gamma twins, which `fit_gllvm(Y; family = Beta())` always uses) must reject
# out-of-support or non-finite OBSERVED responses with an ArgumentError that names the
# fitter and the cell, instead of returning a fit with loglik = -Inf. Masked and `missing`
# cells stay legal.
using Test, GLLVModels, Random
using Distributions: Beta, Gamma

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
            if f === fit_gamma_gllvm || f === fit_beta_gllvm
                # Neither fit type has a `theta` field and `theta_packed` is empty without
                # X_lv, so check the estimates themselves: β, Λ and the dispersion.
                θ = vcat(fit.β, vec(fit.Λ), f === fit_gamma_gllvm ? fit.α : fit.φ)
                @test !isempty(θ) && all(isfinite, θ)
                # The masked value never reaches the warm start or the objective: the fit
                # is identical to one with the ordinary value left in the masked cell.
                ref = f(Y; K = 1, mask = mask)
                @test fit.β == ref.β && fit.Λ == ref.Λ && fit.loglik == ref.loglik
            end
        end
        Zm = Matrix{Union{Missing, Float64}}(Yg); Zm[1, 1] = missing
        @test isfinite(fit_gamma_gllvm(Zm; K = 1).loglik)
        Zc = Matrix{Union{Missing, Int}}(Yp); Zc[2, 3] = missing
        @test isfinite(fit_poisson_gllvm(Zc; K = 1).loglik)
        # A `Matrix{Any}` holding `missing` (what `readdlm` plus `replace("NA" => missing)`
        # gives) fits exactly as the same data typed `Union{Missing, Float64}`.
        for (f, Y) in ((fit_gamma_gllvm, Yg), (fit_beta_gllvm, Yb))
            Za = Matrix{Any}(Y); Za[1, 1] = missing
            Zu = Matrix{Union{Missing, Float64}}(Y); Zu[1, 1] = missing
            fa = f(Za; K = 1)
            @test isfinite(fa.loglik) && fa.loglik == f(Zu; K = 1).loglik
        end
        # A zero weight drops the cell like `mask`, so its value is not checked.
        W = ones(p, n); W[1, 1] = 0
        Zw = copy(Yp); Zw[1, 1] = -1
        fw = fit_poisson_gllvm(Zw; K = 1, weights = W)
        @test isfinite(fw.loglik) && fw.converged
    end

    @testset "the grouped Beta and Gamma twins" begin
        one_ = ones(Int, p)
        X = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)
        @test msg_ok(thrown(() -> fit_beta_gllvm_grouped(with_cell(Yb, 1.5); K = 1, group = one_)),
                     "fit_beta_gllvm_grouped")
        @test msg_ok(thrown(() -> fit_gamma_gllvm_grouped(with_cell(Yg, 0.0); K = 1, group = one_)),
                     "fit_gamma_gllvm_grouped")
        @test msg_ok(thrown(() -> fit_beta_gllvm_grouped_cov(with_cell(Yb, NaN); X = X, K = 1, group = one_)),
                     "fit_beta_gllvm_grouped_cov")
        @test msg_ok(thrown(() -> fit_gamma_gllvm_grouped_cov(with_cell(Yg, -1.0); X = X, K = 1, group = one_)),
                     "fit_gamma_gllvm_grouped_cov")
        # The public routes: Beta always goes to the grouped fitter, Gamma with disp_group.
        @test msg_ok(thrown(() -> fit_gllvm(with_cell(Yb, 0.0); family = Beta(), K = 1)),
                     "fit_beta_gllvm_grouped")
        @test msg_ok(thrown(() -> fit_gllvm(with_cell(Yg, Inf); family = Gamma(), K = 1, disp_group = :species)),
                     "fit_gamma_gllvm_grouped")
        # A masked cell stays legal and has no effect on the fit.
        mask = trues(p, n); mask[1, 1] = false
        for (f, Y, bad) in ((fit_beta_gllvm_grouped, Yb, NaN), (fit_beta_gllvm_grouped, Yb, 1.5),
                            (fit_gamma_gllvm_grouped, Yg, NaN), (fit_gamma_gllvm_grouped, Yg, -1.0))
            Z = copy(Y); Z[1, 1] = bad
            fit = f(Z; K = 1, group = one_, mask = mask)
            ref = f(Y; K = 1, group = one_, mask = mask)
            @test isfinite(fit.loglik) && fit.β == ref.β && fit.Λ == ref.Λ && fit.loglik == ref.loglik
        end
    end

    @testset "clean data fits as before" begin
        @test fit_poisson_gllvm(Yp; K = 1).converged
        @test fit_binomial_gllvm(Yn; K = 1).converged
        @test fit_gamma_gllvm(Yg; K = 1).converged
        @test fit_beta_gllvm(Yb; K = 1).converged
    end
end
