# #730 (remaining fitters): SPDE latent, mixed-family, coevolution GLM, missing-predictor
# FIML and multinomial must reject out-of-support or non-finite OBSERVED responses with an
# ArgumentError that names the fitter (and the first bad cell), instead of returning a fit with
# loglik = -Inf or failing late with an InexactError. The AGHQ wrappers are pinned: they already
# refuse such data (problem builders, or the Laplace fitter they fall back to).
using Test, GLLVModels, Random, Distributions, LinearAlgebra, SparseArrays

@testset "#730 response-domain validation, remaining fitters" begin
    rng = MersenneTwister(21)
    thrown(f) = try; f(); nothing; catch e; e; end
    msg_ok(e, name) = e isa ArgumentError && occursin(name, e.msg)
    with_cell(Y, v, i = 1, j = 1) = (Z = Matrix{Any}(float.(Y)); Z[i, j] = v; identity.(Z))

    @testset "fit_spde_latent_gllvm" begin
        m, L = 4, 3.0
        xs = range(0.0, L; length = m)
        nodes = [xs[i] for j in 1:m for i in 1:m]; nodes = hcat(nodes, [xs[j] for j in 1:m for i in 1:m])
        id(i, j) = (j - 1) * m + i
        tris = Matrix{Int}(undef, 2 * (m - 1)^2, 3); t = 0
        for j in 1:(m - 1), i in 1:(m - 1)
            t += 1; tris[t, :] = [id(i, j), id(i + 1, j), id(i + 1, j + 1)]
            t += 1; tris[t, :] = [id(i, j), id(i + 1, j + 1), id(i, j + 1)]
        end
        locs = nodes[1:3:m * m, :]; M = size(locs, 1); p = 3
        Yc = rand(rng, 0:4, p, M); Yg = randn(rng, p, M); Yb = rand(rng, 0:1, p, M)
        call(Y, fam; kw...) = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = fam, K = 1,
                                                    iterations = 5, newton_maxiter = 20, kw...)
        for bad in (-1.0, 2.5, NaN)
            @test msg_ok(thrown(() -> call(with_cell(Yc, bad), Poisson())), "fit_spde_latent_gllvm")
            @test msg_ok(thrown(() -> call(with_cell(Yc, bad), NegativeBinomial())), "fit_spde_latent_gllvm")
        end
        for bad in (NaN, Inf)
            @test msg_ok(thrown(() -> call(with_cell(Yg, bad), Normal())), "fit_spde_latent_gllvm")
        end
        @test msg_ok(thrown(() -> call(with_cell(Yb, 6), Binomial(); N = fill(5, p, M))),
                     "fit_spde_latent_gllvm")
        @test isfinite(call(float.(Yc), Poisson()).loglik)       # zero counts and in-support data fit
        @test isfinite(call(Yg, Normal()).loglik)                # negative reals are legal for Normal
    end

    @testset "fit_mixed_gllvm (row-wise families, true trait row in the message)" begin
        p, n = 4, 30
        fams = [Normal(), Poisson(), Gamma(), Binomial()]
        Y = vcat(randn(rng, 1, n), float.(rand(rng, 0:5, 1, n)), exp.(randn(rng, 1, n)),
                 float.(rand(rng, 0:1, 1, n)))
        Nm = ones(Int, p, n)
        run(Y) = fit_mixed_gllvm(Y; families = fams, K = 1, N = Nm, iterations = 5)
        for (row, bad) in ((1, NaN), (2, -1.0), (2, 2.5), (3, 0.0), (3, -1.0), (4, 2.0))
            e = thrown(() -> run(with_cell(Y, bad, row, 1)))
            @test msg_ok(e, "fit_mixed_gllvm") && occursin("Y[$row, 1]", e.msg)
        end
        # Normal accepts a negative real, Gamma rejects it, a masked-out `missing` stays legal.
        Ym = Matrix{Union{Missing, Float64}}(Y); Ym[2, 1] = missing
        @test isfinite(run(Ym).loglik)
        Yn = copy(Y); Yn[1, 1] = -3.0
        @test isfinite(run(Yn).loglik)
    end

    @testset "fit_coevolution_glm" begin
        T, n = 3, 8
        Kst = Matrix(1.0I, n, n)
        Yc = float.(rand(rng, 0:4, T, n)); Yb = float.(rand(rng, 0:1, T, n))
        call(Y, fam; kw...) = fit_coevolution_glm(Y, Kst; family = fam, d = 1, iterations = 5, kw...)
        for bad in (-1.0, 2.5)
            @test msg_ok(thrown(() -> call(with_cell(Yc, bad), Poisson())), "fit_coevolution_glm")
        end
        @test msg_ok(thrown(() -> call(with_cell(Yb, 2.0), Binomial(); N = ones(T, n))), "fit_coevolution_glm")
        @test msg_ok(thrown(() -> call(with_cell(randn(rng, T, n), Inf), Normal())), "fit_coevolution_glm")
        Ym = Matrix{Union{Missing, Float64}}(Yc); Ym[1, 1] = missing       # block-NA route stays legal
        @test isfinite(call(Ym, Poisson()).loglik)
    end

    @testset "fit_gllvm_mi and fit_gllvm_mi_multi" begin
        p, n = 3, 25
        Yc = float.(rand(rng, 0:4, p, n)); x = randn(rng, n); X = randn(rng, n, 2)
        Yu = 0.05 .+ 0.9 .* rand(rng, p, n)
        for bad in (-1.0, 2.5, NaN)
            @test msg_ok(thrown(() -> fit_gllvm_mi(Poisson(), with_cell(Yc, bad), x; K = 1)), "fit_gllvm_mi")
            @test msg_ok(thrown(() -> fit_gllvm_mi_multi(Poisson(), with_cell(Yc, bad), X; K = 1)),
                         "fit_gllvm_mi_multi")
        end
        @test msg_ok(thrown(() -> fit_gllvm_mi(Gamma(), with_cell(exp.(randn(rng, p, n)), 0.0), x; K = 1)), "fit_gllvm_mi")
        @test msg_ok(thrown(() -> fit_gllvm_mi(Beta(), with_cell(Yu, 1.0), x; K = 1)), "fit_gllvm_mi")
        @test msg_ok(thrown(() -> fit_gllvm_mi(Binomial(), with_cell(Yc, 3.0), x; K = 1, N = fill(2, p, n))), "fit_gllvm_mi")
        @test thrown(() -> fit_gllvm_mi(Poisson(), Yc, x; K = 1, iterations = 20)) === nothing
    end

    @testset "fit_multinomial_gllvm" begin
        y = rand(rng, 1:3, 1, 40)
        for bad in (2.5, NaN, Inf)
            @test msg_ok(thrown(() -> fit_multinomial_gllvm(with_cell(y, bad))), "multinomial")
        end
        @test msg_ok(thrown(() -> fit_multinomial_gllvm(with_cell(y, 0.0))), "multinomial")   # y < 1 (already refused)
        @test msg_ok(thrown(() -> fit_multinomial_gllvm(with_cell(y, 4.0); n_categories = 3)), "multinomial")
        @test isfinite(fit_multinomial_gllvm(float.(y); iterations = 20).loglik)   # integer-valued floats stay legal
    end

    @testset "AGHQ wrappers already refuse out-of-support data (pinned)" begin
        p, n = 3, 20
        Yc = rand(rng, 0:4, p, n); Yp = float.(Yc)
        for bad in (-1.0, 2.5, NaN)
            for aghq in (3, 1)    # eligible AGHQ path, and the Laplace-rule fallback
                @test thrown(() -> fit_poisson_gllvm(with_cell(Yc, bad); K = 1, aghq = aghq, iterations = 3)) isa ArgumentError
                @test thrown(() -> fit_nb_gllvm_grouped_aghq(with_cell(Yc, bad); K = 1, aghq = aghq, iterations = 3)) isa ArgumentError
            end
        end
        @test thrown(() -> fit_binomial_gllvm(with_cell(rand(rng, 0:1, p, n), 2); K = 1, aghq = 3, iterations = 3)) isa ArgumentError
        @test thrown(() -> GLLVModels.fit_tweedie_gllvm_grouped_aghq(with_cell(exp.(randn(rng, p, n)), -1.0);
                                                                      K = 1, power = 1.5, aghq = 3, iterations = 3)) isa ArgumentError
        @test thrown(() -> GLLVModels.fit_delta_gamma_gllvm_aghq(with_cell(exp.(randn(rng, p, n)), -1.0);
                                                                  K = 1, aghq = 3, iterations = 3)) isa ArgumentError
    end
end
