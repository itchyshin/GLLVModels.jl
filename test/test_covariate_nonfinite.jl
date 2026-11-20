# A NaN, Inf or `missing` in a covariate array used to make the objective NaN or -Inf, so the
# fitter returned `loglik = -Inf, converged = false` with no message. Each covariate-taking
# fitter below now throws an ArgumentError that names the fitter, the array and the first
# offending entry. A covariate value hidden by `mask = false` stays legal (the response cell
# is unobserved), and clean covariates fit as before.
using Test, GLLVModels, Random, Distributions, StatsModels

@testset "non-finite covariates are refused" begin
    rng = MersenneTwister(21)
    p, n = 5, 24
    Yc = [rand(rng, Poisson(exp(0.7 + 0.5 * sin(t + j)))) for t in 1:p, j in 1:n]
    Yz = [rand(rng) < 0.35 ? 0 : max(1, Yc[t, j]) for t in 1:p, j in 1:n]
    Yb = [rand(rng, Binomial(3, 0.4)) for t in 1:p, j in 1:n]
    Yg = 0.3 .+ rand(rng, p, n)
    Yu = clamp.(0.1 .+ 0.8 .* rand(rng, p, n), 0.05, 0.95)
    Yo = [clamp(1 + (t + j) % 3 + (rand(rng) < 0.3), 1, 3) for t in 1:p, j in 1:n]
    Xp = randn(rng, p, n, 1)           # p x n x q
    Xs = randn(rng, n, 2)              # n x q (site level)
    TR = randn(rng, p, 2)
    thrown(f) = try; f(); nothing; catch e; e; end
    function refused(f, caller, name)
        e = thrown(f)
        return e isa ArgumentError && occursin(caller, e.msg) && occursin(name, e.msg) &&
               occursin("finite", e.msg)
    end
    with(X, I, v) = (Z = Array{Any}(X); Z[I...] = v; identity.(Z))
    it = (iterations = 20,)

    @testset "site-level X: rrr, constrained" begin
        for v in (NaN, Inf)
            @test refused(() -> fit_rrr_gllvm(Yc; K = 1, X = with(Xs, (2, 1), v), it...), "fit_rrr_gllvm", "X[2, 1]")
            @test refused(() -> fit_constrained_gllvm(Yc; family = Poisson(), K = 1, X = with(Xs, (2, 1), v), it...),
                          "fit_constrained_gllvm", "X[2, 1]")
        end
    end

    @testset "p x n x q X: two-part, covariate and grouped routes" begin
        for v in (NaN, Inf)
            Xb = with(Xp, (1, 2, 1), v)
            @test refused(() -> fit_zip_gllvm_cov(Yz; K = 1, X = Xb, it...), "fit_zip_gllvm_cov", "X[1, 2, 1]")
            @test refused(() -> fit_zinb_gllvm_cov(Yz; K = 1, X = Xb, it...), "fit_zinb_gllvm_cov", "X[1, 2, 1]")
            @test refused(() -> fit_zib_gllvm_cov(Yb; K = 1, N = 3, X = Xb, it...), "fit_zib_gllvm_cov", "X[1, 2, 1]")
            @test refused(() -> fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xb, it...), "fit_gllvm_cov", "X[1, 2, 1]")
            @test refused(() -> fit_gllvm_speciescov(Yc; family = Poisson(), K = 1, X = Xb, it...), "fit_gllvm_speciescov", "X[1, 2, 1]")
            @test refused(() -> fit_nb_gllvm_grouped_cov(Yc; K = 1, X = Xb, it...), "fit_nb_gllvm_grouped_cov", "X[1, 2, 1]")
            @test refused(() -> fit_beta_gllvm_grouped_cov(Yu; K = 1, X = Xb, it...), "fit_beta_gllvm_grouped_cov", "X[1, 2, 1]")
            @test refused(() -> fit_gamma_gllvm_grouped_cov(Yg; K = 1, X = Xb, it...), "fit_gamma_gllvm_grouped_cov", "X[1, 2, 1]")
            @test refused(() -> fit_nb1_gllvm_grouped_cov(Yc; K = 1, X = Xb, it...), "fit_nb1_gllvm_grouped_cov", "X[1, 2, 1]")
            @test refused(() -> fit_beta_binomial_gllvm_grouped_cov(Yb; K = 1, N = fill(3, p, n), X = Xb, it...),
                          "fit_beta_binomial_gllvm_grouped_cov", "X[1, 2, 1]")
            @test refused(() -> fit_ordinal_gllvm_pertrait_cov(Yo; K = 1, X = Xb, it...), "fit_ordinal_gllvm_pertrait_cov", "X[1, 2, 1]")
        end
    end

    @testset "fourth corner: Xenv and TR, including missing" begin
        for v in (NaN, Inf, missing)
            @test refused(() -> fit_fourthcorner_gllvm(Yc; family = Poisson(), K = 1, Xenv = with(Xs, (3, 2), v), TR = TR, it...),
                          "fit_fourthcorner_gllvm", "Xenv[3, 2]")
        end
        @test refused(() -> fit_fourthcorner_gllvm(Yc; family = Poisson(), K = 1, Xenv = Xs, TR = with(TR, (2, 1), NaN), it...),
                      "fit_fourthcorner_gllvm", "TR[2, 1]")
    end

    @testset "multinomial X" begin
        ycat = [1 + (j % 3) for j in 1:n]; Xm = randn(rng, n, 2)
        for v in (NaN, Inf)
            @test refused(() -> fit_multinomial_gllvm(ycat; X = with(Xm, (5, 2), v), iterations = 20),
                          "fit_multinomial_gllvm", "X[5, 2]")
        end
        @test isfinite(fit_multinomial_gllvm(ycat; X = Xm, iterations = 60).loglik)
    end

    @testset "formula front end reaches the same refusal" begin
        x = randn(rng, n); x[4] = NaN
        e = thrown(() -> gllvm(@formula(y ~ 1 + x), Yc, (x = x,); family = Poisson(), K = 1, it...))
        @test e isa ArgumentError && occursin("finite", e.msg)
    end

    @testset "a covariate hidden by the mask stays legal; clean X still fits" begin
        mask = trues(p, n); mask[1, 2] = false
        Xn = copy(Xp); Xn[1, 2, 1] = NaN
        clean = fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xp, mask = mask, iterations = 60)
        hidden = fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xn, mask = mask, iterations = 60)
        @test isfinite(hidden.loglik)
        @test hidden.loglik ≈ clean.loglik rtol = 1e-8
        # the same NaN at an observed cell is refused
        @test refused(() -> fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xn, it...), "fit_gllvm_cov", "X[1, 2, 1]")
        @test isfinite(fit_rrr_gllvm(Yc; K = 1, X = Xs, iterations = 60).loglik)
    end

    # A mask of the wrong size next to a NaN covariate: the covariate scan indexes the mask,
    # so it must refuse the shape (DimensionMismatch) before reading it, never read past its end.
    @testset "wrong-size mask plus NaN covariate is a DimensionMismatch" begin
        bad_mask = trues(3, 29)                          # Y is 5 x 24
        Xn = copy(Xp); Xn[5, 2, 1] = NaN                 # row 5 does not exist in a 3-row mask
        Xsn = copy(Xs); Xsn[2, 1] = NaN
        mism(f) = (e = thrown(f); e isa DimensionMismatch && occursin("mask", e.msg))
        @test mism(() -> fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xn, mask = bad_mask, it...))
        @test mism(() -> fit_gllvm_cov(Yc; family = Poisson(), K = 1, X = Xp, mask = bad_mask, it...))
        @test mism(() -> fit_gllvm_speciescov(Yc; family = Poisson(), K = 1, X = Xn, mask = bad_mask, it...))
        @test mism(() -> fit_nb_gllvm_grouped_cov(Yc; K = 1, X = Xn, mask = bad_mask, it...))
        # ordinal: the level check runs first and reads the mask by linear index, so use a mask
        # larger than Y here (a smaller one is refused by the level check itself)
        big_mask = trues(6, 30)
        @test mism(() -> fit_ordinal_gllvm_pertrait_cov(Yo; K = 1, X = Xn, mask = big_mask, it...))
        @test mism(() -> fit_ordinal_gllvm_pertrait_cov(Yo; K = 1, X = Xp, mask = big_mask, it...))
        @test mism(() -> fit_fourthcorner_gllvm(Yc; family = Poisson(), K = 1, Xenv = Xsn, TR = TR, mask = bad_mask, it...))
        @test mism(() -> fit_fourthcorner_gllvm(Yc; family = Poisson(), K = 1, Xenv = Xs,
                                                TR = with(TR, (5, 1), NaN), mask = bad_mask, it...))
        # the helper itself, for each unit
        chk = GLLVModels._check_finite_covariates
        @test_throws DimensionMismatch chk(Xn, "f"; obs = bad_mask)
        @test_throws DimensionMismatch chk(Xsn, "f"; obs = bad_mask, unit = :site)
        @test_throws DimensionMismatch chk(TR, "f"; obs = bad_mask, unit = :trait)
        @test chk(Xp, "f"; obs = trues(p, n)) === nothing
    end
end

# #1250: the X_lv sibling. A NaN or Inf in X_lv is refused by name in the six single-family
# fitters that take it; a clean X_lv still fits.
@testset "non-finite X_lv is refused (#1250)" begin
    rng = MersenneTwister(11)
    p, n = 6, 60
    xl = randn(rng, n)
    z = 0.7 .* xl .+ 0.5 .* randn(rng, n)
    lam = [0.8, 0.6, -0.5, 0.7, 0.3, -0.4]
    mu = exp.(0.5 .+ lam * z')
    Y = [rand(rng, Poisson(mu[t, s])) for t in 1:p, s in 1:n]
    Yb = [rand(rng, Binomial(3, 0.4)) for t in 1:p, s in 1:n]
    Yg = 0.3 .+ rand(rng, p, n)
    Yu = clamp.(0.1 .+ 0.8 .* rand(rng, p, n), 0.05, 0.95)
    Yo = [rand(rng, 1:3) for t in 1:p, s in 1:n]
    cases = [("fit_poisson_gllvm", fit_poisson_gllvm, Y, (;)),
             ("fit_nb_gllvm", fit_nb_gllvm, Y, (;)),
             ("fit_binomial_gllvm", fit_binomial_gllvm, Yb, (; N = fill(3, p, n))),
             ("fit_gamma_gllvm", fit_gamma_gllvm, Yg, (;)),
             ("fit_beta_gllvm", fit_beta_gllvm, Yu, (;)),
             ("fit_ordinal_gllvm", fit_ordinal_gllvm, Yo, (;))]
    for bad in (NaN, Inf), (nm, fit, Ym, kw) in cases
        Xlv = reshape(copy(xl), n, 1)
        Xlv[5, 1] = bad
        err = try; fit(Ym; K = 1, X_lv = Xlv, kw...); nothing; catch e; e; end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin(nm, err.msg) && occursin("X_lv[5, 1]", err.msg)
    end
end
