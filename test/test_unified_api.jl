using GLLVModels, Test, Random, Distributions

# Unified-API routing tests for `fit_gllvm`: the new gllvm-style keyword arguments
# (row_eff, disp_group, pervar, num_lv) must each route to EXACTLY the specialised
# fitter they name — same return type and same maximised `loglik` as calling that
# fitter directly. The underlying fits are deterministic in Y (warm start + L-BFGS,
# no internal RNG), so the dispatch-equivalence anchors below should match to well
# within optimiser noise; we seed identically before each paired call regardless.

# Tiny, fast fixtures.
function _count_data(; p = 4, n = 50, K = 1, seed = 1234, hi = 5)
    Random.seed!(seed)
    return rand(0:hi, p, n)
end

@testset "fit_gllvm unified API — keyword routing" begin
    p, n, K = 4, 50, 1
    iters = 30                       # keep L-BFGS short for speed

    # ------------------------------------------------------------------
    # row_eff = :random  ≡  fit_row_random_gllvm
    # ------------------------------------------------------------------
    @testset "row_eff=:random ≡ fit_row_random_gllvm" begin
        Y = _count_data(; p = p, n = n)
        Random.seed!(99)
        a = fit_gllvm(Y; family = Poisson(), K = K, row_eff = :random, iterations = iters)
        Random.seed!(99)
        b = fit_row_random_gllvm(Y; family = Poisson(), K = K, iterations = iters)
        @test typeof(a) === typeof(b)
        @test a isa GLLVModels.RowRandomFit
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)
    end

    # ------------------------------------------------------------------
    # row_eff = :fixed  ≡  fit_roweffect_gllvm
    # ------------------------------------------------------------------
    @testset "row_eff=:fixed ≡ fit_roweffect_gllvm" begin
        Y = _count_data(; p = p, n = n, seed = 222)
        Random.seed!(7)
        a = fit_gllvm(Y; family = Poisson(), K = K, row_eff = :fixed, iterations = iters)
        Random.seed!(7)
        b = fit_roweffect_gllvm(Y; family = Poisson(), K = K, iterations = iters)
        @test typeof(a) === typeof(b)
        @test a isa GLLVModels.RowEffectFit
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)
    end

    # ------------------------------------------------------------------
    # disp_group = :species  ≡  fit_nb_gllvm_grouped(group = 1:p)
    # ------------------------------------------------------------------
    @testset "disp_group=:species ≡ fit_nb_gllvm_grouped" begin
        Y = _count_data(; p = p, n = n, seed = 333, hi = 8)
        fam = NegativeBinomial(1.0, 0.5)
        Random.seed!(11)
        a = fit_gllvm(Y; family = fam, K = K, disp_group = :species, iterations = iters)
        Random.seed!(11)
        b = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p), iterations = iters)
        @test typeof(a) === typeof(b)
        @test a isa GLLVModels.NBGroupedFit
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)

        # an explicit length-p Int group vector routes the same way
        Random.seed!(11)
        c = fit_gllvm(Y; family = fam, K = K, disp_group = collect(1:p), iterations = iters)
        @test isapprox(c.loglik, b.loglik; atol = 1e-6)
    end

    # ------------------------------------------------------------------
    # pervar = true  ≡  fit_gaussian_pervar_gllvm   (family = Normal())
    # ------------------------------------------------------------------
    @testset "pervar=true ≡ fit_gaussian_pervar_gllvm" begin
        Random.seed!(444)
        Y = 0.7 .* randn(p, K) * randn(K, n) .+ 0.5 .* randn(p, n)
        Random.seed!(5)
        a = fit_gllvm(Y; family = Normal(), K = K, pervar = true)
        Random.seed!(5)
        b = fit_gaussian_pervar_gllvm(Y; K = K)
        @test typeof(a) === typeof(b)
        @test a isa GLLVModels.GaussianPerVarFit
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)
    end

    # ------------------------------------------------------------------
    # REGRESSION: plain call still routes to the base family fitter.
    # NB/Beta public default is per-trait φ (API B); Poisson stays shared base.
    # ------------------------------------------------------------------
    @testset "plain dispatch unchanged (regression)" begin
        Y = _count_data(; p = p, n = n, seed = 555)
        Random.seed!(3)
        a = fit_gllvm(Y; family = Poisson(), K = K, iterations = iters)
        Random.seed!(3)
        b = fit_poisson_gllvm(Y; K = K, iterations = iters)
        @test typeof(a) === typeof(b)
        @test a isa PoissonFit
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)

        Ynb = _count_data(; p = p, n = n, seed = 556, hi = 8)
        Random.seed!(12)
        anb = fit_gllvm(Ynb; family = NegativeBinomial(1.0, 0.5), K = K, iterations = iters,
                        disp_group = :species)
        Random.seed!(12)
        bnb = fit_nb_gllvm_grouped(Ynb; K = K, group = collect(1:p), iterations = iters)
        @test anb isa GLLVModels.NBGroupedFit
        @test isapprox(anb.loglik, bnb.loglik; atol = 1e-6)
        # Bare NegativeBinomial() is the shared-r engine since #615.
        Random.seed!(12)
        snb = fit_gllvm(Ynb; family = NegativeBinomial(1.0, 0.5), K = K, iterations = iters)
        Random.seed!(12)
        tnb = fit_nb_gllvm(Ynb; K = K, iterations = iters)
        @test snb isa GLLVModels.NBFit
        @test isapprox(snb.loglik, tnb.loglik; atol = 1e-6)
    end

    # ------------------------------------------------------------------
    # num_lv is an accepted alias for K (gllvm's num.lv).
    # ------------------------------------------------------------------
    @testset "num_lv alias for K" begin
        Y = _count_data(; p = p, n = n, seed = 666)
        Random.seed!(8)
        a = fit_gllvm(Y; family = Poisson(), num_lv = K, iterations = iters)
        Random.seed!(8)
        b = fit_gllvm(Y; family = Poisson(), K = K, iterations = iters)
        @test isapprox(a.loglik, b.loglik; atol = 1e-6)
        # conflicting K and num_lv → error
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = 1, num_lv = 2)
    end

    # ------------------------------------------------------------------
    # Error paths.
    # ------------------------------------------------------------------
    @testset "error paths" begin
        Yc = _count_data(; p = p, n = n, seed = 777)
        # disp_group on a family without grouped support (Poisson) → error
        @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = K,
                                             disp_group = :species)
        # pervar on a non-Normal family → error
        @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = K, pervar = true)
        # combining two variants is not supported → error
        @test_throws ArgumentError fit_gllvm(Yc; family = NegativeBinomial(1.0, 0.5), K = K,
                                             row_eff = :random, disp_group = :species)
        # an unknown row_eff symbol → error
        @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = K,
                                             row_eff = :bogus)
    end
end
