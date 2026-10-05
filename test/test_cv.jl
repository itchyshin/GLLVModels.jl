using GLLVModels
using Test
using Random
using LinearAlgebra
using Statistics
using Distributions
using StableRNGs

@testset "Cross-Validation Engine (cv_gllvm & CVResult)" begin

    # -------------------------------------------------------------------------
    # 1. Split Mechanics & Partition Invariants
    # -------------------------------------------------------------------------
    @testset "Split partitions and coverage" begin
        Random.seed!(101)
        p, n = 5, 30
        Y = randn(p, n)

        # A. Random cell-level split
        cv_rand = cv_gllvm(Y; k_folds = 5, split = :random, family = Normal(), K = 1,
                           rng = Random.MersenneTwister(42))
        @test cv_rand isa CVResult
        @test cv_rand.k_folds == 5
        @test cv_rand.split === :random
        @test size(cv_rand.predictions) == (p, n)
        @test size(cv_rand.residuals) == (p, n)
        @test all(isfinite, cv_rand.predictions)
        @test all(isfinite, cv_rand.residuals)
        @test length(cv_rand.fold_logliks) == 5
        @test length(cv_rand.fold_mses) == 5
        @test length(cv_rand.fits) == 5
        @test cv_rand.mse ≈ mean(cv_rand.fold_mses) atol = 1e-4

        # B. Site-level block split
        cv_site = cv_gllvm(Y; k_folds = 5, split = :site, family = Normal(), K = 1,
                           rng = Random.MersenneTwister(42))
        @test cv_site.split === :site
        @test all(isfinite, cv_site.predictions)
        @test all(isfinite, cv_site.residuals)
        @test length(cv_site.fold_mses) == 5

        # C. Species-level block split
        cv_spec = cv_gllvm(Y; k_folds = 5, split = :species, family = Normal(), K = 1,
                           rng = Random.MersenneTwister(42))
        @test cv_spec.split === :species
        @test all(isfinite, cv_spec.predictions)
        @test all(isfinite, cv_spec.residuals)

        # Reproducibility check with fixed RNG
        cv1 = cv_gllvm(Y; k_folds = 3, split = :random, family = Normal(), K = 1,
                       rng = Random.MersenneTwister(99))
        cv2 = cv_gllvm(Y; k_folds = 3, split = :random, family = Normal(), K = 1,
                       rng = Random.MersenneTwister(99))
        @test cv1.mse ≈ cv2.mse atol = 1e-12
        @test cv1.loglik ≈ cv2.loglik atol = 1e-12
        @test cv1.predictions ≈ cv2.predictions atol = 1e-12
    end

    # -------------------------------------------------------------------------
    # 2. Multi-Family Cross-Validation & Metric Verification
    # -------------------------------------------------------------------------
    @testset "Poisson Count Family CV" begin
        Random.seed!(202)
        p, n, K = 4, 35, 1
        Λ_true = [0.8, 0.6, -0.5, 0.4]
        β_true = [0.5, 0.2, -0.3, 0.1]
        z = randn(n)
        η = β_true .+ reshape(Λ_true, p, 1) * reshape(z, 1, n)
        Y = rand.(Poisson.(exp.(η)))

        cv_pois = cv_gllvm(Y; k_folds = 4, split = :random, family = Poisson(), K = 1,
                           rng = Random.MersenneTwister(123))
        @test isfinite(cv_pois.loglik)
        @test isfinite(cv_pois.mse)
        @test cv_pois.mse > 0.0
        @test all(isfinite, cv_pois.predictions)
        @test all(isfinite, cv_pois.residuals)

        # Dunn–Smyth residuals on Poisson counts should have mean ≈ 0, std ≈ 1
        @test abs(cv_pois.residual_mean) < 0.35
        @test 0.6 < cv_pois.residual_std < 1.4

        # Site-block CV on Poisson
        cv_pois_site = cv_gllvm(Y; k_folds = 3, split = :site, family = Poisson(), K = 1,
                                rng = Random.MersenneTwister(123))
        @test isfinite(cv_pois_site.loglik)
        @test isfinite(cv_pois_site.mse)
    end

    # Held-out species drop rows from the mode search, so Λ and β must be
    # subset to the same training species. Without that, every non-Normal
    # :species fold throws DimensionMismatch inside _laplace_mode (#768).
    @testset "species-split Poisson (#768)" begin
        rng = MersenneTwister(768)
        p, n = 4, 16
        Y = rand(rng, 0:6, p, n)
        cv_spec = cv_gllvm(Y; k_folds = 2, split = :species, family = Poisson(), K = 1,
                           rng = MersenneTwister(768))
        @test cv_spec isa CVResult
        @test cv_spec.split === :species
        @test cv_spec.loglik isa Real
        @test isfinite(cv_spec.loglik)
        @test isfinite(cv_spec.mse)
    end

    @testset "Binomial Binary Family CV" begin
        Random.seed!(303)
        p, n = 4, 40
        Λ_true = [0.7, -0.6, 0.5, 0.3]
        β_true = [0.2, -0.1, 0.0, 0.3]
        z = randn(n)
        η = β_true .+ reshape(Λ_true, p, 1) * reshape(z, 1, n)
        prob = 1.0 ./ (1.0 .+ exp.(-η))
        Y = Int.(rand(p, n) .< prob)

        cv_bin = cv_gllvm(Y; k_folds = 3, split = :random, family = Binomial(), K = 1,
                          rng = Random.MersenneTwister(456))
        @test isfinite(cv_bin.loglik)
        @test isfinite(cv_bin.mse)
        @test all(0.0 .<= cv_bin.predictions .<= 1.0)
        @test all(isfinite, cv_bin.residuals)
        @test abs(cv_bin.residual_mean) < 0.5
    end

    @testset "Negative Binomial Family CV" begin
        Random.seed!(404)
        p, n = 4, 30
        Λ_true = [0.6, 0.4, -0.3, 0.5]
        β_true = [0.8, 0.5, 0.3, 0.6]
        z = randn(n)
        η = β_true .+ reshape(Λ_true, p, 1) * reshape(z, 1, n)
        μ = exp.(η)
        r = 3.0
        Y = rand.(NegativeBinomial.(r, r ./ (r .+ μ)))

        cv_nb = cv_gllvm(Y; k_folds = 3, split = :random, family = NegativeBinomial(), K = 1,
                         rng = Random.MersenneTwister(789))
        @test isfinite(cv_nb.loglik)
        @test isfinite(cv_nb.mse)
        @test all(isfinite, cv_nb.predictions)
        @test all(isfinite, cv_nb.residuals)
    end

    @testset "Gamma & Beta Families CV" begin
        p, n = 3, 25

        # Gamma — StableRNG keeps the synthetic Y identical across Julia versions
        # and platforms; MersenneTwister(seed) is not portable for pinned draws.
        gamma_rng = StableRNG(505)
        Y_gamma = rand(gamma_rng, p, n) .* 2.0 .+ 0.2
        cv_gam = cv_gllvm(Y_gamma; k_folds = 3, split = :random, family = Gamma(), K = 1,
                          rng = StableRNG(11))
        @test isfinite(cv_gam.loglik)
        @test isfinite(cv_gam.mse)
        @test all(cv_gam.predictions .> 0.0)

        # Beta — same StableRNG discipline; fold splits use StableRNG(22) so Linux
        # OpenBLAS does not land on a different Laplace optimum with boundary μ.
        beta_data_rng = StableRNG(505)
        Y_beta = clamp.(rand(beta_data_rng, p, n) .* 0.8 .+ 0.1, 0.01, 0.99)
        cv_beta = cv_gllvm(Y_beta; k_folds = 3, split = :random, family = Beta(), K = 1,
                           rng = StableRNG(22))
        @test isfinite(cv_beta.loglik)
        @test isfinite(cv_beta.mse)
        @test all((0.0 .< cv_beta.predictions) .& (cv_beta.predictions .< 1.0))
    end

    # -------------------------------------------------------------------------
    # 3. Model Dimension Selection via Cross-Validation
    # -------------------------------------------------------------------------
    @testset "CV latent dimension comparison (K=2 vs K=1)" begin
        Random.seed!(606)
        p, K_true, n = 6, 2, 80
        Λ_true = [0.9  0.0;
                  0.7  0.6;
                  0.8 -0.5;
                 -0.6  0.7;
                  0.5 -0.6;
                 -0.7 -0.4]
        z_true = randn(K_true, n)
        Y = Λ_true * z_true .+ 0.3 .* randn(p, n)

        cv_k1 = cv_gllvm(Y; k_folds = 4, split = :random, family = Normal(), K = 1,
                         rng = Random.MersenneTwister(77))
        cv_k2 = cv_gllvm(Y; k_folds = 4, split = :random, family = Normal(), K = 2,
                         rng = Random.MersenneTwister(77))

        # The true 2-latent model should achieve higher out-of-sample logLik and lower MSE
        @test cv_k2.mse < cv_k1.mse
        @test cv_k2.loglik > cv_k1.loglik
    end

    # -------------------------------------------------------------------------
    # 4. CVResult API & Show Extractor Methods
    # -------------------------------------------------------------------------
    @testset "CVResult extractors and display" begin
        Random.seed!(707)
        p, n = 4, 20
        Y = randn(p, n)
        res = cv_gllvm(Y; k_folds = 3, split = :random, family = Normal(), K = 1)

        @test loglikelihood(res) == res.loglik
        @test residuals(res) == res.residuals
        @test residuals(res; type = :dunnsmyth) == res.residuals
        @test predict(res) == res.predictions
        @test fitted(res) == res.predictions

        # Display show test
        buf = IOBuffer()
        show(buf, MIME("text/plain"), res)
        str = String(take!(buf))
        @test occursin("GLLVModels 3-Fold Cross-Validation", str)
        @test occursin("Out-of-sample logLik", str)
        @test occursin("Out-of-sample MSE", str)
        @test occursin("Residual Mean", str)
        @test occursin("Fold breakdown", str)

        show(buf, res)
        str_short = String(take!(buf))
        @test occursin("CVResult(3-fold random", str_short)
    end

    # -------------------------------------------------------------------------
    # 5. Input Validation & Error Handling
    # -------------------------------------------------------------------------
    @testset "CV input validation" begin
        Y = randn(4, 10)
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 1)           # k_folds < 2
        @test_throws ArgumentError cv_gllvm(Y; split = :invalid_split) # invalid split
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 20, split = :site) # k_folds > n_sites
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 10, split = :species) # k_folds > n_species
    end

end

@testset "cv_gllvm threads a Gaussian X through every split" begin
    # Regression: the Gaussian CV path passed the full X to fold fits on subset
    # data, scored sites at a zero mean, and read the covariate β as
    # per-species intercepts. Each split must now use X[t, s, :] ⋅ β.
    rng = StableRNG(41)
    p, n, K = 6, 60, 1
    x = randn(rng, n)
    X = zeros(p, n, 2); X[:, :, 1] .= 1.0; X[:, :, 2] .= x'
    Λ = 0.6 .* randn(rng, p, K)
    Y = 3.0 .+ 2.0 .* x' .+ Λ * randn(rng, K, n) .+ 0.3 .* randn(rng, p, n)

    @testset "site split: held-out sites are predicted by X β" begin
        cv = cv_gllvm(Y; k_folds = 3, split = :site, family = Normal(), K = K,
                      X = X, rng = StableRNG(1))
        @test all(isfinite, cv.predictions)
        for s in 1:n
            @test any(cv.fits) do f
                isapprox(cv.predictions[:, s],
                         [dot(X[t, s, :], f.pars.β) for t in 1:p]; atol = 1e-8)
            end
        end
    end

    @testset "$split split: X lowers out-of-sample error" for split in (:random, :site, :species)
        cvX = cv_gllvm(Y; k_folds = 3, split = split, family = Normal(), K = K,
                       X = X, rng = StableRNG(2))
        cv0 = cv_gllvm(Y; k_folds = 3, split = split, family = Normal(), K = K,
                       rng = StableRNG(2))
        @test all(isfinite, cvX.predictions)
        @test all(f -> length(f.pars.β) == 2, cvX.fits)
        # Oracle: predict every cell at its true mean 3 + 2x.
        oracle = mean(abs2, Y .- (3.0 .+ 2.0 .* x'))
        @test abs(mean(Y .- cvX.predictions)) < 0.1
        if split === :random
            # Other cells at the same site inform ẑ, so X + LV beats the mean alone.
            @test cvX.mse < oracle
        else
            # A held-out site (or species) has no latent information: X β is the best
            # available predictor. Without X the Normal route fits only trait intercepts
            # (#519), so it recovers the mean 3 but misses the 2x slope term entirely
            # (measured: about 9x the oracle's error on this data).
            @test cvX.mse < 1.15 * oracle
            @test cv0.mse > 5 * oracle
        end
    end

    @testset "X with the wrong shape is rejected" begin
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 3, split = :site,
                                            family = Normal(), K = K, X = X[:, 1:10, :])
    end

    @testset "non-Gaussian families reject X" begin
        Yc = rand(StableRNG(3), 0:5, p, n)
        @test_throws ArgumentError cv_gllvm(Yc; k_folds = 3, split = :site,
                                            family = Poisson(), K = K, X = X)
    end
end

@testset "loading GLLVModels raises no identifier-conflict warning" begin
    # Regression: a bare `using Distributions` in src/cv.jl re-imported
    # `Distributions.Multinomial` next to the package's own `Multinomial`,
    # which src/GLLVModels.jl deliberately leaves out, so precompiling the
    # package printed "conflicts with an existing identifier". The warning is
    # emitted when the module is compiled, so load it from source here.
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no --compiled-modules=no -e "using GLLVModels"`
    err = IOBuffer()
    run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err))
    msg = String(take!(err))
    @test !occursin("conflicts with an existing identifier", msg)
end
