# Coverage lane w7 (27 Nov): ranks 16–18 on main baseline (skipped open-PR phylo xLV #17).
# Targets: grouped_nongaussian_fit.jl, cv.jl, lowrank_cholesky.jl — validation and contract branches.
using GLLVModels, Test, Random, LinearAlgebra, Distributions

@testset "coverage 1127cov w7" begin

    @testset "lowrank Cholesky matvec and guards" begin
        Λ = reshape([0.5, 0.2, 0.1], 3, 1)
        d = [1.0, 1.2, 0.8]
        F = GLLVModels.low_rank_chol(Λ, d)
        x = [1.0, -0.5, 2.0]
        y = F * x
        y2 = similar(y)
        mul!(y2, F, x)
        @test y ≈ y2
        X = randn(3, 4)
        Z = F * X
        Z2 = similar(X)
        mul!(Z2, F, X)
        @test Z ≈ Z2
        @test_throws DimensionMismatch mul!(y2, F, [1.0, 2.0])
        @test_throws DimensionMismatch mul!(zeros(2, 2), F, X)
        @test_throws DimensionMismatch mul!(zeros(3, 2), F, randn(2, 4))
        @test isfinite(logdet(F))
    end

    @testset "grouped non-Gaussian validation and init branches" begin
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(Gamma())
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(Beta(NaN, 1.0))
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(
            NegativeBinomial(1.0, 0.3))
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_dispersion_mode(:poisson, :bad)
        @test_throws ArgumentError GLLVModels._grouped_nongaussian_internal_dispersion_mode(:poisson, :bogus)

        p, n = 2, 6
        Random.seed!(1127031)
        data = Float64.(max.(0, rand(Poisson(3), p, n)))
        trials = GLLVModels._grouped_nongaussian_trials(data, nothing, :poisson)
        D = GLLVModels._trait_mean_design(p, n)
        term = GLLVModels.GroupingTerm(:unit; mode = :indep, common = false)
        labels = [[mod1(s, 2) for s in 1:n]]
        θ = GLLVModels._grouped_nongaussian_initial_parameters(
            data, trials, D, [term], :poisson, Poisson(), :none;
            group_labels = labels, moment_start = true)
        @test length(θ) > 0
        @test all(isfinite, θ)

        fit = GLLVModels.fit_grouped_nongaussian(data; family = Poisson(),
            terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)],
            unit = collect(1:n), iterations = 40)
        @test fit isa GLLVModels.GroupedNonGaussianFit
        @test_throws DimensionMismatch GLLVModels.grouped_nongaussian_intervals(
            vcat(data, data[1:1, :]), fit; unit = collect(1:n))
        Ybad = zeros(BigFloat, p, n)
        @test_throws ArgumentError GLLVModels.grouped_nongaussian_intervals(Ybad, fit; unit = collect(1:n))
    end

    @testset "cv.jl helpers and masked Gaussian fold" begin
        p, n = 2, 3
        fit_po = GLLVModels.PoissonFit(
            zeros(p), 0.1 .* ones(p, 1), GLLVModels.LogLink(),
            -1.0, true, 1, nothing, zeros(0), :laplace, nothing, nothing, nothing, nothing)
        Λ, β, link = GLLVModels._cv_extract_params(fit_po)
        @test link isa GLLVModels.LogLink
        @test Λ ≈ fit_po.Λ

        struct _CvBetaOnly
            Λ::Matrix{Float64}
            β::Vector{Float64}
        end
        Λb, βb, linkb = GLLVModels._cv_extract_params(_CvBetaOnly(ones(2, 1), [0.1, 0.2]))
        @test linkb isa GLLVModels.IdentityLink

        struct _CvNestedPars
            pars::NamedTuple
        end
        Λn, βn, _ = GLLVModels._cv_extract_params(_CvNestedPars((
            Λ = ones(2, 1), β = [0.0, 0.0])))
        @test size(Λn) == (2, 1)

        struct _CvParsOnly
            pars::NamedTuple
        end
        σ = 0.42
        mock = _CvParsOnly((σ_eps = σ,))
        ll, r = GLLVModels._cv_cell_eval(mock, Normal(), 0.1, 0.0, 1, 1, nothing, Random.MersenneTwister(1))
        @test isfinite(ll) && isfinite(r)

        kwargs = (weights = ones(p, n), disp_group = ones(Int, p))
        ts = [1]
        sub = GLLVModels._cv_species_subset_kwargs(kwargs, ts, p)
        @test size(sub.weights, 1) == 1
        @test sub.disp_group == [1]

        Y = [1.0 2.0 missing; 0.5 1.5 3.0]
        mask = BitMatrix([true true false; true true true])
        rng = Random.MersenneTwister(1127)
        res = cv_gllvm(Y; family = Normal(), K = 1, k_folds = 2, split = :random,
                       mask = mask, rng = rng, iterations = 20)
        @test res isa GLLVModels.CVResult
        @test isfinite(res.loglik) || res.loglik == -Inf

        Yint = [0 1 0 1; 1 0 1 0]
        res_p = cv_gllvm(Yint; family = Poisson(), K = 1, k_folds = 2, split = :species,
                         rng = Random.MersenneTwister(42), iterations = 25)
        @test res_p isa GLLVModels.CVResult

        fit_p2 = GLLVModels.PoissonFit(
            [0.0, 0.0], reshape([0.2, 0.1], 2, 1), GLLVModels.LogLink(),
            -1.0, true, 1, nothing, zeros(0), :laplace, nothing, nothing, nothing, nothing)
        Yp = [1 2; 0 1; 2 0]
        @test_throws DimensionMismatch GLLVModels._cv_site_mode(
            fit_p2, Poisson(), Yp, 1, nothing, nothing, [1, 2, 3]; X = nothing)
    end
end