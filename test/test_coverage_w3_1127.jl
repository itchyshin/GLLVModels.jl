# Targeted coverage for w3 ranks 4–6: twopart.jl, bridge.jl, grouped_nongaussian_fit.jl
# (refs coverage 27 Nov). Behaviour-only; no src edits.
using Test, GLLVModels, LinearAlgebra, SparseArrays, Random, Distributions

@testset "bridge option coercion (#coverage-w3)" begin
    truthy = GLLVModels._bridge_truthy
    @test truthy(true)
    @test !truthy(false)
    @test truthy(1) && !truthy(0)
    @test truthy("true") && truthy("YES") && truthy("1")
    @test !truthy("false") && !truthy(nothing)
    @test !truthy(:symbol)

    @test_throws ArgumentError GLLVModels._bridge_names(["only"], 3, "trait")

    @test_throws ArgumentError GLLVModels._bridge_family_key("not_a_family")

    @test GLLVModels._bridge_ci_method(Dict("ci_method" => "wald")) == "wald"
    @test_throws ArgumentError GLLVModels._bridge_ci_method(Dict("ci_method" => "lrt"))

    opts = Dict("ci_method" => "bootstrap", "ci_nboot" => 50)
    @test GLLVModels._bridge_ci_nboot(opts) == 50
    bad = Dict("ci_method" => "bootstrap", "ci_nboot" => 2.5)
    @test_throws ArgumentError GLLVModels._bridge_ci_nboot(bad)
end

@testset "bridge CI routing guards (#coverage-w3)" begin
    @test GLLVModels._bridge_ci_guard_lognormal("none") === nothing
    @test_throws ArgumentError GLLVModels._bridge_ci_guard_lognormal("wald")

    @test GLLVModels._bridge_ci_guard_truncated_poisson("none") === nothing
    @test_throws ArgumentError GLLVModels._bridge_ci_guard_truncated_poisson("profile")

    @test GLLVModels._bridge_ci_guard_pertrait_ordinal("ordinal", "none") === nothing
    @test_throws ArgumentError GLLVModels._bridge_ci_guard_pertrait_ordinal(
        "ordinal", "bootstrap")
end

@testset "bridge dispersion payload helpers (#coverage-w3)" begin
    payload = GLLVModels._bridge_dispersion_payload(
        [1.0, 2.0], [1, 2, 1], "phi", "log", "natural")
    @test payload.dispersion_engine_scale == "log"
    @test GLLVModels._bridge_expand_dispersion(payload) ≈ [1.0, 2.0, 1.0]
    @test GLLVModels._bridge_group_df(3, 1, payload) ==
        3 + GLLVModels._bridge_rr_df(3, 1) + 2
end

@testset "bridge integer count contract (#coverage-w3)" begin
    Y = [1.0 2.0; 3.0 4.0]
    @test GLLVModels._bridge_require_integer_counts(Y, "poisson") == [1 2; 3 4]
    bad = copy(Y)
    bad[1, 2] = 1.5
    @test_throws ArgumentError GLLVModels._bridge_require_integer_counts(bad, "zip")
end

@testset "grouped non-Gaussian family guards (#coverage-w3)" begin
    @test GLLVModels._grouped_nongaussian_kind(GLLVModels.Poisson()) === :poisson
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(
        GLLVModels.Beta(2.0, 2.0))
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(
        GLLVModels.Beta(Inf, 1.0))
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(
        GLLVModels.NegativeBinomial(2.0, 0.4))
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_kind(GLLVModels.Gamma(2.0, 1.0))

    @test GLLVModels._grouped_nongaussian_dispersion_mode(:poisson, :shared) === :none
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_dispersion_mode(:beta, :bogus)

    @test GLLVModels._grouped_nongaussian_internal_dispersion_mode(:poisson, :trait) === :none
    @test_throws ArgumentError GLLVModels._grouped_nongaussian_internal_dispersion_mode(
        :beta, :none)

    Z = GLLVModels._grouped_incidence([1, 1, 2, 2], 4)
    loads = [reshape([0.5, 0.4], 2, 1)]
    @test_throws DimensionMismatch GLLVModels._grouped_laplace_design(
        [Z, Z], loads)

    p, n = 1, 8
    unit = repeat(1:4; inner = 2)
    Y = reshape([1, 2, 1, 2, 3, 4, 3, 4], 1, :)
    term = GLLVModels.GroupingTerm
    @test_throws ArgumentError GLLVModels.fit_grouped_nongaussian(Y;
        family = GLLVModels.Poisson(),
        terms = [term(:unit; mode = :indep, common = true)], unit = unit,
        hessian = :observed)

    θ = [0.0, log(1.5)]
    fam = GLLVModels._grouped_nongaussian_family(:beta, θ, 2:2, 1, 4, :shared)
    @test fam isa GLLVModels.Beta && fam.α ≈ 1.5
    @test GLLVModels._grouped_nongaussian_dispersion(θ, 2:2, :beta, :shared) ≈ 1.5
    @test isfinite(GLLVModels._grouped_moment_log_sd(
        Float64[1, 2, 3, 4], [1, 1, 2, 2]))
end

@testset "twopart marginal and driver contracts (#coverage-w3)" begin
    p, n, K = 2, 6, 1
    Y = [0.0 1.2 0.0 2.0 0.0 3.0;
         1.0 0.0 2.5 0.0 0.5 0.0]
    Λc = [0.4 0.2; 0.3 0.1][:, 1:K]
    βz = [0.1, -0.2]
    βc = [0.0, 0.1]
    σ = [0.5, 0.6]
    @test isfinite(GLLVModels.delta_lognormal_marginal_loglik_laplace(
        Y, Λc, βz, βc, σ))
    @test_throws ArgumentError GLLVModels.delta_lognormal_marginal_loglik_laplace(
        Y, Λc, βz, βc, [0.5])

    @test_throws ArgumentError GLLVModels.fit_delta_lognormal_gllvm(Y; K = K,
        predictor = :joint)
    @test_throws ArgumentError GLLVModels.fit_delta_lognormal_gllvm(Y; K = K,
        disp_group = :trait)

    fit = GLLVModels.DeltaLogNormalFit(
        βz, βc, Λc, σ, -100.0, true, 3, :separate, :species, nothing)
    @test occursin("DeltaLogNormalFit", sprint(show, fit))
    @test occursin("p=2", sprint(show, fit))

    GLLVModels._check_twopart_support("coverage_w3", reshape([0.0 1.0], 1, 2),
        GLLVModels._tp_positive_ok, "0 or positive")
    bad = reshape([0.0 NaN], 1, 2)
    @test_throws ArgumentError GLLVModels._check_twopart_support(
        "coverage_w3", bad, GLLVModels._tp_positive_ok, "0 or positive")
end