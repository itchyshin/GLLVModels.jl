using GLLVModels, Test, Random, Distributions
using StatsModels: @formula

@testset "prior-table loading adapter" begin
    loading = (;
        prior = ["ridge(4)"],
        class = ["loading"],
        coef = ["all"],
        dpar = ["mu"],
        scale = ["loading_coef"],
    )
    @test GLLVModels._gllv_prior_table_controls(loading).loading_ridge == 0.5
    default_selectors = (; prior = ["ridge(4)"], class = ["loading"], coef = ["all"])
    @test GLLVModels._gllv_prior_table_controls(default_selectors).dpar == "mu"
    @test GLLVModels._gllv_prior_table_controls(default_selectors).scale == "loading_coef"
    no_penalty = merge(default_selectors, (; prior = ["ridge(0)"]))
    @test isinf(GLLVModels._gllv_prior_table_controls(no_penalty).loading_ridge)
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; coef = ["temperature"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; dpar = ["sigma"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; scale = ["link"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; group = ["site"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; prior = ["normal(0,2.5)"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; prior = ["ridge(-1)"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; prior = ["ridge(NaN)"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(loading, (; unknown = ["x"])))

    Random.seed!(614)
    Y = Int.(rand(5, 32) .< 0.43)
    from_table = fit_gllvm(Y; family = Binomial(), K = 1, priors = loading,
                           iterations = 80, newton_maxiter = 100)
    legacy_sd = inv(sqrt(0.5)) # ridge(lambda) maps to loading_ridge = 1/sqrt(lambda)
    from_legacy = fit_gllvm(Y; family = Binomial(), K = 1, loading_ridge = legacy_sd,
                            iterations = 80, newton_maxiter = 100)
    @test only(from_table.prior_metadata.resolved_targets).class == "loading"
    @test from_table.prior_metadata.input_table.prior == ["ridge(0.5)"]
    @test from_table.prior_metadata.loading_penalty ≈ 0.25 * sum(abs2, from_table.Λ)
    table_objective = from_table.prior_metadata.total_objective
    legacy_objective = -from_legacy.loglik + 0.25 * sum(abs2, from_legacy.Λ)
    # The two paths optimize the same criterion, but their latent-factor
    # coordinates are sign-indeterminate and their finite optimizer stops need
    # not return bitwise-identical objective values.
    @test table_objective ≈ legacy_objective atol=1e-5 rtol=1e-8
    @test from_table.β ≈ from_legacy.β atol=1e-5 rtol=1e-8
    @test from_table.Λ * from_table.Λ' ≈ from_legacy.Λ * from_legacy.Λ' atol=1e-5 rtol=1e-8
    @test from_table.convergence.penalized
    @test from_table.convergence.objective ≈ from_table.loglik + from_table.convergence.log_prior
    @test from_table.convergence.se_type === :penalised

    @test_throws ArgumentError fit_gllvm(Y; family = Normal(), K = 1, priors = loading)
    @test_throws ArgumentError fit_gllvm(Y; family = Binomial(), K = 1,
        priors = loading, loading_ridge = 0.5)

    response_prior = (; prior=["normal(0, 2)"], class=["fixef"],
        coef=["bird::(Intercept)"], dpar=["mu"], scale=["link"])
    named_Y = Int.(rand(MersenneTwister(991), 3, 36) .< 0.45)
    intercept_fit = fit_gllvm(named_Y; family=Binomial(), K=1,
        priors=response_prior, response_names=["bird", "fish", "mammal"],
        iterations=80, newton_maxiter=80)
    @test intercept_fit.prior_metadata !== nothing
    @test first(intercept_fit.prior_metadata.resolved_targets).response == "bird"
    @test intercept_fit.convergence.objective ≈ intercept_fit.loglik +
        intercept_fit.convergence.log_prior
    all_intercepts = (; prior=["normal(0, 2)"], class=["fixef"], coef=["all"])
    all_intercept_fit = fit_gllvm(named_Y; family=Binomial(), K=1,
        priors=all_intercepts, response_names=["bird", "fish", "mammal"],
        iterations=80, newton_maxiter=80)
    @test count(t -> t.class == "fixef", all_intercept_fit.prior_metadata.resolved_targets) == 3
    function assert_prior_route_rejected(; kwargs...)
        caught = try
            fit_gllvm(Y; family = Binomial(), K = 1, priors = loading, kwargs...)
            nothing
        catch err
            err
        end
        @test caught isa ArgumentError
        @test caught !== nothing && occursin("prior table", lowercase(sprint(showerror, caught)))
    end
    assert_prior_route_rejected(grouping = :unsupported)
    assert_prior_route_rejected(phylo = :unsupported)
    assert_prior_route_rejected(row_eff = :random)
    assert_prior_route_rejected(disp_group = :shared)
    assert_prior_route_rejected(pervar = true)
    assert_prior_route_rejected(weights = ones(size(Y)))
    assert_prior_route_rejected(aghq = true)
    assert_prior_route_rejected(X = ones(5, 32, 1))

    fixed = (;
        prior = ["normal(0,2.5)"],
        class = ["fixef"],
        coef = ["temperature"],
        dpar = ["mu"],
        scale = ["link"],
    )
    @test_throws ArgumentError fit_gllvm(Y; family = Binomial(), K = 1, priors = fixed)

    duplicate = (;
        prior = ["ridge(4)", "ridge(9)"],
        class = ["loading", "loading"],
        coef = ["all", "all"],
        dpar = ["mu", "mu"],
        scale = ["loading_coef", "loading_coef"],
    )
    @test_throws ArgumentError fit_gllvm(Y; family = Binomial(), K = 1, priors = duplicate)
end

@testset "prior-table named response coefficients" begin
    normal = (; prior = ["normal(0,2.5)"], class = ["fixef"],
              coef = ["temperature"], dpar = ["mu"], scale = ["link"])
    normal_controls = GLLVModels._gllv_prior_table_controls(normal)
    @test isinf(normal_controls.loading_ridge)
    @test length(normal_controls.coefficient_priors) == 1
    @test only(normal_controls.coefficient_priors) ==
          (; coef = "temperature", kind = :normal, location = 0.0, scale = 2.5, df = nothing)

    student = merge(normal, (; prior = ["student_t(0,1,4)"]))
    student_prior = only(GLLVModels._gllv_prior_table_controls(student).coefficient_priors)
    @test student_prior ==
          (; coef = "temperature", kind = :student_t, location = 0.0, scale = 1.0, df = 4.0)

    combined = (; prior = ["normal(0,2.5)", "ridge(4)"],
                class = ["fixef", "loading"],
                coef = ["temperature", "all"],
                dpar = ["mu", "mu"],
                scale = ["link", "loading_coef"])
    combined_controls = GLLVModels._gllv_prior_table_controls(combined)
    @test combined_controls.loading_ridge == 0.5
    @test only(combined_controls.coefficient_priors).coef == "temperature"

    duplicate = (; prior = ["normal(0,1)", "student_t(0,1,4)"],
                 class = ["fixef", "fixef"], coef = ["temperature", "temperature"],
                 dpar = ["mu", "mu"], scale = ["link", "link"])
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(duplicate)
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(normal, (; scale = ["natural_sd"])))
    @test_throws ArgumentError GLLVModels._gllv_prior_table_controls(
        merge(normal, (; prior = ["student_t(0,1,0)"])))

    Random.seed!(2047)
    p, n = 5, 40
    x = randn(n)
    X = reshape(repeat(x', p, 1), p, n, 1)
    Y = Int.(rand(p, n) .< 0.45)
    fit = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], priors = combined,
        iterations = 120, newton_maxiter = 100)
    @test fit.prior_metadata !== nothing
    @test fit.prior_metadata.resolved_targets == [(class = "fixef", coef = "temperature", dpar = "mu", scale = "link"),
        (class = "loading", coef = "all", dpar = "mu", scale = "loading_coef")]
    @test fit.prior_metadata.fixed_effect_penalty > 0
    @test fit.prior_metadata.loading_penalty ≈ 2 * sum(abs2, fit.Λ)
    @test fit.prior_metadata.total_objective ≈ -fit.loglik +
        fit.prior_metadata.fixed_effect_penalty + fit.prior_metadata.loading_penalty
    @test_throws ArgumentError confint(fit, Y; X = X)

    temperature_index = findfirst(==("temperature"), fit.prior_metadata.coefficient_names)
    @test temperature_index !== nothing
    temperature_estimate = fit.γ[temperature_index]
    expected_normal_penalty = log(2.5) + 0.5 * log(2π) + 0.5 * (temperature_estimate / 2.5)^2
    @test fit.prior_metadata.fixed_effect_penalty ≈ expected_normal_penalty

    formula_fit = gllvm(@formula(y ~ 1 + temperature), Y, (temperature = x,);
        family = Binomial(), K = 1, priors = combined,
        iterations = 120, newton_maxiter = 100)
    @test formula_fit isa GllvmCovFit
    @test formula_fit.prior_metadata.resolved_targets[1].coef == "temperature"
    @test formula_fit.prior_metadata.loading_penalty ≈ 0.5 * 4 * sum(abs2, formula_fit.Λ)

    student_fit = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], priors = student,
        iterations = 120, newton_maxiter = 100)
    student_spec = LocationScale(0.0, 1.0, TDist(4.0))
    @test student_fit.prior_metadata.fixed_effect_penalty ≈
          -logpdf(student_spec, student_fit.γ[1])
    @test student_fit.prior_metadata.loading_penalty == 0
    @test_throws ArgumentError fit_gllvm_cov(Y; family = Poisson(), X = X, K = 1,
        coefficient_names = ["temperature"], priors = normal)
    @test_throws ArgumentError fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], priors = merge(normal, (; coef = ["unknown"])))

    # A `fixef` selector may name either a shared covariate coefficient or a
    # response-specific intercept. The latter is identified by response_names,
    # since a wide numeric Y matrix carries no row names.
    response_names = ["taxon_a", "taxon_b", "taxon_c", "taxon_d", "taxon_e"]
    intercept_prior = (; prior = ["normal(1,0.8)"], class = ["fixef"],
                       coef = ["taxon_a"], dpar = ["mu"], scale = ["link"])
    intercept_fit = fit_gllvm_cov(Y; family = Binomial(), X = zeros(p, n, 0), K = 1,
        response_names = response_names, priors = intercept_prior,
        iterations = 120, newton_maxiter = 100)
    @test intercept_fit.prior_metadata.resolved_targets ==
        [(class = "fixef", coef = "taxon_a", dpar = "mu", scale = "link",
          parameter_role = :response_intercept, response = "taxon_a")]
    @test intercept_fit.prior_metadata.response_names == response_names
    @test intercept_fit.prior_metadata.fixed_effect_penalty ≈
        log(0.8) + 0.5 * log(2π) + 0.5 * ((intercept_fit.β[1] - 1) / 0.8)^2
    default_response_fit = fit_gllvm_cov(Y; family = Binomial(), X = zeros(p, n, 0), K = 1,
        priors = merge(intercept_prior, (; coef = ["trait_1"])),
        iterations = 120, newton_maxiter = 100)
    @test default_response_fit.prior_metadata.response_names == ["trait_$i" for i in 1:p]
    @test default_response_fit.prior_metadata.resolved_targets[1].response == "trait_1"

    mixed_intercept_and_slope = (; prior = ["normal(1,0.8)", "student_t(0,1,4)",
                                               "normal(0,1.5)", "ridge(4)"],
        class = ["fixef", "fixef", "fixef", "loading"],
        coef = ["taxon_a", "taxon_b", "temperature", "all"],
        dpar = ["mu", "mu", "mu", "mu"],
        scale = ["link", "link", "link", "loading_coef"])
    mixed_fit = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], response_names = response_names,
        priors = mixed_intercept_and_slope, iterations = 120, newton_maxiter = 100)
    @test mixed_fit.prior_metadata.resolved_targets[1].parameter_role == :response_intercept
    @test mixed_fit.prior_metadata.resolved_targets[2].parameter_role == :response_intercept
    @test mixed_fit.prior_metadata.resolved_targets[3].parameter_role === :covariate
    @test mixed_fit.prior_metadata.fixed_effect_penalty ≈
        (log(0.8) + 0.5 * log(2π) + 0.5 * ((mixed_fit.β[1] - 1) / 0.8)^2) -
        logpdf(student_spec, mixed_fit.β[2]) +
        (log(1.5) + 0.5 * log(2π) + 0.5 * (mixed_fit.γ[1] / 1.5)^2)
    @test mixed_fit.prior_metadata.loading_penalty ≈ 2 * sum(abs2, mixed_fit.Λ)

    formula_intercept_fit = gllvm(@formula(y ~ 1), Y, NamedTuple();
        family = Binomial(), K = 1, response_names = response_names,
        priors = intercept_prior, iterations = 120, newton_maxiter = 100)
    @test formula_intercept_fit.prior_metadata.resolved_targets[1].response == "taxon_a"
    long_prior_data = (;
        y = vec(Y),
        species = repeat(response_names, n),
        site = repeat(1:n, inner = p),
        temperature = repeat(x, inner = p),
    )
    long_fit = gllvm(@formula(y ~ 1 + temperature), long_prior_data;
        family = Binomial(), K = 1, priors = mixed_intercept_and_slope,
        iterations = 120, newton_maxiter = 100)
    @test long_fit.prior_metadata.resolved_targets[1].response == "taxon_a"

    @test_throws DimensionMismatch fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], response_names = ["taxon_a"], priors = intercept_prior)
    @test_throws ArgumentError fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], response_names = fill("taxon_a", p), priors = intercept_prior)
    ambiguous_prior = merge(intercept_prior, (; coef = ["temperature"]))
    @test_throws ArgumentError fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1,
        coefficient_names = ["temperature"], response_names = ["temperature", "b", "c", "d", "e"],
        priors = ambiguous_prior)
    @test_throws ArgumentError gllvm(@formula(y ~ 0), Y, NamedTuple();
        family = Binomial(), K = 1, response_names = response_names,
        priors = intercept_prior)
end
