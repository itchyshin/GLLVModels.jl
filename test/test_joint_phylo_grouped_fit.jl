using Test
using GLLVModels
using LinearAlgebra
using SparseArrays
using StableRNGs


# ADEMP smoke: one predeclared StableRNG draw from the joint covariance, with
# no recovery or coverage claim.  Y = D*beta + A_phy*f + W_group*b + e.
function _jpgf_fixture()
    rng = StableRNG(84031)
    qbase = [3.0 -0.8 -0.5 0.0;
             -0.8 3.2 0.0 -0.6;
             -0.5 0.0 2.7 0.0;
             0.0 -0.6 0.0 2.9]
    Q = sparse(1.4 .* qbase)
    ii, jj, xx = findnz(Q)
    phy = PrecisionPhy(ii, jj, xx, 4, 2, ["a1", "a2", "tip1", "tip2"],
        logdet(cholesky(Symmetric(Matrix(Q)))), 1.4, [3, 4])
    species_id = repeat([1, 2], 6)
    n, p = length(species_id), 2
    unit = [1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 3, 2]
    incidence = sparse(collect(1:n), unit, ones(n), n, 3)
    loading = reshape([0.55, 0.30], p, 1)
    psi = [0.35, 0.48]
    grouped_variance = [0.16, 0.11]
    selection = zeros(n, phy.n_aug)
    for i in 1:n
        selection[i, phy.species_aug_id[species_id[i]]] = 1.0
    end
    covariance = Matrix(kron(selection * (Matrix(phy.Q) \ selection'), loading * loading') +
        kron(Matrix(incidence * incidence'), Diagonal(grouped_variance)) +
        kron(Matrix(I, n, n), Diagonal(psi)))
    D = GLLVModels._trait_mean_design(p, n)
    beta = [0.15, -0.20]
    response = reshape(D * beta + cholesky(Symmetric(covariance)).L * randn(rng, p * n), p, n)
    return (; phy, species_id, unit, incidence, loading, psi, grouped_variance,
        covariance, D, beta, response)
end

@testset "private joint phylo plus grouped Gaussian fit" begin
    fixture = _jpgf_fixture()
    terms = [GroupingTerm(:unit; mode = :indep, common = false)]
    fit = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
        rank = 1, phylo_mode = :barelowrank, terms = terms, unit = fixture.unit,
        species_id = fixture.species_id, iterations = 120, g_tol = 2e-4)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test fit.stopping_reason === :converged
    @test length(fit.parameters) == 8
    @test length(fit.ordinary_covariances) == 1
    @test fit.phylo_covariance ≈ fit.loading * fit.loading'
    @test all(isfinite, fit.residual_variance)
    @test all(isfinite, diag(only(fit.ordinary_covariances)))

    objective = GLLVModels._joint_phylo_grouped_nll(fit.response, fit.phy, fit.terms,
        fit.incidences; rank = fit.rank, phylo_mode = fit.phylo_mode,
        species_id = fit.species_id, mean_design = fit.mean_design)
    @test isapprox(-objective(fit.parameters), fit.loglik; atol = 1e-8, rtol = 1e-8)
    alternate_unit = reverse(fixture.unit)
    alternate_incidence = GLLVModels._grouped_incidence(alternate_unit, length(alternate_unit))
    alternate_objective = GLLVModels._joint_phylo_grouped_nll(fit.response, fit.phy, fit.terms,
        [alternate_incidence]; rank = fit.rank, phylo_mode = fit.phylo_mode,
        species_id = fit.species_id, mean_design = fit.mean_design)
    @test !isapprox(objective(fit.parameters), alternate_objective(fit.parameters);
        atol = 1e-8, rtol = 1e-8)
    invalid_stencil_theta = fill(1e6, length(fit.parameters))
    @test GLLVModels._fd_failed(objective(invalid_stencil_theta))
    @test !all(isfinite, GLLVModels._fd_hessian(objective, invalid_stencil_theta))
    @test !GLLVModels._pmv_hessian_diagnostics(objective, invalid_stencil_theta).positive_definite

    interval = GLLVModels.joint_phylo_grouped_intervals(fit)
    @test interval.status in (:available, :partial, :not_stationary, :invalid_curvature)
    interval_names = String[row.name for row in interval.intervals]
    @test "beta[1]" in interval_names
    @test "phylo_cov[1,1]" in interval_names
    @test "residual_var[1]" in interval_names
    @test "unit_var[1]" in interval_names
    stalled = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
        rank = 1, phylo_mode = :barelowrank, terms = terms, unit = fixture.unit,
        species_id = fixture.species_id, iterations = 0, g_tol = 2e-4)
    @test !stalled.converged
    @test GLLVModels.joint_phylo_grouped_intervals(stalled).status === :not_converged
    @test_throws ArgumentError GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response,
        fixture.phy; rank = 1, phylo_mode = :barelowrank, terms = terms,
        unit = fixture.unit, species_id = fixture.species_id,
        X = zeros(size(fixture.response, 1) * size(fixture.response, 2), 1), iterations = 0)
    @test_throws ArgumentError GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response,
        fixture.phy; rank = 1, phylo_mode = :explicitunique, terms = terms,
        unit = fixture.unit, species_id = fixture.species_id, iterations = 0)

    source_fixture = _jpgf_fixture()
    source_terms = [GroupingTerm(:unit; mode = :indep, common = false)]
    source_fit = GLLVModels.fit_joint_phylo_grouped_gaussian(source_fixture.response,
        source_fixture.phy; rank = 1, phylo_mode = :barelowrank, terms = source_terms,
        unit = source_fixture.unit, species_id = source_fixture.species_id,
        iterations = 120, g_tol = 2e-4)
    source_objective = GLLVModels._joint_phylo_grouped_nll(source_fit.response, source_fit.phy,
        source_fit.terms, source_fit.incidences; rank = source_fit.rank,
        phylo_mode = source_fit.phylo_mode, species_id = source_fit.species_id,
        mean_design = source_fit.mean_design)
    baseline = source_objective(source_fit.parameters)
    source_fixture.phy.Q.nzval[1] *= 2.0
    source_fixture.species_id[1] = 2
    source_terms[1] = GroupingTerm(:cluster; mode = :indep, common = false)
    @test source_fit.phy.Q.nzval[1] != source_fixture.phy.Q.nzval[1]
    @test source_fit.species_id[1] != source_fixture.species_id[1]
    @test source_fit.terms[1].name === :unit
    @test isapprox(source_objective(source_fit.parameters), baseline; atol = 1e-10, rtol = 1e-10)
end

# residual_mode = :shared on the joint fitter: one common observation residual
# variance (the layout of R's Gaussian route) instead of one per trait. The
# default :trait path must stay exactly as it was. Declared-point algebra only:
# no R number is compared and nothing here is a parity claim.
@testset "joint phylo plus grouped Gaussian fit: residual_mode" begin
    fixture = _jpgf_fixture()
    terms = [GroupingTerm(:unit; mode = :indep, common = false)]
    common = (rank = 1, phylo_mode = :barelowrank, terms = terms, unit = fixture.unit,
        species_id = fixture.species_id)
    p = size(fixture.response, 1)

    # Packed start [beta(2); loading(2); residual log-sd; ordinary log-sd(2)].
    beta = [0.10, -0.05]
    loading = [0.40, 0.25]
    ordinary = log.(sqrt.([0.16, 0.11]))
    s = log(sqrt(0.30))
    theta_trait = vcat(beta, loading, fill(s, p), ordinary)
    theta_shared = vcat(beta, loading, s, ordinary)

    default_fit = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
        common..., start = theta_trait, iterations = 0)
    trait_fit = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
        common..., residual_mode = :trait, start = theta_trait, iterations = 0)
    shared_fit = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
        common..., residual_mode = :shared, start = theta_shared, iterations = 0)

    @testset "default is :trait and unchanged" begin
        @test default_fit.residual_mode === :trait
        @test trait_fit.residual_mode === :trait
        @test default_fit.parameters == trait_fit.parameters
        @test default_fit.parameter_labels == trait_fit.parameter_labels
        @test default_fit.loglik === trait_fit.loglik
        @test length(default_fit.parameters) == 8
        @test count(startswith("log_sd_psi"), default_fit.parameter_labels) == p
        @test "log_sd_psi_shared" ∉ default_fit.parameter_labels
        @test GLLVModels._joint_phylo_grouped_fit_layout(p, 1, :barelowrank, 2, terms).total == 8
    end

    @testset ":shared has p - 1 fewer parameters and shared labels" begin
        @test shared_fit.residual_mode === :shared
        @test length(shared_fit.parameters) == length(default_fit.parameters) - (p - 1)
        @test length(shared_fit.parameter_labels) == length(shared_fit.parameters)
        @test count(==("log_sd_psi_shared"), shared_fit.parameter_labels) == 1
        @test !any(startswith("log_sd_psi["), shared_fit.parameter_labels)
        @test shared_fit.residual_variance ≈ fill(0.30, p) rtol = 1e-12
        layout = GLLVModels._joint_phylo_grouped_fit_layout(p, 1, :barelowrank, 2, terms;
            residual_mode = :shared)
        @test length(layout.residual) == 1
        @test layout.total == 7
        explicit = GLLVModels._joint_phylo_grouped_fit_layout(p, 1, :barelowrank, 2, terms;
            residual_mode = :trait)
        @test explicit == GLLVModels._joint_phylo_grouped_fit_layout(p, 1, :barelowrank, 2, terms)
    end

    @testset ":shared objective equals the :trait objective with equal residual SDs" begin
        objective_trait = GLLVModels._joint_phylo_grouped_nll(trait_fit.response, trait_fit.phy,
            trait_fit.terms, trait_fit.incidences; rank = 1, phylo_mode = :barelowrank,
            species_id = trait_fit.species_id, mean_design = trait_fit.mean_design)
        objective_shared = GLLVModels._joint_phylo_grouped_nll(shared_fit.response, shared_fit.phy,
            shared_fit.terms, shared_fit.incidences; rank = 1, phylo_mode = :barelowrank,
            residual_mode = :shared, species_id = shared_fit.species_id,
            mean_design = shared_fit.mean_design)
        @test isfinite(objective_trait(theta_trait))
        @test objective_shared(theta_shared) ≈ objective_trait(theta_trait) rtol = 1e-12
        @test shared_fit.loglik ≈ trait_fit.loglik rtol = 1e-12
        # Control: unequal residual SDs in :trait give a different value, so the
        # equality above is not vacuous.
        unequal = vcat(beta, loading, [s, s + 0.4], ordinary)
        @test !isapprox(objective_trait(unequal), objective_trait(theta_trait); rtol = 1e-6)
        # A trait-length vector is not a valid :shared coordinate vector.
        @test objective_shared(theta_trait) == GLLVModels._NLL_SENTINEL
    end

    @testset "targets, labels and intervals follow the fit's own mode" begin
        targets = GLLVModels._joint_phylo_grouped_targets(shared_fit)
        names = [t.name for t in targets]
        @test all(("residual_var_shared[$j]" in names) for j in 1:p)
        @test !any(startswith("residual_var["), names)
        shared_targets = [t for t in targets if startswith(t.name, "residual_var_shared")]
        @test all(t -> isapprox(t.value(theta_shared), 0.30; rtol = 1e-12), shared_targets)
        trait_names = [t.name for t in GLLVModels._joint_phylo_grouped_targets(trait_fit)]
        @test all(("residual_var[$j]" in trait_names) for j in 1:p)
        interval = GLLVModels.joint_phylo_grouped_intervals(shared_fit)
        @test "residual_var_shared[1]" in String[row.name for row in interval.intervals]
        postfit = GLLVModels._joint_phylo_grouped_postfit_objective(shared_fit)
        @test isfinite(postfit(shared_fit.parameters))
        @test postfit(shared_fit.parameters) ≈ -shared_fit.loglik rtol = 1e-12
    end

    @testset "refusals" begin
        @test_throws DimensionMismatch GLLVModels.fit_joint_phylo_grouped_gaussian(
            fixture.response, fixture.phy; common..., residual_mode = :shared,
            start = theta_trait, iterations = 0)
        @test_throws DimensionMismatch GLLVModels.fit_joint_phylo_grouped_gaussian(
            fixture.response, fixture.phy; common..., residual_mode = :trait,
            start = theta_shared, iterations = 0)
        @test_throws ArgumentError GLLVModels.fit_joint_phylo_grouped_gaussian(
            fixture.response, fixture.phy; common..., residual_mode = :pooled, iterations = 0)
        @test_throws ArgumentError GLLVModels._joint_phylo_grouped_fit_layout(
            p, 1, :barelowrank, 2, terms; residual_mode = :pooled)
    end

    @testset ":shared optimiser runs from the default start" begin
        fit = GLLVModels.fit_joint_phylo_grouped_gaussian(fixture.response, fixture.phy;
            common..., residual_mode = :shared, iterations = 120, g_tol = 2e-4)
        @test length(fit.parameters) == 7
        @test isfinite(fit.loglik)
        @test allequal(fit.residual_variance)
    end
end
