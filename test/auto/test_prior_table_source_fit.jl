using Test, GLLVModels

@testset "Shared prior table for Gaussian source fits" begin
    K = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0]
    sources = [SourceCovariance(K; groups=1:3, name=:oak, rank=1, unique=true)]
    traits = [:t1, :t2, :t3]
    Y = [.8 -.25 .6; -.1 -.8 .5; .35 .6 -.2]
    theta = [.1,.2,.3,.7,-.3,.4,log(.15),log(.4),log(.2)]
    rows = [
        (prior="normal(0, 1)", class="fixef", coef="t1", dpar="mu",
         scale="link", trait="", kernel=""),
        (prior="gamma(0.4, 2)", class="unique_variance", coef="", dpar="",
         scale="natural_sd", trait="t1", kernel=""),
    ]

    # Table-routed Psi priors must retain the native adapter's fixed-residual
    # requirement. This guard is independent of the optimizer start.
    @test_throws ArgumentError fit_gaussian_sources(Y; sources,
        coefficient_names=traits, trait_names=traits,
        priors=[rows[2]], iterations=0)

    # Fixed-effect-only tables do not need trait selectors, even when the
    # source has unique-variance coordinates whose diagnostics need labels.
    fixed_only_without_traits = fit_gaussian_sources(Y; sources,
        coefficient_names=traits, sigma_eps_fixed=.2, start=theta,
        priors=[rows[1]], iterations=0)
    @test fixed_only_without_traits isa GLLVModels.PenalizedGaussianSourcesFit

    fit = try
        fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta,
            priors=rows, iterations=0)
    catch err
        err
    end
    @test fit isa GLLVModels.PenalizedGaussianSourcesFit
    if fit isa GLLVModels.PenalizedGaussianSourcesFit
        expected_log_prior = GLLVModels.logpdf(GLLVModels.Normal(0, 1), theta[1]) +
            GLLVModels.logpdf(GLLVModels.Gamma(2, .2), exp(theta[7]))
        @test fit.penalised
        @test fit.logLik == loglikelihood(fit)
        @test fit.log_prior ≈ expected_log_prior atol=1e-13
        @test fit.objective ≈ fit.logLik + fit.log_prior atol=1e-13
        @test Set(r.target for r in fit.prior_targets) == Set((:fixef, :unique_variance))
        @test only(filter(r -> r.target === :unique_variance, fit.prior_targets)).measure === :natural_sd
        @test !only(filter(r -> r.target === :unique_variance, fit.prior_targets)).jacobian

        optimized = fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta,
            priors=rows, iterations=8)
        @test optimized.iterations > 0
        @test isfinite(optimized.logLik) && isfinite(optimized.log_prior)
        @test optimized.objective ≈ optimized.logLik + optimized.log_prior atol=1e-12
        data_nll(t) = GLLVModels._gaussian_sources_nll(Y, optimized.sources, t;
            sigma_eps_fixed=.2)
        combined_nll(t) = data_nll(t) + GLLVModels._source_prior_table_penalty(
            t, optimized.prior_targets)
        @test isapprox(optimized.hessian,
            GLLVModels.ForwardDiff.hessian(combined_nll, optimized.parameters);
            atol=1e-11, rtol=1e-11)
        @test isapprox(optimized.hessian_unpenalised,
            GLLVModels.ForwardDiff.hessian(data_nll, optimized.parameters);
            atol=1e-11, rtol=1e-11)

        all_fixed = [merge(rows[1], (coef="all",)), rows[2]]
        all_fit = fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta,
            priors=all_fixed, iterations=0)
        @test count(r -> r.target === :fixef, all_fit.prior_targets) == 3

        student_mixed = [rows[2], (prior="student_t(0, 1.5, 4)",
            class="fixef", coef="t2", dpar="mu", scale="link")]
        student_fit = fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta,
            priors=student_mixed, iterations=0)
        expected_student = GLLVModels.logpdf(GLLVModels.Distributions.LocationScale(
            0, 1.5, GLLVModels.Distributions.TDist(4)), theta[2]) +
            GLLVModels.logpdf(GLLVModels.Gamma(2, .2), exp(theta[7]))
        @test student_fit.log_prior ≈ expected_student atol=1e-13

        ordinary = fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta, iterations=0)
        empty = fit_gaussian_sources(Y; sources, coefficient_names=traits,
            sigma_eps_fixed=.2, trait_names=traits, start=theta,
            priors=NamedTuple[], iterations=0)
        @test typeof(empty) === typeof(ordinary)
        @test propertynames(empty) === propertynames(ordinary)
        @test empty.parameters == ordinary.parameters
        @test empty.loglik == ordinary.loglik

        duplicate = [rows; rows[2]]
        unknown_kernel = [merge(rows[2], (kernel="missing",))]
        unknown_trait = [merge(rows[2], (trait="missing",))]
        wrong_scale = [merge(rows[2], (scale="variance",))]
        for bad in (duplicate, unknown_kernel, unknown_trait, wrong_scale)
            @test_throws ArgumentError fit_gaussian_sources(Y; sources,
                coefficient_names=traits, sigma_eps_fixed=.2, trait_names=traits,
                start=theta, priors=bad, iterations=0)
        end
        for bad_traits in ([:t1], [:t1, :t2, :t3, :t4], [1, 2, 3])
            @test_throws ArgumentError fit_gaussian_sources(Y; sources,
                coefficient_names=traits, sigma_eps_fixed=.2, trait_names=bad_traits,
                start=theta, priors=[rows[2]], iterations=0)
        end
        @test_throws ArgumentError GLLVModels._resolve_gaussian_source_prior_table(
            [merge(rows[1], (coef="all",)), rows[2]], sources, String[], traits, 3)
        Kpine = [1.0 0.0 .2; 0.0 1.0 .1; .2 .1 1.0]
        two_sources = [sources[1], SourceCovariance(Kpine; groups=1:3,
            name=:pine, rank=1, unique=true)]
        @test_throws ArgumentError fit_gaussian_sources(Y; sources=two_sources,
            coefficient_names=traits, sigma_eps_fixed=.2, trait_names=traits,
            start=[theta[1:9]; theta[4:6]; theta[7:9]], priors=[rows[2]], iterations=0)
    end
end
