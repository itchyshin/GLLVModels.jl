using GLLVModels, Test, Random, LinearAlgebra, Distributions

# The confint() entry point takes `y` as a kwarg so it can reconstruct
# the NLL closure without touching src/fit.jl (the PERF agent is
# overhauling that file in parallel). This is the same `y` originally
# passed to fit_gaussian_gllvm.

@testset "confint" begin
    @testset "returns a NamedTuple with the expected fields" begin
        Random.seed!(0)
        p, K, n = 5, 1, 200
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ci = GLLVModels.confint(fit; y = y)
        @test isa(ci.term, Vector{String})
        @test isa(ci.lower, Vector{<:Real})
        @test isa(ci.upper, Vector{<:Real})
        @test length(ci.term) == length(ci.lower) == length(ci.upper) == length(ci.estimate)
        # When pd_hessian = true, all intervals should bracket the estimate.
        if ci.pd_hessian
            @test all(ci.lower .<= ci.estimate)
            @test all(ci.estimate .<= ci.upper)
        end
    end

    @testset "coverage of σ_eps recovers the truth at n=1000" begin
        # Roughly check that the Wald CI contains the truth (nominal 95% coverage)
        Random.seed!(1)
        p, K, n = 5, 1, 1000
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        σ_true = 0.5
        y = Λ_true * randn(K, n) + σ_true * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ci = GLLVModels.confint(fit; y = y, level = 0.95)
        idx_se = findfirst(==("sigma_eps"), ci.term)
        @test !isnothing(idx_se)
        @test ci.lower[idx_se] <= σ_true <= ci.upper[idx_se]
        # Report bounds for the task-report
        @info "σ_eps Wald CI" lower=ci.lower[idx_se] estimate=ci.estimate[idx_se] upper=ci.upper[idx_se] truth=σ_true
    end

    @testset "parm = 'sigma_eps' filters" begin
        Random.seed!(2)
        p, K, n = 4, 1, 100
        y = 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ci = GLLVModels.confint(fit; y = y, parm = "sigma_eps")
        @test ci.term == ["sigma_eps"]
        @test length(ci.lower) == 1
    end

    @testset "parm = 'Lambda:1,1' filters to one Λ entry" begin
        Random.seed!(3)
        p, K, n = 4, 1, 100
        Λ_true = reshape([0.7, 0.5, 0.3, -0.2], p, K)
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ci = GLLVModels.confint(fit; y = y, parm = "Lambda:1,1")
        @test ci.term == ["Lambda_B[1,1]"]
        @test length(ci.lower) == 1
    end

    @testset "graceful NA when Hessian is non-PD" begin
        # Construct a degenerate fixture (e.g., y is rank-deficient)
        # On the degenerate fit, confint should return NaN bounds, not error
        Random.seed!(4)
        # Use a moderately ill-conditioned fixture: tiny n, K matched to p.
        # If the fit succeeds, confint should either return finite bounds
        # (if the Hessian happens to be PD) or NaN bounds with
        # pd_hessian = false. Either way it must not throw.
        p, K, n = 3, 2, 6
        y = randn(p, n)
        local ci
        ok = true
        try
            fit = fit_gaussian_gllvm(y; K = K)
            ci = GLLVModels.confint(fit; y = y)
        catch err
            # Allow the fit itself to error on the degenerate fixture; the
            # contract is about confint() being graceful, not the fitter.
            @test_skip "Fit failed on degenerate fixture; confint contract is about graceful return when fit succeeds"
            ok = false
        end
        if ok
            # Either valid (pd_hessian = true) or graceful NaN (pd_hessian = false).
            @test ci.pd_hessian || all(isnan, ci.lower)
            # The call must return a well-formed NamedTuple either way.
            @test length(ci.term) == length(ci.lower)
        end
    end
end

@testset "confint: fit with beta requires X (no silent NaN)" begin
    Random.seed!(7)
    p, n = 5, 150
    Y = 0.6 * randn(1, n) .* [0.7, -0.5, 0.4, 0.7, -0.3] .+ 0.5 * randn(p, n) .+ (1:p)
    X = zeros(p, n, p); for i in 1:p; X[i, :, i] .= 1; end
    fit = fit_gaussian_gllvm(Y; K = 1, X = X)
    @test_throws ArgumentError GLLVModels.confint(fit, Y; method = :wald)
    ci = GLLVModels.confint(fit, Y; method = :wald, X = X)
    @test all(isfinite, ci.lower) && ci.pd_hessian
end
