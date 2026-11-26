using Test
using GLLVModels
using Random, Distributions

# A response matrix that throws on read once armed, so an error is raised
# inside a family CI adapter's objective and refit closures (#774).
struct _ArmedMatrix <: AbstractMatrix{Int}
    A::Matrix{Int}
    err::Base.RefValue{Any}
end
Base.size(M::_ArmedMatrix) = size(M.A)
Base.getindex(M::_ArmedMatrix, i::Int, j::Int) =
    M.err[] === nothing ? M.A[i, j] : throw(M.err[])

# Throws an ordinary error on one chosen read (the `at`-th), to fail a single objective
# evaluation inside `confint` (#774 first_error).
struct _ThrowOnReadMatrix <: AbstractMatrix{Int}
    A::Matrix{Int}
    count::Base.RefValue{Int}
    at::Base.RefValue{Int}
end
Base.size(M::_ThrowOnReadMatrix) = size(M.A)
function Base.getindex(M::_ThrowOnReadMatrix, i::Int, j::Int)
    M.count[] += 1
    M.count[] == M.at[] && throw(ErrorException("armed read failure"))
    return M.A[i, j]
end

# Issue #774: a bare `catch` in a fitter objective swallowed InterruptException
# (Ctrl-C), so an interrupted fit carried on with the barrier/1e12 sentinel.
# Genuine objective failures must still return the sentinel.
@testset "fitter objectives rethrow InterruptException (#774)" begin
    @testset "mixed: _penalized_negloglik_fg! (behavioural)" begin
        θ = [0.5, -0.5]
        @test_throws InterruptException GLLVModels._penalized_negloglik_fg!(
            0.0, nothing, _ -> throw(InterruptException()), θ)
        # An ordinary error still falls back to the barrier penalty.
        F = GLLVModels._penalized_negloglik_fg!(
            0.0, nothing, _ -> error("objective failed"), θ)
        @test F >= 1e12
    end

    @testset "ordinal: objective wrappers (source-level)" begin
        # The four wrappers are closures inside the fitters and cannot be
        # reached with an injected failure, so pin the source: no bare `catch`,
        # and each wrapper rethrows InterruptException.
        src = read(joinpath(pkgdir(GLLVModels), "src", "families", "ordinal.jl"), String)
        @test !occursin(r"^\s*catch\s*$"m, src)
        @test count("isa InterruptException && rethrow()", src) >= 4
    end

    @testset "family CI bootstrap leftover (#774)" begin
        ad_unsup = GLLVModels._FamilyCI(
            [0.0], _ -> 0.0, ["beta[1]"], [:linear],
            _ -> error("bootstrap not supported for quadratic CIs"),
            _ -> nothing)
        @test_throws ArgumentError GLLVModels._family_bootstrap(
            ad_unsup, [1], 0.95, 12, 1, false)

        ad_int = GLLVModels._FamilyCI(
            [0.0], _ -> 0.0, ["beta[1]"], [:linear],
            _ -> throw(InterruptException()),
            _ -> nothing)
        @test_throws InterruptException GLLVModels._family_bootstrap(
            ad_int, [1], 0.95, 3, 1, false)

        ad_prof = GLLVModels._FamilyCI(
            [0.0, 0.0], _ -> throw(InterruptException()),
            ["beta[1]", "beta[2]"], [:linear, :linear],
            _ -> [0.0], _ -> nothing)
        @test_throws InterruptException GLLVModels._family_profile_refit(
            ad_prof, 1, 0.0, [0.0])

        ad_ok_err = GLLVModels._FamilyCI(
            [0.0, 0.0], _ -> error("objective failed"),
            ["beta[1]", "beta[2]"], [:linear, :linear],
            _ -> [0.0], _ -> nothing)
        ll, ok, _ = GLLVModels._family_profile_refit(
            ad_ok_err, 1, 0.0, [0.0])
        @test isnan(ll)
        @test ok === false
    end

    @testset "family CI adapters: nll and refit (#774)" begin
        rng = Random.MersenneTwister(774)
        p, n = 4, 40
        Λ = 0.5 .* randn(rng, p, 1); z = randn(rng, 1, n)
        Y = [rand(rng, Poisson(exp(0.5 + (Λ * z)[i, j]))) for i in 1:p, j in 1:n]
        fit = fit_poisson_gllvm(Y; K = 1)
        Ya = _ArmedMatrix(Y, Ref{Any}(nothing))
        ad = GLLVModels._family_ci(fit, Ya)
        @test isfinite(ad.nll(ad.θ))

        Ya.err[] = InterruptException()
        @test_throws InterruptException ad.nll(ad.θ)
        @test_throws InterruptException ad.refit(Ya)

        # An ordinary error still becomes the sentinel / failed refit.
        Ya.err[] = ErrorException("objective failed")
        @test ad.nll(ad.θ) == 1e12
        @test ad.refit(Ya) === nothing

        # Source pin for the remaining adapters: no bare `catch` is left.
        src = read(joinpath(pkgdir(GLLVModels), "src", "confint_family.jl"), String)
        @test count(r"^\s*catch\s*$"m, src) == 0
        @test count(r"\bcatch\s*;", src) == 0
    end

    @testset "phylo poisson xlv constrained refit (#747)" begin
        nll = _ -> throw(InterruptException())
        # p=6, K=1, q_lv=1: packed length is 6 + 1 + 6 + 1 = 14.
        theta = zeros(14)
        @test_throws InterruptException GLLVModels._phylo_poisson_xlv_constrained_refit(
            nll, theta, 6, 1, 1, 1, 0.0;
            profile_iterations = 2, penalty_weights = (1.0,))
        # An ordinary error still becomes the 1e12 barrier, then NaN nll.
        out = GLLVModels._phylo_poisson_xlv_constrained_refit(
            _ -> error("objective failed"), theta, 6, 1, 1, 1, 0.0;
            profile_iterations = 2, penalty_weights = (1.0,))
        @test isnan(out.nll)
    end
    @testset "derived / derived-Wald / profile confint: no bare catch (source pin)" begin
        for f in ("confint_derived.jl", "confint_derived_wald.jl", "confint_profile.jl")
            src = read(joinpath(pkgdir(GLLVModels), "src", f), String)
            @test count(r"^\s*catch\s*$"m, src) == 0
            @test count(r"\bcatch\s*;", src) == 0
        end
    end
end

# Option 2 of #774 (maintainer decision 2026-10-31): the first swallowed non-interrupt
# error is reported as `first_error` on the result, with one warning per call.
@testset "confint reports first_error and warns once (#774)" begin
    @testset "recorder" begin
        res, log = GLLVModels._ci_collect() do
            for msg in ("first", "second")
                try
                    error(msg)
                catch e
                    GLLVModels._ci_swallowed(e)
                end
            end
            42
        end
        @test res == 42
        @test log.n == 2
        @test occursin("first", log.first) && !occursin("second", log.first)
        # No recorder installed: swallowing is a silent no-op, an interrupt still escapes.
        @test GLLVModels._CI_ERRLOG[] === nothing
        @test (try; error("x"); catch e; GLLVModels._ci_swallowed(e); end) === nothing
        @test_throws InterruptException (try
            throw(InterruptException())
        catch e
            GLLVModels._ci_swallowed(e)
        end)
        # The log is restored after an exception, and nested counts reach the outer log.
        @test_throws ErrorException GLLVModels._ci_collect(() -> error("boom"))
        @test GLLVModels._CI_ERRLOG[] === nothing
        _, outer = GLLVModels._ci_collect() do
            GLLVModels._ci_collect() do
                try; error("inner"); catch e; GLLVModels._ci_swallowed(e); end
            end
        end
        @test outer.n == 1 && occursin("inner", outer.first)
    end

    @testset "family confint: clean fit has first_error = nothing and no warning" begin
        rng = Random.MersenneTwister(1036)
        p, n = 4, 60
        Λ = 0.5 .* randn(rng, p, 1); z = randn(rng, 1, n)
        Y = [rand(rng, Poisson(exp(0.5 + (Λ * z)[i, j]))) for i in 1:p, j in 1:n]
        fit = fit_poisson_gllvm(Y; K = 1)
        ci = @test_logs confint(fit, Y; method = :wald, parm = "beta")
        @test ci.first_error === nothing
        @test haskey(ci, :term) && haskey(ci, :method)
    end

    @testset "family confint: bootstrap with failing simulate" begin
        # Every replicate's simulate throws an ordinary error (not a "not supported"
        # message): the call finishes with n_converged = 0, records the first error,
        # and `_ci_finish` warns once.
        ad = GLLVModels._FamilyCI(
            [0.0], _ -> 0.0, ["beta[1]"], [:linear],
            _ -> error("refit blew up"), _ -> nothing)
        res, log = GLLVModels._ci_collect() do
            GLLVModels._family_bootstrap(ad, [1], 0.95, 5, 1, false)
        end
        @test res.n_converged == 0
        @test log.n == 6   # the pre-check plus five replicates
        @test occursin("refit blew up", log.first)
        out = @test_logs (:warn, r"6 evaluation\(s\) threw.*refit blew up") GLLVModels._ci_finish(res, log, "confint")
        @test out.first_error == log.first
        @test out.n_converged == 0
    end

    @testset "end to end through confint: one failed read" begin
        rng = Random.MersenneTwister(1037)
        p, n = 4, 40
        Λ = 0.5 .* randn(rng, p, 1); z = randn(rng, 1, n)
        Y = [rand(rng, Poisson(exp(0.5 + (Λ * z)[i, j]))) for i in 1:p, j in 1:n]
        fit = fit_poisson_gllvm(Y; K = 1)
        # Calibrate: count the reads a Wald call makes, then fail one read late in the
        # call (inside a Hessian objective evaluation). The same path is deterministic.
        Yc = _ThrowOnReadMatrix(Y, Ref(0), Ref(0))
        confint(fit, Yc; method = :wald)
        total = Yc.count[]
        Yc = _ThrowOnReadMatrix(Y, Ref(0), Ref(round(Int, 0.9 * total)))
        ci = @test_logs (:warn, r"evaluation\(s\) threw.*armed read failure") confint(
            fit, Yc; method = :wald)
        @test ci.first_error isa String && occursin("armed read failure", ci.first_error)
    end
end
