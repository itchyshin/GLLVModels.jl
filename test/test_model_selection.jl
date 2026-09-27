using GLLVModels, Test, Random, Distributions, Statistics

@testset "select_lv — latent-dimension selection" begin
    @testset "Poisson sweep K = 1:3" begin
        Random.seed!(2024)
        p, K, n = 5, 2, 120
        β_true = log.([4.0, 6.0, 3.0, 5.0, 4.0])
        Λ_true = 0.5 .* randn(p, K)
        η = β_true .+ Λ_true * randn(K, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]

        sel = select_lv(Y; family = Poisson(), Kmax = 3)

        @test sel isa LVSelection
        @test length(sel.K) == 3
        @test sel.K == [1, 2, 3]

        # More latent capacity ⇒ ≥ log-likelihood (allow tiny optimizer slack).
        for k in 1:(length(sel.loglik) - 1)
            @test sel.loglik[k + 1] >= sel.loglik[k] - 1e-3
        end

        @test sel.best_k in 1:3
        @test all(isfinite, sel.aic)
        @test all(isfinite, sel.bic)
        @test all(isfinite, sel.loglik)

        # `best` is a genuine fitted model with a finite log-likelihood.
        @test isfinite(GLLVModels._loglik(sel.best))
        @test sel.best isa PoissonFit

        # AIC default agrees with the BIC selection consistency: best row is the
        # argmin of the chosen criterion (default :bic).
        @test sel.best_k == sel.K[argmin(sel.bic)]

        # bic uses nobs(fit, Y), R's p·n observed-cell count, not the site
        # count n (docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md
        # #1); complete-data Y here has no missing cells, so nobs == p * n.
        for i in eachindex(sel.K)
            @test sel.bic[i] ≈ sel.nparams[i] * log(p * n) - 2 * sel.loglik[i]
        end
    end

    @testset ":aic criterion selects argmin AIC" begin
        Random.seed!(7)
        p, n = 5, 120
        Y = [rand(Poisson(exp(1.4 + 0.4 * randn()))) for t in 1:p, s in 1:n]
        sel = select_lv(Y; family = Poisson(), Kmax = 3, criterion = :aic)
        @test sel.best_k == sel.K[argmin(sel.aic)]
    end
end

# --- Warm-start safeguard (lane auto-d-20260926) ---------------------------------
# A K+1 model nests the K model, so a correct maximum can never have a lower
# log-likelihood. Measured on origin/main 2847b5dbf (NB2, n=300, p=20, K_true=3):
# K=3 and K=4 fits reported converged=true with logLik below K=2. These tests drive
# select_lv's guard with a stand-in fitter so they are exact and fast.
struct _FakeLVFit
    ll::Float64
    np::Int
    converged::Bool
    β::Vector{Float64}
    Λ::Matrix{Float64}
end
GLLVModels._loglik(f::_FakeLVFit) = f.ll
GLLVModels._nparams(f::_FakeLVFit) = f.np
GLLVModels.StatsAPI.aic(f::_FakeLVFit) = 2f.np - 2f.ll
GLLVModels.StatsAPI.bic(f::_FakeLVFit, Y::AbstractMatrix) = f.np * log(length(Y)) - 2f.ll

# Loglik per K on a default start, and on a warm start (used only if the fitter
# accepts Λ_init). K=3 is a bad optimum from the default start only.
const _LL_DEFAULT = Dict(1 => -500.0, 2 => -400.0, 3 => -450.0, 4 => -395.0)
const _LL_WARM    = Dict(3 => -390.0, 4 => -389.0)

function _fake_fitter(calls; warm_ok::Bool, fail_at = nothing, unconv_at = nothing)
    return function (Y; family, K, kwargs...)
        push!(calls, (K = K, warm = haskey(kwargs, :Λ_init), kwargs = kwargs))
        K == fail_at && error("singular at K=$K")
        p = size(Y, 1)
        if haskey(kwargs, :Λ_init)
            warm_ok || throw(ArgumentError("unsupported keyword Λ_init"))
            ll = get(_LL_WARM, K, _LL_DEFAULT[K])
        else
            ll = _LL_DEFAULT[K]
        end
        return _FakeLVFit(ll, 10K, K != unconv_at, zeros(p), fill(0.5, p, K))
    end
end

@testset "select_lv — warm-start safeguard" begin
    Y = zeros(6, 40)

    @testset "healthy sweep: no retries, results unchanged" begin
        calls = Any[]
        f = function (Y; family, K, kwargs...)
            push!(calls, K)
            return _FakeLVFit(-500.0 + 60K, 10K, true, zeros(6), fill(0.5, 6, K))
        end
        sel = select_lv(Y; Kmax = 4, _fitter = f)
        @test calls == [1, 2, 3, 4]
        @test sel.K == [1, 2, 3, 4]
        @test all(a -> a.status === :ok, sel.attempts)
    end

    @testset "non-monotone K is retried from the K-1 solution and accepted" begin
        calls = Any[]
        sel = select_lv(Y; Kmax = 4, _fitter = _fake_fitter(calls; warm_ok = true))
        warm = filter(c -> c.warm, calls)
        # K=3 is retried; then default K=4 (−395) sits below the warm K=3 (−390),
        # so K=4 is retried too and accepted at −389.
        @test [c.K for c in warm] == [3, 4]
        @test only(filter(a -> a.K == 4, sel.attempts)).status === :warm_start
        Λ0 = warm[1].kwargs[:Λ_init]
        @test size(Λ0) == (6, 3)
        @test Λ0[:, 1:2] == fill(0.5, 6, 2)          # previous solution kept
        @test all(iszero, Λ0[1:2, 3])                  # lower-triangular new column
        @test Λ0[3, 3] > 0
        a3 = only(filter(a -> a.K == 3, sel.attempts))
        @test a3.status === :warm_start
        @test a3.loglik == -390.0
        @test 3 in sel.K
    end

    @testset "non-monotone K without warm-start support is excluded, not chosen" begin
        calls = Any[]
        sel = select_lv(Y; Kmax = 3, criterion = :aic,
                        _fitter = _fake_fitter(calls; warm_ok = false))
        a3 = only(filter(a -> a.K == 3, sel.attempts))
        @test a3.status === :nonmonotone
        @test !(3 in sel.K)
        @test sel.best_k == 2
    end

    @testset "a throwing fit is recorded with its reason, not silently dropped" begin
        sel = select_lv(Y; Kmax = 3, _fitter = _fake_fitter(Any[]; warm_ok = true, fail_at = 2))
        a2 = only(filter(a -> a.K == 2, sel.attempts))
        @test a2.status === :failed
        @test occursin("singular", a2.message)
        @test !(2 in sel.K)
    end

    @testset "an unconverged fit is excluded" begin
        sel = select_lv(Y; Kmax = 3, _fitter = _fake_fitter(Any[]; warm_ok = true, unconv_at = 2))
        a2 = only(filter(a -> a.K == 2, sel.attempts))
        @test a2.status === :unconverged
        @test !(2 in sel.K)
    end

    @testset "warm_start = false disables the retry but keeps the guard" begin
        calls = Any[]
        sel = select_lv(Y; Kmax = 3, warm_start = false,
                        _fitter = _fake_fitter(calls; warm_ok = true))
        @test !any(c -> c.warm, calls)
        @test only(filter(a -> a.K == 3, sel.attempts)).status === :nonmonotone
    end

    @testset "InterruptException is not swallowed" begin
        f = (Y; family, K, kwargs...) -> throw(InterruptException())
        @test_throws InterruptException select_lv(Y; Kmax = 2, _fitter = f)
    end
end

# --- Runaway detector (lane auto-d-20260926) -------------------------------------
# Latent variables are standardised (u ~ N(0, I)), so a trait's loading row norm is
# its latent SD on the link scale; ~4 is as strong as real gradients get and 10 is
# saturated (vault note "Two runaway modes in GLLVM loadings"). Mode B (common
# inflation) needs a SCALE check; Mode A (one binary trait separates) needs a RATIO
# check, which is blind to Mode B by construction.
@testset "select_lv — runaway detector" begin
    Y = zeros(6, 40)
    # K = 3 is a common-inflation runaway (all rows × 40) with the best logLik.
    inflate = function (Y; family, K, kwargs...)
        haskey(kwargs, :Λ_init) && return _FakeLVFit(-380.0, 10K, true, zeros(6), fill(0.5, 6, K))
        Λ = K == 3 ? fill(20.0, 6, K) : fill(0.5, 6, K)
        return _FakeLVFit(Dict(1 => -500.0, 2 => -400.0, 3 => -300.0)[K], 10K, true, zeros(6), Λ)
    end

    @testset "Mode B: a common-inflation K is retried, and excluded if still runaway" begin
        sel = select_lv(Y; family = Poisson(), Kmax = 3, criterion = :aic, warm_start = false,
                        _fitter = inflate)
        a3 = only(filter(a -> a.K == 3, sel.attempts))
        @test a3.status === :runaway
        @test occursin("latent SD", a3.message)
        @test !(3 in sel.K)
        @test sel.best_k == 2
    end

    @testset "Mode B: the warm-start refit replaces a runaway when it is healthy" begin
        sel = select_lv(Y; family = Poisson(), Kmax = 3, _fitter = inflate)
        a3 = only(filter(a -> a.K == 3, sel.attempts))
        @test a3.status === :warm_start
        @test a3.loglik == -380.0
    end

    @testset "Mode A: one binary trait separating is caught by the ratio check" begin
        sep = function (Y; family, K, kwargs...)
            Λ = fill(0.5, 6, K); K == 2 && (Λ[4, 1] = 15.0)   # one trait at 30× the median
            K == 3 && (Λ[4, 1] = 48.0)
            return _FakeLVFit(-500.0 + 60K, 10K, true, zeros(6), Λ)
        end
        sel = select_lv(Y; family = Binomial(), Kmax = 3, warm_start = false,
                        max_latent_sd = Inf, _fitter = sep)
        @test only(filter(a -> a.K == 2, sel.attempts)).status === :runaway   # ratio 30 ≥ 25
        @test occursin("ratio", only(filter(a -> a.K == 2, sel.attempts)).message)
        @test sel.K == [1]
    end

    @testset "the ratio check is binomial-only; the scale check can be disabled" begin
        one_big = (Y; family, K, kwargs...) ->
            _FakeLVFit(-500.0 + 60K, 10K, true, zeros(6), (Λ = fill(0.5, 6, K); Λ[4, 1] = 9.0; Λ))
        sel = select_lv(Y; family = Poisson(), Kmax = 2, _fitter = one_big)
        @test sel.K == [1, 2]                                                  # ratio 18, rows < 10
        sel2 = select_lv(Y; family = Poisson(), Kmax = 3, max_latent_sd = Inf,
                         warm_start = false, _fitter = inflate)
        @test 3 in sel2.K
    end
end

# --- Omitting K estimates it (lane auto-d-20260926; public API, awaiting sign-off) --
@testset "fit_gllvm without K selects K via select_lv" begin
    Random.seed!(11)
    p, n = 6, 150
    Λ_true = 0.9 .* randn(p, 1)
    η = log(3.0) .+ Λ_true * randn(1, n)
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]

    fit = @test_logs (:info, r"chose K") match_mode = :any fit_gllvm(Y; family = Poisson())
    sel = select_lv(Y; family = Poisson(), Kmax = min(5, p - 1))
    @test fit isa PoissonFit
    @test size(fit.Λ, 2) == sel.best_k
    @test GLLVModels._loglik(fit) ≈ GLLVModels._loglik(sel.best)

    one = fit_gllvm(Y; family = Poisson(), Kmax = 1)
    @test size(one.Λ, 2) == 1
    @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = 2, Kmax = 3)

    # Explicit K is untouched by the auto path.
    @test GLLVModels._loglik(fit_gllvm(Y; family = Poisson(), K = 1)) ≈
          GLLVModels._loglik(fit_gllvm(Y; family = Poisson(), num_lv = 1))
end
