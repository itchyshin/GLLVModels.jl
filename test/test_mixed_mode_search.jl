using GLLVModels, Test, SHA, TOML, Random, LinearAlgebra, Distributions, ForwardDiff

# `_mixed_laplace_mode` (the per-site inner mode search behind `fit_mixed_gllvm`) could
# return a finite site log-likelihood at a z whose per-site log-posterior gradient was
# not near zero (#503, the same defect class as #479/#507/#509/#500): the per-trait
# Fisher-scoring loop took undamped steps and returned whatever z it held when `maxiter`
# was reached, converged or not. A re-measure on origin/main across three different
# family mixes (this file's own gradient-of-the-score check, `Λ's − z`) found 120/1200
# finite-non-mode sites. `_mixed_laplace_mode` now halves any step that lowers the
# per-site log-posterior and requires BOTH the full proposed step and a scale-aware
# gradient check (the Newton decrement `g'Δ`, not an absolute bound on `g`) to be small
# before declaring convergence: an absolute bound is unusable once `Λ'WΛ + I` is
# ill-conditioned (e.g. a Normal trait whose fitted σ is driven near zero, a Heywood
# case the outer fitter can reach), where a tiny step solves the Newton equation while
# the raw gradient in the stiff direction is still large in floating point. The small
# step shortcut (skip the line search when `‖Δ‖` is already tiny) is also no longer
# unconditional: once a full-size line-search step has been rejected once in a call,
# every later step goes through the line search too, since the undamped fixed-point map
# is not always a contraction near the mode (e.g. a low-shape Gamma trait far out in its
# tail, where the Fisher weight underestimates the observed curvature) and would
# otherwise oscillate to `-Inf` even though a genuine mode exists. `_mixed_loglik_site`
# retries a non-converging site with a 20x iteration budget, then returns -Inf if it
# still cannot certify a stationary point, so the fitter's own 1e12 failure sentinel
# fires instead of a silently wrong value.
#
# `getLV`/`predict` call `_mixed_laplace_mode` directly (not `_mixed_loglik_site`) and
# discard the convergence flag, exactly as the old code discarded nothing (there was no
# flag), so they keep a NO-SENTINEL contract: whatever z the damped search returns,
# converged or not, is what they use, never -Inf.

const _MIXED503_FIXTURE = joinpath(@__DIR__, "fixtures", "mixed_mode_search_503.toml")

function _mixed503_data()
    d = TOML.parsefile(_MIXED503_FIXTURE)
    y = Float64.(d["y"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == d["y_sha256"]
    p, K = d["p"], d["K"]
    β = Float64.(d["beta"])
    Λ = reshape(Float64.(d["Lambda_column_major"]), p, K)
    G = GLLVModels
    fams = [Poisson(), Binomial(), Gamma(d["gamma_shape"], 1.0)]
    links = [G.LogLink(), G.LogitLink(), G.LogLink()]
    return y, Λ, β, fams, links, Float64.(d["z_ref"]), d["v_ref"], d["v_k_on_main"]
end

# The pre-#503 per-site kernel, kept verbatim apart from also reporting whether its
# undamped Fisher loop converged. Used only where it DID converge, to check that the
# fix leaves those values alone.
function _mixed503_old_mode(families, links, y, n, Λ, β; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λ)
    T = Float64
    z = zeros(T, K)
    s = Vector{T}(undef, p)
    W = Vector{T}(undef, p)
    conv = false
    for _ in 1:maxiter
        η = Λ * z
        @inbounds for t in 1:p
            if ismissing(y[t])
                s[t] = zero(T); W[t] = zero(T)
            else
                ηt = G._clamp_eta(β[t] + η[t])
                μt = G._clamp_mu(families[t], G.linkinv(links[t], ηt))
                met = G.mu_eta(links[t], ηt)
                s[t] = G._glm_score(families[t], μt, n[t], met, y[t])
                W[t] = G._glm_weight(families[t], μt, n[t], met)
            end
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = G._safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    return z, conv
end

function _mixed503_old_site(families, links, y, n, Λ, β)
    G = GLLVModels
    p = size(Λ, 1)
    z, conv = _mixed503_old_mode(families, links, y, n, Λ, β)
    η = Λ * z
    T = Float64
    W = Vector{T}(undef, p)
    ℓ = zero(T)
    @inbounds for t in 1:p
        if ismissing(y[t])
            W[t] = zero(T)
        else
            ηt = G._clamp_eta(β[t] + η[t])
            μt = G._clamp_mu(families[t], G.linkinv(links[t], ηt))
            met = G.mu_eta(links[t], ηt)
            W[t] = if G._default_hessian(families[t], links[t]) === :fisher ||
                      G._glm_weight_matches_observed(families[t], links[t])
                G._glm_weight(families[t], μt, n[t], met)
            else
                G._glm_obs_weight(families[t], μt, n[t], met, y[t], links[t], ηt)
            end
            ℓ += G._glm_logpdf(families[t], μt, n[t], y[t])
        end
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Gradient of the per-site log-posterior at z (own GLLVModels score dispatch, matching
# the class-audit's own probe convention): the same quantity the mode search's Newton
# step is asked to zero.
function _mixed503_gradnorm(families, links, y, n, Λ, β, z)
    G = GLLVModels
    p = size(Λ, 1)
    η = Λ * z
    s = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        if ismissing(y[t])
            s[t] = 0.0
        else
            ηt = G._clamp_eta(β[t] + η[t])
            μt = G._clamp_mu(families[t], G.linkinv(links[t], ηt))
            met = G.mu_eta(links[t], ηt)
            s[t] = G._glm_score(families[t], μt, n[t], met, y[t])
        end
    end
    return maximum(abs, Λ' * s .- z)
end

# Fully independent per-site log-posterior: each trait's OWN `Distributions.logpdf`
# and a hand-written link, sharing no code with `_mixed_laplace_mode`/`_glm_*`. Used
# only to certify stationarity via `ForwardDiff`, never for the value regression above.
_ind_linkinv(::Val{:log}, η) = exp(clamp(η, -30.0, 30.0))
_ind_linkinv(::Val{:logit}, η) = 1 / (1 + exp(-clamp(η, -30.0, 30.0)))
_ind_linkinv(::Val{:identity}, η) = η

function _ind_logpost(families_tag, y, n, Λ, β, z)
    η = β .+ Λ * z
    ll = 0.0
    @inbounds for t in eachindex(families_tag)
        tag, dist = families_tag[t]
        μ = _ind_linkinv(Val(tag), η[t])
        ll += _ind_family_logpdf(dist, μ, n[t], y[t])
    end
    return ll - 0.5 * dot(z, z)
end

_ind_family_logpdf(::Poisson, μ, n, y) = logpdf(Poisson(max(μ, 1e-300)), Int(round(y)))
_ind_family_logpdf(::Binomial, μ, n, y) =
    logpdf(Binomial(n, clamp(μ, 1e-12, 1 - 1e-12)), Int(round(y)))
_ind_family_logpdf(f::Gamma, μ, n, y) =
    logpdf(Gamma(shape(f), max(μ, 1e-300) / shape(f)), max(y, 1e-300))
_ind_family_logpdf(f::Beta, μ, n, y) =
    logpdf(Beta(max(μ, 1e-8) * f.α, max(1 - μ, 1e-8) * f.α), clamp(y, 1e-8, 1 - 1e-8))
_ind_family_logpdf(f::NegativeBinomial, μ, n, y) =
    logpdf(NegativeBinomial(f.r, f.r / (f.r + max(μ, 1e-12))), Int(round(y)))
_ind_family_logpdf(f::Normal, μ, n, y) = logpdf(Normal(μ, f.σ), y)

@testset "Mixed-family mode search: damped, and fails loudly (#503)" begin
    y, Λ0, β0, fams0, links0, z_ref, v_ref, _v_old_ref = _mixed503_data()
    p, K = size(Λ0)
    n1 = ones(Int, p)

    @testset "the audit's diverging site now gets its true Laplace value" begin
        # Before: undamped Fisher scoring returned a finite garbage value at a
        # non-mode z (`v_k_on_main` in the fixture, about -1.07e13). Asserted as a
        # relation, not pinned to that one BLAS/platform number (review-514 S6): 100
        # divergent undamped iterations are not guaranteed to reproduce to 1.0 across
        # Julia versions or platforms, only to stay non-converged and enormous.
        z_old, conv_old = _mixed503_old_mode(fams0, links0, y, n1, Λ0, β0)
        @test !conv_old
        v_old, _ = _mixed503_old_site(fams0, links0, y, n1, Λ0, β0)
        @test isfinite(v_old)
        @test abs(v_old) > 1e6

        v = GLLVModels._mixed_loglik_site(fams0, links0, y, n1, Λ0, β0)
        @test isfinite(v)
        @test abs(v - v_ref) < 1e-8
    end

    @testset "the returned mode is a stationary point, independently certified" begin
        z, ok = GLLVModels._mixed_laplace_mode(fams0, links0, y, n1, Λ0, β0)
        @test ok
        @test isapprox(z, z_ref; atol = 1e-6)
        families_tag = [(:log, fams0[1]), (:logit, fams0[2]), (:log, fams0[3])]
        q = zz -> _ind_logpost(families_tag, y, n1, Λ0, β0, zz)
        g = ForwardDiff.gradient(q, z)
        H = ForwardDiff.hessian(q, z)
        @test maximum(abs, g) < 1e-5
        @test isposdef(Symmetric(-H))
    end

    @testset "Heywood-case Normal trait: certified stationary, not -Inf" begin
        # A Normal trait whose σ is driven to near-zero (the regime an outer fit's
        # free log-σ can reach; a Beta/Normal/… mix at a Heywood optimum is exactly
        # where review-514 measured this) mixed with a Poisson trait. Before the
        # scale-aware gradient check, `maximum(abs, g) < grad_tol` on the raw score
        # `Λ's − z` was unattainable once `W = 1/σ²` amplified the unavoidable
        # floating-point residual past `grad_tol`, so a genuine mode was reported as
        # -Inf even though it is a true stationary point.
        K = 1
        Λh = reshape([1.3, -0.8], 2, 1)
        βh = [0.2, -0.3]
        z_true = [0.7]
        σtiny = 1e-8
        y_normal = βh[1] + Λh[1, 1] * z_true[1]   # exact fit: zero residual at z_true
        yh = [y_normal, 4.0]
        famsh = [Normal(0.0, σtiny), Poisson()]
        linksh = [GLLVModels.IdentityLink(), GLLVModels.LogLink()]
        n1h = ones(Int, 2)

        z, ok = GLLVModels._mixed_laplace_mode(famsh, linksh, yh, n1h, Λh, βh)
        @test ok
        v = GLLVModels._mixed_loglik_site(famsh, linksh, yh, n1h, Λh, βh)
        @test isfinite(v)

        families_tag = [(:identity, famsh[1]), (:log, famsh[2])]
        q = zz -> _ind_logpost(families_tag, yh, n1h, Λh, βh, zz)
        g = ForwardDiff.gradient(q, z)
        H = ForwardDiff.hessian(q, z)
        @test isposdef(Symmetric(-H))
        # Scale-free certification (the Newton decrement, matching review-514's own
        # probe): the raw gradient itself is O(1) here purely from floating-point
        # amplification by `1/σ²`, so an absolute bound on `g` would wrongly reject
        # this genuine mode.
        nd = abs(0.5 * dot(g, Symmetric(-H) \ g))
        @test nd < 1e-6
    end

    @testset "non-contracting Fisher fixed point: certified stationary, not -Inf" begin
        # A low-shape Gamma trait far out in its tail (shape 0.7, y/μ ≈ 5, mirroring
        # review-514's own trace: "observed weight 2.6 vs Fisher 0.5") mixed with a
        # Poisson trait. Before the small-step shortcut was gated behind "no full
        # step has been rejected yet in this call", the undamped fixed-point map was
        # not a contraction here and the search oscillated to -Inf even though a
        # genuine mode exists (4/3000 random sites in review-514's own probe).
        Λg = reshape([-1.0246747788910122, -0.33378003105895476], 2, 1)
        βg = [2.1248325556839487, -0.2627633787999423]
        yg = [438.601569107305, 1.0]
        famsg = [Gamma(0.7, 1.0), Poisson()]
        linksg = [GLLVModels.LogLink(), GLLVModels.LogLink()]
        n1g = ones(Int, 2)

        z, ok = GLLVModels._mixed_laplace_mode(famsg, linksg, yg, n1g, Λg, βg; maxiter = 2000)
        @test ok
        v = GLLVModels._mixed_loglik_site(famsg, linksg, yg, n1g, Λg, βg; maxiter = 2000)
        @test isfinite(v)

        families_tag = [(:log, famsg[1]), (:log, famsg[2])]
        q = zz -> _ind_logpost(families_tag, yg, n1g, Λg, βg, zz)
        g = ForwardDiff.gradient(q, z)
        H = ForwardDiff.hessian(q, z)
        @test maximum(abs, g) < 1e-5
        @test isposdef(Symmetric(-H))
    end

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        z, ok = GLLVModels._mixed_laplace_mode(fams0, links0, y, n1, Λ0, β0; maxiter = 0)
        @test !ok
        @test GLLVModels._mixed_loglik_site(fams0, links0, y, n1, Λ0, β0; maxiter = 0) == -Inf
        Y = reshape(y, p, 1)
        N = reshape(n1, p, 1)
        @test GLLVModels.mixed_marginal_loglik_laplace(fams0, links0, Y, N, Λ0, β0;
                                                        maxiter = 0) == -Inf
    end

    @testset "getLV/predict call sites keep a no-sentinel contract" begin
        # `getLV` (src/families/mixed.jl) calls `_mixed_laplace_mode` directly and
        # discards the convergence flag, exactly reproduced here: even at a budget
        # that cannot converge, it is a finite vector, never -Inf/NaN.
        z, ok = GLLVModels._mixed_laplace_mode(fams0, links0, y, n1, Λ0, β0; maxiter = 0)
        @test !ok
        @test all(isfinite, z)
    end

    @testset "where the old undamped loop converged, the value is unchanged (to 1e-8)" begin
        well_behaved = [
            (reshape([0.3, -0.2, 0.4], 3, 1), [0.5, 0.2, -0.1], [1.0, 0.0, 2.0]),
            (reshape([0.2, 0.15, -0.3], 3, 1), [0.1, -0.2, 0.3], [3.0, 1.0, 1.5]),
            (reshape([0.1, -0.15, 0.2], 3, 1), [-0.2, 0.4, 0.1], [5.0, 1.0, 0.8]),
        ]
        n_compared = 0
        worst = 0.0
        for (Λ, β, yc) in well_behaved
            old, conv = _mixed503_old_site(fams0, links0, yc, n1, Λ, β)
            conv || continue
            new = GLLVModels._mixed_loglik_site(fams0, links0, yc, n1, Λ, β)
            worst = max(worst, abs(new - old))
            n_compared += 1
        end
        @test n_compared >= 2
        @test worst < 1e-8
    end
end

# ===========================================================================
# Stress-probe stationarity across several family mixes (relation-only: every
# site the search reports converged must sit at a stationary point with a
# negative-definite Hessian, independently certified by ForwardDiff on each
# family's OWN `Distributions.logpdf`, never a seed-tied numeric outcome, so
# it holds regardless of which Julia version or platform drew the data).
# ===========================================================================

function _stress_probe(mixname, fams, families_tag, links; ntrials = 60, seed = 20260926)
    rng = MersenneTwister(seed)
    p = length(fams)
    n_certified = 0
    n_not_converged = 0
    for _ in 1:ntrials
        K = rand(rng, 1:3)
        sc = rand(rng, (0.5, 1.0, 2.0, 3.0))
        Λ = sc .* randn(rng, p, K)
        β = randn(rng, p) .* 1.2
        n1 = ones(Int, p)
        y = [_stress_y(fams[t], families_tag[t][1], β[t] + sum(Λ[t, :] .* randn(rng, K)), rng)
             for t in 1:p]
        z, ok = GLLVModels._mixed_laplace_mode(fams, links, y, n1, Λ, β; maxiter = 2000)
        # A non-converged draw is COUNTED, not silently skipped (review-514 S6): a
        # bare `ok || continue` with only a loose "at least half certify" floor would
        # not see a regression that pushed a modest fraction of draws to a false
        # -Inf (S1/S2's failure mode measured 4/3000, far below a 50% threshold).
        if !ok
            n_not_converged += 1
            continue
        end
        q = zz -> _ind_logpost(families_tag, y, n1, Λ, β, zz)
        g = ForwardDiff.gradient(q, z)
        H = ForwardDiff.hessian(q, z)
        # Relation-only, relaxed tolerance (mirrors #509's Student-t stress probe):
        # on an ill-conditioned draw the raw gradient can sit slightly above a tight
        # bound in a stiff direction even at a genuine stationary point.
        @test maximum(abs, g) < 2e-3
        @test isposdef(Symmetric(-H))
        n_certified += 1
    end
    # Tightened from a bare majority: measured at 90-95% on both Julia 1.10 and 1.13
    # with this seed, so an 80% floor has margin for genuinely mode-less random draws
    # and for the RNG stream differing slightly across Julia versions, while still
    # catching a regression that reintroduces S1/S2's failure mode.
    @test n_not_converged <= ntrials ÷ 5
    @test n_certified >= ntrials - ntrials ÷ 5
    return n_certified
end

function _stress_y(fam, tag::Symbol, η, rng)
    fam isa Poisson && return Float64(rand(rng, Poisson(max(exp(clamp(η, -5, 6)), 1e-8))))
    fam isa Binomial && return rand(rng, Bool) ? 1.0 : 0.0
    fam isa Gamma && return rand(rng, Gamma(shape(fam), max(exp(clamp(η, -5, 6)), 1e-8) / shape(fam)))
    fam isa Beta && return clamp(rand(rng), 1e-6, 1 - 1e-6)
    fam isa Normal && return η + randn(rng) * fam.σ
    fam isa NegativeBinomial && return Float64(rand(rng, 0:20))
    throw(ArgumentError("no stress sampler for $(typeof(fam))"))
end

@testset "Mixed-family mode search: stress-probe stationarity (#503)" begin
    G = GLLVModels

    @testset "Poisson / Binomial / Gamma" begin
        fams = [Poisson(), Binomial(), Gamma(2.0, 1.0)]
        tags = [(:log, fams[1]), (:logit, fams[2]), (:log, fams[3])]
        links = [G.LogLink(), G.LogitLink(), G.LogLink()]
        _stress_probe("Poisson/Binomial/Gamma", fams, tags, links)
    end

    @testset "Normal / NegativeBinomial / Beta" begin
        fams = [Normal(0.0, 1.5), NegativeBinomial(10.0, 0.5), Beta(3.0, 1.0)]
        tags = [(:identity, fams[1]), (:log, fams[2]), (:logit, fams[3])]
        links = [G.IdentityLink(), G.LogLink(), G.LogitLink()]
        _stress_probe("Normal/NegativeBinomial/Beta", fams, tags, links)
    end

    @testset "Poisson / Gamma / Beta / Binomial (four traits)" begin
        fams = [Poisson(), Gamma(1.5, 1.0), Beta(2.0, 1.0), Binomial()]
        tags = [(:log, fams[1]), (:log, fams[2]), (:logit, fams[3]), (:logit, fams[4])]
        links = [G.LogLink(), G.LogLink(), G.LogitLink(), G.LogitLink()]
        _stress_probe("Poisson/Gamma/Beta/Binomial", fams, tags, links)
    end
end
