using GLLVModels, Test, SHA, TOML, LinearAlgebra

# fit_truncated_nbinom2_gllvm could report converged = true at a Laplace breakdown
# point. The observed truncated-NB2 curvature is negative for small r, so the site
# precision A = I + Λ' diag(W) Λ can approach singularity while the mode search still
# converges, and -1/2 logdet(A) inflates the Laplace value (impossible-loglik audit,
# 2026-09-27, finding F1; same mechanism as the zi_* guard of PR #557). On seed 104 the
# breakdown point was a spurious GLOBAL maximum of the Laplace objective: -1663.100,
# against -1686.335 at the healthy optimum, while the exact marginal there is -1741.2
# (and -1686.39 at the healthy optimum). Seeds 106 and 110 stalled at local breakdown
# points below the healthy optimum and were still reported converged.
# Data: literal draws of the audit's genTNB (p = 4, n = 150, K = 1, r = 0.3).

const _TNB_BD = TOML.parsefile(joinpath(@__DIR__, "fixtures", "truncnb2_laplace_breakdown_audit.toml"))

function _tnb_bd_data(seed)
    d = _TNB_BD["seed$seed"]
    Y = reshape(Int.(d["Y_column_major"]), _TNB_BD["p"], _TNB_BD["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]
    return Y
end

# Exact marginal (K = 1) by a 4001-point trapezoid over z in [-12, 12], using the
# package's own conditional density and clamps; and the smallest site Laplace
# precision eigenvalue at the kernel's own mode.
function _tnb_exact_and_minA(Y, β, Λ, r)
    G = GLLVModels
    fam = TruncatedNegBin2(r); link = LogLink()
    p, n = size(Y); λ = vec(Λ)
    zs = range(-12, 12; length = 4001); h = step(zs)
    exact = 0.0; minA = Inf
    fams = fill(fam, p); N1 = ones(Int, p)
    for i in 1:n
        y = Y[:, i]
        vals = map(zs) do z
            s = -0.5 * z^2 - 0.5 * log(2π)
            for t in 1:p
                η = G._clamp_eta(β[t] + λ[t] * z)
                s += G._glm_logpdf(fam, G._clamp_mu(fam, exp(η)), 1, y[t])
            end
            s
        end
        m = maximum(vals)
        exact += m + log(sum(exp.(vals .- m)) * h)
        ẑ = G._grouped_laplace_mode(fams, y, N1, Λ, β, link)
        η = G._clamp_eta.(β .+ Λ * ẑ)
        μ = G._clamp_mu.(fams, exp.(η))
        W = [G._truncnb2_observed_weight(fam, μ[t], y[t], link) for t in 1:p]
        minA = min(minA, eigmin(Symmetric(Λ' * (W .* Λ) + I)))
    end
    return exact, minA
end

const _TNB_BD_FITS = Dict{Int, Any}()

# A literal parameter vector [β; pack(Λ); log r] from the fixture, sha256-checked.
function _tnb_bd_theta(seed, name)
    d = _TNB_BD["seed$seed"]
    θ = Float64.(d[name])
    @test bytes2hex(sha256(reinterpret(UInt8, θ))) == d[name * "_sha256"]
    p = _TNB_BD["p"]
    return θ[1:p], GLLVModels.unpack_lambda(θ[(p + 1):(2p)], p, 1), exp(θ[end])
end

# These tests assert properties of the guard and of the fit that must hold on every
# platform. They do not pin an optimiser path: L-BFGS here uses finite differences, and
# the path depends on the BLAS kernel (on Julia 1.10.12 the unguarded seed-104 fit ends
# at -1663.1 on aarch64 macOS, -1723.1 under x86_64 Rosetta and -1721.4 on the Linux CI
# runner). What the guard does at a given point is tested at fixed parameters below.
@testset "truncated NB2: breakdown guard at fixed parameters" begin
    Y = _tnb_bd_data(104)
    flr = GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR
    mineig(β, Λ, r) = GLLVModels._truncnb2_min_site_eigen(Y, Λ, β, fill(r, 4))
    lap(β, Λ, r; kw...) = truncated_nbinom2_marginal_loglik_laplace(Y, Λ, β, r; kw...)

    # The spurious maximum: a near-singular site inflates the unguarded Laplace value
    # far above both the healthy optimum and the exact marginal at the same point. The
    # guarded objective walls it off.
    βb, Λb, rb = _tnb_bd_theta(104, "theta_breakdown")
    raw = lap(βb, Λb, rb)
    raw_exact, raw_minA = _tnb_exact_and_minA(Y, βb, Λb, rb)
    @test raw_minA < 1e-3
    @test mineig(βb, Λb, rb) ≈ raw_minA rtol = 1e-6
    @test raw > -1686.335 + 20
    @test raw - raw_exact > 50
    @test lap(βb, Λb, rb; eigmin_floor = -Inf) == raw       # public marginal is unguarded by default
    @test lap(βb, Λb, rb; eigmin_floor = flr) == -Inf

    # The healthy optimum: the guard leaves it untouched, and Laplace is accurate there.
    βh, Λh, rh = _tnb_bd_theta(104, "theta_healthy")
    ex_h, minA_h = _tnb_exact_and_minA(Y, βh, Λh, rh)
    @test minA_h > 5 * flr
    @test lap(βh, Λh, rh; eigmin_floor = flr) == lap(βh, Λh, rh)
    @test abs(lap(βh, Λh, rh) - ex_h) < 3.0

    # A second, small-r basin (r = 0.0085), about 1 log unit below the healthy optimum:
    # not a breakdown (eigenvalue 0.83), so the guard correctly passes it, but the
    # Laplace value overstates the exact marginal there by about 5 units. The guard is
    # not designed to catch this; it is recorded so the limit is explicit.
    βr, Λr, rr = _tnb_bd_theta(104, "theta_ridge")
    ex_r, minA_r = _tnb_exact_and_minA(Y, βr, Λr, rr)
    @test minA_r > 5 * flr
    @test lap(βr, Λr, rr; eigmin_floor = flr) == lap(βr, Λr, rr)
    @test 3.0 < lap(βr, Λr, rr) - ex_r < 10.0
    @test lap(βh, Λh, rh) - lap(βr, Λr, rr) > 0.5
end

@testset "truncated NB2: Laplace breakdown guard (audit F1)" begin
    # Best healthy point known per seed (audit P5 shrunk-start refits; seed 110's
    # -1637.240 was found by this fix and is 0.016 above the audit's -1637.256).
    healthy = Dict(104 => -1686.335, 106 => -1888.350, 110 => -1637.240)
    flr = GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR
    for seed in (104, 106, 110)
        @testset "seed $seed" begin
            Y = _tnb_bd_data(seed)
            fit = _TNB_BD_FITS[seed] = fit_truncated_nbinom2_gllvm(Y; K = 1)
            # `exact` is the quadrature marginal at the fitted parameters themselves.
            exact, minA = _tnb_exact_and_minA(Y, fit.β, fit.Λ, fit.r)
            @test fit.converged
            # Not at a near-singular site: healthy optima measure 0.74 to 1.1 (16 random
            # starts per seed, including the seed-104 small-r basin), breakdown points
            # 8e-6 to 1.3e-4.
            @test fit.min_site_eigen ≈ minA rtol = 1e-8
            @test minA > 5 * flr
            # No spurious maximum above the best healthy point (the breakdown on seed 104
            # was 23 units above it).
            @test fit.loglik < healthy[seed] + 0.1
            if fit.loglik >= healthy[seed] - 0.05
                # At the healthy optimum the Laplace value is a usable approximation to
                # the exact marginal (measured -0.2 to 1.7 units; breakdown: 31 to 139).
                @test abs(fit.loglik - exact) < 3.0
            else
                # Seed 104 only: the small-r basin (see the fixed-parameter testset). The
                # Linux x86_64 default fit (Julia 1.10.12; CI runner, reproduced on an AMD
                # EPYC host) converges at -1687.06, 0.72 below the healthy optimum, with
                # r = 0.028, smallest site eigenvalue 0.88, Laplace minus exact 3.74. On aarch64
                # macOS, 5 of 16 random starts ended in this basin: 1.03 to 1.34 below
                # the healthy optimum, r = 0.001 to 0.009, Laplace minus exact 4.9 to 8.4,
                # smallest site eigenvalue 0.74 to 0.83, so not a breakdown.
                @test seed == 104
                @test fit.r < 0.05
                @test fit.loglik >= healthy[seed] - 2.0
                @test 0.0 < fit.loglik - exact < 10.0
            end
        end
    end
end

@testset "truncated NB2: breakdown guard mechanics" begin
    # The flag rule is load-bearing: Optim reports converged = true for a fit stalled
    # at the wall. With a floor above the healthy optimum's eigenvalue (1.01 on seed
    # 106), both the first fit and the retry end at the guard, and the fit must say so.
    Y6 = _tnb_bd_data(106)
    walled = @test_logs (:warn, r"Laplace breakdown guard") match_mode = :any fit_truncated_nbinom2_gllvm(Y6; K = 1, eigmin_floor = 2.0)
    @test !walled.converged
    @test walled.min_site_eigen < 1.1 * 2.0
end

@testset "truncated NB2: both fits at the guard report the better one" begin
    # r = 0.05 draw (PR #581 review, finding 2). The default start (r = 10) ends at the
    # wall at loglik -3659.3 with r̂ -> 6e-42; the moment-r retry also ends at the wall,
    # at -2008.5 (Julia 1.10.12) or -2006.0 (1.13.0). Both are flagged; the fit reported
    # is the retry, not the far worse first fit.
    Y = _tnb_bd_data(604)
    fit = fit_truncated_nbinom2_gllvm(Y; K = 1)
    @test !fit.converged
    @test fit.min_site_eigen < 1.1 * GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR
    @test fit.loglik > -2100
    @test fit.r > 1e-3
end
