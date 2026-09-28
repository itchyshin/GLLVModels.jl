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

@testset "truncated NB2: Laplace breakdown guard (audit F1)" begin
    # Best healthy point known per seed (audit P5 shrunk-start refits; seed 110's
    # -1637.240 was found by this fix and is 0.016 above the audit's -1637.256). The
    # data have several maxima, so assert "no worse than the best healthy point known";
    # the Laplace-vs-exact check below rules out a spurious higher value.
    healthy = Dict(104 => -1686.335, 106 => -1888.350, 110 => -1637.240)
    for seed in (104, 106, 110)
        @testset "seed $seed" begin
            Y = _tnb_bd_data(seed)
            fit = _TNB_BD_FITS[seed] = fit_truncated_nbinom2_gllvm(Y; K = 1)
            exact, minA = _tnb_exact_and_minA(Y, fit.β, fit.Λ, fit.r)
            # No fit reports converged at a near-singular site.
            @test !(fit.converged && minA < 0.11)
            # The reported loglik is a usable approximation to the exact marginal there
            # (healthy Laplace error on these data: 0.06 to 1.6 units; breakdown: 40 to 139).
            @test abs(fit.loglik - exact) < 3.0
            # The healthy optimum is reached (the spurious maximum on seed 104 was 23 higher).
            @test fit.loglik >= healthy[seed] - 0.01
            @test fit.converged
        end
    end
end

@testset "truncated NB2: breakdown guard mechanics" begin
    Y = _tnb_bd_data(104)
    # The unguarded objective (eigmin_floor = -Inf, main's code path) still ends at a
    # breakdown point that beats the healthy optimum; this pins the defect the guard
    # exists for. (-1663.100 on Julia 1.10.12 and -1678.975 on 1.13.0: the L-BFGS path
    # differs across versions, so the value is not pinned.)
    raw = fit_truncated_nbinom2_gllvm(Y; K = 1, eigmin_floor = -Inf)
    raw_exact, _ = _tnb_exact_and_minA(Y, raw.β, raw.Λ, raw.r)
    @test raw.min_site_eigen < 1e-3
    @test raw.loglik > _TNB_BD_FITS[104].loglik + 5
    @test raw.loglik - raw_exact > 20
    # The guarded objective refuses that point; the public marginal is unguarded by default.
    @test truncated_nbinom2_marginal_loglik_laplace(Y, raw.Λ, raw.β, raw.r) ≈ raw.loglik atol = 1e-6
    @test truncated_nbinom2_marginal_loglik_laplace(Y, raw.Λ, raw.β, raw.r;
              eigmin_floor = GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR) == -Inf
    # The guarded fit records the smallest site eigenvalue at its optimum.
    fit = _TNB_BD_FITS[104]
    _, minA = _tnb_exact_and_minA(Y, fit.β, fit.Λ, fit.r)
    @test fit.min_site_eigen ≈ minA rtol = 1e-8
    @test fit.min_site_eigen >= 1.1 * GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR
    # The flag rule is load-bearing: Optim reports converged = true for a fit stalled
    # at the wall. With a floor above the healthy optimum's eigenvalue (1.01 on seed
    # 106), both the first fit and the retry end at the guard, and the fit must say so.
    Y6 = _tnb_bd_data(106)
    walled = @test_logs (:warn, r"Laplace breakdown guard") match_mode = :any fit_truncated_nbinom2_gllvm(Y6; K = 1, eigmin_floor = 2.0)
    @test !walled.converged
    @test walled.min_site_eigen < 1.1 * 2.0
end
