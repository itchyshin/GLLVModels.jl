using GLLVModels, Test, Random, LinearAlgebra, Statistics
using Logging

# bootstrap_Sigma_r: pinned-R behaviour target (gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9,
# R/bootstrap-sigma.R). Spec and gates: docs/dev-log/plans/2026-11-01-bootstrap-sigma-checkpoint-ledger.md.
#
# Expected values are computed independently of the code under test: from `fit.pars` / the
# TwoLevelFit fields by hand, from the sample covariance of simulated data, and from
# Statistics.quantile over replicates logged by the test's own refit wrapper. The driver's
# internal `_refit` keyword is the test seam (not public API).
#
# WIRING: src/bootstrap_sigma_r.jl is not yet include()d by src/GLLVModels.jl (deferred behind
# PR #1036), so this file includes it into the module when the name is absent. Once wired,
# the guard is a no-op. Not yet registered in test/runtests.jl for the same reason.

const _BSIG_SRC = joinpath(pkgdir(GLLVModels), "src", "bootstrap_sigma_r.jl")
if !isdefined(GLLVModels, :bootstrap_Sigma_r) && isfile(_BSIG_SRC)
    Base.include(GLLVModels, _BSIG_SRC)
end
const _bsig_defined = isdefined(GLLVModels, :bootstrap_Sigma_r)
_bsig(args...; kw...) = _bsig_defined ? GLLVModels.bootstrap_Sigma_r(args...; kw...) :
    error("bootstrap_Sigma_r is not defined (RED): src/bootstrap_sigma_r.jl missing")

_quiet(f) = with_logger(f, NullLogger())
_logged(f; min_level = Logging.Info) = (lg = Test.TestLogger(min_level = min_level);
                                        r = with_logger(f, lg); (r, lg.logs))
_warns(logs) = [string(l.message) for l in logs if l.level == Logging.Warn]
_infos(logs) = [string(l.message) for l in logs if l.level == Logging.Info]

# Non-converged but finite, biased copy of a refit (device from test_bootstrap_decisions.jl).
function _bs_cripple(fb::GllvmFit)
    θ = copy(fb.pars.θ_packed); θ[1] += log(3.0)
    pars = merge(fb.pars, (σ_eps = 3 * fb.pars.σ_eps, θ_packed = θ))
    return GllvmFit(fb.model, pars, fb.logLik, fb.n_iter, false, fb.optim_result, fb.cputime)
end

# GllvmFit refit wrapper: counts calls, logs responses and the fit returned.
function _bs_refit(; bad::Function = _ -> false)
    log = (ys = Matrix{Float64}[], fits = GllvmFit[], n = Ref(0))
    f = function (yb; kwargs...)
        log.n[] += 1
        push!(log.ys, copy(yb))
        fb = fit_gaussian_gllvm(yb; kwargs...)
        out = bad(log.n[]) ? _bs_cripple(fb) : fb
        push!(log.fits, out)
        return out
    end
    return f, log
end

# A converged GllvmFit with Λ scaled by c (deterministic, no optimiser).
_scaled(fit::GllvmFit, c; ok = true) = GllvmFit(fit.model, merge(fit.pars, (Λ = c .* fit.pars.Λ,)),
                                                 fit.logLik, 1, ok, fit.optim_result, 0.0)

# A converged has_diag GllvmFit with σ²_B scaled by c (deterministic, no optimiser): the total
# Σ_B = ΛΛ' + diag(c σ²_B + σ²_W + σ_eps²) then VARIES with c, so communality and R vary.
_scaled_diag(fit::GllvmFit, c) = GllvmFit(fit.model, merge(fit.pars, (σ²_B = c .* fit.pars.σ²_B,)),
                                          fit.logLik, 1, true, fit.optim_result, 0.0)
# independent (by-hand) expectations for the has_diag tier B, straight from fit.pars
_diag_total(pr, c) = pr.Λ * pr.Λ' + Diagonal(c .* pr.σ²_B .+ pr.σ²_W .+ pr.σ_eps^2)

# Hand-built TwoLevelFit with Σ_B != Σ_W (so a tier swap is visible).
function _mk2(sB = 1.0, sW = 1.0; ok = true)
    ΛB = sB .* reshape([0.9, 0.6, -0.4], 3, 1); σB = sB .* [0.30, 0.20, 0.25]
    ΛW = sW .* reshape([0.3, -0.2, 0.5], 3, 1); σW = sW .* [0.60, 0.50, 0.40]
    TwoLevelFit(ΛB, σB, ΛW, σW, Matrix(ΛB * ΛB' + Diagonal(σB)),
                Matrix(ΛW * ΛW' + Diagonal(σW)), 30, -1.0, ok, 1)
end
const _IND = repeat(1:30, inner = 3)
# independent expectations straight from the fields (not via any extractor)
_t2_sigma(f, t) = t === :B ? Matrix(f.Λ_B * f.Λ_B' + Diagonal(f.σ²_B)) :
                             Matrix(f.Λ_W * f.Λ_W' + Diagonal(f.σ²_W))
_t2_comm(f, t) = t === :B ? [f.Λ_B[i]^2 / (f.Λ_B[i]^2 + f.σ²_B[i]) for i in 1:3] :
                            [f.Λ_W[i]^2 / (f.Λ_W[i]^2 + f.σ²_W[i]) for i in 1:3]
_t2_icc(f) = [(f.Λ_B[i]^2 + f.σ²_B[i]) / (f.Λ_B[i]^2 + f.σ²_B[i] + f.Λ_W[i]^2 + f.σ²_W[i]) for i in 1:3]
_cor(S) = S ./ sqrt.(diag(S) * diag(S)')
struct _NotAFit end
# a matrix whose element access raises InterruptException (extraction-time interrupt probe)
struct _BsigBoom <: AbstractMatrix{Float64}; p::Int; end
Base.size(b::_BsigBoom) = (b.p, 1)
Base.getindex(::_BsigBoom, ::Int, ::Int) = throw(InterruptException())

@testset "bootstrap_Sigma_r (pinned-R contract)" begin
    @testset "0. RED baseline: the driver exists" begin
        @test _bsig_defined
    end
    rng = MersenneTwister(20261101)
    p, K, n = 4, 1, 150
    Λ0 = [0.9, 0.6, 0.4, -0.5]
    y = Λ0 * randn(rng, 1, n) + 0.5 * randn(rng, p, n)        # sigma_eps = 0.5 so sigma_eps^2 is visible
    fit = fit_gaussian_gllvm(y; K = 1)                         # Fixture A: K_W = 0, no diag
    fitd = fit_gaussian_gllvm(y; K = 1, has_diag = true)       # Fixture A2: has_diag, K_W = 0
    NB = 12

    @testset "fixtures" begin
        @test fit.converged && fit.model.K_W == 0 && !fit.model.has_diag
        @test fit.pars.σ_eps^2 > 0.1                           # control: contamination is observable
        @test maximum(abs, fit.pars.Λ * fit.pars.Λ' - Diagonal(diag(fit.pars.Λ * fit.pars.Λ'))) > 0.05
        @test fitd.model.has_diag && fitd.model.K_W == 0 && all(isfinite, fitd.pars.θ_packed)
    end

    @testset "1. argument validation, aliases, defaults, @info drops" begin
        r, logs = _logged(() -> _bsig(fit; y = y, n_boot = NB, seed = 1, progress = false))
        @test r.level == [:B]                                  # default vector: W, phy absent, dropped
        @test r.what == [:Sigma, :R, :communality, :cross_corr]  # ICC dropped (R:299-306); cross_corr kept as given
        @test any(occursin("not present", m) && occursin("W", m) && occursin("phy", m) for m in _infos(logs))
        @test any(occursin("ICC", m) for m in _infos(logs))
        @test r.conf == 0.95 && r.ci_method == "percentile" && r.link_residual === :auto
        @test collect(keys(r.point_est)) == [:Sigma_B, :R_B, :communality_B]   # cross_corr skipped (no multinomial trait)
        # unit and B are one tier: dedup, internal name :B (never :unit)
        for lv in ([:unit, :B], (:B, :unit), :unit, :B)
            ra = _quiet(() -> _bsig(fit; y = y, level = lv, what = :Sigma, n_boot = NB, seed = 1, progress = false))
            @test ra.level == [:B]
            @test ra.what == [:Sigma]
        end
        # all requested tiers absent -> error (R:290-295), refused before any refit
        f0, l0 = _bs_refit()
        for lv in (:unit_obs, :W, [:W, :unit_obs], :phy, [:phy, :W])
            @test_throws ArgumentError _bsig(fit; y = y, level = lv, n_boot = NB, _refit = f0, progress = false)
        end
        @test l0.n[] == 0
        @test_throws ArgumentError _bsig(fit; y = y, level = :nonsense, n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, what = [:nonsense], n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, link_residual = :bogus, n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, conf = 1.0, n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, conf = 0.0, n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, conf = -0.1, n_boot = NB)
        @test_throws ArgumentError _bsig(fit; y = y, n_boot = 0)
        @test_throws ArgumentError _bsig(fit; y = y, n_boot = -3)
        @test_throws ArgumentError _bsig(fit; n_boot = NB)             # no y, n_sites or X
        # `what` restricts the output
        r2 = _quiet(() -> _bsig(fit; y = y, what = [:Sigma], n_boot = NB, seed = 1, progress = false))
        @test collect(keys(r2.point_est)) == [:Sigma_B] && isempty(r2.n_effective)
        # nothing reportable (cross_corr only) is refused before simulating
        @test_throws ArgumentError _bsig(fit; y = y, what = [:cross_corr], n_boot = NB, _refit = f0, progress = false)
        @test l0.n[] == 0
    end

    @testset "2. point estimates: absolute values from fit.pars (not the extractors)" begin
        Λ = fit.pars.Λ
        r = _quiet(() -> _bsig(fit; y = y, level = :unit, n_boot = NB, seed = 2, progress = false))
        B = r.point_est.Sigma_B
        @test B ≈ Λ * Λ' rtol = 1e-12                                       # R tier: no sigma_eps^2
        @test r.point_est.R_B ≈ _cor(Λ * Λ') rtol = 1e-10
        @test r.point_est.communality_B ≈ ones(p) rtol = 1e-10              # B = ΛΛ' only, so communality is 1
        s2 = fit.pars.σ_eps^2
        @test diag(sigma_y_site(fit)) .- diag(B) ≈ fill(s2, p) rtol = 1e-10  # the contaminated quantity differs by σ_eps^2
        @test maximum(abs, diag(B) .- diag(sigma_y_site(fit))) > 0.1
        @test !isapprox(B, extract_Sigma(fit; level = :unit_obs).Sigma; rtol = 1e-3)  # never the unit_obs total
        # Fixture A2 (has_diag, K_W = 0): B is the identified total ΛΛ' + diag(σ²_B + σ²_W + σ_eps^2), by hand.
        pr = fitd.pars
        tot = pr.Λ * pr.Λ' + Diagonal(pr.σ²_B .+ pr.σ²_W .+ pr.σ_eps^2)
        rd = _quiet(() -> _bsig(fitd; y = y, level = :B, n_boot = NB, seed = 2, progress = false,
                                _refit = (yb; kw...) -> fitd))
        @test rd.point_est.Sigma_B ≈ tot rtol = 1e-10
        @test rd.point_est.communality_B ≈ diag(pr.Λ * pr.Λ') ./ diag(tot) rtol = 1e-10
        @test all(c -> 0 < c < 1, rd.point_est.communality_B)               # not the A value 1: control
        @test rd.level == [:B]                                              # W absent (R: diag_W unused)
    end

    @testset "3. exactly one refit per replicate, whatever p or what" begin
        for what in ([:Sigma], [:Sigma, :R, :communality, :cross_corr])
            f, log = _bs_refit()
            _quiet(() -> _bsig(fit; y = y, what = what, n_boot = NB, seed = 4, _refit = f, progress = false))
            @test log.n[] == NB
        end
        p6 = 6
        y6 = randn(MersenneTwister(3), p6, 1) * randn(MersenneTwister(4), 1, 120) .+ 0.5 * randn(MersenneTwister(5), p6, 120)
        f6 = fit_gaussian_gllvm(y6; K = 1)
        f, log = _bs_refit()
        _quiet(() -> _bsig(f6; y = y6, n_boot = NB, seed = 4, _refit = f, progress = false))
        @test log.n[] == NB      # one refit per replicate for p = 6 too (the legacy design would need 21 * NB)
    end

    @testset "4. batch and seed semantics" begin
        run(seed; bad = _ -> false, what = :Sigma) = begin
            f, log = _bs_refit(bad = bad)
            r = _quiet(() -> _bsig(fit; y = y, what = what, n_boot = NB, seed = seed, _refit = f, progress = false))
            (r, log)
        end
        a, la = run(7); b, lb = run(7); c, lc = run(8)
        @test la.ys == lb.ys && la.ys != lc.ys                    # same seed identical, different seed differs
        @test length(unique(la.ys)) == NB && all(size(Y) == (p, n) for Y in la.ys)
        @test a.ci_lower.Sigma_B == b.ci_lower.Sigma_B && a.ci_lower.Sigma_B != c.ci_lower.Sigma_B
        # one stream, not seed + b: batch(8) is not batch(7) shifted by one replicate
        @test la.ys[2] != lc.ys[1] && la.ys[3] != lc.ys[2]
        # datasets are identical across `what` (shared batch)
        _, lw = run(7; what = [:Sigma, :R, :communality])
        @test lw.ys == la.ys
        # C4: the batch is drawn BEFORE any refit; refit outcomes cannot change any dataset
        _, lbad = run(7; bad = i -> i % 3 == 0)
        @test lbad.ys == la.ys
        # default seed (nothing) uses the GLOBAL stream: reseeding it reproduces the batch,
        # a different global seed gives a different batch, and consecutive calls differ
        gbatch(k) = begin
            fg, lg = _bs_refit()
            Random.seed!(k)
            _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, _refit = fg, progress = false))
            lg.ys
        end
        @test gbatch(5) == gbatch(5)
        @test gbatch(5) != gbatch(6)
        f1, l1 = _bs_refit(); f2, l2 = _bs_refit()
        _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, _refit = f1, progress = false))
        _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, _refit = f2, progress = false))
        @test l1.ys != l2.ys
        # percentile bounds are Statistics.quantile of the replicates logged by the wrapper
        S11 = [fb.pars.Λ[1, :]' * fb.pars.Λ[1, :] for fb in la.fits]
        @test a.ci_lower.Sigma_B[1, 1] ≈ quantile(S11, 0.025) rtol = 1e-10
        @test a.ci_upper.Sigma_B[1, 1] ≈ quantile(S11, 0.975) rtol = 1e-10
    end

    @testset "5. failure accounting and attrition" begin
        f0, l0 = _bs_refit()
        r0, lg0 = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 5, _refit = f0, progress = false))
        @test r0.n_failed == 0                                    # control
        @test !any(occursin("attrition", m) for m in _warns(lg0))
        f, log = _bs_refit(bad = i -> i % 3 == 0)
        r, lg = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 5, _refit = f, progress = false))
        @test r.n_failed == count(i -> i % 3 == 0, 1:NB)
        keep = [i for i in 1:NB if i % 3 != 0]
        S11 = [log.fits[i].pars.Λ[1, :]' * log.fits[i].pars.Λ[1, :] for i in keep]
        @test r.ci_lower.Sigma_B[1, 1] ≈ quantile(S11, 0.025) rtol = 1e-10   # survivors only
        @test r.ci_upper.Sigma_B[1, 1] ≈ quantile(S11, 0.975) rtol = 1e-10
        @test r.n_effective.communality_B == fill(length(keep), p)            # vector entry has n_effective
        @test !haskey(r.n_effective, :Sigma_B)                                # matrix entry: none (R:728-730)
        @test any(occursin("attrition", m) && occursin("communality_B", m) for m in _warns(lg))  # 8/12 < 80%
        # no attrition warning for matrix-only `what` (R limitation, kept)
        f2, _ = _bs_refit(bad = i -> i % 3 == 0)
        _, lg2 = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :R], n_boot = NB, seed = 5, _refit = f2, progress = false))
        @test !any(occursin("attrition", m) for m in _warns(lg2))
        # a refit that throws is counted failed
        k = Ref(0)
        thrower = (yb; kw...) -> (k[] += 1; k[] == 2 && error("injected"); fit_gaussian_gllvm(yb; kw...))
        rt = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = NB, seed = 5, _refit = thrower, progress = false))
        @test rt.n_failed == 1
        # control 2: a CONVERGED refit with a NaN entry is not failed; its entries are excluded
        j = Ref(0)
        nanfit = function (yb; kw...)
            j[] += 1
            fb = fit_gaussian_gllvm(yb; kw...)
            j[] % 4 == 0 || return fb
            Λn = copy(fb.pars.Λ); Λn[1, 1] = NaN
            GllvmFit(fb.model, merge(fb.pars, (Λ = Λn,)), fb.logLik, fb.n_iter, true, fb.optim_result, fb.cputime)
        end
        rn, lgn = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 5, _refit = nanfit, progress = false))
        @test rn.n_failed == 0
        @test rn.n_effective.communality_B[1] == NB - 3                      # 3 NaN draws excluded, others kept
        @test rn.n_effective.communality_B[3] == NB
        @test any(occursin("attrition", m) && occursin("communality_B", m) for m in _warns(lgn))   # pins min over elements (kills a max mutant)
        # the 0.8 boundary is pinned on both sides and at equality (rule: warn iff retained/n_boot < 0.8)
        att(nb, nbad) = begin
            ctr = Ref(0)
            stub = (yb; kw...) -> (b = (ctr[] += 1); _scaled(fit, 1.0; ok = b > nbad))
            r, lg = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = nb, seed = 1,
                                        _refit = stub, progress = false), min_level = Logging.Warn)
            @test r.n_failed == nbad
            [m for m in _warns(lg) if occursin("attrition", m)]
        end
        @test isempty(att(12, 0))
        @test isempty(att(12, 2))          # 10/12 = 0.833 retained: no warning (kills a 0.9 threshold)
        @test length(att(12, 3)) == 1      # 9/12 = 0.75 retained: warning (kills a 0.7 threshold)
        @test isempty(att(10, 2))          # 8/10 = 0.8 exactly: strict `<`, no warning (kills `<=`)
        @test length(att(10, 3)) == 1      # 7/10 = 0.7 retained: warning
    end

    @testset "6. percentile correctness with a deterministic refit" begin
        nb = 39
        c = [0.5 + 0.05 * b for b in 1:nb]
        for conf in (0.95, 0.8)
            ctr = Ref(0)
            known = (yb; kw...) -> _scaled(fit, c[ctr[] += 1])
            r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = nb, conf = conf, seed = 1, _refit = known, progress = false))
            v = (c .^ 2) .* (fit.pars.Λ[1, :]' * fit.pars.Λ[1, :])
            α = (1 - conf) / 2
            @test r.ci_lower.Sigma_B[1, 1] ≈ quantile(v, α) rtol = 1e-10
            @test r.ci_upper.Sigma_B[1, 1] ≈ quantile(v, 1 - α) rtol = 1e-10   # catches swapped tails / wrong alpha
            @test r.ci_lower.Sigma_B[1, 1] < r.ci_upper.Sigma_B[1, 1]
            @test r.point_est.Sigma_B[1, 1] ≈ fit.pars.Λ[1, :]' * fit.pars.Λ[1, :] rtol = 1e-12  # estimate = original fit
            @test r.conf == conf
        end
        # failed replicates are NaN-shaped and NOT zero: bounds equal the survivors' (guards a 0-fill)
        ctr = Ref(0)
        flaky = (yb; kw...) -> (b = (ctr[] += 1); _scaled(fit, c[b]; ok = b % 5 != 0))
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = nb, seed = 1, _refit = flaky, progress = false))
        surv = [c[b]^2 * (fit.pars.Λ[1, :]' * fit.pars.Λ[1, :]) for b in 1:nb if b % 5 != 0]
        @test r.n_failed == count(b -> b % 5 == 0, 1:nb)
        @test r.ci_lower.Sigma_B[1, 1] ≈ quantile(surv, 0.025) rtol = 1e-10
        @test r.ci_upper.Sigma_B[1, 1] ≈ quantile(surv, 0.975) rtol = 1e-10
        # zero finite draws gives NaN bounds, not an error (no minimum-survivor rule)
        rz = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                                _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = false)))
        @test rz.n_failed == 3 && all(isnan, rz.ci_lower.Sigma_B)
    end

    @testset "7. warnings never throw; ceiling; min_boot boundary" begin
        stub = (yb; kw...) -> fit
        r10, lg10 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 10, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test r10.coverage_ceiling ≈ 9 / 11
        w = _warns(lg10)
        @test length(w) == 1 && occursin("39", w[1]) && !occursin("999", w[1])   # below-min warning only
        for (nb, expect_n, pat) in ((38, 1, "39"), (39, 1, "999"), (100, 1, "999"))
            _, lg = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = nb, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
            ww = _warns(lg)
            @test length(ww) == expect_n && occursin(pat, ww[1])
        end
        _, lg39 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 39, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test !occursin("39 is below", join(_warns(lg39)))                       # 39 == min_boot clears the first warning
        r999, lg999 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 999, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test isempty(_warns(lg999))
        @test r999.coverage_ceiling ≈ 998 / 1000
        # conf = 0.5: min_boot = ceil(2/0.5) - 1 = 3 exactly
        _, lg50 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, conf = 0.5, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test occursin("999", only(_warns(lg50)))
        _, lg50b = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 2, conf = 0.5, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test occursin("min_boot = 3", only(_warns(lg50b)))
        # R's floating point is copied, not "fixed": 2/(1-0.9) > 20 in IEEE doubles, so min_boot = 20 here.
        # That R gives the same 20 follows from the same IEEE arithmetic; it was NOT checked against R output (UNVERIFIED).
        _, lg90 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 19, conf = 0.9, seed = 1, _refit = stub, progress = false), min_level = Logging.Warn)
        @test occursin("min_boot = 20", only(_warns(lg90)))
    end

    @testset "8. keep_draws and progress" begin
        r0 = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = NB, seed = 1, progress = false))
        @test r0.draws === nothing
        r1 = _quiet(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 1, keep_draws = true, progress = false))
        @test r1.draws !== nothing && length(r1.draws) == NB
        @test all(haskey(d, :Sigma_B) && haskey(d, :communality_B) for d in r1.draws)
        # failed draws are NaN-shaped
        fbad, _ = _bs_refit(bad = i -> i == 2)
        r2 = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = NB, seed = 1, keep_draws = true, _refit = fbad, progress = false))
        @test all(isnan, r2.draws[2].Sigma_B) && all(isfinite, r2.draws[1].Sigma_B)
        # progress = true emits one @info per replicate
        _, lgp = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 4, seed = 1, progress = true, _refit = (yb; kw...) -> fit))
        @test count(occursin("replicate", m) for m in _infos(lgp)) == 4
        @test_throws ArgumentError _bsig(fit; y = y, n_boot = 4, n_cores = 2)
    end

    @testset "9. link_residual :auto and :none are bit-identical for Gaussian" begin
        ra = _quiet(() -> _bsig(fit; y = y, n_boot = NB, seed = 3, link_residual = :auto, progress = false))
        rn = _quiet(() -> _bsig(fit; y = y, n_boot = NB, seed = 3, link_residual = :none, progress = false))
        @test ra.point_est == rn.point_est && ra.ci_lower == rn.ci_lower && ra.ci_upper == rn.ci_upper
        @test rn.link_residual === :none
    end

    @testset "10. refusals, each paired with a supported control" begin
        f, log = _bs_refit()
        ctl = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = NB, seed = 1, _refit = f, progress = false))
        @test ctl.level == [:B] && log.n[] == NB
        n_ok = log.n[]
        err(g) = try g(); nothing catch e; e end
        # K_W > 0: split unidentified
        fw = fit_gaussian_gllvm(y; K = 1, K_W = 1)
        e = err(() -> _bsig(fw; y = y, _refit = f, progress = false)); @test e isa ArgumentError && occursin("K_W", e.msg)
        # pinned (lambda_constraint): closes the coverage gap for the bootstrap_Sigma route
        M = fill(NaN, p, 1); M[2, 1] = 0.5
        pinned = fit_gaussian_gllvm(y; K = 1, lambda_constraint = M)
        e = err(() -> _bsig(pinned; y = y, _refit = f, progress = false)); @test e isa ArgumentError && occursin("lambda_constraint", e.msg)
        # retained Gaussian record (offset route)
        rec = fit_gaussian_gllvm(y; K = 1, offset = zeros(p, n))
        @test GLLVModels._has_gaussian_record(rec)
        e = err(() -> _bsig(rec; y = y, _refit = f, progress = false)); @test e isa ArgumentError && occursin("record", e.msg)
        # phylogenetic block
        Σp = Matrix(0.5I(p)) + fill(0.3, p, p)
        fph = fit_gaussian_gllvm(y; K = 1, K_phy = 1, Σ_phy = Σp)
        e = err(() -> _bsig(fph; y = y, _refit = f, progress = false)); @test e isa ArgumentError && occursin("phylogenetic", e.msg)
        # Σ_phy handed to a non-phylo fit
        @test_throws ArgumentError _bsig(fit; y = y, Σ_phy = Σp, progress = false, _refit = f)
        # other classes (no verified simulate-and-refit route)
        for obj in (_NotAFit(), nothing, :not_a_fit)
            e = err(() -> _bsig(obj; progress = false)); @test e isa ArgumentError && occursin("unsupported fit class", e.msg)
        end
        @test log.n[] == n_ok                                   # no refusal triggered a refit
    end

    @testset "11. simulator draws have the fitted covariance (empirical, large n)" begin
        # GllvmFit: batch covariance within MC tolerance of ΛΛ' + σ_eps^2 I (hand-built, not _derived_site_cov)
        ys = Matrix{Float64}[]
        _quiet(() -> _bsig(fit; n_sites = 20000, what = :Sigma, n_boot = 2, seed = 11, progress = false,
                           _refit = (yb; kw...) -> (push!(ys, copy(yb)); fit)))
        A = fit.pars.Λ * fit.pars.Λ' + fit.pars.σ_eps^2 * I
        Yall = hcat(ys...)
        @test maximum(abs, cov(Yall') .- A) < 0.05
        @test maximum(abs, mean(Yall, dims = 2)) < 0.05            # zero mean with no X
        # a non-trivial control: the covariance is NOT ΛΛ' alone
        @test maximum(abs, cov(Yall') .- fit.pars.Λ * fit.pars.Λ') > 0.15
        # TwoLevelFit: total cov = Σ_B + Σ_W and same-individual cross-covariance = Σ_B
        t2 = _mk2()
        nind, nobs = 4000, 2
        ind = repeat(1:nind, inner = nobs)
        t2big = TwoLevelFit(t2.Λ_B, t2.σ²_B, t2.Λ_W, t2.σ²_W, t2.Σ_B, t2.Σ_W, nind, 0.0, true, 0)
        Y2 = Matrix{Float64}[]
        _quiet(() -> _bsig(t2big; individual = ind, what = :Sigma, n_boot = 1, seed = 5, progress = false,
                           _refit = (yb, i; kw...) -> (push!(Y2, copy(yb)); t2big)))
        Yb = Y2[1]
        @test maximum(abs, cov(Yb') .- (t2.Σ_B .+ t2.Σ_W)) < 0.08
        o1 = Yb[:, 1:nobs:end]; o2 = Yb[:, 2:nobs:end]
        cross = (o1 * o2') ./ nind
        @test maximum(abs, cross .- t2.Σ_B) < 0.08
        @test maximum(abs, cross .- (t2.Σ_B .+ t2.Σ_W)) > 0.2     # control: distinguishes B from B + W
    end

    @testset "12. TwoLevelFit (stretch): tiers, ICC, order, stubs" begin
        f = _mk2(1.0, 1.7)
        stub = (yb, i; kw...) -> _mk2(1.0, 1.7)
        r = _quiet(() -> _bsig(f; individual = _IND, n_boot = 5, seed = 1, progress = false, _refit = stub))
        @test r.level == [:B, :W]
        @test r.what == [:Sigma, :R, :communality, :ICC, :cross_corr]   # default `what`, in order, nothing dropped
        @test collect(keys(r.point_est)) == [:Sigma_B, :R_B, :communality_B, :Sigma_W, :R_W, :communality_W, :ICC_site]
        for t in (:B, :W)
            @test getproperty(r.point_est, Symbol(:Sigma_, t)) ≈ _t2_sigma(f, t) rtol = 1e-12
            @test getproperty(r.point_est, Symbol(:R_, t)) ≈ _cor(_t2_sigma(f, t)) rtol = 1e-10
            @test getproperty(r.point_est, Symbol(:communality_, t)) ≈ _t2_comm(f, t) rtol = 1e-12
        end
        @test r.point_est.Sigma_B != r.point_est.Sigma_W
        @test maximum(abs, r.point_est.Sigma_B .- r.point_est.Sigma_W) > 0.1
        @test r.point_est.ICC_site ≈ _t2_icc(f) rtol = 1e-12
        @test r.n_effective.ICC_site == fill(5, 3) && !haskey(r.n_effective, :Sigma_B)
        # aliases: unit/unit_obs/B/W dedup, order of first appearance
        rr = _quiet(() -> _bsig(f; individual = _IND, level = [:unit_obs, :W, :unit], what = [:Sigma, :ICC], n_boot = 3, seed = 1, progress = false, _refit = stub))
        @test rr.level == [:W, :B] && haskey(rr.point_est, :ICC_site)
        # only B: ICC dropped with @info
        r1, lg1 = _logged(() -> _bsig(f; individual = _IND, level = :unit, what = [:Sigma, :ICC], n_boot = 3, seed = 1, progress = false, _refit = stub))
        @test r1.level == [:B] && r1.what == [:Sigma] && !haskey(r1.point_est, :ICC_site)
        @test any(occursin("ICC", m) for m in _infos(lg1))
        # replicate summaries use the right tier too (data-dependent stub scales)
        r3 = _quiet(() -> _bsig(f; individual = _IND, n_boot = 4, seed = 2, keep_draws = true, progress = false,
                                _refit = (yb, i; kw...) -> _mk2(1.3, 2.1)))
        g = _mk2(1.3, 2.1)
        @test all(d.Sigma_B ≈ _t2_sigma(g, :B) && d.Sigma_W ≈ _t2_sigma(g, :W) && d.communality_W ≈ _t2_comm(g, :W) &&
                  d.ICC_site ≈ _t2_icc(g) for d in r3.draws)
        @test all(d.R_B ≈ _cor(_t2_sigma(g, :B)) && d.R_W ≈ _cor(_t2_sigma(g, :W)) &&
                  d.communality_B ≈ _t2_comm(g, :B) for d in r3.draws)
        # one refit per replicate; failure accounting; percentile exactness
        cnt = Ref(0)
        _quiet(() -> _bsig(f; individual = _IND, n_boot = 7, seed = 3, progress = false, _refit = (yb, i; kw...) -> (cnt[] += 1; f)))
        @test cnt[] == 7
        nb41 = 41; stubs = [_mk2(0.5 + b / 40, 1.0 + 0.1 * b) for b in 1:nb41]; c = Ref(0)
        rp = _quiet(() -> _bsig(f; individual = _IND, n_boot = nb41, conf = 0.9, seed = 5, progress = false, _refit = (yb, i; kw...) -> stubs[c[] += 1]))
        M = reduce(hcat, [vec(_t2_sigma(s, :W)) for s in stubs])
        @test vec(rp.ci_lower.Sigma_W) ≈ [quantile(M[i, :], 0.05) for i in 1:9] rtol = 1e-10
        @test vec(rp.ci_upper.Sigma_W) ≈ [quantile(M[i, :], 0.95) for i in 1:9] rtol = 1e-10
        Mi = reduce(hcat, [_t2_icc(s) for s in stubs])
        @test rp.ci_lower.ICC_site ≈ [quantile(Mi[i, :], 0.05) for i in 1:3] rtol = 1e-10
        @test rp.ci_upper.ICC_site ≈ [quantile(Mi[i, :], 0.95) for i in 1:3] rtol = 1e-10
        # R_* and communality_* bounds (both tiers), against Statistics.quantile of by-hand draws
        for t in (:B, :W)
            MR = reduce(hcat, [vec(_cor(_t2_sigma(s, t))) for s in stubs])
            lo = getproperty(rp.ci_lower, Symbol(:R_, t)); hi = getproperty(rp.ci_upper, Symbol(:R_, t))
            @test vec(lo) ≈ [quantile(MR[i, :], 0.05) for i in 1:9] rtol = 1e-10
            @test vec(hi) ≈ [quantile(MR[i, :], 0.95) for i in 1:9] rtol = 1e-10
            @test all(lo .<= hi) && any(lo .< hi .- 1e-4)       # tails not swapped, and R actually varies
            MC = reduce(hcat, [_t2_comm(s, t) for s in stubs])
            @test getproperty(rp.ci_lower, Symbol(:communality_, t)) ≈ [quantile(MC[i, :], 0.05) for i in 1:3] rtol = 1e-10
            @test getproperty(rp.ci_upper, Symbol(:communality_, t)) ≈ [quantile(MC[i, :], 0.95) for i in 1:3] rtol = 1e-10
        end
        c2 = Ref(0)
        rf = _quiet(() -> _bsig(f; individual = _IND, n_boot = 20, seed = 7, progress = false,
                                _refit = (yb, i; kw...) -> (c2[] += 1; c2[] % 5 == 0 ? error("boom") : _mk2(0.8 + c2[] / 50, 1.0; ok = c2[] % 7 != 0))))
        @test rf.n_failed == count(b -> b % 5 == 0 || b % 7 == 0, 1:20)
        # refusals
        @test_throws ArgumentError _bsig(f; level = :unit, progress = false)                  # no `individual`
        @test_throws ArgumentError _bsig(f; individual = repeat(1:10, inner = 3), progress = false)  # wrong group count
        @test_throws ArgumentError _bsig(f; individual = _IND, level = :phy, progress = false) # all-absent
        # ICC edge: vB + vW <= 0 gives NaN, never Inf or an error
        z = TwoLevelFit(zeros(3, 1), zeros(3), zeros(3, 1), zeros(3), zeros(3, 3), zeros(3, 3), 30, 0.0, true, 0)
        rz = _quiet(() -> _bsig(z; individual = _IND, n_boot = 3, seed = 1, progress = false, _refit = (yb, i; kw...) -> z))
        @test all(isnan, rz.point_est.ICC_site) && all(isnan, rz.ci_lower.ICC_site) && rz.n_effective.ICC_site == zeros(Int, 3)
        @test !any(isinf, rz.point_est.ICC_site)
    end

    @testset "13. TwoLevelFit tiny real simulate-and-refit route" begin
        rng = MersenneTwister(20261101)
        pp, nind, nobs = 3, 40, 4
        ΛB = reshape([0.9, 0.6, -0.4], pp, 1); ΛW = reshape([0.4, -0.3, 0.5], pp, 1)
        ind = repeat(1:nind, inner = nobs)
        Y = Matrix{Float64}(undef, pp, nind * nobs)
        for i in 1:nind
            b = ΛB * randn(rng, 1) + sqrt.([0.3, 0.2, 0.25]) .* randn(rng, pp)
            for j in ((i - 1) * nobs + 1):(i * nobs)
                Y[:, j] = b + ΛW * randn(rng, 1) + sqrt.([0.5, 0.4, 0.45]) .* randn(rng, pp)
            end
        end
        tf = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)
        r, lg = _logged(() -> _bsig(tf; individual = ind, n_boot = 4, seed = 1, progress = false))
        @test r.n_boot == 4
        @test r.n_failed == 0            # real refit of well-posed data: seeds 1-3 all gave 0 failures (probe, 2026-11-01)
        @test r.point_est.Sigma_B == tf.Σ_B && r.point_est.Sigma_W == tf.Σ_W
        @test r.point_est.ICC_site ≈ [tf.Σ_B[t, t] / (tf.Σ_B[t, t] + tf.Σ_W[t, t]) for t in 1:pp] rtol = 1e-12
        @test all(r.ci_lower.Sigma_B .<= r.ci_upper.Sigma_B) && all(isfinite, r.ci_lower.Sigma_B)
        @test all(r.ci_lower.Sigma_W .<= r.ci_upper.Sigma_W) && all(isfinite, r.ci_upper.R_W)
        @test r.n_effective.ICC_site == fill(4, pp)       # every replicate refit and extracted
        @test any(occursin("min_boot", m) || occursin("39", m) for m in _warns(lg))
    end

    @testset "14. GllvmFit R_B and communality_B bounds on replicates that vary (has_diag)" begin
        # Fixture A has communality identically 1 (no information). Fixture A2 with a deterministic stub that
        # scales sigma^2_B makes the identified total, hence R_B and communality_B, vary across replicates.
        nb = 41; cs = [0.2 + 0.1 * b for b in 1:nb]
        ctr = Ref(0)
        stub = (yb; kw...) -> _scaled_diag(fitd, cs[ctr[] += 1])
        r = _quiet(() -> _bsig(fitd; y = y, what = [:Sigma, :R, :communality], n_boot = nb, conf = 0.9, seed = 1,
                               progress = false, _refit = stub))
        tots = [_diag_total(fitd.pars, c) for c in cs]
        dL = diag(fitd.pars.Λ * fitd.pars.Λ')
        MC = reduce(hcat, [dL ./ diag(T) for T in tots])
        lo = r.ci_lower.communality_B; hi = r.ci_upper.communality_B
        @test lo ≈ [quantile(MC[i, :], 0.05) for i in 1:p] rtol = 1e-10
        @test hi ≈ [quantile(MC[i, :], 0.95) for i in 1:p] rtol = 1e-10
        @test all(lo .< hi .- 1e-4)                              # communality actually varies; tails not swapped
        @test r.n_effective.communality_B == fill(nb, p)
        MR = reduce(hcat, [vec(_cor(T)) for T in tots])
        loR = r.ci_lower.R_B; hiR = r.ci_upper.R_B
        @test vec(loR) ≈ [quantile(MR[i, :], 0.05) for i in 1:p^2] rtol = 1e-10
        @test vec(hiR) ≈ [quantile(MR[i, :], 0.95) for i in 1:p^2] rtol = 1e-10
        @test all(loR .<= hiR) && any(loR .< hiR .- 1e-4)         # R varies, tails not swapped
        MS = reduce(hcat, [vec(T) for T in tots])
        @test vec(r.ci_lower.Sigma_B) ≈ [quantile(MS[i, :], 0.05) for i in 1:p^2] rtol = 1e-10
        @test vec(r.ci_upper.Sigma_B) ≈ [quantile(MS[i, :], 0.95) for i in 1:p^2] rtol = 1e-10
        # the same varying fixture, with every 4th replicate failed: survivors only, and n_effective counts them
        ctr2 = Ref(0)
        flaky = function (yb; kw...)
            b = (ctr2[] += 1)
            f = _scaled_diag(fitd, cs[b])
            return b % 4 == 0 ? GllvmFit(f.model, f.pars, f.logLik, 1, false, f.optim_result, 0.0) : f
        end
        rf = _quiet(() -> _bsig(fitd; y = y, what = [:communality], n_boot = nb, conf = 0.9, seed = 1, progress = false, _refit = flaky))
        keep = [b for b in 1:nb if b % 4 != 0]
        @test rf.n_failed == nb - length(keep)
        @test rf.ci_lower.communality_B ≈ [quantile(MC[i, keep], 0.05) for i in 1:p] rtol = 1e-10
        @test rf.ci_upper.communality_B ≈ [quantile(MC[i, keep], 0.95) for i in 1:p] rtol = 1e-10
        @test rf.n_effective.communality_B == fill(length(keep), p)
    end

    @testset "15. rr-only communality_B is exactly 1 with a zero-width interval, flagged by @info" begin
        r, lg = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 1, progress = false))
        @test r.point_est.communality_B == ones(p)
        @test r.ci_lower.communality_B == ones(p) && r.ci_upper.communality_B == ones(p)   # vacuous, pinned on purpose
        @test r.n_effective.communality_B == fill(NB, p)
        @test any(occursin("communality_B is exactly 1", m) for m in _infos(lg))
        # no such message when communality is not requested, or for a has_diag fit (informative communality)
        _, lg2 = _logged(() -> _bsig(fit; y = y, what = [:Sigma], n_boot = NB, seed = 1, progress = false))
        @test !any(occursin("exactly 1", m) for m in _infos(lg2))
        rd, lgd = _logged(() -> _bsig(fitd; y = y, what = [:communality], n_boot = NB, seed = 1, progress = false,
                                      _refit = (yb; kw...) -> fitd))
        @test !any(occursin("exactly 1", m) for m in _infos(lgd))
        @test all(c -> 0 < c < 1, rd.point_est.communality_B)
    end

    @testset "16. has_diag real simulate-and-refit route and data-size priority" begin
        # real refit keyword passthrough (has_diag, K_W, K_phy, X) and the has_diag simulator run end to end
        rr, lgr = _logged(() -> _bsig(fitd; y = y, what = [:Sigma, :communality], n_boot = 6, seed = 3, progress = false))
        @test rr.n_boot == 6
        @test rr.n_failed == 0                  # probe 2026-11-01: seeds 1-3 gave 0 failures on this well-posed fixture
        @test rr.point_est.Sigma_B ≈ sigma_y_site(fitd) rtol = 1e-10
        @test all(rr.ci_lower.Sigma_B .<= rr.ci_upper.Sigma_B) && all(isfinite, rr.ci_lower.Sigma_B)
        @test all(0 .< rr.ci_lower.communality_B .<= rr.ci_upper.communality_B .< 1)
        @test rr.n_effective.communality_B == fill(6, p)
        # priority: n_sites > X > y for the number of simulated columns
        f0, l0 = _bs_refit()
        _quiet(() -> _bsig(fit; y = y, n_sites = 7, what = :Sigma, n_boot = 3, seed = 1, _refit = f0, progress = false))
        @test all(size(Y, 2) == 7 for Y in l0.ys)
        f1, l1 = _bs_refit()
        _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, _refit = f1, progress = false))
        @test all(size(Y, 2) == n for Y in l1.ys)
    end

    @testset "17. InterruptException is never swallowed; other refit errors are failed replicates" begin
        @test_throws InterruptException _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                                                           _refit = (yb; kw...) -> throw(InterruptException())))
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                               _refit = (yb; kw...) -> error("boom")))
        @test r.n_failed == 3
    end

    @testset "18. refit receives the fitted structure (K, K_B/K_W, center, X); mean path" begin
        # GllvmFit with K = 2 (every other fixture has K = 1, so a hard-coded K = 1 went unseen)
        r2 = MersenneTwister(18)
        p6, n6 = 6, 300
        L2 = [0.9 0.0; 0.6 0.5; 0.4 -0.6; -0.5 0.4; 0.3 0.7; 0.7 -0.3]
        y6k2 = L2 * randn(r2, 2, n6) + 0.5 * randn(r2, p6, n6)
        f2 = fit_gaussian_gllvm(y6k2; K = 2)
        @test f2.model.K == 2 && f2.converged
        kwlog = Any[]
        _quiet(() -> _bsig(f2; y = y6k2, what = :Sigma, n_boot = 2, seed = 1, progress = false,
                           _refit = (yb; kw...) -> (push!(kwlog, (; kw...)); f2)))
        @test all(k.K == 2 && k.K_W == 0 && k.has_diag == false for k in kwlog)
        empty!(kwlog)
        _quiet(() -> _bsig(fitd; y = y, what = :Sigma, n_boot = 2, seed = 1, progress = false,
                           _refit = (yb; kw...) -> (push!(kwlog, (; kw...)); fitd)))
        @test all(k.K == 1 && k.has_diag == true for k in kwlog)

        # TwoLevelFit with K_B != K_W (test 13 has K_B == K_W == 1, so a swap went unseen)
        ΛB = [0.9 0.1; 0.6 -0.3; -0.4 0.5]; σB = [0.30, 0.20, 0.25]
        ΛW = reshape([0.3, -0.2, 0.5], 3, 1); σW = [0.60, 0.50, 0.40]
        tk = TwoLevelFit(ΛB, σB, ΛW, σW, Matrix(ΛB * ΛB' + Diagonal(σB)), Matrix(ΛW * ΛW' + Diagonal(σW)), 30, -1.0, true, 1)
        tlog = Any[]; tstub = (yb, i; kw...) -> (push!(tlog, (; kw...)); tk)
        _quiet(() -> _bsig(tk; individual = _IND, what = :Sigma, n_boot = 2, seed = 1, progress = false, _refit = tstub))
        @test all(k.K_B == 2 && k.K_W == 1 && k.center == true for k in tlog)
        empty!(tlog)
        _quiet(() -> _bsig(tk; individual = _IND, what = :Sigma, n_boot = 2, seed = 1, progress = false, center = false, _refit = tstub))
        @test all(k.center == false for k in tlog)

        # fitted covariate: refit gets X, datasets carry the fitted mean X*beta, no X -> refusal
        rX = MersenneTwister(77); pX, nX = 4, 200
        Xc = randn(rX, pX, nX, 1)
        yX = 2.0 .* Xc[:, :, 1] .+ [0.9, 0.6, 0.4, -0.5] * randn(rX, 1, nX) .+ 0.5 * randn(rX, pX, nX)
        fx = fit_gaussian_gllvm(yX; K = 1, X = Xc)
        @test fx.converged && length(fx.pars.β) == 1 && abs(fx.pars.β[1] - 2.0) < 0.3
        @test_throws ArgumentError _bsig(fx; n_sites = nX, what = :Sigma, n_boot = 2, progress = false, _refit = (yb; kw...) -> fx)
        ys = Matrix{Float64}[]; kwlog = Any[]
        xstub = (yb; kw...) -> (push!(ys, copy(yb)); push!(kwlog, (; kw...)); fx)
        _quiet(() -> _bsig(fx; X = Xc, what = :Sigma, n_boot = 2, seed = 1, progress = false, _refit = xstub))
        @test all(haskey(k, :X) && k.X !== nothing && size(k.X) == (pX, nX, 1) for k in kwlog)
        μX = fx.pars.β[1] .* Xc[:, :, 1]
        @test cor(vec(ys[1]), vec(μX)) > 0.7                       # mean is carried (zero-mean sim gives ~0)
        @test maximum(abs, mean(ys[1] .- μX, dims = 2)) < 0.3
        # priority: X (hence its n) beats y for the number of simulated columns
        empty!(ys)
        _quiet(() -> _bsig(fx; y = yX[:, 1:11], X = Xc, what = :Sigma, n_boot = 2, seed = 1, progress = false, _refit = xstub))
        @test all(size(Y, 2) == nX for Y in ys)

        # the common user route: fit_gllvm(Y; family = Normal()) estimates trait intercepts (mean_design)
        ri = MersenneTwister(5)
        yi = [5.0, -3.0, 0.0, 10.0] .+ [0.9, 0.6, 0.4, -0.5] * randn(ri, 1, 150) .+ 0.5 * randn(ri, 4, 150)
        fi = GLLVModels.fit_gllvm(yi; family = GLLVModels.Normal(), K = 1)
        ys = Matrix{Float64}[]; kwlog = Any[]
        istub = (yb; kw...) -> (push!(ys, copy(yb)); push!(kwlog, (; kw...)); fi)
        _quiet(() -> _bsig(fi; n_sites = 20000, what = :Sigma, n_boot = 1, seed = 2, progress = false, _refit = istub))
        @test maximum(abs, vec(mean(ys[1], dims = 2)) .- fi.pars.β) < 0.05   # simulated mean = fitted intercepts
        @test maximum(abs, fi.pars.β) > 3                                    # control: intercepts are large
        @test all(k.X !== nothing && size(k.X) == (4, 20000, 4) for k in kwlog)
    end

    @testset "19. extraction errors, attrition element choice, guards, progress=false, Inf draws" begin
        # an extraction error on a CONVERGED refit blanks that replicate to NaN and is NOT counted failed
        bad_extract(fb) = GllvmFit(fb.model, merge(fb.pars, (Λ = nothing,)), fb.logLik, fb.n_iter, true, fb.optim_result, fb.cputime)
        j = Ref(0)
        ex = (yb; kw...) -> ((j[] += 1) % 4 == 0 ? bad_extract(fit_gaussian_gllvm(yb; kw...)) : fit_gaussian_gllvm(yb; kw...))
        re = _quiet(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 5, keep_draws = true, progress = false, _refit = ex))
        @test re.n_failed == 0
        @test count(d -> all(isnan, d.Sigma_B), re.draws) == 3
        @test re.n_effective.communality_B == fill(NB - 3, p)
        # attrition takes the MINIMUM over elements: NaN in element 3 (not 1) must still warn
        j3 = Ref(0)
        nan3 = function (yb; kw...)
            j3[] += 1
            fb = fit_gaussian_gllvm(yb; kw...)
            j3[] % 4 == 0 || return fb
            Λn = copy(fb.pars.Λ); Λn[3, 1] = NaN
            GllvmFit(fb.model, merge(fb.pars, (Λ = Λn,)), fb.logLik, fb.n_iter, true, fb.optim_result, fb.cputime)
        end
        r3, lg3 = _logged(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = NB, seed = 5, progress = false, _refit = nan3))
        @test r3.n_effective.communality_B[3] == NB - 3 && r3.n_effective.communality_B[1] == NB
        @test any(occursin("attrition", m) for m in _warns(lg3))
        # R's .safe_icc_ratio / communality guard: NaN for a non-positive or non-finite denominator
        # (a zero denominator alone cannot tell the guard from 0/0 = NaN)
        Σb = Matrix(Diagonal([-1.0, Inf, 1.0])); Σw = Matrix(0.5I(3))
        tg = TwoLevelFit(reshape([0.5, 0.5, 0.5], 3, 1), [0.1, 0.1, 0.1], reshape([0.3, 0.3, 0.3], 3, 1), [0.1, 0.1, 0.1], Σb, Σw, 30, 0.0, true, 0)
        rg = _quiet(() -> _bsig(_mk2(); individual = _IND, what = [:communality, :ICC], n_boot = 2, seed = 1, keep_draws = true,
                                progress = false, _refit = (yb, i; kw...) -> tg))
        d1 = rg.draws[1]
        @test isnan(d1.ICC_site[1]) && isnan(d1.ICC_site[2]) && isfinite(d1.ICC_site[3])
        @test isnan(d1.communality_B[1]) && isnan(d1.communality_B[2]) && isfinite(d1.communality_B[3])
        @test rg.n_effective.ICC_site == [0, 0, 2] && rg.n_effective.communality_B == [0, 0, 2]
        # a refit returning the wrong class is a failed replicate, not an uncaught FieldError
        rw = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false, _refit = (yb; kw...) -> _NotAFit()))
        @test rw.n_failed == 3
        # progress = false is silent
        _, lgq = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false, _refit = (yb; kw...) -> fit))
        @test !any(occursin("replicate", m) for m in _infos(lgq))
        # DOCUMENTED divergence from R (docstring): a non-finite draw is dropped. Pinned so a change is deliberate.
        infit(fb) = GllvmFit(fb.model, merge(fb.pars, (Λ = [Inf; fb.pars.Λ[2:end, :]],)), fb.logLik, 1, true, fb.optim_result, 0.0)
        ri = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false, _refit = (yb; kw...) -> infit(fit)))
        @test isnan(ri.ci_lower.Sigma_B[1, 1])
    end

    @testset "20. TwoLevelFit grouping is by label: shuffled columns, unbalanced groups, string ids" begin
        t2 = _mk2()
        nind = 4000; rg = MersenneTwister(31)
        sizes = rand(rg, 2:3, nind)
        labels = String[]; for g in 1:nind, _ in 1:sizes[g]; push!(labels, "id" * string(g)); end
        perm = randperm(rg, length(labels)); lab = labels[perm]
        t2big = TwoLevelFit(t2.Λ_B, t2.σ²_B, t2.Λ_W, t2.σ²_W, t2.Σ_B, t2.Σ_W, nind, 0.0, true, 0)
        Ys = Matrix{Float64}[]; inds = Any[]
        _quiet(() -> _bsig(t2big; individual = lab, what = :Sigma, n_boot = 1, seed = 6, progress = false,
                           _refit = (yb, i; kw...) -> (push!(Ys, copy(yb)); push!(inds, i); t2big)))
        Yb = Ys[1]; @test size(Yb, 2) == length(lab) && inds[1] == lab
        cols = Dict{String, Vector{Int}}(); for (j, l) in enumerate(lab); push!(get!(cols, l, Int[]), j); end
        same = sum(Yb[:, c[1]] * Yb[:, c[2]]' for c in values(cols)) ./ nind
        @test maximum(abs, same .- t2.Σ_B) < 0.08                              # same label share the B effect
        other = [Yb[:, 1:1000] * Yb[:, 1001:2000]' ./ 1000][1]                 # arbitrary other-column pairs
        @test maximum(abs, same .- t2.Σ_B) < maximum(abs, same .- zeros(3, 3)) # control: not independent
    end

    @testset "21. mutation-hardening: gaps found by the line-level mutation campaign" begin
        # (a) the DEFAULT progress = true emits one @info per replicate (every other test passes progress explicitly)
        _, lgd = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, _refit = (yb; kw...) -> fit))
        @test count(occursin("replicate", m) for m in _infos(lgd)) == 3
        @test any(occursin("replicate 2 / 3", m) for m in _infos(lgd))
        # (b) `center` defaults to true and is passed through to the TwoLevelFit refit
        f2 = _mk2(1.0, 1.7); seen = Any[]
        spy = (yb, i; kw...) -> (push!(seen, get(kw, :center, :absent)); f2)
        _quiet(() -> _bsig(f2; individual = _IND, n_boot = 2, seed = 1, progress = false, _refit = spy))
        _quiet(() -> _bsig(f2; individual = _IND, n_boot = 2, seed = 1, progress = false, center = false, _refit = spy))
        @test seen == Any[true, true, false, false]
        # (c) the n_boot lower bound is inclusive: n_boot = 1 is accepted and its single draw is both bounds
        r1 = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 1, seed = 1, progress = false,
                                _refit = (yb; kw...) -> _scaled(fit, 1.0)))
        @test r1.n_boot == 1 && r1.n_failed == 0
        @test r1.ci_lower.Sigma_B ≈ r1.point_est.Sigma_B rtol = 1e-12
        @test r1.ci_upper.Sigma_B ≈ r1.point_est.Sigma_B rtol = 1e-12
        # (d) the number of simulated sites comes from size(X, 2) when only X is given (X is p x n x q, n != p)
        X3 = randn(MersenneTwister(31), p, n, 1)
        fx = fit_gaussian_gllvm(y; K = 1, X = X3)
        fxr, lxr = _bs_refit()
        _quiet(() -> _bsig(fx; X = X3, what = :Sigma, n_boot = 3, seed = 1, _refit = fxr, progress = false))
        @test all(size(Y) == (p, n) for Y in lxr.ys)
        # (e) communality is NaN (never Inf or negative) when the tier total diagonal is <= 0
        ΛB = reshape([0.9, 0.6, -0.4], 3, 1)
        zt = TwoLevelFit(ΛB, [0.3, 0.2, 0.25], reshape([0.3, -0.2, 0.5], 3, 1), [0.6, 0.5, 0.4],
                         Matrix(Diagonal([0.0, -1.0, 1.0])), Matrix(Diagonal([0.7, 0.7, 0.7])), 30, 0.0, true, 0)
        rzt = _quiet(() -> _bsig(zt; individual = _IND, what = :communality, level = :B, n_boot = 3, seed = 1,
                                 progress = false, _refit = (yb, i; kw...) -> zt))
        @test isnan(rzt.point_est.communality_B[1]) && isnan(rzt.point_est.communality_B[2])
        @test rzt.point_est.communality_B[3] ≈ 0.16 rtol = 1e-12
        @test rzt.n_effective.communality_B == [0, 0, 3]
        @test !any(isinf, rzt.point_est.communality_B)
        # (f) TwoLevelFit batch: every individual is simulated (none left uninitialised), in the documented draw order
        ys2 = Matrix{Float64}[]
        _quiet(() -> _bsig(f2; individual = _IND, what = :Sigma, n_boot = 2, seed = 5, progress = false,
                           _refit = (yb, i; kw...) -> (push!(ys2, copy(yb)); f2)))
        rr2 = MersenneTwister(5)
        LBx = GLLVModels._psd_sqrt_factor(f2.Σ_B); LWx = GLLVModels._psd_sqrt_factor(f2.Σ_W)
        want = [begin
                    yb = zeros(3, length(_IND))
                    for g in 1:30
                        bi = LBx * randn(rr2, 3)
                        for j in (3g - 2):(3g)
                            yb[:, j] = bi .+ LWx * randn(rr2, 3)
                        end
                    end
                    yb
                end for _ in 1:2]
        @test ys2 == want
        # (g) InterruptException raised while EXTRACTING a converged replicate is never swallowed
        boomfit = (yb; kw...) -> GllvmFit(fit.model, merge(fit.pars, (Λ = _BsigBoom(p),)),
                                         fit.logLik, 1, true, fit.optim_result, 0.0)
        @test_throws InterruptException _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 2, seed = 1,
                                                           progress = false, _refit = boomfit))
        # (h) refusal / warning / info message text (cosmetic mutants: pin the user-facing wording)
        err(g) = try g(); nothing catch e; e end
        e = err(() -> _bsig(fit; y = y, what = Symbol[], n_boot = NB, progress = false))
        @test e isa ArgumentError && occursin("what is empty", e.msg)
        e = err(() -> _bsig(fit; y = y, n_boot = 0, progress = false))
        @test e isa ArgumentError && occursin("n_boot must be >= 1", e.msg)
        e = err(() -> _bsig(fit; y = y, level = :W, n_boot = NB, progress = false))
        @test e isa ArgumentError && occursin("Available: B", e.msg)
        e = err(() -> _bsig(f2; individual = repeat(1:10, inner = 3), progress = false))
        @test e isa ArgumentError && occursin("has 10 groups", e.msg) && occursin("nindiv = 30", e.msg)
        _, lg100 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 100, seed = 1, progress = false,
                                       _refit = (yb; kw...) -> fit), min_level = Logging.Warn)
        @test occursin("n_boot = 100 < 999", only(_warns(lg100)))
        _, lgr = _logged(() -> _bsig(fit; y = y, what = :communality, n_boot = 3, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> fit))
        @test any(occursin("(has_diag = false)", m) && occursin("refit with has_diag = true", m) for m in _infos(lgr))
        ctrb = Ref(0)
        _, lga = _logged(() -> _bsig(fit; y = y, what = :communality, n_boot = 12, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = (ctrb[] += 1) > 4)), min_level = Logging.Warn)
        @test any(occursin("66.7%", m) for m in _warns(lga))
    end
    @testset "22. hardening regressions: validation, reporting, shape guards, bounds, show" begin
        err(g) = try g(); nothing catch e; e end
        stub = (yb; kw...) -> fit
        # (a) input validation, each with a clear message and no refit attempted
        for (lab, kw, pat) in (
                ("n_sites = 1", (n_sites = 1,), "n_sites must be >= 2"),
                ("n_sites = 0", (n_sites = 0,), "n_sites must be >= 2"),
                ("n_sites = -3", (n_sites = -3,), "n_sites must be >= 2"),
                ("vector y", (y = vec(y),), "must be a p x n matrix"),
                ("transposed y", (y = permutedims(y),), "did you pass n x p"),
                ("y with 3 rows", (y = y[1:3, :],), "has 3 rows"),
                ("2-d X", (X = ones(1, n),), "p x n_sites x q"),
                ("X vs n_sites", (X = ones(p, 50, 1), n_sites = 20), "must agree"))
            e = err(() -> _bsig(fit; n_boot = NB, what = :Sigma, progress = false, _refit = stub, kw...))
            @test e isa ArgumentError && occursin(pat, e.msg)
        end
        e = err(() -> _bsig(fit; y = y, n_boot = NB, seed = -1, what = :Sigma, progress = false, _refit = stub))
        @test e isa ArgumentError && occursin("seed must be >= 0", e.msg)
        e = err(() -> _bsig(_mk2(); individual = [1], progress = false))
        @test e isa ArgumentError && occursin("too short", e.msg)
        e = err(() -> _bsig(_mk2(); individual = repeat(1:10, inner = 3), progress = false))
        @test e isa ArgumentError && occursin("one entry per observation", e.msg)
        # controls: the valid forms of the same calls run
        r = _quiet(() -> _bsig(fit; y = y, n_boot = 2, n_sites = 2, what = :Sigma, seed = 1, progress = false, _refit = stub))
        @test r.n_boot == 2
        # what = :ICC / :cross_corr on a GllvmFit: informative message, no empty parentheses
        for w in (:ICC, :cross_corr)
            e = err(() -> _bsig(fit; y = y, what = w, n_boot = NB, progress = false))
            @test e isa ArgumentError && !occursin("()", e.msg) && occursin("TwoLevelFit", e.msg)
        end

        # (c) total failure is announced whatever `what` is (matrix-only requests have no n_effective)
        for w in (:Sigma, [:Sigma, :R], :communality)
            r, lg = _logged(() -> _bsig(fit; y = y, what = w, n_boot = 5, seed = 1, progress = false,
                                        _refit = (yb; kw...) -> error("boom")), min_level = Logging.Warn)
            @test r.n_failed == 5
            @test any(occursin("5 of 5 replicates were lost", m) for m in _warns(lg))
        end
        # no warning when nothing is lost (control); ANY loss warns
        _, lg0 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 10, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> fit), min_level = Logging.Warn)
        @test !any(occursin("were lost", m) for m in _warns(lg0))
        _, lg2 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 10, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = false)), min_level = Logging.Warn)
        @test any(occursin("10 of 10 replicates were lost", m) for m in _warns(lg2))
        j = Ref(0)
        _, lg3 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 10, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = (j[] += 1) > 2)), min_level = Logging.Warn)
        @test any(occursin("2 of 10 replicates were lost", m) && occursin("2 failed refits", m) for m in _warns(lg3))
        j5 = Ref(0)
        _, lg5 = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 10, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = (j5[] += 1) > 1)), min_level = Logging.Warn)
        @test any(occursin("1 of 10 replicates were lost", m) for m in _warns(lg5))   # a single lost replicate warns

        # (b) extraction errors are not in n_failed (documented) but ARE reported as lost replicates
        bad_ex(fb) = GllvmFit(fb.model, merge(fb.pars, (Λ = nothing,)), fb.logLik, fb.n_iter, true, fb.optim_result, fb.cputime)
        re, lge = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 5, seed = 1, progress = false,
                                      _refit = (yb; kw...) -> bad_ex(fit)), min_level = Logging.Warn)
        @test re.n_failed == 0
        @test any(occursin("5 of 5 replicates were lost", m) && occursin("5 extraction errors", m) for m in _warns(lge))

        # (b) point-estimate NaN is structural, not attrition
        tg = TwoLevelFit(reshape([0.9, 0.6, -0.4], 3, 1), [0.3, 0.2, 0.25], reshape([0.3, -0.2, 0.5], 3, 1), [0.6, 0.5, 0.4],
                         Matrix(Diagonal([0.0, -1.0, 1.0])), Matrix(Diagonal([0.7, 0.7, 0.7])), 30, 0.0, true, 0)
        rg, lgg = _logged(() -> _bsig(tg; individual = _IND, what = :communality, level = :B, n_boot = 4, seed = 1,
                                      progress = false, _refit = (yb, i; kw...) -> tg), min_level = Logging.Warn)
        @test rg.n_effective.communality_B == [0, 0, 4]
        @test !any(occursin("attrition", m) for m in _warns(lgg))
        # control: a genuine loss on a finite entry still warns
        j4 = Ref(0)
        _, lgh = _logged(() -> _bsig(fit; y = y, what = :communality, n_boot = 12, seed = 1, progress = false,
                                     _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = (j4[] += 1) > 4)), min_level = Logging.Warn)
        @test any(occursin("attrition", m) for m in _warns(lgh))

        # (b) a replicate summary of the wrong shape is a NaN draw, never indexed into the percentiles
        w4 = TwoLevelFit([0.9, 0.6, 0.4, -0.5][:, :], fill(0.3, 4), [0.3, -0.2, 0.5, 0.1][:, :], fill(0.5, 4),
                         Matrix(1.0I(4)), Matrix(1.0I(4)), 30, 0.0, true, 0)
        rs = _quiet(() -> _bsig(_mk2(); individual = _IND, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                                _refit = (yb, i; kw...) -> w4))
        @test rs.n_failed == 0
        @test all(isnan, rs.ci_lower.Sigma_B) && all(isnan, rs.ci_upper.Sigma_W)
        @test size(rs.ci_lower.Sigma_B) == (3, 3)

        # (b) correlation bounds never leave [-1, 1]: K = 1 rr-only has |R| = 1 up to rounding
        for c in (0.3, 0.7, 1.1, 1.3, 2.9)
            rb = _quiet(() -> _bsig(fit; y = y, what = :R, n_boot = 9, seed = 1, progress = false,
                                    _refit = (yb; kw...) -> _scaled(fit, c)))
            @test all(abs.(rb.ci_lower.R_B) .<= 1.0) && all(abs.(rb.ci_upper.R_B) .<= 1.0)
        end
        cs = Iterators.Stateful(Iterators.cycle((0.37, 0.9, 1.13, 1.7, 2.1, 0.61)))
        rv = _quiet(() -> _bsig(fit; y = y, what = :R, n_boot = 30, seed = 1, progress = false,
                                _refit = (yb; kw...) -> _scaled(fit, popfirst!(cs))))
        @test all(abs.(rv.ci_lower.R_B) .<= 1.0) && all(abs.(rv.ci_upper.R_B) .<= 1.0)

        # (d) show: one-line form unchanged; text/plain lists estimates and intervals compactly
        rsh = _quiet(() -> _bsig(fit; y = y, what = [:Sigma, :communality], n_boot = 4, seed = 1, progress = false,
                                 _refit = (yb; kw...) -> _scaled(fit, 1.0; ok = (j[] += 1) % 2 == 0)))
        @test occursin("BootstrapSigmaR(level=", sprint(show, rsh)) && !occursin('\n', sprint(show, rsh))
        txt = sprint(show, MIME("text/plain"), rsh)
        @test occursin("Sigma_B", txt) && occursin("communality_B", txt) && occursin("n_effective", txt)
        @test occursin("replicates failed", txt) && occursin(" [", txt)
        @test occursin(string(round(rsh.point_est.Sigma_B[1, 1]; sigdigits = 4)), txt)
    end
end
