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
# WIRING: src/bootstrap_sigma_r.jl is include()d by src/GLLVModels.jl and this file is
# registered in test/runtests.jl (PR #1056). The guard below is a no-op safety net.

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
        @test _bsig_defined && hasmethod(GLLVModels.bootstrap_Sigma_r, Tuple{GllvmFit})
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
        # n_cores refusals (the threaded behaviour itself is in the n_cores testset at the end of the file)
        @test_throws ArgumentError _bsig(fit; y = y, n_boot = 4, n_cores = Threads.nthreads() + 1)
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

    @testset "10b. guard and result-shape gaps (mutation review 2026-11-02)" begin
        stubf = (yb; kw...) -> fit
        err(g) = try g(); nothing catch e; e end
        # more groups than nindiv is refused (the guard is `==`, not `>=`); fewer is covered in tests 12 and 21
        t30 = _mk2()
        e = err(() -> _bsig(t30; individual = repeat(1:31, inner = 3), n_boot = 2, seed = 1, progress = false,
                            _refit = (yb, i; kw...) -> t30))
        @test e isa ArgumentError && occursin("has 31 groups but the fit has nindiv = 30", e.msg)
        # control: exactly nindiv groups is accepted
        @test _quiet(() -> _bsig(t30; individual = _IND, n_boot = 2, seed = 1, progress = false,
                                 _refit = (yb, i; kw...) -> t30)).n_boot == 2
        # a fit whose only phylogenetic part is the unique variance (K_phy = 0) is refused too
        Σp = Matrix(0.5I(p)) + fill(0.3, p, p)
        fpu = fit_gaussian_gllvm(y; K = 1, K_phy = 0, has_phy_unique = true, Σ_phy = Σp)
        @test fpu.model.has_phy_unique && fpu.model.K_phy == 0
        e = err(() -> _bsig(fpu; y = y, n_boot = 2, seed = 1, progress = false, _refit = stubf))
        @test e isa ArgumentError && occursin("phylogenetic", e.msg)
        # boot_median is an empty NamedTuple by design (docstring), for both fit classes
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 1, progress = false, _refit = stubf))
        @test r.boot_median == NamedTuple() && isempty(r.boot_median)
        r2 = _quiet(() -> _bsig(t30; individual = _IND, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                                _refit = (yb, i; kw...) -> t30))
        @test r2.boot_median == NamedTuple()
        # coverage_ceiling = (n_boot - 1) / (n_boot + 1) for a TwoLevelFit as well
        @test r2.coverage_ceiling ≈ 2 / 4
        r3 = _quiet(() -> _bsig(t30; individual = _IND, what = :Sigma, n_boot = 9, seed = 1, progress = false,
                                _refit = (yb, i; kw...) -> t30))
        @test r3.coverage_ceiling ≈ 8 / 10
    end

    @testset "11. simulator draws have the fitted covariance (empirical, large n)" begin
        # GllvmFit: batch covariance within MC tolerance of ΛΛ' + σ_eps^2 I (hand-built, not _derived_site_cov)
        ys = Matrix{Float64}[]
        _quiet(() -> _bsig(fit; n_sites = 20000, what = :Sigma, n_boot = 2, seed = 11, progress = false,
                           _refit = (yb; kw...) -> (push!(ys, copy(yb)); fit)))
        A = fit.pars.Λ * fit.pars.Λ' + fit.pars.σ_eps^2 * I
        Yall = hcat(ys...)
        @test maximum(abs, cov(Yall') .- A) < 0.02                 # clean max ~0.004; a +2% scale (+4% variance) gives ~0.034
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

    @testset "11b. real-refit calibration: replicate Sigma_B is centred on the fitted Sigma_B (not on a rescaled one)" begin
        # built-in refit (no stub), n_sites = 2000 so the finite-sample bias of the MLE is negligible.
        # Measured 2026-11-02: clean trace ratio - 1 about +0.01 at n_boot = 30; a +2% simulator scale gives about +0.04.
        rc = _quiet(() -> _bsig(fit; n_sites = 2000, what = :Sigma, n_boot = 200, seed = 3, keep_draws = true, progress = false))
        @test rc.n_failed == 0
        tr_draws = mean(tr(d.Sigma_B) for d in rc.draws)
        @test abs(tr_draws / tr(rc.point_est.Sigma_B) - 1) < 0.02
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
        @test maximum(abs, same .- t2.Σ_B) < maximum(abs, t2.Σ_B)             # control: the tolerance is far below the signal
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
        @test occursin("Sigma_B", txt) && occursin("communality_B", txt) && occursin("n_effective = ", txt)
        @test occursin("replicates failed", txt) && occursin(r"\[1,1\] \S+ \[\S+, \S+\]", txt)
        @test occursin(string(round(rsh.point_est.Sigma_B[1, 1]; sigdigits = 4)), txt)
    end
    @testset "23. survivors of the independent mutation review (PR #1059 review)" begin
    stub = (yb; kw...) -> fit
    err(g) = try g(); nothing catch e; e end
    @testset "S02 warn threshold boundary" begin
        _, lg = _logged(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 998, seed = 1, progress = false, _refit = stub), min_level = Logging.Warn)
        @test occursin("n_boot = 998 < 999", only(_warns(lg)))
    end
    @testset "S03 X must have exactly n_sites columns (both directions)" begin
        e = err(() -> _bsig(fit; X = ones(p, 10, 1), n_sites = 20, n_boot = 2, what = :Sigma, progress = false, _refit = stub))
        @test e isa ArgumentError && occursin("must agree", e.msg)
    end
    @testset "S04 show prints both bounds for matrix entries" begin
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 4, seed = 1, progress = false, _refit = (yb; kw...) -> _scaled(fit, 1.0)))
        @test occursin(r"\[1,1\] \S+ \[\S+, \S+\]", sprint(show, MIME("text/plain"), r))
    end
    @testset "S05 refusal order: fit structure before argument checks" begin
        fw = fit_gaussian_gllvm(y; K = 1, K_W = 1)
        e = err(() -> _bsig(fw; y = y, level = :nonsense, progress = false, _refit = stub))
        @test e isa ArgumentError && occursin("K_W", e.msg)
    end
    @testset "S06 +-Inf draws are dropped" begin
        infit = GllvmFit(fit.model, merge(fit.pars, (Λ = [Inf; fit.pars.Λ[2:end, :]],)), fit.logLik, 1, true, fit.optim_result, 0.0)
        r = _quiet(() -> _bsig(fit; y = y, what = :communality, n_boot = 3, seed = 1, progress = false, keep_draws = true, _refit = (yb; kw...) -> infit))
        @test r.n_effective.communality_B[1] == 0
        ok = _scaled(fit, 1.0)
        c = Ref(0)
        r2 = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 5, seed = 1, progress = false,
                                _refit = (yb; kw...) -> (c[] += 1) == 1 ? infit : _scaled(fit, 1.0 + c[] / 10)))
        @test isfinite(r2.ci_upper.Sigma_B[1, 1]) && isfinite(r2.ci_lower.Sigma_B[1, 1])
    end
    @testset "S07/S08 built-in refit: optimizer settings only for has_diag" begin
        # same data as the driver's batch, refit directly with / without the extra kwargs
        ys = Matrix{Float64}[]
        _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 9, progress = false, _refit = (yb; kw...) -> (push!(ys, copy(yb)); fit)))
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 3, seed = 9, progress = false, keep_draws = true))
        for (b, yb) in enumerate(ys)
            d = fit_gaussian_gllvm(yb; K = 1, K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false, Σ_phy = nothing, X = nothing)
            @test r.draws[b].Sigma_B == GLLVModels._r_tier_total(d, :unit)   # bit-identical: no extra kwargs
        end
        ysd = Matrix{Float64}[]
        _quiet(() -> _bsig(fitd; y = y[:, 1:8], what = :Sigma, n_boot = 6, seed = 9, progress = false, _refit = (yb; kw...) -> (push!(ysd, copy(yb)); fitd)))
        rd = _quiet(() -> _bsig(fitd; y = y[:, 1:8], what = :Sigma, n_boot = 6, seed = 9, progress = false, keep_draws = true))
        for (b, yb) in enumerate(ysd)
            d = fit_gaussian_gllvm(yb; K = 1, K_W = 0, has_diag = true, K_phy = 0, has_phy_unique = false, Σ_phy = nothing, X = nothing, iterations = 2000, f_tol = 1e-12)
            exp = d.converged ? sigma_y_site(d) : fill(NaN, p, p)
            @test isequal(rd.draws[b].Sigma_B, exp)
        end
    end
    @testset "S09 seed = 0 is valid" begin
        r = _quiet(() -> _bsig(fit; y = y, what = :Sigma, n_boot = 2, seed = 0, progress = false, _refit = stub))
        @test r.n_boot == 2
    end
    @testset "S10 ICC: zero denominator with nonzero numerator is NaN, not Inf" begin
        Σb = Matrix(Diagonal([1.0, 2.0, 3.0])); Σw = Matrix(Diagonal([-1.0, -2.0, 1.0]))
        tg = TwoLevelFit(reshape([0.5, 0.5, 0.5], 3, 1), [0.1, 0.1, 0.1], reshape([0.3, 0.3, 0.3], 3, 1), [0.1, 0.1, 0.1], Σb, Σw, 30, 0.0, true, 0)
        r = _quiet(() -> _bsig(tg; individual = _IND, what = :ICC, n_boot = 2, seed = 1, progress = false, _refit = (yb, i; kw...) -> tg))
        @test isnan(r.point_est.ICC_site[1]) && isnan(r.point_est.ICC_site[2]) && r.point_est.ICC_site[3] ≈ 3 / 4
    end
    @testset "S11 attrition message names the FIRST of tied entries" begin
        c = Ref(0)
        _, lg = _logged(() -> _bsig(_mk2(); individual = _IND, n_boot = 12, seed = 1, progress = false,
                                    _refit = (yb, i; kw...) -> _mk2(1.0, 1.0; ok = (c[] += 1) > 4)), min_level = Logging.Warn)
        @test any(occursin("Summary communality_B kept only", m) for m in _warns(lg))
    end
    @testset "S12 duplicate `what` entries are collapsed" begin
        r = _quiet(() -> _bsig(fit; y = y, what = [:Sigma, :Sigma], n_boot = 2, seed = 1, progress = false, _refit = stub))
        @test r.what == [:Sigma] && collect(keys(r.point_est)) == [:Sigma_B]
    end
    end
    @testset "24. review follow-ups: optimizer kwargs seen by the seam, clamp scope, wrong-shape count" begin
        # a user _refit keeps its own controls: the driver injects no optimizer options into the seam
        kwa = Any[]
        for ft in (fitd, fit)
            empty!(kwa)
            _quiet(() -> _bsig(ft; y = y, what = :Sigma, n_boot = 2, seed = 1, progress = false,
                               _refit = (yb; kw...) -> (push!(kwa, (; kw...)); ft)))
            @test !any(haskey(k, :iterations) || haskey(k, :f_tol) for k in kwa) && length(kwa) == 2
        end
        # a genuine |R| > 1 is NOT hidden by the rounding clamp
        sb = [1.0 2.0 0.1; 2.0 1.0 0.1; 0.1 0.1 1.0]
        tbad = TwoLevelFit(reshape([0.5, 0.5, 0.5], 3, 1), [0.1, 0.1, 0.1], reshape([0.3, 0.3, 0.3], 3, 1), [0.1, 0.1, 0.1],
                           sb, Matrix(0.5I(3)), 30, 0.0, true, 0)
        rbad = _quiet(() -> _bsig(tbad; individual = _IND, what = :R, level = :B, n_boot = 3, seed = 1, progress = false,
                                  _refit = (yb, i; kw...) -> tbad))
        @test rbad.point_est.R_B[1, 2] ≈ 2.0 && rbad.ci_lower.R_B[1, 2] ≈ 2.0 && rbad.ci_upper.R_B[1, 2] ≈ 2.0
        # a wrong-shaped summary is a NaN draw AND is counted as lost (not in n_failed)
        w4 = TwoLevelFit([0.9, 0.6, 0.4, -0.5][:, :], fill(0.3, 4), [0.3, -0.2, 0.5, 0.1][:, :], fill(0.5, 4),
                         Matrix(1.0I(4)), Matrix(1.0I(4)), 30, 0.0, true, 0)
        rw, lgw = _logged(() -> _bsig(_mk2(); individual = _IND, what = :Sigma, n_boot = 3, seed = 1, progress = false,
                                      _refit = (yb, i; kw...) -> w4), min_level = Logging.Warn)
        @test rw.n_failed == 0 && any(occursin("3 of 3 replicates were lost", m) && occursin("3 extraction errors", m) for m in _warns(lgw))
        # n_sites between 2 and p: accepted, and the all-NaN outcome is announced
        rn, lgn = _logged(() -> _bsig(fit; n_sites = 3, what = :Sigma, n_boot = 3, seed = 1, progress = false), min_level = Logging.Warn)
        @test size(rn.ci_lower.Sigma_B) == (p, p) && rn.n_boot == 3
        # 3 sites < p = 4: the refit of every replicate is not converged, so all 3 are failed, every bound is NaN, and
        # the loss is announced (measured on this fixture and seed 2026-11-02)
        @test rn.n_failed == 3 && all(isnan, rn.ci_lower.Sigma_B) && all(isnan, rn.ci_upper.Sigma_B)
        @test any(occursin("3 of 3 replicates were lost", m) && occursin("3 failed refits", m) for m in _warns(lgn))
    end
    @testset "25. has_diag K=2 built-in refit reaches the optimum, not the 500-iteration cap" begin
        # Replicate 65 of bootstrap seed 101 on the has_diag K=2 fixture (n = 200 units, p = 6, trait intercepts).
        # R gllvmTMB on the same data gives logLik -1422.97095347 (convergence 0); the stock Julia fit stops at
        # the 500-iteration cap with logLik -1422.97217 (recorded 2026-11-01). (CSV parsed by hand: no extra test dependency.)
        rows = [parse.(Float64, split(l, ',')) for l in readlines(joinpath(@__DIR__, "data", "bsig_c_s101_b65.csv")) if !isempty(strip(l))]
        Yb = Matrix{Float64}(reduce(hcat, rows))                     # p x n
        nb_ = size(Yb, 2); Xb = zeros(6, nb_, 6)
        for t in 1:6; Xb[t, :, t] .= 1.0; end
        f0 = _quiet(() -> fit_gaussian_gllvm(Yb; K = 2, has_diag = true, X = Xb))
        @test !f0.converged && f0.n_iter == 500                      # pins the cause (stock cap)
        _, refit = GLLVModels._bsig_setup(f0, 1, nothing, nb_, Xb, nothing)
        fb = _quiet(() -> refit(Yb))                                  # the driver's built-in refit
        # Not gated on fb.converged: the refit needs ~1750 of its 2000 iterations on a flat ridge, so whether the
        # relative f_tol test fires before the cap depends on the platform's floating-point path (observed to fail
        # intermittently on Julia 1.13 CI, never on 1.10 or on Totoro with Julia 1.12). What the testset pins is that
        # the built-in refit options are active (n_iter beyond the stock 500) and that the fit reaches R's optimum.
        @test fb.n_iter > f0.n_iter
        @test fb.logLik >= f0.logLik
        @test isapprox(fb.logLik, -1422.97095347239; atol = 1e-4)     # the external (R) optimum
    end

    @testset "R_* clamp tolerance: 1e-9 overshoot is clamped, 1e-6 stays visible" begin
        # src/bootstrap_sigma_r.jl clamps R_* bounds to [-1, 1] only for a rounding overshoot (<= 1 + 1e-8).
        # Hand-built TwoLevelFit whose Σ_B has unit variances and off-diagonal 1 + δ, so R_B[1, 2] = 1 + δ
        # exactly as the driver reads it; every replicate is the same fit, so both bounds equal that value.
        function overshoot(δ)
            Σb = [1.0 1.0+δ 0.0; 1.0+δ 1.0 0.0; 0.0 0.0 1.0]
            tg = TwoLevelFit(reshape([0.5, 0.5, 0.5], 3, 1), [0.1, 0.1, 0.1], reshape([0.3, 0.3, 0.3], 3, 1),
                             [0.1, 0.1, 0.1], Σb, Matrix(0.5I(3)), 30, 0.0, true, 0)
            return _quiet(() -> _bsig(_mk2(); individual = _IND, what = [:R], n_boot = 5, seed = 1,
                                      progress = false, _refit = (yb, i; kw...) -> tg))
        end
        rs = overshoot(1e-9)                                  # rounding-sized: clamped to exactly 1
        @test rs.ci_lower.R_B[1, 2] == 1.0 && rs.ci_upper.R_B[1, 2] == 1.0
        rb = overshoot(1e-6)                                  # genuine excursion: left visible, not clamped
        @test isapprox(rb.ci_lower.R_B[1, 2], 1 + 1e-6; atol = 1e-12)
        @test isapprox(rb.ci_upper.R_B[1, 2], 1 + 1e-6; atol = 1e-12)
        @test rb.ci_upper.R_B[1, 2] > 1.0
        rn = overshoot(-2.0 - 1e-6)                           # mirror: -1 - 1e-6 stays visible below -1
        @test rn.ci_lower.R_B[1, 2] < -1.0
        rm9 = overshoot(-2.0 - 1e-9)                          # mirror rounding-sized: clamped to exactly -1
        @test rm9.ci_lower.R_B[1, 2] == -1.0 && rm9.ci_upper.R_B[1, 2] == -1.0
    end
end

# ============================================================================================
# PHY TIER, PrecisionMultivariateFit (BEGIN PHY-TIER). Target: R bootstrap_Sigma level = "phy"
# (gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9: R/bootstrap-sigma.R, redraw at
# R/methods-gllvmTMB.R). Expected values below are computed from raw fit fields and from the exact
# model covariance, never through the code under test. The evidence level of this tier is
# STRUCTURAL: no numerical comparison with an R run exists. The mutation runner extracts this
# region (header lines 1-81 plus everything from the BEGIN marker), so keep it self-contained.
# ============================================================================================
const _PG = GLLVModels

# trait-major complete mean design (row (o-1)*d + t), built independently of the package helper
_pmv_trait_design(d, m) = kron(ones(m, 1), Matrix{Float64}(I, d, d))

# Hand-built PrecisionMultivariateFit (no optimiser): converged, arbitrary but chosen parameters.
function _pmv_hand(phy, sid; beta, loading, unique, residual, mode = unique === nothing ? :barelowrank : :explicitunique,
                   residual_mode = :trait, D = nothing, ok = true)
    d = size(loading, 1); m = length(sid); rank = size(loading, 2)
    D = D === nothing ? _pmv_trait_design(d, m) : D
    names = Union{String, Symbol}[Symbol("b", j) for j in 1:length(beta)]
    PrecisionMultivariateFit(Vector{Float64}(beta), Matrix{Float64}(loading),
        unique === nothing ? nothing : Vector{Float64}(unique), Vector{Float64}(residual),
        mode, rank, phy, collect(Int, sid), zeros(1), -1.0, ok, 0.0, 1.0, true, 1.0, 1,
        :converged, zeros(d, m), Matrix{Float64}(D), (d, m), names, residual_mode, String[])
end
# copy of a fit with some fields replaced (deterministic stand-in for a refit)
_pmv_with(f; loading = f.loading, unique = f.phylo_unique_variance, residual = f.residual_variance,
          ok = f.converged) =
    PrecisionMultivariateFit(f.beta, loading, unique, residual, f.mode, size(loading, 2), f.phy,
        f.species_id, f.parameters, f.loglik, ok, f.gradient_norm, f.hessian_min_eigenvalue,
        f.hessian_positive_definite, f.hessian_condition_number, f.iterations, f.stopping_reason,
        f.response, f.mean_design, f.response_shape, f.coefficient_names, f.residual_mode,
        f.parameter_labels)

# Independent test-side data simulator (same template as test_precision_multivariate_fit.jl).
function _pmv_truth_data(rng, phy, sid; beta, loading, unique, residual)
    d = length(beta)
    F = cholesky(Symmetric(Matrix(phy.Q))).U
    g = F \ randn(rng, phy.n_aug, size(loading, 2)); h = F \ randn(rng, phy.n_aug, d)
    Y = Matrix{Float64}(undef, d, length(sid))
    for o in eachindex(sid), t in 1:d
        nd = phy.species_aug_id[sid[o]]
        Y[t, o] = beta[t] + sum(loading[t, k] * g[nd, k] for k in 1:size(loading, 2)) +
            (unique === nothing ? 0.0 : sqrt(unique[t]) * h[nd, t]) + sqrt(residual[t]) * randn(rng)
    end
    return Y
end

# exact model covariance of vec(Y) (trait fastest) and the mean vector of a PMV fit
function _pmv_exact_cov(f)
    d, m = f.response_shape
    Ainv = inv(Matrix(f.phy.Q))
    nodes = [f.phy.species_aug_id[s] for s in f.species_id]
    Ao = [Ainv[nodes[o], nodes[o2]] for o in 1:m, o2 in 1:m]
    LL = f.loading * f.loading'
    uu = f.phylo_unique_variance === nothing ? zeros(d) : f.phylo_unique_variance
    return kron(Ao, LL + Diagonal(uu)) + kron(Matrix{Float64}(I, m, m), Diagonal(f.residual_variance))
end

# records every refit call (keywords included) and returns a real refit on the data it was given
function _pmv_capture()
    log = (calls = NamedTuple[],)
    f = function (Y, phy; kw...)
        push!(log.calls, (Y = copy(Y), phy = phy, kw = Dict{Symbol, Any}(kw)))
        return _PG.fit_precision_multivariate(Y, phy; kw...)
    end
    return f, log
end

@testset "bootstrap_Sigma_r phy tier: PrecisionMultivariateFit (structural evidence only)" begin
    phy6 = PrecisionPhy(GLLVModels.random_balanced_tree(6; branch_length = 0.35))
    phy4 = PrecisionPhy(GLLVModels.random_balanced_tree(4; branch_length = 0.35))
    sid6 = repeat(1:6, inner = 4)
    sid4 = [1, 2, 3, 4, 1, 3]
    bt = [0.4, -0.3, 0.7]
    Ltrue = reshape([0.8, -0.5, 0.6], 3, 1)
    utrue = [0.30, 0.20, 0.25]; rtrue = [0.10, 0.12, 0.10]
    Y6 = _pmv_truth_data(MersenneTwister(20261102), phy6, sid6; beta = bt, loading = Ltrue,
                         unique = utrue, residual = rtrue)
    fpb = _quiet(() -> _PG.fit_precision_multivariate(Y6, phy6; rank = 1, mode = :barelowrank, species_id = sid6))
    fpu = _quiet(() -> _PG.fit_precision_multivariate(Y6, phy6; rank = 1, mode = :explicitunique, species_id = sid6))
    NB = 10
    stub(f) = (Y, phy; kw...) -> f                      # every replicate "refits" to f itself

    @testset "P0 fixtures; PMV is no longer 'unsupported'; a real refusal still is" begin
        @test fpb.converged && fpu.converged
        @test fpb.phylo_unique_variance === nothing && fpu.phylo_unique_variance !== nothing
        @test phy6.species_aug_id != collect(1:6) && phy4.species_aug_id != collect(1:4)  # non-identity node map
        @test hasmethod(GLLVModels.bootstrap_Sigma_r, Tuple{PrecisionMultivariateFit})
        r = _quiet(() -> _bsig(fpu; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = stub(fpu)))
        @test r isa GLLVModels.BootstrapSigmaR
        # paired control: other classes keep their refusal text
        e = try _bsig(_NotAFit(); progress = false); nothing catch ex; ex end
        @test e isa ArgumentError && occursin("unsupported fit class", e.msg)
    end

    @testset "P1 point estimates from raw fields: total = LL' + diag(unique), no residual; exact keys" begin
        cases = (
            ("barelowrank rank 2, trait residual",
             _pmv_hand(phy6, sid6; beta = bt, loading = [0.7 0.2; -0.4 0.5; 0.3 -0.6], unique = nothing,
                       residual = [0.3, 0.4, 0.5])),
            ("explicitunique rank 1, trait residual",
             _pmv_hand(phy6, sid6; beta = bt, loading = Ltrue, unique = utrue, residual = [1.5, 1.0, 2.0])),
            ("explicitunique rank 1, shared residual",
             _pmv_hand(phy6, sid6; beta = bt, loading = Ltrue, unique = utrue, residual = fill(0.9, 3),
                       residual_mode = :shared)))
        for (nm, f) in cases
            r = _quiet(() -> _bsig(f; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f)))
            L = f.loading
            u = f.phylo_unique_variance === nothing ? zeros(3) : f.phylo_unique_variance
            S = L * L' + Diagonal(u)
            @test collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
            @test r.level == [:phy]
            @test r.point_est.Sigma_phy ≈ S atol = 1e-12
            @test r.point_est.R_phy ≈ _cor(S) atol = 1e-12
            @test r.point_est.communality_phy ≈ vec(sum(abs2, L; dims = 2)) ./ diag(S) atol = 1e-12
            # every replicate refits to f, so both bounds equal the point estimate
            @test r.ci_lower.Sigma_phy ≈ S atol = 1e-12
            @test r.ci_upper.Sigma_phy ≈ S atol = 1e-12
            @test r.n_failed == 0
        end
    end

    @testset "P2 residual variance is excluded from Sigma_phy and communality_phy" begin
        f = _pmv_hand(phy6, sid6; beta = bt, loading = Ltrue, unique = utrue, residual = [1.5, 1.0, 2.0])
        r = _quiet(() -> _bsig(f; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f)))
        S = Ltrue * Ltrue' + Diagonal(utrue)
        withres = S + Diagonal(f.residual_variance)
        @test maximum(abs, withres - S) > 1                     # control: contamination is observable
        @test r.point_est.Sigma_phy ≈ S atol = 1e-12
        @test !isapprox(r.point_est.Sigma_phy, withres; atol = 0.5)
        @test r.point_est.communality_phy ≈ vec(Ltrue .^ 2) ./ diag(S) atol = 1e-12
        @test !isapprox(r.point_est.communality_phy, vec(Ltrue .^ 2) ./ diag(withres); atol = 0.05)
    end

    @testset "P1c barelowrank: communality_phy is exactly 1, zero width, flagged by @info" begin
        f = _pmv_hand(phy6, sid6; beta = bt, loading = [0.7 0.2; -0.4 0.5; 0.3 -0.6], unique = nothing,
                      residual = [0.3, 0.4, 0.5])
        r, logs = _logged(() -> _bsig(f; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f)))
        @test r.point_est.communality_phy == ones(3)
        @test r.ci_lower.communality_phy == ones(3) && r.ci_upper.communality_phy == ones(3)
        @test any(occursin("communality_phy is exactly 1", m) for m in _infos(logs))
        # control: with a unique component there is no such flag and communality < 1
        g = _pmv_hand(phy6, sid6; beta = bt, loading = Ltrue, unique = utrue, residual = rtrue)
        r2, logs2 = _logged(() -> _bsig(g; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(g)))
        @test !any(occursin("communality_phy is exactly 1", m) for m in _infos(logs2))
        @test all(<(1), r2.point_est.communality_phy)
        # not requested -> not flagged
        _, logs3 = _logged(() -> _bsig(f; level = :phy, what = :Sigma, n_boot = 4, seed = 1, progress = false, _refit = stub(f)))
        @test !any(occursin("communality_phy is exactly 1", m) for m in _infos(logs3))
    end

    @testset "P3 simulator law: exact moments against the model covariance" begin
        # d = 2 traits, 4 tips, 6 observations (tips 1 and 3 repeated), covariate-bearing mean design
        m4 = length(sid4)
        x = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5]
        D = zeros(2 * m4, 3)
        for o in 1:m4
            D[(o - 1) * 2 + 1, 1] = 1.0; D[(o - 1) * 2 + 2, 2] = 1.0
            D[(o - 1) * 2 + 1, 3] = x[o]; D[(o - 1) * 2 + 2, 3] = x[o]
        end
        cases = (
            ("rank 1, explicitunique, unequal per-trait residuals",
             _pmv_hand(phy4, sid4; beta = [0.4, -0.3, 0.6], loading = reshape([0.9, -0.6], 2, 1),
                       unique = [0.30, 0.55], residual = [0.50, 1.20], D = D)),
            ("rank 2, barelowrank, d = 3",
             _pmv_hand(phy4, sid4; beta = [0.2, 0.1, -0.2], loading = [0.8 0.3; -0.5 0.6; 0.4 -0.7],
                       unique = nothing, residual = [0.4, 0.9, 0.6])),
            ("shared residual, explicitunique, d = 3",
             _pmv_hand(phy4, sid4; beta = [0.0, 0.5, -0.5], loading = reshape([0.7, 0.4, -0.8], 3, 1),
                       unique = [0.35, 0.60, 0.25], residual = fill(0.8, 3), residual_mode = :shared)))
        N = 40000
        for (k, (nm, f)) in enumerate(cases)
            d, m = f.response_shape
            Ys = _PG._bsig_pmv_simulate_batch(MersenneTwister(31000 + k), f, N)
            @test length(Ys) == N && all(size(Y) == (d, m) for Y in Ys[1:5])
            S = reduce(hcat, (vec(Y) for Y in Ys))              # (d*m) x N
            K = _pmv_exact_cov(f)
            μ = f.mean_design * f.beta
            sm = vec(mean(S; dims = 2)); sc = cov(S; dims = 2)
            @test all(abs.(sm .- μ) .<= 4.5 .* sqrt.(diag(K) ./ N))
            tol = 4.5 .* sqrt.((diag(K) * diag(K)' .+ K .^ 2) ./ N)       # analytic Gaussian SE of a covariance entry
            bad = count(abs.(sc .- K) .> tol)
            @test bad == 0
            # control: the test has power (a tree-free covariance with the same diagonal is rejected)
            Kn = Diagonal(diag(K)) * 1.0
            @test count(abs.(sc .- Kn) .> tol) > 0
        end
    end

    @testset "P3c the exact covariance equals the fitter's own likelihood (ties the simulator to the fit)" begin
        for f0 in (_pmv_hand(phy6, sid6; beta = bt, loading = Ltrue, unique = utrue, residual = [0.4, 0.9, 0.6]),
                   _pmv_hand(phy6, sid6; beta = bt, loading = [0.7 0.2; -0.4 0.5; 0.3 -0.6], unique = nothing,
                             residual = [0.3, 0.4, 0.5]))
            d, m = f0.response_shape
            Yd = randn(MersenneTwister(5), d, m)
            K = _pmv_exact_cov(f0)
            res = vec(Yd .- reshape(f0.mean_design * f0.beta, d, m))
            dense = -0.5 * (logdet(cholesky(Symmetric(K))) + res' * (K \ res) + length(res) * log(2π))
            sparse_ll = _PG.multivariate_phylo_precision_loglik(Yd' .- reshape(f0.mean_design * f0.beta, d, m)',
                f0.phy, f0.loading, f0.residual_variance; sigma2_phy = 1.0,
                phylo_unique_variance = f0.phylo_unique_variance, species_id = f0.species_id)
            @test isapprox(dense, sparse_ll; atol = 1e-9, rtol = 1e-12)
        end
    end

    @testset "P4 refit receives the fitted structure; no warm start, no f_tol; reproduces the original fit" begin
        d = 3; m = length(sid6)
        xcov = [0.3 * mod(o, 5) - 0.6 for o in 1:m]
        X3 = zeros(d, m, d + 1)
        for t in 1:d; X3[t, :, t] .= 1.0; X3[t, :, d + 1] .= xcov; end
        Yx = Y6 .+ reshape(0.5 .* xcov', 1, m)
        names = ["a", "b", "c", "slope"]
        fx = _quiet(() -> _PG.fit_precision_multivariate(Yx, phy6; rank = 1, mode = :explicitunique,
                          residual_mode = :shared, species_id = sid6, X = X3, coefficient_names = names))
        for (nm, f) in (("covariate X, shared residual", fx), ("default trait means", fpu), ("barelowrank", fpb))
            cap, log = _pmv_capture()
            r = _quiet(() -> _bsig(f; level = :phy, n_boot = 3, seed = 5, progress = false, _refit = cap))
            @test length(log.calls) == 3
            for c in log.calls
                @test c.phy === f.phy                                        # the SAME tree object
                @test Set(keys(c.kw)) == Set([:rank, :mode, :residual_mode, :species_id, :X, :coefficient_names])
                @test c.kw[:rank] == f.rank && c.kw[:mode] === f.mode && c.kw[:residual_mode] === f.residual_mode
                @test c.kw[:species_id] == f.species_id
                @test c.kw[:X] == f.mean_design
                @test c.kw[:coefficient_names] == f.coefficient_names
            end
            # the kwargs reproduce the original fit exactly when given the original data
            kw = log.calls[1].kw
            f2 = _quiet(() -> _PG.fit_precision_multivariate(f.response, f.phy; kw...))
            @test maximum(abs, f2.parameters .- f.parameters) < 1e-10
            @test length(f2.beta) == length(f.beta) && f2.coefficient_names == f.coefficient_names
        end
        @test length(fx.beta) == d + 1                                      # the covariate design was really used
    end

    @testset "P5 exactly n_boot refits; one batch from one stream; seeds" begin
        cap, log = _pmv_capture()
        n_calls = Ref(0)
        counting = (Y, phy; kw...) -> (n_calls[] += 1; cap(Y, phy; kw...))
        _quiet(() -> _bsig(fpu; level = :phy, n_boot = 7, seed = 11, progress = false, _refit = counting))
        @test n_calls[] == 7 && length(log.calls) == 7
        # the responses handed to the refit are exactly the batch from MersenneTwister(seed)
        expect = _PG._bsig_pmv_simulate_batch(MersenneTwister(11), fpu, 7)
        @test all(log.calls[b].Y == expect[b] for b in 1:7)
        # same seed bit-identical, different seed different
        runb(seed) = _quiet(() -> _bsig(fpu; level = :phy, n_boot = 6, seed = seed, progress = false, keep_draws = true))
        ra, rb, rc = runb(3), runb(3), runb(4)
        @test all(ra.draws[b].Sigma_phy == rb.draws[b].Sigma_phy for b in 1:6)
        @test any(ra.draws[b].Sigma_phy != rc.draws[b].Sigma_phy for b in 1:6)
        # the whole batch is drawn BEFORE any refit: a refit that consumes the default RNG changes nothing
        Random.seed!(77)
        ys_default = Matrix{Float64}[]
        eater = (Y, phy; kw...) -> (push!(ys_default, copy(Y)); rand(); rand(); randn(); fpu)
        _quiet(() -> _bsig(fpu; level = :phy, n_boot = 5, progress = false, _refit = eater))
        Random.seed!(77)
        expect2 = _PG._bsig_pmv_simulate_batch(Random.default_rng(), fpu, 5)
        @test ys_default == expect2
    end

    @testset "P6 failure accounting: injected non-converged, throwing and wrong-class refits" begin
        L1 = Ltrue
        f0 = _pmv_hand(phy6, sid6; beta = bt, loading = L1, unique = utrue, residual = rtrue)
        c(b) = 0.5 + 0.1 * b
        function injected(bad_nonconv, bad_throw, bad_class)
            n = Ref(0)
            return function (Y, phy; kw...)
                n[] += 1; b = n[]
                b in bad_throw && error("injected refit failure")
                b in bad_class && return _NotAFit()
                return _pmv_with(f0; loading = c(b) .* L1, ok = !(b in bad_nonconv))
            end
        end
        nb = 20
        nonconv, thr, cls = [3, 12], [7], [15]
        r, logs = _logged(() -> _bsig(f0; level = :phy, n_boot = nb, seed = 1, progress = false,
                                      _refit = injected(nonconv, thr, cls)))
        @test r.n_failed == 4
        @test any(occursin("4 of $nb replicates were lost", w) for w in _warns(logs))
        surv = [b for b in 1:nb if !(b in vcat(nonconv, thr, cls))]
        vals = [c(b)^2 * L1[1]^2 + utrue[1] for b in surv]             # Sigma_phy[1,1] of each survivor
        @test r.ci_lower.Sigma_phy[1, 1] ≈ quantile(vals, 0.025) atol = 1e-12
        @test r.ci_upper.Sigma_phy[1, 1] ≈ quantile(vals, 0.975) atol = 1e-12
        @test r.n_effective.communality_phy == fill(length(surv), 3)
        @test !any(occursin("attrition", w) for w in _warns(logs))      # 16/20 = 80% retained: no attrition warning
        # attrition below 80%: 3 of 10 lost
        r2, logs2 = _logged(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false,
                                        _refit = injected([2], [5], [9])))
        @test r2.n_failed == 3
        @test any(occursin("attrition", w) && occursin("70.0%", w) for w in _warns(logs2))
        # control: no failure, no warning about lost replicates
        r3, logs3 = _logged(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false,
                                        _refit = injected(Int[], Int[], Int[])))
        @test r3.n_failed == 0 && !any(occursin("lost", w) for w in _warns(logs3))
        # a real refit stopped after one iteration is non-converged: a failure, not an exception
        nc = (Y, phy; kw...) -> _quiet(() -> _PG.fit_precision_multivariate(Y, phy; kw..., iterations = 1))
        r4 = _quiet(() -> _bsig(fpu; level = :phy, n_boot = 4, seed = 2, progress = false, _refit = nc))
        @test r4.n_failed == 4 && all(isnan, r4.ci_lower.Sigma_phy)
    end

    @testset "P7 refusals, each paired with a supported control" begin
        err(g) = try g(); nothing catch e; e end
        cap, log = _pmv_capture()
        ctl = _quiet(() -> _bsig(fpu; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = cap))
        @test ctl.level == [:phy] && length(log.calls) == 3
        n_ok = length(log.calls)
        Σ = Matrix(0.5I(3)) + fill(0.3, 3, 3)
        for (kw, word) in ((:Σ_phy, "Σ_phy"), (:y, "`y`"), (:n_sites, "`n_sites`"), (:X, "`X`"), (:individual, "`individual`"))
            val = kw === :Σ_phy ? Σ : kw === :y ? Y6 : kw === :n_sites ? 24 : kw === :X ? zeros(3, 24, 3) : sid6
            e = err(() -> _bsig(fpu; level = :phy, n_boot = 3, progress = false, _refit = cap, (kw => val,)...))
            @test e isa ArgumentError && occursin(word, e.msg) && occursin("PrecisionMultivariateFit", e.msg)
        end
        # tiers that do not exist for this class
        for lv in (:B, :W, :unit, :unit_obs, [:B, :W])
            e = err(() -> _bsig(fpu; level = lv, n_boot = 3, progress = false, _refit = cap))
            @test e isa ArgumentError && occursin("None of the requested level(s) are present", e.msg)
        end
        # default level: B and W dropped with an @info, phy kept; ICC dropped; cross_corr never produced
        r, logs = _logged(() -> _bsig(fpu; n_boot = 3, seed = 1, progress = false, _refit = cap))
        @test r.level == [:phy]
        @test any(occursin("not present", m) && occursin("B, W", m) for m in _infos(logs))
        @test any(occursin("ICC", m) for m in _infos(logs))
        @test collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
        # nothing reportable
        for wh in ([:ICC], [:cross_corr], [:ICC, :cross_corr])
            e = err(() -> _bsig(fpu; level = :phy, what = wh, n_boot = 3, progress = false, _refit = cap))
            @test e isa ArgumentError
        end
        # n_cores > 1
        @test_throws ArgumentError _bsig(fpu; level = :phy, n_cores = Threads.nthreads() + 1, n_boot = 3, progress = false, _refit = cap)
        @test length(log.calls) == n_ok + 3                           # only the supported control (default level) refit
        # JointPhyloGroupedGaussianFit is handled by Stage 2 (tier :phy only): see the JPG-TIER region below.
        # It is no longer 'unsupported'; its own refusals (unidentified or non-converged original) are tested there.
    end

    @testset "P8 real simulate-and-refit route, both modes (6 tips, d = 3)" begin
        for (nm, f) in (("barelowrank", fpb), ("explicitunique", fpu))
            r = _quiet(() -> _bsig(f; level = :phy, n_boot = 6, seed = 20261102, progress = false))
            @test r.n_boot == 6 && 0 <= r.n_failed <= 6
            @test r.level == [:phy] && collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
            for k in keys(r.point_est)
                lo = r.ci_lower[k]; hi = r.ci_upper[k]
                @test size(lo) == size(r.point_est[k]) && size(hi) == size(lo)
                @test all((lo .<= hi) .| isnan.(lo) .| isnan.(hi))
            end
            @test all(-1 .<= r.ci_lower.R_phy .<= 1) && all(-1 .<= r.ci_upper.R_phy .<= 1)
            L = f.loading; u = f.phylo_unique_variance === nothing ? zeros(3) : f.phylo_unique_variance
            @test r.point_est.Sigma_phy ≈ L * L' + Diagonal(u) atol = 1e-12
            @test r.n_failed < 6                                          # the route actually produces refits
        end
    end

    @testset "P9 show, level order and the R_phy clamp scope" begin
        r = _quiet(() -> _bsig(fpu; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(fpu)))
        txt = sprint(show, MIME"text/plain"(), r)
        @test occursin("Sigma_phy (3x3 matrix; diagonal estimate [lower, upper]):", txt)
        @test occursin("communality_phy (estimate [lower, upper], n_effective):", txt)
        @test occursin("level=[:phy]", sprint(show, r))
        # rounding-sized overshoot of R_phy is clamped, a genuine excursion stays visible (testset
        # "R_* clamp tolerance" analogue); hand fit with R_phy[1,2] = 1 + delta exactly
        function overshoot(δ, s = 1.0)                      # R_phy[1,2] = s * (1 + δ) exactly
            u2 = 0.64 / (1 + δ)^2 - 0.64
            g = _pmv_hand(phy6, sid6; beta = [0.0, 0.0], loading = reshape([0.6, s * 0.8], 2, 1), unique = [0.0, u2],
                          residual = [0.1, 0.1])               # not simulated from: it is only what every refit returns
            f = _pmv_hand(phy6, sid6; beta = [0.0, 0.0], loading = reshape([0.6, s * 0.8], 2, 1), unique = [0.1, 0.1],
                          residual = [0.1, 0.1])               # the (valid) original fit that is simulated from
            return _quiet(() -> _bsig(f; level = :phy, what = [:R], n_boot = 5, seed = 1, progress = false, _refit = stub(g)))
        end
        rs = overshoot(1e-9)
        @test rs.ci_lower.R_phy[1, 2] == 1.0 && rs.ci_upper.R_phy[1, 2] == 1.0
        rb = overshoot(1e-6)
        @test isapprox(rb.ci_upper.R_phy[1, 2], 1 + 1e-6; atol = 1e-10)
        @test rb.ci_upper.R_phy[1, 2] > 1.0
        rm9 = overshoot(1e-9, -1.0)
        @test rm9.ci_lower.R_phy[1, 2] == -1.0 && rm9.ci_upper.R_phy[1, 2] == -1.0
        @test overshoot(1e-6, -1.0).ci_lower.R_phy[1, 2] < -1.0
    end
end
# END PHY-TIER

# ============================================================================================
# PHY TIER, JointPhyloGroupedGaussianFit (BEGIN JPG-TIER). Stage 2, tier :phy only.
# Evidence level: STRUCTURAL (contract read from the pinned R source) PLUS JULIA-DESCRIPTIVE (pre-run
# failure fractions on fixtures); there is NO numerical comparison with an R run and NO R parity claim.
# Documented DEVIATIONS pinned here: a replicate whose refit throws, does not converge or is not
# component-identified is a failure (R: convergence only); B and W are dropped (R would report them).
# Expected values are computed from raw fit fields and from the exact model covariance, never through
# the code under test. The mutation runner extracts this region (header lines 1-81 plus everything from
# the BEGIN marker), so keep it self-contained (it defines its own helpers; it needs only the header).
# ============================================================================================
using SparseArrays
const _PG = GLLVModels

_jpg_trait_design(d, m) = kron(ones(m, 1), Matrix{Float64}(I, d, d))
_jpg_terms(names) = GroupingTerm[GroupingTerm(n; mode = :indep, common = false) for n in names]
# n x groups incidence from the test's own first-seen coding (independent of the package helper)
function _jpg_inc(labels)
    lv = unique(labels); n = length(labels)
    return sparse(collect(1:n), [findfirst(==(x), lv) for x in labels], ones(n), n, length(lv))
end
_jpg_same_partition(a, b) = length(a) == length(b) &&
    all((a[i] == a[j]) == (b[i] == b[j]) for i in eachindex(a), j in eachindex(a))

# Hand-built JointPhyloGroupedGaussianFit (no optimiser): converged, chosen parameters. `labels` is a
# NamedTuple keyed by term name (its key order is the term order).
function _jpg_hand(phy, sid, labels; beta, loading, unique, ordvars, residual, ok = true,
                   hess_pd = true, D = nothing)
    d = size(loading, 1); m = length(sid); rank = size(loading, 2)
    D = D === nothing ? _jpg_trait_design(d, m) : D
    names = Union{String, Symbol}[Symbol("b", j) for j in 1:length(beta)]
    L = Matrix{Float64}(loading); u = unique === nothing ? nothing : Vector{Float64}(unique)
    pc = L * L' + (u === nothing ? zeros(d, d) : Matrix(Diagonal(u)))
    incs = SparseMatrixCSC{Float64, Int}[_jpg_inc(labels[k]) for k in keys(labels)]
    return JointPhyloGroupedGaussianFit(Vector{Float64}(beta), L, u, Matrix{Float64}(pc),
        Vector{Float64}(residual), [Matrix(Diagonal(Vector{Float64}(v))) for v in ordvars],
        _jpg_terms(collect(keys(labels))), phy, collect(Int, sid), zeros(1), ["x"], -1.0, ok, 0.0, 1.0,
        hess_pd, 1.0, 1, :converged, zeros(d, m), Matrix{Float64}(D), (d, m), names, incs,
        u === nothing ? :barelowrank : :explicitunique, rank)
end
# copy with some fields replaced (deterministic stand-in for a refit); `break_ident` swaps every
# incidence for the identity (every observation its own group), which aliases the ordinary variance
# with the residual variance: component-NON-identified.
function _jpg_with(f; loading = f.loading, unique = f.phylo_unique_variance, ok = f.converged,
                   hess_pd = f.hessian_positive_definite, break_ident = false)
    d, m = f.response_shape
    pc = loading * loading' + (unique === nothing ? zeros(d, d) : Matrix(Diagonal(unique)))
    incs = break_ident === :throw ? SparseMatrixCSC{Float64, Int}[sparse(1.0I, m - 1, m - 1) for _ in f.incidences] :  # diagnosing it throws
           break_ident ? SparseMatrixCSC{Float64, Int}[sparse(1.0I, m, m) for _ in f.incidences] : f.incidences
    return JointPhyloGroupedGaussianFit(f.beta, Matrix{Float64}(loading), unique, Matrix{Float64}(pc),
        f.residual_variance, f.ordinary_covariances, f.terms, f.phy, f.species_id, f.parameters,
        f.parameter_labels, f.loglik, ok, f.gradient_norm, f.hessian_min_eigenvalue, hess_pd,
        f.hessian_condition_number, f.iterations, f.stopping_reason, f.response, f.mean_design,
        f.response_shape, f.coefficient_names, incs, f.phylo_mode, size(loading, 2))
end

# exact model covariance of vec(Y) (trait fastest): phylo (total) + one block per ordinary term + residual
function _jpg_exact_cov(f; ordinary = true)
    d, m = f.response_shape
    Ainv = inv(Matrix(f.phy.Q))
    nodes = [f.phy.species_aug_id[s] for s in f.species_id]
    Ao = [Ainv[nodes[o], nodes[o2]] for o in 1:m, o2 in 1:m]
    LL = f.loading * f.loading'
    uu = f.phylo_unique_variance === nothing ? zeros(d) : f.phylo_unique_variance
    K = kron(Ao, LL + Diagonal(uu)) + kron(Matrix{Float64}(I, m, m), Diagonal(f.residual_variance))
    if ordinary
        for k in eachindex(f.incidences)
            B = Matrix(f.incidences[k]); K = K + kron(B * B', Diagonal(diag(f.ordinary_covariances[k])))
        end
    end
    return K
end

# independent test-side data simulator (per observation, per group; no package code)
function _jpg_truth_data(rng, phy, sid, labels; beta, loading, uniq, ordvars, residual)
    d = length(beta); m = length(sid)
    F = cholesky(Symmetric(Matrix(phy.Q))).U
    g = F \ randn(rng, phy.n_aug, size(loading, 2)); h = F \ randn(rng, phy.n_aug, d)
    Z = [sqrt.(v) .* randn(rng, d, length(unique_labels)) for (v, unique_labels) in
         zip(ordvars, [unique(labels[k]) for k in keys(labels)])]
    Y = Matrix{Float64}(undef, d, m)
    for o in 1:m, t in 1:d
        nd = phy.species_aug_id[sid[o]]
        y = beta[t] + sum(loading[t, k] * g[nd, k] for k in 1:size(loading, 2)) +
            (uniq === nothing ? 0.0 : sqrt(uniq[t]) * h[nd, t]) + sqrt(residual[t]) * randn(rng)
        for (kk, k) in enumerate(keys(labels))
            y += Z[kk][t, findfirst(==(labels[k][o]), unique(labels[k]))]
        end
        Y[t, o] = y
    end
    return Y
end

# records every refit call (keywords included) and returns `ret` (a fit) or runs a real refit
function _jpg_capture(ret = nothing)
    log = (calls = NamedTuple[],)
    f = function (Y, phy; kw...)
        push!(log.calls, (Y = copy(Y), phy = phy, kw = Dict{Symbol, Any}(kw)))
        return ret === nothing ? _PG.fit_joint_phylo_grouped_gaussian(Y, phy; kw...) : ret
    end
    return f, log
end

@testset "bootstrap_Sigma_r phy tier: JointPhyloGroupedGaussianFit (structural plus Julia descriptive)" begin
    phy4 = PrecisionPhy(GLLVModels.random_balanced_tree(4; branch_length = 0.35))
    phy6 = PrecisionPhy(GLLVModels.random_balanced_tree(6; branch_length = 0.35))
    sid4 = [1, 2, 3, 4, 1, 3, 2, 4, 1, 2, 3, 4]
    sid6 = repeat(1:6, inner = 4)
    u12 = [3, 1, 1, 2, 3, 3, 1, 2, 2, 3, 1, 3]                       # 3 unbalanced groups (4, 3, 5)
    c12 = ["a", "b", "a", "c", "b", "a", "c", "c", "b", "a", "a", "b"] # 3 string groups, crossing u12
    L1 = reshape([0.7, 0.4, -0.8], 3, 1)
    # f2: d = 2 bare; f3u: d = 3 explicitunique, two crossing terms (unit, cluster); f3r2: d = 3 rank 2
    f2 = _jpg_hand(phy4, sid4, (unit = u12,); beta = [0.4, -0.3], loading = reshape([0.9, -0.6], 2, 1),
                   unique = nothing, ordvars = [[0.5, 0.3]], residual = [0.4, 0.6])
    f3u = _jpg_hand(phy4, sid4, (unit = u12, cluster = c12); beta = [0.0, 0.5, -0.5], loading = L1,
                    unique = [0.35, 0.60, 0.25], ordvars = [[0.45, 0.30, 0.55], [0.25, 0.50, 0.35]],
                    residual = [0.40, 0.30, 0.50])
    f3r2 = _jpg_hand(phy4, sid4, (cluster2 = c12,); beta = [0.2, 0.1, -0.2], loading = [0.8 0.3; -0.5 0.6; 0.4 -0.7],
                     unique = nothing, ordvars = [[0.40, 0.55, 0.30]], residual = [0.4, 0.9, 0.6])
    stub(f) = (Y, phy; kw...) -> f                                   # every replicate "refits" to f itself
    ident(f) = _PG._joint_phylo_grouped_component_identification(f).reason

    @testset "J0 fixtures; JPG is no longer 'unsupported'; a real refusal still is" begin
        @test ident(f2) === :identified && ident(f3u) === :identified && ident(f3r2) === :identified
        @test ident(_jpg_with(f2; break_ident = true)) === :nonidentifiable   # control: the breaker works
        @test f2.converged && f3u.converged && f3r2.converged
        @test phy4.species_aug_id != collect(1:4)                              # non-identity node map
        r = _quiet(() -> _bsig(f3u; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = stub(f3u)))
        @test r isa GLLVModels.BootstrapSigmaR && r.level == [:phy]
        e = try _bsig(_NotAFit(); progress = false); nothing catch ex; ex end   # paired control
        @test e isa ArgumentError && occursin("unsupported fit class", e.msg)
        @test occursin("JointPhyloGroupedGaussianFit", e.msg)                   # the refusal text names the supported class
    end

    @testset "J1 point estimates from raw fields: LL' + diag(unique); no residual, no ordinary variance" begin
        for (nm, f) in (("d2 bare, 1 term", f2), ("d3 explicitunique, 2 terms", f3u), ("d3 rank 2, cluster2", f3r2))
            r = _quiet(() -> _bsig(f; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f)))
            d = size(f.loading, 1)
            L = f.loading
            u = f.phylo_unique_variance === nothing ? zeros(d) : f.phylo_unique_variance
            S = L * L' + Diagonal(u)
            @test collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
            @test r.level == [:phy]
            @test r.point_est.Sigma_phy ≈ S atol = 1e-12
            @test r.point_est.R_phy ≈ _cor(S) atol = 1e-12
            @test r.point_est.communality_phy ≈ vec(sum(abs2, L; dims = 2)) ./ diag(S) atol = 1e-12
            @test r.ci_lower.Sigma_phy ≈ S atol = 1e-12
            @test r.ci_upper.Sigma_phy ≈ S atol = 1e-12
            @test r.n_failed == 0
            # contamination is observable (control) and absent
            withres = S + Diagonal(f.residual_variance)
            withord = S + sum(Diagonal(diag(c)) for c in f.ordinary_covariances)
            @test maximum(abs, withres - S) > 0.3 && maximum(abs, withord - S) > 0.3
            @test !isapprox(r.point_est.Sigma_phy, withres; atol = 0.2)
            @test !isapprox(r.point_est.Sigma_phy, withord; atol = 0.2)
            @test !isapprox(r.point_est.communality_phy, vec(sum(abs2, L; dims = 2)) ./ diag(withord); atol = 0.02)
        end
    end

    @testset "J1c barelowrank: communality_phy is exactly 1, zero width, flagged by @info" begin
        r, logs = _logged(() -> _bsig(f2; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f2)))
        @test r.point_est.communality_phy == ones(2)
        @test r.ci_lower.communality_phy == ones(2) && r.ci_upper.communality_phy == ones(2)
        @test any(occursin("communality_phy is exactly 1", m) for m in _infos(logs))
        r2, logs2 = _logged(() -> _bsig(f3u; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(f3u)))
        @test !any(occursin("communality_phy is exactly 1", m) for m in _infos(logs2))
        @test all(<(1), r2.point_est.communality_phy)
        _, logs3 = _logged(() -> _bsig(f2; level = :phy, what = :Sigma, n_boot = 4, seed = 1, progress = false, _refit = stub(f2)))
        @test !any(occursin("communality_phy is exactly 1", m) for m in _infos(logs3))
    end

    @testset "J2 simulator law: exact moments, ordinary terms redrawn through the incidences" begin
        # a covariate-bearing mean design for case 1
        m4 = length(sid4); x = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, -2.0, 0.3, 0.0, 1.2, -0.7]
        D = zeros(2 * m4, 3)
        for o in 1:m4
            D[(o - 1) * 2 + 1, 1] = 1.0; D[(o - 1) * 2 + 2, 2] = 1.0
            D[(o - 1) * 2 + 1, 3] = x[o]; D[(o - 1) * 2 + 2, 3] = x[o]
        end
        f2x = _jpg_hand(phy4, sid4, (unit = u12,); beta = [0.4, -0.3, 0.6], loading = reshape([0.9, -0.6], 2, 1),
                        unique = nothing, ordvars = [[0.5, 0.3]], residual = [0.4, 0.6], D = D)
        N = 40000
        for (k, (nm, f)) in enumerate((("d2 bare, covariate design, 1 unbalanced term", f2x),
                                        ("d3 explicitunique, 2 crossing terms", f3u),
                                        ("d3 rank 2 bare, cluster2", f3r2)))
            d, m = f.response_shape
            Ys = _PG._bsig_jpg_simulate_batch(MersenneTwister(52000 + k), f, N)
            @test length(Ys) == N && all(size(Y) == (d, m) for Y in Ys[1:5])
            S = reduce(hcat, [vec(Y) for Y in Ys])          # a Vector (not a generator): linear-time hcat
            K = _jpg_exact_cov(f)
            μ = f.mean_design * f.beta
            sm = vec(mean(S; dims = 2)); sc = cov(S; dims = 2)
            @test all(abs.(sm .- μ) .<= 4.5 .* sqrt.(diag(K) ./ N))
            tol = 4.5 .* sqrt.((diag(K) * diag(K)' .+ K .^ 2) ./ N)
            @test count(abs.(sc .- K) .> tol) == 0
            # controls: the test has power against a missing ordinary term and against a tree-free law
            @test count(abs.(sc .- _jpg_exact_cov(f; ordinary = false)) .> tol) > 0
            @test count(abs.(sc .- Diagonal(diag(K))) .> tol) > 0
        end
    end

    @testset "J2c the exact covariance equals the fitter's own likelihood (ties simulator law to the fit)" begin
        for f0 in (f3u, f2)
            d, m = f0.response_shape
            Yd = randn(MersenneTwister(6), d, m)
            K = _jpg_exact_cov(f0)
            res = vec(Yd .- reshape(f0.mean_design * f0.beta, d, m))
            dense = -0.5 * (logdet(cholesky(Symmetric(K))) + res' * (K \ res) + length(res) * log(2π))
            jl = _PG.joint_phylo_grouped_gaussian_loglik(Yd .- reshape(f0.mean_design * f0.beta, d, m), f0.phy,
                f0.loading, f0.residual_variance; phylo_unique_variance = f0.phylo_unique_variance,
                species_id = f0.species_id, ordinary_incidences = f0.incidences,
                ordinary_trait_variances = [diag(c) for c in f0.ordinary_covariances])
            @test isapprox(dense, jl; atol = 1e-9, rtol = 1e-12)
        end
    end

    @testset "J3 refit receives the fitted structure; term-to-keyword label mapping; no warm start" begin
        base = Set([:rank, :phylo_mode, :terms, :species_id, :X, :coefficient_names])
        # case id => (fit, expected extra keywords, original labels by term name)
        uo_nested = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]; un_of = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        cases = (
            ("unit only, shuffled unbalanced", _jpg_hand(phy4, sid4, (unit = u12,); beta = [0.0, 0.0], loading = reshape([0.9, -0.6], 2, 1),
                unique = nothing, ordvars = [[0.5, 0.3]], residual = [0.4, 0.6]), Set([:unit]), (unit = u12,)),
            ("terms in non-canonical order [cluster, unit], string labels", _jpg_hand(phy4, sid4, (cluster = c12, unit = u12);
                beta = [0.0, 0.0, 0.0], loading = L1, unique = nothing, ordvars = [[0.3, 0.2, 0.4], [0.5, 0.4, 0.3]],
                residual = [0.4, 0.3, 0.5]), Set([:unit, :cluster]), (cluster = c12, unit = u12)),
            ("unit + unit_obs (nested)", _jpg_hand(phy4, sid4, (unit = un_of, unit_obs = uo_nested); beta = [0.0, 0.0, 0.0], loading = L1,
                unique = nothing, ordvars = [[0.3, 0.2, 0.4], [0.5, 0.4, 0.3]], residual = [0.4, 0.3, 0.5]),
                Set([:unit, :unit_obs]), (unit = un_of, unit_obs = uo_nested)),
            ("unit_obs only (a unit stand-in is needed for the nesting check)", _jpg_hand(phy4, sid4, (unit_obs = uo_nested,);
                beta = [0.0, 0.0, 0.0], loading = L1, unique = nothing, ordvars = [[0.3, 0.2, 0.4]], residual = [0.4, 0.3, 0.5]),
                Set([:unit, :unit_obs]), (unit_obs = uo_nested,)),
            ("cluster + cluster2", _jpg_hand(phy4, sid4, (cluster = u12, cluster2 = c12); beta = [0.0, 0.0, 0.0], loading = L1,
                unique = nothing, ordvars = [[0.3, 0.2, 0.4], [0.5, 0.4, 0.3]], residual = [0.4, 0.3, 0.5]),
                Set([:cluster, :cluster2]), (cluster = u12, cluster2 = c12)))
        for (nm, f, extra, orig) in cases
            cap, log = _jpg_capture(f)
            _quiet(() -> _bsig(f; level = :phy, n_boot = 3, seed = 5, progress = false, _refit = cap))
            @test length(log.calls) == 3
            for c in log.calls
                @test c.phy === f.phy                                                # the SAME tree object
                @test Set(keys(c.kw)) == union(base, extra)                          # no start, no f_tol, no iterations/g_tol
                @test c.kw[:rank] == f.rank && c.kw[:phylo_mode] === f.phylo_mode
                @test c.kw[:terms] == f.terms && c.kw[:species_id] == f.species_id
                @test c.kw[:X] == f.mean_design && c.kw[:coefficient_names] == f.coefficient_names
                for k in keys(orig)                                                  # each keyword carries ITS term's partition
                    @test _jpg_same_partition(c.kw[k], orig[k])
                end
                # distinct terms with different partitions must not be swapped
                length(orig) == 2 && @test !_jpg_same_partition(c.kw[first(keys(orig))], c.kw[last(keys(orig))])
            end
            # the keywords are accepted by the real fitter and rebuild exactly the original incidences
            kw = log.calls[1].kw
            d, m = f.response_shape
            q = length(f.beta)
            lay = _PG._joint_phylo_grouped_fit_layout(d, f.rank, f.phylo_mode, q, f.terms)
            Ytry = randn(MersenneTwister(9), d, m)
            real = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Ytry, f.phy; kw..., start = zeros(lay.total), iterations = 0))
            @test length(real.incidences) == length(f.incidences)
            @test all(Matrix(real.incidences[k] * real.incidences[k]') == Matrix(f.incidences[k] * f.incidences[k]')
                      for k in eachindex(f.incidences))
            @test real.terms == f.terms && real.species_id == f.species_id
        end
    end

    @testset "J3c a real fit is reproduced: the refit keywords rebuild the original likelihood" begin
        Yr = _jpg_truth_data(MersenneTwister(20261103), phy6, sid6, (unit = repeat(1:8, inner = 3),); beta = [0.4, -0.3, 0.7],
                             loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = nothing, ordvars = [[0.25, 0.20, 0.30]],
                             residual = [0.30, 0.40, 0.35])
        fr = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yr, phy6; rank = 1, phylo_mode = :barelowrank,
                terms = _jpg_terms([:unit]), unit = repeat(1:8, inner = 3), species_id = sid6))
        @test fr isa JointPhyloGroupedGaussianFit
        cap, log = _jpg_capture(fr)
        _quiet(() -> _bsig(fr; level = :phy, n_boot = 2, seed = 3, progress = false, _refit = cap))
        kw = log.calls[1].kw
        again = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(fr.response, fr.phy; kw..., start = fr.parameters, iterations = 0))
        @test isapprox(again.loglik, fr.loglik; atol = 1e-9, rtol = 1e-12)
        @test again.coefficient_names == fr.coefficient_names && length(again.beta) == length(fr.beta)
    end

    @testset "J4 failure accounting: thrown, non-converged, non-identified and wrong-class refits all count" begin
        f0 = f3u
        c(b) = 0.5 + 0.05 * b
        function injected(nonconv, thrown, nonident, wrong, nonpd = Int[])
            n = Ref(0)
            return function (Y, phy; kw...)
                n[] += 1; b = n[]
                b in thrown && error("injected refit failure")
                b in wrong && return _NotAFit()
                return _jpg_with(f0; loading = c(b) .* f0.loading, ok = !(b in nonconv),
                                 break_ident = b in nonident, hess_pd = !(b in nonpd))
            end
        end
        nb = 24
        nonconv, thr, nid, cls = [3, 12], [7], [5, 19], [15]
        r, logs = _logged(() -> _bsig(f0; level = :phy, n_boot = nb, seed = 1, progress = false,
                                      _refit = injected(nonconv, thr, nid, cls)))
        @test r.n_failed == 6                                  # 2 non-converged + 1 thrown + 2 non-identified + 1 wrong class
        @test any(occursin("6 of $nb replicates were lost", w) &&
                  occursin("Failed refits: 1 thrown, 3 non-converged, 2 non-identified", w) for w in _warns(logs))
        surv = [b for b in 1:nb if !(b in vcat(nonconv, thr, nid, cls))]
        vals = [c(b)^2 * f0.loading[1]^2 + f0.phylo_unique_variance[1] for b in surv]
        @test r.ci_lower.Sigma_phy[1, 1] ≈ quantile(vals, 0.025) atol = 1e-12
        @test r.ci_upper.Sigma_phy[1, 1] ≈ quantile(vals, 0.975) atol = 1e-12
        @test r.n_effective.communality_phy == fill(length(surv), 3)       # failed replicates are NaN, not draws
        @test r.diagnostics.n_thrown == 1 && r.diagnostics.n_not_converged == 3   # 2 non-converged + the wrong-class return
        @test r.diagnostics.n_non_identified == 2
        @test r.diagnostics.n_thrown + r.diagnostics.n_not_converged + r.diagnostics.n_non_identified == r.n_failed == 6
        # a non-identified replicate is NOT converted to a success by being converged (control: same count w/o breaker)
        r0 = _quiet(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false, _refit = injected(Int[], Int[], Int[], Int[])))
        @test r0.n_failed == 0 && r0.diagnostics.n_non_identified == 0 && r0.diagnostics.n_thrown == 0
        r1 = _quiet(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false, _refit = injected(Int[], Int[], [4], Int[])))
        @test r1.n_failed == 1 && r1.diagnostics.n_non_identified == 1
        # a replicate whose identification cannot even be diagnosed (it throws) is counted as non-identified, never as a success
        thr_id = _quiet(() -> _bsig(f0; level = :phy, n_boot = 4, seed = 1, progress = false,
                                    _refit = (Y, phy; kw...) -> _jpg_with(f0; break_ident = :throw)))
        @test thr_id.n_failed == 4 && thr_id.diagnostics.n_non_identified == 4 && all(isnan, thr_id.ci_lower.Sigma_phy)
        @test_throws Exception _PG._joint_phylo_grouped_component_identification(_jpg_with(f0; break_ident = :throw))  # control: it does throw
        # attrition warning (unchanged rule): 3 of 10 lost = 70% retained
        r2, logs2 = _logged(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false, _refit = injected([2], [5], [9], Int[])))
        @test r2.n_failed == 3
        @test any(occursin("attrition", w) && occursin("70.0%", w) for w in _warns(logs2))
        # a real refit stopped after one iteration is non-converged: a failure, not an exception
        nc = (Y, phy; kw...) -> _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Y, phy; kw..., iterations = 1))
        Yq = _jpg_truth_data(MersenneTwister(4), phy6, sid6, (unit = repeat(1:8, inner = 3),); beta = [0.4, -0.3, 0.7],
                             loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = nothing, ordvars = [[0.25, 0.2, 0.3]],
                             residual = [0.3, 0.4, 0.35])
        fq = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yq, phy6; rank = 1, phylo_mode = :barelowrank,
                terms = _jpg_terms([:unit]), unit = repeat(1:8, inner = 3), species_id = sid6))
        r4 = _quiet(() -> _bsig(fq; level = :phy, n_boot = 3, seed = 2, progress = false, _refit = nc))
        @test r4.n_failed == 3 && r4.diagnostics.n_not_converged == 3 && all(isnan, r4.ci_lower.Sigma_phy)
    end

    @testset "J5 Hessian curvature: count of converged replicates with a non-PD Hessian; warning on the original" begin
        f0 = f3u
        mk(nonpd, nonconv = Int[]) = (n = Ref(0); (Y, phy; kw...) -> (n[] += 1; b = n[];
            _jpg_with(f0; loading = (0.5 + 0.05 * b) .* f0.loading, hess_pd = !(b in nonpd), ok = !(b in nonconv))))
        r, logs = _logged(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false, _refit = mk([2, 3, 8])))
        @test r.diagnostics.n_hessian_not_pd == 3 && r.diagnostics.n_converged == 10
        @test r.n_failed == 0                                            # curvature does NOT make a replicate a failure (as in R)
        @test r.diagnostics.original_hessian_pd === true
        @test any(occursin("3 of 10 converged", m) && occursin("Hessian", m) for m in _infos(logs))
        @test !any(occursin("Hessian", w) for w in _warns(logs))         # original Hessian is PD: no warning
        # only CONVERGED replicates are counted; a non-converged one with a non-PD Hessian is not
        r2 = _quiet(() -> _bsig(f0; level = :phy, n_boot = 10, seed = 1, progress = false, _refit = mk([2, 3, 8], [2])))
        @test r2.diagnostics.n_hessian_not_pd == 2 && r2.diagnostics.n_converged == 9 && r2.n_failed == 1
        # control: none non-PD, no message
        r3, logs3 = _logged(() -> _bsig(f0; level = :phy, n_boot = 6, seed = 1, progress = false, _refit = mk(Int[])))
        @test r3.diagnostics.n_hessian_not_pd == 0
        @test !any(occursin("non-positive-definite", m) || occursin("converged replicates", m) for m in _infos(logs3))
        # original Hessian not PD: the run proceeds, with a warning, and records it
        fnp = _jpg_with(f0; hess_pd = false)
        r4, logs4 = _logged(() -> _bsig(fnp; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = stub(fnp)))
        @test r4 isa GLLVModels.BootstrapSigmaR && r4.diagnostics.original_hessian_pd === false
        @test any(occursin("Hessian", w) && occursin("not positive definite", w) for w in _warns(logs4))
        # a refusal does not emit that warning (it comes after validation)
        _, logs5 = _logged(() -> try _bsig(fnp; level = :B, n_boot = 4, progress = false, _refit = stub(fnp)) catch; nothing end)
        @test !any(occursin("Hessian", w) for w in _warns(logs5))
    end

    @testset "J6 refusals, each paired with a supported control" begin
        err(g) = try g(); nothing catch e; e end
        cap, log = _jpg_capture(f3u)
        ctl = _quiet(() -> _bsig(f3u; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = cap))
        @test ctl.level == [:phy] && length(log.calls) == 3
        n_ok = length(log.calls)
        # the original fit must be identified and converged
        fni = _jpg_with(f3u; break_ident = true)
        e = err(() -> _bsig(fni; level = :phy, n_boot = 3, progress = false, _refit = cap))
        @test e isa ArgumentError && occursin("identif", e.msg) && occursin("JointPhyloGroupedGaussianFit", e.msg)
        fnc = _jpg_with(f3u; ok = false)
        e = err(() -> _bsig(fnc; level = :phy, n_boot = 3, progress = false, _refit = cap))
        @test e isa ArgumentError && occursin("converge", e.msg) && occursin("JointPhyloGroupedGaussianFit", e.msg)
        @test length(log.calls) == n_ok                                   # neither refusal triggered a refit
        # a real non-converged fit is refused (paired with a real converged control, J7)
        Yq = _jpg_truth_data(MersenneTwister(4), phy6, sid6, (unit = repeat(1:8, inner = 3),); beta = [0.4, -0.3, 0.7],
                             loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = nothing, ordvars = [[0.25, 0.2, 0.3]],
                             residual = [0.3, 0.4, 0.35])
        stalled = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yq, phy6; rank = 1, phylo_mode = :barelowrank,
                terms = _jpg_terms([:unit]), unit = repeat(1:8, inner = 3), species_id = sid6, iterations = 0))
        @test !stalled.converged
        @test err(() -> _bsig(stalled; level = :phy, n_boot = 3, progress = false, _refit = cap)) isa ArgumentError
        # arguments that would be silently ignored
        Σ = Matrix(0.5I(3)) + fill(0.3, 3, 3)
        for (kw, word) in ((:Σ_phy, "Σ_phy"), (:y, "`y`"), (:n_sites, "`n_sites`"), (:X, "`X`"), (:individual, "`individual`"))
            val = kw === :Σ_phy ? Σ : kw === :y ? zeros(3, 12) : kw === :n_sites ? 12 : kw === :X ? zeros(3, 12, 3) : sid4
            e = err(() -> _bsig(f3u; level = :phy, n_boot = 3, progress = false, _refit = cap, (kw => val,)...))
            @test e isa ArgumentError && occursin(word, e.msg) && occursin("JointPhyloGroupedGaussianFit", e.msg)
        end
        # tiers that do not exist for this class
        for lv in (:B, :W, :unit, :unit_obs, [:B, :W])
            e = err(() -> _bsig(f3u; level = lv, n_boot = 3, progress = false, _refit = cap))
            @test e isa ArgumentError && occursin("None of the requested level(s) are present", e.msg)
        end
        # default level: B and W dropped, with an @info that says what R would do
        r, logs = _logged(() -> _bsig(f3u; n_boot = 3, seed = 1, progress = false, _refit = cap))
        @test r.level == [:phy]
        @test any(occursin("not present", m) && occursin("B, W", m) && occursin("R would report", m) for m in _infos(logs))
        @test any(occursin("ICC", m) for m in _infos(logs))
        @test collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
        # level = :phy alone: no B/W message
        _, logs2 = _logged(() -> _bsig(f3u; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = cap))
        @test !any(occursin("R would report", m) for m in _infos(logs2))
        for wh in ([:ICC], [:cross_corr], [:ICC, :cross_corr])
            @test err(() -> _bsig(f3u; level = :phy, what = wh, n_boot = 3, progress = false, _refit = cap)) isa ArgumentError
        end
        @test_throws ArgumentError _bsig(f3u; level = :phy, n_cores = Threads.nthreads() + 1, n_boot = 3, progress = false, _refit = cap)
        # only the three supported controls (J6 control, default-level control, level = :phy control) refit
        @test length(log.calls) == n_ok + 3 + 3
        # the original-fit refusal precedes argument validation (nothing is simulated or refit)
        e = err(() -> _bsig(fni; level = :phy, n_boot = 3, progress = false, n_cores = Threads.nthreads() + 1, _refit = cap))
        @test e isa ArgumentError && occursin("identif", e.msg)
    end

    @testset "J7 real simulate-and-refit route (6 tips, d = 3, one :unit term), both modes" begin
        unit = repeat(1:8, inner = 3)
        for (nm, mode, uq) in (("barelowrank", :barelowrank, nothing), ("explicitunique", :explicitunique, [0.30, 0.20, 0.25]))
            Yr = _jpg_truth_data(MersenneTwister(20261103), phy6, sid6, (unit = unit,); beta = [0.4, -0.3, 0.7],
                                 loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = uq, ordvars = [[0.25, 0.20, 0.30]],
                                 residual = [0.30, 0.40, 0.35])
            f = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yr, phy6; rank = 1, phylo_mode = mode,
                    terms = _jpg_terms([:unit]), unit = unit, species_id = sid6))
            @test f.converged && ident(f) === :identified                  # control: the original qualifies
            r = _quiet(() -> _bsig(f; level = :phy, n_boot = 5, seed = 20261103, progress = false))
            @test r.n_boot == 5 && 0 <= r.n_failed <= 5 && r.n_failed < 5
            @test r.level == [:phy] && collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]
            for k in keys(r.point_est)
                lo = r.ci_lower[k]; hi = r.ci_upper[k]
                @test size(lo) == size(r.point_est[k]) && size(hi) == size(lo)
                @test all((lo .<= hi) .| isnan.(lo) .| isnan.(hi))
            end
            @test all(-1 .<= r.ci_lower.R_phy .<= 1) && all(-1 .<= r.ci_upper.R_phy .<= 1)
            u = f.phylo_unique_variance === nothing ? zeros(3) : f.phylo_unique_variance
            @test r.point_est.Sigma_phy ≈ f.loading * f.loading' + Diagonal(u) atol = 1e-12
            @test r.diagnostics.n_thrown + r.diagnostics.n_not_converged + r.diagnostics.n_non_identified == r.n_failed
        end
    end

    @testset "J7b the per-replicate identification @warn is silenced (counted, not spammed)" begin
        noisy = (Y, phy; kw...) -> (@warn "joint phylo grouped covariance components are not separately identifiable; test"; f3u)
        _, direct = _logged(() -> noisy(zeros(3, 12), phy4))
        @test any(occursin("separately identifiable", w) for w in _warns(direct))   # control: the stub does warn
        r, logs = _logged(() -> _bsig(f3u; level = :phy, n_boot = 4, seed = 1, progress = false, _refit = noisy))
        @test !any(occursin("separately identifiable", w) for w in _warns(logs))
        @test r.n_failed == 0
        # an exception raised inside the silenced region is still a counted failure, never swallowed silently
        boom = (Y, phy; kw...) -> error("boom")
        rb = _quiet(() -> _bsig(f3u; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = boom))
        @test rb.n_failed == 3 && rb.diagnostics.n_thrown == 3
        # InterruptException is never swallowed
        @test_throws InterruptException _bsig(f3u; level = :phy, n_boot = 3, seed = 1, progress = false,
                                              _refit = (Y, phy; kw...) -> throw(InterruptException()))
    end

    @testset "J8 exactly n_boot refits; one batch from one stream; seeds; other classes carry no diagnostics" begin
        cap, log = _jpg_capture(f3u)
        n_calls = Ref(0)
        counting = (Y, phy; kw...) -> (n_calls[] += 1; cap(Y, phy; kw...))
        _quiet(() -> _bsig(f3u; level = :phy, n_boot = 7, seed = 11, progress = false, _refit = counting))
        @test n_calls[] == 7
        expect = _PG._bsig_jpg_simulate_batch(MersenneTwister(11), f3u, 7)
        @test all(log.calls[b].Y == expect[b] for b in 1:7)
        runb(seed) = _quiet(() -> _bsig(f3u; level = :phy, n_boot = 6, seed = seed, progress = false, keep_draws = true,
                                        _refit = (Y, phy; kw...) -> _jpg_with(f3u; loading = (0.3 + 0.1 * mean(Y .^ 2)) .* f3u.loading)))
        ra, rb, rc = runb(3), runb(3), runb(4)
        @test all(ra.draws[b].Sigma_phy == rb.draws[b].Sigma_phy for b in 1:6)
        @test any(ra.draws[b].Sigma_phy != rc.draws[b].Sigma_phy for b in 1:6)
        # the whole batch is drawn BEFORE any refit: a refit that consumes the default RNG changes nothing
        Random.seed!(77)
        ys_default = Matrix{Float64}[]
        eater = (Y, phy; kw...) -> (push!(ys_default, copy(Y)); rand(); rand(); randn(); f3u)
        _quiet(() -> _bsig(f3u; level = :phy, n_boot = 5, progress = false, _refit = eater))
        Random.seed!(77)
        @test ys_default == _PG._bsig_jpg_simulate_batch(Random.default_rng(), f3u, 5)
        # other classes carry an empty diagnostics tuple
        t2 = _mk2()
        r2 = _quiet(() -> _bsig(t2; individual = _IND, level = :B, n_boot = 2, seed = 1, progress = false,
                                _refit = (Yb, ind; kw...) -> t2))
        @test r2.diagnostics == NamedTuple()
    end

    @testset "J9 show reports the failure breakdown and curvature for this class only" begin
        f0 = f3u
        inj = (n = Ref(0); (Y, phy; kw...) -> (n[] += 1; _jpg_with(f0; loading = (0.5 + 0.05 * n[]) .* f0.loading,
                                                                   ok = n[] != 2, hess_pd = n[] != 3, break_ident = n[] == 4)))
        r = _quiet(() -> _bsig(f0; level = :phy, n_boot = 8, seed = 1, progress = false, _refit = inj))
        txt = sprint(show, MIME"text/plain"(), r)
        @test occursin("Sigma_phy (3x3 matrix; diagonal estimate [lower, upper]):", txt)
        @test occursin("non-positive-definite Hessian in 1 of 7 converged replicates", txt)
        @test occursin("1 non-converged, 1 non-identified", txt)
        t = _quiet(() -> _bsig(_mk2(); individual = _IND, level = :B, n_boot = 2, seed = 1, progress = false,
                               _refit = (Yb, ind; kw...) -> _mk2()))
        @test !occursin("Hessian", sprint(show, MIME"text/plain"(), t))
    end
end

# Driver-level guard for group terms the upstream identification gate cannot judge (finding S2-F1): a term with
# fewer than 3 groups, or one aliased with fixed group dummies in the mean design, is refused; a term whose
# estimated variance is below 1e-8 of the trait's total variance gets a boundary warning. The thresholds are
# this driver's choices, not R's.
@testset "bootstrap_Sigma_r phy tier: JointPhyloGroupedGaussianFit group-term guard (S2-F1)" begin
    phy4 = PrecisionPhy(GLLVModels.random_balanced_tree(4; branch_length = 0.35))
    sid4 = [1, 2, 3, 4, 1, 3, 2, 4, 1, 2, 3, 4]
    u12 = [3, 1, 1, 2, 3, 3, 1, 2, 2, 3, 1, 3]
    c12 = ["a", "b", "a", "c", "b", "a", "c", "c", "b", "a", "a", "b"]
    L2 = reshape([0.9, -0.6], 2, 1)
    mk(labels; ordvars = [[0.5, 0.3] for _ in 1:length(labels)], D = nothing, beta = [0.4, -0.3]) =
        _jpg_hand(phy4, sid4, labels; beta = beta, loading = L2, unique = nothing, ordvars = ordvars,
                  residual = [0.4, 0.6], D = D)
    stub(f) = (Y, phy; kw...) -> f
    err(g) = try g(); nothing catch e; e end
    run(f) = _logged(() -> _bsig(f; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = stub(f)))
    ident(f) = _PG._joint_phylo_grouped_component_identification(f).reason

    @testset "G1 fewer than 3 groups is refused, naming the term and its group count" begin
        ctl = mk((unit = u12,))
        @test ident(ctl) === :identified
        r, lg = run(ctl)
        @test r isa GLLVModels.BootstrapSigmaR && !any(occursin("boundary", w) for w in _warns(lg))   # 3 groups: accepted, no guard warning
        for (nm, lab, n) in (("one group", ones(Int, 12), 1), ("two groups", repeat([1, 2], 6), 2))
            f = mk((unit = lab,))
            @test ident(f) === :identified                                    # the upstream gate lets it through
            e = err(() -> run(f))
            @test e isa ArgumentError && occursin("unit", e.msg) && occursin("$n group", e.msg) &&
                  occursin("at least 3", e.msg) && occursin("JointPhyloGroupedGaussianFit", e.msg)
        end
        # the offending term is named when it is the second one, and the first term is fine
        f = mk((unit = u12, cluster = repeat(["a", "b"], 6)))
        e = err(() -> run(f))
        @test e isa ArgumentError && occursin("cluster", e.msg) && occursin("2 group", e.msg) && !occursin("term :unit", e.msg)
        # the refusal precedes any refit (exactly 3 groups is the accepted boundary: the control above)
        n_calls = Ref(0)
        cnt = (Y, phy; kw...) -> (n_calls[] += 1; ctl)
        err(() -> _quiet(() -> _bsig(mk((unit = ones(Int, 12),)); level = :phy, n_boot = 3, progress = false, _refit = cnt)))
        @test n_calls[] == 0
    end

    @testset "G2 aliasing with fixed group dummies in the mean design is refused" begin
        inc = _jpg_inc(u12)
        T = _jpg_trait_design(2, 12)
        # all group dummies per trait, no intercept: the group effect is exactly the fixed effects
        fa = mk((unit = u12,); D = Matrix(kron(inc, Matrix(1.0I, 2, 2))), beta = zeros(6))
        @test ident(fa) === :identified
        e = err(() -> run(fa))
        @test e isa ArgumentError && occursin("unit", e.msg) && occursin("alias", e.msg) && occursin("trait 1", e.msg)
        # intercept plus dummies for all but one group spans the same space
        fb = mk((unit = u12,); D = hcat(T, Matrix(kron(inc[:, 1:2], Matrix(1.0I, 2, 2)))), beta = zeros(6))
        @test err(() -> run(fb)) isa ArgumentError
        # only trait 2 aliased: trait-1 intercept, trait-2 group dummies
        e1 = [1.0 0.0]'; e2 = [0.0 1.0]'
        fc = mk((unit = u12,); D = hcat(kron(ones(12), e1), Matrix(kron(inc, e2))), beta = zeros(4))
        e = err(() -> run(fc))
        @test e isa ArgumentError && occursin("trait 2", e.msg) && occursin("alias", e.msg)
        # controls: intercept plus ONE group dummy leaves G - 2 >= 1 free contrasts; covariates unrelated to groups
        fd = mk((unit = u12,); D = hcat(T, Matrix(kron(inc[:, 1], Matrix(1.0I, 2, 2)))), beta = zeros(4))
        r, lg = run(fd)
        @test r isa GLLVModels.BootstrapSigmaR && !any(occursin("boundary", w) for w in _warns(lg))
        x = repeat(collect(range(-1.0, 1.0; length = 12)), inner = 1)
        fe = mk((unit = u12,); D = hcat(T, Matrix(kron(x, Matrix(1.0I, 2, 2)))), beta = zeros(4))
        @test run(fe)[1] isa GLLVModels.BootstrapSigmaR
    end

    @testset "G3 a variance far below the trait's total variance warns (boundary), once per term" begin
        for v in (1e-12, 1e-10)
            f = mk((unit = u12,); ordvars = [[v, 0.3]])
            r, lg = run(f)
            @test r isa GLLVModels.BootstrapSigmaR                              # a warning, not a refusal
            ws = [w for w in _warns(lg) if occursin("boundary", w)]
            @test length(ws) == 1 && occursin("unit", ws[1]) && occursin("trait 1", ws[1]) && !occursin("trait 2", ws[1])
        end
        # above the threshold: silent (1e-6 of the total is NOT flagged)
        f = mk((unit = u12,); ordvars = [[1e-6, 0.3]])
        @test !any(occursin("boundary", w) for w in _warns(run(f)[2]))
        # relative, not absolute: a tiny variance on a tiny-scale trait is not flagged
        fs = _jpg_hand(phy4, sid4, (unit = u12,); beta = [0.4, -0.3], loading = 1e-6 .* L2, unique = nothing,
                       ordvars = [[1e-10, 1e-10]], residual = [1e-6, 1e-6])
        @test !any(occursin("boundary", w) for w in _warns(run(fs)[2]))
        # two terms: only the small one is named
        f2 = mk((unit = u12, cluster = c12); ordvars = [[0.5, 0.3], [0.4, 1e-12]])
        ws = [w for w in _warns(run(f2)[2]) if occursin("boundary", w)]
        @test length(ws) == 1 && occursin("cluster", ws[1]) && occursin("trait 2", ws[1])
        # both traits of one term tiny: still one warning for the term, listing both traits
        fb = mk((unit = u12,); ordvars = [[1e-12, 1e-12]])
        ws = [w for w in _warns(run(fb)[2]) if occursin("boundary", w)]
        @test length(ws) == 1 && occursin("trait 1, 2", ws[1])
        # different traits in different terms: each warning names only its own trait
        fx = mk((unit = u12, cluster = c12); ordvars = [[1e-12, 0.3], [0.4, 1e-12]])
        ws = [w for w in _warns(run(fx)[2]) if occursin("boundary", w)]
        @test length(ws) == 2
        wu = only(w for w in ws if occursin(":unit", w)); wc = only(w for w in ws if occursin(":cluster", w))
        @test occursin("trait 1", wu) && !occursin("trait 2", wu) && occursin("trait 2", wc) && !occursin("trait 1", wc)
        # the reference total includes the residual and every other ordinary term
        fres = _jpg_hand(phy4, sid4, (unit = u12,); beta = [0.4, -0.3], loading = L2, unique = nothing,
                         ordvars = [[1e-7, 0.3]], residual = [1000.0, 0.6])
        ws = [w for w in _warns(run(fres)[2]) if occursin("boundary", w)]
        @test length(ws) == 1 && occursin("trait 1", ws[1])
        foth = mk((unit = u12, cluster = c12); ordvars = [[1e-7, 0.3], [1000.0, 0.4]])
        ws = [w for w in _warns(run(foth)[2]) if occursin("boundary", w)]
        @test length(ws) == 1 && occursin(":unit", ws[1])
        # the warning comes after validation: a refused call does not emit it
        fr = mk((unit = ones(Int, 12),); ordvars = [[1e-12, 0.3]])
        _, lgr = _logged(() -> err(() -> _bsig(fr; level = :phy, n_boot = 3, progress = false, _refit = stub(fr))))
        @test !any(occursin("boundary", w) for w in _warns(lgr))
    end
end
# END JPG-TIER

# DRIVER-FIXES: refit warning flood (F1), String level/what (F4), `what` echo and refusal text (F5)
@testset "bootstrap_Sigma_r driver fixes (warning flood, String level, what echo)" begin
    phy8 = PrecisionPhy(GLLVModels.random_balanced_tree(8; branch_length = 0.35))
    sid8 = repeat(1:8, inner = 2)
    Y8 = _pmv_truth_data(MersenneTwister(20261103), phy8, sid8; beta = [0.4, -0.3],
                         loading = reshape([0.8, -0.5], 2, 1), unique = [0.3, 0.2], residual = [0.1, 0.12])
    fp = _quiet(() -> _PG.fit_precision_multivariate(Y8, phy8; rank = 1, mode = :explicitunique, species_id = sid8))
    stub(f) = (Y, phy; kw...) -> f

    @testset "F1 PMV refit warnings do not flood; the driver's own warnings stay" begin
        r, lg = _logged(() -> _bsig(fp; level = :phy, n_boot = 3, seed = 1, progress = false),
                        min_level = Logging.Warn)
        @test r isa GLLVModels.BootstrapSigmaR
        ws = _warns(lg)
        @test length(ws) <= 5
        @test !any(occursin("structurally non-identifiable", m) for m in ws)
        @test any(occursin("below min_boot", m) for m in ws)          # the driver's own warning is kept
        # a failed refit is still counted and announced by the driver
        _, lg2 = _logged(() -> _bsig(fp; level = :phy, n_boot = 3, seed = 1, progress = false,
                                     _refit = (Y, phy; kw...) -> error("boom")), min_level = Logging.Warn)
        @test any(occursin("3 of 3 replicates were lost", m) for m in _warns(lg2))
    end

    @testset "F4 String / vector-of-String level and what are accepted" begin
        ref = _quiet(() -> _bsig(fp; level = :phy, what = :Sigma, n_boot = 3, seed = 1, progress = false, _refit = stub(fp)))
        for lv in ("phy", ["phy"], [:phy], :phy), wh in ("Sigma", ["Sigma"], [:Sigma])
            r = _quiet(() -> _bsig(fp; level = lv, what = wh, n_boot = 3, seed = 1, progress = false, _refit = stub(fp)))
            @test r.level == [:phy] && r.what == [:Sigma]
            @test r.point_est.Sigma_phy == ref.point_est.Sigma_phy
        end
        e = try _bsig(fp; level = "nope", n_boot = 3, progress = false, _refit = stub(fp)); nothing catch ex; ex end
        @test e isa ArgumentError && occursin("unknown level :nope", e.msg)
        e = try _bsig(fp; level = :phy, what = "nope", n_boot = 3, progress = false, _refit = stub(fp)); nothing catch ex; ex end
        @test e isa ArgumentError && occursin("unknown summary :nope", e.msg)
    end

    @testset "F5 what echoes the request; refusal for ICC / cross_corr alone names the right classes" begin
        r = _quiet(() -> _bsig(fp; level = :phy, n_boot = 3, seed = 1, progress = false, _refit = stub(fp)))
        @test :cross_corr in r.what                                   # documented: echoes the request, mirrors R
        @test collect(keys(r.point_est)) == [:Sigma_phy, :R_phy, :communality_phy]   # authoritative list
        for w in (:ICC, :cross_corr)
            e = try _bsig(fp; level = :phy, what = w, n_boot = 3, progress = false, _refit = stub(fp)); nothing catch ex; ex end
            @test e isa ArgumentError && !occursin("()", e.msg)
            @test occursin("PrecisionMultivariateFit", e.msg) && occursin("TwoLevelFit", e.msg)
            @test occursin("JointPhyloGroupedGaussianFit", e.msg)
        end
    end
end

# N-CORES: replicates on Julia threads. Draws must be bit-identical to n_cores = 1 for the same seed.
@testset "bootstrap_Sigma_r n_cores (threaded replicates)" begin
    nth = Threads.nthreads()
    cores = [k for k in (1, 2, 4) if k <= nth]       # the threaded legs need `julia -t 4`
    nth < 4 && @info "n_cores testset: only n_cores $(cores) can run with Threads.nthreads() = $nth (start Julia with -t 4)"
    same(a, b) = all(isequal(getfield(a, f), getfield(b, f)) for f in
        (:point_est, :ci_lower, :ci_upper, :n_effective, :n_boot, :n_failed, :level, :what, :draws, :diagnostics))
    # run with n_cores = 1 as reference, then every available k; the logs (warnings) must match too
    function agree(mk; min_level = Logging.Warn)
        ref, lg1 = _logged(() -> mk(1), min_level = min_level)
        ok = true
        for k in cores
            k == 1 && continue
            r, lgk = _logged(() -> mk(k), min_level = min_level)
            ok &= same(ref, r) && sort(_warns(lg1)) == sort(_warns(lgk))
        end
        return ok, ref, _warns(lg1)
    end

    @testset "N1 refusals: n_cores < 1, above Threads.nthreads(), non-integer" begin
        rng = MersenneTwister(1); yy = randn(rng, 4, 40)
        ff = fit_gaussian_gllvm(yy; K = 1)
        for k in (0, -1)
            e = try _bsig(ff; y = yy, n_boot = 2, n_cores = k, progress = false); nothing catch ex; ex end
            @test e isa ArgumentError && occursin("n_cores must be >= 1", e.msg)
        end
        e = try _bsig(ff; y = yy, n_boot = 2, n_cores = nth + 1, progress = false); nothing catch ex; ex end
        @test e isa ArgumentError && occursin("exceeds Threads.nthreads() = $nth", e.msg) && occursin("-t", e.msg)
        @test_throws TypeError _bsig(ff; y = yy, n_boot = 2, n_cores = 1.5, progress = false)
        # the refusal happens before any refit
        n = Threads.Atomic{Int}(0)
        try _bsig(ff; y = yy, n_boot = 2, n_cores = nth + 1, progress = false, _refit = (yb; kw...) -> (Threads.atomic_add!(n, 1); ff)) catch; end
        @test n[] == 0
    end

    @testset "N2 GllvmFit stub: identical draws and failure count for n_cores in $(cores)" begin
        rng = MersenneTwister(20261102); yy = [0.9, 0.6, 0.4, -0.5] * randn(rng, 1, 120) + 0.5 * randn(rng, 4, 120)
        ff = fit_gaussian_gllvm(yy; K = 1)
        # depends on the replicate data only (never on call order): ok iff hash(Y) % 4 != 0, scale varies with Y
        stub = (yb; kw...) -> _scaled(ff, 1.0 + 0.2 * abs(mean(yb)) * 5; ok = hash(yb) % 4 != 0)
        ok, ref, ws = agree(k -> _bsig(ff; y = yy, what = [:Sigma, :communality], n_boot = 41, seed = 7, keep_draws = true,
                                       progress = false, n_cores = k, _refit = stub))
        @test ok
        @test 0 < ref.n_failed < 41                      # the control is live: some replicates fail, some do not
        @test any(occursin("replicates were lost", w) && occursin("$(ref.n_failed) failed refits", w) for w in ws)
        @test count(d -> all(isnan, d.Sigma_B), ref.draws) == ref.n_failed
    end

    @testset "N3 TwoLevelFit stub" begin
        f = _mk2()
        stub = (yb, i; kw...) -> _mk2(0.8 + abs(mean(yb)), 1.0 + abs(mean(yb[:, 1:2:end])); ok = hash(yb) % 5 != 0)
        ok, ref, _ = agree(k -> _bsig(f; individual = _IND, n_boot = 25, seed = 3, keep_draws = true, progress = false,
                                      n_cores = k, _refit = stub))
        @test ok && 0 < ref.n_failed < 25
    end

    @testset "N4 PMV and JPG stubs (thrown, non-converged and converged mixed)" begin
        phy8 = PrecisionPhy(GLLVModels.random_balanced_tree(8; branch_length = 0.35)); sid8 = repeat(1:8, inner = 2)
        Y8 = _pmv_truth_data(MersenneTwister(20261103), phy8, sid8; beta = [0.4, -0.3],
                             loading = reshape([0.8, -0.5], 2, 1), unique = [0.3, 0.2], residual = [0.1, 0.12])
        fp = _quiet(() -> _PG.fit_precision_multivariate(Y8, phy8; rank = 1, mode = :explicitunique, species_id = sid8))
        fpn = _quiet(() -> _PG.fit_precision_multivariate(Y8, phy8; rank = 1, mode = :explicitunique, species_id = sid8, iterations = 1))
        @test fp.converged && !fpn.converged
        pstub = (Y, phy; kw...) -> (h = hash(Y) % 3; h == 0 ? error("boom") : h == 1 ? fpn : fp)
        ok, ref, _ = agree(k -> _bsig(fp; level = :phy, n_boot = 30, seed = 5, keep_draws = true, progress = false,
                                      n_cores = k, _refit = pstub))
        @test ok && ref.n_failed > 0 && ref.n_failed < 30

        phy6 = PrecisionPhy(GLLVModels.random_balanced_tree(6; branch_length = 0.35)); sid6 = repeat(1:6, inner = 4)
        uni = repeat(1:8, inner = 3)
        Yr = _jpg_truth_data(MersenneTwister(20261103), phy6, sid6, (unit = uni,); beta = [0.4, -0.3, 0.7],
                             loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = nothing, ordvars = [[0.25, 0.20, 0.30]],
                             residual = [0.30, 0.40, 0.35])
        fj = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yr, phy6; rank = 1, phylo_mode = :barelowrank,
                terms = _jpg_terms([:unit]), unit = uni, species_id = sid6))
        jstub = (Y, phy; kw...) -> hash(Y) % 4 == 0 ? error("boom") : fj
        ok, ref, _ = agree(k -> _bsig(fj; level = :phy, n_boot = 30, seed = 5, keep_draws = true, progress = false,
                                      n_cores = k, _refit = jstub))
        @test ok && 0 < ref.n_failed < 30
        @test ref.diagnostics.n_thrown == ref.n_failed          # JPG status counters also reduce identically
    end

    @testset "N5 real refits: GllvmFit (rr and has_diag), TwoLevelFit, PMV, JPG" begin
        rng = MersenneTwister(20261101)
        yy = [0.9, 0.6, 0.4, -0.5] * randn(rng, 1, 150) + 0.5 * randn(rng, 4, 150)
        f1 = fit_gaussian_gllvm(yy; K = 1); f2 = fit_gaussian_gllvm(yy; K = 1, has_diag = true)
        for ff in (f1, f2)
            ok, ref, _ = agree(k -> _bsig(ff; y = yy, what = [:Sigma, :communality], n_boot = 12, seed = 11,
                                          keep_draws = true, progress = false, n_cores = k))
            @test ok && ref.n_failed == 0
        end
        pp, nind, nobs = 3, 20, 3; ind = repeat(1:nind, inner = nobs)
        ΛB = reshape([0.9, 0.6, -0.4], pp, 1); ΛW = reshape([0.4, -0.3, 0.5], pp, 1)
        Y = Matrix{Float64}(undef, pp, nind * nobs)
        for i in 1:nind
            b = ΛB * randn(rng, 1) + sqrt.([0.3, 0.2, 0.25]) .* randn(rng, pp)
            for j in ((i - 1) * nobs + 1):(i * nobs)
                Y[:, j] = b + ΛW * randn(rng, 1) + sqrt.([0.5, 0.4, 0.45]) .* randn(rng, pp)
            end
        end
        tf = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)
        ok, ref, _ = agree(k -> _bsig(tf; individual = ind, n_boot = 8, seed = 2, keep_draws = true, progress = false, n_cores = k))
        @test ok && ref.n_failed == 0

        phy8 = PrecisionPhy(GLLVModels.random_balanced_tree(8; branch_length = 0.35)); sid8 = repeat(1:8, inner = 2)
        Y8 = _pmv_truth_data(MersenneTwister(20261103), phy8, sid8; beta = [0.4, -0.3],
                             loading = reshape([0.8, -0.5], 2, 1), unique = [0.3, 0.2], residual = [0.1, 0.12])
        fp = _quiet(() -> _PG.fit_precision_multivariate(Y8, phy8; rank = 1, mode = :explicitunique, species_id = sid8))
        ok, ref, _ = agree(k -> _bsig(fp; level = :phy, n_boot = 8, seed = 4, keep_draws = true, progress = false, n_cores = k))
        @test ok

        phy6 = PrecisionPhy(GLLVModels.random_balanced_tree(6; branch_length = 0.35)); sid6 = repeat(1:6, inner = 4)
        uni = repeat(1:8, inner = 3)
        Yr = _jpg_truth_data(MersenneTwister(20261103), phy6, sid6, (unit = uni,); beta = [0.4, -0.3, 0.7],
                             loading = reshape([0.8, -0.5, 0.6], 3, 1), uniq = nothing, ordvars = [[0.25, 0.20, 0.30]],
                             residual = [0.30, 0.40, 0.35])
        fj = _quiet(() -> _PG.fit_joint_phylo_grouped_gaussian(Yr, phy6; rank = 1, phylo_mode = :barelowrank,
                terms = _jpg_terms([:unit]), unit = uni, species_id = sid6))
        ok, ref, _ = agree(k -> _bsig(fj; level = :phy, n_boot = 8, seed = 4, keep_draws = true, progress = false, n_cores = k))
        @test ok
    end

    @testset "N6 a throwing refit in one thread is one failed replicate; InterruptException propagates" begin
        rng = MersenneTwister(5); yy = randn(rng, 4, 60); ff = fit_gaussian_gllvm(yy; K = 1)
        bad = Ref(UInt64(0))
        # first replicate-data hash is the thrower: found with a serial dry run so the test is order-free
        seen = UInt64[]
        _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 9, seed = 2, progress = false,
                           _refit = (yb; kw...) -> (push!(seen, hash(yb)); ff)))
        bad[] = seen[5]
        thrower = (yb; kw...) -> hash(yb) == bad[] ? error("injected") : ff
        ok, ref, _ = agree(k -> _bsig(ff; y = yy, what = :Sigma, n_boot = 9, seed = 2, keep_draws = true, progress = false,
                                      n_cores = k, _refit = thrower))
        @test ok && ref.n_failed == 1 && all(isnan, ref.draws[5].Sigma_B) && all(isfinite, ref.draws[4].Sigma_B)
        for k in cores
            k == 1 && continue
            @test_throws InterruptException _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 9, seed = 2, progress = false,
                n_cores = k, _refit = (yb; kw...) -> hash(yb) == bad[] ? throw(InterruptException()) : ff))
                        boom = GllvmFit(ff.model, merge(ff.pars, (Λ = _BsigBoom(4),)), ff.logLik, 1, true, ff.optim_result, 0.0)
            @test_throws InterruptException _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 9, seed = 2, progress = false,
                n_cores = k, _refit = (yb; kw...) -> hash(yb) == bad[] ? boom : ff))
        end
    end

    @testset "N7 progress and logger handling under threads" begin
        rng = MersenneTwister(6); yy = randn(rng, 4, 60); ff = fit_gaussian_gllvm(yy; K = 1)
        for k in cores
            _, lg = _logged(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 12, seed = 1, progress = true, n_cores = k,
                                        _refit = (yb; kw...) -> ff))
            msgs = [m for m in _infos(lg) if occursin("replicate", m) && !occursin("replicates", m)]
            @test sort(msgs) == sort(["bootstrap_Sigma_r: replicate $b / 12" for b in 1:12])   # each exactly once
            # a NullLogger around the call silences every thread (task-inherited logger)
            @test _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 6, seed = 1, progress = true, n_cores = k,
                                     _refit = (yb; kw...) -> ff)) isa GLLVModels.BootstrapSigmaR
        end
    end

    @testset "N8 real PMV refit emits no structural-redundancy warning flood on threads" begin
        phy8 = PrecisionPhy(GLLVModels.random_balanced_tree(8; branch_length = 0.35)); sid8 = repeat(1:8, inner = 2)
        Y8 = _pmv_truth_data(MersenneTwister(20261103), phy8, sid8; beta = [0.4, -0.3],
                             loading = reshape([0.8, -0.5], 2, 1), unique = [0.3, 0.2], residual = [0.1, 0.12])
        fp = _quiet(() -> _PG.fit_precision_multivariate(Y8, phy8; rank = 1, mode = :explicitunique, species_id = sid8))
        for k in cores
            _, lg = _logged(() -> _bsig(fp; level = :phy, n_boot = 6, seed = 1, progress = false, n_cores = k), min_level = Logging.Warn)
            @test length(_warns(lg)) <= 5 && !any(occursin("structurally non-identifiable", m) for m in _warns(lg))
        end
    end

    @testset "N9 exactly n_cores workers run concurrently (the threaded path is really used)" begin
        rng = MersenneTwister(8); yy = randn(rng, 4, 60); ff = fit_gaussian_gllvm(yy; K = 1)
        for k in cores
            lk = ReentrantLock(); tasks = Set{UInt}(); inflight = Threads.Atomic{Int}(0); peak = Threads.Atomic{Int}(0)
            stub = function (yb; kw...)
                n = Threads.atomic_add!(inflight, 1) + 1
                Threads.atomic_max!(peak, n)
                lock(() -> push!(tasks, objectid(current_task())), lk)
                sleep(0.05)
                Threads.atomic_sub!(inflight, 1)
                return ff
            end
            _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 16, seed = 1, progress = false, n_cores = k, _refit = stub))
            @test length(tasks) == k && peak[] == k
        end
    end

    @testset "N10 every replicate is refit exactly once under contention (fast stub, many replicates)" begin
        rng = MersenneTwister(9); yy = randn(rng, 4, 30); ff = fit_gaussian_gllvm(yy; K = 1)
        for k in cores
            k == 1 && continue
            calls = Threads.Atomic{Int}(0)
            r = _quiet(() -> _bsig(ff; y = yy, what = :communality, n_boot = 3000, seed = 1, progress = false, n_cores = k,
                                   _refit = (yb; kw...) -> (Threads.atomic_add!(calls, 1); ff)))
            @test calls[] == 3000 && r.n_failed == 0 && all(==(3000), r.n_effective.communality_B)
        end
    end

    @testset "N11 an interrupt on the MAIN task stops the workers (no orphans)" begin
        kk = maximum(cores)
        if kk > 1
            rng = MersenneTwister(10); yy = randn(rng, 4, 30); ff = fit_gaussian_gllvm(yy; K = 1)
            done = Threads.Atomic{Int}(0)
            main = current_task()
            Threads.@spawn (sleep(0.4); schedule(main, InterruptException(); error = true))
            @test_throws InterruptException _quiet(() -> _bsig(ff; y = yy, what = :Sigma, n_boot = 400, seed = 1,
                progress = false, n_cores = kk, _refit = (yb; kw...) -> (sleep(0.05); Threads.atomic_add!(done, 1); ff)))
            at_return = done[]
            sleep(1.0)
            @test done[] == at_return            # nothing kept running after the call returned
            @test 0 < at_return < 100            # and it stopped early (400 replicates would take about 5 s at kk = 4)
        end
    end

    @testset "N12 the fitter's warn-once identification warning is emitted exactly once under threads" begin
        kk = maximum(cores)
        if kk > 1
            # direct hammer: kk tasks released together call the helper with the key un-set, many trials
            nwarn = Int[]
            for trial in 1:300
                GLLVModels._reset_unidentified_warnings!()
                go = Threads.Atomic{Bool}(false)
                lg = Test.TestLogger(min_level = Logging.Warn)
                with_logger(lg) do
                    ts = [Threads.@spawn((while !go[]; end; GLLVModels._warn_unidentified_once(:has_diag, "hammer"))) for _ in 1:kk]
                    go[] = true
                    foreach(wait, ts)
                end
                push!(nwarn, length(lg.logs))
            end
            @test all(==(1), nwarn)
            # through the driver: re-armed set, real has_diag refits
            rng = MersenneTwister(11); yy = [0.9, 0.6, 0.4, -0.5] * randn(rng, 1, 120) + 0.5 * randn(rng, 4, 120)
            fd = _quiet(() -> fit_gaussian_gllvm(yy; K = 1, has_diag = true))
            for trial in 1:5
                GLLVModels._reset_unidentified_warnings!()
                _, lg = _logged(() -> _bsig(fd; y = yy, what = :Sigma, n_boot = 16, seed = trial, progress = false, n_cores = kk),
                                min_level = Logging.Warn)
                @test count(occursin("has_diag = true with one observation", m) for m in _warns(lg)) == 1
            end
        end
    end
end
