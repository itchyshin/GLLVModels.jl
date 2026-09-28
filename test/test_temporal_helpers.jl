# gllvm-parity-tag: P1
#
# Temporal helper routes against gllvmTMB P1 (9539352f6): forecast_temporal,
# profile_temporal, compare_temporal, bootstrap_temporal, simulate, in-sample
# predict and the refusal set. Spec section 4.1 rows program-forecast.R:47,
# :85, :104; program-profile.R:1, :30; program-selection.R:1;
# program-bootstrap.R:1, :18; program-simulation.R:32, :41; ar1-methods.R:20
# (predict, simulate and refusals). R numbers are read from
# test/fixtures/temporal_p1/{forecast,profile,compare,bootstrap}.toml.
using Test, GLLVModels, LinearAlgebra, Random, Statistics
const GMH = GLLVModels
include(joinpath(@__DIR__, "fixtures", "temporal_p1", "fixture_helpers.jl"))

helper_refuses(f, pattern; kind=:rlang_error) = begin
    err = try; f(); nothing; catch e; e; end
    err isa TemporalContractError && occursin(pattern, err.message) && err.kind === kind
end
# A random stream of zeros: the simulated draw then equals its centre.
struct TemporalZeroRNG <: Random.AbstractRNG end
Random.randn(::TemporalZeroRNG, n::Integer) = zeros(n)
Random.randn(::TemporalZeroRNG) = 0.0
theta_scale(structure, v) = ismissing(v) ? missing :
    structure == "ar1" ? atanh(v / (1 - 1e-6)) : log(v)

@testset "temporal helpers vs gllvmTMB P1" begin
    F = temporal_p1_load("fits.toml")
    fit_row(id) = only(filter(c -> c["id"] == id, F["fits"]))
    at_r(id) = (c = fit_row(id); temporal_p1_fit(F, c; start=temporal_p1_vec(c["par"]), iterations=0))

    @testset "receipt integrity" begin
        for name in ("forecast.toml", "profile.toml", "compare.toml", "bootstrap.toml")
            @test temporal_p1_sha(name) == TEMPORAL_P1_SHA256[name]
            @test temporal_p1_load(name)["gllvmTMB_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        end
    end

    @testset "forecast_temporal" begin
        R = temporal_p1_load("forecast.toml")
        # program-forecast.R:47 and :104 at R's coordinates: est and se.fit to 1e-8.
        fa = at_r(R["ar1"]["fit_id"])
        future = (series=String.(R["ar1"]["future_series"]), occasion=temporal_p1_vec(R["ar1"]["future_occasion"]),
            trait=String.(R["ar1"]["future_trait"]))
        fc = forecast_temporal(fa, future; se_fit=true)
        @test maximum(abs, fc.est .- temporal_p1_vec(R["ar1"]["est"])) <= 1e-8
        @test maximum(abs, fc.se_fit .- temporal_p1_vec(R["ar1"]["se_fit"])) <= 1e-8
        @test keys(fc)[1:3] == (:series, :occasion, :trait)
        c = fit_row(R["ar1"]["fit_id"])
        neg = temporal_p1_fit(F, c; start=temporal_p1_vec(R["ar1"]["negative_par"]), iterations=0)
        @test neg.time_value ≈ -0.6 atol = 1e-12
        fn = forecast_temporal(neg, future; se_fit=true)
        @test maximum(abs, fn.est .- temporal_p1_vec(R["ar1"]["negative_est"])) <= 1e-8
        @test maximum(abs, fn.se_fit .- temporal_p1_vec(R["ar1"]["negative_se_fit"])) <= 1e-8
        fo = at_r(R["ou"]["fit_id"])
        future_ou = (series=String.(R["ou"]["future_series"]), elapsed=temporal_p1_vec(R["ou"]["future_elapsed"]),
            trait=String.(R["ou"]["future_trait"]))
        fco = forecast_temporal(fo, future_ou; se_fit=true)
        @test maximum(abs, fco.est .- temporal_p1_vec(R["ou"]["est"])) <= 1e-8
        @test maximum(abs, fco.se_fit .- temporal_p1_vec(R["ou"]["se_fit"])) <= 1e-8

        # Independent dense conditioning on a Julia fit (program-forecast.R:47).
        jf = temporal_p1_fit(F, c)
        obs = jf.data
        all_series = vcat(obs.series, future.series); all_time = vcat(obs.occasion, future.occasion)
        all_trait = vcat(obs.trait, future.trait)
        traits = sort(unique(all_trait)); tr = [findfirst(==(t), traits) for t in all_trait]
        n_o = length(obs.value); m = length(all_series)
        phi = jf.time_value
        Vall = [all_series[i] == all_series[j] ? phi^abs(all_time[i] - all_time[j]) * jf.Sigma_T[tr[i], tr[j]] : 0.0
                for i in 1:m, j in 1:m] + jf.sigma_eps^2 * I
        Voo = Vall[1:n_o, 1:n_o]; Von = Vall[1:n_o, n_o+1:end]; Vnn = Vall[n_o+1:end, n_o+1:end]
        mu = [jf.beta[k] for k in tr]
        expected = mu[n_o+1:end] .+ Von' * (Voo \ (obs.value .- mu[1:n_o]))
        expected_sd = sqrt.(max.(diag(Vnn .- Von' * (Voo \ Von)), 0))
        jfc = forecast_temporal(jf, future; se_fit=true)
        @test maximum(abs, jfc.est .- expected) <= 1e-8
        @test maximum(abs, jfc.se_fit .- expected_sd) <= 1e-8

        # Layout refusals (program-forecast.R:85) with R's classes.
        base = (series=future_ou.series, elapsed=future_ou.elapsed, trait=future_ou.trait)
        @test helper_refuses(() -> forecast_temporal(fo, merge(base, (series=fill("new_series", 6),))),
            "existing series"; kind=:gllvmTMB_temporal_forecast_new_series)
        @test helper_refuses(() -> forecast_temporal(fo, merge(base, (elapsed=fill(1.0, 6),))),
            "strictly after"; kind=:gllvmTMB_temporal_forecast_not_future)
        @test helper_refuses(() -> forecast_temporal(fo, (; (k => v[2:end] for (k, v) in pairs(base))...)),
            "complete trait panel"; kind=:gllvmTMB_temporal_forecast_panel)
        dep = temporal_p1_fit(F, fit_row("profile_ar1__dep__ar1"))
        @test helper_refuses(() -> forecast_temporal(dep, future), "temporal_indep()` only";
            kind=:gllvmTMB_temporal_forecast_mode)
        rep = temporal_p1_fit(F, fit_row("sim_rep__indep__ar1"))
        @test helper_refuses(() -> forecast_temporal(rep, future), "replicated temporal panels";
            kind=:gllvmTMB_temporal_forecast_replicated)
        @test helper_refuses(() -> forecast_temporal(fa, future; se_fit=1), "TRUE or FALSE")
        @test helper_refuses(() -> forecast_temporal(fa, merge(future, (occasion=future.occasion .+ 0.5,))),
            "integer occasions")

        # OU forecasts are invariant to a time-origin shift (program-forecast.R:104).
        shifted_data = merge(fo.data, (elapsed=fo.data.elapsed .+ 100,))
        cfo = fit_row(R["ou"]["fit_id"])
        fo_shift = fit_temporal_gllvm(shifted_data; formula=@formula(value ~ 0 + trait),
            temporal=temporal_p1_term(cfo), start=fo.parameters, iterations=0)
        fs = forecast_temporal(fo_shift, merge(future_ou, (elapsed=future_ou.elapsed .+ 100,)); se_fit=true)
        @test maximum(abs, fs.est .- fco.est) <= 1e-10
        @test maximum(abs, fs.se_fit .- fco.se_fit) <= 1e-10
    end

    @testset "profile_temporal" begin
        P = temporal_p1_load("profile.toml")
        worst_trace = 0.0
        for key in ("ar1", "ar1_constrained", "ou", "ar1_default")
            r = P[key]; c = fit_row(r["fit_id"])
            f = at_r(r["fit_id"])
            idx = GMH.TemporalLayout(size(f.X, 2), f.spec).time
            pr = isempty(r["parm_range_offset"]) ? (-Inf, Inf) :
                Tuple(f.parameters[idx] .+ temporal_p1_vec(r["parm_range_offset"]))
            out = profile_temporal(f; ystep=r["ystep"], ytol=r["ytol"], parm_range=pr)
            @test keys(out) == (:estimate, :lower, :upper)
            @test abs(out.estimate - r["estimate"]) <= 1e-8
            for side in (:lower, :upper)
                rv = temporal_p1_num(r[String(side)]); jv = getproperty(out, side)
                @test ismissing(rv) == ismissing(jv)
                if !ismissing(rv) && !ismissing(jv)
                    @test isinf(rv) == isinf(jv)
                    if isfinite(rv)
                        @test abs(theta_scale(c["structure"], jv) - theta_scale(c["structure"], rv)) <= 1e-4
                    else
                        @test jv == rv
                    end
                end
            end
            # The walk visits the same displacements as TMB::tmbprofile.
            tr = GMH._temporal_tmbprofile(f; ystep=r["ystep"], ytol=r["ytol"], parm_range=pr)
            rt = temporal_p1_vec(r["trace_theta"])
            @test length(tr.theta) == length(rt)
            if length(tr.theta) == length(rt)
                @test maximum(abs, tr.theta .- rt) <= 1e-10
                rv = [temporal_p1_num(v) for v in r["trace_value"]]
                @test all(ismissing.(rv) .== ismissing.(tr.value))
                d = maximum(abs, skipmissing(rv .- tr.value))
                @test d <= 1e-5
                worst_trace = max(worst_trace, d)
            end
        end
        println("profile trace max |value_J - value_R| = ", worst_trace)

        # program-profile.R:1 on a Julia fit.
        fj = temporal_p1_fit(F, fit_row("profile_ar1__indep__ar1"))
        idx = GMH.TemporalLayout(size(fj.X, 2), fj.spec).time
        out = profile_temporal(fj; ystep=0.25, ytol=1)
        @test out.estimate ≈ (1 - 1e-6) * tanh(fj.parameters[idx]) atol = 1e-10
        tr = GMH._temporal_tmbprofile(fj; ystep=0.25, ytol=1)
        at_mle = argmin(abs.(tr.theta .- fj.parameters[idx]))
        @test abs(tr.value[at_mle] - (-fj.loglik)) <= 1e-8
        @test maximum(skipmissing(tr.value)) > -fj.loglik + 0.1
        constrained = profile_temporal(fj; ystep=0.1, ytol=1,
            parm_range=(fj.parameters[idx] - 0.01, fj.parameters[idx] + 0.01))
        @test ismissing(constrained.lower) && ismissing(constrained.upper)
        @test helper_refuses(() -> profile_temporal(temporal_p1_fit(F, fit_row("profile_ar1__dep__ar1"))),
            "temporal_indep")
        @test helper_refuses(() -> profile_temporal(temporal_p1_fit(F, fit_row("sim_rep__indep__ar1"))),
            "unreplicated Gaussian")

        # program-profile.R:30: OU profiles are invariant to a time-origin shift.
        cou = fit_row("profile_ou__indep__ou")
        fou = temporal_p1_fit(F, cou)
        tbl = temporal_p1_table(F["datasets"]["profile_ou"])
        fsh = fit_temporal_gllvm(merge(tbl, (elapsed=tbl.elapsed .+ 100,)); formula=@formula(value ~ 0 + trait),
            temporal=temporal_p1_term(cou))
        @test abs(fou.loglik - fsh.loglik) <= 1e-8
        p1 = profile_temporal(fou; ystep=0.25, ytol=1); p2 = profile_temporal(fsh; ystep=0.25, ytol=1)
        for k in (:estimate, :lower, :upper)
            a = getproperty(p1, k); b = getproperty(p2, k)
            @test ismissing(a) == ismissing(b)
            ismissing(a) || @test (isinf(a) ? a == b : abs(a - b) <= 1e-8 * max(1, abs(a)))
        end
    end

    @testset "compare_temporal" begin
        C = temporal_p1_load("compare.toml")
        for key in ("selection", "all_modes", "replicated")
            r = C[key]
            fs = [at_r(id) for id in r["fit_ids"]]
            out = compare_temporal(; (Symbol(m) => f for (m, f) in zip(r["model"], fs))...)
            @test collect(String.(keys(out))) == String.(r["columns"])
            @test out.model == String.(r["model"])
            @test maximum(abs, out.logLik .- temporal_p1_vec(r["logLik"])) <= 1e-8
            @test out.df == Int.(r["df"])
            @test maximum(abs, out.AIC .- temporal_p1_vec(r["AIC"])) <= 1e-8
        end
        # program-selection.R:1 semantics on Julia fits: AIC = -2 logLik + 2 df, no LRT.
        ar1 = temporal_p1_fit(F, fit_row("selection__indep__ar1"))
        ou = temporal_p1_fit(F, fit_row("selection__indep__ou"))
        out = compare_temporal(ar1=ar1, ou=ou)
        @test keys(out) == (:model, :logLik, :df, :AIC, :convergence)
        @test all(isfinite, out.AIC)
        @test out.logLik == [ar1.loglik, ou.loglik]
        @test out.df == [length(ar1.parameters), length(ou.parameters)]
        @test maximum(abs, out.AIC .- (-2 .* out.logLik .+ 2 .* out.df)) <= 1e-12
        @test !haskey(out, :p_value)
        @test helper_refuses(() -> compare_temporal(ar1=ar1), "at least two named")
        @test helper_refuses(() -> compare_temporal(ar1, ou), "must be named")
        @test helper_refuses(() -> compare_temporal(a=ar1, b=fit_gaussian_gllvm(randn(3, 10); K=1)),
            "native temporal fit")
        other = temporal_p1_fit(F, fit_row("profile_ar1__indep__ar1"))
        @test helper_refuses(() -> compare_temporal(a=ar1, b=other), "identical response rows")
    end

    @testset "bootstrap_temporal" begin
        # program-bootstrap.R:1: every attempt is retained.
        sel = temporal_p1_fit(F, fit_row("selection__indep__ar1"))
        out = bootstrap_temporal(sel; n_boot=2, seed=7)
        @test out.replicate == [1, 2]
        @test keys(out) == (:replicate, :seed, :convergence, :objective, :time_estimate, :error)
        @test all(isfinite, out.seed)
        @test all(i -> (!ismissing(out.objective[i]) && isfinite(out.objective[i])) || !isempty(out.error[i]), 1:2)
        @test helper_refuses(() -> bootstrap_temporal(temporal_p1_fit(F, fit_row("profile_ar1__dep__ar1")); n_boot=1),
            "temporal_indep")
        @test helper_refuses(() -> bootstrap_temporal(sel; n_boot=0), "positive integer")
        @test helper_refuses(() -> bootstrap_temporal(sel; n_boot=1, seed=-1), "non-negative whole number")
        # program-bootstrap.R:18: reproducible seeds, OU scale, caller RNG untouched.
        fou = temporal_p1_fit(F, fit_row("profile_ou__indep__ou"))
        a = bootstrap_temporal(fou; n_boot=2, seed=260914)
        b = bootstrap_temporal(fou; n_boot=2, seed=260914)
        @test a.seed == b.seed
        @test all(i -> ismissing(a.time_estimate[i]) ? ismissing(b.time_estimate[i]) :
            abs(a.time_estimate[i] - b.time_estimate[i]) <= 1e-10, 1:2)
        @test all(a.time_estimate[i] > 0 for i in 1:2 if isempty(a.error[i]))
        Random.seed!(260917); expected_next = rand()
        Random.seed!(260917); bootstrap_temporal(fou; n_boot=1, seed=260914)
        @test rand() == expected_next
        Random.seed!(260917); bootstrap_temporal(fou; n_boot=1)
        @test rand() == expected_next
        idx = GMH.TemporalLayout(size(fou.X, 2), fou.spec).time
        @test abs(fou.time_value - exp(fou.parameters[idx])) <= 1e-12

        # Distribution summary against R's n_boot = 200 run (spec section 5).
        B = temporal_p1_load("bootstrap.toml")
        fi = temporal_p1_fit(F, fit_row(B["fit_id"]))
        jb = bootstrap_temporal(fi; n_boot=200, seed=260931)
        ok = isempty.(jb.error)
        @test sum(ok) >= 190 && B["n_converged"] >= 190
        te = Float64.(jb.time_estimate[ok])
        rmean = B["time_estimate_mean"]; rsd = B["time_estimate_sd"]
        se = sqrt(rsd^2 / B["n_converged"] + std(te)^2 / length(te))
        @test abs(mean(te) - rmean) <= 4 * se
        @test 0.75 <= std(te) / rsd <= 1 / 0.75
        println("bootstrap time_estimate mean/sd: Julia ", mean(te), " / ", std(te), "; R ", rmean, " / ", rsd)
    end

    @testset "simulate and in-sample predict" begin
        # program-simulation.R:32 and :41 with a zero-returning RNG.
        f = temporal_p1_fit(F, fit_row("sim_ar1__latent_unique__ar1"))
        @test maximum(abs, simulate(f; condition_on_RE=false, rng=TemporalZeroRNG())[:, 1] .- f.X * f.beta) <= 1e-12
        pr = predict(f)
        @test maximum(abs, simulate(f; condition_on_RE=true, rng=TemporalZeroRNG())[:, 1] .- pr.est) <= 1e-12
        # ar1-methods.R:20 (predict, simulate shape and refusals).
        @test keys(pr) == (:series, :occasion, :trait, :est)
        @test pr.series == f.data.series && pr.occasion == f.data.occasion && pr.trait == f.data.trait
        @test size(simulate(f; nsim=2, rng=Random.Xoshiro(99))) == (length(f.y), 2)
        @test helper_refuses(() -> predict(f, f.data), r"newdata.*temporal";
            kind=:gllvmTMB_temporal_predict_newdata)
        for g in (() -> confint(f), () -> bootstrap_ci(f), () -> ordination_uncertainty(f))
            @test helper_refuses(g, "not available for `temporal_latent()` fits";
                kind=:gllvmTMB_temporal_inference_unsupported)
        end
        # The unconditional draw has the model's marginal covariance (Monte Carlo check).
        fi = temporal_p1_fit(F, fit_row("forecast__indep__ar1"))
        V = GMH._temporal_covariance(fi.parameters, fi.spec, size(fi.X, 2))
        draws = simulate(fi; nsim=20000, rng=Random.Xoshiro(20260927))
        C = cov(draws; dims=2)
        k = findall(!=(0), V)
        sd_entry = [sqrt((V[i, j]^2 + V[i, i] * V[j, j]) / 20000) for (i, j) in Tuple.(k)]
        @test maximum(abs.(C[k] .- V[k]) ./ sd_entry) <= 5
    end
end

@testset "temporal latent scores (ar1-methods.R:20, engine.R:119)" begin
    F = temporal_p1_load("fits.toml")
    row(id) = only(filter(c -> c["id"] == id, F["fits"]))
    fl = temporal_p1_fit(F, row("sim_ar1__latent_unique__ar1"))
    lv = getLV(fl)
    S = length(fl.spec.pair_table.pair_id)
    @test size(lv.scores) == (S, 1)
    @test lv.pair_index.pair_id == fl.spec.pair_table.pair_id
    @test lv.loadings[1, 1] >= 0                        # first-loading anchor
    @test lv.sign.anchor_trait == "t1"
    # Scores equal the conditional mean of the states given y (dense check).
    V = GMH._temporal_covariance(fl.parameters, fl.spec, size(fl.X, 2))
    r = fl.y .- fl.X * fl.beta
    Czy = [fl.spec.row_series[o] == fl.spec.pair_table.series[s] ?
           fl.time_value^abs(fl.spec.pair_table.time[s] - fl.spec.row_time[o]) * fl.loadings[fl.spec.trait_id[o], 1] : 0.0
           for s in 1:S, o in eachindex(r)]
    @test maximum(abs, lv.scores[:, 1] .- lv.sign.multiplier .* (Czy * (V \ r))) <= 1e-10
    @test getLV(temporal_p1_fit(F, row("sim_ar1__indep__ar1"))) === nothing
    @test getLV(temporal_p1_fit(F, row("sim_ar1__dep__ar1"))) === nothing
    fd = temporal_p1_fit(F, row("sim_ar1__dep__ar1"))
    @test size(extract_temporal(fd).loadings) == (3, 3)
end
