# gllvm-parity-tag: P1
#
# Temporal source beside ordinary unit / unit_obs terms (temporal port slice 2;
# gllvmTMB P1 9539352f6). Spec section 4.1 rows sixth-source-api.R:44,
# sixth-source-engine.R:34, :43, :52, :69, :91, :104,
# program-composed-simulation.R:12, :36, :84, and the partial twins
# program-bootstrap.R:40 and program-selection.R:22 (an ordinary `unit` source
# in place of R's kernel fixture: evidence for the composed refusal only, never
# for the kernel pair). Numeric parity (fixed point, dense oracle, between
# optima) is in test_temporal_composed_receipts.jl.
using Test, GLLVModels, LinearAlgebra, Random, Statistics
const GMS = GLLVModels
include(joinpath(@__DIR__, "fixtures", "temporal_p1", "fixture_helpers.jl"))

composed_refuses(f, pattern; kind=:rlang_error) = begin
    err = try; f(); nothing; catch e; e; end
    err isa TemporalContractError && occursin(pattern, err.message) && err.kind === kind
end
struct ComposedZeroRNG <: Random.AbstractRNG end
Random.randn(::ComposedZeroRNG, n::Integer) = zeros(n)
Random.randn(::ComposedZeroRNG) = 0.0

# R's expand.grid order (series fastest, then occasion, then trait).
function composed_grid(series, occasions; traits=["t1", "t2", "t3"])
    rows = [(s, o, t) for t in traits for o in occasions for s in series]
    return (series=[r[1] for r in rows], occasion=Float64[r[2] for r in rows],
        trait=[r[3] for r in rows])
end
# sixth-source-engine.R:1-10 (value = trait index + occasion / 10).
engine_fixture() = begin
    d = composed_grid(["s1", "s2", "s3"], [1, 3, 7])
    tr = Dict("t1" => 1, "t2" => 2, "t3" => 3)
    merge(d, (unit=copy(d.series), value=[tr[t] + o / 10 for (t, o) in zip(d.trait, d.occasion)]))
end
TI = temporal_indep(:(0 + trait | series), :occasion)
fitc(d; kw...) = fit_temporal_gllvm(d; formula=@formula(value ~ 0 + trait), temporal=TI, kw...)

@testset "temporal + ordinary unit / unit_obs terms" begin
    dat = engine_fixture()

    @testset "series/unit partition only with a stable unit component (api.R:44)" begin
        d = composed_grid(["a", "b"], [1, 3, 8])
        d = merge(d, (unit_group=fill("one_stable_unit", length(d.series)),
            value=collect(1:length(d.series)) ./ 10))
        @test fitc(d; unit=:unit_group) isa TemporalGaussianFit
        @test composed_refuses(() -> fitc(d; unit=:unit_group,
            structure=[:(indep(0 + trait | unit_group))]), r"same partition as.*unit")
    end

    @testset "ordinary unit intercept beside the temporal source (engine.R:34)" begin
        @test fitc(dat; unit=:series, structure=[:(1 | series)]) isa TemporalGaussianFit
    end

    @testset "distinct series label, same unit partition (engine.R:43)" begin
        d = merge(dat, (unit_label="unit-" .* dat.series,))
        @test fitc(d; unit=:unit_label) isa TemporalGaussianFit
    end

    @testset "unit_obs stays unit-nested (engine.R:52)" begin
        d = merge(dat, (within_unit=copy(dat.series),))
        @test fitc(d; unit=:series, unit_obs=:within_unit) isa TemporalGaussianFit
        d = merge(d, (crossed_obs=[isodd(i) ? "shared" : "other" for i in eachindex(d.series)],))
        @test composed_refuses(() -> fitc(d; unit=:series, unit_obs=:crossed_obs), r"unit_obs.*nested")
    end

    @testset "composes with every ordinary unit and unit_obs mode (engine.R:69)" begin
        d = merge(dat, (within_unit=copy(dat.series),))
        for level in (:series, :within_unit), mode in (:indep, :dep, :latent)
            bar = Expr(:call, :|, :(0 + trait), level)
            term = mode === :latent ? Expr(:call, :latent, bar, Expr(:kw, :d, 1)) : Expr(:call, mode, bar)
            f = fitc(d; unit=:series, unit_obs=:within_unit, structure=[term])
            @test f isa TemporalGaussianFit
            @test extract_temporal(f).parameters.mode == "indep"   # the temporal tier stays active
        end
    end

    @testset "ordinary unit ordination beside the temporal source (engine.R:91)" begin
        f = fitc(dat; unit=:series, structure=[:(latent(0 + trait | series, d = 1, unique = false))])
        o = extract_ordination(f; level=:unit)
        @test size(o.scores, 1) == length(unique(dat.series))
        @test o.row_id == sort(unique(dat.series))
        @test !haskey(o, :pair_id)
    end

    @testset "composed simulation redraws every ordinary tier (engine.R:104)" begin
        d = merge(dat, (within_unit=string.(dat.series, " ", Int.(dat.occasion)),))
        f = fitc(d; unit=:series, unit_obs=:within_unit,
            structure=[:(indep(0 + trait | series)), :(indep(0 + trait | within_unit))])
        @test size(simulate(f; nsim=1, rng=Random.Xoshiro(41))) == (length(d.value), 1)
        @test size(simulate(f; nsim=2, condition_on_RE=true, rng=Random.Xoshiro(41))) == (length(d.value), 2)
    end

    # program-composed-simulation.R:1-10 fixture.
    comp = merge(dat, (unit_obs=string.(dat.series, " ", Int.(dat.occasion)),))
    fcomp = fitc(comp; unit=:series, unit_obs=:unit_obs,
        structure=[:(indep(0 + trait | series)), :(indep(0 + trait | unit_obs))])

    @testset "unconditional simulation redraws temporal and ordinary tiers (composed-simulation.R:12)" begin
        draw = simulate(fcomp; nsim=1, condition_on_RE=false, rng=ComposedZeroRNG())
        @test maximum(abs, draw[:, 1] .- fcomp.X * fcomp.beta) <= 1e-12
    end

    @testset "conditional simulation keeps the full fitted predictor (composed-simulation.R:36)" begin
        draw = simulate(fcomp; nsim=1, condition_on_RE=true, rng=ComposedZeroRNG())
        @test maximum(abs, draw[:, 1] .- predict(fcomp).est) <= 1e-12
        # R's report$eta at R's optimum (composed.toml, composed__indep_BW).
        C = temporal_p1_load("composed.toml")
        c = only(filter(r -> r["id"] == "composed__indep_BW", C["fits"]))
        at_r = temporal_p1_composed_fit(C, c; start=temporal_p1_vec(c["par"]), iterations=0)
        rdraw = simulate(at_r; nsim=1, condition_on_RE=true, rng=ComposedZeroRNG())
        @test maximum(abs, rdraw[:, 1] .- temporal_p1_vec(c["eta"])) <= 1e-7
    end

    @testset "composed simulation matches independent Gaussian moments (composed-simulation.R:84)" begin
        # Independent covariance, built from the fitted blocks as R's test does
        # (composed-simulation.R:58-82).
        tr = [findfirst(==(t), fcomp.spec.traits) for t in comp.trait]
        n = length(comp.value)
        V = zeros(n, n)
        psi = fcomp.psi; SB = fcomp.Sigma_B; SW = fcomp.Sigma_W; phi = fcomp.time_value
        for i in 1:n, j in 1:n
            tr[i] == tr[j] || continue
            comp.series[i] == comp.series[j] && (V[i, j] += phi^abs(comp.occasion[i] - comp.occasion[j]) * psi[tr[i]] + SB[tr[i], tr[i]])
            comp.unit_obs[i] == comp.unit_obs[j] && (V[i, j] += SW[tr[i], tr[i]])
        end
        V += fcomp.sigma_eps^2 * I
        draws = simulate(fcomp; nsim=10000, condition_on_RE=false, rng=Random.Xoshiro(260909))
        sampled = cov(draws; dims=2)
        for (i, j) in ((1, 1), (1, 4), (1, 2), (1, 10))
            mc_se = sqrt((V[i, i] * V[j, j] + V[i, j]^2) / (size(draws, 2) - 1))
            @test abs(sampled[i, j] - V[i, j]) <= 4.2 * mc_se
        end
    end

    @testset "helpers refuse composed fits with R's tier names" begin
        fu = fitc(dat; unit=:series, structure=[:(1 | series)])
        @test GMS._temporal_other_tiers(fu) == ["re_int"]
        # program-bootstrap.R:40 (partial twin: unit source in place of kernel_indep).
        @test composed_refuses(() -> bootstrap_temporal(fu; n_boot=1), "temporal source by itself";
            kind=:gllvmTMB_temporal_bootstrap_composed)
        # program-selection.R:22 (partial twin).
        @test composed_refuses(() -> compare_temporal(left=fu, right=fu), "source by itself";
            kind=:gllvmTMB_temporal_selection_composed)
        @test composed_refuses(() -> forecast_temporal(fu, (series=["s1"], occasion=[8.0], trait=["t1"])),
            "temporal source by itself"; kind=:gllvmTMB_temporal_forecast_composed)
        @test composed_refuses(() -> profile_temporal(fu), "requires the temporal source by itself")
        err = try; bootstrap_temporal(fu; n_boot=1); catch e; e; end
        @test occursin("covariance tier(s): re_int", err.message)
        fbw = fcomp
        @test GMS._temporal_other_tiers(fbw) == ["diag_B", "diag_W"]
    end

    @testset "sigma_eps suppression rule (R/fit-multi.R:6959-6967)" begin
        # Unreplicated temporal fit with a per-row unit_obs diagonal: sigma_eps stays free.
        @test "log_sigma_eps" in fcomp.parameter_names
        @test fcomp.spec.composition.sigma_fixed === nothing
        # Replicated fit with a per-row unit_obs diagonal: sigma_eps is fixed and
        # leaves the parameter vector, so df drops by one.
        d = composed_grid(["s1", "s2", "s3"], [1, 2, 3])
        d = (series=repeat(d.series, 2), occasion=repeat(d.occasion, 2), trait=repeat(d.trait, 2),
            measurement=vcat(fill("m1", length(d.series)), fill("m2", length(d.series))))
        d = merge(d, (value=randn(Random.Xoshiro(3), length(d.series)),
            obs=string.(d.series, " ", Int.(d.occasion), " ", d.measurement),
            cell=string.(d.series, " ", Int.(d.occasion))))
        TR = temporal_indep(:(0 + trait | series), :occasion; replicate=:measurement)
        frow = (@test_logs (:info, r"Auto-suppressing `sigma_eps`") fit_temporal_gllvm(d;
            formula=@formula(value ~ 0 + trait), temporal=TR, unit=:series, unit_obs=:obs,
            structure=[:(indep(0 + trait | obs))]))
        @test !("log_sigma_eps" in frow.parameter_names)
        @test frow.sigma_eps == max(1e-3 * std(d.value), 1e-6)
        fcell = fit_temporal_gllvm(d; formula=@formula(value ~ 0 + trait), temporal=TR,
            unit=:series, unit_obs=:cell, structure=[:(indep(0 + trait | cell))])
        @test "log_sigma_eps" in fcell.parameter_names
        @test dof(frow) == dof(fcell) - 1
        @test aic(frow) == -2 * loglikelihood(frow) + 2 * dof(frow)
    end

    @testset "grouping and term admission" begin
        d = merge(dat, (other=copy(dat.series),))
        @test_throws ArgumentError fitc(d; unit=:series, structure=[:(indep(0 + trait | other))])
        @test_throws ArgumentError fitc(dat; unit=:series, structure=[:(unique(0 + trait | series))])
        @test_throws ArgumentError fitc(dat; unit=:series,
            structure=[:(indep(0 + trait | series)), :(dep(0 + trait | series))])
        # Ordinary-only models without a temporal term are unchanged.
        @test fitc(dat) isa TemporalGaussianFit
        @test isempty(GMS._temporal_other_tiers(fitc(dat)))
    end

    @testset "update replays the call with named overrides" begin
        f = fitc(dat; unit=:series, structure=[:(1 | series)])
        g = update(f)
        @test g isa TemporalGaussianFit && g.parameter_names == f.parameter_names
        @test abs(g.loglik - f.loglik) <= 1e-10
        d2 = merge(dat, (value=dat.value .+ 0.01 .* (1:length(dat.value)),))
        h = update(f; data=d2)
        @test h.y == d2.value
        k = update(f; temporal=temporal_latent(:(0 + trait | series), :occasion; unique=true))
        @test extract_temporal(k).parameters.mode == "latent"
        @test "log_sigma_re_int" in k.parameter_names
    end
end
