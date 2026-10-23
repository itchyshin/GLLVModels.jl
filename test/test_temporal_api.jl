# gllvm-parity-tag: P1
#
# Temporal constructors, pre-pass and admission (gllvmTMB P1 9539352f6; spec
# section 4.1 rows sixth-source-api.R:1, :16, :26; ar1-parser.R:4, :25;
# ar1-methods.R:3, :50, :59; sixth-source-engine.R:12, :169, :196).
# Julia-only: every expectation is a structural or message contract.
using Test, GLLVModels
const GMA = GLLVModels

temporal_grid(series, occasions; traits=["t1", "t2", "t3"]) = begin
    rows = [(s, o, t) for t in traits for o in occasions for s in series]  # expand.grid order
    (series=[r[1] for r in rows], occasion=Float64[r[2] for r in rows],
     trait=[r[3] for r in rows])
end
with_value(d, v) = merge(d, (value=Float64.(v),))
fitq(d, term; kw...) = fit_temporal_gllvm(d; formula=@formula(value ~ 0 + trait), temporal=term, kw...)
# Message match with the R regex fragment, and the R class carried as `kind`.
function refuses(f, pattern; kind=:rlang_error)
    err = try
        f(); nothing
    catch e
        e
    end
    return err isa TemporalContractError && occursin(pattern, err.message) && err.kind === kind
end

@testset "temporal constructors, pre-pass and admission" begin
    @testset "constructors preserve the requested covariance mode (api.R:1, :16)" begin
        indep = temporal_indep(:(0 + trait | series), :occasion)
        dep = temporal_dep(:(0 + trait | series), :occasion)
        latent = temporal_latent(:(0 + trait | series), :occasion; d=1, unique=true)
        @test indep isa TemporalTerm && dep isa TemporalTerm && latent isa TemporalTerm
        @test (indep.mode, dep.mode, latent.mode) == (:indep, :dep, :latent)
        @test latent.unique
        @test indep.unique && !dep.unique            # R: unique is TRUE for indep, FALSE for dep
        @test temporal_indep(:(0 + trait | series), :occasion; structure=:ar1).structure === :ar1
        @test temporal_indep(:(0 + trait | series), :elapsed; structure=:ou).structure === :ou
        @test temporal_indep(:(0 + trait | series), :elapsed; structure="ou").structure === :ou
    end

    @testset "constructor boundary checks (ar1-parser.R:25)" begin
        term = temporal_latent(:(0 + trait | series), :occasion; d=1)
        @test term.d == 1 && term.structure === :ar1
        @test refuses(() -> temporal_latent(:(0 + trait | series), :occasion; d=2), "rank one")
        @test refuses(() -> temporal_indep(:(0 + trait | series), :(occasion + 1)), "bare column")
        @test refuses(() -> temporal_indep(:(0 + trait), :occasion), "formula of the form")
        @test refuses(() -> temporal_indep(:(0 + trait | series), :occasion; replicate=:(a + b)),
            "must be NULL or a bare column")
        @test refuses(() -> temporal_indep(:(0 + trait | series), :occasion; structure=:rw), "either \"ar1\" or \"ou\"")
        @test refuses(() -> temporal_latent(:(0 + trait | series), :occasion; unique=1), "TRUE or FALSE")
        # Every temporal refusal carries R's action line.
        err = try; temporal_latent(:(0 + trait | series), :occasion; d=2); catch e; e; end
        @test endswith(err.message, "See `temporal_latent()` for the admitted temporal workflow.")
    end

    base = with_value(temporal_grid(["a", "b"], [1, 3, 7]), 1:18)

    @testset "parser keeps ordered integer gaps (api.R:26, ar1-parser.R:4)" begin
        spec = GMA._parse_temporal_term(temporal_indep(:(0 + trait | series), :occasion),
            Base.structdiff(base, NamedTuple()); trait=:trait, response=:value)
        @test spec.pair_table.time[spec.pair_table.series .== "a"] == [1.0, 3.0, 7.0]
        @test spec.structure === :ar1
        spec_l = GMA._parse_temporal_term(temporal_latent(:(0 + trait | series), :occasion; unique=true),
            base; trait=:trait, response=:value)
        @test spec_l.pair_table.time[spec_l.pair_table.series .== "a"] == [1.0, 3.0, 7.0]
        el = merge(base, (elapsed=[Dict(1.0 => 0.0, 3.0 => 1.5, 7.0 => 5.0)[o] for o in base.occasion],))
        spec_ou = GMA._parse_temporal_term(temporal_dep(:(0 + trait | series), :elapsed; structure=:ou),
            el; trait=:trait, response=:value)
        @test spec_ou.structure === :ou
        @test spec_ou.elapsed[2] == 1.5 && spec_ou.gap == zeros(Int, 6)
        @test refuses(() -> GMA._parse_temporal_term(temporal_indep(:(0 + trait | series), :elapsed),
            el; trait=:trait, response=:value), "integer-valued")
    end

    @testset "pre-pass refusals keep R's messages (ar1-parser.R:25, methods.R:59)" begin
        t = temporal_indep(:(0 + trait | series), :occasion)
        @test refuses(() -> fitq(base, temporal_latent(:(1 | series), :occasion)), "trait-intercept block")
        @test refuses(() -> fitq((; (k => v[2:end] for (k, v) in pairs(base))...), t), "complete trait panel")
        dup = (; (k => vcat(v, v[1:1]) for (k, v) in pairs(base))...)
        @test refuses(() -> fitq(dup, t), "replicate")
        miss = merge(base, (value=Union{Missing,Float64}[i == 1 ? missing : Float64(i) for i in 1:18],))
        @test refuses(() -> fitq(miss, temporal_latent(:(0 + trait | series), :occasion)), "complete Gaussian response")
        @test refuses(() -> fitq(base, temporal_indep(:(0 + trait | unit), :occasion)), "missing column")
        two = with_value(temporal_grid(["a", "b"], [1, 3, 7]; traits=["t1", "t2"]), 1:12)
        @test refuses(() -> fitq(two, t), "at least three traits")
        short = with_value(temporal_grid(["a", "b"], [1, 3]), 1:12)
        @test refuses(() -> fitq(short, t), "three strictly ordered occasions")
        noninteger = merge(base, (occasion=base.occasion .+ 0.5,))
        @test refuses(() -> fitq(noninteger, t), "integer-valued")
        @test refuses(() -> fitq(base, temporal_indep(:(0 + trait | series), :occasion; replicate=:measurement)),
            "must name a column")
    end

    @testset "replicated workflow admission and refusals" begin
        r = temporal_grid(["a", "b"], [1, 3, 7])
        rep = (series=vcat(r.series, r.series), occasion=vcat(r.occasion, r.occasion),
            trait=vcat(r.trait, r.trait), measurement=vcat(fill("m1", 18), fill("m2", 18)),
            value=collect(1.0:36.0) .+ sin.(1:36))
        term = temporal_indep(:(0 + trait | series), :occasion; replicate=:measurement)
        f = fitq(rep, term)
        @test f.spec.workflow === :replicated
        @test extract_temporal(f).parameters.workflow == "replicated"
        @test refuses(() -> fitq((; (k => vcat(v, v[1:1]) for (k, v) in pairs(rep))...), term), "duplicate series--occasion--replicate--trait")
        @test refuses(() -> fitq((; (k => v[2:end] for (k, v) in pairs(rep))...), term), "one complete trait panel")
        one = merge(rep, (measurement=vcat(fill("m1", 18), fill("m2", 18)),))
        keep = [i <= 18 || rep.series[i] == "a" for i in 1:36]
        @test refuses(() -> fitq((; (k => v[keep] for (k, v) in pairs(one))...), term), "at least two measurements")
    end

    @testset "deferred providers and duplicate temporal terms (engine.R:169)" begin
        fx = with_value(temporal_grid(["s1", "s2", "s3"], [1, 3, 7]), 1:27)
        t = temporal_indep(:(0 + trait | series), :occasion)
        for other in (:(phylo_indep(0 + trait | series)), :(animal_indep(0 + trait | series)),
                      :(spatial_indep(0 + trait | series)), :(kernel_indep(series)))
            @test refuses(() -> fitq(fx, t; structure=[other]), "requires replicated AR1")
        end
        @test refuses(() -> fitq(fx, t; structure=[:(temporal_dep(0 + trait | series; time = occasion))]),
            "Only one temporal covariance term")
        @test refuses(() -> fitq(fx, temporal_dep(:(0 + trait | series), :occasion);
            structure=[:(phylo_indep(0 + trait | series))]), "cannot be combined with another covariance source")
        # Ordinary unit/unit_obs composition (slice 2) is admitted and is a composed
        # fit, never silently ignored (test_temporal_composed.jl twins it).
        @test GMA._temporal_other_tiers(fitq(fx, t; structure=[:(indep(0 + trait | series))])) == ["diag_B"]
    end

    @testset "dedicated state tier (engine.R:12) and extractor labels (methods.R:3)" begin
        fx = with_value(temporal_grid(["s1", "s2", "s3"], [1, 3, 7]), 1:27)
        f = fitq(fx, temporal_indep(:(0 + trait | series), :occasion))
        s = f.spec
        @test length(s.pair_table.pair_id) == 9
        @test length(s.state_id) == length(f.y)
        @test all(in(-1:8), s.predecessor .- 1)     # R's 0-based convention, -1 at a series start
        @test any(==(2), s.gap) && any(==(4), s.gap)
        @test f.loadings === nothing && f.psi !== nothing
        e = extract_temporal(f)
        @test all(==("temporal_indep_variance"), e.variance.component)
        @test length(e.variance.value) == 3
        fl = fitq(with_value(fx, (1:27) ./ 10), temporal_latent(:(0 + trait | series), :occasion; unique=true))
        el = extract_temporal(fl)
        @test el.parameters.mode == "latent" && el.parameters.structure == "ar1"
        @test all(==("temporal_Psi_variance"), el.variance.component)
        @test length(el.pair_index.pair_id) == 9
        @test size(el.loadings) == (3, 1)
        @test refuses(() -> extract_temporal(fit_gaussian_gllvm(randn(3, 10); K=1)), "requires a fit made with a temporal covariance term")
    end

    @testset "all modes and structures reach the temporal tier (engine.R:196)" begin
        fx = with_value(temporal_grid(["s1", "s2", "s3"], [1, 3, 7]),
            [parse(Int, t[2:end]) + o / 10 for (t, o) in zip(temporal_grid(["s1", "s2", "s3"], [1, 3, 7]).trait,
                temporal_grid(["s1", "s2", "s3"], [1, 3, 7]).occasion)])
        fx = merge(fx, (elapsed=[Dict(1.0 => 0.0, 3.0 => 0.5, 7.0 => 2.0)[o] for o in fx.occasion],))
        for (mk, st, tc, u) in ((temporal_indep, :ar1, :occasion, false), (temporal_dep, :ar1, :occasion, false),
                (temporal_latent, :ar1, :occasion, false), (temporal_latent, :ar1, :occasion, true),
                (temporal_indep, :ou, :elapsed, false), (temporal_dep, :ou, :elapsed, false),
                (temporal_latent, :ou, :elapsed, false), (temporal_latent, :ou, :elapsed, true))
            term = mk === temporal_latent ? mk(:(0 + trait | series), tc; structure=st, unique=u) :
                mk(:(0 + trait | series), tc; structure=st)
            f = fitq(fx, term)
            @test f isa TemporalGaussianFit
            @test f.spec.mode === term.mode && f.spec.structure === st
            @test isfinite(f.loglik)
        end
    end

    @testset "stable sign anchor (methods.R:50)" begin
        s = GMA._temporal_report_sign(reshape([0.0, -2.0, 0.5], 3, 1); traits=["first", "largest", "third"])
        @test s.first_loading_negligible
        @test s.anchor_trait == "largest"
        @test s.multiplier == -1
    end
end
