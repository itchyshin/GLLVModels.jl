using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Poisson

# `confint(fit, Y; method = m, ...)` either runs method `m` or throws an
# ArgumentError that names the supported methods. It never returns a Wald
# interval for a request for a profile or bootstrap interval, and never drops a
# keyword it does not understand. Derived quantities (communality, icc, rho,
# proportion, phylo_signal) are reached through the same call by `parm`.
#
# Route classifier: the same field-shape rule as tools/core070_inference_batch.jl.
function _route_tag(r)
    f = propertynames(r)
    :n_converged in f && return :bootstrap
    :se_transformed in f && return :wald_derived
    (:se in f && :pd_hessian in f) && return :wald_packed
    :method in f && return :profile
    return :unknown
end

# Message check shared by the refusal tests: a named ArgumentError that lists the
# methods the caller can use.
function _refuses_listing_methods(thunk)
    err = try
        thunk()
        nothing
    catch e
        e
    end
    err isa ArgumentError || return false
    msg = sprint(showerror, err)
    return occursin(":wald", msg) && occursin(":profile", msg) && occursin(":bootstrap", msg)
end

# The message of the ArgumentError a call throws, or `nothing` when it does not throw one.
function _argument_error_message(thunk)
    try
        thunk()
    catch e
        return e isa ArgumentError ? sprint(showerror, e) : nothing
    end
    return nothing
end

# ---- fixtures ---------------------------------------------------------------
# Plain default Gaussian fit: integration === nothing.
function _fixture_plain()
    rng = MersenneTwister(11)
    p, n, K = 4, 60, 1
    Λ = reshape([0.7, 0.5, 0.4, -0.3], p, K)
    Y = Λ * randn(rng, K, n) + 0.5 * randn(rng, p, n)
    return fit_gaussian_gllvm(Y; K = K), Y
end

# Structured Gaussian fit (B/W diagonal tiers + phylogenetic block): never an
# AGHQ record, so it takes the packed (non-record) route.
function _fixture_structured()
    rng = MersenneTwister(20260902)
    p, n = 3, 30
    Σ_phy = [1.0 0.2 0.1; 0.2 1.0 0.15; 0.1 0.15 1.0]
    Λ = reshape([0.5, 0.3, -0.4], p, 1)
    β = [0.4, -0.2, 0.1]
    Y = zeros(p, n)
    for s in 1:n
        z = randn(rng, 1)
        for t in 1:p
            Y[t, s] = β[t] + (Λ * z)[t] + 0.2 * randn(rng) + 0.15 * randn(rng)
        end
    end
    fit = fit_gaussian_gllvm(Y; K = 1, has_diag = true, has_phy_unique = true, Σ_phy = Σ_phy)
    return fit, Y, Σ_phy
end

# AGHQ Gaussian record fit with one covariate.
function _fixture_record()
    rng = MersenneTwister(20260901)
    p, n = 3, 30
    X = randn(rng, p, n, 1)
    Λ = reshape([0.6, 0.4, -0.5], p, 1)
    β = [0.5, -0.3, 0.2]
    Y = zeros(p, n)
    for s in 1:n
        z = randn(rng, 1)
        for t in 1:p
            Y[t, s] = β[t] + 0.3 * X[t, s, 1] + (Λ * z)[t] + 0.3 * randn(rng)
        end
    end
    return fit_gaussian_gllvm(Y; K = 1, X = X, aghq = 3), Y, X
end

@testset "confint honours method= or refuses it" begin
    fitS, YS, Σ_phy = _fixture_structured()
    fitP, YP = _fixture_plain()
    fitA, YA, XA = _fixture_record()

    @testset "structured Gaussian (packed route)" begin
        parm = "sigma_B[1]"
        # the evidence case: before this fix both of these returned a Wald interval
        pr = confint(fitS, YS; parm = parm, method = :profile, Σ_phy = Σ_phy, profile_iterations = 20)
        @test _route_tag(pr) == :profile
        @test pr.method === :profile && pr.term == [parm]
        direct = profile_ci(fitS, parm; y = YS, Σ_phy = Σ_phy, profile_iterations = 20)
        @test isequal(pr.lower[1], direct.lower) && isequal(pr.upper[1], direct.upper)
        @test pr.status == [direct.method]

        bt = confint(fitS, YS; parm = parm, method = :bootstrap, Σ_phy = Σ_phy, n_boot = 10, seed = 4)
        @test _route_tag(bt) == :bootstrap
        @test bt.method === :bootstrap && bt.term == [parm]
        # the same refits as bootstrap_ci; an SD is reported on the raw scale like the other routes
        db = bootstrap_ci(fitS; parms = parm, y = YS, Σ_phy = Σ_phy, n_boot = 10, seed = 4)
        @test bt.n_converged == db.n_converged
        i_B1 = findfirst(==(parm), first(GLLVModels._confint_all_term_names(fitS)))
        @test bt.estimate == [exp(fitS.pars.θ_packed[i_B1])]
        @test all(l -> isnan(l) || l >= 0, bt.lower)
        # a linear term is bootstrap_ci's interval, value for value
        lt = confint(fitS, YS; parm = "Lambda_B[1,1]", method = :bootstrap, Σ_phy = Σ_phy, n_boot = 10, seed = 4)
        dl = bootstrap_ci(fitS; parms = "Lambda_B[1,1]", y = YS, Σ_phy = Σ_phy, n_boot = 10, seed = 4)
        @test isequal(lt.lower, dl.lower) && isequal(lt.upper, dl.upper) && lt.estimate == dl.estimate

        # several terms, a group selector, and the keyword-only form
        several = confint(fitS, YS; parm = ["sigma_eps", "sigma_W"], method = :profile,
                          Σ_phy = Σ_phy, profile_iterations = 20)
        @test several.term == ["sigma_eps", "sigma_W[1]", "sigma_W[2]", "sigma_W[3]"]
        @test length(several.status) == 4
        kwform = confint(fitS; y = YS, parm = "sigma_eps", method = :profile, Σ_phy = Σ_phy,
                         profile_iterations = 20)
        @test _route_tag(kwform) == :profile

        # Wald unchanged, and spelled out or defaulted gives the same object
        w0 = confint(fitS, YS; parm = parm, Σ_phy = Σ_phy)
        w1 = confint(fitS, YS; parm = parm, method = :wald, Σ_phy = Σ_phy)
        w2 = confint(fitS; y = YS, parm = parm, Σ_phy = Σ_phy)
        @test _route_tag(w0) == :wald_packed
        @test isequal(w0, w1) && isequal(w0, w2)

        @test _refuses_listing_methods(() -> confint(fitS, YS; parm = parm, method = :bogus, Σ_phy = Σ_phy))
        @test _refuses_listing_methods(() -> confint(fitS; y = YS, parm = parm, method = :bogus, Σ_phy = Σ_phy))
        # the message names the fit and the parameter
        msg = try
            confint(fitS, YS; parm = parm, method = :fisher_z, Σ_phy = Σ_phy)
        catch e
            sprint(showerror, e)
        end
        @test occursin("method = :fisher_z", msg) && occursin("sigma_B[1]", msg) && occursin("available:", msg)
        # the method is a Symbol; a string gets a named error, not a MethodError
        @test_throws ArgumentError confint(fitS, YS; parm = parm, method = "profile", Σ_phy = Σ_phy)
        @test_throws ArgumentError confint(fitS; y = YS, parm = parm, method = "profile", Σ_phy = Σ_phy)
        # a profile or bootstrap request needs the data
        @test_throws ArgumentError confint(fitS; parm = parm, method = :profile, Σ_phy = Σ_phy)
        @test_throws ArgumentError confint(fitS; parm = parm, method = :bootstrap, Σ_phy = Σ_phy)
    end

    @testset "plain default Gaussian fit (packed route)" begin
        pr = confint(fitP, YP; parm = "sigma_eps", method = :profile, profile_iterations = 20)
        @test _route_tag(pr) == :profile
        d = profile_ci(fitP, "sigma_eps"; y = YP, profile_iterations = 20)
        @test isequal(pr.lower[1], d.lower) && isequal(pr.upper[1], d.upper)
        @test pr.estimate[1] ≈ fitP.pars.σ_eps

        bt = confint(fitP, YP; parm = "Lambda:1,1", method = :bootstrap, n_boot = 10, seed = 2)
        @test _route_tag(bt) == :bootstrap
        db = bootstrap_ci(fitP; parms = "Lambda_B[1,1]", y = YP, n_boot = 10, seed = 2)
        @test isequal(bt.lower, db.lower) && isequal(bt.upper, db.upper)
        # an SD is reported on the raw scale: positive, like the Wald and profile routes
        bs = confint(fitP, YP; parm = "sigma_eps", method = :bootstrap, n_boot = 12, seed = 2)
        @test bs.estimate == [fitP.pars.σ_eps] && all(>(0), bs.lower) && all(bs.lower .< bs.estimate .< bs.upper)

        # n_boot and seed are honoured (they were silently dropped)
        a = confint(fitP, YP; parm = "sigma_eps", method = :bootstrap, n_boot = 10, seed = 1)
        b = confint(fitP, YP; parm = "sigma_eps", method = :bootstrap, n_boot = 10, seed = 1)
        c = confint(fitP, YP; parm = "sigma_eps", method = :bootstrap, n_boot = 12, seed = 1)
        @test isequal(a.lower, b.lower) && size(a.replicates, 1) == 10 && size(c.replicates, 1) == 12

        @test _refuses_listing_methods(() -> confint(fitP, YP; parm = "sigma_eps", method = :bogus))
        # a keyword the route does not use is refused, not swallowed
        @test_throws ArgumentError confint(fitP, YP; foo = 1)
        @test_throws ArgumentError confint(fitP, YP; method = :bootstrap, n_boot = 10, parallel = true)
        @test_throws ArgumentError confint(fitP, YP; mask = trues(size(YP)))
        @test_throws ArgumentError confint(fitP, YP; objective = :va)
        # inert values stay accepted so existing generic callers keep working
        @test isequal(confint(fitP, YP; mask = nothing, offset = nothing, N = nothing, objective = :fit),
                      confint(fitP, YP))
        # level is validated on every method
        @test_throws ArgumentError confint(fitP, YP; method = :profile, level = 1.5)
        @test_throws ArgumentError confint(fitP, YP; method = :bootstrap, level = 0.0)
    end

    @testset "stderror stays Wald-only" begin
        @test_throws ArgumentError GLLVModels.stderror(fitP, YP; method = :profile)
        @test_throws ArgumentError GLLVModels.stderror(fitP; y = YP, method = :bootstrap)
        @test GLLVModels.stderror(fitP, YP; method = :wald) == GLLVModels.stderror(fitP, YP)
    end

    @testset "AGHQ Gaussian record" begin
        pr = confint(fitA, YA; X = XA, method = :profile, parm = "sigma_eps", profile_iterations = 20)
        @test _route_tag(pr) == :profile
        bt = confint(fitA, YA; X = XA, method = :bootstrap, n_boot = 3, seed = 1, parm = "sigma_eps")
        @test _route_tag(bt) == :bootstrap
        @test _refuses_listing_methods(() -> confint(fitA, YA; X = XA, method = :bogus))
        @test _refuses_listing_methods(() -> confint(fitA; y = YA, X = XA, method = :bogus))
        @test_throws ArgumentError confint(fitA, YA; X = XA, foo = 1)
        # the keyword-only form carries `method` too
        @test _route_tag(confint(fitA; y = YA, X = XA, method = :profile, parm = "sigma_eps",
                                 profile_iterations = 20)) == :profile
    end

    @testset "Poisson (family route)" begin
        rng = MersenneTwister(7)
        μ = exp.(0.3 .+ 0.4 * randn(rng, 4, 40))
        Yc = [rand(rng, Poisson(μ[i, j])) for i in 1:4, j in 1:40]
        fitC = fit_poisson_gllvm(Yc; K = 1)
        @test _route_tag(confint(fitC, Yc; method = :wald)) == :wald_packed
        @test _route_tag(confint(fitC, Yc; method = :profile, parm = "beta[1]", profile_iterations = 20)) == :profile
        @test _route_tag(confint(fitC, Yc; method = :bootstrap, parm = "beta[1]", n_boot = 3, seed = 3)) == :bootstrap
        @test _refuses_listing_methods(() -> confint(fitC, Yc; method = :bogus))
    end
end

@testset "confint routes derived quantities by parm" begin
    fitS, YS, Σ_phy = _fixture_structured()
    fitP, YP = _fixture_plain()
    fitA, YA, XA = _fixture_record()
    spec = GLLVModels._derived_spec(fitS)
    kw = (Σ_phy = Σ_phy,)

    @testset "Wald equals the direct functions, bit for bit" begin
        for t in 1:3
            r = confint(fitS, YS; parm = "communality[$t]", method = :wald, kw...)
            d = communality_wald_ci(fitS, t; y = YS, Σ_phy = Σ_phy)
            @test _route_tag(r) == :wald_derived
            @test r.term == ["communality[$t]"] && r.method === :wald
            @test isequal((r.estimate[1], r.lower[1], r.upper[1], r.se_transformed[1]),
                          (d.estimate, d.lower, d.upper, d.se_transformed))
            @test r.transform == [:logit] && r.pd_hessian == [d.pd_hessian] && r.status == [d.method]

            h = confint(fitS, YS; parm = "phylo_signal[$t]", method = :wald, kw...)
            dh = phylo_signal_wald_ci(fitS, t; y = YS, Σ_phy = Σ_phy)
            @test isequal((h.estimate[1], h.lower[1], h.upper[1]), (dh.estimate, dh.lower, dh.upper))
        end
        rho = confint(fitS, YS; parm = "rho[1,2]", method = :wald, kw...)
        dr = correlation_wald_ci(fitS, 1, 2; y = YS, Σ_phy = Σ_phy)
        @test isequal((rho.estimate[1], rho.lower[1], rho.upper[1]), (dr.estimate, dr.lower, dr.upper))
        @test rho.term == ["rho[1,2]"] && rho.transform == [:fisher_z]

        # icc is extract_ICC_site; its interval is icc_wald_ci on the packed icc closure
        icc = confint(fitS, YS; parm = "icc[2]", method = :wald, kw...)
        di = icc_wald_ci(fitS, GLLVModels._make_icc_closure(spec, 2); y = YS, Σ_phy = Σ_phy)
        @test isequal((icc.estimate[1], icc.lower[1], icc.upper[1]), (di.estimate, di.lower, di.upper))
        @test icc.estimate[1] ≈ extract_ICC_site(fitS)[2] atol = 1e-12

        # proportion components
        pshared = confint(fitS, YS; parm = "proportion:shared[1]", method = :wald, kw...)
        c1 = communality_wald_ci(fitS, 1; y = YS, Σ_phy = Σ_phy)
        @test pshared.term == ["proportion:shared[1]"]
        @test isequal((pshared.lower[1], pshared.upper[1]), (c1.lower, c1.upper))
        for comp in (:unique_B, :unique_Wd, :residual)
            pc = confint(fitS, YS; parm = "proportion:$comp[3]", method = :wald, kw...)
            @test pc.estimate[1] ≈ GLLVModels.proportions(fitS; component = comp)[3] atol = 1e-12
        end

        # all traits, one Hessian: same numbers as one call per trait
        all_c = confint(fitS, YS; parm = "communality", method = :wald, kw...)
        @test all_c.term == ["communality[1]", "communality[2]", "communality[3]"]
        one_by_one = [communality_wald_ci(fitS, t; y = YS, Σ_phy = Σ_phy) for t in 1:3]
        @test isequal(all_c.lower, [d.lower for d in one_by_one]) && isequal(all_c.upper, [d.upper for d in one_by_one])
        # gllvmTMB's tier spellings name its aligned estimands (extract_communality,
        # extract_correlations, extract_proportions), which differ from these quantities,
        # so they are refused rather than silently mapped (review of #709).
        for bad in ("communality:unit:2", "correlation:unit:1,2", "rho:unit:1,2",
                    "proportion:shared_unit[1]", "proportion:unique_unit[1]")
            @test_throws ArgumentError confint(fitS, YS; parm = bad, method = :wald, kw...)
        end
        @test isequal(confint(fitS, YS; parm = "repeatability[2]", method = :wald, kw...).lower, icc.lower)
        @test isequal(confint(fitS, YS; parm = "correlation[1,2]", method = :wald, kw...).lower, rho.lower)
        # a vector of derived names concatenates in order
        two = confint(fitS, YS; parm = ["communality[1]", "rho[1,2]"], method = :wald, kw...)
        @test two.term == ["communality[1]", "rho[1,2]"] && two.transform == [:logit, :fisher_z]
        # the default method for a derived parm is unchanged: Wald
        @test isequal(confint(fitS, YS; parm = "communality[1]", kw...), confint(fitS, YS; parm = "communality[1]", method = :wald, kw...))
    end

    @testset "record and plain fits take the same routes" begin
        rA = confint(fitA, YA; X = XA, parm = "communality[1]", method = :wald)
        dA = communality_wald_ci(fitA, 1; y = YA, X = XA)
        @test isequal((rA.lower[1], rA.upper[1]), (dA.lower, dA.upper))
        rP = confint(fitP, YP; parm = "rho[1,2]", method = :wald)
        dP = correlation_wald_ci(fitP, 1, 2; y = YP)
        @test isequal((rP.lower[1], rP.upper[1]), (dP.lower, dP.upper))
        # the keyword-only form reaches derived parms too
        @test isequal(confint(fitP; y = YP, parm = "rho[1,2]").lower, rP.lower)
    end

    @testset "profile routes to the existing derived profile functions" begin
        f_i = GLLVModels._make_icc_closure(spec, 1)
        r = confint(fitS, YS; parm = "icc[1]", method = :profile, penalty_weight = 1e4, kw...)
        @test _route_tag(r) == :profile && r.method === :profile && r.term == ["icc[1]"]
        d = GLLVModels.profile_ci_derived(fitS, f_i; y = YS, Σ_phy = Σ_phy, penalty_weight = 1e4)
        @test r.estimate[1] == d.estimate
        # an interior profile interval is the direct function's, unchanged
        if all(isfinite, (d.lower, d.upper)) && 0 < d.lower && d.upper < 1
            @test isequal((r.lower[1], r.upper[1]), (d.lower, d.upper))
        end
        @test r.lower[1] <= r.estimate[1] <= r.upper[1] || isnan(r.lower[1]) || isnan(r.upper[1])

        ps = confint(fitS, YS; parm = "phylo_signal[1]", method = :profile, penalty_weight = 1e4, kw...)
        dps = profile_ci_phylo_signal(fitS, 1; y = YS, Σ_phy = Σ_phy, penalty_weight = 1e4)
        @test isequal((ps.lower[1], ps.upper[1], ps.estimate[1], ps.boundary[1]),
                      (dps.lower, dps.upper, dps.estimate, dps.boundary))
        @test ps.status == [dps.method]
    end

    @testset "profile is withdrawn for communality, rho and proportion, as in gllvmTMB" begin
        # gllvmTMB 9539352f6 raises gllvmTMB_nonlinear_profile_withdrawn for these three;
        # the refusal names the withdrawal, so it is not the bad-method message.
        for (parm, what) in (("communality[1]", "communality"), ("rho[1,2]", "correlations"),
                             ("proportion:shared[1]", "variance proportions"),
                             ("correlation[1,2]", "correlations"), ("communality", "communality"))
            msg = _argument_error_message(() -> confint(fitS, YS; parm = parm, method = :profile, kw...))
            @test msg !== nothing
            @test occursin("nonlinear profile intervals for $what are withdrawn", msg)
            @test occursin("parm $parm", msg) && occursin(":bootstrap", msg)
            @test !occursin("is not available for", msg)  # not the bad-method refusal
        end
        # withdrawn on a record fit and on a plain fit too, before any refit
        @test occursin("withdrawn", _argument_error_message(
            () -> confint(fitA, YA; X = XA, parm = "communality[1]", method = :profile)))
        @test occursin("withdrawn", _argument_error_message(
            () -> confint(fitP, YP; parm = "rho[1,2]", method = :profile)))
        # a vector that includes a withdrawn kind is refused as a whole
        @test occursin("withdrawn", _argument_error_message(
            () -> confint(fitS, YS; parm = ["phylo_signal[1]", "rho[1,2]"], method = :profile, kw...)))
        # the other methods of the same quantities still run
        @test _route_tag(confint(fitS, YS; parm = "proportion:shared[1]", method = :wald, kw...)) == :wald_derived
    end

    @testset "rho: method = :fisher_z is the :wald Fisher-z interval" begin
        w = confint(fitS, YS; parm = "rho[1,2]", method = :wald, kw...)
        z = confint(fitS, YS; parm = "rho[1,2]", method = :fisher_z, kw...)
        @test isequal(z, w)                      # every field, bit for bit
        @test z.transform == [:fisher_z] && z.method === :wald
        @test isequal(z, confint(fitS, YS; parm = "rho[1,2]", kw...))  # = the default
        @test isequal(confint(fitP, YP; parm = "rho", method = :fisher_z),
                      confint(fitP, YP; parm = "rho", method = :wald))
        # accepted for rho only: refused for the other derived quantities and packed terms
        for parm in ("communality[1]", "icc[1]", "proportion:shared[1]", "phylo_signal[1]", "sigma_eps")
            msg = _argument_error_message(() -> confint(fitS, YS; parm = parm, method = :fisher_z, kw...))
            @test msg !== nothing && occursin("method = :fisher_z is not available", msg)
        end
        @test _argument_error_message(
            () -> confint(fitS, YS; parm = ["communality[1]", "rho[1,2]"], method = :fisher_z, kw...)) !== nothing
    end

    @testset "bootstrap routes to bootstrap_ci_derived" begin
        r = confint(fitS, YS; parm = "communality[1]", method = :bootstrap, n_boot = 10, seed = 9, kw...)
        d = GLLVModels.bootstrap_ci_derived(fitS, f -> communality(f)[1]; y = YS, Σ_phy = Σ_phy,
                                            n_boot = 10, seed = 9)
        @test _route_tag(r) == :bootstrap && r.method === :bootstrap
        @test isequal((r.estimate[1], r.lower[1], r.upper[1]), (d.estimate, d.lower, d.upper))
        @test r.n_converged == d.n_converged && r.n_valid == [d.n_valid]
        @test isequal(r.replicates[:, 1], d.replicates)

        rho = confint(fitS, YS; parm = "rho[1,2]", method = :bootstrap, n_boot = 10, seed = 10, kw...)
        drho = GLLVModels.bootstrap_ci_derived(fitS, f -> correlation(f)[1, 2]; y = YS, Σ_phy = Σ_phy,
                                               n_boot = 10, seed = 10)
        @test isequal((rho.lower[1], rho.upper[1]), (drho.lower, drho.upper))

        ic = confint(fitS, YS; parm = "icc[1]", method = :bootstrap, n_boot = 10, seed = 12, kw...)
        dic = GLLVModels.bootstrap_ci_derived(fitS, f -> extract_ICC_site(f)[1]; y = YS, Σ_phy = Σ_phy,
                                              n_boot = 10, seed = 12)
        @test isequal((ic.lower[1], ic.upper[1]), (dic.lower, dic.upper))
    end

    @testset "validation: named errors that list what is available" begin
        @test occursin("available: :wald, :bootstrap", _argument_error_message(
            () -> confint(fitS, YS; parm = "communality[1]", method = :bogus, kw...)))
        @test occursin("available: :wald, :fisher_z, :bootstrap", _argument_error_message(
            () -> confint(fitS, YS; parm = "rho[1,2]", method = :bogus, kw...)))
        @test _refuses_listing_methods(() -> confint(fitS, YS; parm = "icc[1]", method = :bogus, kw...))
        @test _refuses_listing_methods(() -> confint(fitS, YS; parm = "phylo_signal", method = :fisher_z, kw...))
        # parse and range errors name the offending token
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[9]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[0]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "rho[1,1]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "rho[1]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "communality:unit_obs", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[a]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "proportion:nonsense", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "icc:unit", kw...)
        # quantities the fit does not define
        @test_throws ArgumentError confint(fitP, YP; parm = "phylo_signal")
        @test_throws ArgumentError confint(fitP, YP; parm = "proportion:unique_W")
        @test_throws ArgumentError confint(fitP, YP; parm = "proportion:unique_B")
        # a derived name and a packed term name do not mix
        @test_throws ArgumentError confint(fitS, YS; parm = ["communality[1]", "sigma_eps"], kw...)
        # keywords of another route are refused, not swallowed
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[1]", foo = 1, kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[1]", profile_iterations = 5, kw...)
        # a derived call needs the data
        @test_throws ArgumentError confint(fitS; parm = "communality[1]", kw...)
        @test_throws ArgumentError confint(fitS, YS; parm = "communality[1]", level = 2.0, kw...)
    end
end

@testset "confint refuses a profile or bootstrap request on a pinned-loadings fit" begin
    rng = MersenneTwister(5)
    p, n = 4, 80
    Λ = reshape([0.8, 0.5, 0.4, -0.3], p, 1)
    Y = Λ * randn(rng, 1, n) + 0.5 * randn(rng, p, n)
    M = fill(NaN, p, 1)
    M[2, 1] = 0.5
    pinned = fit_gaussian_gllvm(Y; K = 1, lambda_constraint = M)
    # bootstrap_ci and profile_ci refit without the pins: a different model
    @test_throws ArgumentError confint(pinned, Y; parm = "sigma_eps", method = :profile)
    @test_throws ArgumentError confint(pinned, Y; parm = "sigma_eps", method = :bootstrap, n_boot = 3)
    @test_throws ArgumentError confint(pinned, Y; parm = "communality[1]", method = :bootstrap, n_boot = 3)
end

@testset "direct profile and bootstrap entry points refuse a pinned-loadings fit (refs #794)" begin
    rng = MersenneTwister(7)
    p, n = 5, 200
    Λ = reshape([0.8, 0.3, 0.5, -0.4, 0.6], p, 1)
    Y = Λ * randn(rng, 1, n) + 0.5 * randn(rng, p, n)
    M = fill(NaN, p, 1)
    M[2, 1] = 0.3
    pinned = fit_gaussian_gllvm(Y; K = 1, lambda_constraint = M)
    i_pin = only(GLLVModels._lambda_constraint_pinned_theta_indices(pinned))
    @test GLLVModels._profile_all_term_names(pinned)[1][i_pin] == "Lambda_B[2,1]"
    # Each of these refits without the pins, so it would answer for another model.
    # The error names `loading_profile`, which profiles a pinned fit correctly.
    calls = (
        () -> GLLVModels.profile_ci(pinned, i_pin; y = Y),
        () -> GLLVModels.profile_ci(pinned, "sigma_eps"; y = Y),
        () -> GLLVModels.tmbprofile_wrapper(pinned, i_pin; y = Y),
        () -> GLLVModels.tmbprofile_wrapper(pinned, "sigma_eps"; y = Y),
        () -> GLLVModels.profile_curve_targets(pinned, [1]; y = Y),
        () -> GLLVModels.bootstrap_ci(pinned; y = Y, n_boot = 2),
        () -> GLLVModels.bootstrap_ci_derived(pinned, fb -> GLLVModels.communality(fb)[1]; y = Y, n_boot = 2),
        () -> GLLVModels.profile_ci_derived(pinned, θ -> θ[1]; y = Y),
        () -> GLLVModels.loading_profile_exploratory(pinned, 3, 1; y = Y),
    )
    for f in calls
        err = try
            f()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin("loading_profile", err.msg)
        @test err isa ArgumentError && occursin("lambda_constraint", err.msg)
    end
    # The pinned-fit profile itself still runs.
    lp = loading_profile(pinned; y = Y, n_grid = 3)
    @test !isempty(lp.table)

    # Unpinned fits are unaffected.
    plain = fit_gaussian_gllvm(Y; K = 1)
    r = GLLVModels.profile_ci(plain, 2; y = Y)
    @test isfinite(r.lower) && isfinite(r.upper) && r.lower < plain.pars.θ_packed[2] < r.upper
    tw = GLLVModels.tmbprofile_wrapper(plain, 2; y = Y)
    @test (tw.lower, tw.upper) == (r.lower, r.upper)
    b = GLLVModels.bootstrap_ci(plain; y = Y, n_boot = 2, parms = "sigma_eps")
    @test length(b.term) == 1
end
