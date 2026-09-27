# Integrated species distribution model (iSDM) twins of gllvmTMB's public
# door `gllvmTMB(..., family = isdm_sources(...))` at P1 (gllvmTMB 9539352f6):
# constructors, contract refusals, long-table assembly, the per-cell kernel,
# the one-step implicit gradient, and fits. Identities and refusals only; the
# paired R-versus-Julia numbers live in test/parity/isdm_cases.jl.
# R test file and line of each twin: docs/design/isdm-port-spec.md section 3.4.
using Test, GLLVModels, Distributions, ForwardDiff, LinearAlgebra

include(joinpath(@__DIR__, "fixtures", "isdm", "isdm_fixture_io.jl"))

const _CL = (Binomial(), CLogLogLink())

# "class" tolerance of the port spec: the call throws an ArgumentError whose
# message STARTS with R's cli-rendered first line.
macro test_first_line(prefix, ex)
    quote
        local m = try
            $(esc(ex)); "(no error thrown)"
        catch e
            e isa ArgumentError ? e.msg : sprint(showerror, e)
        end
        @test startswith(m, $(esc(prefix)))
    end
end

# The public-door fixture of test-isdm-public-door.R:119-143, on the declared
# route (source names gbif / survey): 2 cells x 2 traits x 2 sources.
function _isdm_door_fixture()
    cell = repeat(["c1", "c2"], 4)
    trait = repeat(repeat(["sp1", "sp2"], inner = 2), 2)
    src = repeat(["gbif", "survey"], inner = 4)
    value = [3.0, 2, 4, 1, 1, 0, 1, 0]
    fail = [0, 0, 0, 0, 2, 3, 1, 2]
    return (cell_id = cell, trait = trait, isdm_source = src,
            isdm_gbif = Float64.(src .== "gbif"), log_support = fill(0.1, 8),
            value = value, succ = value, fail = fail)
end

const _DOOR_FORMULA = :(value ~ 0 + trait + trait & isdm_gbif + offset(log_support) +
                        latent(0 + trait | cell_id, d = 1, unique = FALSE))

_subset(nt, keep) = NamedTuple{keys(nt)}(Tuple(v[keep] for v in values(nt)))
_replace_col(nt, k, v) = merge(nt, NamedTuple{(k,)}((v,)))

@testset "iSDM (gllvmTMB P1 public door)" begin

@testset "isdm_sources() validates its declaration (test-isdm-multisource.R:33-51, 74-84)" begin
    fam = isdm_sources(gbif = Poisson(), literature = Poisson(), survey = _CL)
    @test fam isa IsdmSources
    @test fam.names == [:gbif, :literature, :survey]
    @test fam.fid == [2, 2, 1] && fam.lid == [0, 0, 2]
    @test_first_line "`isdm_sources()` needs at least two named sources" isdm_sources(gbif = Poisson())
    @test_throws r"at least two" isdm_sources(Poisson(), _CL)
    @test_throws r"declared twice" isdm_sources(:a => Poisson(), :a => _CL)
    @test_throws r"an observation law that is not admitted" isdm_sources(a = Poisson(), b = Binomial())
    @test_throws r"an observation law that is not admitted" isdm_sources(a = Poisson(), b = (Binomial(), ProbitLink()))
    @test_throws r"an observation law that is not admitted" isdm_sources(a = Poisson(), b = Normal())
    @test_first_line "An integrated declaration needs at least one count arm" isdm_sources(survey_a = _CL, survey_b = _CL)
    # The Pair spelling of a law and of a name are equivalent to the keyword form.
    @test isdm_sources(:gbif => Poisson(), :survey => (Binomial() => CLogLogLink())).fid == [2, 1]
end

@testset "isdm_source() keeps the law and the formula (test-isdm-source-formula.R:60-76)" begin
    w = isdm_source(Poisson(); observation = :(~ access + popdens))
    @test w isa IsdmSource
    @test w.family isa Poisson && w.link isa LogLink
    @test w.observation == Expr(:call, :~, :(access + popdens))
    @test_first_line "`observation` must be a one-sided formula" isdm_source(Poisson(); observation = :(access + popdens))
    @test_first_line "`family` must be an R <family> object" isdm_source(Any[]; observation = :(~ access))
    wrapped = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ access + popdens)),
                           inat = isdm_source(Poisson(); observation = :(~ access + popdens)),
                           survey = isdm_source(Poisson(); observation = :(~ observer + method)))
    @test wrapped.names == [:gbif, :inat, :survey]
    @test Set(keys(wrapped.observation)) == Set(wrapped.names)
    bare = isdm_sources(gbif = Poisson(), inat = Poisson(), survey = _CL)
    @test isempty(bare.observation)
    mixed = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ 0 + access)),
                         inat = Poisson(), survey = Poisson())
    @test mixed.observation[:inat] === nothing
end

@testset "the cloglog offset opens only inside the contract (test-isdm-public-door.R:5-70)" begin
    off = [1.5, 2.5]
    @test GLLVModels._isdm_prepare_offset(copy(off), [2, 1], [0, 2]; allow_isdm_cloglog = true) == off
    @test_first_line "offsets are supported for count families (poisson, nbinom) only" GLLVModels._isdm_prepare_offset(
        copy(off), [2, 1], [0, 2]; allow_isdm_cloglog = false)
    # gaussian, binomial-logit, binomial-probit and Beta stay refused with the flag on.
    for (fid, lid) in ((0, 0), (1, 0), (1, 1), (7, 0))
        @test_throws r"count families" GLLVModels._isdm_prepare_offset([2.0], [fid], [lid];
                                                                        allow_isdm_cloglog = true)
    end
    @test GLLVModels._isdm_prepare_offset([0.0, 0.0], [0, 1], [0, 0]) == [0.0, 0.0]   # zero always passes
    @test_throws r"non-finite" GLLVModels._isdm_prepare_offset([Inf], [2], [0])
end

@testset "the contract requires every source within every trait (public-door 71-144; multisource 136-159)" begin
    fam = isdm_sources(gbif = Poisson(), survey = _CL)
    core = GLLVModels._isdm_declared_core
    @test core(fam, ["gbif", "survey"], [2, 1], [0, 2], ["A", "A"], 2) === true
    @test core(fam, ["gbif", "gbif", "survey", "survey"], [2, 2, 1, 1], [0, 0, 2, 2],
               ["A", "A", "B", "B"], 4) === false
    @test core(fam, ["gbif", "survey", "survey", "survey"], [2, 1, 1, 1], [0, 2, 2, 2],
               ["A", "B", "B", "B"], 4) === false
    # the declared route admits the n = 2 shape under an honest declaration
    @test core(fam, ["gbif", "survey"], [2, 1], [0, 2], ["sp1", "sp1"], 2) === true
    # an all-count declaration is never admitted (it needs no relaxation)
    allc = isdm_sources(a = Poisson(), b = Poisson())
    @test core(allc, ["a", "b"], [2, 2], [0, 0], ["sp1", "sp1"], 2) === false
end

@testset "observation designs are source-masked (test-isdm-source-formula.R:77-147)" begin
    dat = (trait = repeat(["sp1", "sp2"], 3),
           isdm_source = repeat(["gbif", "inat", "survey"], inner = 2),
           access = Union{Missing, Float64}[0.2, -0.1, 0.4, -0.3, missing, missing],
           popdens = Union{Missing, Float64}[1.0, 1.2, 0.8, 0.7, missing, missing],
           observer = Union{Missing, String}[missing, missing, missing, missing, "o1", "o2"],
           method = Union{Missing, String}[missing, missing, missing, missing, "walk", "point"])
    X = Float64.(hcat(dat.trait .== "sp1", dat.trait .== "sp2")); Xn = ["traitsp1", "traitsp2"]
    fam = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ access + popdens)),
                       inat = isdm_source(Poisson(); observation = :(~ access + popdens)),
                       survey = isdm_source(Poisson(); observation = :(~ observer + method)))
    out, names, _ = GLLVModels._isdm_observation_design(X, Xn, dat, dat.isdm_source, fam)
    added = setdiff(names, Xn)
    @test all(startswith("isdm_source:"), added)
    for s in ("gbif", "survey")
        cols = findall(startswith("isdm_source:$s"), names)
        @test all(out[dat.isdm_source .!= s, cols] .== 0)
    end
    @test !any(isnan, out)

    noint, nn, _ = GLLVModels._isdm_observation_design(X, Xn, dat, dat.isdm_source,
        isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ 0 + access)),
                     inat = Poisson(), survey = Poisson()))
    j = findfirst(==("isdm_source:gbif:access"), nn)
    @test j !== nothing
    @test all(noint[dat.isdm_source .!= "gbif", j] .== 0)

    sdat = (trait = repeat(["sp1", "sp2"], 6), isdm_source = repeat(["gbif", "survey"], inner = 6),
            observer = Union{Missing, String}[fill(missing, 6); "o1"; "o1"; "o2"; "o2"; "o1"; "o2"],
            method = Union{Missing, String}[fill(missing, 6); "walk"; "point"; "point"; "walk"; "walk"; "point"])
    SX = Float64.(hcat(sdat.trait .== "sp1", sdat.trait .== "sp2"))
    sb, sn, _ = GLLVModels._isdm_observation_design(SX, Xn, sdat, sdat.isdm_source,
        isdm_sources(gbif = Poisson(), survey = isdm_source(Poisson(); observation = :(~ observer + method))))
    scols = findall(startswith("isdm_source:survey:"), sn)
    @test length(scols) > 0
    @test all(sb[sdat.isdm_source .!= "survey", scols] .== 0)

    @test_throws r"uses variable\(s\) not found in `data`" GLLVModels._isdm_observation_design(
        X, Xn, dat, dat.isdm_source,
        isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ unavailable)),
                     inat = Poisson(), survey = Poisson()))
    dna = _replace_col(dat, :access, Union{Missing, Float64}[missing, -0.1, 0.4, -0.3, missing, missing])
    @test_throws r"has missing values after source filtering" GLLVModels._isdm_observation_design(
        X, Xn, dna, dna.isdm_source, fam)
    dupn = ["traitsp1", "traitsp2", "traitsp1:isdm_sourceinat", "traitsp2:isdm_sourceinat"]
    @test_first_line "Top-level `isdm_source` fixed effects duplicate" GLLVModels._isdm_observation_design(
        hcat(X, X), dupn, dat, dat.isdm_source, fam)
end

@testset "fit-time refusals in R's order (multisource 85-117; public-door 145-184)" begin
    ms = read_isdm_csv("isdm_ms3.csv")
    fam = isdm_sources(gbif = Poisson(), literature = Poisson(), survey = _CL)
    f = :(value ~ 0 + trait + trait & env + offset(log_support) +
          latent(0 + trait | cell_id, d = 1, unique = FALSE))
    bad = _replace_col(ms, :isdm_source, [i == 1 ? "mystery" : s for (i, s) in enumerate(ms.isdm_source)])
    @test_first_line "length(family) must match the number of distinct levels in" isdm_table(f, bad; family = fam)
    inc = _subset(ms, .!((ms.trait .== "sp1") .& (ms.isdm_source .== "survey")))
    @test_first_line "Response family/link cannot currently vary across rows within a trait" isdm_table(f, inc; family = fam)
    @test_first_line "`weights` is not admitted for the integrated multi-source model" isdm_table(
        f, ms; family = fam, weights = ones(length(ms.value)))

    door = _isdm_door_fixture()
    dfam = isdm_sources(gbif = Poisson(), survey = _CL)
    @test isdm_table(_DOOR_FORMULA, door; family = dfam).admitted
    @test_first_line "`weights` is not admitted" isdm_table(_DOOR_FORMULA, door; family = dfam,
                                                          weights = repeat([1.0, 3.0], inner = 4))
    @test_first_line "The integrated multi-source model admits only single-trial detection rows" isdm_table(
        _DOOR_FORMULA, door; family = dfam, n_trials = door.succ .+ door.fail)

    # Julia-side scope refusals (D-296): missing responses; R's latent() default.
    dm = merge(door, (value = Union{Missing, Float64}[missing; door.value[2:end]],))
    @test_throws r"refuses missing responses" isdm_table(_DOOR_FORMULA, dm; family = dfam)
    fdef = :(value ~ 0 + trait + trait & isdm_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 1))
    @test_throws r"unique = TRUE\) is R's default" isdm_table(fdef, door; family = dfam)
    fphy = :(value ~ 0 + trait + offset(log_support) + phylo_latent(species, d = 1))
    @test_throws r"not admitted on the integrated door" isdm_table(fphy, door; family = dfam)
    @test_throws r"zero or one latent\(\)" isdm_table(
        :(value ~ 0 + trait + latent(0 + trait | cell_id, d = 1, unique = FALSE) +
          latent(0 + trait | cell_id, d = 1, unique = FALSE)), door; family = dfam)
    @test_throws r"must be the unit column" isdm_table(
        :(value ~ 0 + trait + latent(0 + trait | trait, d = 1, unique = FALSE)), door; family = dfam)

    # every declared source x trait arm needs an observed response (R/isdm-sources.R:445-477)
    @test_first_line "Every declared integrated source-trait arm needs an observed response" GLLVModels._isdm_assert_observed_arms(
        ["gbif", "survey", "gbif", "survey"], ["A", "A", "B", "B"], [1, 1, 1, 0], [:gbif, :survey])
    @test GLLVModels._isdm_assert_observed_arms(["gbif", "survey"], ["A", "A"], [1, 1], [:gbif, :survey])
end

@testset "long-table assembly" begin
    c = isdm_case("predict")
    tab = isdm_table(c.formula, read_isdm_csv(c.csv); family = c.family)
    @test tab.admitted && tab.K == 1
    @test tab.X_names == ["traitsp1", "traitsp2", "traitsp1:env", "traitsp2:env",
                          "traitsp1:src_gbif", "traitsp2:src_gbif"]
    @test length(tab.y) == 120 && length(tab.rows_by_unit) == 30
    @test all(length(r) == 4 for r in tab.rows_by_unit)
    @test tab.fid == [s == 1 ? 2 : 1 for s in tab.source_id]
    @test tab.offset ≈ log.(ifelse.(tab.fid .== 2, 1.5, 0.9)) atol = 1e-14
    @test sort(reduce(vcat, tab.rows_by_unit)) == 1:120
end

@testset "cloglog kernel is Dual-safe on the tail grid" begin
    for y in (0.0, 1.0), η in (-40.0, -20.5, -20.0, -19.5, 0.0, 5.0, 40.0, 699.5, 700.0, 720.0)
        v = GLLVModels._isdm_dbinom_cloglog(y, η)
        w = GLLVModels._isdm_cloglog_obs_weight(y, η)
        @test isfinite(v) && isfinite(w) && w >= 0
    end
    # in the ordinary range the copy is the Bernoulli-cloglog log-density
    for y in (0, 1), η in (-3.0, -0.4, 0.0, 1.2)
        p = -expm1(-exp(η))
        @test GLLVModels._isdm_dbinom_cloglog(y, η) ≈ (y == 1 ? log(p) : log1p(-p)) rtol = 1e-12
        @test GLLVModels._isdm_cloglog_score(y, η) ≈
              ForwardDiff.derivative(e -> y == 1 ? log(-expm1(-exp(e))) : -exp(e), η) rtol = 1e-10
    end
end

@testset "one-step implicit gradient agrees with central differences" begin
    c = isdm_case("predict")
    tab = isdm_table(c.formula, read_isdm_csv(c.csv); family = c.family)
    ft = fit_isdm_gllvm(tab)
    pX = size(tab.X, 2)
    θhat = GLLVModels._isdm_pack(ft.b_fix, ft.Λ)
    b0, L0 = GLLVModels._isdm_start(tab, 1)
    θ0 = GLLVModels._isdm_pack(b0, L0)
    marg(θ) = begin
        b, Λ = GLLVModels._isdm_unpack(θ, pX, 2, 1)
        isdm_marginal_loglik_laplace(tab, Λ, b; tol = 1e-13)
    end
    for θ in (θ0, (θ0 .+ θhat) ./ 2, θhat)
        ga = GLLVModels.isdm_laplace_grad(tab, θ)
        @test ga !== nothing
        h = 1e-5
        gfd = [(marg(θ .+ h .* (1:length(θ) .== i)) - marg(θ .- h .* (1:length(θ) .== i))) / (2h)
               for i in eachindex(θ)]
        @test norm(ga .- gfd) <= 1e-6 * max(1.0, norm(gfd))
    end
    # the K = 0 route (a GLM through the same kernel) with a detection source
    c0 = isdm_case("srcform_mixed")
    t0 = isdm_table(c0.formula, read_isdm_csv(c0.csv); family = c0.family)
    b = GLLVModels._isdm_start(t0, 0)[1]
    m0(bb) = isdm_marginal_loglik_laplace(t0, zeros(2, 0), bb)
    g0 = GLLVModels.isdm_laplace_grad(t0, b)
    gfd0 = [(m0(b .+ 1e-6 .* (1:length(b) .== i)) - m0(b .- 1e-6 .* (1:length(b) .== i))) / 2e-6
            for i in eachindex(b)]
    @test norm(g0 .- gfd0) <= 1e-6 * max(1.0, norm(gfd0))
end

@testset "fits: one count + one detection source, three sources, K = 0, all-count" begin
    c = isdm_case("predict")
    ft = fit_isdm_gllvm(c.formula, read_isdm_csv(c.csv); family = c.family)
    @test ft isa IsdmFit
    @test ft.converged && all(ft.cell_converged)
    @test isfinite(ft.loglik) && size(ft.Λ) == (2, 1) && size(ft.zhat) == (1, 30)
    tab = ft.table
    @test ft.loglik ≈ isdm_marginal_loglik_laplace(tab, ft.Λ, ft.b_fix) atol = 1e-10
    @test ft.eta ≈ tab.X * ft.b_fix .+ tab.offset .+
          [dot(ft.Λ[tab.trait_id[o], :], ft.zhat[:, tab.unit_id[o]]) for o in eachindex(tab.y)] atol = 1e-12

    c3 = isdm_case("ms3")
    f3 = fit_isdm_gllvm(c3.formula, read_isdm_csv(c3.csv); family = c3.family)
    @test f3.converged && all(f3.cell_converged)

    # all-Poisson source formulas, no latent term (test-isdm-source-formula.R:148-170)
    cp = isdm_case("srcform_pois")
    fp = fit_isdm_gllvm(cp.formula, read_isdm_csv(cp.csv); family = cp.family)
    @test fp.converged && size(fp.Λ, 2) == 0 && !fp.table.admitted
    ja = findfirst(==("isdm_source:gbif:access"), fp.b_names)
    @test abs(fp.b_fix[ja] - 0.5) <= 0.15
    gb = [fp.table.sources.names[s] == :gbif for s in fp.table.source_id]
    @test all(fp.table.X[.!gb, ja] .== 0)
    @test any(startswith("isdm_source:survey:"), fp.b_names)
    @test !any(isnan, fp.b_fix)

    # mixed Poisson and cloglog source formulas (test-isdm-source-formula.R:171-200)
    cm = isdm_case("srcform_mixed")
    fm = fit_isdm_gllvm(cm.formula, read_isdm_csv(cm.csv); family = cm.family)
    @test fm.converged && fm.table.admitted
    sv = findall(startswith("isdm_source:survey:"), fm.b_names)
    @test length(sv) > 0
    sr = [fm.table.sources.names[s] == :survey for s in fm.table.source_id]
    @test all(fm.table.X[.!sr, sv] .== 0)

    # an all-count declaration fits through the same kernel (D-296 item 3); weights fenced
    ms = read_isdm_csv("isdm_ms3.csv")
    cnt = _subset(ms, ms.isdm_source .!= "survey")
    fa = isdm_sources(gbif = Poisson(), literature = Poisson())
    fc = :(value ~ 0 + trait + trait & env + offset(log_support) +
           latent(0 + trait | cell_id, d = 1, unique = FALSE))
    fac = fit_isdm_gllvm(fc, cnt; family = fa)
    @test fac.converged && !fac.table.admitted
    @test_throws r"`weights` is not available on the Julia integrated door" isdm_table(
        fc, cnt; family = fa, weights = ones(length(cnt.value)))
end

end
