# Integrated species distribution model (iSDM) twins of gllvmTMB's public
# door `gllvmTMB(..., family = isdm_sources(...))` at P1 (gllvmTMB 9539352f6):
# constructors, contract refusals, long-table assembly, the per-cell kernel,
# the one-step implicit gradient, and fits. Identities and refusals only; the
# paired R-versus-Julia numbers live in test/parity/isdm_cases.jl.
# R test file and line of each twin: docs/design/isdm-port-spec.md section 3.4.
using Test, GLLVModels, Distributions, ForwardDiff, LinearAlgebra, Statistics

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

    # Julia-side scope refusal (D-296): missing responses.
    dm = merge(door, (value = Union{Missing, Float64}[missing; door.value[2:end]],))
    @test_throws r"refuses missing responses" isdm_table(_DOOR_FORMULA, dm; family = dfam)
    # R's latent() default, unique = TRUE, is admitted (the unit-level unique
    # variance, theta_diag_B); a non-literal `unique` is refused.
    fdef = :(value ~ 0 + trait + trait & isdm_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 1))
    @test isdm_table(fdef, door; family = dfam).unique
    @test isdm_table(:(value ~ 0 + trait + trait & isdm_gbif + offset(log_support) +
                       latent(0 + trait | cell_id, d = 1, unique = TRUE)), door; family = dfam).unique
    @test !isdm_table(_DOOR_FORMULA, door; family = dfam).unique
    @test !isdm_table(:(value ~ 0 + trait + offset(log_support)), door; family = dfam).unique
    @test_throws r"`unique` must be the literal TRUE or FALSE" isdm_table(
        :(value ~ 0 + trait + latent(0 + trait | cell_id, d = 1, unique = maybe)), door; family = dfam)
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

@testset "latent(unique = TRUE): the unit-level unique variance (R's theta_diag_B)" begin
    c = isdm_psi_case("psi4")
    dat = read_isdm_csv(c.csv)
    tab = isdm_table(c.formula, dat; family = c.family)
    @test tab.unique && tab.K == 1 && length(tab.trait_levels) == 4
    f_nou = Meta.parse(replace(string(c.formula), "d = 1)" => "d = 1, unique = FALSE)"))
    tab0 = isdm_table(f_nou, dat; family = c.family)
    @test !tab0.unique && tab0.X_names == tab.X_names
    pX = size(tab.X, 2); p = 4
    b0, L0 = GLLVModels._isdm_start(tab, 1)

    # K = 0 on a unique table is refused up front, as R refuses latent(d = 0)
    # ("loading rank must be between 1 and the number of rows", measured at P1):
    # R has no unique variance without a latent() term.
    @test_throws r"K = 0 on a table built from latent\(\.\.\., unique = TRUE\)" fit_isdm_gllvm(tab; K = 0)
    @test fit_isdm_gllvm(tab0; K = 0).converged       # the loadings-only table still fits a GLM

    # theta_diag_B is required on a unique table and refused on any other.
    @test_throws r"pass `theta_diag_B`" isdm_marginal_loglik_laplace(tab, L0, b0)
    @test_throws r"`theta_diag_B` was given" isdm_marginal_loglik_laplace(tab0, L0, b0; theta_diag_B = zeros(p))
    @test_throws DimensionMismatch isdm_marginal_loglik_laplace(tab, L0, b0; theta_diag_B = zeros(p - 1))

    # A vanishing unique SD recovers the loadings-only marginal.
    @test isdm_marginal_loglik_laplace(tab, L0, b0; theta_diag_B = fill(-40.0, p)) ≈
          isdm_marginal_loglik_laplace(tab0, L0, b0) atol = 1e-9

    # The augmented kernel is the joint Laplace over (z, s_B): with K = 1 and one
    # cell it equals a direct Laplace in the original s_B scale (no change of
    # variables), evaluated with ForwardDiff's Hessian of the joint log density.
    θd = [-0.4, -1.1, -0.35, -1.0]
    rows = tab.rows_by_unit[1]
    eta0 = (tab.X * b0 .+ tab.offset)[rows]
    y = tab.y[rows]; fid = tab.fid[rows]; tr = tab.trait_id[rows]
    Λa = GLLVModels._isdm_augment(L0, θd)
    v_aug, zaug, ok = GLLVModels._isdm_cell_loglik(y, fid, tr, eta0, Λa; tol = 1e-13)
    @test ok
    joint(w) = begin                       # w = [z; s_B], s_B in its own scale
        acc = -0.5 * w[1]^2
        for t in 1:p
            sd = exp(θd[t])
            acc += -0.5 * (w[1 + t] / sd)^2 - log(sd)
        end
        for o in eachindex(y)
            acc += GLLVModels._isdm_row_logpdf(fid[o], y[o], eta0[o] + L0[tr[o], 1] * w[1] + w[1 + tr[o]])
        end
        acc
    end
    ŵ = vcat(zaug[1], exp.(θd) .* zaug[2:end])
    @test norm(ForwardDiff.gradient(joint, ŵ)) <= 1e-8
    H = -ForwardDiff.hessian(joint, ŵ)
    @test v_aug ≈ joint(ŵ) - 0.5 * logdet(Symmetric(H)) rtol = 1e-12

    # The one-step implicit gradient over theta = [b; pack(Λ); theta_diag_B]
    # (R's opt$par order) agrees with central differences of the exact Laplace
    # value. FINDING (measured): the kernel's mode search, a copy of the
    # `_mixed_laplace_mode` rule, accepts a mode at the floating-point floor
    # with a residual step up to sqrt(eps) * (1 + |z|); the log-determinant is
    # first-order in that residual, so a cell's value can move by ~2e-8 between
    # neighbouring theta (cell 49 at the start, theta_rr_B[1] + 1e-5). That is far
    # below the 1e-6 receipts but swamps a 1e-5 central difference, so the
    # reference polishes each cell's mode with three Newton steps on the
    # ForwardDiff Hessian before evaluating the value.
    θ0 = GLLVModels._isdm_pack(b0, L0, zeros(p))
    θ1 = GLLVModels._isdm_pack(b0, L0, θd)
    marg(θ) = begin
        b, Λ, t, Λa = GLLVModels._isdm_unpack_unique(θ, pX, p, 1)
        e0 = tab.X * b .+ tab.offset
        acc = 0.0
        for rows in tab.rows_by_unit
            yy = tab.y[rows]; ff = tab.fid[rows]; tt = tab.trait_id[rows]; ee = e0[rows]
            z, ok = GLLVModels._isdm_cell_mode(yy, ff, tt, ee, Λa; tol = 1e-13)
            lp(w) = GLLVModels._isdm_cell_logpost(yy, ff, tt, ee, Λa, w)
            for _ in 1:3
                z = z .- ForwardDiff.hessian(lp, z) \ ForwardDiff.gradient(lp, z)
            end
            acc += GLLVModels._isdm_cell_value_at(yy, ff, tt, ee, Λa, z)
        end
        acc
    end
    for θ in (θ0, θ1)
        ga = GLLVModels.isdm_laplace_grad(tab, θ)
        @test ga !== nothing && length(ga) == pX + p + p
        h = 1e-5
        gfd = [(marg(θ .+ h .* (1:length(θ) .== i)) - marg(θ .- h .* (1:length(θ) .== i))) / (2h)
               for i in eachindex(θ)]
        @test norm(ga .- gfd) <= 1e-6 * max(1.0, norm(gfd))
    end

    # The fit: fields, eta at the modes, and predict re-adding s_B on seen units only.
    ft = fit_isdm_gllvm(tab)
    @test ft.converged && ft.unique && length(ft.theta_diag_B) == p && size(ft.s_B) == (p, 80)
    @test occursin("unique", sprint(show, ft))
    @test ft.loglik ≈ isdm_marginal_loglik_laplace(tab, ft.Λ, ft.b_fix;
                                                   theta_diag_B = ft.theta_diag_B) atol = 1e-10
    eta_chk = tab.X * ft.b_fix .+ tab.offset .+
        [dot(ft.Λ[tab.trait_id[o], :], ft.zhat[:, tab.unit_id[o]]) + ft.s_B[tab.trait_id[o], tab.unit_id[o]]
         for o in eachindex(tab.y)]
    @test maximum(abs.(ft.eta .- eta_chk)) <= 1e-12
    @test predict(ft).est == ft.eta
    @test maximum(abs.(predict(ft; newdata = dat).est .- ft.eta)) <= 1e-12
    nu = merge(dat, (cell_id = [i <= 12 ? "c_new" : v for (i, v) in enumerate(dat.cell_id)],))
    pu = predict(ft; newdata = nu).est
    fixed_only = tab.X * ft.b_fix .+ tab.offset
    @test maximum(abs.(pu[1:12] .- fixed_only[1:12])) <= 1e-12
    @test maximum(abs.(pu[13:end] .- ft.eta[13:end])) <= 1e-12
end

@testset "predict and fitted (test-isdm-predict.R, non-spatial blocks)" begin
    c = isdm_case("predict"); dat = read_isdm_csv(c.csv)
    ft = fit_isdm_gllvm(c.formula, dat; family = c.family)
    tab = ft.table
    out = predict(ft)
    ic = dat.isdm_source .== "gbif"; ip = .!ic
    # 55-66: in-sample link prediction is report$eta, one row per data row
    @test out isa NamedTuple && length(out.est) == 120 && haskey(out, :est)
    @test out.est == ft.eta
    # 67-83: each row's own inverse link
    resp = predict(ft; type = :response)
    @test resp.est[ic] == exp.(out.est[ic])
    @test resp.est[ip] == -expm1.(-exp.(out.est[ip]))
    @test all(0 .<= resp.est[ip] .<= 1) && all(resp.est[ic] .> 0)
    # 84-93: newdata = training reproduces in-sample
    @test isapprox(predict(ft; newdata = dat).est, out.est; atol = 1e-10)
    # 94-110: re_form zero on newdata is fixed effects + offset, and differs from full
    wre = predict(ft; newdata = dat, re_form = :all)
    fo = predict(ft; newdata = dat, re_form = :zero)
    @test std(wre.est .- fo.est) > 0
    @test isapprox(fo.est, tab.X * ft.b_fix .+ tab.offset; atol = 1e-10)
    # 111-127: only the newdata refusal is twinned; in-sample se.fit is fenced (D-296)
    @test_first_line "`se.fit = TRUE` is not yet supported together with `newdata`." predict(
        ft; newdata = dat, se_fit = true)
    @test_throws ArgumentError predict(ft; se_fit = true)
    # 128-149: an unseen unit falls back to the fixed-only prediction
    first_cell = sort(unique(dat.cell_id))[1]
    rows1 = dat.cell_id .== first_cell
    nn = merge(_subset(dat, rows1), (cell_id = fill("cNEW", count(rows1)), env = fill(0.25, count(rows1))))
    @test predict(ft; newdata = nn).est == predict(ft; newdata = nn, re_form = :zero).est
    # 150-175: newdata response uses each row's arm
    ir = predict(ft; type = :response)
    nr = predict(ft; newdata = dat, type = :response)
    @test isapprox(nr.est, ir.est; atol = 1e-10)
    @test all(0 .<= nr.est[ip] .<= 1)
    nl = predict(ft; newdata = dat, type = :link)
    @test nr.est[ip] == -expm1.(-exp.(nl.est[ip]))
    # 176-211: re_form honoured in-sample for every zero form; fitted forwards
    z = predict(ft; re_form = :zero)
    @test z.est == tab.X * ft.b_fix .+ tab.offset
    @test std(out.est .- z.est) > 0
    @test predict(ft; re_form = nothing).est == z.est
    @test predict(ft; re_form = 0).est == z.est
    @test predict(ft; re_form = missing).est == z.est
    @test fitted(ft; type = :link, re_form = :zero).est == z.est
    # R warns (class gllvmTMB_predict_re_form_unsupported); Julia has no formula-valued
    # re_form, so an unsupported form is an error (a recorded fence).
    @test_throws ArgumentError predict(ft; re_form = :intercept)
    # 467-489: the source column is carried, est last, fitted inherits
    @test keys(out) == (:cell_id, :trait, :isdm_source, :est)
    @test out.isdm_source == dat.isdm_source
    @test keys(fitted(ft; type = :link)) == keys(out)
    @test fitted(ft).est == resp.est                       # fitted defaults to the response scale
    # 511-529: zeroing the offset in newdata moves the link prediction by exactly log(support)
    nd0 = merge(dat, (log_support = zeros(length(dat.value)),))
    @test isapprox(predict(ft; newdata = dat).est .- predict(ft; newdata = nd0).est,
                   log.(dat.support); atol = 1e-12)
    # 530-556: newdata without the response column
    nov = Base.structdiff(dat, NamedTuple{(:value,)})
    @test predict(ft; newdata = nov).est == predict(ft; newdata = dat).est
    @test isapprox(predict(ft; newdata = nov).est, out.est; atol = 1e-10)
    @test isapprox(predict(ft; newdata = nov, type = :response).est, resp.est; atol = 1e-10)
    # source-column refusals (R/methods-gllvmTMB.R:2876-2905)
    @test_first_line "Integrated-source prediction needs source column `isdm_source` in `newdata`." predict(
        ft; newdata = Base.structdiff(dat, NamedTuple{(:isdm_source,)}))
    @test_first_line "New data names undeclared integrated source" predict(
        ft; newdata = merge(dat, (isdm_source = fill("mystery", 120),)))
    @test_first_line "Integrated-source prediction does not allow missing source labels" predict(
        ft; newdata = merge(dat, (isdm_source = Union{Missing, String}[missing; dat.isdm_source[2:end]],)))
    @test_first_line "`newdata` is missing the offset variable" predict(
        ft; newdata = Base.structdiff(dat, NamedTuple{(:log_support,)}))

    # source-observation columns are rebuilt from the frozen basis (K = 0, mixed laws)
    cm = isdm_case("srcform_mixed"); dm = read_isdm_csv(cm.csv)
    fm = fit_isdm_gllvm(cm.formula, dm; family = cm.family)
    @test isapprox(predict(fm; newdata = dm).est, predict(fm).est; atol = 1e-10)
    @test predict(fm).est == fm.eta
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
