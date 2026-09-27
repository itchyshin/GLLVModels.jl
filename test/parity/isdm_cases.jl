# gllvm-parity-tag: P1
#
# Paired iSDM twins against gllvmTMB at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9),
# Julia-only: the R side ran ONCE, from test/fixtures/isdm/export_p1_fixtures.R,
# and its numbers are recorded in test/fixtures/isdm/r_values_p1.toml and
# test/fixtures/isdm/cloglog_grid_p1.csv. Both engines read the same
# sha256-pinned CSV fixtures. Case ids and tolerances: docs/design/isdm-port-spec.md
# section 3.4 ("Paired numeric twins").
#
# Run: julia --project=. test/parity/isdm_cases.jl
using Test, GLLVModels, Distributions, LinearAlgebra, TOML

include(joinpath(@__DIR__, "..", "fixtures", "isdm", "isdm_fixture_io.jl"))

const RV = isdm_r_values()
const JE = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "julia_estimates_p1.toml"))

relgap(a, b) = abs(a - b) / max(abs(b), eps())
function by_name(names_from, vals, names_to)
    Set(names_from) == Set(names_to) || error("coefficient names do not pair: $names_from vs $names_to")
    return [vals[findfirst(==(n), names_from)] for n in names_to]
end
# Exact observed weight of a y = 1 Bernoulli-cloglog row, -d2/d eta2 of
# log(1 - exp(-exp(eta))), evaluated in 256-bit arithmetic.
function _exact_cloglog_weight(eta)
    setprecision(BigFloat, 256) do
        λ = exp(BigFloat(eta)); e = expm1(λ)
        Float64(λ * (λ * exp(λ) - e) / e^2)
    end
end

function r_lambda(c)
    K = Int(c["K"])
    K == 0 && return zeros(2, 0)
    return reshape(Float64.(c["Lambda_B_colmajor"]), :, K)
end

# P1-ISDM-ADMISSION-20: the 20 CORE070-ISDM-*-PAIRED-CONTROL predicates of
# docs/dev-log/core070/isdm-batch-contract.json (helpers:
# test/parity/fixtures/core070_isdm_admission.R), run natively in Julia and
# compared with their R replay at P1 (admission_p1.toml). ISDM-LEGACY is the
# legacy two-source route: re-measured in R, an R-only disposition (D-296).
function _adm_fixture(sources = ["count", "detect"])
    trait = String[]; src = String[]; unit = Int[]
    for u in 1:2, s in sources, t in ("a", "b")          # expand.grid: trait fastest
        push!(trait, t); push!(src, s); push!(unit, u)
    end
    n = length(trait)
    return (trait = trait, isdm_source = src, unit = unit,
            value = [s == "detect" ? 1.0 : 2.0 for s in src],
            log_support = log.((1:n) .+ 1.0),
            x = Union{Missing, Float64}[s == "count" ? Float64(i) : missing for (i, s) in enumerate(src)],
            z = Union{Missing, Float64}[s == "detect" ? Float64(i) : missing for (i, s) in enumerate(src)])
end
_adm_laws() = isdm_sources(count = Poisson(), detect = (Binomial(), CLogLogLink()))
function _adm_ids(f, d)
    idx = [findfirst(==(Symbol(s)), f.names) for s in d.isdm_source]
    return f.fid[idx], f.lid[idx]
end
function _adm_admitted(; f = _adm_laws(), d = _adm_fixture(), ids = _adm_ids(f, d), traits = d.trait)
    GLLVModels._isdm_declared_core(f, d.isdm_source, ids[1], ids[2], traits, length(d.isdm_source))
end
_adm_sub(d, keep) = NamedTuple{keys(d)}(Tuple(v[keep] for v in values(d)))
function _adm_design()
    d = _adm_fixture()
    f = isdm_sources(count = isdm_source(Poisson(); observation = :(~ x)),
                     detect = isdm_source((Binomial(), CLogLogLink()); observation = :(~ z)))
    X = Float64.(hcat(d.trait .== "a", d.trait .== "b")); Xn = ["traita", "traitb"]
    M, names, _ = GLLVModels._isdm_observation_design(X, Xn, d, d.isdm_source, f)
    return d, f, X, Xn, M, names
end
function _adm_offset(off; allow = true, d = _adm_fixture())
    fid, lid = _adm_ids(_adm_laws(), d)
    v = off === nothing ? zeros(length(d.value)) : GLLVModels._isdm_eval_expr(off, d, length(d.value))
    return GLLVModels._isdm_prepare_offset(v, fid, lid; allow_isdm_cloglog = allow)
end

const _ADMISSION_JULIA = Dict{String, Function}(
    "ISDM-ALIASED" => () -> begin
        d, f, X, Xn, M, names = _adm_design()
        size(M, 2) == 5 && GLLVModels._isdm_qr_rank(M) == 5 && M[:, 1:2] == X && names[1:2] == Xn
    end,
    "ISDM-ALIGN" => () -> begin
        # declared in the reverse of the selector's sorted level order: alignment is by name
        d = _adm_fixture()
        f = isdm_sources(detect = isdm_source((Binomial(), CLogLogLink()); observation = :(~ z)),
                         count = isdm_source(Poisson(); observation = :(~ x)))
        f.names == [:detect, :count] && f.observation[:detect] == Expr(:call, :~, :z) &&
            f.links[1] isa CLogLogLink && _adm_admitted(f = f, d = d)
    end,
    "ISDM-COUNT" => () -> begin
        f = isdm_sources(count = Poisson(), detect = Poisson())
        f.names == [:count, :detect] && !_adm_admitted(f = f)
    end,
    "ISDM-EXTRA-SOURCE" => () -> begin
        d = _adm_fixture(); ids = _adm_ids(_adm_laws(), d)
        d2 = merge(d, (isdm_source = ["unknown"; d.isdm_source[2:end]],))
        !_adm_admitted(d = d2, ids = ids)
    end,
    "ISDM-MASKED-ARM" => () -> begin
        d = _adm_fixture()
        _adm_admitted(d = merge(d, (value = [s == "detect" ? NaN : v for (s, v) in zip(d.isdm_source, d.value)],)))
    end,
    "ISDM-MASKED-COLUMNS" => () -> begin
        d, f, X, Xn, M, names = _adm_design()
        cc = findall(startswith("isdm_source:count:"), names)
        dc = findall(startswith("isdm_source:detect:"), names)
        !isempty(cc) && !isempty(dc) && all(M[d.isdm_source .!= "count", cc] .== 0) &&
            all(M[d.isdm_source .!= "detect", dc] .== 0) && M[:, 1:2] == X && names[1:2] == Xn
    end,
    "ISDM-MISSING-IN-TRAIT" => () -> begin
        d = _adm_fixture()
        !_adm_admitted(d = _adm_sub(d, .!((d.trait .== "a") .& (d.isdm_source .== "detect"))))
    end,
    "ISDM-MISSING-SOURCE" => () -> begin
        d = _adm_fixture()
        !_adm_admitted(d = _adm_sub(d, d.isdm_source .== "count"))
    end,
    "ISDM-MIXED" => () -> _adm_admitted(),
    "ISDM-NO-OFFSET" => () -> _adm_offset(nothing) == zeros(length(_adm_fixture().value)),
    "ISDM-NO-TRAITS" => () -> !_adm_admitted(traits = nothing),
    "ISDM-SUPPORT" => () -> _adm_offset(:log_support) == _adm_fixture().log_support,
    "ISDM-THREE" => () -> begin
        f = isdm_sources(count = Poisson(), detect = (Binomial(), CLogLogLink()), literature = Poisson())
        _adm_admitted(f = f, d = _adm_fixture(String.(f.names)))
    end,
    "ISDM-UNBALANCED" => () -> begin
        d = _adm_fixture()
        _adm_admitted(d = _adm_sub(d, 2:length(d.value)))
    end,
    "ISDM-WITHIN-TRAIT-ADMIT" => () ->
        GLLVModels._isdm_assert_trait_scale([2, 1], [0, 2], ["a", "a"]; allow_isdm_mixed = true) === nothing,
    "ISDM-WRAPPER-LAW" => () -> isdm_source(Binomial(); observation = :(~ x)) isa IsdmSource,
    "ISDM-WRONG-ID" => () -> begin
        fid, lid = _adm_ids(_adm_laws(), _adm_fixture()); fid = copy(fid); fid[1] = 1
        !_adm_admitted(ids = (fid, lid))
    end,
    "ISDM-WRONG-LINK" => () -> begin
        fid, lid = _adm_ids(_adm_laws(), _adm_fixture()); lid = copy(lid); lid[1] = 2
        !_adm_admitted(ids = (fid, lid))
    end,
    "ISDM-ZERO-ORDINARY" => () -> _adm_offset(0; allow = false) == zeros(length(_adm_fixture().value)),
)

@testset "P1-ISDM-ADMISSION-20" begin
    adm = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "admission_p1.toml"))
    @test adm["gllvmtmb_sha"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
    @test length(adm["cases"]) == 20
    @test Set(keys(_ADMISSION_JULIA)) == setdiff(Set(keys(adm["cases"])), Set(["ISDM-LEGACY"]))
    for (id, r) in sort(collect(adm["cases"]))
        id == "ISDM-LEGACY" && continue                  # R-only disposition
        @test r == "TRUE"
        @test _ADMISSION_JULIA[id]() === true
    end
end

@testset "iSDM P1 paired twins" begin

@test RV["gllvmtmb_sha"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

@testset "P1-ISDM-CLOGLOG-GRID" begin
    g = read_isdm_csv("cloglog_grid_p1.csv")
    @test length(g.eta) == 48 && length(unique(g.eta)) == 24
    for i in eachindex(g.eta)
        y, η = g.y[i], g.eta[i]
        v = GLLVModels._isdm_dbinom_cloglog(y, η)
        w = GLLVModels._isdm_cloglog_obs_weight(y, η)
        s = GLLVModels._isdm_cloglog_score(y, η)
        @test isapprox(v, g.value[i]; rtol = 1e-10, atol = 0.0) || (v == 0 && g.value[i] == 0)
        # Where 1 - exp(-exp(eta)) cancels catastrophically (the middle branch just
        # above eta = -20), R's AD weight is itself rounding noise: it differs from
        # the exact weight (BigFloat) by more than 1e-6 relative. At such points two
        # AD engines cannot agree beyond that noise, so they are compared at rel 1e-7
        # and reported as a finding; every other point is held to the spec's 1e-8.
        noisy = y == 1 && relgap(g.weight[i], _exact_cloglog_weight(η)) > 1e-6
        @test isapprox(w, g.weight[i]; rtol = noisy ? 1e-7 : 1e-8, atol = 0.0) ||
              (w == 0 && g.weight[i] == 0)
        @test isapprox(s, g.score[i]; rtol = 1e-8, atol = 0.0) || (s == 0 && g.score[i] == 0)
    end
end

for name in ISDM_CASES
    @testset "$name" begin
        r = RV["cases"][name]; j = JE["julia"][name]
        c = isdm_case(name)
        dat = read_isdm_csv(c.csv)
        @test bytes2hex(open(sha256, isdm_fixture_path(c.csv))) == r["fixture_sha256"]
        @test length(dat.value) == r["fixture_rows"]
        tab = isdm_table(c.formula, dat; family = c.family)
        rb = by_name(r["b_fix_names"], Float64.(r["b_fix"]), tab.X_names)
        RΛ = r_lambda(r)

        # P1-ISDM-LOGLIK-XOBJ: Julia's marginal at R's optimum equals R's
        # objective there, and R's objective at Julia's optimum equals Julia's.
        @test abs(isdm_marginal_loglik_laplace(tab, RΛ, rb) - r["loglik"]) <= 1e-6
        jb = by_name(j["b_fix_names"], Float64.(j["b_fix"]), tab.X_names)
        JΛ = j["K"] == 0 ? zeros(2, 0) : GLLVModels.unpack_lambda(Float64.(j["theta_rr_B"]), 2, Int(j["K"]))
        jll = isdm_marginal_loglik_laplace(tab, JΛ, jb)
        @test abs(jll - j["loglik"]) <= 1e-9          # the recorded Julia estimate reproduces
        @test abs(RV["xobj"][name] - jll) <= 1e-6

        # A fresh Julia fit reproduces the recorded estimate and converges; R converged too.
        ft = fit_isdm_gllvm(tab)
        @test ft.converged && all(ft.cell_converged)
        @test r["convergence"] == 0
        @test abs(ft.loglik - j["loglik"]) <= 1e-8
        @test ft.loglik >= r["loglik"] - 1e-6           # same objective: Julia is not below R

        # P1-ISDM-ESTIMATES at R's door optimum: b_fix by name and Λ Λ' at rel 1e-4,
        # eta at abs 1e-4. Near-zero Λ Λ' entries (the ms3 fixture has no latent
        # signal, both engines put Λ at ~1e-6) are compared on the absolute scale 1e-6.
        if size(RΛ, 2) > 0
            LLr = reshape(Float64.(r["LLt_colmajor"]), 2, 2)
            LLj = ft.Λ * ft.Λ'
            @test all(isapprox.(LLj, LLr; rtol = 1e-4, atol = 1e-6))
        end
        @test maximum(abs.(ft.eta .- Float64.(r["eta"]))) <= 1e-4
        gaps = relgap.(ft.b_fix, rb)
        if name == "predict"
            @test maximum(gaps) <= 1e-4
        else
            # FINDING (recorded, not a tolerance change): R's nlminb stops at
            # "relative convergence (4)" with max|gradient| 3.9e-4 to 9.0e-4 on these
            # fits, so small coefficients sit up to 2e-5 (absolute) short of the
            # optimum; Julia's log-likelihood is higher on every case. The polished
            # comparison below closes the gap on the same TMB objective.
            @test_broken maximum(gaps) <= 1e-4
        end

        # The same comparison against R's polished optimum (nlminb restarted from
        # the door's optimum on the same TMB objective, rel.tol 1e-14).
        pb = by_name(r["b_fix_names"], Float64.(r["polished_b_fix"]), tab.X_names)
        @test maximum(relgap.(ft.b_fix, pb)) <= 1e-4
        @test abs(ft.loglik - r["polished_loglik"]) <= 1e-6

        # P1-ISDM-PREDICT: link and response, in-sample, fixed-only, and on the
        # training rows with the offset zeroed; abs 1e-4. R's output carries a
        # `species` column ("placeholder" on this fit, spec Q8); the twin compares
        # the four shared columns.
        if haskey(r, "predict_link")
            @test r["predict_columns"] == ["cell_id", "species", "trait", "isdm_source", "est"]
            out = predict(ft)
            @test keys(out) == (:cell_id, :trait, :isdm_source, :est)
            @test out.isdm_source == dat.isdm_source && out.cell_id == dat.cell_id && out.trait == dat.trait
            nd0 = merge(dat, (log_support = zeros(length(dat.value)),))
            for (got, key) in ((out.est, "predict_link"),
                               (predict(ft; type = :response).est, "predict_response"),
                               (predict(ft; re_form = :zero).est, "predict_link_zero_re"),
                               (predict(ft; newdata = nd0).est, "predict_newdata_offset0_link"),
                               (predict(ft; newdata = nd0, type = :response).est,
                                "predict_newdata_offset0_response"))
                @test maximum(abs.(got .- Float64.(r[key]))) <= 1e-4
            end
        end
    end
end

end
