# gllvm-parity-tag: P1
#
# Admission twins: R-at-P1 vs Julia fitted cases built to reach positive
# admissions that the four fits of test/parity/isdm_cases.jl do not reach
# (docs/dev-log/core070/true-parity-latest/audit-isdm-546-2026-10-01.md):
#   adm_aliased    ISDM-ALIASED       a fit that drops an aliased candidate column
#   adm_align      ISDM-ALIGN         family declared in a different order from the data's levels
#   adm_nooffset   ISDM-NO-OFFSET     no offset() term
#   adm_zeroord    ISDM-ZERO-ORDINARY all-count declaration (cloglog exception off), zero offset
#   adm_unbalanced ISDM-UNBALANCED    latent fit on an incomplete cell x trait x source grid
# Two rows stay free and are recorded as reproducers below: ISDM-MASKED-ARM
# (R fits NA responses, the Julia door refuses them) and ISDM-WRAPPER-LAW (both
# engines refuse a logit-binomial source inside isdm_sources).
#
# Julia-only: the R side ran ONCE, from test/fixtures/isdm/export_admission_twins_p1.R,
# and its numbers are recorded in test/fixtures/isdm/r_values_admission_p1.toml.
# Both engines read the same sha256-checked CSV fixtures. Tolerances are those of
# isdm_cases.jl: logLik 1e-6 (Julia marginal at R's optimum vs R; R objective at
# Julia's optimum vs Julia), b_fix and eta max abs 1e-4.
#
# Run: julia --project=. test/parity/isdm_admission_twins.jl
using Test, GLLVModels, Distributions, LinearAlgebra, TOML

include(joinpath(@__DIR__, "..", "fixtures", "isdm", "isdm_fixture_io.jl"))
include(joinpath(@__DIR__, "..", "fixtures", "isdm", "isdm_admission_cases.jl"))

const ARV = isdm_admission_r_values()
const AJE = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "julia_estimates_admission_p1.toml"))

function adm_by_name(names_from, vals, names_to)
    Set(names_from) == Set(names_to) || error("coefficient names do not pair: $names_from vs $names_to")
    return [vals[findfirst(==(n), names_from)] for n in names_to]
end
adm_r_lambda(c) = Int(c["K"]) == 0 ? zeros(2, 0) : reshape(Float64.(c["Lambda_B_colmajor"]), :, Int(c["K"]))

@testset "iSDM admission twins (P1)" begin
@test ARV["gllvmtmb_sha"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
for name in ISDM_ADMISSION_CASES
    @testset "$name" begin
        r = ARV["cases"][name]; j = AJE["julia"][name]
        c = isdm_admission_case(name)
        dat = read_isdm_csv(c.csv)
        @test bytes2hex(open(sha256, isdm_fixture_path(c.csv))) == r["fixture_sha256"]
        @test length(dat.value) == r["fixture_rows"]
        tab = isdm_table(c.formula, dat; family = c.family)
        rb = adm_by_name(r["b_fix_names"], Float64.(r["b_fix"]), tab.X_names)
        RΛ = adm_r_lambda(r)
        # Julia's marginal at R's optimum equals R's objective there, and R's
        # objective at Julia's optimum equals Julia's.
        @test abs(isdm_marginal_loglik_laplace(tab, RΛ, rb) - r["loglik"]) <= 1e-6
        jb = adm_by_name(j["b_fix_names"], Float64.(j["b_fix"]), tab.X_names)
        JΛ = j["K"] == 0 ? zeros(2, 0) : GLLVModels.unpack_lambda(Float64.(j["theta_rr_B"]), 2, Int(j["K"]))
        jll = isdm_marginal_loglik_laplace(tab, JΛ, jb)
        @test abs(jll - j["loglik"]) <= 1e-9
        @test abs(ARV["xobj"][name] - jll) <= 1e-6
        # A fresh Julia fit reproduces the recorded estimate and converges; R converged too.
        ft = fit_isdm_gllvm(tab)
        @test ft.converged && all(ft.cell_converged)
        @test r["convergence"] == 0
        @test abs(ft.loglik - j["loglik"]) <= 1e-8
        @test ft.loglik >= r["loglik"] - 1e-6
        @test maximum(abs.(ft.b_fix .- rb)) <= 1e-4
        @test maximum(abs.(ft.eta .- Float64.(r["eta"]))) <= 1e-4

        # The admission path each fit reaches (so agreement would fail if Julia's
        # path differed from R's).
        if name == "adm_aliased"
            # both engines drop the exact multiple and keep the same columns, in R's order
            @test "isdm_source:gbif:access2" ∉ tab.X_names
            @test "isdm_source:gbif:access2" ∉ r["b_fix_names"]
            @test tab.X_names == r["b_fix_names"]
            @test any(==("isdm_source:gbif:access"), tab.X_names)
        elseif name == "adm_align"
            # declared survey-first, data levels gbif-first: re-ordering happens, by name
            @test r["declared_sources"] == ["survey", "gbif"] && r["factor_levels"] == ["gbif", "survey"]
            @test String.(c.family.names) == ["survey", "gbif"]
            @test [String(c.family.names[i]) for i in tab.source_id[1:2]] == ["gbif", "gbif"]
            # and the aligned fit is the in-order fit
            tab2 = isdm_table(c.formula, dat; family = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink())))
            @test abs(isdm_marginal_loglik_laplace(tab2, zeros(2, 0), rb) - r["loglik"]) <= 1e-6
        elseif name == "adm_nooffset"
            @test tab.offset_expr === nothing && all(tab.offset .== 0) && tab.admitted
        elseif name == "adm_zeroord"
            # all-count declaration: the mixed contract is not admitted, so the cloglog
            # exception is off; the offset column is identically zero
            @test !tab.admitted && all(tab.offset .== 0) && all(tab.fid .== 2)
            @test all(dat.log_support .== 0)
        elseif name == "adm_unbalanced"
            n_full = 2 * 2 * length(unique(dat.cell_id))
            @test length(dat.value) < n_full
            @test all(any((dat.trait .== t) .& (dat.isdm_source .== s)) for t in unique(dat.trait), s in unique(dat.isdm_source))
            @test any(length(rows) < 4 for rows in tab.rows_by_unit)   # a cell with an arm missing
            @test tab.K == 1 && size(RΛ, 2) == 1
        end
    end
end

@testset "reproducer: ISDM-MASKED-ARM (rows stay free)" begin
    # R at P1 fits NA responses (dropped before fitting); the Julia integrated door
    # refuses missing responses by design (isdm_table, "Masked-response fitting is a
    # separate missing-data capability"). The engines disagree on admission, so no
    # twin is bound. Reproducer: R's recorded fit, Julia's refusal, and Julia on the
    # rows R keeps (which agrees with R), showing the gap is the door and not the model.
    rp = ARV["reproducers"]["adm_maskedna"]
    dat = read_isdm_csv(rp["fixture"])
    @test bytes2hex(open(sha256, isdm_fixture_path(rp["fixture"]))) == rp["fixture_sha256"]
    @test count(ismissing, dat.value) == rp["na_rows"] == 10
    form = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support))
    fam = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))
    @test_throws ArgumentError isdm_table(form, dat; family = fam)
    err = try; isdm_table(form, dat; family = fam); catch e; sprint(showerror, e); end
    @test occursin("refuses missing responses", err)
    keep = findall(!ismissing, dat.value)
    kept = NamedTuple{keys(dat)}(Tuple([v[keep] for v in values(dat)]))
    kept = merge(kept, (value = Float64.(kept.value),))
    @test length(keep) == rp["rows"] - rp["na_rows"]
    ft = fit_isdm_gllvm(isdm_table(form, kept; family = fam))
    @test ft.converged && abs(ft.loglik - rp["loglik"]) <= 1e-6
end

@testset "reproducer: ISDM-WRAPPER-LAW (rows stay free)" begin
    # A logit-binomial wrapper builds as an object in both engines but cannot be
    # fitted: isdm_sources refuses its law in R at P1 and in Julia. Agreement on a
    # refusal; nothing numeric to compare.
    rp = ARV["reproducers"]["wrapper_logit"]
    @test startswith(rp["isdm_sources_result"], "REFUSED:")
    @test isdm_source(Binomial(); observation = :(~ x)) isa IsdmSource
    @test_throws ArgumentError isdm_sources(gbif = Poisson(),
        survey = isdm_source(Binomial(); observation = :(~ access)))
end
end
