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
