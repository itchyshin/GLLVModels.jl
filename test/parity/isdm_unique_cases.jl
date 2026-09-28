# gllvm-parity-tag: P1
#
# Paired twins of R's default `latent(..., unique = TRUE)` on the integrated
# door (the per-trait unit-level unique variance, TMB parameter theta_diag_B)
# against gllvmTMB at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9),
# Julia-only: the R side ran ONCE, from test/fixtures/isdm/export_psi_fixtures.R,
# and its numbers are recorded in test/fixtures/isdm/r_values_psi_p1.toml. Both
# engines read the same sha256-pinned CSV fixtures.
#
#   psi4             four traits, one factor: identified (p >= 2K + 1); the
#                    full receipt set is asserted.
#   predict_default  the two-trait fixtures of test/parity/isdm_cases.jl fitted
#   ms3_default      with R's default formula. With p = 2, K = 1 the unique SDs
#                    are not identified and run toward the boundary in both
#                    engines (R: 4e-5 and 2e-5; 6e-6 and 4e-5); the
#                    log-likelihood and the cross-objective identity are
#                    asserted, theta_diag_B is documented, not asserted.
#
# Run: julia --project=. test/parity/isdm_unique_cases.jl
using Test, GLLVModels, Distributions, LinearAlgebra, TOML

include(joinpath(@__DIR__, "..", "fixtures", "isdm", "isdm_fixture_io.jl"))

const RVU = isdm_psi_r_values()
const JEU = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "julia_estimates_psi_p1.toml"))

_relgap(a, b) = abs(a - b) / max(abs(b), eps())
function _by_name(names_from, vals, names_to)
    Set(names_from) == Set(names_to) || error("coefficient names do not pair: $names_from vs $names_to")
    return [vals[findfirst(==(n), names_from)] for n in names_to]
end

@testset "iSDM unique-variance paired twins (P1)" begin

@test RVU["gllvmtmb_sha"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

for name in ISDM_PSI_CASES
    @testset "$name" begin
        r = RVU["cases"][name]; j = JEU["julia"][name]
        c = isdm_psi_case(name)
        dat = read_isdm_csv(c.csv)
        @test bytes2hex(open(sha256, isdm_fixture_path(c.csv))) == r["fixture_sha256"]
        @test length(dat.value) == r["fixture_rows"]
        tab = isdm_table(c.formula, dat; family = c.family)
        p = length(tab.trait_levels); K = Int(r["K"])
        @test tab.unique && tab.K == K == 1

        # R's parameter vector is [b_fix; theta_rr_B; theta_diag_B], every
        # trait's theta_diag_B free (no diag_B_skip), with z_B and s_B integrated.
        pX = length(r["b_fix_names"])
        @test r["par_names"] == vcat(fill("b_fix", pX), fill("theta_rr_B", p * K - K * (K - 1) ÷ 2),
                                     fill("theta_diag_B", p))
        @test r["random_names"] == ["z_B", "s_B"]
        @test all(==(0), r["diag_B_skip"])
        @test exp.(Float64.(r["theta_diag_B"])) ≈ Float64.(r["sd_B"]) rtol = 1e-14

        rb = _by_name(r["b_fix_names"], Float64.(r["b_fix"]), tab.X_names)
        RΛ = reshape(Float64.(r["Lambda_B_colmajor"]), p, K)
        @test GLLVModels.unpack_lambda(Float64.(r["theta_rr_B"]), p, K) ≈ RΛ rtol = 1e-14
        Rθd = Float64.(r["theta_diag_B"])

        # Cross-objective both ways: Julia's marginal at R's optimum equals R's
        # objective there, and R's objective at Julia's optimum equals Julia's.
        @test abs(isdm_marginal_loglik_laplace(tab, RΛ, rb; theta_diag_B = Rθd) - r["loglik"]) <= 1e-6
        jb = _by_name(j["b_fix_names"], Float64.(j["b_fix"]), tab.X_names)
        JΛ = GLLVModels.unpack_lambda(Float64.(j["theta_rr_B"]), p, K)
        jll = isdm_marginal_loglik_laplace(tab, JΛ, jb; theta_diag_B = Float64.(j["theta_diag_B"]))
        @test abs(jll - j["loglik"]) <= 1e-9            # the recorded Julia estimate reproduces
        @test abs(RVU["xobj"][name] - jll) <= 1e-6

        # A fresh Julia fit reproduces the recorded estimate and converges; R converged.
        ft = fit_isdm_gllvm(tab)
        @test ft.converged && all(ft.cell_converged)
        @test r["convergence"] == 0 && r["polished_convergence"] == 0
        @test abs(ft.loglik - j["loglik"]) <= 1e-8
        @test ft.loglik >= r["loglik"] - 1e-6            # same objective: Julia is not below R
        @test abs(ft.loglik - r["polished_loglik"]) <= 1e-6

        if name == "psi4"
            # Estimates against R's polished optimum (nlminb restarted from the
            # door's optimum at rel.tol 1e-14, certified code 0, max|gradient|
            # 5.4e-6): b_fix by name at rel 1e-4 (measured 1.3e-6), Λ Λ' and the
            # unique SDs at abs 1e-6 (measured 2.3e-7 and 4.2e-7).
            pb = _by_name(r["b_fix_names"], Float64.(r["polished_b_fix"]), tab.X_names)
            PΛ = GLLVModels.unpack_lambda(Float64.(r["polished_theta_rr_B"]), p, K)
            pθd = Float64.(r["polished_theta_diag_B"])
            @test maximum(_relgap.(ft.b_fix, pb)) <= 1e-4
            @test maximum(abs.(ft.Λ * ft.Λ' .- PΛ * PΛ')) <= 1e-6
            @test maximum(abs.(exp.(ft.theta_diag_B) .- exp.(pθd))) <= 1e-6
            # R's door optimum stopped at "relative convergence (4)" with
            # max|gradient| 6.2e-4, so it is compared on loose absolute bounds
            # (measured max gaps: b_fix 2.9e-5, Λ Λ' 1.7e-5, sd_B 3.2e-5, eta
            # 3.5e-5) so that a regression fails.
            @test maximum(abs.(ft.b_fix .- rb)) <= 1e-4
            LLr = reshape(Float64.(r["LLt_colmajor"]), p, p)
            @test maximum(abs.(ft.Λ * ft.Λ' .- LLr)) <= 1e-4
            @test maximum(abs.(exp.(ft.theta_diag_B) .- Float64.(r["sd_B"]))) <= 1e-4
            @test maximum(abs.(ft.eta .- Float64.(r["eta"]))) <= 1e-4
            # interior: the unique SDs are identified here
            @test all(0.2 .< exp.(ft.theta_diag_B) .< 1.0)

            # Predict, against R's door optimum: in-sample (the unique effect s_B
            # is part of eta), response, fixed-only, training rows with the
            # offset zeroed (re-adds Λ z and s_B), and rows moved to an unseen
            # unit (fixed-only there, as in R). Link scale at abs 1e-4 (measured
            # max 3.5e-5); response scale at rel 1e-4 (the response is exp(eta)
            # up to ~10 on count rows; measured abs max 1.6e-4).
            @test r["predict_columns"] == ["cell_id", "species", "trait", "isdm_source", "est"]
            nd0 = merge(dat, (log_support = zeros(length(dat.value)),))
            ndu = merge(dat, (cell_id = [i <= 12 ? "c_new" : v for (i, v) in enumerate(dat.cell_id)],))
            for (got, key) in ((predict(ft).est, "predict_link"),
                               (predict(ft; re_form = :zero).est, "predict_link_zero_re"),
                               (predict(ft; newdata = nd0).est, "predict_newdata_offset0_link"),
                               (predict(ft; newdata = ndu).est, "predict_newdata_unseen_link"))
                @test maximum(abs.(got .- Float64.(r[key]))) <= 1e-4
            end
            for (got, key) in ((predict(ft; type = :response).est, "predict_response"),
                               (predict(ft; newdata = nd0, type = :response).est,
                                "predict_newdata_offset0_response"))
                @test maximum(_relgap.(got, Float64.(r[key]))) <= 1e-4
            end
        else
            # Documented, not asserted: theta_diag_B runs toward the boundary.
            # Measured: R sd_B = [4.05e-5, 2.28e-5] (predict_default) and
            # [6.19e-6, 3.94e-5] (ms3_default); Julia sd = O(1e-7). Only Λ Λ' and
            # b_fix, which do not depend on where the vanishing SDs stop, are
            # compared, on the absolute scale of the p = 2 twins (measured max
            # 4.5e-8 and 8.3e-9).
            @test maximum(abs.(ft.b_fix .- rb)) <= 1e-5
            LLr = reshape(Float64.(r["LLt_colmajor"]), p, p)
            @test maximum(abs.(ft.Λ * ft.Λ' .- LLr)) <= 1e-5
        end
    end
end

end
