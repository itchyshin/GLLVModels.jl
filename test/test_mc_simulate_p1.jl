# Monte-Carlo moment twins against gllvmTMB P1 (maintainer ruling 2026-10-05, vault D-319, item N4):
#   postfit/POSTFIT-SURFACE-simulate_unit_trait  and  postfit-policy/POST-SIMULATE-DEFAULT.
# The rule is stated in test/mc_simulate_helpers_p1.jl (and, identically, in test/fixtures/gen_mc_simulate_p1.R):
# per moment, |m_R - m_J| <= z * sqrt(s_R^2/B_k + s_J^2/B_k), z Bonferroni over the K moments at familywise
# alpha 0.01; and a discrimination control that must fail the same rule.

using Test, TOML
include(joinpath(@__DIR__, "mc_simulate_helpers_p1.jl"))

const MC_FX = TOML.parsefile(joinpath(@__DIR__, "fixtures", "mc_simulate_p1.toml"))

@testset "Monte-Carlo moment twins (D-319 N4, gllvmTMB P1)" begin
    B, α = MC_FX["B"], MC_FX["alpha"]

    @testset "simulate_unit_trait" begin
        fx = MC_FX["unit_trait"]
        z = fx["z"]
        @test isapprox(z, mc_z(α, fx["K"]); atol = 1e-12)
        j = mc_unit_trait_julia(fx, B)
        diff, tol = mc_rule(fx["r_mean"], fx["r_sd"], fx["r_n"], j.mean, j.sd, j.n, z)
        @test length(diff) == fx["K"]
        @test all(diff .<= tol)   # MC-RULE-ASSERT unit_trait
        alt = mc_unit_trait_julia(fx, B; psi_scale = 2.0, seed0 = 20_000)
        diff_alt, tol_alt = mc_rule(fx["r_mean"], fx["r_sd"], fx["r_n"], alt.mean, alt.sd, alt.n, z)
        @test any(diff_alt .> tol_alt)   # discrimination control: psi_B doubled must fail
    end

    @testset "simulate() default (latent scores redrawn)" begin
        fx = MC_FX["simulate_default"]
        z = fx["z"]
        @test isapprox(z, mc_z(α, fx["K"]); atol = 1e-12)
        fit, j = mc_simulate_default_julia(fx, B)
        @test isapprox(GLLVModels.loglikelihood(fit), fx["loglik"]; atol = 1e-4)   # same fitted model
        d1, t1 = mc_rule(fx["r_mean_m1"], fx["r_sd_m1"], fx["r_n_m1"], j.m1.mean, j.m1.sd, j.m1.n, z)
        d2, t2 = mc_rule(fx["r_mean_m2"], fx["r_sd_m2"], fx["r_n_m2"], j.m2.mean, j.m2.sd, j.m2.n, z)
        @test length(d1) + length(d2) == fx["K"]
        @test all(vcat(d1, d2) .<= vcat(t1, t2))   # MC-RULE-ASSERT simulate_default
        c1, u1 = mc_rule(fx["cond_mean_m1"], fx["cond_sd_m1"], fx["r_n_m1"], j.m1.mean, j.m1.sd, j.m1.n, z)
        c2, u2 = mc_rule(fx["cond_mean_m2"], fx["cond_sd_m2"], fx["r_n_m2"], j.m2.mean, j.m2.sd, j.m2.n, z)
        @test any(vcat(c1, c2) .> vcat(u1, u2))   # discrimination control: R condition_on_RE = TRUE must fail
    end
end
