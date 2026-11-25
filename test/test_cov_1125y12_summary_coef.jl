# Lane cursor-cloud-1125y12 — summary_table.jl
using GLLVModels, Test, Random

const _ST = GLLVModels

@testset "cov 1125y12 coef_table and GllvmCoefTable show" begin
    rng = MersenneTwister(11251205)
    Y = rand(rng, 0:8, 3, 12)

    @testset "_coef_table_from_ci" begin
        ci = (term = ["beta[1]", "beta[2]", "bad"],
              estimate = [0.5, -0.2, 1.0],
              se = [0.1, 0.05, NaN],
              lower = [0.3, -0.3, NaN],
              upper = [0.7, -0.1, NaN])
        ct = _ST._coef_table_from_ci(ci)
        @test ct isa GllvmCoefTable
        @test ct.z[1] ≈ 5.0 rtol = 1e-12
        @test ct.z[2] ≈ -4.0 rtol = 1e-12
        @test isnan(ct.z[3]) && isnan(ct.pvalue[3])
        @test 0 < ct.pvalue[1] < 1e-4
    end

    @testset "coef_table from Poisson fit" begin
        fit = fit_poisson_gllvm(Y; K = 1, iterations = 80, g_tol = 1e-4)
        ct = coef_table(fit, Y)
        @test length(ct.term) == length(ct.estimate)
        @test all(isfinite, ct.estimate)
        shown = sprint(show, "text/plain", ct)
        @test occursin("GllvmCoefTable", shown)
        @test occursin("term", shown)
        ct90 = coef_table(fit, Y; level = 0.90, parm = "beta")
        @test all(ct90.lower .< ct90.upper)
    end
end