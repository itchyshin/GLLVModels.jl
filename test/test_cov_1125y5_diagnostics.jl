# Lane 1125y5: diagnostics.jl compare_*, kernel separability, aliases.
using Test, GLLVModels, Random, LinearAlgebra, Statistics
const G = GLLVModels

@testset "1125y5 diagnostics compare and angles" begin
    Λa = [0.9 0.0; 0.1 0.8; 0.2 -0.3]
    Λb = [0.85 0.05; 0.15 0.75; 0.25 -0.25]
    ang = G._principal_angles(Λa, Λb)
    @test all(0 .<= ang .<= π / 2)
    cmpL = compare_loadings(Λa, Λb)
    @test haskey(cmpL, :frobenius)
    cmpF = compare_loadings(fit_gaussian_gllvm(randn(3, 20); K = 1, iterations = 5),
        fit_gaussian_gllvm(randn(3, 20); K = 1, iterations = 5))
    @test cmpF.frobenius_norm_LLt isa Real

    Random.seed!(3)
    p, n = 5, 60
    y = randn(p, n)
    f1 = fit_gaussian_gllvm(y; K = 1, iterations = 10)
    f2 = fit_gaussian_gllvm(y .+ 0.01; K = 1, iterations = 10)
    sig = compare_fits_Sigma_table(f1, f2)
    @test sig.frobenius_norm isa Float64
    @test size(sig.Sigma1) == (p, p)

    sep = diagnose_kernel_separability(f1)
    @test sep.separable === missing
    @test occursin("W-tier", sep.message)

    tbl = fit_diagnostic_table(f1; y = y)
    @test tbl isa NamedTuple
    tbl2 = G.diagnostic_table(f1; y = y)
    @test tbl2.check == tbl.check
    deprecated = G.compare_Sigma_table(f1, f2)
    @test deprecated.frobenius_norm ≈ sig.frobenius_norm

    chk = gllvmTMB_check_consistency(f1, y; n_sim = 20, seed = 1)
    @test haskey(chk, :marginal_p_value)
    auto = check_auto_residual(f1)
    @test auto isa NamedTuple
    sm = sanity_multi(f1; y = y)
    @test sm.loadings_finite isa Bool
    dg = gllvmTMB_diagnose(f1; y = y)
    @test dg isa NamedTuple
    @test check_gllvmTMB(f1; y = y) isa NamedTuple
    pc = predictive_check(f1, y; nsim = 25)
    @test pc isa NamedTuple
    @test hasproperty(pc, :p_value) || hasproperty(pc, :observed_statistic)
end