using GLLVModels, Test, SHA, TOML

# fit_nb_gllvm_grouped (the default NB2 route, per-trait dispersion) could stall with a
# trait's r pushed to the Poisson boundary, below the optimum (#477). On this stored draw
# (tools/nb2_parity_data_draw.jl 46 1.0) it used to stop at logLik -836.924 with two traits
# at r ~ 1e10 and 1e55. The optimum, which Julia's own objective confirms at gllvmTMB's
# answer (frozen b4d5fee64), is -835.3693 with every r finite (R: 1.09, 1.12, 2.74, 1.62, 0.81).

@testset "NB2 per-trait dispersion does not stall at the Poisson boundary (#477)" begin
    d = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb2_finite_dispersion_data.toml"))
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]

    fit = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
    @test fit isa NBGroupedFit
    @test !any(fit.dispersion_boundary)
    @test all(r -> 0.5 < r < 5.0, fit.r_group)
    @test fit.loglik ≈ -835.369277 atol = 1e-5
end
