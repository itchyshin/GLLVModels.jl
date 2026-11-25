# Lane cursor-cloud-1125y11 — kernel_latent.jl admission branches
using GLLVModels, Test, LinearAlgebra, Random

@testset "cov 1125y11 kernel latent guards" begin
    rng = MersenneTwister(1125113)
    p, n = 3, 9
    Y = randn(rng, p, n)
    groups = repeat(1:3; inner = 3)
    K = randn(rng, 3, 3); K = K * K' + I

    @testset "rank routing dep versus latent" begin
        f1 = fit_kernel_latent_gllvm(Y, K, groups, 1; name = :k1, iterations = 30, g_tol = 1e-3)
        f3 = fit_kernel_latent_gllvm(Y, K, groups, p; name = :kp, iterations = 30, g_tol = 1e-3)
        @test f1.sources[1].mode === :latent
        @test f1.sources[1].rank == 1
        @test f3.sources[1].mode === :dep
        @test isfinite(f1.loglik) && isfinite(f3.loglik)
    end

    for (msg, call) in [
        ("missing d", () -> fit_kernel_latent_gllvm(Y, K, groups)),
        ("missing groups", () -> fit_kernel_latent_gllvm(Y, K)),
        ("non-square K", () -> fit_kernel_latent_gllvm(Y, K[1:2, :], groups, 1)),
        ("short groups", () -> fit_kernel_latent_gllvm(Y, K, groups[1:(end - 1)], 1)),
        ("bad group index", () -> fit_kernel_latent_gllvm(Y, K, [1, 2, 9], 1)),
        ("d=0", () -> fit_kernel_latent_gllvm(Y, K, groups, 0)),
        ("d>p", () -> fit_kernel_latent_gllvm(Y, K, groups, p + 1)),
        ("unique", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; unique = true)),
        ("rho", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; rho = 0.3)),
        ("K knob", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; K = 2)),
        ("Σ_phy knob", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; Σ_phy = I)),
        ("sources knob", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; sources = [])),
        ("family knob", () -> fit_kernel_latent_gllvm(Y, K, groups, 1; family = Normal())),
    ]
        @testset "$msg" begin
            @test_throws ArgumentError call()
        end
    end
end