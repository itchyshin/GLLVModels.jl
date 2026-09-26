using GLLVModels, Optim, Random, Distributions, LinearAlgebra, Printf
include(joinpath(@__DIR__, "probe_common.jl"))
for seed in (1, 3, 7, 8, 2)
    Y = sim(:gamma, seed; p = 8, n = 100)
    run("gamma-analytic s$seed", () -> fit_gamma_gllvm(Y; K = 2))
    fa = fit_gamma_gllvm(Y; K = 2)
    run("gamma-FD s$seed", () -> fit_gamma_gllvm(Y; K = 2, gradient = :finite))
    ff = fit_gamma_gllvm(Y; K = 2, gradient = :finite)
    run("gamma-FD-warm s$seed", () -> fit_gamma_gllvm(Y; K = 2, gradient = :finite, β_init = ff.β, Λ_init = ff.Λ, α_init = ff.α))
    run("gamma-FD-warm-from-analytic s$seed", () -> fit_gamma_gllvm(Y; K = 2, gradient = :finite, β_init = fa.β, Λ_init = fa.Λ, α_init = fa.α))
    @printf("  s%d: loglik analytic=%.4f FD=%.4f diff=%.4f\n", seed, fa.loglik, ff.loglik, fa.loglik - ff.loglik)
end
