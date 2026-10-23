# Repro for #753: em_fit_phylo clamps σ_eps² to eps() and reports converged=true
# Recorded against origin/main at e6cfcd4ee7b5bcf1ba4ff2f0b6413bf80f31efe3
# (issue pin was 824d22a4; behaviour confirmed on current main).
#
# Usage:
#   julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_753_em_phylo.jl

using GLLVModels, Random

println("commit = ", readchomp(`git rev-parse HEAD`))
Random.seed!(2)
p, n = 5, 30
Σ = [0.6^abs(i - j) for i in 1:p, j in 1:p]
λ = [1.0, 0.8, -0.5, 0.3, 0.9]
y = λ * randn(n)'
fit = em_fit_phylo(y, 1, Σ; max_iter = 3000, assert_monotone = false)
println("σ_eps=", fit.σ_eps, "  conv=", fit.converged,
        "  n_iter=", fit.n_iter, "  logLik=", fit.logLik)
println("sqrt(eps())=", sqrt(eps()),
        "  on_floor=", fit.σ_eps == sqrt(eps()))
println("dof=", GLLVModels.dof(fit), "  aic=", GLLVModels.aic(fit))
if hasproperty(fit, :sigma_eps_boundary)
    println("sigma_eps_boundary=", fit.sigma_eps_boundary)
end
