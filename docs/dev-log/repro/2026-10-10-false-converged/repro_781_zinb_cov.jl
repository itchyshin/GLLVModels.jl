# Repro for #781: fit_zinb_gllvm_cov(iterations=0) reports converged=true
# Recorded against origin/main at e6cfcd4ee7b5bcf1ba4ff2f0b6413bf80f31efe3
# (issue pin was 824d22a4; behaviour confirmed on current main).
#
# Usage:
#   julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_781_zinb_cov.jl

using GLLVModels, Random, Distributions

println("commit = ", readchomp(`git rev-parse HEAD`))
Random.seed!(11)
Y = Float64.(rand(Poisson(2), 4, 60)); X = randn(4, 60, 1)

f = fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0)
println("warmstart: conv=", f.converged, "  iters=", f.iterations,
        "  loglik=", f.loglik)

f2 = fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0,
        θ_init = vcat(fill(30.0, 4), 0.0, fill(30.0, 4), 0.0, zeros(4), 0.0))
println("absurd θ_init: conv=", f2.converged, "  loglik=", f2.loglik)

print("NaN θ_init: ")
try
    f3 = fit_zinb_gllvm_cov(Y; X = X, K = 1, iterations = 0,
            θ_init = vcat(fill(NaN, 4), 0.0, fill(30.0, 4), 0.0, zeros(4), 0.0))
    println("conv=", f3.converged, "  loglik=", f3.loglik)
catch e
    println(typeof(e), " ", sprint(showerror, e))
end
