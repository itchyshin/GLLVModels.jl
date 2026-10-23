# Diagnostic for dataset 2 of gamma.jl: why did the public fitter stop after 2
# iterations with converged = true, 43 log-lik units below a fresh start?
include_harness = false
using GLLVModels, Distributions, Random, LinearAlgebra, Printf, Logging
const GM = GLLVModels
const Optim = GM.Optim
# pull build_problem / simulate definitions without re-running the loop
src = read("/tmp/claude-503/sibling-screen/gamma.jl", String)
cut = findfirst("const TOL_STALL", src)
eval(Meta.parseall(src[1:first(cut) - 1]))

Y = simulate(2, 2.0)
prob = build_problem(Y; K = K, group = collect(1:p))
res = prob.run(prob.θ0)
println(res)
fl = propertynames(res)
println("fields: ", fl)
for f in (:x_converged, :f_converged, :g_converged, :g_residual, :f_increased, :ls_success,
          :f_calls, :g_calls, :iterations)
    f in fl && println(rpad(string(f), 14), getfield(res, f))
end
θ̂ = Optim.minimizer(res)
f0 = prob.negll(prob.θ0); f̂ = prob.negll(θ̂)
@printf("negll(θ0) = %.6f   negll(θ̂) = %.6f   |θ̂-θ0|∞ = %.3e\n", f0, f̂, maximum(abs, θ̂ .- prob.θ0))
# central finite-difference gradient at θ0 and θ̂
function fdgrad(f, θ; h = 1e-6)
    g = similar(θ)
    for j in eachindex(θ)
        e = zeros(length(θ)); e[j] = h
        g[j] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end
g0 = fdgrad(prob.negll, prob.θ0); ĝ = fdgrad(prob.negll, θ̂)
@printf("|grad(θ0)|∞ = %.3e   |grad(θ̂)|∞ = %.3e\n", maximum(abs, g0), maximum(abs, ĝ))
println("grad(θ̂) = ", round.(ĝ; sigdigits = 3))
# is the objective failing (1e12 sentinel) along the descent direction?
println("negll along -grad(θ̂)/|grad| at step sizes:")
d = -ĝ ./ norm(ĝ)
for s in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0)
    @printf("  s=%-6g  negll=%.6f\n", s, prob.negll(θ̂ .+ s .* d))
end
# trace the first iterations
res_tr = Optim.optimize(prob.negll, prob.θ0,
    Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
    Optim.Options(g_tol = 1e-5, iterations = 500, store_trace = true, extended_trace = true);
    autodiff = :finite)
for st in Optim.trace(res_tr)
    @printf("iter %d  f=%.6f  |g|=%.3e\n", st.iteration, st.value, st.g_norm)
end
