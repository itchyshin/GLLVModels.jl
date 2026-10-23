# Diagnostic follow-up for beta.jl: characterise d05 (stall, no boundary) and d03 (high-phi mode).
# Reuses beta.jl's simulate/replicate by including it with main() suppressed.
src = read("/tmp/claude-503/sibling-screen/beta.jl", String)
src = replace(src, r"\nmain\(\)\s*$" => "\n")
include_string(Main, src)
using Printf
function fdgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for j in eachindex(θ)
        e = zeros(length(θ)); e[j] = h
        g[j] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end
for d in (3, 5)
    Y, φtrue, _ = simulate(d)
    R = replicate(Y)
    fit = fit_beta_gllvm_grouped(Y; K = K, group = collect(1:p))
    θ̂ = theta_of(fit)
    θa = copy(R.θ0); θa[R.first_logd:end] .= 0.0
    ra = Optim.optimize(R.negll, θa, R.ls, R.opts; autodiff = :finite)
    θA = Optim.minimizer(ra)
    println("\n[d$(lpad(d,2,'0'))] phi_true=$φtrue")
    for (lab, θ) in (("fitter", θ̂), ("alt_a", θA))
        g = fdgrad(R.negll, θ)
        Λ = GM.unpack_lambda(θ[(p+1):(p+R.rr)], p, K)
        @printf("  %-6s loglik=%.6f  max|grad|=%.3e  phi=[%s]\n", lab, -R.negll(θ), maximum(abs, g),
                join([@sprintf("%.4g", x) for x in exp.(θ[R.first_logd:end])], ","))
        println("         row-norms |Lambda_t|=", round.(sqrt.(vec(sum(Λ.^2; dims=2))); digits=3),
                "  beta=", round.(θ[1:p]; digits=3))
    end
    # re-run the fitter's optimizer from its own returned point (does it move?)
    r0 = Optim.optimize(R.negll, θ̂, R.ls, R.opts; autodiff = :finite)
    @printf("  restart from fitter point unchanged: loglik=%.6f iters=%d\n", -Optim.minimum(r0), Optim.iterations(r0))
    # (c) analog of (b) for the outlying (non-boundary) group: set the group with max |log phi - median| to 0
    lφ = θ̂[R.first_logd:end]; j = argmax(abs.(lφ .- sort(lφ)[3]))
    θc = copy(θ̂); θc[R.first_logd - 1 + j] = 0.0
    rc = Optim.optimize(R.negll, θc, R.ls, R.opts; autodiff = :finite)
    @printf("  (c) fitter point with outlying group %d log phi=0: loglik=%.6f phi=[%s]\n", j, -Optim.minimum(rc),
            join([@sprintf("%.4g", x) for x in exp.(Optim.minimizer(rc)[R.first_logd:end])], ","))
    # 1-D slice in log phi_j from the fitter point (other params held at fitter values)
    print("  slice loglik vs log phi_$j (others fixed at fitter): ")
    for v in (0.0, 0.5, 1.0, 2.0, 3.0, lφ[j], 5.0, 7.0)
        θs = copy(θ̂); θs[R.first_logd - 1 + j] = v
        @printf("[%.2f: %.3f] ", v, -R.negll(θs))
    end
    println()
end
