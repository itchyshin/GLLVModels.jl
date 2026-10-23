using GLLVModels, Distributions, Random, LinearAlgebra, Printf, Logging
const GM = GLLVModels
const Optim = GM.Optim
src = read("/tmp/claude-503/sibling-screen/gamma.jl", String)
cut = findfirst("const TOL_STALL", src)
eval(Meta.parseall(src[1:first(cut) - 1]))
for ds in 1:10
    α = ds <= 5 ? 2.0 : 200.0
    Y = simulate(ds, α)
    prob = build_problem(Y; K = K, group = collect(1:p))
    θa = copy(prob.θ0); θa[prob.first_log:end] .= 0.0
    Λ0 = GM.unpack_lambda(prob.θ0[p+1:p+prob.rr], p, K)
    @printf("ds %2d  min(Y)=%.3e  max|Λ0|=%.3f  negll(θ0)=%.4e  negll(θ0 with logα=0)=%.4e\n",
            ds, minimum(Y), maximum(abs, Λ0), prob.negll(prob.θ0), prob.negll(θa))
end
# per-site breakdown for ds 2 at θ0
Y = simulate(2, 2.0); prob = build_problem(Y; K = K, group = collect(1:p))
θ = prob.θ0; β = θ[1:p]; Λ = GM.unpack_lambda(θ[p+1:p+prob.rr], p, K)
println("ds2 β0 = ", round.(β; sigdigits=4)); println("ds2 Λ0 = ", round.(Λ; sigdigits=4))
fams = [Gamma(2.0, 1.0) for t in 1:p]
site = [GM._gamma_grouped_loglik_site(fams, Y[:, i], ones(Int, p), Λ, β, GM.LogLink();
         hessian = :observed) for i in 1:size(Y, 2)]
o = sortperm(site)[1:3]
for i in o
    @printf("site %d  loglik=%.4e  y=%s\n", i, site[i], string(round.(Y[:, i]; sigdigits=3)))
end
