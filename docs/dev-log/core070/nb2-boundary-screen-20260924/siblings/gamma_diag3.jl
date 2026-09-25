using GLLVModels, Distributions, Random, LinearAlgebra, Printf
const GM = GLLVModels
src = read("/tmp/claude-503/sibling-screen/gamma.jl", String)
cut = findfirst("const TOL_STALL", src)
eval(Meta.parseall(src[1:first(cut) - 1]))
Y = simulate(2, 2.0); prob = build_problem(Y; K = K, group = collect(1:p))
θ = prob.θ0; β = θ[1:p]; Λ = GM.unpack_lambda(θ[p+1:p+prob.rr], p, K)
y = Y[:, 7]; link = GM.LogLink()
for α in (2.0, 1.0)
    fams = [Gamma(α, 1.0) for t in 1:p]
    z = zeros(K); println("α = $α, site 7 Fisher-scoring mode search (replica of _gamma_grouped_loglik_site loop):")
    for it in 1:100
        η  = GM._clamp_eta.(β .+ Λ * z)
        μ  = GM._clamp_mu.(fams, GM.linkinv.(Ref(link), η))
        me = GM.mu_eta.(Ref(link), η)
        s  = GM._glm_score.(fams, μ, ones(Int, p), me, y)
        W  = GM._gamma_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link))
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = GM._safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && (println("  break: nonfinite Δ at it $it"); break)
        z = z .+ Δ
        (it <= 8 || it % 20 == 0 || maximum(abs, Δ) < 1e-9) && @printf("  it %3d z=%s |Δ|=%.3e\n", it, string(round.(z; sigdigits=4)), maximum(abs, Δ))
        maximum(abs, Δ) < 1e-9 && break
    end
end
