# Class A kernel probe: does the site kernel return a FINITE value at a z that is not the mode?
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
# log-posterior gradient at z for a scalar family with offset eta0 = beta+off
function resid(fam, y, n, Λ, η0, link, z)
    η = G._clamp_eta.(η0 .+ Λ * z)
    μ = G._clamp_mu.(Ref(fam), G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    s = G._glm_score.(Ref(fam), μ, n, me, y)
    return maximum(abs, Λ' * s .- z)
end
function residv(fams, y, n, Λ, η0, link, z)
    η = G._clamp_eta.(η0 .+ Λ * z)
    μ = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    s = G._glm_score.(fams, μ, n, me, y)
    return maximum(abs, Λ' * s .- z)
end
rng = MersenneTwister(20260924)
# ---- 1. covariates.jl _laplace_mode_off / _laplace_site_off (fit_gllvm_cov, row_eff=:fixed) vs backtracked generic
for (fam, link, name) in ((Poisson(), LogLink(), "Poisson/log"), (Binomial(), LogitLink(), "Binomial/logit"))
    nbad = 0; nbad_finite = 0; maxdiff = 0.0; ex = nothing; ntr = 4000
    for trial in 1:ntr
        p = rand(rng, 4:12); K = rand(rng, 1:3)
        sc = rand(rng, (0.5, 1.0, 2.0, 3.0))
        Λ = sc .* randn(rng, p, K); β = randn(rng, p) .* 1.5
        n = ones(Int, p)
        y = fam isa Poisson ? [rand(rng, Poisson(exp(clamp(β[t] + sc*2*randn(rng), -5, 6)))) for t in 1:p] :
                              [rand(rng, Bool) ? 1 : 0 for t in 1:p]
        z_off = G._laplace_mode_off(fam, y, n, Λ, β, link)
        r_off = resid(fam, y, n, Λ, β, link, z_off)
        v_off = G._laplace_site_off(fam, y, n, Λ, β, link)
        z_ref = G._laplace_mode(fam, y, n, Λ, β, link)
        r_ref = resid(fam, y, n, Λ, β, link, z_ref)
        v_ref = G.laplace_loglik_site(fam, y, n, Λ, β, link)
        if r_off > 1e-4 && r_ref < 1e-6
            nbad += 1
            if isfinite(v_off)
                nbad_finite += 1
                d = abs(v_off - v_ref)
                if d > maxdiff
                    maxdiff = d; ex = (p=p, K=K, sc=sc, r_off=r_off, r_ref=r_ref, v_off=v_off, v_ref=v_ref, z_off=z_off, z_ref=z_ref)
                end
            end
        end
    end
    println("[covariates _laplace_mode_off] $name: $nbad/$ntr non-mode returns (ref converged), $nbad_finite finite; max |Δloglik| = $maxdiff")
    ex === nothing || println("   worst: ", ex)
end
# ---- 2. NB1 grouped kernel (_nb1_grouped_loglik_site) vs backtracked _grouped_laplace_mode
let nbad = 0, nbad_finite = 0, maxdiff = 0.0, ex = nothing, ntr = 3000
    for trial in 1:ntr
        p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rand(rng, (0.5, 1.0, 2.0, 3.0))
        Λ = sc .* randn(rng, p, K); β = randn(rng, p) .* 1.5
        φ = exp.(randn(rng, p) .* 1.5)
        fams = [G.NB1(φ[t]) for t in 1:p]; n = ones(Int, p)
        y = [rand(rng, NegativeBinomial(max(exp(clamp(β[t] + sc*2*randn(rng), -5, 6)),1e-6)/φ[t], 1/(1+φ[t]))) for t in 1:p]
        # reproduce the kernel's own z by re-running its loop logic: use the site value and compare to the
        # value at the reference mode. Recompute the kernel z via a copy of the loop.
        z = zeros(K)
        for _ in 1:100
            η = G._clamp_eta.(β .+ Λ * z); μ = G._clamp_mu.(fams, G.linkinv.(Ref(LogLink()), η)); me = G.mu_eta.(Ref(LogLink()), η)
            s = G._glm_score.(fams, μ, n, me, y); W = G._glm_weight.(fams, μ, n, me)
            A = Symmetric(Λ' * (W .* Λ) + I); Δ = G._safe_solve(A, Λ' * s .- z)
            (Δ === nothing || !all(isfinite, Δ)) && break
            z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
        end
        r_k = residv(fams, y, n, Λ, β, LogLink(), z)
        v_k = G._nb1_grouped_loglik_site(fams, y, n, Λ, β, LogLink(); hessian = :fisher)
        z_ref = G._grouped_laplace_mode(fams, y, n, Λ, β, LogLink())
        r_ref = residv(fams, y, n, Λ, β, LogLink(), z_ref)
        if r_k > 1e-4 && r_ref < 1e-6
            nbad += 1
            if isfinite(v_k)
                nbad_finite += 1
                # value at ref mode with the same (fisher) log-det
                η = G._clamp_eta.(β .+ Λ * z_ref); μ = G._clamp_mu.(fams, G.linkinv.(Ref(LogLink()), η)); me = G.mu_eta.(Ref(LogLink()), η)
                W = G._glm_weight.(fams, μ, n, me); A = Symmetric(Λ' * (W .* Λ) + I)
                v_ref = sum(G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p) - 0.5*dot(z_ref,z_ref) - 0.5*logdet(A)
                d = abs(v_k - v_ref)
                d > maxdiff && (maxdiff = d; ex = (p=p, K=K, sc=sc, r_k=r_k, v_k=v_k, v_ref=v_ref))
            end
        end
    end
    println("[NB1 grouped _nb1_grouped_loglik_site] $nbad/$ntr non-mode returns (ref converged), $nbad_finite finite; max |Δloglik| = $maxdiff")
    ex === nothing || println("   worst: ", ex)
end
