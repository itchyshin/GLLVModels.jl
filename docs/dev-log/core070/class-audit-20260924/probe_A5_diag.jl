using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff
const G = GLLVModels
# instrumented copy of families/laplace.jl:_laplace_mode (lines 102-184), reporting exit reason
function mode_diag(family, y, n, Λ, β, link; maxiter=100, tol=1e-9)
    K = size(Λ, 2); z = zeros(K); restarted = false; reason = :maxiter; it = 0; last_step = NaN; lastΔ = NaN
    for i in 1:maxiter
        it = i
        η = G._clamp_eta.(β .+ Λ * z); μ = G._clamp_mu.(Ref(family), G.linkinv.(Ref(link), η)); me = G.mu_eta.(Ref(link), η)
        s = G._glm_score.(Ref(family), μ, n, me, y); W = G._glm_weight.(Ref(family), μ, n, me)
        A = Symmetric(Λ' * (W .* Λ) + I); g = Λ' * s .- z; Δ = G._safe_solve(A, g)
        if Δ === nothing || !all(isfinite, Δ)
            if !restarted; z = zeros(K); restarted = true; continue; end
            reason = :nonfinite_step; break
        end
        step_taken = 1.0
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = G._laplace_mode_logpost(family, y, n, Λ, β, link, z)
            if isfinite(q0)
                accepted = false; step = 1.0
                for _h in 1:30
                    zt = z .+ step .* Δ; q1 = G._laplace_mode_logpost(family, y, n, Λ, β, link, zt)
                    if isfinite(q1) && q1 >= q0; z = zt; step_taken = step; accepted = true; break; end
                    step *= 0.5
                end
                accepted || (reason = :backtrack_failed; break)
            else
                z = z .+ Δ
            end
        end
        last_step = step_taken; lastΔ = maximum(abs, Δ)
        if step_taken * maximum(abs, Δ) < tol
            reason = step_taken < 1 ? :tiny_backtracked_step_tol_exit : :converged
            break
        end
    end
    return z, reason, it, last_step, lastΔ
end
rng = MersenneTwister(99)
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
for (fam, link, nm, draw) in ((NegativeBinomial(2.0, 0.5), LogLink(), "NB2", μ -> rand(rng, NegativeBinomial(2.0, 2.0/(2.0+μ)))),
                              (Gamma(2.0, 1.0), LogLink(), "Gamma", μ -> rand(rng, Gamma(2.0, μ/2))))
    counts = Dict{Symbol,Int}(); ex = Dict{Symbol,Any}()
    for _ in 1:1500
        p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc()
        Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
        y = [draw(exp(clamp(β[t] + dot(Λ[t, :], zt), -5, 6))) for t in 1:p]
        Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K)); n = ones(Int, p)
        z, reason, it, ls, lΔ = mode_diag(fam, y, n, Λe, β, link)
        zp = G._laplace_mode(fam, y, n, Λe, β, link)
        @assert isapprox(z, zp; atol=1e-12) || (reason == :maxiter)  # instrumented copy must match
        q = zz -> sum(G._glm_logpdf(fam, G._clamp_mu(fam, G.linkinv(link, G._clamp_eta(β[t] + dot(Λe[t, :], zz)))), 1, y[t]) for t in 1:p) - 0.5 * dot(zz, zz)
        gk = maximum(abs, ForwardDiff.gradient(q, z))
        if gk > 1e-4
            counts[reason] = get(counts, reason, 0) + 1
            haskey(ex, reason) || (ex[reason] = (it=it, last_step=ls, lastΔ=lΔ, gk=gk, max_abs_eta=maximum(abs, β .+ Λe*z)))
        end
    end
    println("$nm backtracked generic kernel, non-mode exits by reason: ", counts)
    for (k, v) in ex; println("   ", k, " => ", v); end
end
