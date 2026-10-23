using GLLVModels, Test, SHA, TOML, Random, LinearAlgebra, Distributions

# fit_gamma_gllvm_grouped could report converged = true after 2 iterations at a point 43
# log-likelihood units below the optimum (#479). The per-site mode search in
# `_gamma_grouped_loglik_site` took full Fisher-scoring steps with no damping. At the
# fitter's warm start it diverged on one site (z about 1e11) and still returned a FINITE
# site log-likelihood (about -1.4e23), so the 1e12 sentinel in the objective never fired
# and L-BFGS stopped on a garbage gradient. The search now halves any step that lowers the
# per-site log-posterior, falls back to a damped Newton search on the observed Gamma/log
# curvature when Fisher scoring does not converge, and returns -Inf for a site whose
# search still fails, so the fitter's sentinel fires.
# Data: the sibling screen's dataset 2 (seed 2, alpha 2); recipe in the fixture header.

const _G479_FIXTURE = joinpath(@__DIR__, "fixtures", "gamma_grouped_mode_search_seed2.toml")

function _g479_data()
    d = TOML.parsefile(_G479_FIXTURE)
    Y = reshape(Float64.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Y)))) == d["data_sha256"]
    return Y, d["K"]
end

# The fitter's own warm start (log row means, SVD loadings), as in fit_gamma_gllvm_grouped.
function _g479_warmstart(Y, K)
    p, n = size(Y)
    Z = log.(max.(Y, 1e-6))
    β0 = vec(sum(Z; dims = 2)) ./ n
    F = svd(Z .- β0)
    Λ0 = zeros(p, K)
    for j in 1:K
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    return β0, Λ0
end

_g479_fams(αvec) = [Gamma(float(a), 1.0) for a in αvec]

# The pre-#479 per-site kernel, kept verbatim apart from also reporting whether its
# undamped Fisher loop converged. Used only where it DID converge, to check that the fix
# leaves those values alone.
function _g479_old_site(fams, y, Λ, β; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λ)
    n = ones(Int, p)
    link = G.LogLink()
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        η  = G._clamp_eta.(β .+ Λ * z)
        μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
        me = G.mu_eta.(Ref(link), η)
        s  = G._glm_score.(fams, μ, n, me, y)
        W  = G._gamma_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link))
        Δ  = G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    η  = G._clamp_eta.(β .+ Λ * z)
    μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    W  = G._gamma_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link))
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Independent per-site Laplace value: Distributions.logpdf, analytic Gamma/log score and
# observed curvature, damped Newton to a tight tolerance. Shares no code with the kernel.
function _g479_indep_site(y, Λ, β, αvec)
    h(z) = sum(logpdf(Gamma(αvec[t], exp(β[t] + dot(Λ[t, :], z)) / αvec[t]), y[t])
               for t in eachindex(y)) - 0.5 * dot(z, z)
    z = zeros(size(Λ, 2))
    for _ in 1:200
        μ = exp.(β .+ Λ * z)
        g = Λ' * (αvec .* (y ./ μ .- 1)) .- z
        H = Λ' * ((αvec .* y ./ μ) .* Λ) + I
        d = H \ g
        step = 1.0
        while h(z .+ step .* d) < h(z) && step > 1e-10
            step /= 2
        end
        z = z .+ step .* d
        maximum(abs, step .* d) < 1e-13 && break
    end
    μ = exp.(β .+ Λ * z)
    H = Λ' * ((αvec .* y ./ μ) .* Λ) + I
    return h(z) - 0.5 * logdet(H)
end

@testset "Gamma grouped mode search: damped, and fails loudly (#479)" begin
    Y, K = _g479_data()
    p, n = size(Y)
    β0, Λ0 = _g479_warmstart(Y, K)
    link = GLLVModels.LogLink()
    ones_p = ones(Int, p)

    @testset "warm-start site 7: the diverging site now gets its true Laplace value" begin
        # Before: undamped Fisher scoring cycled out to z about 1e11 and returned -1.4169e23.
        fams = _g479_fams(fill(2.0, p))
        y = Y[:, 7]
        v = GLLVModels._gamma_grouped_loglik_site(fams, y, ones_p, Λ0, β0, link)
        @test isfinite(v)
        @test abs(v - _g479_indep_site(y, Λ0, β0, fill(2.0, p))) < 1e-8
        # every other warm-start site agrees with the independent evaluator too
        worst = maximum(abs(GLLVModels._gamma_grouped_loglik_site(fams, Y[:, i], ones_p, Λ0, β0, link) -
                            _g479_indep_site(Y[:, i], Λ0, β0, fill(2.0, p))) for i in 1:n)
        @test worst < 1e-8
    end

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        fams = _g479_fams(fill(2.0, p))
        # Two iterations cannot reach tol = 1e-9 from z = 0 at any of these sites.
        for i in (7, 10, 42)
            @test GLLVModels._gamma_grouped_loglik_site(fams, Y[:, i], ones_p, Λ0, β0, link;
                                                        maxiter = 2) == -Inf
        end
        @test GLLVModels.gamma_grouped_marginal_loglik_laplace(Y, Λ0, β0, fill(2.0, p);
                                                               maxiter = 2) == -Inf
        # ... so the fitter's sentinel fires and the fit does not claim convergence.
        fit = GLLVModels.fit_gamma_gllvm_grouped(Y; K = K, newton_maxiter = 2)
        @test !fit.converged
        @test fit.loglik == -Inf
    end

    @testset "where the old loop converged, the value is unchanged (to 1e-8)" begin
        # Three parameter points on the fixture (the warm start, the warm start with
        # log α = 0, a per-trait α) plus a well-behaved simulated dataset at its truth and
        # at perturbed points. At every site where the pre-#479 undamped loop converged,
        # the new kernel must return the same value.
        Random.seed!(479)
        p2, n2 = 6, 60
        Λt = 0.3 .* randn(p2, K)
        βt = 0.5 .* randn(p2)
        αt = [1.5, 2.0, 3.0, 5.0, 8.0, 20.0]
        Zt = randn(K, n2)
        Y2 = [rand(Gamma(αt[t], exp(βt[t] + dot(Λt[t, :], Zt[:, i])) / αt[t])) for t in 1:p2, i in 1:n2]
        cases = [(Y, Λ0, β0, fill(2.0, p)), (Y, Λ0, β0, fill(1.0, p)),
                 (Y, Λ0, β0, [1.7, 1.6, 3.5, 2.1, 1.7]),
                 (Y2, Λt, βt, αt), (Y2, 1.5 .* Λt, βt .+ 0.2, αt), (Y2, Λt, βt, 0.5 .* αt)]
        n_compared = 0
        worst = 0.0
        for (Yc, Λ, β, αvec) in cases
            fams = _g479_fams(αvec)
            for i in axes(Yc, 2)
                old, conv = _g479_old_site(fams, Yc[:, i], Λ, β)
                conv || continue
                new = GLLVModels._gamma_grouped_loglik_site(fams, Yc[:, i], ones(Int, size(Yc, 1)),
                                                            Λ, β, link)
                worst = max(worst, abs(new - old))
                n_compared += 1
            end
        end
        @test n_compared >= 400          # measured: 418 of the 420 site evaluations
        @test worst < 1e-8
    end

    @testset "public fit reaches the optimum instead of stalling at iteration 2" begin
        # Before: loglik -610.224727, converged = true, iterations = 2. A fresh start with
        # log α = 0 reaches -567.232613 with every site's mode search converged.
        fit = GLLVModels.fit_gamma_gllvm_grouped(Y; K = K)
        @test fit.loglik >= -567.232613 - 1e-5
        @test fit.converged
    end

    @testset "same kernel: one shared group (the bridge's no-X Gamma route)" begin
        # bridge.jl fits Gamma without X as fit_gamma_gllvm_grouped(...; group = fill(1, p)).
        # Before: -610.109101 at iteration 2; a log α = 0 start reaches -569.700833.
        fit = GLLVModels.fit_gamma_gllvm_grouped(Y; K = K, group = ones(Int, p))
        @test fit.loglik >= -569.700833 - 1e-5
        @test fit.converged
    end

    @testset "same kernel: covariate route (fit_gamma_gllvm_grouped_cov)" begin
        # With X = 0 the covariate has no effect, so the optimum is the no-X one.
        # Before: -610.224727 at iteration 2, like the no-X fitter.
        fit = GLLVModels.fit_gamma_gllvm_grouped_cov(Y; X = zeros(p, n, 1), K = K)
        @test fit.loglik >= -567.232613 - 1e-5
        @test fit.converged
    end
end
