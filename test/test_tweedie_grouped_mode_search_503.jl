using GLLVModels, Test, Random, Distributions, LinearAlgebra

# `_tweedie_grouped_loglik_site` ran undamped Fisher scoring and returned whatever z the
# loop held when it stopped, converged or not, so a site whose search did not converge
# still produced a finite log-likelihood (#503; the same defect #479 fixed for Gamma and
# #507 for NB1). The search (`_tweedie_grouped_mode`) now halves a large step that lowers
# the per-site log-posterior, reports whether it converged, and the site kernel returns
# -Inf for a search that did not, so the fitter's failure sentinel fires and the fit's
# `converged` flag (from `_tweedie_verdict`: Optim.converged plus the gradient, boundary
# and objective-failed checks) is false instead of true at a garbage surface.
#
# No assertion below is tied to one seed's outcome: each is a relation that holds on every
# Julia version (random streams differ across versions, #182).

const _G = GLLVModels

function _tw503_sim(seed; p = 5, n = 40, K = 1)
    rng = MersenneTwister(seed)
    β = log.(rand(rng, p) .* 2 .+ 0.5)
    Λ = randn(rng, p, K) .* 0.4
    Y = zeros(p, n)
    for s in 1:n
        z = randn(rng, K)
        for t in 1:p
            μ = exp(β[t] + dot(Λ[t, :], z))
            k = rand(rng, Poisson(μ))
            Y[t, s] = k == 0 ? 0.0 : sum(rand(rng, Gamma(2.0, μ / (2.0 * μ + 1e-9)), k))
        end
    end
    return Y, β, Λ
end

# The pre-#503 per-site kernel, verbatim apart from also reporting whether its undamped
# Fisher loop converged. Used only where it DID converge, to check the fix leaves those
# values alone.
function _tw503_old_site(fams, y, Λ, β; hessian = :observed, maxiter = 100, tol = 1e-9)
    p, K = size(Λ)
    n = ones(Int, p)
    link = _G.LogLink()
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        η  = _G._clamp_eta.(β .+ Λ * z)
        μ  = _G._clamp_mu.(fams, _G.linkinv.(Ref(link), η))
        me = _G.mu_eta.(Ref(link), η)
        s  = _G._glm_score.(fams, μ, n, me, y)
        W  = _G._tweedie_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link), η)
        Δ  = _G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    η  = _G._clamp_eta.(β .+ Λ * z)
    μ  = _G._clamp_mu.(fams, _G.linkinv.(Ref(link), η))
    me = _G.mu_eta.(Ref(link), η)
    W  = _G._tweedie_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link), η)
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(_G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Closed-form ∇_z of the per-site log-posterior under the log link: shares no code with
# the kernel. score_η = (y − μ)/(φ μ^(p−1)).
function _tw503_grad(y, Λ, β, φvec, pw, z)
    μ = exp.(β .+ Λ * z)
    score = (y .- μ) ./ (φvec .* μ .^ (pw - 1))
    return Λ' * score .- z
end

@testset "Tweedie grouped mode search: converged is reported, and fails loudly (#503)" begin
    link = _G.LogLink()

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        Y, β, Λ = _tw503_sim(503)
        p = size(Y, 1)
        φvec = fill(1.0, p)
        fams = [_G.TweedieED(φvec[t], 1.5) for t in 1:p]
        y = Y[:, 1]
        n1 = ones(Int, p)
        # A zero-iteration budget cannot reach tol = 1e-9 from z = 0, and the 20x retry
        # (20 * 0 = 0) does not change that: a budget-independent non-convergence case.
        @test _G._tweedie_grouped_loglik_site(fams, y, n1, Λ, β, link; maxiter = 0) == -Inf
        z, ok = _G._tweedie_grouped_mode(fams, y, n1, Λ, β, link; maxiter = 0)
        @test !ok
        @test _G.tweedie_grouped_marginal_loglik_laplace(Y, Λ, β, φvec, 1.5; maxiter = 0) == -Inf
    end

    @testset "every site the search reports converged sits at a mode of the log-posterior" begin
        pw = 1.5
        n_checked = 0
        for seed in 1:6, scale in (1.0, 2.0, 4.0)
            Y, β, Λ0 = _tw503_sim(seed; p = 6, n = 8)
            Λ = scale .* Λ0
            p = size(Y, 1)
            φvec = fill(1.0, p)
            fams = [_G.TweedieED(φvec[t], pw) for t in 1:p]
            n1 = ones(Int, p)
            for s in 1:size(Y, 2)
                y = Y[:, s]
                z, ok = _G._tweedie_grouped_mode(fams, y, n1, Λ, β, link)
                ok || continue
                n_checked += 1
                @test maximum(abs, _tw503_grad(y, Λ, β, φvec, pw, z)) < 1e-4
                # a converged site has a finite value, and the site kernel agrees
                v = _G._tweedie_grouped_loglik_site(fams, y, n1, Λ, β, link; hessian = :fisher)
                @test isfinite(v)
            end
        end
        @test n_checked >= 60
    end

    @testset "where the old undamped loop converged, the value is unchanged (to 1e-8)" begin
        # Small, well-behaved literal points (tame counts, moderate dispersion), so the
        # old loop converges and the fix has to leave those sites alone.
        points = [
            (reshape([0.3, -0.2, 0.4, 0.1], 4, 1), [0.5, 0.2, -0.1, 0.3],
             [1.0, 1.5, 0.8, 2.0], [3.0, 0.0, 2.0, 4.0], 1.5),
            (reshape([0.2, 0.15, -0.3, 0.25, -0.1], 5, 1), [0.1, -0.2, 0.3, 0.0, 0.2],
             [0.5, 1.0, 2.0, 0.8, 1.5], [2.0, 1.0, 0.0, 3.0, 4.0], 1.3),
            (reshape([0.1, -0.15, 0.2, 0.05], 4, 1), [-0.2, 0.4, 0.1, -0.3],
             [1.2, 0.9, 1.8, 2.5], [1.0, 0.0, 2.0, 3.0], 1.7),
            (reshape([0.25, -0.1, 0.3, 0.2, -0.2, 0.1], 6, 1), [0.3, 0.0, -0.2, 0.4, 0.1, -0.1],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 3.0, 0.0, 1.0, 2.0], 1.5),
        ]
        n_compared = 0
        worst = 0.0
        for (Λ, β, φvec, y, pw) in points
            p = length(y)
            fams = [_G.TweedieED(φvec[t], pw) for t in 1:p]
            old, conv = _tw503_old_site(fams, y, Λ, β)
            conv || continue
            new = _G._tweedie_grouped_loglik_site(fams, y, ones(Int, p), Λ, β, link)
            worst = max(worst, abs(new - old))
            n_compared += 1
        end
        @test n_compared >= 3
        @test worst < 1e-8
    end

    @testset "fit-level: converged comes from the fit, and a fit that cannot converge says so" begin
        Y, _, _ = _tw503_sim(2024; p = 5, n = 40, K = 1)

        # (1) every site's search fails (no Newton budget): the objective sits on its
        # failure sentinel everywhere, so the fit must report converged = false and no
        # finite log-likelihood. Before the change the kernel returned a finite value at
        # the unconverged z = 0 and the fitter optimised that surface.
        f0 = fit_tweedie_gllvm_grouped(Y; K = 1, newton_maxiter = 0)
        @test !f0.converged
        @test f0.loglik == -Inf

        # (2) an optimiser stopped by its iteration cap is not converged.
        f1 = fit_tweedie_gllvm_grouped(Y; K = 1, iterations = 1)
        @test !f1.converged

        # (3) a fit that does converge reports a log-likelihood equal to the marginal at
        # its own parameters: the flag and the value describe the same point.
        f = fit_tweedie_gllvm_grouped(Y; K = 1)
        if f.converged
            φvec = [f.φ[f.group[t]] for t in 1:size(Y, 1)]
            ll = _G.tweedie_grouped_marginal_loglik_laplace(Y, f.Λ, f.β, φvec, f.power;
                                                            hessian = f.hessian)
            @test isfinite(f.loglik)
            @test isapprox(f.loglik, ll; rtol = 1e-6)
        end
        @test f.converged isa Bool
    end
end
