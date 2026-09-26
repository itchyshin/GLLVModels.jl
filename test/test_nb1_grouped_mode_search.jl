using GLLVModels, Test, SHA, TOML, Random, LinearAlgebra, Distributions, SpecialFunctions

# `_nb1_grouped_loglik_site` could return a finite site log-likelihood at a z whose
# per-site log-posterior gradient was not near zero (#503, the same defect #479 fixed
# for the Gamma grouped kernel): the per-site mode search took undamped Fisher-scoring
# steps and returned whatever z the loop held when it stopped, converged or not. The
# class-audit's stress probe (Lambda scale up to 3x) measured a 3.3-3.6% non-mode rate
# with the diverged z still producing a finite value that escaped the fitter's 1e12
# sentinel. The search (`_nb1_grouped_mode`) now halves any step that lowers the
# per-site log-posterior, falls back to a damped Newton search on the observed NB1/log
# curvature under LogLink when Fisher scoring does not converge, and returns -Inf for a
# site whose search still fails, so the fitter's sentinel fires instead of a garbage
# value silently reaching L-BFGS.

const _NB1_503_FIXTURE = joinpath(@__DIR__, "fixtures", "nb1_grouped_mode_search_503.toml")

function _nb1_503_data()
    d = TOML.parsefile(_NB1_503_FIXTURE)
    y = Float64.(d["y"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == d["y_sha256"]
    p, K = d["p"], d["K"]
    β = Float64.(d["beta"])
    φ = Float64.(d["phi"])
    Λ = reshape(Float64.(d["Lambda_column_major"]), p, K)
    return y, Λ, β, φ, d["v_ref"]
end

_nb1_503_fams(φvec) = [GLLVModels.NB1(float(f)) for f in φvec]

# The pre-#503 per-site kernel, kept verbatim apart from also reporting whether its
# undamped Fisher loop converged. Used only where it DID converge, to check that the
# fix leaves those values alone.
function _nb1_503_old_site(fams, y, Λ, β; maxiter = 100, tol = 1e-9)
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
        W  = G._nb1_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link))
        Δ  = G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    η  = G._clamp_eta.(β .+ Λ * z)
    μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    W  = G._nb1_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link))
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Independent per-site Laplace value at a given z: closed-form NB1/log gradient and
# observed curvature (shares no code with the kernel), damped Newton to a tight
# tolerance, so this is a from-scratch check that a claimed value is actually AT a
# stationary point of the log-posterior with a negative-definite Hessian (the
# log-posterior is concave there; -H is `Λ'WΛ + I` with W = -∂²ℓ/∂η² >= 0 built in).
function _nb1_503_grad_and_negdefH(y, Λ, β, φvec, z)
    p, K = size(Λ)
    η = β .+ Λ * z
    μ = exp.(η)
    r = μ ./ φvec
    s_μ = (digamma.(y .+ r) .- digamma.(r) .- log1p.(φvec)) ./ φvec
    score = μ .* s_μ                       # ∂ℓ/∂η for NB1/log
    g = Λ' * score .- z                    # ∇_z log-posterior
    W = -μ .* s_μ .- (μ ./ φvec).^2 .* (trigamma.(y .+ r) .- trigamma.(r))
    H = -(Λ' * (W .* Λ) + I)               # Hessian of the log-posterior
    return g, H
end

@testset "NB1 grouped mode search: damped, and fails loudly (#503)" begin
    y, Λ0, β0, φ0, v_ref = _nb1_503_data()
    p, K = size(Λ0)
    n1 = ones(Int, p)
    link = GLLVModels.LogLink()

    @testset "the audit's diverging site now gets its true Laplace value" begin
        # Before: undamped Fisher scoring returned a finite garbage value at a
        # non-mode z (v_k_on_main in the fixture, about -1.53e10). The fixed kernel
        # must return the reference value instead.
        fams = _nb1_503_fams(φ0)
        v = GLLVModels._nb1_grouped_loglik_site(fams, y, n1, Λ0, β0, link; hessian = :fisher)
        @test isfinite(v)
        @test abs(v - v_ref) < 1e-8
    end

    @testset "the returned mode is a stationary point with negative-definite Hessian" begin
        fams = _nb1_503_fams(φ0)
        z, ok = GLLVModels._nb1_grouped_mode(fams, y, n1, Λ0, β0, link, :fisher)
        @test ok
        g, H = _nb1_503_grad_and_negdefH(y, Λ0, β0, φ0, z)
        @test maximum(abs, g) < 1e-6
        @test isposdef(Symmetric(-H))       # H negative-definite <=> -H positive-definite
    end

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        fams = _nb1_503_fams(φ0)
        # Two iterations cannot reach tol = 1e-9 from z = 0 on this fixture's site.
        @test GLLVModels._nb1_grouped_loglik_site(fams, y, n1, Λ0, β0, link;
                                                  maxiter = 2) == -Inf
        Y = reshape(y, p, 1)
        @test GLLVModels.nb1_grouped_marginal_loglik_laplace(Y, Λ0, β0, φ0; maxiter = 2) == -Inf
    end

    @testset "every replicate `_nb1_grouped_mode` reports converged has converged = true and sits at a mode" begin
        # Sweep several parameter points derived from the fixture (scaled Lambda, a
        # perturbed beta, a rescaled phi) plus the fixture itself; relation-only checks
        # (no seed-tied numeric outcome), so they hold regardless of which Julia
        # version drew what upstream.
        cases = [(Λ0, β0, φ0), (1.5 .* Λ0, β0, φ0), (Λ0, β0 .+ 0.3, φ0),
                 (Λ0, β0, 0.5 .* φ0), (3.0 .* Λ0, β0, φ0)]
        n_checked = 0
        for (Λ, β, φvec) in cases
            fams = _nb1_503_fams(φvec)
            z, ok = GLLVModels._nb1_grouped_mode(fams, y, n1, Λ, β, link, :fisher)
            ok || continue
            n_checked += 1
            g, H = _nb1_503_grad_and_negdefH(y, Λ, β, φvec, z)
            @test maximum(abs, g) < 1e-6
            @test isposdef(Symmetric(-H))
            # the reported value agrees with an independent evaluation at that z
            v = GLLVModels._nb1_grouped_loglik_site(fams, y, n1, Λ, β, link; hessian = :fisher)
            η = β .+ Λ * z
            μ = exp.(η)
            W = GLLVModels._nb1_grouped_laplace_weight.(Ref(:fisher), fams, μ, GLLVModels.mu_eta.(Ref(link), η), y, Ref(link))
            A = Symmetric(Λ' * (W .* Λ) + I)
            ℓ = sum(GLLVModels._glm_logpdf(fams[t], μ[t], n1[t], y[t]) for t in 1:p)
            v_indep = ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
            @test abs(v - v_indep) < 1e-8
        end
        @test n_checked >= 4
    end

    @testset "where the old undamped loop converged, the value is unchanged (to 1e-8)" begin
        # The fixture's own site is chosen because the old loop diverges there under
        # every perturbation tried; a handful of small, well-behaved literal NB1
        # points (tame counts, moderate dispersion) are added so the old loop has
        # cases where it actually converges, to check the fix leaves those alone.
        well_behaved = [
            (reshape([0.3, -0.2, 0.4, 0.1], 4, 1), [0.5, 0.2, -0.1, 0.3],
             [1.0, 1.5, 0.8, 2.0], [3.0, 5.0, 2.0, 4.0]),
            (reshape([0.2, 0.15, -0.3, 0.25, -0.1], 5, 1), [0.1, -0.2, 0.3, 0.0, 0.2],
             [0.5, 1.0, 2.0, 0.8, 1.5], [2.0, 1.0, 6.0, 3.0, 4.0]),
            (reshape([0.1, -0.15, 0.2, 0.05], 4, 1), [-0.2, 0.4, 0.1, -0.3],
             [1.2, 0.9, 1.8, 2.5], [1.0, 4.0, 2.0, 3.0]),
        ]
        cases = [(Λ0, β0, φ0), (1.5 .* Λ0, β0, φ0), (Λ0, β0 .+ 0.3, φ0),
                 (Λ0, β0, 0.5 .* φ0), (3.0 .* Λ0, β0, φ0)]
        n_compared = 0
        worst = 0.0
        for (Λ, β, φvec) in cases
            fams = _nb1_503_fams(φvec)
            old, conv = _nb1_503_old_site(fams, y, Λ, β)
            conv || continue
            new = GLLVModels._nb1_grouped_loglik_site(fams, y, n1, Λ, β, link)
            worst = max(worst, abs(new - old))
            n_compared += 1
        end
        for (Λ, β, φvec, yc) in well_behaved
            fams = _nb1_503_fams(φvec)
            nc = ones(Int, length(yc))
            old, conv = _nb1_503_old_site(fams, yc, Λ, β)
            @test conv          # these points are chosen precisely so the old loop converges
            conv || continue
            new = GLLVModels._nb1_grouped_loglik_site(fams, yc, nc, Λ, β, link)
            worst = max(worst, abs(new - old))
            n_compared += 1
        end
        @test n_compared >= 3
        @test worst < 1e-8
    end
end
