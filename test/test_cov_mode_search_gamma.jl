using GLLVModels, Test, SHA, TOML, Random, LinearAlgebra, Distributions, Logging

# fit_gllvm_cov(Y; family = Gamma(...), X, K) threw DomainError("Gamma: alpha > 0") on
# ordinary Gamma data. Two faults, both in src/families/covariates.jl:
# (1) the offset-aware per-site mode search `_laplace_mode_off` took full Fisher-scoring
#     steps with no damping and no convergence check. At the fitter's own warm start it
#     diverged on one site (z about 1e11) and returned a FINITE site value (about -7.2e22),
#     so the objective's 1e12 sentinel never fired and L-BFGS saw a garbage gradient;
# (2) the line search then jumped to an extreme log alpha, and the family marker was
#     built from exp(log alpha) = 0.0 OUTSIDE the objective's `try`, so Distributions'
#     DomainError escaped instead of returning the sentinel.
# The search now halves any step that lowers the per-site log-posterior, falls back to a
# damped Newton search on the observed curvature when Fisher scoring does not converge,
# and a site that still fails returns -Inf. The family is built inside the `try`.
# The same kernel serves fit_gllvm_speciescov, fit_fourthcorner_gllvm,
# fit_roweffect_gllvm and fit_constrained_gllvm, which threw the same error on this data.
# Data: the sibling screen's dataset 2 (seed 2, alpha 2) and a heavily dispersed dataset
# (seed 1, alpha 0.3); recipes in the fixture headers.

const _CMS_FIXTURES = joinpath(@__DIR__, "fixtures")

function _cms_data(file, pinned_sha)
    d = TOML.parsefile(joinpath(_CMS_FIXTURES, file))
    Y = reshape(Float64.(d["Y_column_major"]), d["p"], d["n"])
    h = bytes2hex(sha256(reinterpret(UInt8, vec(Y))))
    @test h == d["data_sha256"] == pinned_sha
    return Y, d["K"]
end

# The warm start fit_gllvm_cov builds for Gamma (log row means, SVD loadings).
function _cms_warmstart(Y, K)
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

# The pre-fix `_laplace_mode_off` and `_laplace_site_off`, kept verbatim apart from also
# reporting whether the undamped loop converged. Used only where it DID converge, to
# check that the fix leaves those values and modes alone. No mask (none of the cases
# below has missing cells).
function _cms_old_site(family, y, n, Λ, η0, link; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    K = size(Λ, 2)
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        η  = G._clamp_eta.(η0 .+ Λ * z)
        μ  = G._clamp_mu.(Ref(family), G.linkinv.(Ref(link), η))
        me = G.mu_eta.(Ref(link), η)
        s  = G._glm_score.(Ref(family), μ, n, me, y)
        W  = G._glm_weight.(Ref(family), μ, n, me)
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = G._safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z  = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    hessian = G._default_hessian(family, link)
    η  = G._clamp_eta.(η0 .+ Λ * z)
    μ  = G._clamp_mu.(Ref(family), G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    W  = if hessian === :fisher || G._glm_weight_matches_observed(family, link)
        G._glm_weight.(Ref(family), μ, n, me)
    else
        [G._glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) for t in eachindex(y)]
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(G._glm_logpdf(family, μ[t], n[t], y[t]) for t in eachindex(y))
    if any(w -> w < zero(w), W)
        issuccess(cholesky(A; check = false)) || return -Inf, conv, z
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv, z
end

# Independent per-site Laplace value for Gamma/log: Distributions.logpdf, the analytic
# score and observed curvature, and its own damped Newton to a tight tolerance. Shares
# no code with the kernel.
function _cms_indep_gamma_site(y, Λ, η0, α)
    h(z) = sum(logpdf(Gamma(α, exp(η0[t] + dot(Λ[t, :], z)) / α), y[t])
               for t in eachindex(y)) - 0.5 * dot(z, z)
    z = zeros(size(Λ, 2))
    for _ in 1:200
        μ = exp.(η0 .+ Λ * z)
        g = Λ' * (α .* (y ./ μ .- 1)) .- z
        H = Λ' * ((α .* y ./ μ) .* Λ) + I
        d = H \ g
        step = 1.0
        while h(z .+ step .* d) < h(z) && step > 1e-10
            step /= 2
        end
        z = z .+ step .* d
        maximum(abs, step .* d) < 1e-13 && break
    end
    μ = exp.(η0 .+ Λ * z)
    H = Λ' * ((α .* y ./ μ) .* Λ) + I
    return h(z) - 0.5 * logdet(H)
end

_cms_quiet(f) = with_logger(f, NullLogger())

@testset "fit_gllvm_cov offset mode search: damped, fails loudly, no DomainError" begin
    G = GLLVModels
    Y, K = _cms_data("cov_gamma_mode_search_seed2.toml",
                     "2a1f201e83d6136945f9564cacad96f9e9480c09ccfbe35d491243636b7ae38a")
    p, n = size(Y)
    β0, Λ0 = _cms_warmstart(Y, K)
    link = G.LogLink()
    ones_p = ones(Int, p)
    gam2 = Gamma(2.0, 1.0)

    @testset "warm-start site 7: the diverging site now gets its true Laplace value" begin
        # Before: undamped Fisher scoring ran out to z about 1e11 and returned -7.2248e22.
        v = G._laplace_site_off(gam2, Y[:, 7], ones_p, Λ0, β0, link)
        @test isfinite(v)
        @test abs(v - _cms_indep_gamma_site(Y[:, 7], Λ0, β0, 2.0)) < 1e-8
        worst = maximum(abs(G._laplace_site_off(gam2, Y[:, i], ones_p, Λ0, β0, link) -
                            _cms_indep_gamma_site(Y[:, i], Λ0, β0, 2.0)) for i in 1:n)
        @test worst < 1e-8
        @test isfinite(G._marginal_loglik_offset(gam2, Y, fill(1, p, n), Λ0, β0, zeros(p, n), link))
    end

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        # Two iterations cannot reach tol = 1e-9 from z = 0 at any of these sites.
        for i in (7, 10, 42)
            @test G._laplace_site_off(gam2, Y[:, i], ones_p, Λ0, β0, link; maxiter = 2) == -Inf
        end
        @test G._marginal_loglik_offset(gam2, Y, fill(1, p, n), Λ0, β0, zeros(p, n), link;
                                        maxiter = 2) == -Inf
        # ... so the fitter's sentinel fires and the fit does not claim convergence.
        fit = _cms_quiet(() -> G.fit_gllvm_cov(Y; family = gam2, X = zeros(p, n, 1), K = K,
                                               newton_maxiter = 2))
        @test !fit.converged
        @test fit.loglik == -Inf
        # getLV's mode helper still returns a plain score (no NaN, no flag): this change
        # is to the objective only.
        @test all(isfinite, G._laplace_mode_off(gam2, Y[:, 7], ones_p, Λ0, β0, link; maxiter = 2))
    end

    @testset "where the old search converged, values and modes are unchanged (to 1e-8)" begin
        # Gamma on the fixture (warm start at alpha 2 and 1) plus well-behaved simulated
        # data for Gamma, Poisson, NB2, Beta and Binomial at the truth and at two
        # perturbed points. At every site where the pre-fix undamped loop converged, the
        # new kernel must return the same value and the same mode.
        rng = MersenneTwister(20260925)
        p2, n2 = 5, 60
        Λt = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
        xs = randn(rng, n2)
        Zt = randn(rng, K, n2)
        cases = Any[(gam2, Y, Λ0, β0 .+ zeros(p, n)), (Gamma(1.0, 1.0), Y, Λ0, β0 .+ zeros(p, n))]
        for (fam, βt, perturbed_disp) in
                ((Gamma(2.0, 1.0), [0.5, 1.0, -0.5, 1.5, 0.0], Gamma(1.0, 1.0)),
                 (Poisson(), [0.5, 1.0, -0.5, 1.5, 0.0], Poisson()),
                 (NegativeBinomial(5.0, 0.5), [0.5, 1.0, -0.5, 1.5, 0.0], NegativeBinomial(2.5, 0.5)),
                 (Beta(10.0, 1.0), [0.0, 0.5, -0.5, 1.0, -1.0], Beta(5.0, 1.0)),
                 (Binomial(), [0.0, 0.5, -0.5, 1.0, -1.0], Binomial()))
            lk = G._cov_default_link(fam)
            O = [βt[t] + 0.3 * xs[i] for t in 1:p2, i in 1:n2]
            Ys = [G._cov_sample(fam, G.linkinv(lk, O[t, i] + dot(Λt[t, :], Zt[:, i])), 1, rng)
                  for t in 1:p2, i in 1:n2]
            push!(cases, (fam, float.(Ys), Λt, O))
            push!(cases, (fam, float.(Ys), 1.5 .* Λt, O .+ 0.3))
            push!(cases, (perturbed_disp, float.(Ys), Λt, O))
        end
        n_compared = 0
        n_total = 0
        worst_v = 0.0
        worst_z = 0.0
        for (fam, Yc, Λ, E) in cases
            lk = G._cov_default_link(fam)
            nn = ones(Int, size(Yc, 1))
            for i in axes(Yc, 2)
                n_total += 1
                old, conv, zold = _cms_old_site(fam, Yc[:, i], nn, Λ, E[:, i], lk)
                conv || continue
                new = G._laplace_site_off(fam, Yc[:, i], nn, Λ, E[:, i], lk)
                znew = G._laplace_mode_off(fam, Yc[:, i], nn, Λ, E[:, i], lk)
                worst_v = max(worst_v, abs(new - old))
                worst_z = max(worst_z, maximum(abs, znew .- zold))
                n_compared += 1
            end
        end
        @test n_total == 2 * 80 + 15 * 60
        @test n_compared >= n_total - 10   # measured: 1059 of 1060 (not warm-start site 7)
        @test worst_v < 1e-8
        @test worst_z < 1e-8
    end

    @testset "fit_gllvm_cov Gamma returns a finite fit instead of throwing" begin
        # Before: DomainError from Gamma(0.0, 1.0) in the line search, for both designs.
        # X = 0 is the no-covariate shared-shape model, whose optimum fit_gamma_gllvm
        # reaches at -569.700833.
        fit0 = _cms_quiet(() -> G.fit_gllvm_cov(Y; family = gam2, X = zeros(p, n, 1), K = K))
        @test isfinite(fit0.loglik)
        @test fit0.converged
        @test fit0.loglik >= -569.700833 - 1e-5
        X1 = randn(MersenneTwister(99), p, n, 1)
        fit1 = _cms_quiet(() -> G.fit_gllvm_cov(Y; family = gam2, X = X1, K = K))
        @test isfinite(fit1.loglik)
        @test fit1.converged
        @test fit1.loglik >= fit0.loglik - 1e-6          # nested: a covariate cannot lower the max

        # The confidence-interval objective builds the family the same way; an
        # out-of-range log alpha must give its sentinel, not a DomainError.
        ci = G._family_ci(fit0, Y; X = zeros(p, n, 1))
        θ = copy(ci.θ)
        θ[end] = -1000.0                                  # exp(-1000) == 0.0
        @test ci.nll(θ) == 1e12
    end

    @testset "same kernel: sibling fitters on the seed-2 data" begin
        # Before: both threw the same DomainError. With X = 0 (or C = 0) each nests the
        # no-covariate model, so its optimum is at least -569.700833.
        fs = _cms_quiet(() -> G.fit_gllvm_speciescov(Y; family = gam2, X = zeros(p, n, 1), K = K))
        @test fs.converged
        @test fs.loglik >= -569.700833 - 1e-5
        ff = _cms_quiet(() -> G.fit_fourthcorner_gllvm(Y; family = gam2,
                                                       Xenv = randn(MersenneTwister(99), n, 1),
                                                       TR = randn(MersenneTwister(98), p, 1), K = K))
        @test ff.converged
        @test ff.loglik >= -569.700833 - 1e-5
    end

    @testset "dispersion guard: heavily dispersed Gamma data (alpha 0.3)" begin
        # Before: DomainError. With the mode search fixed but the family still built
        # outside the `try`, this dataset STILL threw, so it tests the guard on its own.
        Yd, Kd = _cms_data("cov_gamma_dispersion_guard_a03_seed1.toml",
                           "d6663d884c3f25dbef6d23c5ec179b66c0b485f9413ebb338fa30da796e1e8bd")
        fd = _cms_quiet(() -> G.fit_gllvm_cov(Yd; family = gam2, X = zeros(size(Yd)..., 1), K = Kd))
        @test isfinite(fd.loglik)
        @test fd.converged
        @test 0.15 < fd.dispersion < 0.6                  # true alpha 0.3
    end
end
