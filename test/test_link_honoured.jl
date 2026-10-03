# A fitter must optimise the objective of the link it is asked for.
#
# Before this file: `fit_beta_gllvm` and `fit_gamma_gllvm` accepted `link = ...`, used it for
# the warm start and stored it on the fit, but built their objective WITHOUT it, so a probit /
# cloglog / identity Beta fit (and an identity Gamma fit) optimised the logit / log model and
# reported the result as converged. The predictor-informed (`X_lv`) packed objectives of Beta,
# Gamma and NB2 dropped `link` the same way. Separately, the Poisson and NB2 analytic gradients
# are log-link-only, yet were engaged for any link whose default curvature is Fisher, so an
# identity-link fit stopped where the true gradient is far from zero. The variational (ELBO)
# fitters accepted `link` while their ELBO is derived for one link only.
#
# Every check below is an INDEPENDENT one: the stored `loglik` is compared with the Laplace
# marginal of the REQUESTED link evaluated at the returned parameters, the true objective's
# gradient is checked at the returned optimum, and a simulation with a known truth checks
# that the intercepts are recovered on the requested link's scale.
module LinkHonoured

using GLLVModels, Test, Random, Distributions, LinearAlgebra
const G = GLLVModels

# max |central finite-difference gradient| of `f` at `θ`
function fd_grad_max(f, θ; h = 1e-6)
    m = 0.0
    for i in eachindex(θ)
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        m = max(m, abs((f(θp) - f(θm)) / (2h)))
    end
    return m
end

# One latent variable, fixed loadings `λ`, intercepts `β`; returns (Y, β, λ).
function sim_beta(link; p = 6, n = 300, seed = 1, φ = 25.0,
                  β = collect(range(-0.8, 0.8; length = p)),
                  λ = [0.7, 0.5, -0.4, 0.6, 0.3, -0.5])
    rng = MersenneTwister(seed)
    z = randn(rng, n)
    Y = zeros(p, n)
    for i in 1:n, t in 1:p
        μ = clamp(G.linkinv(link, β[t] + λ[t] * z[i]), 1e-3, 1 - 1e-3)
        Y[t, i] = rand(rng, Beta(μ * φ, (1 - μ) * φ))
    end
    return Y, β, λ
end

function sim_gamma(link; p = 6, n = 300, seed = 1, α = 8.0,
                   β = 4.0 .+ collect(range(-1.0, 1.0; length = p)),
                   λ = [0.5, 0.3, -0.3, 0.4, 0.2, -0.3])
    rng = MersenneTwister(seed)
    z = randn(rng, n)
    Y = zeros(p, n)
    for i in 1:n, t in 1:p
        μ = max(G.linkinv(link, β[t] + λ[t] * z[i]), 1e-3)
        Y[t, i] = rand(rng, Gamma(α, μ / α))
    end
    return Y, β, λ
end

function sim_count(link, family; p = 6, n = 200, seed = 1, base = 6.0, r = 5.0,
                   β = base .+ collect(range(-1.0, 1.0; length = p)),
                   λ = [0.5, 0.3, -0.3, 0.4, 0.2, -0.3])
    rng = MersenneTwister(seed)
    z = randn(rng, n)
    Y = zeros(p, n)
    for i in 1:n, t in 1:p
        μ = max(G.linkinv(link, β[t] + λ[t] * z[i]), 0.05)
        Y[t, i] = family === :poisson ? rand(rng, Poisson(μ)) :
                                        rand(rng, NegativeBinomial(r, r / (r + μ)))
    end
    return Y, β, λ
end

@testset "fit_beta_gllvm honours a non-default link" begin
    p, K = 6, 1
    rr = G.rr_theta_len(p, K)
    cases = (
        ("probit",   G.ProbitLink(),   nothing),
        ("cloglog",  G.CLogLogLink(),  nothing),
        # identity needs a mean inside (0, 1): a narrow band around 0.5
        ("identity", G.IdentityLink(), (β = 0.5 .+ collect(range(-0.2, 0.2; length = p)),
                                        λ = 0.06 .* [1.0, 0.7, -0.6, 0.8, 0.4, -0.7])),
    )
    for (nm, link, sim_kw) in cases
        @testset "$nm" begin
            Y, β, _ = sim_kw === nothing ? sim_beta(link; seed = 11) :
                                           sim_beta(link; seed = 11, sim_kw...)
            fit = fit_beta_gllvm(Y; K = K, link = link)
            @test fit.link isa typeof(link)
            # (1) the stored loglik IS the requested-link marginal at the returned parameters
            ll_req = G.beta_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.φ;
                                                    link = link, hessian = fit.hessian)
            @test isapprox(fit.loglik, ll_req; atol = 1e-8)
            # ... and it is NOT the logit-link marginal of the same parameters
            ll_logit = G.beta_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.φ;
                                                      link = G.LogitLink(),
                                                      hessian = fit.hessian)
            @test !(isapprox(fit.loglik, ll_logit; atol = 1e-3))
            # (2) the returned point is a stationary point of the requested-link objective
            θ̂ = vcat(fit.β, G.pack_lambda(fit.Λ), log(fit.φ))
            obj = θ -> -G.beta_marginal_loglik_laplace(Y, G.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                       θ[1:p], exp(θ[end]);
                                                       link = link, hessian = fit.hessian)
            @test fd_grad_max(obj, θ̂) < 1e-2
            # (3) a known intercept vector is recovered on the requested link's scale
            tol = nm == "identity" ? 0.06 : 0.25
            @test maximum(abs.(fit.β .- β)) < tol
        end
    end

    @testset "predictor-informed (X_lv) packed objective uses the link" begin
        link = G.ProbitLink()
        Y, _, _ = sim_beta(link; n = 200, seed = 12)
        X = randn(MersenneTwister(5), 200, 1)
        fit = fit_beta_gllvm(Y; K = K, link = link, X_lv = X)
        off = G._lv_mean_eta(fit.Λ, X, fit.alpha_lv)
        ll_req = G.beta_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.φ; link = link,
                                                offset = off, hessian = fit.hessian)
        @test isapprox(fit.loglik, ll_req; atol = 1e-8)
    end

end

@testset "fit_gamma_gllvm honours a non-default link" begin
    p, K = 6, 1
    rr = G.rr_theta_len(p, K)
    link = G.IdentityLink()
    @testset "identity" begin
        Y, β, _ = sim_gamma(link; seed = 21)
        fit = fit_gamma_gllvm(Y; K = K, link = link)
        @test fit.link isa G.IdentityLink
        ll_req = G.gamma_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.α;
                                                 link = link, hessian = fit.hessian)
        @test isapprox(fit.loglik, ll_req; atol = 1e-8)
        ll_log = G.gamma_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.α;
                                                 link = G.LogLink(), hessian = fit.hessian)
        @test !(isapprox(fit.loglik, ll_log; atol = 1e-3))
        θ̂ = vcat(fit.β, G.pack_lambda(fit.Λ), log(fit.α))
        obj = θ -> -G.gamma_marginal_loglik_laplace(Y, G.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                    θ[1:p], exp(θ[end]);
                                                    link = link, hessian = fit.hessian)
        @test fd_grad_max(obj, θ̂) < 1e-2
        # intercepts are on the MEAN scale under the identity link (truth ~ 3 to 5; the log-link
        # objective returns values near log(mean) ~ 1.4, an error of ~3)
        @test maximum(abs.(fit.β .- β)) < 0.5
    end

    @testset "predictor-informed (X_lv) packed objective uses the link" begin
        Y, _, _ = sim_gamma(link; n = 200, seed = 22)
        X = randn(MersenneTwister(6), 200, 1)
        fit = fit_gamma_gllvm(Y; K = K, link = link, X_lv = X)
        off = G._lv_mean_eta(fit.Λ, X, fit.alpha_lv)
        ll_req = G.gamma_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.α; link = link,
                                                 offset = off, hessian = fit.hessian)
        @test isapprox(fit.loglik, ll_req; atol = 1e-8)
    end

    @testset "reached through fit_gllvm" begin
        Y, _, _ = sim_gamma(link; n = 150, seed = 23)
        fit = fit_gllvm(Y; family = G.Gamma(), K = K, link = link)
        ll_req = G.gamma_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.α; link = link,
                                                 hessian = fit.hessian)
        @test isapprox(fit.loglik, ll_req; atol = 1e-8)
    end
end

@testset "NB2 X_lv packed objective honours a non-default link" begin
    p, K = 6, 1
    link = G.IdentityLink()
    Y, _, _ = sim_count(link, :nb; seed = 31)
    X = randn(MersenneTwister(7), size(Y, 2), 1)
    fit = fit_nb_gllvm(Y; K = K, link = link, X_lv = X)
    off = G._lv_mean_eta(fit.Λ, X, fit.alpha_lv)
    ll_req = G.nb_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.r; link = link,
                                          offset = off, hessian = fit.hessian)
    @test isapprox(fit.loglik, ll_req; atol = 1e-8)
end

@testset "analytic gradients are only used for the link they were derived for" begin
    p, K = 6, 1
    rr = G.rr_theta_len(p, K)
    link = G.IdentityLink()

    @testset "Poisson identity" begin
        Y, _, _ = sim_count(link, :poisson; seed = 41)
        fit = fit_poisson_gllvm(Y; K = K, link = link)
        θ̂ = vcat(fit.β, G.pack_lambda(fit.Λ))
        obj = θ -> -G.poisson_marginal_loglik_laplace(Y, G.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                      θ[1:p], link; hessian = fit.hessian)
        @test isapprox(fit.loglik, -obj(θ̂); atol = 1e-8)
        @test fd_grad_max(obj, θ̂) < 1e-2
        ffd = fit_poisson_gllvm(Y; K = K, link = link, gradient = :finite)
        @test isapprox(fit.loglik, ffd.loglik; atol = 1e-6)
    end

    @testset "NB2 identity" begin
        Y, _, _ = sim_count(link, :nb; seed = 42)
        fit = fit_nb_gllvm(Y; K = K, link = link)
        θ̂ = vcat(fit.β, G.pack_lambda(fit.Λ), log(fit.r))
        obj = θ -> -G.nb_marginal_loglik_laplace(Y, G.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                 θ[1:p], exp(θ[end]); link = link,
                                                 hessian = fit.hessian)
        @test isapprox(fit.loglik, -obj(θ̂); atol = 1e-8)
        @test fd_grad_max(obj, θ̂) < 1e-2
        ffd = fit_nb_gllvm(Y; K = K, link = link, gradient = :finite)
        @test isapprox(fit.loglik, ffd.loglik; atol = 1e-6)
    end
end

@testset "variational fitters refuse a link their ELBO is not derived for" begin
    p, n, K = 6, 60, 1
    Yb, _, _ = sim_beta(G.LogitLink(); n = n, seed = 51)
    Yg, _, _ = sim_gamma(G.LogLink(); n = n, seed = 52)
    Yc, _, _ = sim_count(G.LogLink(), :nb; n = n, seed = 53, base = 1.5)
    Yi = round.(Int, Yc)
    Ybin = Int.(rand(MersenneTwister(54), p, n) .< 0.5)
    @test_throws ArgumentError fit_beta_gllvm_va(Yb; K = K, link = G.ProbitLink())
    @test_throws ArgumentError fit_gamma_gllvm_va(Yg; K = K, link = G.IdentityLink())
    @test_throws ArgumentError fit_exponential_gllvm_va(Yg; K = K, link = G.IdentityLink())
    @test_throws ArgumentError fit_nb_gllvm_va(Yi; K = K, link = G.IdentityLink())
    @test_throws ArgumentError fit_binomial_gllvm_va(Ybin; K = K, link = G.ProbitLink())
    @test_throws ArgumentError fit_delta_gamma_gllvm_va(Yg; K = K, link = G.IdentityLink())
    # the canonical link is still accepted
    @test fit_beta_gllvm_va(Yb; K = K, link = G.LogitLink()) isa BetaFit
    @test fit_gamma_gllvm_va(Yg; K = K, link = G.LogLink()) isa GammaFit
end

# Default-link fits must not move. These reference values were produced on main BEFORE the
# link fix (same simulation, same call); the default-link code path is unchanged.
@testset "default-link fits are unchanged" begin
    K = 1
    Yb, _, _ = sim_beta(G.LogitLink(); n = 150, seed = 61)
    fb = fit_beta_gllvm(Yb; K = K)
    @test isapprox(fb.loglik, 713.1872485087047; atol = 1e-5)
    @test isapprox(fb.φ, 25.216934087488305; rtol = 1e-5)
    @test fit_beta_gllvm(Yb; K = K, link = G.LogitLink()).loglik == fb.loglik

    Yg, _, _ = sim_gamma(G.LogLink(); n = 150, seed = 62, β = 1.0 .+ collect(range(-0.5, 0.5; length = 6)))
    fg = fit_gamma_gllvm(Yg; K = K)
    @test isapprox(fg.loglik, -1326.0092575393442; atol = 1e-5)
    @test isapprox(fg.α, 8.323843742421477; rtol = 1e-5)
    @test fit_gamma_gllvm(Yg; K = K, link = G.LogLink()).loglik == fg.loglik
end

end # module
