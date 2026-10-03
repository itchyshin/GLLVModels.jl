using GLLVModels, Test, Random, LinearAlgebra, Distributions, SpecialFunctions, StableRNGs
using StatsModels: @formula

# #149: the Julia closed-form Gaussian fit refuses n_sites < p, while R's Laplace
# engine has no such gate. Approved scope: the Laplace routes must accept n < p
# (they already did; nothing in src/ had to change for them). The closed-form
# Gaussian fitter replaces the n >= p guard by a rank rule (K below the rank of
# the data the fitter sees), as an ArgumentError, and so does every route built
# on it, Lognormal() included. This file pins both halves. It also records that the Gaussian LIKELIHOOD itself has no
# n >= p dependence: the guard belongs to the fitter (see the last testset).
#
# Every expected value below is computed outside the package: a hand-written
# Newton/Laplace reference for the Laplace objective, a dense MvNormal for the
# Gaussian likelihood, and loglik values pinned from the tree before this file.

# ---------------------------------------------------------------------------
# Deterministic data. Uniforms come from a StableRNG and the counts are their
# inverse CDF (`quantile`), so no sampler algorithm sits between the seed and
# the data: the same seed gives the same matrix on every Julia / Distributions.
# ---------------------------------------------------------------------------
_nltp_logistic(x) = inv(1 + exp(-x))

function _nltp_truth(seed, p, n, K; β0 = 0.6, scale = 0.6)
    rng = StableRNG(seed)
    Λ = scale .* randn(rng, p, K)
    β = β0 .+ 0.3 .* randn(rng, p)
    Z = randn(rng, K, n)
    η = [β[t] + dot(Λ[t, :], Z[:, i]) for t in 1:p, i in 1:n]
    return rng, Λ, β, η
end

function _nltp_poisson(seed, p, n, K)
    rng, Λ, β, η = _nltp_truth(seed, p, n, K)
    Y = [quantile(Distributions.Poisson(exp(η[t, i])), rand(rng)) for t in 1:p, i in 1:n]
    return Y, Λ, β
end

function _nltp_binary(seed, p, n, K)
    rng, Λ, β, η = _nltp_truth(seed, p, n, K; β0 = 0.0, scale = 0.7)
    Y = [rand(rng) < _nltp_logistic(η[t, i]) ? 1 : 0 for t in 1:p, i in 1:n]
    return Y, Λ, β
end

function _nltp_nb2(seed, p, n, K; r = 3.0)
    rng, Λ, β, η = _nltp_truth(seed, p, n, K)
    Y = [quantile(Distributions.NegativeBinomial(r, r / (r + exp(η[t, i]))), rand(rng))
         for t in 1:p, i in 1:n]
    return Y, Λ, β
end

# ---------------------------------------------------------------------------
# Independent Laplace reference (not the package's code). Site i contributes
#     ∫ exp(g_i(z)) N(z; 0, I_K) dz ≈ exp(g_i(z*) - z*'z*/2) / sqrt(det(I + Λ' W Λ)),
# with z* the mode of g_i(z) - z'z/2 found by plain Newton, g_i the conditional
# log-likelihood of the site's p cells at η = β + Λ z, and W the negative second
# derivative of g_i in η. Both families below use their canonical link, so the
# observed and the expected curvature coincide and the reference does not depend
# on the package's `hessian` choice.
# ---------------------------------------------------------------------------
function _nltp_ref_site(y, β, Λ, loglik_η, score_η, weight_η)
    K = size(Λ, 2)
    z = zeros(K)
    for _ in 1:500
        η = β .+ Λ * z
        g = Λ' * score_η.(y, η) .- z
        A = Λ' * Diagonal(weight_η.(η)) * Λ + I
        δ = A \ g
        z .+= δ
        norm(δ) < 1e-13 && break
    end
    η = β .+ Λ * z
    A = Λ' * Diagonal(weight_η.(η)) * Λ + I
    return sum(loglik_η.(y, η)) - dot(z, z) / 2 - logdet(A) / 2
end

# Poisson, log link: g = y η - e^η - log y!, score y - e^η, weight e^η.
_nltp_ref_poisson(Y, β, Λ) = sum(
    _nltp_ref_site(Float64.(Y[:, i]), β, Λ,
        (y, η) -> y * η - exp(η) - loggamma(y + 1),
        (y, η) -> y - exp(η),
        η -> exp(η)) for i in 1:size(Y, 2))

# Bernoulli, logit link: g = y η - log(1 + e^η), score y - σ(η), weight σ(η)(1 - σ(η)).
_nltp_ref_binary(Y, β, Λ) = sum(
    _nltp_ref_site(Float64.(Y[:, i]), β, Λ,
        (y, η) -> y * η - log1p(exp(η)),
        (y, η) -> y - _nltp_logistic(η),
        η -> _nltp_logistic(η) * (1 - _nltp_logistic(η))) for i in 1:size(Y, 2))

_nltp_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

const _NLTP_P = 12     # traits
const _NLTP_N = 8      # sites, so n < p

@testset "n < p (#149)" begin

@testset "#149 Poisson fits with n < p (p = 12, n = 8)" begin
    for K in (1, 2)
        Y, Λtrue, βtrue = _nltp_poisson(3001, _NLTP_P, _NLTP_N, K)
        @test size(Y, 2) < size(Y, 1)
        fit = fit_gllvm(Y; family = Poisson(), K = K)
        @test fit.converged isa Bool
        @test isfinite(fit.loglik)
        @test fit.loglik < 0
        @test size(fit.Λ) == (_NLTP_P, K)
        @test length(fit.β) == _NLTP_P
        # The reported value is the Laplace marginal at the returned parameters,
        # whether or not the optimiser declared convergence.
        @test fit.loglik ≈ _nltp_ref_poisson(Y, fit.β, fit.Λ) atol = 1e-6
        # A converged fit is at least as good as the generating parameters.
        fit.converged && @test fit.loglik ≥ _nltp_ref_poisson(Y, βtrue, Λtrue) - 1e-6
    end
    # Sites far below p still run (p = 12, n = 2).
    Y2, _, _ = _nltp_poisson(3006, _NLTP_P, 2, 1)
    fit2 = fit_gllvm(Y2; family = Poisson(), K = 1)
    @test isfinite(fit2.loglik)
    @test fit2.converged isa Bool
end

@testset "#149 formula route (site covariate) fits with n < p" begin
    Y, _, _ = _nltp_poisson(3007, _NLTP_P, _NLTP_N, 1)
    site = (temp = collect(range(-1.0, 1.0; length = _NLTP_N)),)
    fit = gllvm(@formula(y ~ 1 + temp), Y, site; family = Poisson(), K = 1)
    @test isfinite(fit.loglik)
    @test fit.converged isa Bool
end

@testset "#149 Binomial fits with n < p" begin
    Y, Λtrue, βtrue = _nltp_binary(3002, _NLTP_P, _NLTP_N, 1)
    @test size(Y, 2) < size(Y, 1)
    # With 8 sites the logit Laplace fit runs to the saturation region; that is
    # reported through `converged` / `saturation` (and a warning), not by an error.
    fit = _nltp_quiet(() -> fit_gllvm(Y; family = Binomial(), K = 1))
    @test fit.converged isa Bool
    @test isfinite(fit.loglik)
    @test fit.loglik ≤ 0                      # a probability of binary data
    @test size(fit.Λ) == (_NLTP_P, 1)
    @test fit.loglik ≥ _nltp_ref_binary(Y, βtrue, Λtrue) - 1e-6
end

@testset "#149 NB2 fits with n < p" begin
    Y, _, _ = _nltp_nb2(3003, _NLTP_P, _NLTP_N, 1)
    fit = _nltp_quiet(() -> fit_gllvm(Y; family = NegativeBinomial(), K = 1))
    @test fit.converged isa Bool
    @test isfinite(fit.loglik)
    @test fit.loglik < 0
end

@testset "#149 the Laplace objective has no n >= p dependence" begin
    # Sites are independent given (β, Λ), so the marginal of an n = 16 >= p = 12
    # data set is the sum of its two n = 8 < p halves, and any n < p subset is the
    # sum of its single sites. Each is also checked against the hand-written
    # reference above.
    Y, Λ, β = _nltp_poisson(3004, _NLTP_P, 16, 2)
    ll(Ysub) = GLLVModels.poisson_marginal_loglik_laplace(Ysub, Λ, β)
    @test ll(Y[:, 1:8]) + ll(Y[:, 9:16]) ≈ ll(Y) atol = 1e-9
    @test ll(Y[:, 1:8]) ≈ _nltp_ref_poisson(Y[:, 1:8], β, Λ) atol = 1e-9
    @test ll(Y[:, 9:16]) ≈ _nltp_ref_poisson(Y[:, 9:16], β, Λ) atol = 1e-9
    @test ll(Y) ≈ _nltp_ref_poisson(Y, β, Λ) atol = 1e-9
    cols = [2, 5, 11]                          # n = 3 << p
    @test ll(Y[:, cols]) ≈ sum(ll(Y[:, [c]]) for c in cols) atol = 1e-9
    @test ll(Y[:, [7]]) ≈ _nltp_ref_poisson(Y[:, [7]], β, Λ) atol = 1e-9   # n = 1

    Yb, Λb, βb = _nltp_binary(3005, _NLTP_P, 16, 2)
    llb(Ysub) = GLLVModels.binomial_marginal_loglik_laplace(
        Ysub, ones(Int, size(Ysub)), Λb, βb, GLLVModels.LogitLink())
    @test llb(Yb[:, 1:8]) + llb(Yb[:, 9:16]) ≈ llb(Yb) atol = 1e-9
    @test llb(Yb[:, 1:8]) ≈ _nltp_ref_binary(Yb[:, 1:8], βb, Λb) atol = 1e-9
    @test llb(Yb[:, 9:16]) ≈ _nltp_ref_binary(Yb[:, 9:16], βb, Λb) atol = 1e-9
    @test llb(Yb) ≈ _nltp_ref_binary(Yb, βb, Λb) atol = 1e-9
end

@testset "#149 fits with n >= p are unchanged (values pinned from the tree before this file)" begin
    # Regular, non-saturated fits with n >= p. The numbers below were produced by
    # the unmodified tree; the n < p work must not move them.
    Y, _, _ = _nltp_poisson(2001, 6, 20, 1)
    f = fit_gllvm(Y; family = Poisson(), K = 1)
    @test f.converged
    @test f.loglik ≈ -209.37091572283185 rtol = 1e-6
    @test sum(abs2, f.Λ) ≈ 1.9280626037432802 rtol = 1e-4
    @test sum(f.β) ≈ 3.002001156034836 rtol = 1e-4

    Yb, _, _ = _nltp_binary(2002, 6, 60, 1)
    fb = _nltp_quiet(() -> fit_gllvm(Yb; family = Binomial(), K = 1))
    @test fb.converged
    @test fb.loglik ≈ -243.33719644395075 rtol = 1e-6
    @test sum(abs2, fb.Λ) ≈ 2.5147584097795104 rtol = 1e-4
    @test isapprox(sum(fb.β), -0.027505335588489166; atol = 1e-4)

    Yn, _, _ = _nltp_nb2(2003, 6, 20, 1)
    fn = _nltp_quiet(() -> fit_gllvm(Yn; family = NegativeBinomial(), K = 1))
    @test fn.converged
    @test fn.loglik ≈ -228.77530606765785 rtol = 1e-6
end

@testset "#149 the Gaussian guard is a rank rule: K below the rank of the data" begin
    G = GLLVModels.fit_gaussian_gllvm
    dense(f, Y) = begin
        Σ = Symmetric(Matrix(f.pars.Λ) * Matrix(f.pars.Λ)' + f.pars.σ_eps^2 * I)
        sum(logpdf(MvNormal(zeros(size(Y, 1)), Σ), Y[:, i]) for i in 1:size(Y, 2))
    end

    # n < p with K below the rank (rank 3 here): the fit runs, on every route.
    Yg = randn(StableRNG(149), 5, 3)           # p = 5 > n = 3
    f = G(Yg; K = 1)
    @test isfinite(f.logLik)
    @test f.logLik ≈ dense(f, Yg) atol = 1e-8
    @test (fit_gllvm(Yg; family = Normal(), K = 1); true)
    @test isfinite(G(Yg; K = 2).logLik)
    bf = bridge_fit(; y = Yg, family = "gaussian", d = 1)
    @test bf isa NamedTuple && isfinite(bf.loglik)

    # K at the rank (3) with rank < p: refused, and the message names both.
    err = try G(Yg; K = 3); nothing catch e; e end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("rank = 3", msg)
    @test occursin("K = 3", msg)
    @test_throws ArgumentError G(Yg; K = 4)
    @test_throws ArgumentError fit_gllvm(Yg; family = Normal(), K = 3)
    @test_throws ArgumentError bridge_fit(; y = Yg, family = "gaussian", d = 3)
    mask = trues(5, 3); mask[2, 3] = false
    @test_throws ArgumentError G(Yg; K = 3, mask = mask)
    @test_throws ArgumentError G(Yg; K = 3, offset = zeros(5, 3))
    # The K-selection sweep over K = 1:2 stays below the rank and runs.
    @test (select_lv(Yg; family = Normal(), Kmax = 2); true)

    # n >= p but rank-deficient (trait 5 = trait 1 + trait 2): rank 4 < p = 5.
    Yr = randn(StableRNG(153), 5, 8)
    Yr[5, :] .= Yr[1, :] .+ Yr[2, :]
    @test_throws ArgumentError G(Yr; K = 4)
    msg = sprint(showerror, try G(Yr; K = 4); nothing catch e; e end)
    @test occursin("rank = 4", msg) && occursin("K = 4", msg)
    fr = G(Yr; K = 1)
    @test isfinite(fr.logLik)

    # Full rank at n = p: K = 1 fits (the old boundary), value is the dense marginal.
    Ye = randn(StableRNG(150), 5, 5)
    fe = G(Ye; K = 1)
    @test fe.logLik ≈ dense(fe, Ye) atol = 1e-8

    # Estimated intercepts centre the data first: n = p = 4 with one intercept
    # per trait leaves rank 3, so K = 3 is refused and K = 2 fits.
    Yc = randn(StableRNG(154), 4, 4)
    Xi = zeros(4, 4, 4); for t in 1:4; Xi[t, :, t] .= 1; end
    @test_throws ArgumentError G(Yc; K = 3, X = Xi)
    @test isfinite(G(Yc; K = 2, X = Xi).logLik)
    @test isfinite(G(Yc; K = 3).logLik)       # no intercepts: raw rank is 4 = p
end

@testset "#149 Lognormal reaches the same guard through the Gaussian fitter" begin
    # family = Lognormal() fits a Gaussian GLLVM to the centred log(Y), so the
    # rank rule applies to it as well: n < p now fits with K below the centred
    # rank, and K at that rank is refused. (Routing Lognormal onto the Laplace
    # path, as gllvmTMB does, is a separate change in src/families/lognormal.jl.)
    Yl = exp.(randn(StableRNG(151), 5, 3))     # p = 5 > n = 3, centred rank 2
    fl1 = fit_gllvm(Yl; family = Lognormal(), K = 1)
    @test isfinite(fl1.loglik)
    @test_throws ArgumentError fit_gllvm(Yl; family = Lognormal(), K = 2)
    @test_throws ArgumentError bridge_fit(; y = Yl, family = "lognormal", d = 2)
    err = try GLLVModels.fit_lognormal_gllvm(Yl; K = 2); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("rank = 2", sprint(showerror, err))

    # n >= p (p = 6, n = 8): the fit runs and its loglik is the y-scale lognormal
    # marginal at the returned parameters (dense MvNormal on log y, minus sum log y).
    Y6 = exp.(0.3 .+ 0.6 .* randn(StableRNG(152), 6, 8))
    fl = fit_gllvm(Y6; family = Lognormal(), K = 1)
    @test isfinite(fl.loglik)
    Σl = Symmetric(fl.Λ * fl.Λ' + fl.σ^2 * I)
    dense = sum(logpdf(MvNormal(fl.β, Σl), log.(Y6[:, i])) for i in 1:8) - sum(log, Y6)
    @test fl.loglik ≈ dense atol = 1e-8
end

@testset "#149 the Gaussian likelihood itself is exact at n < p" begin
    # The closed-form marginal itself is exact at n < p: compare with a dense MvNormal.
    rng = StableRNG(7)
    p, n, K, σ = _NLTP_P, _NLTP_N, 2, 0.5
    Λ = 0.7 .* randn(rng, p, K)
    Y = Λ * randn(rng, K, n) .+ σ .* randn(rng, p, n)
    Σ = Symmetric(Λ * Λ' + σ^2 * I)
    dense = sum(logpdf(MvNormal(zeros(p), Σ), Y[:, i]) for i in 1:n)
    @test GLLVModels.gaussian_marginal_loglik(Y, Λ, σ) ≈ dense atol = 1e-9
end

end
