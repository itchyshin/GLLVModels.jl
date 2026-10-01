using GLLVModels, Test, Random, LinearAlgebra, Distributions

# The covariate fitters (`fit_gllvm_cov`, `fit_nb_gllvm_grouped_cov`) drive dense BFGS with an
# exact gradient (implicit one-Newton-step + ForwardDiff). This gates it against a central
# finite difference of the very objective the fitter minimises, at a point away from the
# optimum, for every route that uses it. No tolerance is loosened to pass: the bound is
# 1e-6 on a gradient whose entries are O(1e1 to 3e2), and the measured gap is ~5e-9 to 1.3e-7 (central-difference noise at h = 1e-6).

const _CAG = GLLVModels
const _CAG_TOL = 1e-6

function _cag_data(family, p, n, q, seed)
    rng = MersenneTwister(seed)
    X = randn(rng, p, n, q)
    z = randn(rng, n, 2)
    Λ = [0.6 0.0; -0.4 0.5; 0.3 0.4; 0.5 -0.3; -0.2 0.6; 0.4 0.2][1:p, :]
    η = [0.3 * (t % 3 - 1) + 0.4 * X[t, s, 1] - 0.2 * X[t, s, min(2, q)] +
         dot(Λ[t, :], z[s, :]) for t in 1:p, s in 1:n]
    Y = if family isa Poisson
        [rand(rng, Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    elseif family isa Binomial
        [rand(rng) < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]
    elseif family isa NegativeBinomial
        [rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + exp(η[t, s])))) for t in 1:p, s in 1:n]
    elseif family isa Beta
        [clamp(rand(rng, Distributions.Beta(4 / (1 + exp(-η[t, s])), 4 / (1 + exp(η[t, s])))),
                1e-3, 1 - 1e-3) for t in 1:p, s in 1:n]
    elseif family isa Gamma
        [rand(rng, Distributions.Gamma(2.0, exp(η[t, s]) / 2.0)) for t in 1:p, s in 1:n]
    else # Exponential
        [rand(rng, Distributions.Exponential(exp(η[t, s]))) for t in 1:p, s in 1:n]
    end
    return X, Y
end

function _cag_fd(f, θ; h = 1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        θp = copy(θ); θm = copy(θ); θp[i] += h; θm[i] -= h
        g[i] = (f(θp) - f(θm)) / (2h)
    end
    return g
end

@testset "covariate fitters: exact gradient vs central finite difference" begin
    p, n, q, K = 6, 40, 2, 2
    rr = _CAG.rr_theta_len(p, K)

    for (name, fam) in (("Poisson", Poisson()), ("Binomial", Binomial()),
                        ("NegativeBinomial", NegativeBinomial(3.0, 0.5)),
                        ("Beta", Beta(4.0, 1.0)), ("Gamma", Gamma(2.0, 1.0)),
                        ("Exponential", Exponential(1.0)))
        @testset "fit_gllvm_cov $name" begin
            X, Y = _cag_data(fam, p, n, q, 11)
            lk = _CAG._cov_default_link(fam)
            hd = _CAG._cov_has_disp(fam)
            Nm = fill(1, p, n)
            rng = MersenneTwister(5)
            θ = vcat(0.1 .* randn(rng, p), 0.2 .* randn(rng, q), 0.3 .* randn(rng, rr),
                     hd ? [log(2.5)] : Float64[])
            Λ(θ) = _CAG.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
            negll(θ) = -_CAG._marginal_loglik_offset(
                _CAG._cov_family(fam, hd ? exp(θ[p + q + rr + 1]) : NaN), Y, Nm, Λ(θ),
                θ[1:p], _CAG._build_offset(X, θ[(p + 1):(p + q)]), lk;
                maxiter = 100, tol = 1e-12)
            g = _CAG._cov_negll_grad(fam, Y, Nm, X, θ, p, q, K, rr, hd, lk, nothing, 100, 1e-12)
            @test g !== nothing
            d = maximum(abs.(g .- _cag_fd(negll, θ)))
            @test d < _CAG_TOL
            get(ENV, "COV_GRAD_VERBOSE", "") == "1" && println("  max|g-fd| $name = ", d, "  max|g| = ", maximum(abs.(g)))
        end
    end

    @testset "fit_gllvm_cov masked Poisson" begin
        fam = Poisson()
        X, Y = _cag_data(fam, p, n, q, 12)
        msk = trues(p, n); msk[2, 3] = false; msk[5, 17] = false; msk[1, 30] = false
        Nm = fill(1, p, n)
        rng = MersenneTwister(6)
        θ = vcat(0.1 .* randn(rng, p), 0.2 .* randn(rng, q), 0.3 .* randn(rng, rr))
        negll(θ) = -_CAG._marginal_loglik_offset(
            fam, Y, Nm, _CAG.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K), θ[1:p],
            _CAG._build_offset(X, θ[(p + 1):(p + q)]), LogLink();
            mask = msk, maxiter = 100, tol = 1e-12)
        g = _CAG._cov_negll_grad(fam, Y, Nm, X, θ, p, q, K, rr, false, LogLink(), msk, 100, 1e-12)
        @test g !== nothing
        d = maximum(abs.(g .- _cag_fd(negll, θ)))
        @test d < _CAG_TOL
        get(ENV, "COV_GRAD_VERBOSE", "") == "1" && println("  max|g-fd| masked Poisson = ", d, "  max|g| = ", maximum(abs.(g)))
    end

    @testset "fit_nb_gllvm_grouped_cov (per-group NB2 dispersion)" begin
        X, Y = _cag_data(NegativeBinomial(3.0, 0.5), p, n, q, 13)
        group = [1, 1, 2, 2, 3, 3]; G = 3
        gidx = group
        for hess in (:observed, :fisher)
            rng = MersenneTwister(7)
            θ = vcat(0.1 .* randn(rng, p), 0.2 .* randn(rng, q), 0.3 .* randn(rng, rr),
                     log.([2.0, 3.0, 5.0]))
            negll(θ) = begin
                rg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
                -_CAG.nb_grouped_marginal_loglik_laplace(
                    Y, _CAG.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K), θ[1:p],
                    [rg[gidx[t]] for t in 1:p]; link = LogLink(), mask = nothing,
                    offset = _CAG._build_offset(X, θ[(p + 1):(p + q)]), hessian = hess,
                    maxiter = 100, tol = 1e-12)
            end
            g = _CAG._nb_grouped_cov_negll_grad(Y, X, θ, p, q, K, rr, G, gidx, LogLink(),
                                                nothing, hess, 100, 1e-12)
            @test g !== nothing
            d = maximum(abs.(g .- _cag_fd(negll, θ)))
            @test d < _CAG_TOL
            get(ENV, "COV_GRAD_VERBOSE", "") == "1" && println("  max|g-fd| grouped NB2 $hess = ", d, "  max|g| = ", maximum(abs.(g)))
        end
    end
end
