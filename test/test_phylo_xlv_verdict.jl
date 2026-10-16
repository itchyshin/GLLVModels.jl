using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Binomial, NegativeBinomial, Gamma
import Optim

# #505 (the #485 class) for the phylogenetic xlv point fitters (binomial, NB2, Gamma).
# `Optim.converged` also fires on a zero-length line-search step, so these fitters used to
# report `converged = true` at finite-difference gradients of order 1-100. The assertions
# are RELATIONS over several seeds, never one seed's outcome: a fit reported converged has
# a small gradient. CI draws different data per Julia version from the same seed, and the
# optimiser path differs between Linux and macOS.

const _XLV_V_TREE = "(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);"

# Central finite-difference gradient (inf-norm) of `f` at `x`.
function _xlv_v_gnorm(f, x; h = 1e-6)
    g = 0.0
    for i in eachindex(x)
        e = zeros(length(x)); e[i] = h
        g = max(g, abs(f(x + e) - f(x - e)) / (2h))
    end
    return g
end

# The reporting bound: GLLVModels' own stall rule, evaluated here at the returned optimum.
_xlv_v_ok(g, nll) = isfinite(g) && g <= max(1e-6, GLLVModels._PHYLO_STALL_RTOL * max(1.0, abs(nll)))

function _xlv_v_data(family, rng, n)
    p = 6
    X = reshape(collect(range(-1.0, 1.0; length = n)), n, 1)
    beta = family === :gamma ? log.(fill(3.0, p)) .+ 0.2 .* randn(rng, p) : 0.3 .* randn(rng, p)
    Lam = reshape(0.4 .* randn(rng, p), p, 1)
    Z = X * [0.5] .+ 0.4 .* randn(rng, n)
    eta = beta .+ Lam * Z'
    if family === :binomial
        N = fill(20, p, n)
        Y = [rand(rng, Binomial(20, 1 / (1 + exp(-eta[t, s])))) for t in 1:p, s in 1:n]
        return (; Y, N, X, beta)
    elseif family === :nb
        Y = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(eta[t, s])))) for t in 1:p, s in 1:n]
        return (; Y, N = nothing, X, beta)
    else
        Y = [rand(rng, Gamma(2.5, exp(eta[t, s]) / 2.5)) for t in 1:p, s in 1:n]
        return (; Y, N = nothing, X, beta)
    end
end

function _xlv_v_fit(family, d, phy; sigma2_init)
    p = phy.n_leaves
    kw = (; K = 1, X_lv = d.X, beta_init = d.beta, sigma2_phy_init = sigma2_init,
          iterations = 200, g_tol = 1e-6)
    if family === :binomial
        fit = GLLVModels._fit_phylo_binomial_xlv(d.Y, d.N, phy; kw...)
        nll = th -> GLLVModels._phylo_binomial_xlv_nll_packed(th, d.Y, d.N, p, 1, phy, d.X;
                                                              q_lv = 1, maxiter = 80, tol = 1e-9)
    elseif family === :nb
        fit = GLLVModels._fit_phylo_nb_xlv(d.Y, phy; kw...)
        nll = th -> GLLVModels._phylo_nb_xlv_nll_packed(th, d.Y, p, 1, phy, d.X;
                                                        q_lv = 1, maxiter = 80, tol = 1e-9)
    else
        fit = GLLVModels._fit_phylo_gamma_xlv(d.Y, phy; kw...)
        nll = th -> GLLVModels._phylo_gamma_xlv_nll_packed(th, d.Y, p, 1, phy, d.X;
                                                           q_lv = 1, maxiter = 80, tol = 1e-9)
    end
    return fit, nll
end

@testset "phylo xlv fitters: converged implies a small gradient ($family)" for family in
        (:binomial, :nb, :gamma)
    phy = GLLVModels.augmented_phy(_XLV_V_TREE)
    n_checked = 0
    for seed in 1:6, s2 in (0.5, 1e8)
        d = _xlv_v_data(family, MersenneTwister(1000 * seed), 30)
        fit, nll = _xlv_v_fit(family, d, phy; sigma2_init = s2)
        g = _xlv_v_gnorm(nll, fit.theta_packed)
        n_checked += 1
        # The relation under test: a fit reported converged has a small gradient.
        fit.converged && @test _xlv_v_ok(g, -fit.loglik)
        # The new field, kept separate from the relation so the relation is what fails on base.
        if hasproperty(fit, :stop_reason)
            @test fit.converged == (fit.stop_reason === :converged)
        end
    end
    @test n_checked == 12
end

@testset "phylo xlv fitters expose a stop_reason" begin
    phy = GLLVModels.augmented_phy(_XLV_V_TREE)
    d = _xlv_v_data(:binomial, MersenneTwister(1000), 30)
    fit, _ = _xlv_v_fit(:binomial, d, phy; sigma2_init = 0.5)
    @test fit.stop_reason in (:converged, :not_converged, :gradient_not_small,
                              :gradient_at_failure_value, :objective_failure)
end

@testset "_phylo_xlv_verdict: Optim stall on a nonsmooth objective" begin
    # A kink Optim cannot descend: the line search takes a zero-length step with a
    # gradient of magnitude 50. The raw flag may say converged; the verdict must not.
    f = x -> 50.0 * abs(x[1] - 0.3) + 1.0
    res = Optim.optimize(f, [5.0],
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
        Optim.Options(g_tol = 1e-6, iterations = 200); autodiff = :finite)
    v = GLLVModels._phylo_xlv_verdict(res, 1e-6)
    gres = Optim.g_residual(res)
    if gres > 1.0
        @test !v.converged
        @test v.reason === :gradient_not_small
    end
    # An objective sitting at the failure value is reported as its own reason.
    plateau = Optim.optimize(x -> 1e12, [0.0], Optim.LBFGS(), Optim.Options(iterations = 5);
                             autodiff = :finite)
    vp = GLLVModels._phylo_xlv_verdict(plateau, 1e-6)
    @test !vp.converged
    @test vp.loglik == -Inf
    @test vp.reason === :objective_failure
end
