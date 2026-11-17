using GLLVModels, Test, LinearAlgebra, ForwardDiff
using Distributions: Beta, logpdf

# refs #1125 (same cause as the NB case). The inner mode search of
# `_phylo_beta_xlv_marginal_loglik` is damped Newton with a Fisher step weight. Near the boundary of (0, 1) the Fisher curvature can fall below the observed one and the loop crawls.
# On the data below it did not reach the mode within the default maxiter = 80 and the
# evaluator returned -Inf. The step weight is now max(observed, Fisher) (as for NB, #1125,
# and grouped Tweedie, #1122); the log-det still uses the observed weight.

# Independent reference: dense leaf-scale Laplace with ForwardDiff observed information
# and step-halving Newton, written without any package mode-search code.
function _indep_laplace_beta_xlv_1125(Y, beta, Lambda, alpha_lv, par, sigma2, phy, X_lv)
    p = size(Y, 1)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Sigma_a = sigma2 .* (inv(Qc)[leaf_pos, leaf_pos])
    Pa = inv(Symmetric(Sigma_a))
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)[:, 1]
    function logpost(x)
        e = x[1]; a = x[2:end]
        eta = beta .+ mean_eta .+ Lambda[:, 1] .* e .+ a
        q = -0.5 * e^2 - 0.5 * dot(a, Pa * a)
        for t in 1:p
            mu = 1 / (1 + exp(-eta[t]))
            q += logpdf(Beta(mu * par, (1 - mu) * par), Y[t, 1])
        end
        return q
    end
    x = zeros(p + 1)
    for _ in 1:500
        g = ForwardDiff.gradient(logpost, x)
        H = -ForwardDiff.hessian(logpost, x)
        d = isposdef(Symmetric(H)) ? Symmetric(H) \ g : g
        s = 1.0
        q0 = logpost(x)
        while s > 1e-12 && !(logpost(x .+ s .* d) >= q0)
            s /= 2
        end
        x = x .+ s .* d
        norm(g) < 1e-11 && break
    end
    H = -ForwardDiff.hessian(logpost, x)
    @assert isposdef(Symmetric(H)) && norm(ForwardDiff.gradient(logpost, x)) < 1e-8
    return logpost(x) + 0.5 * logdet(Symmetric(Pa)) - 0.5 * logdet(Symmetric(H))
end

@testset "phylo Beta xlv mode search, step weight (#1125 siblings)" begin
    phy3 = GLLVModels.augmented_phy("((A:0.3,B:0.3):0.2,C:0.5);")
    beta = [0.2, -0.3, 0.1]; alpha_lv = reshape([0.3], 1, 1); X_lv = reshape([0.5], 1, 1)
    # (Y, Lambda, sigma2, phi). Each was -Inf at the default maxiter = 80 before the change.
    cases = [([0.938, 0.971, 0.997], [0.54, 0.06, -0.28], 0.74, 10.83),
             ([0.004, 0.051, 0.412], [-0.69, 0.7, -0.29], 0.11, 16.3)]
    for (Yv, Lv, s2, par) in cases
        Y = reshape(Yv, 3, 1); L1 = reshape(Lv, 3, 1)
        ref = _indep_laplace_beta_xlv_1125(Y, beta, L1, alpha_lv, par, s2, phy3, X_lv)
        v = GLLVModels._phylo_beta_xlv_marginal_loglik(Y, beta, L1, alpha_lv, par, s2,
                phy3, X_lv)
        @test isfinite(v)
        @test isapprox(v, ref; atol = 1e-6)
        # the converged value does not depend on maxiter
        v2 = GLLVModels._phylo_beta_xlv_marginal_loglik(Y, beta, L1, alpha_lv, par, s2,
                phy3, X_lv; maxiter = 400)
        @test isapprox(v2, v; atol = 1e-6)
    end
    # A mild case that already converged stays on the reference.
    Ymild = reshape([0.55, 0.4, 0.6], 3, 1); Lm = reshape([0.3, 0.2, 0.4], 3, 1)
    vm = GLLVModels._phylo_beta_xlv_marginal_loglik(Ymild, beta, Lm, alpha_lv, 6.0, 0.4,
            phy3, X_lv)
    @test isapprox(vm, _indep_laplace_beta_xlv_1125(Ymild, beta, Lm, alpha_lv, 6.0, 0.4, phy3, X_lv); atol = 1e-6)
end
