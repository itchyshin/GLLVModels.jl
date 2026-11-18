using GLLVModels, Test, Random, LinearAlgebra, ForwardDiff
using Distributions: NegativeBinomial, logpdf

# refs #1125. `_phylo_nb_xlv_marginal_loglik` returned -Inf on one site, three leaves,
# with counts far above the fixed-effect means (Y = [3, 25, 120]). The mode search is
# damped Newton with a Fisher step weight; where y >> mu Fisher under-states the
# curvature, the iterates alternate around the mode and the gradient shrinks by only
# ~4% per iteration, so `maxiter = 200` ran out and the search reported failure. The
# step weight is now max(observed, Fisher) (as in the grouped Tweedie search, #1122).

# Independent reference: dense leaf-scale Laplace with ForwardDiff observed information
# and step-halving Newton, written without any package mode-search code.
function _indep_laplace_nb_xlv_1125(Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)
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
            mu = exp(eta[t])
            q += logpdf(NegativeBinomial(r, r / (r + mu)), Y[t, 1])
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

@testset "phylo NB xlv mode search, outlying counts (#1125)" begin
    phy3 = GLLVModels.augmented_phy("((A:0.3,B:0.3):0.2,C:0.5);")
    beta = log.([4.0, 3.0, 5.0]); alpha_lv = reshape([0.3], 1, 1); X_lv = reshape([0.5], 1, 1)
    L1 = reshape([0.3, 0.2, 0.4], 3, 1)

    # Case B of the issue: was -Inf at maxiter = 200 for tol 1e-12, 1e-10, 1e-9.
    YB = reshape([3, 25, 120], 3, 1)
    ref = _indep_laplace_nb_xlv_1125(YB, beta, L1, alpha_lv, 1.7, 0.4, phy3, X_lv)
    @test isapprox(ref, -21.99365; atol = 1e-5)
    for tol in (1e-12, 1e-10, 1e-9)
        v = GLLVModels._phylo_nb_xlv_marginal_loglik(YB, beta, L1, alpha_lv, 1.7, 0.4,
                phy3, X_lv; maxiter = 200, tol = tol)
        @test isfinite(v)
        @test isapprox(v, ref; atol = 1e-6)
    end

    # Neighbouring cases that already worked (A, C of the issue) stay on the reference.
    YA = reshape([50, 18, 160], 3, 1)
    vA = GLLVModels._phylo_nb_xlv_marginal_loglik(YA, beta, L1, alpha_lv, 1.7, 0.4,
            phy3, X_lv; maxiter = 200, tol = 1e-10)
    @test isapprox(vA, _indep_laplace_nb_xlv_1125(YA, beta, L1, alpha_lv, 1.7, 0.4, phy3, X_lv); atol = 1e-6)
    LC = reshape([0.8, -0.6, 0.5], 3, 1); YC = reshape([30, 2, 400], 3, 1)
    vC = GLLVModels._phylo_nb_xlv_marginal_loglik(YC, beta, LC, alpha_lv, 0.8, 0.8,
            phy3, X_lv; maxiter = 200, tol = 1e-10)
    @test isapprox(vC, _indep_laplace_nb_xlv_1125(YC, beta, LC, alpha_lv, 0.8, 0.8, phy3, X_lv); atol = 1e-6)

    # No-change check: the value at the test_phylo_nb_xlv.jl fixture (p = 6, n = 7,
    # seed 742) is unchanged by the step-weight change (recorded on main 12aa00216).
    # The two step weights stop at slightly different iterates, so the tolerance is the
    # mode-search stopping error at tol = 1e-10 (the shift observed is 3.5e-10).
    Random.seed!(742)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves; n = 7
    betaF = log.([4.2, 3.7, 4.5, 3.4, 4.0, 3.8])
    LambdaF = reshape(range(0.08, 0.32; length = p), p, 1)
    alphaF = reshape([0.30], 1, 1)
    XF = reshape(collect(range(-1.1, 1.1; length = n)), n, 1)
    r = 1.7
    eta = betaF .+ LambdaF * transpose(XF * alphaF)
    YF = [rand(NegativeBinomial(r, r / (r + exp(eta[t, s])))) for t in 1:p, s in 1:n]
    llF = GLLVModels._phylo_nb_xlv_marginal_loglik(YF, betaF, LambdaF, alphaF, r, 0.40,
            phy, XF; maxiter = 120, tol = 1e-10)
    @test isapprox(llF, -112.42439520345326; atol = 1e-9)
end
