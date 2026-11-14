# Regression for the phylo x Gamma x predictor-informed LV Laplace mode search (#1048).
#
# The Newton mode search ends on `step * max|delta| < tol`. With the Gamma
# observed-information path the step stalls near 1e-7 (the log-posterior changes by
# about 1e-14 per step, below rounding at |q| ~ 300), the `q1 >= q0` line search then
# rejects every trial, and the search used to report "no mode" -> marginal -Inf ->
# the Optim objective sentinel 1e12 at ONE finite-difference neighbour of the
# optimiser's iterate. LBFGS then saw a gradient of about 1e16 and stopped after a
# few iterations with `stop_reason = :gradient_at_failure_value`.
#
# `theta_bad` is that neighbour (a 6e-6 step in one Lambda coordinate) for the data
# below, recorded on Julia 1.13.3. The marginal at it must be finite and must agree
# with the marginal at the iterate, which is 6e-6 away. On a platform where the
# rounding falls differently the assertions hold trivially; they fail only if a
# rounding-level mode-search stall is again reported as failure.
using GLLVModels, Test, Random
using Distributions: Gamma

@testset "Phylo x Gamma xlv mode search: rounding-floor stall is convergence" begin
    Random.seed!(753)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 28
    K = 1
    q_lv = 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    beta = log.([3.0, 2.5, 2.8, 2.3, 2.7, 2.4])
    Lambda = reshape([0.18, -0.15, 0.16, -0.13, 0.11, -0.10], p, K)
    alpha_lv = reshape([0.40], q_lv, K)
    alpha_shape = 2.5
    epsilon = 0.05 .* randn(n, K)
    Z_truth = X_lv * alpha_lv + epsilon
    eta = beta .+ Lambda * transpose(Z_truth)
    Y = [rand(Gamma(alpha_shape, exp(eta[t, s]) / alpha_shape)) for t in 1:p, s in 1:n]

    theta_min = [1.0619271339444578, 0.9359183768482601, 0.9204248767527815,
                 0.9796796349331031, 0.9780045461263662, 0.8565203959475254,
                 0.08951997290949985, 0.35282512054693593, -0.49340445570601343,
                 0.19912048842563762, -0.02534142028745768, 0.21429996321626146,
                 -0.04775383544726281, 1.0527458796597025, -1.50472532326496]
    theta_bad = copy(theta_min)
    theta_bad[10] = 0.19912654388009002

    nll(theta) = GLLVModels._phylo_gamma_xlv_nll_packed(theta, Y, p, K, phy, X_lv;
                                                        q_lv = q_lv, maxiter = 120, tol = 1e-10)
    nll_min = nll(theta_min)
    nll_bad = nll(theta_bad)
    @test nll_min < 1e11
    @test nll_bad < 1e11
    @test isapprox(nll_bad, nll_min; atol = 1e-3)
end
