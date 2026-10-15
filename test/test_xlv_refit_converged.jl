# Issue #752: the beta and ordinal xlv constrained refits used to report
# `converged = true` whenever the constraint was met within 1e-3, even when
# Nelder-Mead stopped on the iteration cap. A toy smooth objective with a known
# minimum exercises the private refit directly; no fitted model is needed.
using GLLVModels, Test
const _G752 = GLLVModels

@testset "xlv constrained refit: converged reflects the refit (#752)" begin
    p, K, q = 6, 2, 2
    rr = _G752.rr_theta_len(p, K)
    weights = (1e2, 1e3, 1e4, 1e5, 1e6)

    toy_nll(c, p) = th -> sum(abs2, th .- c) +
                          5 * sum(abs2, th[1:p] .* th[(p + 1):(2p)]) +
                          sum(abs.(th))^2 / 50

    @testset "beta" begin
        n = p + q * K + rr + 2
        th0 = fill(0.3, n)
        nll = toy_nll(collect(range(-1, 1, length = n)), p)
        tgt = _G752._phylo_beta_xlv_effects_from_packed(th0, p, K, q)[1] + 0.4
        refit(it) = _G752._phylo_beta_xlv_constrained_refit(
            nll, th0, p, K, q, 1, tgt;
            profile_iterations = it, penalty_weights = weights)
        short = refit(30)
        long = refit(600)
        # The iteration cap is hit, so the refit must not claim convergence,
        # whatever the constraint error is.
        @test !short.converged
        @test isfinite(short.constraint_error)
        # The stalled refit is far from the minimum: this is the anti-conservative
        # deviance the flag has to warn about.
        @test short.nll > long.nll + 1
    end

    @testset "ordinal" begin
        C = 4
        n = q * K + rr + C
        th0 = fill(0.3, n)
        nll = toy_nll(collect(range(-1, 1, length = n)), p)
        tgt = _G752._phylo_ordinal_xlv_effects_from_packed(th0, p, K, q, C)[1] + 0.4
        refit(it) = _G752._phylo_ordinal_xlv_constrained_refit(
            nll, th0, p, K, q, C, 1, tgt;
            profile_iterations = it, penalty_weights = weights)
        short = refit(30)
        long = refit(600)
        @test !short.converged
        @test isfinite(short.constraint_error)
        @test short.nll > long.nll + 1
    end
end
