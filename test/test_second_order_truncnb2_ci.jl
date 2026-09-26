# TruncatedNegBin2Fit / TruncatedNegBin2PerTraitFit Wald wiring (second-order holdout
# clearance; per-trait added under pr-493's review, see #499 for the open per-trait
# local-optimum stall this test does not attempt to fix).

using Test
using Random
using GLLVModels
using Distributions: NegativeBinomial

# Same DGP as tools/truncnb2_parity_data_draw.jl (NATIVE-12 design, p=5, K=1,
# beta=log([4,5,3.5,4.5,4]), Lambda=0.2*parity_loadings_p5k2()), so `seed` alone
# reproduces the exact draws the review measured on Julia 1.10.
_pertrait_loadings_p5k2() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]

function _draw_truncnb2_pertrait(seed::Integer, r_true::Real, n::Integer)
    p, K = 5, 1
    Random.seed!(seed)
    β = log.([4.0, 5.0, 3.5, 4.5, 4.0])
    Λ = 0.2 .* _pertrait_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -3.0, 3.5))
        while true
            v = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
            if v >= 1
                Y[t, s] = v
                break
            end
        end
    end
    return Y
end

@testset "TruncatedNegBin2Fit second-order Wald CI" begin
    Random.seed!(73)
    p, K, n = 5, 1, 120
    β = 1.0 .+ 0.15 .* randn(p)
    Λ = 0.3 .* ones(p, K)
    r = 4.0
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β .+ Λ * randn(K)
        for t in 1:p
            μ = max(exp(η[t]), 1e-12)
            # reject-sample zeros from NB2
            y = 0
            while y < 1
                y = rand(NegativeBinomial(r, r / (r + μ)))
            end
            Y[t, s] = y
        end
    end
    fit = fit_truncated_nbinom2_gllvm(Y; K = K)
    @test fit isa TruncatedNegBin2Fit
    @test fit.converged

    ci = confint(fit, Y; method = :wald)
    @test ci.method === :wald
    @test "r" in ci.term
    @test "beta[1]" in ci.term
    r_row = findfirst(==("r"), ci.term)
    @test r_row !== nothing
    @test ci.estimate[r_row] ≈ fit.r atol = 1e-8
    @test isfinite(ci.se[r_row]) && ci.se[r_row] > 0
    if isfinite(ci.lower[r_row]) && isfinite(ci.upper[r_row])
        @test 0 < ci.lower[r_row] < ci.estimate[r_row] < ci.upper[r_row]
    end

    w = confint(fit, Y; method = :wald, parm = "beta[1]")
    @test w.term == ["beta[1]"]
    @test w.estimate[1] ≈ fit.β[1] atol = 1e-8
    @test isfinite(w.se[1])
end

@testset "TruncatedNegBin2PerTraitFit second-order Wald CI" begin
    # Basic confint call (pr-493 BLOCKING 1): the new user-reachable route
    # (`confint(fit_truncated_nbinom2_gllvm_pertrait(Y; K), Y)`) used to throw a
    # MethodError before #493 added TruncatedNegBin2PerTraitFit to _CIFit.
    Y_interior = _draw_truncnb2_pertrait(61, 4.0, 150)
    fit_interior = fit_truncated_nbinom2_gllvm_pertrait(Y_interior; K = 1)
    @test fit_interior isa TruncatedNegBin2PerTraitFit
    @test fit_interior.converged

    ci_interior = confint(fit_interior, Y_interior; method = :wald)
    @test ci_interior.method === :wald
    @test all(in(ci_interior.term), ["r[$t]" for t in 1:5])
    @test "beta[1]" in ci_interior.term
    r1 = findfirst(==("r[1]"), ci_interior.term)
    @test r1 !== nothing
    @test ci_interior.estimate[r1] ≈ fit_interior.r[1] atol = 1e-8

    w = confint(fit_interior, Y_interior; method = :wald, parm = "beta[1]")
    @test w.term == ["beta[1]"]
    @test w.estimate[1] ≈ fit_interior.β[1] atol = 1e-8
    @test isfinite(w.se[1])

    # Regime 1 (interior, pr-493 review "seed 61/n150"): every trait's r sits well
    # inside the Poisson-limit boundary on both engines, so the joint Wald Hessian
    # is positive definite and every bound is finite.
    @test ci_interior.pd_hessian == true
    @test isempty(ci_interior.boundary_terms)
    @test all(isfinite, ci_interior.se)
    @test all(isfinite, ci_interior.lower)
    @test all(isfinite, ci_interior.upper)

    # Regime 2 (boundary, pr-493 review "seed 62/n150"): trait 1's r sits at the
    # Poisson limit (~1e10). Before the fix, the per-trait adapter passed the
    # 6-arg `_FamilyCI(...)`, so `boundary` was all-false, `pd_hessian` came back
    # `true`, and r[1] got a finite-looking but meaningless SE (~1860 on the log
    # scale) with an `Inf` upper bound — the same T14 F1 failure the grouped
    # NB2/NB1/Beta/Gamma adapters already guard against. After the fix, r[1] is
    # conditioned out of the joint Hessian like those adapters: `pd_hessian` is
    # `false` and `boundary_terms` names it (the mirror-image local-optimum stall
    # on some other seeds, e.g. 65/n150, is a fitter defect tracked in #499, not
    # fixed here).
    Y_boundary = _draw_truncnb2_pertrait(62, 4.0, 150)
    fit_boundary = fit_truncated_nbinom2_gllvm_pertrait(Y_boundary; K = 1)
    @test fit_boundary isa TruncatedNegBin2PerTraitFit
    @test fit_boundary.r[1] > 1e6

    ci_boundary = confint(fit_boundary, Y_boundary; method = :wald)
    @test ci_boundary.pd_hessian == false
    @test "r[1]" in ci_boundary.boundary_terms
end

@testset "R paired truncated_nbinom2 cell (live Δ, beta[] block)" begin
    if get(ENV, "GLLVM_PARITY_TESTS", "0") != "1"
        @test_skip "set GLLVM_PARITY_TESTS=1 with R + gllvmTMB for live second-order Δ"
    else
        using RCall
        include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "common.jl"))
        include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "cells.jl"))
        d = run_one_cell("truncated_nbinom2")
        @test get(d, "skip_reason", nothing) === nothing
        @test get(d, "parameterisation_gap", true) == false
        se_rel = d["se_max_relative_delta"]
        @test se_rel !== nothing && isfinite(se_rel)

        # Contract §4 bars (second-order-parity-contract.md:144-146), each-own-optimum
        # column, scaled by cond(H)_R/1e3 above 1e3 (D2). Previously only checked for
        # finiteness, not the FAIL-to-PASS move the fixture re-point earned (pr-493
        # review, SHOULD-FIX "gate the win").
        cond_scale = max(1.0, d["r_condition_number"] / 1e3)
        @test se_rel <= 1e-2 * cond_scale
        vcov_rel = d["vcov_frobenius_relative_delta"]
        @test vcov_rel !== nothing && isfinite(vcov_rel)
        @test vcov_rel <= 1e-2 * cond_scale
        ci_rel = d["ci_endpoint_rel_half_width"]
        @test ci_rel !== nothing && isfinite(ci_rel)
        @test ci_rel <= 5e-2
    end
end
