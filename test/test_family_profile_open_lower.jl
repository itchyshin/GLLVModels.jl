using GLLVModels, Test, Distributions

# `_family_profile` on a log-scale parameter whose profile deviance levels off
# below the χ²₁ cutoff as the parameter goes to 0 (seen for the zero-truncated NB2
# dispersion r on the #581 fixture draw 104: D ≈ 2.35 as r → 0). The 95% profile
# interval is then open at 0, and the lower bound is reported as 0 rather than NaN.
# Synthetic adapters: θ = [a (linear), u = log s (log scale)], nll = a²/2 + g(u).

const _OPL = GLLVModels
_opl_ad(g) = _OPL._FamilyCI([0.0, 0.0], θ -> θ[1]^2 / 2 + g(θ[2]), ["a", "s"],
                            [:linear, :log], rng -> nothing, Y -> nothing)
const _OPL_CUT = quantile(Chisq(1), 0.95)

@testset "family profile: open lower end on a log-scale parameter reports 0" begin
    # D(u) = 2.35 (1 − eᵘ)²: 2.35 as u → −∞, crosses the cutoff only above û = 0.
    ci = _OPL._family_profile(_opl_ad(u -> 1.175 * (1 - exp(u))^2), [2], 0.95)
    @test ci.lower[1] == 0.0
    @test ci.upper[1] ≈ 1 + sqrt(_OPL_CUT / 2.35) rtol = 1e-3
    @test ci.status[1] === :profile
end

@testset "family profile: a finite lower bound on a log-scale parameter is unchanged" begin
    # D(u) = u²: bounds exp(±1.96).
    ci = _OPL._family_profile(_opl_ad(u -> u^2 / 2), [2], 0.95)
    @test ci.lower[1] ≈ exp(-sqrt(_OPL_CUT)) rtol = 1e-3
    @test ci.upper[1] ≈ exp(sqrt(_OPL_CUT)) rtol = 1e-3
end

@testset "family profile: open end is not claimed when small values cannot be evaluated" begin
    # Levels off at 2.35 but the objective is non-finite below u = -5 (as for a
    # failed refit), so there is no evidence the deviance stays below the cutoff.
    g(u) = u < -5 ? Inf : 1.175 * (1 - exp(u))^2
    ci = _OPL._family_profile(_opl_ad(g), [2], 0.95)
    @test isnan(ci.lower[1])
    @test ci.status[1] === :partial
end

@testset "family profile: open lower end is found without walking the lower search" begin
    # Floor first: when D at 1e-6 × the estimate is already below the cutoff, the
    # lower bracket search (a refit per expansion step, 583 s on the #581 draw 104)
    # is skipped. Record every u the objective sees; apart from the Wald Hessian's
    # stencil near û = 0, nothing may land between the floor and the estimate.
    us = Float64[]
    g(u) = (push!(us, u); 1.175 * (1 - exp(u))^2)
    ci = _OPL._family_profile(_opl_ad(g), [2], 0.95)
    @test ci.lower[1] == 0.0
    @test count(u -> log(1e-6) + 1 < u < -0.5, us) == 0
end
