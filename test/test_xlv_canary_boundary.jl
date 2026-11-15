using GLLVModels, Test
include(joinpath(@__DIR__, "xlv_canary_boundary.jl"))

# Gate logic of the boundary-stop acceptance used by the NB and binomial xlv canaries (refs #1048).
# The reference fit and gradient must be reachable only when the stop is the failure-value stop AND
# sigma2_phy is at the floor; everything else is decided without calling the plain-model objective.
@testset "xlv canary boundary-stop acceptance" begin
    calls = Ref(0)
    quad = th -> (calls[] += 1; 0.5 * sum(abs2, th .- 0.25))      # plain nll, minimum 0 at 0.25
    start = fill(0.25, 3)
    mkfit(; converged = false, reason = :gradient_at_failure_value, s2 = 8.4e-308,
            ll = 0.0, theta = vcat(fill(0.25, 3), log(1e-307))) =
        (converged = converged, stop_reason = reason, sigma2_phy = s2, loglik = ll, theta_packed = theta)
    ok(f; kw...) = _xlv_converged_or_boundary(f; nll_plain = quad, theta_start_plain = start, kw...)

    # a converged fit is accepted without touching the plain objective
    calls[] = 0
    @test ok(mkfit(converged = true, reason = :converged, s2 = 0.3))
    @test calls[] == 0

    # boundary stop with all five conditions met is accepted
    calls[] = 0
    @test ok(mkfit())
    @test calls[] > 0

    # the branch is not reachable with the variance off the floor: not accepted, and the plain
    # objective is never evaluated
    calls[] = 0
    @test !ok(mkfit(s2 = 0.3))
    @test !ok(mkfit(s2 = 1e-8))
    @test calls[] == 0

    # nor with another stop reason
    calls[] = 0
    for r in (:gradient_not_small, :not_converged, :objective_failure)
        @test !ok(mkfit(reason = r))
    end
    @test calls[] == 0

    # each remaining condition fails on its own
    @test !ok(mkfit(ll = NaN))                                       # (iii) not finite
    @test !ok(mkfit(ll = -5.0))                                      # (iii) below the start value
    @test !ok(mkfit(theta = vcat([0.25, NaN, 0.25], log(1e-307))))   # (iv) non-finite parameter
    @test !ok(mkfit(theta = vcat(fill(3.0, 3), log(1e-307)), ll = 0.0))  # (iv) large gradient
    @test !ok(mkfit(ll = 0.1))                                       # (v) differs from the reference fit
end
