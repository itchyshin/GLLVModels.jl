# #1006: fit_zip_gllvm (hessian = :observed, the default) reports converged = false on a
# large share of simulated ZIP datasets (5 of 10 on Simulator 2 cell 10, p = 10, n = 150, K = 2).
#
# Reproduction at test size: p = 6, n = 50, K = 1, true ZI probability 0.2 (simulated once;
# literal data so the test does not depend on an RNG stream). The observed-curvature Laplace
# surface keeps rising toward a singular site precision: the default fit ends on the
# breakdown guard (smallest site eigenvalue = ZI_LAPLACE_EIGMIN_FLOOR), at a logLik above the
# value the same surface takes at the interior Fisher optimum. So this is a property of the
# objective, not of the optimiser; whether the default should handle it is an open design
# question (see the PR for #1006). The `@test_broken` below records the unresolved behaviour
# and will start to error (as an unexpected pass) once a fix lands.
using Test, GLLVModels, Logging
const _G1006 = GLLVModels

const _Y1006 = Float64[
    6 2 3 1 4 0 1 2 1 0 0 1 4 0 3 3 0 3 1 2 0 0 2 4 2 0 0 2 1 0 4 1 3 1 2 1 2 3 2 0 0 0 3 1 0 0 2 0 1 3;
    28 8 0 6 5 5 3 1 6 4 6 10 16 6 0 5 25 7 4 3 3 7 49 1 2 9 8 5 4 3 14 0 11 0 25 3 2 0 0 28 4 11 4 1 20 5 11 6 6 0;
    0 3 2 5 4 3 0 0 3 3 7 0 3 2 1 2 1 4 2 6 1 0 1 6 3 4 2 2 4 3 8 0 1 3 3 5 5 3 1 0 3 0 0 4 0 3 0 4 1 3;
    0 2 0 4 4 2 3 3 6 1 6 2 2 0 2 0 0 2 2 6 0 3 2 1 3 3 3 9 2 0 0 4 0 0 2 2 2 0 2 1 3 0 3 5 0 2 2 4 1 2;
    11 3 0 4 1 2 2 0 3 0 1 2 7 3 11 1 9 3 0 0 3 2 29 0 1 3 1 2 0 0 7 1 0 0 0 0 1 0 1 5 1 5 4 0 4 0 3 0 4 9;
    3 5 0 5 0 2 4 4 0 7 4 0 1 1 0 5 0 1 8 4 4 3 0 2 6 3 1 3 0 7 2 0 1 5 2 0 0 0 0 0 2 0 1 3 0 2 3 2 1 0]

@testset "#1006 ZIP observed-curvature Laplace: breakdown-wall optimum" begin
    obs = with_logger(NullLogger()) do; fit_zip_gllvm(_Y1006; K = 1) end
    fis = fit_zip_gllvm(_Y1006; K = 1, hessian = :fisher)
    floor_ = _G1006.ZI_LAPLACE_EIGMIN_FLOOR

    @testset "default fit ends on the breakdown guard and is reported not converged" begin
        @test !obs.converged
        me = _G1006._zip_min_site_eigen(_Y1006, obs.Λc, obs.βz, obs.βc, nothing, 100, 1e-9)
        @test me < 1.1 * floor_
    end

    @testset "no interior maximum: the wall point beats the Fisher optimum on the observed surface" begin
        at_fisher = _G1006.zip_marginal_loglik_laplace(_Y1006, fis.Λc, fis.βz, fis.βc; hessian = :observed)
        @test obs.loglik > at_fisher + 0.5
    end

    @testset "hessian = :fisher converges, away from the wall" begin
        @test fis.converged
        me = _G1006._zip_min_site_eigen(_Y1006, fis.Λc, fis.βz, fis.βc, nothing, 100, 1e-9)
        @test me > 1.1 * floor_
    end

    # Unresolved (#1006): the default fit should converge on data like these.
    @test_broken obs.converged
end
