# Internals of the iSDM Laplace kernel (src/families/isdm_laplace.jl) and fitter
# (src/families/isdm_fit.jl) that no other test names: the log-scale cloglog
# probability, the Fisher weight of the mode search, the retrying cell mode, the
# link-scale pseudodata for the warm start, and the once-only experimental
# notice. Each is checked against an independent reference.
using Test, GLLVModels, Distributions, ForwardDiff, LinearAlgebra

include(joinpath(@__DIR__, "fixtures", "isdm", "isdm_fixture_io.jl"))

# Reference log(1 - exp(-exp(eta))) in 400-bit BigFloat; the precision absorbs the
# cancellation in the left tail.
_ref_log_cloglog_p(eta) = Float64(setprecision(BigFloat, 400) do
    log(1 - exp(-exp(BigFloat(eta))))
end)

@testset "iSDM cloglog internals" begin

@testset "_isdm_log_cloglog_p matches a BigFloat reference" begin
    grid = collect(-40.0:0.5:5.0)
    vals = GLLVModels._isdm_log_cloglog_p.(grid)
    @test all(isfinite, vals)
    relerr(η) = abs(GLLVModels._isdm_log_cloglog_p(η) - _ref_log_cloglog_p(η)) /
                abs(_ref_log_cloglog_p(η))
    # left series branch (eta <= -20) and the well-conditioned middle
    @test all(relerr(η) < 1e-10 for η in -40.0:0.5:-20.0)
    @test all(relerr(η) < 1e-10 for η in -17.0:0.5:2.0)
    # FINDING (measured): just above the -20 cut the middle branch evaluates
    # log(1 - exp(-lambda)) with lambda ~ 1e-9 and loses digits to cancellation
    # (the cut is a copy of R's gll_log_cloglog_p). Worst relative error on
    # eta in [-19.5, -17.5] is 5.35e-10 (absolute 1.04e-8 at eta = -19.5), above
    # the 1e-10 target; src/ is out of scope for this test-only change.
    @test_broken all(relerr(η) < 1e-10 for η in -19.5:0.5:-17.5)
    @test all(relerr(η) < 1e-6 for η in -19.5:0.5:-17.5)
    # right of eta = 2 the value is ~ -exp(-exp(eta)) <= 6e-4 and tends to 0; the
    # relative error is meaningless there (the reference is 1e-65 at eta = 5), so
    # bound the absolute error instead (measured worst 5.2e-17).
    @test all(abs(GLLVModels._isdm_log_cloglog_p(η) - _ref_log_cloglog_p(η)) < 1e-15
              for η in 2.5:0.5:5.0)
    # strictly increasing while log(1 - exp(-exp(eta))) is representable (it
    # rounds to exactly 0.0 from eta ~ 4.5, a plateau), non-decreasing beyond
    inc = grid .<= 4.0
    @test all(diff(vals[inc]) .> 0)
    @test all(diff(vals) .>= 0)
    # the cut itself: the left branch (eta <= -20) is accurate; the first point
    # right of it shares the middle-branch cancellation above (rel 1.0e-9 measured).
    for η in (-20.0 - 1e-9, -20.0)
        @test GLLVModels._isdm_log_cloglog_p(η) ≈ _ref_log_cloglog_p(η) rtol = 1e-10
    end
    @test_broken GLLVModels._isdm_log_cloglog_p(-20.0 + 1e-9) ≈
                 _ref_log_cloglog_p(-20.0 + 1e-9) rtol = 1e-10
    # far right tail: log(1 - exp(-exp(eta))) is 0 at the 700 cut
    @test GLLVModels._isdm_log_cloglog_p(700.0) == 0.0
    @test GLLVModels._isdm_log_cloglog_p(720.0) == 0.0
end

@testset "_isdm_dbinom_cloglog is the Bernoulli-cloglog log density" begin
    for η in (-30.0, -3.0, 0.0, 1.5)
        @test GLLVModels._isdm_dbinom_cloglog(1.0, η) ≈ _ref_log_cloglog_p(η) rtol = 1e-10
        @test GLLVModels._isdm_dbinom_cloglog(0.0, η) ≈ -exp(η) rtol = 1e-12
    end
end

@testset "_isdm_cloglog_fisher is the expected information" begin
    for η in (-8.0, -3.0, -1.0, 0.0, 1.0, 2.0)
        # brute-force E over y in {0, 1} of the observed weight -d2/d eta2
        brute = sum(exp(GLLVModels._isdm_dbinom_cloglog(y, η)) *
                    GLLVModels._isdm_cloglog_obs_weight(y, η) for y in (0.0, 1.0))
        @test GLLVModels._isdm_cloglog_fisher(η) ≈ brute rtol = 1e-8
        # and the same by differentiating the log density directly
        d2(y) = -ForwardDiff.derivative(e -> ForwardDiff.derivative(
            x -> GLLVModels._isdm_dbinom_cloglog(y, x), e), η)
        p = 1 - exp(-exp(η))
        @test GLLVModels._isdm_cloglog_fisher(η) ≈ p * d2(1.0) + (1 - p) * d2(0.0) rtol = 1e-8
        # closed form: (d mu / d eta)^2 / (mu (1 - mu)) with d mu / d eta = exp(eta - exp(eta))
        @test GLLVModels._isdm_cloglog_fisher(η) ≈
              exp(2η - 2exp(η)) / (p * (1 - p)) rtol = 1e-8
    end
    @test all(GLLVModels._isdm_cloglog_fisher.(-6.0:1.0:3.0) .> 0)
end

# One cell of the door fixture's shape: 2 traits x (count, detection) rows.
_CELL = (y = [3.0, 2.0, 1.0, 0.0], fid = [2, 2, 1, 1], tr = [1, 2, 1, 2],
               eta0 = [0.2, -0.1, -0.5, -0.8], Λ = reshape([0.7, -0.4], 2, 1))

@testset "_isdm_cell_mode_retry" begin
    c = _CELL
    z0, ok0 = GLLVModels._isdm_cell_mode(c.y, c.fid, c.tr, c.eta0, c.Λ; tol = 1e-12)
    z1, ok1 = GLLVModels._isdm_cell_mode_retry(c.y, c.fid, c.tr, c.eta0, c.Λ; tol = 1e-12)
    @test ok0 && ok1
    @test z1 ≈ z0 atol = 1e-10
    lp(w) = GLLVModels._isdm_cell_logpost(c.y, c.fid, c.tr, c.eta0, c.Λ, w)
    @test norm(ForwardDiff.gradient(lp, z1)) <= 1e-6
    # a good start reaches the same mode
    zg, okg = GLLVModels._isdm_cell_mode_retry(c.y, c.fid, c.tr, c.eta0, c.Λ;
                                               tol = 1e-12, z_init = z0)
    @test okg
    @test zg ≈ z0 atol = 1e-10
    # a deliberately bad start (far in the tail of every row) recovers
    zb, okb = GLLVModels._isdm_cell_mode_retry(c.y, c.fid, c.tr, c.eta0, c.Λ;
                                               tol = 1e-12, z_init = [25.0])
    @test okb
    @test all(isfinite, zb)
    @test norm(ForwardDiff.gradient(lp, zb)) <= 1e-6
    @test zb ≈ z0 atol = 1e-6
    # the retry spends 20x the budget: with maxiter = 2 the first pass stops short
    # of the mode (converged = false), the retry reaches it.
    zf, okf = GLLVModels._isdm_cell_mode(c.y, c.fid, c.tr, c.eta0, c.Λ;
                                         maxiter = 2, tol = 1e-12, z_init = [25.0])
    @test !okf
    zr, okr = GLLVModels._isdm_cell_mode_retry(c.y, c.fid, c.tr, c.eta0, c.Λ;
                                               maxiter = 2, tol = 1e-12, z_init = [25.0])
    @test okr
    @test norm(ForwardDiff.gradient(lp, zr)) <= 1e-6
    @test zr ≈ z0 atol = 1e-6
end

@testset "_isdm_pseudo_eta is finite on all-zero and all-one detection rows" begin
    door = (cell_id = repeat(["c1", "c2"], 4),
            trait = repeat(repeat(["sp1", "sp2"], inner = 2), 2),
            isdm_source = repeat(["gbif", "survey"], inner = 4),
            isdm_gbif = Float64.(repeat(["gbif", "survey"], inner = 4) .== "gbif"),
            log_support = fill(0.1, 8),
            value = [3.0, 2, 4, 1, 1, 0, 1, 0])
    fam = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))
    f = :(value ~ 0 + trait + trait & isdm_gbif + offset(log_support) +
          latent(0 + trait | cell_id, d = 1, unique = FALSE))
    for (label, surv) in (("all zero", zeros(4)), ("all one", ones(4)), ("mixed", [1.0, 0, 0, 1]))
        v = copy(door.value); v[5:8] .= surv
        tab = isdm_table(f, merge(door, (value = v,)); family = fam)
        ηt = GLLVModels._isdm_pseudo_eta(tab)
        @test length(ηt) == length(tab.y)
        @test all(isfinite, ηt)
        det = tab.fid .== 1
        @test all(ηt[det] .+ tab.offset[det] .≈ log.(-log.(1 .- (tab.y[det] .+ 0.5) ./ 2)))
        cnt = tab.fid .== 2
        @test all(ηt[cnt] .+ tab.offset[cnt] .≈ log.(tab.y[cnt] .+ 0.5))
        # the warm start built from it is finite too
        b0, L0 = GLLVModels._isdm_start(tab, 1)
        @test all(isfinite, b0) && all(isfinite, L0)
    end
end

@testset "_isdm_experimental_notice fires once" begin
    was_shown = GLLVModels._ISDM_NOTICE_SHOWN[]
    GLLVModels._ISDM_NOTICE_SHOWN[] = false
    @test_logs (:info, r"integrated multi-source route is experimental") GLLVModels._isdm_experimental_notice()
    @test GLLVModels._ISDM_NOTICE_SHOWN[]
    @test_logs GLLVModels._isdm_experimental_notice()      # second call is silent
    GLLVModels._ISDM_NOTICE_SHOWN[] = was_shown
end

end
