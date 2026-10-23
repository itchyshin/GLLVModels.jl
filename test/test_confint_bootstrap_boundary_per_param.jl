using GLLVModels, Test, Random
const GM = GLLVModels

# A bootstrap refit that converged but ran ONE parameter to its upper boundary (the
# `upper_boundary` field, #542 option 3) is still a valid draw for every other parameter.
# It used to be dropped from every quantile; where a per-group dispersion sits at the limit
# on most draws (the truncated-NB2 per-trait case in #645: 12 of 12 refits), that left no
# usable replicate and made the β and Λ intervals NaN as well. Synthetic adapter, so the
# check does not depend on any fit or platform.
@testset "a boundary-flagged replicate still informs the unflagged parameters" begin
    m = 3
    names = ["a", "b", "log_r"]
    kinds = [:linear, :linear, :log]
    sim = rng -> randn(rng)
    # Every replicate: parameters 1 and 2 vary, parameter 3 is flagged at its upper limit.
    refit_flag3 = y -> (θ = [y, 2y, log(1e8)], converged = true, loglik = -1.0,
                        upper_boundary = [false, false, true])
    ad = GM._FamilyCI(zeros(m), θ -> 0.0, names, kinds, sim, refit_flag3)
    res = GM._family_bootstrap(ad, collect(1:m), 0.95, 40, 7, false)
    @test all(isfinite, res.lower[1:2]) && all(isfinite, res.upper[1:2])
    @test res.lower[1] < 0 < res.upper[1]
    @test res.upper[3] == Inf          # all usable replicates sat at the boundary
    @test isnan(res.lower[3])          # and none gave an interior draw for it

    # A non-converged flagged refit still informs nothing.
    refit_bad = y -> (θ = [y, 2y, log(1e8)], converged = false, loglik = -1.0,
                      upper_boundary = [false, false, true])
    ad_bad = GM._FamilyCI(zeros(m), θ -> 0.0, names, kinds, sim, refit_bad)
    res_bad = GM._family_bootstrap(ad_bad, collect(1:m), 0.95, 40, 7, false)
    @test all(isnan, res_bad.lower[1:2])

    # Mixed: half the replicates flag parameter 3, half are interior. The interior half
    # gives parameter 3 its draws, and every replicate gives parameters 1 and 2 theirs.
    refit_mix = let k = Ref(0)
        y -> (k[] += 1; isodd(k[]) ?
              (θ = [y, 2y, log(1e8)], converged = true, loglik = -1.0,
               upper_boundary = [false, false, true]) :
              (θ = [y, 2y, log(5.0) + 0.1y], converged = true, loglik = -1.0))
    end
    ad_mix = GM._FamilyCI(zeros(m), θ -> 0.0, names, kinds, sim, refit_mix)
    res_mix = GM._family_bootstrap(ad_mix, collect(1:m), 0.95, 40, 7, false)
    @test all(isfinite, res_mix.lower)
    @test res_mix.upper[3] == Inf      # 20 of 40 flagged > 2.5%
end
