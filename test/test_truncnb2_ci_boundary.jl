using GLLVModels, Test, TOML

# Truncated-NB2 interval adapters at the dispersion boundary (T14 F1). The fit-level
# `boundary` flag conditions an r past the Poisson limit (r > 1e6, where the likelihood
# is flat in r) out of the Wald Hessian, so it does not get a meaningless finite SE.
# The per-trait adapter already set it; the shared-r adapter did not.
#
# The bootstrap refits deliberately do NOT report `upper_boundary` (#542 option 3; the
# question #585 left open). The bootstrap drops a flagged replicate from every
# parameter's quantiles, and data simulated from a per-trait fit with one trait at the
# Poisson limit put that trait there again in every refit (12 of 12 on the #585 seed-93
# draw), so flagging would leave the per-trait bootstrap with no intervals at all.

const _GM = GLLVModels
const _CB = TOML.parsefile(joinpath(@__DIR__, "fixtures", "truncnb2_dispersion_boundary.toml"))
const _CB_Y = reshape(Int.(_CB["upper_Y_column_major"]), _CB["upper_p"], _CB["n"])

# Fits pinned at a chosen r: start there and run zero optimiser iterations, so the
# returned r is the start on every machine (`eigmin_floor = -Inf` keeps the shared-r
# breakdown retry from replacing it).
_cb_shared(r) = fit_truncated_nbinom2_gllvm(_CB_Y; K = 1, r_init = r, iterations = 0,
                                            eigmin_floor = -Inf)

@testset "truncated NB2 shared r: adapter flags r at the Poisson limit" begin
    ad_hi = _GM._family_ci(_cb_shared(1e8), _CB_Y)
    @test ad_hi.names[end] == "r"
    @test ad_hi.boundary[end]
    @test !any(ad_hi.boundary[1:(end - 1)])
    ad_mid = _GM._family_ci(_cb_shared(3.0), _CB_Y)
    @test !any(ad_mid.boundary)
end

@testset "truncated NB2 shared r: Wald conditions a boundary r out" begin
    ad = _GM._family_ci(_cb_shared(1e8), _CB_Y)
    ci = _GM._family_wald(ad, collect(eachindex(ad.θ)), 0.95)
    @test "r" in ci.boundary_terms
    @test !ci.pd_hessian
end

@testset "truncated NB2 per-trait: adapter flags the trait at the Poisson limit" begin
    p = _CB["upper_p"]
    f = fit_truncated_nbinom2_gllvm_pertrait(_CB_Y; K = 1,
                                             r_init = [fill(3.0, p - 1); 1e9], iterations = 0)
    ad = _GM._family_ci(f, _CB_Y)
    @test ad.boundary[end] && !any(ad.boundary[1:(end - 1)])
end
