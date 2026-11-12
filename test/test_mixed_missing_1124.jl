# Issue 1124: fit_mixed_gllvm with `missing` cells in a non-Normal row.
# The Normal warm start already skipped missing cells; the Poisson, NegativeBinomial,
# Binomial, Gamma and Beta warm starts threw a MethodError. These tests check, for each
# of the five families, that the fit runs and that its log-likelihood equals an
# independent evaluation in which each site with a missing cell is computed on the
# sub-model that simply omits that trait (rows of beta, Lambda, families, links removed).
using Test
using GLLVModels
using Distributions
using LinearAlgebra
using Random

const _MM = GLLVModels

function _mm_rows(rng, n)
    z = randn(rng, n)
    base = 0.1 .+ 0.6 .* z .+ 0.3 .* randn(rng, n)
    pois = Float64.(rand.(rng, Poisson.(exp.(0.5 .- 0.3 .* z))))
    rows = Dict(
        "Poisson"  => (Poisson(), Float64.(rand.(rng, Poisson.(exp.(1.0 .+ 0.4 .* z))))),
        "NegBin"   => (NegativeBinomial(), Float64.(rand.(rng, Poisson.(exp.(1.2 .+ 0.4 .* z))))),
        "Binomial" => (Binomial(), Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.6 .* z)))),
        "Gamma"    => (Gamma(), exp.(0.3 .+ 0.4 .* z) .* (0.5 .+ rand(rng, n))),
        "Beta"     => (Beta(), clamp.(1 ./ (1 .+ exp.(-0.5 .* z .+ 0.3 .* randn(rng, n))), 0.02, 0.98)),
    )
    return base, pois, rows
end

# Independent route: per site, drop the missing traits and evaluate the reduced model.
function _mm_reference_loglik(fit, Y, N)
    p, n = size(Y)
    fams = [_MM._with_dispersion(fit.families[t],
                isfinite(fit.dispersion[t]) ? fit.dispersion[t] : 1.0) for t in 1:p]
    total = 0.0
    for s in 1:n
        keep = [t for t in 1:p if !ismissing(Y[t, s])]
        isempty(keep) && continue            # no observed cell: contributes exactly 0
        total += _MM._mixed_loglik_site(fams[keep], fit.links[keep],
                    Float64[Y[t, s] for t in keep], N[keep, s],
                    fit.Λ[keep, :], fit.β[keep])
    end
    return total
end

@testset "mixed non-Normal missing (#1124)" begin
    n = 40
    for name in ("Poisson", "NegBin", "Binomial", "Gamma", "Beta")
        @testset "$name row" begin
            rng = MersenneTwister(11)
            base, pois, rows = _mm_rows(rng, n)
            fam, y = rows[name]
            fams = Any[Normal(), fam, Poisson()]
            Yc = permutedims(hcat(base, y, pois))
            N = ones(Int, 3, n)

            # (a) a few scattered missing cells in the family's row
            Y = Matrix{Union{Missing,Float64}}(Yc)
            Y[2, 5] = missing; Y[2, 17] = missing; Y[2, 33] = missing
            f = fit_mixed_gllvm(Y; families = fams, K = 1)
            @test f isa MixedFamilyFit
            @test isfinite(f.loglik)
            # (b) log-likelihood equals the independent sub-model evaluation
            @test f.loglik ≈ _mm_reference_loglik(f, Y, N) rtol = 1e-9
            # the same value from the public marginal at the fitted parameters
            fams_t = [_MM._with_dispersion(f.families[t],
                        isfinite(f.dispersion[t]) ? f.dispersion[t] : 1.0) for t in 1:3]
            @test mixed_marginal_loglik_laplace(fams_t, f.links, Y, N, f.Λ, f.β) ≈ f.loglik rtol = 1e-9
            # post-fit methods accept the matrix with missing cells
            @test all(isfinite, getLV(f, Y))
            @test all(isfinite, predict(f, Y))
            @test all(isfinite, correlation(f, Y))

            # (c1) every cell of one site missing: runs, and that site contributes 0
            Y2 = copy(Y); Y2[:, 9] .= missing
            f2 = fit_mixed_gllvm(Y2; families = fams, K = 1)
            @test isfinite(f2.loglik)
            @test f2.loglik ≈ _mm_reference_loglik(f2, Y2, N) rtol = 1e-9
        end
    end

    @testset "Int Union{Missing,Int} matrix, Poisson and NegativeBinomial rows" begin
        rng = MersenneTwister(12)
        _, pois, rows = _mm_rows(rng, n)
        Yi = Matrix{Union{Missing,Int}}(round.(Int,
                permutedims(hcat(pois, rows["NegBin"][2], pois .+ 1))))
        Yi[2, 5] = missing; Yi[1, 20] = missing
        fi = fit_mixed_gllvm(Yi; families = [Poisson(), NegativeBinomial(), Poisson()], K = 1)
        @test isfinite(fi.loglik)
        @test fi.loglik ≈ _mm_reference_loglik(fi, Yi, ones(Int, 3, n)) rtol = 1e-9
    end

    @testset "complete matrix is unchanged by the Union eltype" begin
        # A Union{Missing,Float64} matrix with no missing cell must give the bit-identical
        # fit to the plain Float64 matrix (the warm start and objective are unchanged).
        rng = MersenneTwister(13)
        base, pois, rows = _mm_rows(rng, n)
        fams = Any[Normal(), Poisson(), Gamma(), Beta()]
        Yc = permutedims(hcat(base, rows["Poisson"][2], rows["Gamma"][2], rows["Beta"][2]))
        fa = fit_mixed_gllvm(Yc; families = fams, K = 1)
        fb = fit_mixed_gllvm(Matrix{Union{Missing,Float64}}(Yc); families = fams, K = 1)
        @test fa.loglik === fb.loglik
        @test fa.β == fb.β
        @test fa.Λ == fb.Λ
    end
end
