# `missing` in Y on the Gaussian fitters typed `AbstractMatrix{<:Real}` (refs #1147, section 3).
# `fit_gaussian_sources` and `fit_phylo_latent_gllvm` used to fail at dispatch with a MethodError; the
# other four already refuse with a named ArgumentError and are pinned here so all six stay covered.
using Test, GLLVModels, Random, LinearAlgebra

@testset "missing in Y on AbstractMatrix{<:Real} fitters (#1147)" begin
    rng = MersenneTwister(21)
    p, n = 6, 40
    Y = 0.8 .* randn(rng, p, 1) * randn(rng, 1, n) .+ 0.5 .* randn(rng, p, n)
    Ypos = exp.(0.3 .* Y)
    Ydz = copy(Ypos); Ydz[rand(rng, p, n) .< 0.25] .= 0.0
    S = let A = randn(rng, p, p); M = A * A' ./ p + I; M ./ maximum(diag(M)); end
    A20 = let A = randn(rng, 20, 20); A * A' ./ 20 + I; end
    unit = repeat(1:20, inner = 2); sid = repeat(1:p, outer = 7)[1:n]; species = string.(sid)
    pp = PrecisionPhy(GLLVModels.random_balanced_tree(p; branch_length = 0.4))
    term = [GroupingTerm(:unit; mode = :latent, rank = 1)]
    U(M) = Matrix{Union{Missing,eltype(M)}}(M)
    function M1(M); Z = U(M); Z[1, 2] = missing; Z; end
    thrown(f) = try; f(); nothing; catch e; e; end
    cases = (
        ("fit_gaussian_sources", Y, y -> fit_gaussian_sources(y; sources = [SourceCovariance(A20; groups = unit, name = :a, mode = :indep, common = true)])),
        ("fit_precision_multivariate", Y, y -> GLLVModels.fit_precision_multivariate(y, pp; species_id = sid)),
        ("fit_grouped_gaussian", Y, y -> GLLVModels.fit_grouped_gaussian(y; terms = term, unit = unit)),
        ("fit_phylo_latent_gllvm", Y, y -> fit_phylo_latent_gllvm(y, species; d = 1, vcv = S, tip_labels = string.(1:p))),
        ("fit_lognormal_gllvm", Ypos, y -> fit_lognormal_gllvm(y; K = 1)),
        ("fit_delta_lognormal_gllvm", Ydz, y -> fit_delta_lognormal_gllvm(y; K = 1)),
    )
    for (name, M, f) in cases
        @testset "$name" begin
            e = thrown(() -> f(M1(M)))
            @test e isa ArgumentError
            @test e isa ArgumentError && occursin(name, e.msg)
        end
    end
    # A Union eltype with nothing missing fits like the plain matrix.
    @testset "Union eltype, nothing missing" begin
        for (name, M, f) in cases[[1, 4]]
            @test thrown(() -> f(U(M))) === nothing
        end
    end
end
