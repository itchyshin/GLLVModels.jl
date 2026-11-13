# NA input on the Gaussian fitters that have no `mask`. A `Matrix{Union{Missing,T}}` used to reach
# them as a no-method error at dispatch (Y typed `AbstractMatrix{<:Real}`) or deep inside the fit,
# and NaN went through `em_fa` to a NaN result. Now `missing` is a typed ArgumentError naming the
# fitter and the first cell, NaN is refused by `em_fa`, and a `Union{Missing,T}` eltype with
# nothing missing fits exactly like the plain matrix.
using Test, GLLVModels, Random, LinearAlgebra

@testset "NA input on the maskless Gaussian fitters" begin
    rng = MersenneTwister(21)
    p, n = 6, 40
    Y = 0.8 .* randn(rng, p, 1) * randn(rng, 1, n) .+ 0.5 .* randn(rng, p, n)
    Ypos = exp.(0.3 .* Y)
    Ydz = copy(Ypos); Ydz[rand(rng, p, n) .< 0.25] .= 0.0
    Xc = randn(rng, p, n, 1)
    S = let A = randn(rng, p, p); M = A * A' ./ p + I; M ./ maximum(diag(M)); end
    Kstar = let A = randn(rng, n, n); A * A' ./ n + I; end
    A20 = let A = randn(rng, 20, 20); A * A' ./ 20 + I; end
    ind = repeat(1:(n ÷ 2), inner = 2)
    grp = repeat(1:8, outer = 5)
    Zr = randn(rng, n, 1)
    tree = GLLVModels.random_balanced_tree(p; branch_length = 0.4)
    pp = PrecisionPhy(tree)
    sid = repeat(1:p, outer = 7)[1:n]
    species = string.(sid)
    term = [GroupingTerm(:unit; mode = :latent, rank = 1)]
    unit = repeat(1:20, inner = 2)

    U(M) = Matrix{Union{Missing,eltype(M)}}(M)
    function M1(M); Z = U(M); Z[1, 2] = missing; Z; end
    thrown(f) = try; f(); nothing; catch e; e; end
    typed(e, name) = e isa ArgumentError && occursin(name, e.msg) &&
        occursin("(row, column) = (1, 2)", e.msg) && occursin("`missing`", e.msg)
    function objective(f)                       # log-likelihood where the fit has one, else the loadings
        f isa Tuple && return f[3]
        for nm in (:logLik, :loglik, :ll)
            hasproperty(f, nm) && return getproperty(f, nm)
        end
        return f.Λ
    end
    quiet(f) = redirect_stderr(devnull) do; f(); end

    cases = (
        ("em_fa", Y, y -> em_fa(y, 1)),
        ("fit_gaussian_reml", Y, y -> fit_gaussian_reml(y, Xc; K = 1)),
        ("fit_twolevel_gaussian", Y, y -> fit_twolevel_gaussian(y, ind; K_B = 1, K_W = 1)),
        ("fit_gaussian_random_slope", Y, y -> fit_gaussian_random_slope(y, grp, Zr; K = 1)),
        ("em_fit_phylo", Y, y -> em_fit_phylo(y, 1, S)),
        ("em_fit_phylo_squarem", Y, y -> em_fit_phylo_squarem(y, 1, S)),
        ("fit_coevolution_gaussian", Y, y -> fit_coevolution_gaussian(y, Kstar; d = 1)),
        ("fit_gaussian_sources", Y, y -> fit_gaussian_sources(y; sources = [SourceCovariance(A20;
            groups = unit, name = :a, mode = :indep, common = true)])),
        ("fit_precision_multivariate", Y, y -> GLLVModels.fit_precision_multivariate(y, pp; species_id = sid)),
        ("fit_grouped_gaussian", Y, y -> GLLVModels.fit_grouped_gaussian(y; terms = term, unit = unit)),
        ("fit_phylo_latent_gllvm", Y, y -> fit_phylo_latent_gllvm(y, species; d = 1, vcv = S,
            tip_labels = string.(1:p))),
        ("fit_lognormal_gllvm", Ypos, y -> fit_lognormal_gllvm(y; K = 1)),
        ("fit_delta_lognormal_gllvm", Ydz, y -> fit_delta_lognormal_gllvm(y; K = 1)),
    )

    for (name, M, call) in cases
        @testset "$name" begin
            e = thrown(() -> quiet(() -> call(M1(M))))
            @test typed(e, name)
            base = quiet(() -> call(M))
            alt = quiet(() -> call(U(M)))
            @test objective(alt) == objective(base)
        end
    end

    @testset "a Matrix{Any} is a typed error, not a StackOverflowError" begin
        for (name, M, call) in cases
            e = thrown(() -> quiet(() -> call(Matrix{Any}(M))))
            @test e isa ArgumentError && occursin(name, e.msg) && occursin("numeric response matrix", e.msg)
        end
    end

    @testset "em_fa refuses NaN (it returned NaN silently)" begin
        Yn = copy(Y); Yn[1, 2] = NaN
        e = thrown(() -> em_fa(Yn, 1))
        @test e isa ArgumentError && occursin("NaN", e.msg) && occursin("(row, column) = (1, 2)", e.msg)
    end
end
