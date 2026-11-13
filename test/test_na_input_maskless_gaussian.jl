# NA input on seven Gaussian fitters that have no `mask` and already accept any `AbstractMatrix`
# (refs #1147). A `Matrix{Union{Missing,T}}` with a missing cell used to fail inside the fit with a
# `MethodError`; it is now an ArgumentError naming the fitter and the first cell. `em_fa` also
# refuses NaN and Inf (it returned an all-NaN result). No signature changes: a Union eltype with
# nothing missing behaves as before. The fitters typed `AbstractMatrix{<:Real}` are not covered.
using Test, GLLVModels, Random, LinearAlgebra

@testset "NA input on maskless Gaussian fitters (#1147)" begin
    rng = MersenneTwister(21)
    p, n = 6, 40
    Y = 0.8 .* randn(rng, p, 1) * randn(rng, 1, n) .+ 0.5 .* randn(rng, p, n)
    Xc = randn(rng, p, n, 1)
    S = let A = randn(rng, p, p); M = A * A' ./ p + I; M ./ maximum(diag(M)); end
    Kstar = let A = randn(rng, n, n); A * A' ./ n + I; end
    ind = repeat(1:(n ÷ 2), inner = 2)
    grp = repeat(1:8, outer = 5)
    Zr = randn(rng, n, 1)

    U(M) = Matrix{Union{Missing,eltype(M)}}(M)
    function M1(M); Z = U(M); Z[1, 2] = missing; Z; end
    thrown(f) = try; f(); nothing; catch e; e; end
    typed(e, name) = e isa ArgumentError && occursin(name, e.msg) &&
        occursin("(row, column) = (1, 2)", e.msg) && occursin("`missing`", e.msg)
    function objective(f)                       # the log-likelihood field, whatever the fit type calls it
        f isa Tuple && return f[3]
        for nm in propertynames(f)
            (lowercase(String(nm)) in ("ll", "llik") || endswith(lowercase(String(nm)), "loglik")) && return getproperty(f, nm)
        end
        error("no log-likelihood field on $(typeof(f))")
    end
    quiet(f) = redirect_stderr(devnull) do; f(); end

    # (name, call, does the Union eltype with nothing missing already fit on main?)
    cases = (
        ("em_fa", y -> em_fa(y, 1), true),
        ("fit_gaussian_reml", y -> fit_gaussian_reml(y, Xc; K = 1), false),
        ("fit_twolevel_gaussian", y -> fit_twolevel_gaussian(y, ind; K_B = 1, K_W = 1), true),
        ("fit_gaussian_random_slope", y -> fit_gaussian_random_slope(y, grp, Zr; K = 1), true),
        ("em_fit_phylo", y -> em_fit_phylo(y, 1, S), true),
        ("em_fit_phylo_squarem", y -> em_fit_phylo_squarem(y, 1, S), true),
        ("fit_coevolution_gaussian", y -> fit_coevolution_gaussian(y, Kstar; d = 1), true),
    )

    for (name, call, union_fits) in cases
        @testset "$name" begin
            @test typed(thrown(() -> quiet(() -> call(M1(Y)))), name)
            base = quiet(() -> call(Y))
            @test all(isfinite, objective(base))
            union_fits && @test objective(quiet(() -> call(U(Y)))) == objective(base)
        end
    end

    @testset "em_fa refuses NaN and Inf (it returned NaN silently)" begin
        for (what, v) in (("NaN", NaN), ("Inf", Inf), ("Inf", -Inf))
            Yb = copy(Y); Yb[1, 2] = v
            e = thrown(() -> em_fa(Yb, 1))
            @test e isa ArgumentError && occursin(what, e.msg) && occursin("(row, column) = (1, 2)", e.msg)
        end
    end
end
