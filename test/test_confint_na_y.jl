# confint, profile_ci and bootstrap_ci on a Gaussian fit made without a `mask` used to accept a `y`
# holding `missing` or NaN: Wald and profile returned all-NaN intervals, the bootstrap ignored the
# cells. They now throw an ArgumentError naming the first cell. A fit made with a mask keeps its
# own data route, and the complete matrix still gives finite intervals.
using Test, GLLVModels, Random

@testset "confint family and NA in y" begin
    Random.seed!(2)
    p, n, K = 6, 40, 1
    Y = 0.7 .* randn(p, K) * randn(K, n) .+ 0.4 .* randn(p, n)
    Ym = Matrix{Union{Float64,Missing}}(Y); Ym[1, 2] = missing
    Yn = copy(Y); Yn[1, 2] = NaN
    thrown(f) = try; f(); nothing; catch e; e; end
    isarg(e, what) = e isa ArgumentError && occursin(what, e.msg) && occursin("(row, column) = (1, 2)", e.msg)

    fit = fit_gaussian_gllvm(Y; K = K)
    term = confint(fit, Y).term[1]
    for (what, y) in (("missing", Ym), ("NaN", Yn))
        for m in (:wald, :bootstrap)
            @test isarg(thrown(() -> confint(fit, y; method = m, n_boot = 4, seed = 1)), what)
        end
        @test isarg(thrown(() -> confint(fit, y; method = :profile, parm = term)), what)
        @test isarg(thrown(() -> profile_ci(fit, 1; y = y)), what)
        @test isarg(thrown(() -> bootstrap_ci(fit; y = y, n_boot = 4, seed = 1)), what)
    end
    @test all(isfinite, confint(fit, Y).lower)

    mk = observed_mask(Ym)
    mfit = fit_gaussian_gllvm(Ym; K = K, mask = mk)
    @test thrown(() -> confint(mfit, Ym)) === nothing
end
