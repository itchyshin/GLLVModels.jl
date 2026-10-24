using GLLVModels, Test, Random, Distributions

# #763: an explicit `mask` must not declare a `missing` cell observed. The fitters replace
# `missing` by a placeholder (0 / 1), so a mask that says "observed" there would fit the
# placeholder as data. The contract: such a mask throws an `ArgumentError` naming the first
# offending cell and the count; a mask that agrees with `missing` (or no mask) is unchanged.
@testset "explicit mask + missing in Y (#763)" begin
    # ---- helper unit checks -------------------------------------------------
    Ym = Union{Missing,Int}[1 missing; 2 3]
    @test GLLVModels._resolve_obs_mask(nothing, Ym) == [true false; true true]
    ok = [true false; true true]
    @test GLLVModels._resolve_obs_mask(ok, Ym) === ok
    sub = [true false; false true]                      # marks one more cell unobserved: legal
    @test GLLVModels._resolve_obs_mask(sub, Ym) === sub
    full = trues(2, 2)
    err = try GLLVModels._resolve_obs_mask(full, Ym); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("(1, 2)", err.msg) && occursin("1 ", err.msg)
    @test GLLVModels._resolve_obs_mask(nothing, [1 2; 3 4]) === nothing
    @test GLLVModels._resolve_obs_mask(full, [1 2; 3 4]) === full   # no missing: untouched

    # ---- fitters --------------------------------------------------------------
    Random.seed!(2)
    p, n = 5, 60
    β0 = [1.5, 1.0, 0.8, 1.2, 0.6]
    Λ0 = reshape([0.5, -0.4, 0.3, 0.6, -0.2], p, 1)
    z  = randn(1, n)
    μ  = exp.(β0 .+ Λ0 * z)
    Yc = [rand(Poisson(μ[t, i])) for t in 1:p, i in 1:n]
    Yp  = Yc .+ 5                                  # large counts so a 0 placeholder is visible
    Ytr = max.(Yc, 1)                              # zero-truncated families need y >= 1
    Yg  = Float64.(Yc) .+ 0.5                      # positive responses
    holes = (1:2, 1:30)

    function holed(Y)
        M = Matrix{Union{Missing,eltype(Y)}}(Y)
        M[holes...] .= missing
        return M
    end

    cases = [
        ("poisson",            holed(Yp),  (Y; kw...) -> fit_poisson_gllvm(Y; K = 1, kw...)),
        ("nb",                 holed(Yp),  (Y; kw...) -> fit_nb_gllvm(Y; K = 1, kw...)),
        ("nb1",                holed(Yp),  (Y; kw...) -> fit_nb1_gllvm(Y; K = 1, kw...)),
        ("truncated_poisson",  holed(Ytr), (Y; kw...) -> fit_truncated_poisson_gllvm(Y; K = 1, kw...)),
        ("truncated_nbinom2",  holed(Ytr), (Y; kw...) -> fit_truncated_nbinom2_gllvm(Y; K = 1, kw...)),
        ("truncated_nb2_pertrait", holed(Ytr), (Y; kw...) -> fit_truncated_nbinom2_gllvm_pertrait(Y; K = 1, kw...)),
        ("censored_poisson",   holed(Yp),  (Y; kw...) -> fit_censored_poisson_gllvm(Y; K = 1, kw...)),
        ("gamma",              holed(Yg),  (Y; kw...) -> fit_gamma_gllvm(Y; K = 1, kw...)),
        ("exponential",        holed(Yg),  (Y; kw...) -> fit_exponential_gllvm(Y; K = 1, kw...)),
        ("tweedie",            holed(Yg),  (Y; kw...) -> fit_tweedie_gllvm(Y; K = 1, kw...)),
    ]
    for (name, Y, fit) in cases
        @testset "$name" begin
            f0 = fit(Y)                                         # mask derived from `missing`
            # a mask that covers nothing is refused, naming the first cell and the count
            err = try fit(Y; mask = trues(p, n)); nothing catch e; e end
            @test err isa ArgumentError
            @test err isa ArgumentError && occursin("(1, 1)", err.msg) &&
                  occursin("$(count(ismissing, Y))", err.msg)
            # a mask that agrees with `missing` gives the identical fit
            f2 = fit(Y; mask = .!ismissing.(Y))
            @test f2.loglik == f0.loglik
            @test f2.β == f0.β
        end
    end
end
