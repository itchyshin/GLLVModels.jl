using GLLVModels, Test, Random, Distributions

# #763: an explicit `mask` must never declare a `missing` cell observed. Before the
# fix the fitters replaced `missing` with a placeholder (0 / 1) and, when the caller's
# mask said "observed" there, fitted the placeholder as data. The contract: a
# `missing` cell is unobserved whatever the mask says, so
#   mask = trues(p, n)   ==   no mask   ==   mask = .!ismissing.(Y).
@testset "explicit mask + missing in Y (#763)" begin
    # ---- helper unit checks -------------------------------------------------
    Ym = Union{Missing,Int}[1 missing; 2 3]
    @test GLLVModels._resolve_obs_mask(nothing, Ym) == [true false; true true]
    @test GLLVModels._resolve_obs_mask(trues(2, 2), Ym) == [true false; true true]
    @test GLLVModels._resolve_obs_mask([true true; false true], Ym) == [true false; false true]
    @test GLLVModels._resolve_obs_mask(nothing, [1 2; 3 4]) === nothing
    full = trues(2, 2)
    @test GLLVModels._resolve_obs_mask(full, [1 2; 3 4]) === full   # no missing: untouched

    # ---- fitters --------------------------------------------------------------
    Random.seed!(2)
    p, n = 5, 60
    β0 = [1.5, 1.0, 0.8, 1.2, 0.6]
    Λ0 = reshape([0.5, -0.4, 0.3, 0.6, -0.2], p, 1)
    z  = randn(1, n)
    μ  = exp.(β0 .+ Λ0 * z)
    Yc = [rand(Poisson(μ[t, i])) for t in 1:p, i in 1:n]
    Yp = Yc .+ 5                                   # large counts so a 0 placeholder is visible
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
    ]
    for (name, Y, fit) in cases
        @testset "$name" begin
            f0 = fit(Y)                                         # mask derived from `missing`
            f1 = fit(Y; mask = trues(p, n))                     # mask that covers nothing
            f2 = fit(Y; mask = .!ismissing.(Y))                 # mask that agrees
            @test f1.loglik ≈ f0.loglik rtol = 1e-6
            @test f1.β ≈ f0.β atol = 1e-4
            @test f2.loglik ≈ f0.loglik rtol = 1e-6
        end
    end
end
