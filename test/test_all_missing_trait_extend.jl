# Extends test_all_missing_trait.jl (the shared check `_check_every_trait_observed`) to the
# fitters that still accepted a trait (row of the p x n response) with no observed cell:
#   - the ordinal fitters: fit_ordinal_gllvm (accepted it, converged = false),
#     fit_ordinal_gllvm_pertrait, fit_ordinal_gllvm_pertrait_cov, fit_ordinal_gllvm_pertrait_aghq
#     and fit_gllvm(Ordinal()) (these refused with a different "needs >= 2 observed
#     categories" message); the ordinal fitters take `mask` only, not `missing` in Y;
#   - fit_mixed_gllvm, which takes `missing` in Y (it has no `mask` keyword).
# The fitters that take neither a mask nor `missing` (two-part, hurdle, Student t, Lognormal,
# rrr, quadratic, constrained ordination) have nothing to check.
# A site (column) with no observed cell stays accepted (the maintainer's choice, Nov 2026).

using GLLVModels, Test, Random, Distributions

const _AX_IT = (iterations = 3, g_tol = 1e-2)

function _ax_msg(f)
    try
        f()
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return nothing
end

@testset "All-missing trait: ordinal and mixed-family fitters" begin
    rng = MersenneTwister(1113)
    p, n = 4, 12
    Yo = rand(rng, 1:3, p, n)
    Yo[2, 2] = 3                       # the single observed cell of trait 2 below has level 3
    Xo = randn(MersenneTwister(5), p, n, 1)

    ORD = [
        ("ordinal", (Y; kw...) -> fit_ordinal_gllvm(Y; K = 1, _AX_IT..., kw...)),
        ("ordinal_pertrait", (Y; kw...) -> fit_ordinal_gllvm_pertrait(Y; K = 1, _AX_IT..., kw...)),
        ("ordinal_pertrait_cov", (Y; kw...) -> fit_ordinal_gllvm_pertrait_cov(Y; X = Xo, K = 1, _AX_IT..., kw...)),
        ("ordinal_pertrait_aghq", (Y; kw...) -> GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y; K = 1, aghq = 3, _AX_IT..., kw...)),
        ("fit_gllvm(Ordinal())", (Y; kw...) -> fit_gllvm(Y; family = Ordinal(), K = 1, _AX_IT..., kw...)),
    ]
    @testset "$name" for (name, fit) in ORD
        mask = trues(p, n); mask[2, :] .= false
        msg = _ax_msg(() -> fit(Yo; mask = mask))
        @test msg !== nothing
        msg !== nothing && @test occursin("trait 2", msg) && occursin("no observed cell", msg)
        # a trait observed at a single cell only is accepted
        mask2 = trues(p, n); mask2[2, 1:n] .= false; mask2[2, 2] = true
        @test isfinite(fit(Yo; mask = mask2).loglik)
        # complete data: an all-true mask equals no mask, bit for bit
        @test fit(Yo; mask = trues(p, n)).loglik == fit(Yo).loglik
        # a site with no observed cell is still accepted (not decided by the check; unchanged)
        mask3 = trues(p, n); mask3[:, 5] .= false
        @test isfinite(fit(Yo; mask = mask3).loglik)
    end

    @testset "fit_mixed_gllvm (missing in Y)" begin
        fams = [Normal(), Poisson(), Normal(), Poisson()]
        Y = Matrix{Any}(undef, p, n)
        for i in 1:p, j in 1:n
            Y[i, j] = fams[i] isa Normal ? randn(rng) : rand(rng, 0:4)
        end
        fit(Y) = fit_mixed_gllvm(Y; families = fams, K = 1, _AX_IT...)
        Ym = Matrix{Union{Missing,Any}}(Y); Ym[3, :] .= missing
        msg = _ax_msg(() -> fit(Ym))
        @test msg !== nothing
        msg !== nothing && @test occursin("trait 3", msg) && occursin("no observed cell", msg)
        # also through the fit_gllvm front end
        msg = _ax_msg(() -> fit_gllvm(Ym; family = fams, K = 1, _AX_IT...))
        @test msg !== nothing && occursin("trait 3", msg)
        # `missing` is supported on the Normal traits only (the warm start of the count
        # traits has no NA path), so the accepted-case checks use Normal trait 3. An
        # all-missing count trait is refused by the same check, before that warm start.
        Yc = Matrix{Union{Missing,Any}}(Y); Yc[2, :] .= missing
        msg = _ax_msg(() -> fit(Yc))
        @test msg !== nothing && occursin("trait 2", msg) && occursin("no observed cell", msg)
        # a Normal trait observed at a single cell only is accepted
        Y1 = Matrix{Union{Missing,Any}}(Y); Y1[3, 2:n] .= missing
        @test isfinite(fit(Y1).loglik)
        # one missing Normal cell: accepted
        Y2 = Matrix{Union{Missing,Any}}(Y); Y2[1, 1] = missing
        @test isfinite(fit(Y2).loglik)
        # complete data through the same entry: Union{Missing} copy equals the plain matrix
        @test fit(Matrix{Union{Missing,Any}}(Y)).loglik == fit(Y).loglik
    end
end
