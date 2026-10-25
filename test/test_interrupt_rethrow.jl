using Test
using GLLVModels

# Issue #774: a bare `catch` in a fitter objective swallowed InterruptException
# (Ctrl-C), so an interrupted fit carried on with the barrier/1e12 sentinel.
# Genuine objective failures must still return the sentinel.
@testset "fitter objectives rethrow InterruptException (#774)" begin
    @testset "mixed: _penalized_negloglik_fg! (behavioural)" begin
        θ = [0.5, -0.5]
        @test_throws InterruptException GLLVModels._penalized_negloglik_fg!(
            0.0, nothing, _ -> throw(InterruptException()), θ)
        # An ordinary error still falls back to the barrier penalty.
        F = GLLVModels._penalized_negloglik_fg!(
            0.0, nothing, _ -> error("objective failed"), θ)
        @test F >= 1e12
    end

    @testset "ordinal: objective wrappers (source-level)" begin
        # The four wrappers are closures inside the fitters and cannot be
        # reached with an injected failure, so pin the source: no bare `catch`,
        # and each wrapper rethrows InterruptException.
        src = read(joinpath(pkgdir(GLLVModels), "src", "families", "ordinal.jl"), String)
        @test !occursin(r"^\s*catch\s*$"m, src)
        @test count("isa InterruptException && rethrow()", src) >= 4
    end

    @testset "phylo poisson xlv constrained refit (#747)" begin
        nll = _ -> throw(InterruptException())
        # p=6, K=1, q_lv=1: packed length is 6 + 1 + 6 + 1 = 14.
        theta = zeros(14)
        @test_throws InterruptException GLLVModels._phylo_poisson_xlv_constrained_refit(
            nll, theta, 6, 1, 1, 1, 0.0;
            profile_iterations = 2, penalty_weights = (1.0,))
        # An ordinary error still becomes the 1e12 barrier, then NaN nll.
        out = GLLVModels._phylo_poisson_xlv_constrained_refit(
            _ -> error("objective failed"), theta, 6, 1, 1, 1, 0.0;
            profile_iterations = 2, penalty_weights = (1.0,))
        @test isnan(out.nll)
    end
end
