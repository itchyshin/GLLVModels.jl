using GLLVModels, Test, TOML

# A zero-truncated NB2 fit whose dispersion runs below r = 1e-6 is degenerate (extreme
# overdispersion, the likelihood flat in r), yet Optim still stops with converged = true.
# One count of 10^13 in otherwise ordinary data drives both fitters there, and they must
# report converged = false. Above r = 1e6 (the Poisson limit) r is not identified, but the
# fit is usually sound, so the fitters only warn and keep Optim's verdict.

const _TNB2_BD = TOML.parsefile(joinpath(@__DIR__, "fixtures",
                                         "truncnb2_dispersion_boundary.toml"))
_tnb2_bd_Y(key, p = _TNB2_BD["p"]) = reshape(Int.(_TNB2_BD[key]), p, _TNB2_BD["n"])

@testset "truncated NB2 (shared r): r below 1e-6 is not converged" begin
    f = @test_logs (:warn, r"below 1e-6") match_mode = :any fit_truncated_nbinom2_gllvm(
        _tnb2_bd_Y("shared_Y_column_major"); K = 1)
    @test f.r < 1e-6                     # premise: the fit is at the lower boundary
    @test !f.converged
end

@testset "truncated NB2 (per-trait r): any r_t below 1e-6 is not converged" begin
    f = @test_logs (:warn, r"below 1e-6") match_mode = :any fit_truncated_nbinom2_gllvm_pertrait(
        _tnb2_bd_Y("pertrait_Y_column_major"); K = 1)
    @test any(<(1e-6), f.r)
    @test !f.converged
end

@testset "truncated NB2 (per-trait r): r_t above 1e6 warns but stays converged" begin
    f = @test_logs (:warn, r"Poisson limit") match_mode = :any fit_truncated_nbinom2_gllvm_pertrait(
        _tnb2_bd_Y("upper_Y_column_major", _TNB2_BD["upper_p"]); K = 1)
    @test any(>(1e6), f.r) && all(>=(1e-6), f.r)   # premise: upper boundary only
    @test f.converged
end
