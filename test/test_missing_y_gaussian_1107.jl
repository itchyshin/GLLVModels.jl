# #1107: the Gaussian fitters used to throw an internal MethodError when `Y` held `missing`
# (ppca_init / a Float64 copy). They now throw an ArgumentError that names the fitter and the
# first missing cell. A `Union{Missing,T}` eltype with no `missing` actually present fits like
# the plain matrix. An explicit `mask` keeps working on fit_gaussian_gllvm.
using Test, GLLVModels, Random, Distributions

@testset "#1107 Gaussian fitters and missing in Y" begin
    Random.seed!(1)
    p, n, K = 6, 40, 1
    Y = 0.7 .* randn(p, K) * randn(K, n) .+ 0.4 .* randn(p, n)
    Ym = Matrix{Union{Float64,Missing}}(Y)
    Ym[1, 2] = missing
    Ym[3, 7] = missing
    Ynm = Matrix{Union{Float64,Missing}}(Y)     # Union eltype, nothing missing

    thrown(f) = try; f(); nothing; catch e; e; end
    typed(e, name) = e isa ArgumentError && occursin(name, e.msg) &&
        occursin("(row, column) = (1, 2)", e.msg) && occursin("2 missing", e.msg)

    @testset "missing cells give a typed ArgumentError" begin
        @test typed(thrown(() -> fit_gaussian_gllvm(Ym; K = K)), "fit_gaussian_gllvm")
        @test typed(thrown(() -> fit_gllvm(Ym; family = Normal(), K = K)), "fit_gaussian_gllvm")
        @test typed(thrown(() -> fit_gaussian_pervar_gllvm(Ym; K = K)), "fit_gaussian_pervar_gllvm")
    end

    @testset "Union eltype without missing fits like the plain matrix" begin
        base = fit_gaussian_gllvm(Y; K = K)
        alt = fit_gaussian_gllvm(Ynm; K = K)
        @test alt.logLik == base.logLik
        @test alt.converged == base.converged
        gb = fit_gllvm(Y; family = Normal(), K = K)
        ga = fit_gllvm(Ynm; family = Normal(), K = K)
        @test ga.logLik == gb.logLik
    end

    @testset "explicit mask still fits (unchanged)" begin
        fit = fit_gaussian_gllvm(Ym; K = K, mask = observed_mask(Ym))
        @test isfinite(fit.logLik)
    end

    @testset "plain matrices unchanged" begin
        @test isfinite(fit_gaussian_gllvm(Y; K = K).logLik)
        @test isfinite(fit_gllvm(Y; family = Normal(), K = K).logLik)
    end
end
