# #1449: fit_gllvm_cov must refuse Ordinal / ZIP / ZINB with a clear ArgumentError
# (dedicated covariate fitters and gllvm(@formula(...)) are the supported routes).

using GLLVModels, Test, Random, Distributions

function _site_design(x::AbstractVector, p::Integer)
    n = length(x)
    X = zeros(p, n, 1)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = x[s]
    end
    return X
end

function _argerr_msg(f)
    try
        f()
    catch e
        return e isa ArgumentError ? e.msg : "not an ArgumentError: $(typeof(e))"
    end
    return "no throw"
end

@testset "fit_gllvm_cov refuses Ordinal / ZIP / ZINB (#1449)" begin
    Random.seed!(31)
    p, n, K = 3, 50, 1
    x = randn(n)
    X = _site_design(x, p)
    Yord = [rand(1:3) for t in 1:p, s in 1:n]
    Yzip = [rand() < 0.25 ? 0 : rand(1:5) for t in 1:p, s in 1:n]
    Yzinb = copy(Yzip)

    entries = [
        ("Ordinal", Ordinal(), Yord, "fit_ordinal_gllvm_pertrait_cov", "gllvm(@formula"),
        ("ZIPoisson", ZIPoisson(), Yzip, "fit_zip_gllvm_cov", "gllvm(@formula"),
        ("ZINegBin", ZINegBin(), Yzinb, "fit_zinb_gllvm_cov", "gllvm(@formula"),
    ]

    @testset "$label" for (label, family, Y, fitter, route) in entries
        msg = _argerr_msg(() -> fit_gllvm_cov(Y; family = family, X = X, K = K, iterations = 150))
        @test occursin("fit_gllvm_cov", msg)
        @test occursin(label, msg)
        @test occursin(fitter, msg)
        @test occursin(route, msg)
    end

    @testset "supported Poisson route unchanged" begin
        Yp = [rand(Poisson(3)) for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Yp; family = Poisson(), X = X, K = K, iterations = 50)
        @test fit isa GllvmCovFit
        @test isfinite(fit.loglik)
    end
end