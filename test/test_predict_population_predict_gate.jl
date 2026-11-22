using Test, GLLVModels, Distributions, Random

const _POP_SUPPORTED_SUBSTR = (
    "GllvmCovFit",
    "RRRFit",
    "GroupedGaussianFit",
    "predict_spatial",
)

function _expect_population_unsupported(f, Xn)
    err = nothing
    try
        predict(f(), Xn; type = :link)
    catch e
        err = e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("population-level predict", msg)
    for needle in _POP_SUPPORTED_SUBSTR
        @test occursin(needle, msg)
    end
    @test !occursin("MethodError", msg)
end

@testset "population predict unsupported gate (#1192)" begin
    Random.seed!(1)
    p, n = 5, 60
    x = randn(n)
    X = zeros(p, n, 1)
    for s in 1:n, t in 1:p
        X[t, s, 1] = x[s]
    end
    Yint = [rand(Poisson(exp(0.3 + 0.4 * x[s]))) for t in 1:p, s in 1:n]
    Y = Float64.(Yint)
    Xn = zeros(p, 2, 1)
    Xn[:, 1, 1] .= -0.5
    Xn[:, 2, 1] .= 1.0

    @test size(predict(fit_gllvm_cov(Yint; family = Poisson(), X = X, K = 1), Xn; type = :link)) == (p, 2)

    for (name, f) in (
        ("NBGroupedCovFit", () -> fit_nb_gllvm_grouped_cov(Yint; X = X, K = 1)),
        ("ZIPCovFit", () -> fit_zip_gllvm_cov(Y; X = X, K = 1)),
        ("GllvmFit+Gaussian design", () -> fit_gaussian_gllvm(Y; K = 1, X = X)),
    )
        @testset "$name" begin
            _expect_population_unsupported(f, Xn)
        end
    end
end