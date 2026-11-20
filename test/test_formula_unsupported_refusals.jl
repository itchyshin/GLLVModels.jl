using GLLVModels, Test, Random, Distributions, StatsModels

# Unsupported family markers, family values, keywords and function terms on the formula route
# with covariates used to stop with a raw `MethodError` naming an internal function (#1190).
@testset "formula covariate route: named ArgumentError instead of a raw MethodError" begin
    rng = MersenneTwister(1)
    p, n = 4, 60
    x1 = randn(rng, n)
    eta = [0.3 * j + 0.5 * x1[s] for j in 1:p, s in 1:n]
    Yp = [Float64(rand(rng, Poisson(exp(0.5 * e)))) for e in eta]
    Yg = eta .+ 0.5 .* randn(rng, p, n)
    Ypos = abs.(Yg) .+ 0.1
    site = (x1 = x1,)
    msg(f) = try f(); nothing catch e; e end

    @testset "family marker without a covariate route: $(nameof(typeof(fam)))" for fam in
            (TruncatedPoisson(), Lognormal(), COMPoisson(), HurdlePoisson(), OrderedBeta(), BetaHurdle())
        err = msg(() -> gllvm(@formula(y ~ 1 + x1), Ypos, site; family = fam, K = 1))
        @test err isa ArgumentError
        @test occursin(string(nameof(typeof(fam))), err.msg)
        @test !occursin("_cov_default_link", err.msg)
    end

    @testset "family that is not a family object" for fam in ("poisson", :poisson, fill(Poisson(), p))
        err = msg(() -> gllvm(@formula(y ~ 1 + x1), Yp, site; family = fam, K = 1))
        @test err isa ArgumentError
        @test occursin("family", err.msg) && !occursin("_cov_default_link", err.msg)
    end

    @testset "misspelled keyword is named, never swallowed" begin
        err = msg(() -> gllvm(@formula(y ~ 1 + x1), Yp, site; family = Poisson(), K = 1, numlv = 1))
        @test err isa ArgumentError
        @test occursin("`numlv`", err.msg) && !occursin("fit_gllvm_cov", err.msg)
        # an accepted keyword still passes
        @test gllvm(@formula(y ~ 1 + x1), Yp, site; family = Poisson(), K = 1, iterations = 5) isa GLLVModels.AnyGllvmFit
    end

    @testset "function term that cannot take the column" begin
        err = msg(() -> gllvm(@formula(y ~ 1 + scale(x1)), Yg, site; K = 1))
        @test err isa ArgumentError
        @test occursin("scale", err.msg) && occursin("compute the transformed column", err.msg)
    end
end
