# refs #730: `K` (or `num_lv`) that is negative or not an integer is an ArgumentError
# naming `K` at the two public entry points, `fit_gllvm` and `gllvm`. Before, `K = -1`
# surfaced as "invalid Array dimensions" from inside the fitter and `K = 1.0` as a
# TypeError on the fitter's keyword. `K = 0` (a GLM fit) and `K > p` are not touched here.
using Test, GLLVModels, Random, Distributions

@testset "K argument check at fit_gllvm and gllvm" begin
    rng = MersenneTwister(3)
    p, n = 5, 40
    Yc = rand(rng, 0:6, p, n)
    Yb = rand(rng, 0:1, p, n)
    Yp = exp.(randn(rng, p, n))
    Yu = rand(rng, p, n) .* 0.8 .+ 0.1
    site = (x1 = randn(rng, n),)

    thrown(f) = try; f(); nothing; catch e; e; end
    isK(e, caller) = e isa ArgumentError && startswith(e.msg, "$caller: K must be a non-negative integer") &&
                     occursin("got K = ", e.msg)

    cases = [
        ("Poisson",         Yc, (; family = Poisson())),
        ("NB2 shared",      Yc, (; family = NegativeBinomial())),
        ("NB2 per species", Yc, (; family = NegativeBinomial(), disp_group = :species)),
        ("ZIP",             Yc, (; family = ZIPoisson())),
        ("Gamma",           Yp .+ 0.1, (; family = Gamma())),
        ("Beta",            Yu, (; family = Beta())),
        ("Binomial",        Yb, (; family = Binomial())),
        ("Poisson row_eff", Yc, (; family = Poisson(), row_eff = :fixed)),
    ]
    for (label, Y, kw) in cases
        @testset "$label" begin
            @test isK(thrown(() -> fit_gllvm(Y; K = -1, kw...)), "fit_gllvm")
            @test isK(thrown(() -> fit_gllvm(Y; K = 1.0, kw...)), "fit_gllvm")
            @test isK(thrown(() -> fit_gllvm(Y; num_lv = -2, kw...)), "fit_gllvm")
        end
    end

    @testset "Gaussian keeps its own range message" begin
        e = thrown(() -> fit_gllvm(Float64.(Yc); K = -1))
        @test e isa ArgumentError && occursin("K must lie in 1:p", e.msg)
        @test isK(thrown(() -> fit_gllvm(Float64.(Yc); K = 1.0)), "fit_gllvm")
    end

    @testset "gllvm() front end" begin
        @test isK(thrown(() -> gllvm(@formula(y ~ 1), Yc, site; family = Poisson(), K = -1)), "gllvm")
        @test isK(thrown(() -> gllvm(@formula(y ~ 1 + x1), Yc, site; family = Poisson(), K = -1)), "gllvm")
        @test isK(thrown(() -> gllvm(@formula(y ~ 1 + x1), Yc, site;
                                     family = NegativeBinomial(), K = -3)), "gllvm")
    end

    @testset "legal K is unchanged" begin
        @test isfinite(fit_gllvm(Yc; family = Poisson(), K = 0).loglik)     # GLM fit, not an error
        @test isfinite(fit_gllvm(Yc; family = Poisson(), K = 1).loglik)
        @test isfinite(fit_gllvm(Yc; family = Poisson(), num_lv = 1).loglik)
        # K > p keeps the range message of the packing layer
        e = thrown(() -> fit_gllvm(Yc; family = Poisson(), K = p + 3))
        @test e isa ArgumentError && occursin("K must lie in 0:p", e.msg)
        # K and num_lv that disagree keep their own error
        e = thrown(() -> fit_gllvm(Yc; family = Poisson(), K = 1, num_lv = 2))
        @test e isa ArgumentError && occursin("disagree", e.msg)
    end
end
