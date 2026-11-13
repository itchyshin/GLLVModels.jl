# P81 ZERO-SITES (refs #1188): a response matrix with no sites (p x 0) or no species (0 x n)
# is an ArgumentError naming the empty dimension. Before, p x 0 returned logLik = 0.0 with
# converged = true (Poisson, Binomial, Gamma, NB2) and 0 x n ended in a raw BoundsError or
# an empty-reduction error inside the fitter.
using Test, GLLVModels, Random, Distributions

@testset "empty Y is an ArgumentError naming the empty dimension (fit_gllvm)" begin
    thrown(f) = try; f(); nothing; catch e; e; end
    fams = [("Poisson", Poisson(), Int), ("Binomial", Binomial(), Int), ("Gamma", Gamma(), Float64),
            ("NB2", NegativeBinomial(), Int), ("Beta", Beta(), Float64), ("ZIP", ZIPoisson(), Int),
            ("Gaussian", Normal(), Float64)]
    for (label, fam, T) in fams
        @testset "$label" begin
            e = thrown(() -> fit_gllvm(zeros(T, 5, 0); family = fam, K = 1))
            @test e isa ArgumentError
            @test e isa ArgumentError && occursin("no sites (0 columns)", e.msg) && startswith(e.msg, "fit_gllvm:")
            e = thrown(() -> fit_gllvm(zeros(T, 0, 8); family = fam, K = 0))
            @test e isa ArgumentError
            @test e isa ArgumentError && occursin("no species (0 rows)", e.msg)
        end
    end
    @testset "both empty, ordinal, and the per-column route" begin
        e = thrown(() -> fit_gllvm(zeros(Int, 0, 0); family = Poisson(), K = 0))
        @test e isa ArgumentError && occursin("no species and no sites", e.msg)
        e = thrown(() -> fit_gllvm(zeros(Int, 4, 0); family = GLLVModels.Ordinal(), K = 1))
        @test e isa ArgumentError && occursin("no sites", e.msg)
        e = thrown(() -> fit_gllvm(zeros(Int, 4, 0); family = [Poisson(), Poisson(), Poisson(), Poisson()], K = 1))
        @test e isa ArgumentError && occursin("no sites", e.msg)
    end
    @testset "gllvm() front end" begin
        e = thrown(() -> gllvm(@formula(y ~ 1), zeros(Int, 5, 0), (x1 = Float64[],); family = Poisson(), K = 1))
        @test e isa ArgumentError && occursin("no sites", e.msg)
    end
    @testset "non-empty Y is unchanged" begin
        Y = rand(MersenneTwister(1), 0:4, 4, 30)
        @test isfinite(fit_gllvm(Y; family = Poisson(), K = 1).loglik)
        @test isfinite(fit_gllvm(Float64.(Y) .+ randn(MersenneTwister(2), 4, 30); K = 1).logLik)
    end
end
