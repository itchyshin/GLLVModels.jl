using GLLVModels, Test, Random, LinearAlgebra

# Issue #729: `pd` must use `isposdef` on `I_obs` (not diag(inv(I_obs)) > 0) and
# must be false when `emf.converged` is false.

@testset "em_observed_information pd guard (#729)" begin

    @testset "indefinite I_obs with positive diag(inv)" begin
        I_bad = [-0.9885664662877819 0.5778239535645899 -1.4515847952585692;
                  0.5778239535645899 1.4759754167072543 1.416776309318;
                 -1.4515847952585692 1.416776309318 0.6596308520101298]
        @test !isposdef(Symmetric(I_bad))
        @test all(diag(inv(Symmetric(I_bad))) .> 0)
        @test !GLLVModels._observed_information_is_pd(I_bad)
    end

    @testset "unconverged EMPhyloFit never reports pd" begin
        Random.seed!(1)
        p, n = 8, 60
        Σ = [0.6^abs(i - j) for i in 1:p, j in 1:p]
        y = cholesky(Symmetric(Σ)).L * randn(p, n) .+ 0.5 .* randn(p, n)
        emf = em_fit_phylo(y, 1, Σ; max_iter = 2)
        @test !emf.converged
        oi = em_observed_information(emf, y, Σ)
        @test !oi.pd
    end
end
