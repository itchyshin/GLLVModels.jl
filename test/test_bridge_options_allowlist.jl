using GLLVModels, Test, Random, LinearAlgebra

# Refs #936: each bridge_fit route rejects unknown options keys with a route tag.

@testset "bridge_fit per-route options allow-list (#936)" begin
    Random.seed!(936)
    Y = randn(4, 24)
    Yc = rand(0:4, 4, 24)

    @testset "one-part poisson rejects stray keys" begin
        base = bridge_fit(; y = Float64.(Yc), family = "poisson", d = 1,
            options = Dict("ci_method" => "none"))
        @test base.converged
        err = @test_throws ArgumentError bridge_fit(; y = Float64.(Yc),
            family = "poisson", d = 1,
            options = Dict("ci_method" => "none", "row_eff" => "fixed"))
        @test occursin("one-part poisson", sprint(showerror, err.value))
        @test occursin("row_eff", sprint(showerror, err.value))
        err2 = @test_throws ArgumentError bridge_fit(; y = Float64.(Yc),
            family = "poisson", d = 1, options = Dict("reml" => true))
        @test occursin("reml", sprint(showerror, err2.value))
    end

    @testset "one-part gaussian REML route keeps reml" begin
        br = bridge_fit(; y = Y, family = "gaussian", d = 1,
            options = Dict("reml" => true))
        @test br.model == "gaussian_reml_rr"
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = 1,
            options = Dict("reml" => true, "phylo_model" => "diagnostic"))
    end

    @testset "mixed-family rejects unlisted CI extras" begin
        fam = ["gaussian", "poisson", "binomial"]
        Ymix = [
            0.2 0.4 -0.1 0.3 0.5 -0.2 0.1 0.6
            1.0 3.0 2.0 4.0 1.0 2.0 5.0 3.0
            0.0 1.0 1.0 0.0 1.0 0.0 1.0 1.0
        ]
        br = bridge_fit(; y = Ymix, family = fam, d = 1,
            options = Dict("ci_method" => "wald"))
        @test br.model == "mixed_rr"
        @test_throws ArgumentError bridge_fit(; y = Ymix, family = fam, d = 1,
            options = Dict("ci_method" => "wald", "ci_nboot" => 50))
    end

    @testset "gaussian sources rejects unknown keys" begin
        n = size(Y, 2)
        C = Matrix{Float64}(I, n, n)
        spec = Dict{String, Any}(
            "name" => "ord", "covariance" => C, "groups" => collect(1:n),
            "mode" => "latent", "rank" => 1, "unique" => false, "common" => false)
        br = bridge_fit(; y = Y, family = "gaussian", d = 1, sources = [spec],
            options = Dict("ci_method" => "none", "iterations" => 200))
        @test isfinite(br.loglik)
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = 1,
            sources = [spec],
            options = Dict("ci_method" => "none", "reml" => false))
    end

    @testset "phylo diagnostic allows only phylo_model" begin
        pp = PrecisionPhy(augmented_phy(
            "(((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.1,((E:0.1,F:0.1):0.1,(G:0.1,H:0.1):0.1):0.1);");
            correlation = false)
        yv = randn(pp.n_leaves)
        br = bridge_fit(; y = yv, family = "gaussian", phylo = pp,
            options = Dict("phylo_model" => "diagnostic"))
        @test br.diagnostic_only
        @test_throws ArgumentError bridge_fit(; y = yv, phylo = pp,
            options = Dict("ci_method" => "none"))
    end
end