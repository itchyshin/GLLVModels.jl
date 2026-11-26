# Lane cursor-cloud-1125y8: grouped_profile_interval.jl private adapters.
using Test, GLLVModels, Random

const G = GLLVModels

@testset "cov 1125y8 grouped profile helpers" begin
    @testset "_grouped_profile_unavailable_side" begin
        s = G._grouped_profile_unavailable_side(:unavailable)
        @test s.status === :unavailable && isnan(s.endpoint) && isempty(s.points)
        s2 = G._grouped_profile_unavailable_side(:foo, [(a = 1,)])
        @test length(s2.points) == 1
    end

    @testset "_grouped_profile_collect_receipts" begin
        a = (points = [1, 2],)
        b = [3, 4]
        @test G._grouped_profile_collect_receipts(a, nothing, b) == [1, 2, 3, 4]
        @test G._grouped_profile_collect_receipts(5) == [5]
        @test isempty(G._grouped_profile_collect_receipts(nothing, nothing))
    end

    @testset "_grouped_profile_interval_result shape" begin
        refitter = (baseline_nll = 10.0, selected_full_index = 3, selected_label = "x",
            provenance = (response = :y,))
        lower = (status = :root_verified, endpoint = 0.1)
        upper = (status = :root_verified, endpoint = 2.0)
        out = G._grouped_profile_interval_result(:available, :ok, refitter, 3.84, (fixed_variance = 0.5,),
            lower, upper, [lower, upper])
        @test out.status === :available
        @test out.lower === lower
        @test length(out.receipts) == 2
        @test out.selected_label == "x"
    end

    @testset "_grouped_gaussian_variance_profile argument guards" begin
        Y = reshape(randn(12), 1, :)
        fit = G.fit_grouped_gaussian(Y; terms = [G.GroupingTerm(:unit; mode = :indep, common = false)],
            unit = repeat(1:6; inner = 2), iterations = 50)
        @test_throws ArgumentError G._grouped_gaussian_variance_profile(fit, Y; level = 1.0)
        @test_throws ArgumentError G._grouped_gaussian_variance_profile(fit, Y; max_expand = 0)
        @test_throws ArgumentError G._grouped_gaussian_variance_profile(fit, Y; maxiter = -1)
    end
end