# Lane cursor-cloud-1125y15 — src/na_input.jl, grouped profile helpers
using GLLVModels, Test, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y15 NA refusal and grouped profile plumbing" begin
    @testset "_refuse_na_response missing cells" begin
        Y = [1.0 missing; 2.0 3.0]
        err = try
            G._refuse_na_response("em_fa", Y)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("em_fa", err.msg)
        @test occursin("missing", err.msg)
        @test occursin("(1, 2)", err.msg)
        G._refuse_na_response("em_fa", [1.0 2.0; 3.0 4.0]) === nothing
    end

    @testset "_refuse_na_response NaN and Inf" begin
        @test_throws ArgumentError G._refuse_na_response("test_fit", [1.0 NaN]; nan = true)
        @test_throws ArgumentError G._refuse_na_response("test_fit", [Inf 2.0]; nan = true)
        G._refuse_na_response("test_fit", [1.0 2.0]; nan = true) === nothing
    end

    @testset "_grouped_profile_drop_coordinate" begin
        v = [10.0, 20.0, 30.0, 40.0]
        @test G._grouped_profile_drop_coordinate(v, 1) == [20.0, 30.0, 40.0]
        @test G._grouped_profile_drop_coordinate(v, 4) == [10.0, 20.0, 30.0]
        @test length(G._grouped_profile_drop_coordinate(v, 2)) == 3
    end

    @testset "_profile_lr_refit guards" begin
        @test_throws ArgumentError G._profile_lr_refit(NaN, 1.0, (_, _) -> nothing)
        @test_throws ArgumentError G._profile_lr_refit(1.0, -1.0, (_, _) -> nothing)
        @test_throws ArgumentError G._profile_lr_refit(1.0, 1.0, (_, _) -> nothing; tol_D = 0)
        bad = G._profile_lr_refit(0.0, 1.0, (v, _) -> (accepted = false, objective_nll = 1.0,
            status = :rejected, provenance = (fixed_variance = v,)))
        @test bad.refit_accepted == false
        @test isnan(bad.lr)
    end

    @testset "_profile_provenance_identity" begin
        prov = (model = :gaussian, rank = 2, stage = :evaluate, fixed_variance = 0.5)
        id = G._profile_provenance_identity(prov)
        @test id.model == :gaussian
        @test id.rank == 2
        @test !haskey(id, :stage)
        @test_throws ArgumentError G._profile_provenance_identity((stage = :x,))
        @test_throws ArgumentError G._profile_provenance_identity("not a named tuple")
    end

    @testset "_profile_trace_reenters" begin
        pts = [
            (fixed_variance = 0.0, lr = 10.0, refit_accepted = true),
            (fixed_variance = 0.5, lr = 1.0, refit_accepted = true),
            (fixed_variance = 1.0, lr = 8.0, refit_accepted = true),
        ]
        @test G._profile_trace_reenters(pts, 0.0, :upper, 3.84, 1e-6) == true
        @test G._profile_trace_reenters(pts, 0.0, :upper, 20.0, 1e-6) == false
    end

    @testset "_grouped_profile_gradient finite difference smoke" begin
        f = x -> dot(x, x)
        g = G._grouped_profile_gradient(f, [1.0, -2.0])
        @test g.components ≈ [2.0, -4.0] rtol = 1e-5
        @test isfinite(g.maximum)
    end
end