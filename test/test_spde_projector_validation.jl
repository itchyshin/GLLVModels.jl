using GLLVModels, Test, SparseArrays

function _arg_error(f)
    try
        f()
        return nothing
    catch e
        e isa ArgumentError || rethrow()
        return e
    end
end

@testset "spde_projector location validation" begin
    nodes, tris = spde_mesh_grid((0.0, 1.0), (0.0, 1.0); nx = 5, ny = 5)

    @testset "non-finite coordinates" begin
        err = _arg_error(() -> spde_projector(nodes, tris, [NaN 0.5; 0.5 0.5]))
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("row 1", msg)
        @test occursin("NaN", msg)

        err = _arg_error(() -> spde_projector(nodes, tris, [0.5 0.5; Inf 0.2]))
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("row 2", msg)
        @test occursin("Inf", msg)
    end

    @testset "far outside mesh" begin
        err = _arg_error(() -> spde_projector(nodes, tris, [0.5 0.5; 1e6 1e6]))
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("row 2", msg)
        @test occursin("outside the mesh", msg)
    end

    @testset "near-boundary snap exposes count" begin
        nsnapped = Ref(0)
        A = spde_projector(nodes, tris, [1.01 0.5; 0.5 0.5]; nsnapped = nsnapped)
        @test size(A) == (2, size(nodes, 1))
        @test nsnapped[] == 1
        @test all(isapprox.(vec(sum(A; dims = 2)), 1.0; atol = 1e-10))
    end
end
