using GLLVModels, Test, Random

# Geometry-only checks. No SPDE fit, no precision matrix.

function _tri_signed_area(ax, ay, bx, by, cx, cy)
    return 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
end

function _mesh_area(nodes, tris)
    s = 0.0
    for k in 1:size(tris, 1)
        a = tris[k, 1]; b = tris[k, 2]; c = tris[k, 3]
        s += _tri_signed_area(nodes[a, 1], nodes[a, 2],
                              nodes[b, 1], nodes[b, 2],
                              nodes[c, 1], nodes[c, 2])
    end
    return s
end

@testset "SPDE Delaunay in-circle cutoff scales with extent (#748)" begin
    # Unit-square hull with two interior points (order-1 coordinates).
    pts = [0.0 0.0;
           1.0 0.0;
           1.0 1.0;
           0.0 1.0;
           0.5 0.25;
           0.3 0.7]

    @testset "ordinary unit-square mesh still builds" begin
        nodes, tris = GLLVModels.spde_mesh_delaunay(pts)
        @test size(nodes, 1) == 6
        @test size(tris, 1) >= 4
        @test all(1 .<= tris .<= 6)
        area = _mesh_area(nodes, tris)
        @test area ≈ 1.0 atol = 1e-12
        for k in 1:size(tris, 1)
            a = tris[k, 1]; b = tris[k, 2]; c = tris[k, 3]
            @test _tri_signed_area(nodes[a, 1], nodes[a, 2],
                                   nodes[b, 1], nodes[b, 2],
                                   nodes[c, 1], nodes[c, 2]) > 0
        end
    end

    @testset "1e-8 scale does not silently return a bad mesh" begin
        nodes_u, tris_u = GLLVModels.spde_mesh_delaunay(pts)
        area_u = _mesh_area(nodes_u, tris_u)

        scale = 1e-8
        nodes_t, tris_t = GLLVModels.spde_mesh_delaunay(pts .* scale)
        @test size(nodes_t, 1) == 6
        # Same connectivity as the order-1 mesh: not sparse, not overlapping.
        @test size(tris_t, 1) == size(tris_u, 1)
        area_t = _mesh_area(nodes_t, tris_t)
        @test area_t / scale^2 ≈ area_u rtol = 1e-8
        for k in 1:size(tris_t, 1)
            a = tris_t[k, 1]; b = tris_t[k, 2]; c = tris_t[k, 3]
            @test _tri_signed_area(nodes_t[a, 1], nodes_t[a, 2],
                                   nodes_t[b, 1], nodes_t[b, 2],
                                   nodes_t[c, 1], nodes_t[c, 2]) > 0
        end
    end
end

# Hull coverage (#1221): the mesh must cover the whole convex hull.
function _hull_vertices(P)
    idx = sortperm(1:size(P, 1); by = i -> (P[i, 1], P[i, 2]))
    cr(o, a, b) = (P[a, 1] - P[o, 1]) * (P[b, 2] - P[o, 2]) - (P[a, 2] - P[o, 2]) * (P[b, 1] - P[o, 1])
    function half(order)
        h = Int[]
        for i in order
            while length(h) >= 2 && cr(h[end-1], h[end], i) <= 0
                pop!(h)
            end
            push!(h, i)
        end
        pop!(h)
        return h
    end
    return vcat(half(idx), half(reverse(idx)))
end

@testset "SPDE Delaunay covers the full convex hull (#1221)" begin
    for scale in (1e-3, 1.0, 1e3), seed in (1, 2, 3), N in (30, 200)
        rng = Random.MersenneTwister(seed)
        pts = scale .* (rand(rng, N, 2) .+ [5.0 -3.0])
        nodes, tris = GLLVModels.spde_mesh_delaunay(pts)
        hull = _hull_vertices(pts)
        H = length(hull)
        hx = pts[hull, 1]; hy = pts[hull, 2]
        hull_area = 0.5 * abs(sum(hx .* circshift(hy, -1) .- circshift(hx, -1) .* hy))
        @test _mesh_area(nodes, tris) ≈ hull_area rtol = 1e-9
        @test size(tris, 1) == 2N - 2 - H
    end
end
