# #754: fit_branch_re must not report an interior converged fit at the
# variance floor, and must reject constant / non-finite y by name.
# These three cases are the issue repro; each would fail on origin/main
# e6cfcd4ee (noise-only: converged=true, no boundary; constant y: logLik
# +11870; NaN y: ErrorException pointing at the optimiser).

using GLLVModels, Test, Random, Statistics

function _newick_balanced_32(; bl::Real = 0.3)
    nodes = ["L$(t):" * string(bl) for t in 1:32]
    while length(nodes) > 1
        nn = String[]
        i = 1
        while i + 1 <= length(nodes)
            push!(nn, "(" * nodes[i] * "," * nodes[i + 1] * "):" * string(bl))
            i += 2
        end
        i == length(nodes) && push!(nn, nodes[i])
        nodes = nn
    end
    return nodes[1] * ";"
end

@testset "fit_branch_re convergence and y checks (#754)" begin
    phy = edge_phy(_newick_balanced_32())

    @testset "noise-only fit reports a variance boundary" begin
        Random.seed!(5)
        y = randn(32)
        f = fit_branch_re(phy, y)
        vy = var(y; corrected = false)
        @test f.sigma2_boundary
        @test !f.converged
        @test f.σ² <= GLLVModels._BRANCH_RE_VARIANCE_BOUNDARY_TOL * vy
        @test isfinite(f.negll)
        # A huge positive log-likelihood is the constant-y failure mode, not this one.
        @test f.negll > 0
    end

    @testset "boundary flags are invariant to rescaling y" begin
        Random.seed!(5)
        y0 = randn(32)
        flags = Tuple{Bool,Bool}[]
        for c in (1.0, 1e-5, 1e5)
            f = fit_branch_re(phy, c .* y0)
            push!(flags, (f.converged, f.sigma2_boundary))
        end
        @test flags[1] == flags[2] == flags[3]
        @test flags[1] == (false, true)
    end

    @testset "constant y raises ArgumentError naming the data" begin
        err = try
            fit_branch_re(phy, fill(1.0, 32))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("y", msg)
        @test occursin("constant", msg)
        @test !occursin("no finite likelihood", msg)
    end

    @testset "NaN y raises ArgumentError naming the data" begin
        Random.seed!(5)
        err = try
            fit_branch_re(phy, vcat(randn(31), NaN))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("y", msg)
        @test occursin("non-finite", msg)
        @test occursin("32", msg) || occursin(r"indices", msg)
        @test !occursin("no finite likelihood", msg)
    end
end
