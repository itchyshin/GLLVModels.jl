# Repro for #754: fit_branch_re false convergence / constant-y / NaN-y
# Recorded against origin/main at e6cfcd4ee7b5bcf1ba4ff2f0b6413bf80f31efe3
# (issue pin was 824d22a4; behaviour confirmed on current main).
#
# Usage:
#   julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_754_branch_re.jl

using GLLVModels, Random

function _newick_balanced(p::Integer; bl::Real = 0.3)
    p > 1 || error("p must be > 1")
    nodes = ["L$(t):" * string(bl) for t in 1:p]
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

println("commit = ", readchomp(`git rev-parse HEAD`))
phy = edge_phy(_newick_balanced(32))
Random.seed!(5)

print("noise only: ")
f = fit_branch_re(phy, randn(32))
println("s2=", f.σ², "  s2eps=", f.σ²_eps,
        "  conv=", f.converged, "  iters=", f.n_iter,
        "  negll=", f.negll,
        hasproperty(f, :sigma2_boundary) ? "  boundary=$(f.sigma2_boundary)" : "")

print("constant y: ")
try
    g = fit_branch_re(phy, fill(1.0, 32))
    println("s2=", g.σ², "  s2eps=", g.σ²_eps,
            "  conv=", g.converged, "  negll=", g.negll)
catch e
    println(typeof(e), " ", sprint(showerror, e))
end

print("NaN in y:   ")
try
    h = fit_branch_re(phy, vcat(randn(31), NaN))
    println("conv=", h.converged, "  negll=", h.negll)
catch e
    println(typeof(e), " ", sprint(showerror, e))
end
