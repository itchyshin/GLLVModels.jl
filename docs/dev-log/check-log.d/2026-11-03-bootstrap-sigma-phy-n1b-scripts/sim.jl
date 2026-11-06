# Simulate small phylo Gaussian dataset (PMV-compatible): tree, d=3, rank 1, explicit unique, shared residual.
using GLLVModels, LinearAlgebra, SparseArrays, Random, DelimitedFiles
out = ARGS[1]; seed = parse(Int, ARGS[2])
rng = MersenneTwister(seed)
ntip = parse(Int, ARGS[3]); reps = parse(Int, ARGS[4]); d = 5
# random ultrametric tree of height 1 (recursive random split)
function rt(names, h, rng)
    if length(names) == 1
        return names[1]
    end
    k = rand(rng, 1:length(names)-1)
    sh = shuffle(rng, names)
    l, r = sh[1:k], sh[k+1:end]
    hl = h * (0.35 + 0.55*rand(rng)) # height of child node
    function sub(g)
        length(g)==1 ? g[1] * ":" * string(h) :
            "(" * rt_inner(g, hl, rng) * "):" * string(h - hl)
    end
    return sub(l) * "," * sub(r)
end
function rt_inner(g, hnode, rng)
    k = rand(rng, 1:length(g)-1); sh = shuffle(rng, g)
    l, r = sh[1:k], sh[k+1:end]
    function sub(gg)
        if length(gg)==1; return gg[1] * ":" * string(hnode); end
        hc = hnode * (0.2 + 0.7*rand(rng))
        return "(" * rt_inner(gg, hc, rng) * "):" * string(hnode - hc)
    end
    return sub(l) * "," * sub(r)
end
tips = ["sp" * lpad(i,2,'0') for i in 1:ntip]
newick = "(" * rt(tips, 1.0, rng) * ");"
write(joinpath(out, "tree.nwk"), newick * "\n")
ap = GLLVModels.augmented_phy(newick)
pp = PrecisionPhy(ap)
Q = Matrix(pp.Q)
A = inv(Symmetric(Q)); A = (A + A') / 2
labels = pp.node_labels
tipidx = [findfirst(==(t), labels) for t in tips]
At = A[tipidx, tipidx]
println("tip cov diag: ", round.(diag(At); digits=4), " scale=", pp.scale)
Lc = cholesky(Symmetric(At + 1e-12I)).L
lam = [0.9, 0.6, -0.5, 0.7, 0.4]; s = [0.50, 0.40, 0.60, 0.30, 0.50]; beta = [0.4, -0.2, 0.3, 0.0, 0.1]; sig = 0.3
g = Lc * randn(rng, ntip)
U = [sqrt(s[t]) .* (Lc * randn(rng, ntip)) for t in 1:d]
m = ntip * reps
Y = zeros(d, m); sid = zeros(Int, m)
for sp in 1:ntip, r in 1:reps
    o = (sp-1)*reps + r; sid[o] = tipidx[sp] - (pp.n_aug - pp.n_leaves)  # PMV tip index = position in leaf_names
    for t in 1:d
        Y[t, o] = beta[t] + lam[t]*g[sp] + U[t][sp] + sig*randn(rng)
    end
end
open(joinpath(out, "long.csv"), "w") do io
    println(io, "species,trait,replicate,value")
    for sp in 1:ntip, t in 1:d, r in 1:reps
        println(io, tips[sp], ",", "t", t, ",", r, ",", repr(Y[t, (sp-1)*reps+r]))
    end
end
writedlm(joinpath(out, "Y.csv"), Y, ',')
writedlm(joinpath(out, "sid.csv"), sid, ',')
# Julia Q for comparison with R engine Q
QI,QJ,QV = findnz(pp.Q)
open(joinpath(out, "Qjulia.csv"), "w") do io
    println(io, "i,j,v"); for k in eachindex(QI); println(io, QI[k], ",", QJ[k], ",", repr(QV[k])); end
end
open(joinpath(out,"Qlabels.csv"),"w") do io; for l in labels; println(io,l); end; end
println("newick: ", newick)
