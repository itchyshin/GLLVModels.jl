# N1b Section 5: Julia simulator moments. Per-draw statistics: 15 within-observation trait covariances and 15 same-species
# cross-replicate trait covariances (species-level covariance), centred on the fitted mean. Also writes theory from the fit.
using GLLVModels, DelimitedFiles, Printf, Random, LinearAlgebra, Statistics
const G = GLLVModels
ds = ARGS[1]; root = ARGS[2]; od = ARGS[3]; nsim = parse(Int, ARGS[4]); mkpath(od)
dd = joinpath(root, "data", "d" * ds)
Y = readdlm(joinpath(dd, "Y.csv"), ',', Float64)
sid = Int.(vec(readdlm(joinpath(dd, "sid.csv"), ',')))
pp = PrecisionPhy(G.augmented_phy(strip(read(joinpath(dd, "tree.nwk"), String))))
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
fit = quiet() do
    G.fit_precision_multivariate(Y, pp; rank=1, mode=:explicitunique, residual_mode=:shared, species_id=sid)
end
e = Dict(G._bsig_entries(fit, [:phy], [:Sigma]))
Sg = Matrix(e[:Sigma_phy]); d, m = fit.response_shape
rv = fit.residual_variance
Tw = Sg + Diagonal(fill(rv isa Number ? rv : rv[1], d)); Tc = Sg
mu = reshape(fit.mean_design * fit.beta, d, m)
@printf("tip variance check: scale %.6f; residual %s; trait-4 Sigma_phy diag %.5f\n", pp.scale, string(rv), Sg[4,4])
lt = [(i, j) for j in 1:d for i in j:d]
# pairs of observations sharing a species
by = Dict{Int,Vector{Int}}(); for o in 1:m; push!(get!(by, sid[o], Int[]), o); end
pairs = [(v[1], v[2]) for v in values(by) if length(v) >= 2]
open(joinpath(od, "theory_$(ds).csv"), "w") do io
    println(io, "stat,i,j,theory"); for (i,j) in lt; println(io, "within,$i,$j,", @sprintf("%.17g", Tw[i,j])); end
    for (i,j) in lt; println(io, "cross,$i,$j,", @sprintf("%.17g", Tc[i,j])); end
end
function stats(Yb)
    r = Yb .- mu
    w = (r * r') ./ m
    c = zeros(d, d)
    for (a, b) in pairs; c .+= r[:, a] * r[:, b]'; end
    c = (c + c') ./ (2 * length(pairs))
    vcat([w[i,j] for (i,j) in lt], [c[i,j] for (i,j) in lt])
end
Ys = G._bsig_pmv_simulate_batch(Random.MersenneTwister(31337), fit, nsim)
S = reduce(hcat, [stats(Yb) for Yb in Ys])'
open(joinpath(od, "s5J_$(ds).csv"), "w") do io
    for b in 1:nsim; println(io, join([@sprintf("%.17g", v) for v in S[b, :]], ",")); end
end
println("DONE s5 J")
