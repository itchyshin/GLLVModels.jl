# Reproducer (not run by CI) for two namespace rows left unbound by
# test/test_namespace_numeric_p1_twin.jl because the engines disagree.
# Run from the repo root:  julia --project=. test/fixtures/repro_namespace_twin_gaps_p1.jl
# Reads R's recorded values from test/fixtures/ns_numeric_p1.toml (gllvmTMB 9539352f6).
using GLLVModels, TOML, LinearAlgebra
using Distributions: Normal
fx = TOML.parsefile(joinpath(@__DIR__, "ns_numeric_p1.toml"))
tn = String.(fx["trait_names"]); p, n = Int(fx["p"]), Int(fx["n_unit"])
Y = zeros(p, n)
for line in eachline(joinpath(@__DIR__, fx["main"]["data_file"]))
    startswith(line, "\"unit\"") && continue
    a = split(line, ",")
    Y[findfirst(==(strip(a[2], '"')), tn), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
end

# 1. vcov.gllvmTMB_multi: R returns the full 6 x 6 covariance of the trait means; Julia's public
#    vcov(fit, Y) returns Diagonal(se^2) over ALL parameters, so every off-diagonal is 0.
fit = fit_gllvm(Y; family = Normal(), K = 2)
V = Matrix(vcov(fit, Y))[1:p, 1:p]
Vr = permutedims(reshape(Float64.(fx["main"]["vcov"]), p, p))
println("vcov: max |diag diff| = ", maximum(abs.(diag(V) .- diag(Vr))),
        "   max |R off-diagonal| = ", maximum(abs.(Vr - Diagonal(diag(Vr)))),
        "   max |Julia off-diagonal| = ", maximum(abs.(V - Diagonal(diag(V)))))

# 2. extract_communality / extract_proportions under unique at one tier. Same likelihood, different split:
g = fx["gap_unique"]
fu = fit_gllvm(Y; family = Normal(), K = 2, has_diag = true)
println("unique fit: logLik Julia = ", fu.logLik, "  R = ", g["loglik"])
println("communality  R     = ", round.(Float64.(g["communality_unit"]); digits = 4), "   (R sigma_eps = ", g["sigma_eps"], ")")
println("communality  Julia = ", round.(extract_communality(fu); digits = 4), "   (Julia sigma_eps = ", fu.pars.σ_eps, ")")
println("Julia splits the per-trait residual across sigma_eps, sigma2_B and sigma2_W (not separately identified with one")
println("observation per cell); R pins sigma_eps near 0 and puts it all in Psi_B.")
