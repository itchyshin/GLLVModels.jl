# namespace-3 (D5) batch, Julia half: matched point and one independent fit on the R half's draw.
#
# usage: julia --project=<env with GLLVModels> tools/core070_namespace_3_d5_batch.jl <outdir>
# <outdir> is the R half's output directory (tools/core070_namespace_3_d5_batch.R). Writes
# julia-results.json there. Applies no pass/fail bar: tools/core070_namespace_3_p1_receipts.py
# recomputes every difference from the raw values and applies the contract's bars.
#
# Step 1 (matched point): R's parameters, mapped by TMB block name into Julia's packed vector
#   [beta; loading; log_sd_psi_shared; log_sd_species], evaluated with residual_mode = :shared
#   (iterations = 0), then extract_phylo_signal at that point.
# Step 2 (independent fit): one fit from the default start, residual_mode = :shared, 1500 iterations.
using GLLVModels, DelimitedFiles, Printf, LinearAlgebra
const G = GLLVModels

length(ARGS) == 1 || error("usage: julia --project=<env> tools/core070_namespace_3_d5_batch.jl <outdir>")
od = ARGS[1]

readtable(path) = (m = readdlm(path, ','; header = true); (string.(m[2][:]), m[1]))

phy, tips = G._phylo_latent_tree_precision(strip(read(joinpath(od, "tree.nwk"), String)))
hdr, tab = readtable(joinpath(od, "frozen_A.csv"))
col(name) = tab[:, findfirst(==(name), hdr)]
species, site, trait, value = string.(col("species")), string.(col("site")),
    string.(col("trait")), Float64.(col("value"))
trait_names = unique(trait); site_names = unique(site)
p = length(trait_names); nsite = length(site_names); nsp = length(tips)
sid_of(s) = (i = findfirst(==(s), tips); i === nothing && error("species $s is not a tree tip"); i)
n = nsite * nsp
Y = fill(NaN, p, n); species_id = zeros(Int, n)
for r in eachindex(value)               # observations are site-major: obs = (site - 1) * nsp + tip
    o = (findfirst(==(site[r]), site_names) - 1) * nsp + sid_of(species[r])
    Y[findfirst(==(trait[r]), trait_names), o] = value[r]
    species_id[o] = sid_of(species[r])
end
all(isfinite, Y) && all(>(0), species_id) || error("data are not a complete species x site x trait grid")
terms = [GroupingTerm(:cluster; mode = :indep, common = false)]
common = (; rank = 1, phylo_mode = :barelowrank, terms = terms, cluster = species_id,
    species_id = species_id, residual_mode = :shared)

_, ptab = readtable(joinpath(od, "R_params_A.csv"))
pname = string.(ptab[:, 1]); pval = Float64.(ptab[:, 3])
block(name) = pval[[i for i in eachindex(pname) if pname[i] == name]]
want = Dict("b_fix" => p, "theta_rr_phy" => p, "log_sigma_eps" => 1, "theta_diag_species" => p)
bad = [k for (k, len) in want if length(block(k)) != len]
if !isempty(bad)
    println("R-to-Julia map FAILED for: ", join(bad, ", "))
    println("names saved by R: ", join(sort(unique(pname)), ", "))
    exit(2)
end
theta = vcat(block("b_fix"), block("theta_rr_phy"), block("log_sigma_eps"), block("theta_diag_species"))

at = G.fit_joint_phylo_grouped_gaussian(Y, phy; common..., start = theta, iterations = 0)
sig_at = G.extract_phylo_signal(at; species_level = :cluster)
t0 = time()
fit = G.fit_joint_phylo_grouped_gaussian(Y, phy; common..., iterations = 1500)
el = time() - t0
sig = G.extract_phylo_signal(fit; species_level = :cluster)

arr(v) = "[" * join((repr(Float64(x)) for x in v), ", ") * "]"
q(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
open(joinpath(od, "julia-results.json"), "w") do io
    println(io, "{")
    println(io, "  \"schema\": \"core070-namespace-3-d5-julia-results/v1\",")
    println(io, "  \"julia_version\": ", q(VERSION), ",")
    println(io, "  \"julia_threads\": ", Threads.nthreads(), ",")
    println(io, "  \"check_bounds\": ", q(Base.JLOptions().check_bounds), ",")
    println(io, "  \"matched\": {\"logLik\": ", repr(Float64(at.loglik)), ", \"H2\": ", arr(sig_at.H2),
        ", \"Psi\": ", arr(sig_at.Psi), ", \"V_eta\": ", arr(sig_at.V_eta), "},")
    println(io, "  \"independent\": {\"logLik\": ", repr(Float64(fit.loglik)), ", \"H2\": ", arr(sig.H2),
        ", \"Psi\": ", arr(sig.Psi), ", \"V_eta\": ", arr(sig.V_eta), ", \"converged\": ", fit.converged,
        ", \"stopping_reason\": ", q(fit.stopping_reason), ", \"iterations_cap\": 1500, \"elapsed_s\": ", repr(el), "}")
    println(io, "}")
end
@printf("matched logLik %.12f; independent logLik %.12f converged=%s %.2f s\n", at.loglik, fit.loglik, fit.converged, el)
