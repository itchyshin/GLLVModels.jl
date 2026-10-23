# Julia side of the P1 grouping-level paired Gaussian receipts.
#
# Usage (from the repo root):
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/core070_grouping_p1_julia.jl <out.toml>
#
# Reads the same literal fixture CSVs as tools/core070_grouping_p1_r.R, refuses any
# whose sha256 differs from fixtures/manifest.json, and for each grouping level
#   (a) measures name parity by CALLING fit_gllvm: the paired fit passes the
#       keyword (a successful fit of the requested term is the positive result),
#       and a negative control calls fit_gllvm with a misspelt keyword and
#       records the error;
#   (b) fits the diagonal Gaussian model (GroupingTerm(level; mode=:indep)) by ML,
#       independently of R (default start, no R coordinates), and writes logLik,
#       fixed effects, the level's trait covariance (extract_Sigma, part=:total)
#       and sigma_eps.
using GLLVModels
using SHA
using TOML

const FIXTURES = joinpath("docs", "dev-log", "core070", "true-parity-latest", "receipts", "grouping",
    "fixtures")
const COLUMN = Dict("unit" => "unit", "unit_obs" => "unit_obs", "cluster" => "cluster_id",
    "cluster2" => "cluster2_id")

function read_fixture(level)
    path = joinpath(FIXTURES, level * ".csv")
    manifest = read(joinpath(FIXTURES, "manifest.json"), String)
    want = match(Regex("\"" * level * "\\.csv\": \"([0-9a-f]{64})\""), manifest).captures[1]
    got = bytes2hex(open(sha256, path))
    got == want || error("$path sha256 $got != manifest $want")
    lines = split(chomp(read(path, String)), '\n')
    header = split(lines[1], ',')
    rows = [Dict(zip(header, split(l, ','))) for l in lines[2:end]]
    nobs = maximum(parse(Int, r["obs_row"]) for r in rows)
    Y = fill(NaN, 2, nobs)
    labels = Dict(c => Vector{Symbol}(undef, nobs) for c in ("unit", "unit_obs", "cluster_id", "cluster2_id"))
    for r in rows
        i = parse(Int, r["obs_row"])
        t = r["trait"] == "trait_1" ? 1 : 2
        Y[t, i] = parse(Float64, r["value"])
        for c in keys(labels)
            labels[c][i] = Symbol(r[c])
        end
    end
    all(isfinite, Y) || error("$path does not fill a complete 2 x $nobs matrix")
    return path, got, Y, labels, length(rows)
end

function level_kwargs(level, labels)
    level == "unit" && return (unit = labels["unit"],)
    level == "unit_obs" && return (unit = labels["unit"], unit_obs = labels["unit_obs"])
    level == "cluster" && return (cluster = labels["cluster_id"],)
    level == "cluster2" && return (cluster2 = labels["cluster2_id"],)
    error(level)
end

out_path = only(ARGS)
results = Dict{String,Any}()
for level in ("unit", "unit_obs", "cluster", "cluster2")
    path, digest, Y, labels, nrows = read_fixture(level)
    terms = [GroupingTerm(Symbol(level); mode = :indep)]
    kw = level_kwargs(level, labels)

    bogus = Symbol(level * "_zz_not_a_keyword")
    neg = try
        fit_gllvm(Y; family = GLLVModels.Normal(), grouping = terms, kw...,
            NamedTuple{(bogus,)}((labels[COLUMN[level]],))...)
        nothing
    catch err
        sprint(showerror, err)
    end

    t0 = time()
    fit = fit_gllvm(Y; family = GLLVModels.Normal(), grouping = terms, kw..., iterations = 1000)
    secs = time() - t0
    # Membership control: move observation 1 into the group of the first later
    # observation whose group differs, and refit; a keyword that were accepted but
    # ignored would leave logLik unchanged. Same rule as the R runner, so the moved
    # fits are paired too; for unit_obs the target lies in the same unit.
    col = COLUMN[level]
    moved_labels = copy(labels)
    moved_labels[col] = copy(labels[col])
    target = labels[col][findfirst(!=(labels[col][1]), labels[col])]
    moved_labels[col][1] = target
    moved_fit = fit_gllvm(Y; family = GLLVModels.Normal(), grouping = terms,
        level_kwargs(level, moved_labels)..., iterations = 1000)
    S = extract_Sigma(fit; level = Symbol(level), part = :total).Sigma
    results[level] = Dict(
        "keyword" => level, "keyword_accepted" => true,
        "call" => "fit_gllvm(Y; family=Normal(), grouping=[GroupingTerm(:$level; mode=:indep)], " *
                  join(("$k=<labels>" for k in keys(kw)), ", ") * ", iterations=1000)",
        "fit_type" => string(nameof(typeof(fit))),
        "term_names" => [string(t.name) for t in fit.terms],
        "negative_control" => Dict("keyword" => string(bogus), "rejected" => neg !== nothing,
            "message" => something(neg, "")),
        "membership_control" => Dict("moved_obs_row" => 1, "to_group" => string(target),
            "logLik" => moved_fit.loglik, "converged" => moved_fit.converged),
        "fixture" => path, "fixture_sha256" => digest,
        "n_rows" => nrows, "n_groups" => length(unique(labels[COLUMN[level]])),
        "fit_seconds" => secs, "converged" => fit.converged, "iterations" => fit.iterations,
        "logLik" => fit.loglik, "beta" => collect(fit.parameters[1:2]),
        "Sigma_diag" => [S[1, 1], S[2, 2]], "Sigma_offdiag" => S[1, 2],
        "sigma_eps" => fit.sigma_eps)
end

out = Dict("engine" => "Julia GLLVModels", "julia_version" => string(VERSION),
    "glvmodels_path" => pathof(GLLVModels), "levels" => results)
open(out_path, "w") do io
    TOML.print(io, out; sorted = true)
end
println("wrote ", out_path)
