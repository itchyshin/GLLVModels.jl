#!/usr/bin/env julia
# Pre-run Julia side (campaign plan 4.3). Usage: julia --project=<GLLVModels.jl> prerun_J.jl <cell>
using GLLVModels, Distributions, LinearAlgebra, Printf, DelimitedFiles
cell = ARGS[1]
TAG = get(ENV, "PRERUN_TAG", cell)   # diagnostic runs only: reduced-size subsets
NSUB = parse(Int, get(ENV, "PRERUN_NSUB", "0"))   # sites (or series for temporal); 0 = all
PSUB = parse(Int, get(ENV, "PRERUN_PSUB", "0"))   # species (or occasions for temporal); 0 = all
mkpath("out")
lines = String[]
emit(s) = (push!(lines, s); open(f -> foreach(l -> println(f, l), lines), joinpath("out", "$(TAG)_J_summary.txt"), "w"))
emit("cell=$cell tag=$TAG nsub=$NSUB psub=$PSUB"); emit("julia_version=$(VERSION)")
emit("gllvmodels_commit=$(get(ENV, "GLLVM_JL_SHA", "unknown"))")

function readcsv(path)  # numeric/quoted CSV -> (header, rows of strings)
    ls = readlines(path); hdr = [strip(x, '"') for x in split(ls[1], ",")]
    rows = [[strip(x, '"') for x in split(l, ",")] for l in ls[2:end] if !isempty(l)]
    return hdr, rows
end
col(hdr, rows, name) = [r[findfirst(==(name), hdr)] for r in rows]

function try_confint(f, args...; kwargs...)
    t = time()
    try
        ci = confint(args...; kwargs...)
        emit("pd_hessian=$(hasproperty(ci, :pd_hessian) ? ci.pd_hessian : "NA")")
        emit("wall_confint_sec=$(@sprintf("%.3f", time() - t))")
    catch e
        emit("confint_error=$(first(sprint(showerror, e), 300))")
        emit("wall_confint_sec=$(@sprintf("%.3f", time() - t))")
    end
end

if cell == "ordinal"
    hdr, rows = readcsv("data/ordinal_p20_n500_K2.csv")
    site = parse.(Int, col(hdr, rows, "site")); tr = col(hdr, rows, "trait"); val = parse.(Int, col(hdr, rows, "value"))
    names_ = sort(unique(tr)); p = length(names_); n = maximum(site)
    Y = zeros(Int, p, n); for (s, t, v) in zip(site, tr, val); Y[findfirst(==(t), names_), s] = v; end
    t0 = time(); fit = fit_gllvm(Y; family = Ordinal(), K = 2); wall = time() - t0
    emit("wall_fit_sec=$(@sprintf("%.3f", wall))"); emit("converged=$(fit.converged)"); emit("logLik=$(@sprintf("%.10f", fit.loglik))")
    try_confint(fit, fit, Y)
elseif cell == "temporal"
    hdr, rows = readcsv("data/temporal_s25_t20_p20_d1.csv")
    sr = String.(col(hdr, rows, "series")); oc = parse.(Float64, col(hdr, rows, "occasion"))
    keep = trues(length(sr))
    NSUB > 0 && (keep .&= [parse(Int, s[2:end]) <= NSUB for s in sr])
    PSUB > 0 && (keep .&= oc .<= PSUB)
    tbl = (series = sr[keep], trait = String.(col(hdr, rows, "trait"))[keep],
           occasion = oc[keep], value = parse.(Float64, col(hdr, rows, "value"))[keep])
    t0 = time()
    fit = fit_temporal_gllvm(tbl; formula = @formula(value ~ 0 + trait),
        temporal = temporal_latent(:(0 + trait | series), :occasion; d = 1, structure = :ar1, unique = false),
        g_tol = 1e-8, iterations = 2000)
    wall = time() - t0
    emit("wall_fit_sec=$(@sprintf("%.3f", wall))"); emit("converged=$(fit.converged)")
    emit("logLik=$(@sprintf("%.10f", fit.loglik))")
    for nm in (:iterations, :optimizer_converged, :gradient_norm); hasproperty(fit, nm) && emit("$nm=$(getproperty(fit, nm))"); end
elseif cell == "isdm"
    include(joinpath(dirname(pathof(GLLVModels)), "..", "test", "fixtures", "isdm", "isdm_fixture_io.jl"))
    d = read_isdm_csv(abspath("data/isdm_c500_sp20_s2_K2.csv"))
    cl = (Binomial(), CLogLogLink())
    fm = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 2, unique = FALSE))
    t0 = time()
    ft = fit_isdm_gllvm(fm, d; family = isdm_sources(gbif = Poisson(), survey = cl), trait = :trait, unit = :cell_id)
    wall = time() - t0
    emit("wall_fit_sec=$(@sprintf("%.3f", wall))"); emit("converged=$(ft.converged)"); emit("cells_converged=$(all(ft.cell_converged))")
    emit("iterations=$(ft.iterations)"); emit("logLik=$(@sprintf("%.10f", ft.loglik))")
elseif cell in ("beetle", "fungi")
    hdr, rows = readcsv("data/$(cell)_wide.csv")
    covs = cell == "beetle" ? ["pH", "Moist", "Org"] : ["TEMPR", "PRECIP", "logAREA"]
    sp = setdiff(hdr, vcat("site", covs)); PSUB > 0 && (sp = sp[1:PSUB]); p = length(sp)
    NSUB > 0 && (rows = rows[1:NSUB]); n = length(rows)
    Y = Int[parse(Int, rows[s][findfirst(==(sp[t]), hdr)]) for t in 1:p, s in 1:n]
    covdata = NamedTuple{Tuple(Symbol.(covs))}(Tuple([parse.(Float64, col(hdr, rows, c)) for c in covs]))
    fam = cell == "beetle" ? NegativeBinomial() : Binomial()
    f = cell == "beetle" ? @formula(y ~ 1 + pH + Moist + Org) : @formula(y ~ 1 + TEMPR + PRECIP + logAREA)
    t0 = time(); fit = gllvm(f, Y, covdata; family = fam, K = 2); wall = time() - t0
    emit("fit_type=$(typeof(fit))"); emit("wall_fit_sec=$(@sprintf("%.3f", wall))")
    emit("converged=$(fit.converged)"); emit("logLik=$(@sprintf("%.10f", fit.loglik))")
else
    error("unknown cell $cell")
end
emit("DONE")
