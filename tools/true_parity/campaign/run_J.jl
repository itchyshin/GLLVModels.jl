#!/usr/bin/env julia
# True-parity campaign (C3, C4), Julia side. Signed itchyshin/GLLVModels.jl#684 item 4.
# Fits one cell with GLLVModels on the SAME literal CSV that gen_data.R wrote and run_R.R read, and
# writes the RAW outputs to <out>/<cell>_J.toml. It never reads R's results (Julia starts from its own
# default start, never from R's coordinates), classifies nothing and signs nothing;
# tools/true_parity/campaign/write_receipts.py reads this file and run_R.R's JSON.
#
# Usage: julia --project=<GLLVModels.jl checkout> run_J.jl <cell>
#   env: CAMPAIGN_DATA, CAMPAIGN_OUT, CAMPAIGN_SMALL=1 (development runs, never a receipt),
#        GLLVM_JL_SHA (the commit the checkout is at, recorded), CAMPAIGN_COND=0 to skip cond(H).
# Julia wall times include first-call compilation (nothing is warmed up), as in the pre-run.
using GLLVModels, LinearAlgebra, Printf, SHA, TOML, Distributions
const GM = GLLVModels
cell = ARGS[1]
data_dir = get(ENV, "CAMPAIGN_DATA", "data"); out_dir = get(ENV, "CAMPAIGN_OUT", "out")
SMALL = !isempty(get(ENV, "CAMPAIGN_SMALL", "")); sfx = SMALL ? "_small" : ""
DO_COND = get(ENV, "CAMPAIGN_COND", "1") != "0"
mkpath(out_dir)

# ---- readers (R write.csv output; the root project has no CSV package) ----------------------
function splitcsv(line)
    out = String[]; buf = IOBuffer(); inq = false
    for c in line
        if c == '"'; inq = !inq
        elseif c == ',' && !inq; push!(out, String(take!(buf)))
        else; write(buf, c); end
    end
    push!(out, String(take!(buf))); out
end
function readcsv(path)
    ls = filter(!isempty, readlines(path)); hdr = splitcsv(ls[1])
    cols = Dict(h => String[] for h in hdr)
    for l in ls[2:end]; for (h, v) in zip(hdr, splitcsv(l)); push!(cols[h], v); end; end
    hdr, cols
end
csvpath(name) = joinpath(data_dir, name * sfx * ".csv")
fsha(path) = bytes2hex(open(sha256, path))
fl(x) = parse.(Float64, x)
rows(M) = [collect(Float64, M[i, :]) for i in 1:size(M, 1)]   # TOML-friendly matrix (array of rows)
function wide_from_long(cols; T = Float64)  # long (site, trait, value) -> p x n, traits sorted
    tr = sort(unique(cols["trait"])); ti = Dict(t => i for (i, t) in enumerate(tr))
    sites = unique(cols["site"]); si = Dict(s => i for (i, s) in enumerate(sites))
    Y = zeros(T, length(tr), length(sites))
    for (s, t, v) in zip(cols["site"], cols["trait"], cols["value"]); Y[ti[t], si[s]] = T == Int ? parse(Int, v) : parse(Float64, v); end
    Y, tr
end

R = Dict{String,Any}("engine" => "GLLVModels.jl", "cell" => cell, "small" => SMALL, "julia_version" => string(VERSION),
    "gllvmodels_commit" => get(ENV, "GLLVM_JL_SHA", "unknown"), "host" => gethostname(),
    "JULIA_NUM_THREADS" => Threads.nthreads(), "OPENBLAS_NUM_THREADS" => get(ENV, "OPENBLAS_NUM_THREADS", "unset"))
function flush_out(final = false)
    f = joinpath(out_dir, cell * sfx * "_J.toml")
    open(f * ".tmp", "w") do io; TOML.print(io, R); end
    mv(f * ".tmp", f; force = true)
end
put!(k, v) = (R[k] = v; flush_out())
timeit(f) = (t = time(); r = f(); (r, time() - t))

# SEs and fixed-effect names/estimates from the public Wald confint (beta[..], gamma[..] terms)
function fixed_block!(ci)
    idx = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ci.term)
    put!("beta_terms", ci.term[idx]); put!("beta", ci.estimate[idx]); put!("beta_se", ci.se[idx])
    put!("pd_hessian", ci.pd_hessian)
end
function cond_block!(fit, Y; kw...)
    DO_COND || return
    try
        V, t = timeit(() -> Matrix(vcov(fit, Y; kw...)))
        put!("wall_vcov_sec", t)
        put!("cond_H", all(isfinite, V) ? cond(Symmetric((V .+ V') ./ 2)) : NaN)   # cond(H) = cond(inverse Wald covariance)
    catch e
        put!("cond_H_error", first(sprint(showerror, e), 300))
    end
end
function ci_block!(fit, Y; kw...)
    try
        ci, t = timeit(() -> confint(fit, Y; kw...))
        put!("wall_confint_sec", t); fixed_block!(ci)
        cond_block!(fit, Y; kw...)
    catch e
        put!("confint_error", first(sprint(showerror, e), 300))
    end
end

put!("data_file", basename(csvpath(cell in ("spider", "beetle", "fungi", "urban") ? cell * "_wide" : cell)))
if cell in ("gaussian", "poisson", "nb2", "binomial", "ordinal")
    path = csvpath(cell); put!("data_sha256", fsha(path)); hdr, cols = readcsv(path)
    T = cell == "gaussian" ? Float64 : Int
    Y, tr = wide_from_long(cols; T = T); p, n = size(Y); put!("trait_levels", tr); put!("p", p); put!("n", n)
    if cell == "gaussian"
        # per-trait intercepts as an explicit design (R's `0 + trait`); as the P0 harness and test_fixed_effects.jl
        X = zeros(p, n, p); for t in 1:p; X[t, :, t] .= 1.0; end
        put!("call", "fit_gaussian_gllvm(Y; K = 2, X = <per-trait intercept design>)")
        fit, w = timeit(() -> fit_gaussian_gllvm(Y; K = 2, X = X)); put!("wall_fit_sec", w)
        put!("converged", fit.converged); put!("logLik", fit.logLik); put!("iterations", fit.n_iter)
        put!("LLt", rows(fit.pars.Λ * fit.pars.Λ')); put!("sigma_eps", fit.pars.σ_eps)
        ci_block!(fit, Y; X = X)
    elseif cell == "poisson"
        put!("call", "fit_poisson_gllvm(Y; K = 2, hessian = :observed)")
        fit, w = timeit(() -> fit_poisson_gllvm(Y; K = 2, hessian = :observed)); put!("wall_fit_sec", w)
        put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
        put!("LLt", rows(fit.Λ * fit.Λ')); ci_block!(fit, Y)
    elseif cell == "binomial"
        put!("call", "fit_binomial_gllvm(Y; K = 2)")
        fit, w = timeit(() -> fit_binomial_gllvm(Y; K = 2)); put!("wall_fit_sec", w)
        put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
        put!("LLt", rows(fit.Λ * fit.Λ')); ci_block!(fit, Y)
    elseif cell == "nb2"
        put!("call", "fit_gllvm(Y; family = NegativeBinomial(), K = 2, disp_group = :species, g_tol = 1e-7, iterations = 800)")
        fit, w = timeit(() -> fit_gllvm(Y; family = GM.NegativeBinomial(), K = 2, disp_group = :species, g_tol = 1e-7, iterations = 800)); put!("wall_fit_sec", w)
        put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
        put!("LLt", rows(fit.Λ * fit.Λ')); put!("dispersion_phi", collect(Float64, fit.r_group))
        put!("dispersion_boundary", collect(Bool, fit.dispersion_boundary)); ci_block!(fit, Y)
    else  # ordinal
        put!("call", "fit_gllvm(Y; family = Ordinal(), K = 2)")
        fit, w = timeit(() -> fit_gllvm(Y; family = Ordinal(), K = 2)); put!("wall_fit_sec", w)
        put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
        put!("LLt", rows(fit.Λ * fit.Λ')); put!("tau_size", collect(size(fit.τ))); put!("C", fit.C)
        put!("tau", fit.τ isa AbstractMatrix ? rows(fit.τ) : collect(Float64, fit.τ)); ci_block!(fit, Y)
    end
elseif cell == "temporal"
    path = csvpath("temporal"); put!("data_sha256", fsha(path)); hdr, cols = readcsv(path)
    tbl = (series = String.(cols["series"]), trait = String.(cols["trait"]), occasion = fl(cols["occasion"]), value = fl(cols["value"]))
    put!("call", "fit_temporal_gllvm(tbl; formula = @formula(value ~ 0 + trait), temporal = temporal_latent(:(0 + trait | series), :occasion; d = 1, structure = :ar1, unique = false), g_tol = 1e-8, iterations = 2000)")
    fit, w = timeit(() -> fit_temporal_gllvm(tbl; formula = @formula(value ~ 0 + trait),
        temporal = temporal_latent(:(0 + trait | series), :occasion; d = 1, structure = :ar1, unique = false),
        g_tol = 1e-8, iterations = 2000))
    put!("wall_fit_sec", w); put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
    put!("stopping_reason", string(fit.stopping_reason)); put!("gradient_norm", fit.gradient_norm)
    put!("hessian_positive_definite", fit.hessian_positive_definite); put!("hessian_min_eigenvalue", fit.hessian_min_eigenvalue)
    et = extract_temporal(fit)
    put!("temporal_phi", et.time.value); put!("temporal_phi_parameter", et.time.parameter)
    put!("temporal_loadings", collect(Float64, et.loadings[:, 1])); put!("temporal_loadings_traits", collect(String, fit.spec.traits))
    put!("LLt", rows(et.loadings * et.loadings')); put!("beta", collect(Float64, fit.beta)); put!("beta_terms", collect(String, fit.coefficient_names))
elseif cell == "isdm"
    include(joinpath(dirname(pathof(GLLVModels)), "..", "test", "fixtures", "isdm", "isdm_fixture_io.jl"))
    path = abspath(csvpath("isdm")); put!("data_sha256", fsha(path)); d = read_isdm_csv(path)
    cl = (Binomial(), CLogLogLink())
    fm = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) + latent(0 + trait | cell_id, d = 2, unique = false))
    put!("call", "fit_isdm_gllvm(fm, d; family = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink())), trait = :trait, unit = :cell_id)")
    ft, w = timeit(() -> fit_isdm_gllvm(fm, d; family = isdm_sources(gbif = Poisson(), survey = cl), trait = :trait, unit = :cell_id))
    put!("wall_fit_sec", w); put!("converged", ft.converged); put!("cells_converged", all(ft.cell_converged))
    put!("logLik", ft.loglik); put!("iterations", ft.iterations)
    put!("beta", ft.b_fix); put!("beta_terms", collect(String, ft.b_names)); put!("LLt", rows(ft.Λ * ft.Λ'))
    pr = predict(ft)
    put!("predict_link", collect(Float64, pr.est)); put!("predict_cell_id", String.(pr.cell_id))
    put!("predict_trait", String.(pr.trait)); put!("predict_source", String.(pr.isdm_source))
    put!("predict_response", collect(Float64, predict(ft; type = :response).est))
    nd0 = merge(d, (log_support = zeros(length(d.value)),))
    put!("predict_newdata_offset0_link", collect(Float64, predict(ft; newdata = nd0).est))
    put!("predict_newdata_offset0_response", collect(Float64, predict(ft; newdata = nd0, type = :response).est))
elseif cell == "crabs"
    path = csvpath("crabs"); put!("data_sha256", fsha(path)); hdr, cols = readcsv(path)
    trs = ["FL", "RW", "CL", "CW", "BD"]; p = 5; n = length(cols["specimen"])
    Y = Float64[parse(Float64, cols[trs[t]][s]) for t in 1:p, s in 1:n]
    grps = sort(unique(cols["grp"])); gi = [findfirst(==(g), grps) for g in cols["grp"]]; q = p * length(grps)
    X = zeros(p, n, q); names_ = String[]
    for (g, gname) in enumerate(grps), t in 1:p; push!(names_, string(trs[t], ":", gname)); end
    for s in 1:n, t in 1:p; X[t, s, (gi[s] - 1) * p + t] = 1.0; end   # R's `0 + trait:grp` column order (trait fastest)
    put!("call", "fit_gaussian_gllvm(Y; K = 1, X = <per-trait x group design>)"); put!("p", p); put!("n", n); put!("beta_design_names", names_)
    fit, w = timeit(() -> fit_gaussian_gllvm(Y; K = 1, X = X)); put!("wall_fit_sec", w)
    put!("converged", fit.converged); put!("logLik", fit.logLik); put!("iterations", fit.n_iter)
    put!("LLt", rows(fit.pars.Λ * fit.pars.Λ')); put!("sigma_eps", fit.pars.σ_eps)
    ci_block!(fit, Y; X = X)
    put!("eta", vec(predict(fit, Y; type = :link, X = X)))
elseif cell == "urban"
    path = csvpath("urban_wide"); put!("data_sha256", fsha(path)); hdr, cols = readcsv(path)
    sp = filter(h -> h != "review", hdr); n = length(cols["review"]); p = length(sp)
    Y = Int[parse(Int, cols[sp[t]][s]) for t in 1:p, s in 1:n]; put!("p", p); put!("n", n); put!("trait_levels", sp)
    put!("call", "fit_binomial_gllvm(Y; K = 2, link = ProbitLink())")
    fit, w = timeit(() -> fit_binomial_gllvm(Y; K = 2, link = GM.ProbitLink())); put!("wall_fit_sec", w)
    put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
    put!("LLt", rows(fit.Λ * fit.Λ'))
    put!("beta", collect(Float64, fit.β)); put!("beta_terms", ["beta[$i]" for i in 1:p])
    put!("ci_skipped", "C4 quantities need no standard errors; the p = 51 finite-difference Hessian is the expensive step")
    put!("eta", vec(predict(fit, Y; type = :link)))
elseif cell in ("spider", "beetle", "fungi")
    covs = cell == "spider" ? ["ConWate", "BareSand", "CovMoss"] : cell == "beetle" ? ["pH", "Moist", "Org"] : ["TEMPR", "PRECIP", "logAREA"]
    path = csvpath(cell * "_wide"); put!("data_sha256", fsha(path)); hdr, cols = readcsv(path)
    sp = filter(h -> !(h in covs) && h != "site", hdr); n = length(cols["site"]); p = length(sp)
    Y = Int[parse(Int, cols[sp[t]][s]) for t in 1:p, s in 1:n]; put!("p", p); put!("n", n); put!("trait_levels", sp)
    covdata = NamedTuple{Tuple(Symbol.(covs))}(Tuple([fl(cols[c]) for c in covs]))
    q = length(covs); X = zeros(p, n, q); for k in 1:q, s in 1:n, t in 1:p; X[t, s, k] = covdata[k][s]; end
    fam = cell == "fungi" ? GM.Binomial() : GM.NegativeBinomial()
    f = cell == "spider" ? @formula(y ~ 1 + ConWate + BareSand + CovMoss) : cell == "beetle" ? @formula(y ~ 1 + pH + Moist + Org) :
        @formula(y ~ 1 + TEMPR + PRECIP + logAREA)
    put!("call", "gllvm(@formula(y ~ 1 + <3 covariates>), Y, covdata; family = $(nameof(typeof(fam)))(), K = 2)")
    fit, w = timeit(() -> gllvm(f, Y, covdata; family = fam, K = 2)); put!("wall_fit_sec", w)
    put!("fit_type", string(typeof(fit))); put!("converged", fit.converged); put!("logLik", fit.loglik); put!("iterations", fit.iterations)
    put!("LLt", rows(fit.Λ * fit.Λ'))
    if hasproperty(fit, :r_group)
        put!("dispersion_phi", collect(Float64, fit.r_group)); put!("dispersion_boundary", collect(Bool, fit.dispersion_boundary))
    end
    if cell == "spider"
        ci_block!(fit, Y; X = X)      # 28 x 12: cheap, gives pd_hessian and cond(H)
    else
        put!("beta", vcat(collect(Float64, fit.β), collect(Float64, fit.γ))); put!("beta_terms", vcat(["beta[$i]" for i in 1:p], ["gamma[$k]" for k in 1:q]))
        put!("ci_skipped", "C4 quantities need no standard errors; the finite-difference Hessian at this size is the expensive step")
    end
    if fit isa GM.NBGroupedCovFit
        Z = getLV(fit, Y, X; rotate = false)
        O = zeros(p, n); for k in 1:q, s in 1:n, t in 1:p; O[t, s] += X[t, s, k] * fit.γ[k]; end
        put!("eta", vec(fit.β .+ O .+ fit.Λ * Z'))
    else
        put!("eta", vec(predict(fit, Y, X; type = :link)))
    end
else
    error("unknown cell $cell")
end
put!("finished_unix", time()); put!("DONE", true)
println("DONE ", cell, " logLik=", get(R, "logLik", NaN), " converged=", get(R, "converged", missing), " wall_fit=", round(get(R, "wall_fit_sec", NaN); digits = 1))
