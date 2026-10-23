# Julia side of the phylo_latent twin receipts at gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9).
#
#   julia --project=. tools/phylo_latent/compare_phylo_latent_p1.jl fit CASE DATA_JSON OUT_JSON
#       Fit `fit_phylo_latent_gllvm` on the literal fixture and write the Julia
#       receipt (theta in both coordinate orders, logLik, wall time).
#   julia --project=. tools/phylo_latent/compare_phylo_latent_p1.jl compare CASE DATA_JSON JULIA_JSON R_JSON
#       Evaluate the Julia objective at R's optimum and print every paired
#       quantity against the signed tolerances.
#
# The replay test `test/test_phylo_latent_paired_p1.jl` includes this file for
# the shared helpers; nothing here runs at include time. Only root-project
# dependencies are used (the P1 CI job runs tagged tests with --project=.),
# so the receipts are read and written by the minimal JSON codec below.
using SHA, LinearAlgebra, GLLVModels

# --- minimal JSON codec (objects, arrays, strings, numbers, true/false/null) ---
struct PLJson
    d::Dict{String,Any}
end
_plwrap(x) = x isa Dict{String,Any} ? PLJson(x) : x isa Vector{Any} ? map(_plwrap, x) : x
Base.getproperty(o::PLJson, s::Symbol) = _plwrap(getfield(o, :d)[String(s)])
Base.haskey(o::PLJson, s::Symbol) = haskey(getfield(o, :d), String(s))

mutable struct _PLCursor
    s::String
    i::Int
end
function _plskip!(c)
    while c.i <= ncodeunits(c.s) && c.s[c.i] in (' ', '\n', '\r', '\t')
        c.i += 1
    end
end
function _plvalue!(c)
    _plskip!(c)
    ch = c.s[c.i]
    if ch == '{'
        c.i += 1; out = Dict{String,Any}(); _plskip!(c)
        c.s[c.i] == '}' && (c.i += 1; return out)
        while true
            _plskip!(c); key = _plvalue!(c); _plskip!(c)
            c.s[c.i] == ':' || error("JSON: expected ':' at $(c.i)"); c.i += 1
            out[key] = _plvalue!(c); _plskip!(c)
            c.s[c.i] == ',' ? (c.i += 1) : c.s[c.i] == '}' ? (c.i += 1; return out) :
                error("JSON: expected ',' or '}' at $(c.i)")
        end
    elseif ch == '['
        c.i += 1; out = Any[]; _plskip!(c)
        c.s[c.i] == ']' && (c.i += 1; return out)
        while true
            push!(out, _plvalue!(c)); _plskip!(c)
            c.s[c.i] == ',' ? (c.i += 1) : c.s[c.i] == ']' ? (c.i += 1; return out) :
                error("JSON: expected ',' or ']' at $(c.i)")
        end
    elseif ch == '"'
        c.i += 1; io = IOBuffer()
        while true
            b = c.s[c.i]
            if b == '"'
                c.i += 1; return String(take!(io))
            elseif b == '\\'
                e = c.s[c.i + 1]
                if e == 'u'
                    write(io, Char(parse(UInt16, c.s[(c.i + 2):(c.i + 5)]; base = 16))); c.i += 6
                else
                    write(io, Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b',
                        'f' => '\f', '"' => '"', '\\' => '\\', '/' => '/')[e]); c.i += 2
                end
            else
                write(io, b); c.i = nextind(c.s, c.i)
            end
        end
    elseif startswith(SubString(c.s, c.i), "true")
        c.i += 4; return true
    elseif startswith(SubString(c.s, c.i), "false")
        c.i += 5; return false
    elseif startswith(SubString(c.s, c.i), "null")
        c.i += 4; return nothing
    else
        j = c.i
        while j <= ncodeunits(c.s) && c.s[j] in "+-0123456789.eE"
            j += 1
        end
        tok = c.s[c.i:(j - 1)]; c.i = j
        return occursin(r"[.eE]", tok) ? parse(Float64, tok) : parse(Int, tok)
    end
end
"""Parse a JSON file into nested `PLJson` / `Vector` / scalar values."""
function pl_read_json(path)
    c = _PLCursor(read(path, String), 1)
    v = _plvalue!(c); _plskip!(c)
    c.i > ncodeunits(c.s) || error("JSON: trailing characters in $path")
    return _plwrap(v)
end
_pljson(io, x::AbstractString) = print(io, '"', escape_string(x), '"')
_pljson(io, x::Bool) = print(io, x ? "true" : "false")
_pljson(io, ::Nothing) = print(io, "null")
_pljson(io, x::Integer) = print(io, x)
_pljson(io, x::AbstractFloat) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
_pljson(io, x::Symbol) = _pljson(io, String(x))
function _pljson(io, x::AbstractVector)
    print(io, '['); for (k, v) in enumerate(x); k > 1 && print(io, ", "); _pljson(io, v); end; print(io, ']')
end
function _pljson(io, x::AbstractDict)
    print(io, "{\n")
    ks = sort!(collect(keys(x)))
    for (k, key) in enumerate(ks)
        print(io, "  "); _pljson(io, String(key)); print(io, ": "); _pljson(io, x[key])
        print(io, k < length(ks) ? ",\n" : "\n")
    end
    print(io, '}')
end

const PL_P1_SOURCE_PIN = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const PL_P1_CASES = ("struct_phy_tree_rr", "struct_phy_dense_rr", "cov_phylo_latent_rsz")
const PL_P1_TOL = (loglik_rtol = 1e-6, cross_abs = 1e-8, estimate_rtol = 1e-4,
                   logdet_abs = 1e-8, cross_gradient = 1e-4)

_pl_matrix(rows) = permutedims(reduce(hcat, [Float64.(collect(r)) for r in rows]))

"""Little-endian Float64 column-major SHA-256 of a response matrix (R's `yhash`)."""
function pl_y_hash(Y::AbstractMatrix{Float64})
    io = IOBuffer()
    for x in vec(Y)
        write(io, htol(x))
    end
    return bytes2hex(sha256(take!(io)))
end

"""Read a fixture JSON: response, labels, tree, dense correlation matrix."""
function pl_read_fixture(path)
    fx = pl_read_json(path)
    Y = _pl_matrix(fx.Y_traits_by_observations)
    pl_y_hash(Y) == fx.data_sha256 || error("fixture response hash mismatch")
    return (Y = Y, species = String.(collect(fx.observation_species)),
            tips = String.(collect(fx.tip_labels)), newick = String(fx.newick),
            vcv = _pl_matrix(fx.vcv_corr), rank = Int(fx.rank),
            data_sha256 = String(fx.data_sha256), fixture = String(fx.fixture))
end

pl_route(case) = case == "struct_phy_dense_rr" ? :vcv : :tree

"""The twin's precision and observation map for one case (same code path as the fit)."""
function pl_precision(fx, case)
    if pl_route(case) === :tree
        phy, labels = GLLVModels._phylo_latent_tree_precision(fx.newick)
    else
        levels = sort(unique(fx.species))
        phy = GLLVModels._phylo_latent_dense_precision(fx.vcv, fx.tips, levels)
        labels = levels
    end
    position = Dict(l => i for (i, l) in enumerate(labels))
    return phy, [position[s] for s in fx.species]
end

"""R coordinate order `[b_fix; log_sigma_eps; theta_rr_phy]` to Julia `[beta; rr; log_sd]`."""
pl_r_to_julia(theta, n_traits) = vcat(theta[1:n_traits], theta[(n_traits + 2):end], theta[n_traits + 1])
pl_julia_to_r(theta, n_traits) = vcat(theta[1:n_traits], theta[end], theta[(n_traits + 1):(end - 1)])

"""Julia marginal negative log-likelihood at a Julia-order `theta`."""
function pl_julia_objective(fx, case, theta_julia)
    phy, species_id = pl_precision(fx, case)
    phy = GLLVModels._validate_precision_fit_input(phy)
    return GLLVModels._precision_multivariate_nll(fx.Y, phy, theta_julia; rank = fx.rank,
        mode = :barelowrank, residual_mode = :shared, species_id = species_id)
end

function pl_fit(fx, case; kwargs...)
    source = pl_route(case) === :tree ? (tree = fx.newick,) : (vcv = fx.vcv, tip_labels = fx.tips)
    return fit_phylo_latent_gllvm(fx.Y, fx.species; d = fx.rank, source..., kwargs...)
end

_pl_rows(M) = [collect(M[i, :]) for i in 1:size(M, 1)]

function pl_fit_receipt(case, data_path, out_path)
    isfile(out_path) && error("use a fresh output path; receipts are immutable")
    case in PL_P1_CASES || error("unknown case $case")
    fx = pl_read_fixture(data_path)
    started = time_ns()
    fit = pl_fit(fx, case)
    elapsed = (time_ns() - started) / 1e9
    n_traits = size(fx.Y, 1)
    phy = fit.phy
    receipt = Dict{String,Any}(
        "schema_version" => "phylo-latent-p1-julia-receipt-1", "case" => case,
        "status" => "recorded", "source_pin" => PL_P1_SOURCE_PIN,
        "julia_version" => string(VERSION),
        "git_head" => strip(read(`git -C $(@__DIR__) rev-parse HEAD`, String)),
        "data_file_sha256" => bytes2hex(sha256(read(data_path))),
        "data_sha256" => fx.data_sha256, "route" => String(pl_route(case)),
        "entry" => "fit_phylo_latent_gllvm", "residual_mode" => "shared",
        "theta_julia_order" => fit.parameters,
        "theta_r_order" => pl_julia_to_r(fit.parameters, n_traits),
        "parameter_labels_julia" => fit.parameter_labels,
        "loglik" => fit.loglik, "objective" => -fit.loglik,
        "converged" => fit.converged, "gradient_norm" => fit.gradient_norm,
        "iterations" => fit.iterations, "stopping_reason" => String(fit.stopping_reason),
        "hessian_positive_definite" => fit.hessian_positive_definite,
        "hessian_condition_number" => fit.hessian_condition_number,
        "Sigma_phy" => _pl_rows(fit.loading * fit.loading'),
        "beta" => fit.beta, "sigma_eps2" => fit.residual_variance[1],
        "log_det_Q" => phy.log_det, "n_aug" => phy.n_aug,
        "species_aug_id" => phy.species_aug_id,
        "fit_elapsed_seconds" => elapsed, "qualified" => false)
    open(out_path, "w") do io
        _pljson(io, receipt); println(io)
    end
    println("Julia $case loglik=$(fit.loglik) converged=$(fit.converged) elapsed=$(round(elapsed; digits = 2))s")
    return receipt
end

"""Every paired quantity for one case, from the fixture and both receipts."""
function pl_compare(case, data_path, julia_path, r_path)
    fx = pl_read_fixture(data_path)
    jr = pl_read_json(julia_path)
    rr = pl_read_json(r_path)
    n_traits = size(fx.Y, 1)
    r_theta = Float64.(collect(rr.theta_hat))
    j_theta = Float64.(collect(jr.theta_julia_order))
    julia_at_r = pl_julia_objective(fx, case, pl_r_to_julia(r_theta, n_traits))
    julia_at_j = pl_julia_objective(fx, case, j_theta)
    Sigma_r = _pl_matrix(rr.Sigma_phy)
    Sigma_j = _pl_matrix(jr.Sigma_phy)
    beta_r = r_theta[1:n_traits]
    s2_r = exp(2 * r_theta[n_traits + 1])
    phy, _ = pl_precision(fx, case)
    return (
        loglik_rel = abs(jr.loglik - rr.loglik) / abs(rr.loglik),
        julia_own_reeval_abs = abs(julia_at_j - jr.objective),
        cross_julia_at_r_abs = abs(julia_at_r - rr.objective),
        cross_r_at_julia_abs = abs(rr.cross.r_objective_at_julia_theta - julia_at_j),
        r_gradient_at_julia = rr.cross.r_gradient_max_abs_at_julia_theta,
        Sigma_rel = maximum(abs.(Sigma_j .- Sigma_r)) / maximum(abs.(Sigma_r)),
        beta_rel = maximum(abs.(Float64.(collect(jr.beta)) .- beta_r) ./ max.(abs.(beta_r), 1e-12)),
        sigma_eps2_rel = abs(jr.sigma_eps2 - s2_r) / s2_r,
        logdet_abs = abs(-phy.log_det - rr.log_det_A_phy_rr),
        n_aug_match = phy.n_aug == rr.n_aug_phy,
        julia_at_r = julia_at_r, r_objective = rr.objective,
        julia_objective = julia_at_j, r_at_julia = rr.cross.r_objective_at_julia_theta)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    mode = isempty(ARGS) ? "" : ARGS[1]
    if mode == "fit" && length(ARGS) == 4
        pl_fit_receipt(ARGS[2], ARGS[3], ARGS[4])
    elseif mode == "compare" && length(ARGS) == 5
        c = pl_compare(ARGS[2:5]...)
        for (k, v) in pairs(c)
            println(rpad(string(k), 26), v)
        end
    else
        error("usage: compare_phylo_latent_p1.jl fit CASE DATA OUT | compare CASE DATA JULIA R")
    end
end
