# Behavioural probe of the GLLVModels fit-time surfaces that the core070 data
# rows name: weights, offset, mask, and `missing` cells in Y.
#
# Why this exists: tools/core070_data_batch.jl reads Base.kwarg_decl() on
# gllvm()/fit_gllvm(). Both entry points end in `kwargs...`, so kwarg_decl is a
# name census of the dispatcher's own keywords, not of what it forwards. This
# probe calls each surface on a tiny fixture through the public dispatcher and
# records what actually happens: refused (the error type and first line), or
# accepted, and whether the maximised logLik moved against the same fit without
# the surface. `mask` and `missing` are also compared with each other.
#
# Fixture: p = 4 traits, n = 24 sites, K = 1, fixed seeds; the offset is
# non-constant (N(0, 0.3^2) per cell) so an intercept cannot absorb it; the
# mask drops the same two cells that are set to `missing`; weights are
# U(0.5, 1.5) per cell. No R call, no oracle, no frozen source.
#
# Usage: julia --project=<repo> tools/core070_data_surface_probe.jl [output.json]

using GLLVModels
using Random
using StatsModels: @formula, loglikelihood
using Distributions: Normal, Poisson, NegativeBinomial, Binomial, Beta, Gamma

const PARITY_PIN = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P0")))
PARITY_PIN in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(PARITY_PIN))")
const REFERENCE_COMMIT = PARITY_PIN == "P1" ?
    "9539352f66f2db2cc26b1c393e67212a359b60c9" : "b4d5fee64def88bc768dda1f1f77c29b295edd86"

const P, N, K = 4, 24, 1
const LOGLIK_EFFECT_TOL = 1e-6
const DROPPED = [(1, 3), (3, 17)]

function fixture(kind::Symbol, seed::Integer)
    rng = MersenneTwister(seed)
    η = 0.4 .+ 0.8 .* randn(rng, P) * randn(rng, N)'
    if kind === :gaussian
        return η .+ 0.5 .* randn(rng, P, N)
    elseif kind === :count
        return [rand(rng) < 0.5 ? rand(rng, 0:6) : rand(rng, 0:2) for _ in 1:P, _ in 1:N]
    elseif kind === :binary
        return [Int(rand(rng) < 1 / (1 + exp(-x))) for x in η]
    elseif kind === :unit
        return [clamp(1 / (1 + exp(-(x + 0.3 * randn(rng)))), 0.02, 0.98) for x in η]
    elseif kind === :positive
        return [exp(0.3 * x) * (0.5 + rand(rng)) for x in η]
    end
    error("unknown fixture kind $kind")
end

with_missing(Y) = (Ym = Matrix{Union{Missing,eltype(Y)}}(Y); for (i, j) in DROPPED; Ym[i, j] = missing; end; Ym)
drop_mask() = (M = trues(P, N); for (i, j) in DROPPED; M[i, j] = false; end; M)

# Each path: (label, family expression as recorded, fixture kind, fitter closure).
fitter(fam; extra...) = (Y; kw...) -> fit_gllvm(Y; family = fam, K = K, extra..., kw...)
const PATHS = [
    ("fit_gllvm(Y; family=Normal())", :gaussian, fitter(Normal())),
    ("fit_gllvm(Y; family=Normal(), pervar=true)", :gaussian, fitter(Normal(); pervar = true)),
    ("fit_gllvm(Y; family=Poisson())", :count, fitter(Poisson())),
    ("fit_gllvm(Y; family=NegativeBinomial(1.0, 0.5))", :count, fitter(NegativeBinomial(1.0, 0.5))),
    ("fit_gllvm(Y; family=NB1())", :count, fitter(NB1())),
    ("fit_gllvm(Y; family=Binomial())", :binary, fitter(Binomial())),
    ("fit_gllvm(Y; family=Beta())", :unit, fitter(Beta())),
    ("fit_gllvm(Y; family=Gamma())", :positive, fitter(Gamma())),
    ("gllvm(@formula(y ~ 1), Y, data; family=Poisson())", :count,
        (Y; kw...) -> gllvm(@formula(y ~ 1), Y, (x = collect(1.0:N),); family = Poisson(), K = K, kw...)),
]

firstline(e) = first(split(sprint(showerror, e), '\n'))
shorten(s, n = 200) = length(s) <= n ? s : first(s, n) * "..."

function attempt(f, Y; kw...)
    try
        fit = f(Y; kw...)
        return (ok = true, loglik = Float64(loglikelihood(fit)), fit_type = string(nameof(typeof(fit))),
                error_type = nothing, error = nothing)
    catch e
        return (ok = false, loglik = nothing, fit_type = nothing,
                error_type = string(nameof(typeof(e))), error = shorten(firstline(e)))
    end
end

function classify(base, r)
    r.ok || return "refused"
    base.ok || return "accepted_baseline_failed"
    abs(r.loglik - base.loglik) > LOGLIK_EFFECT_TOL ? "accepted_changes_loglik" : "accepted_no_loglik_change"
end

entry(base, r) = Dict{String,Any}("status" => classify(base, r), "loglik" => r.loglik, "fit_type" => r.fit_type,
                                   "error_type" => r.error_type, "error" => r.error)

seed_rng = MersenneTwister(20260927)
offset_mat = 0.3 .* randn(seed_rng, P, N)
weights_mat = 0.5 .+ rand(seed_rng, P, N)

paths_out = Any[]
for (k, (label, kind, f)) in enumerate(PATHS)
    Y = fixture(kind, 100 + k)
    base = attempt(f, Y)
    res = Dict{String,Any}(
        "weights" => entry(base, attempt(f, Y; weights = weights_mat)),
        "offset" => entry(base, attempt(f, Y; offset = offset_mat)),
        "mask" => entry(base, attempt(f, Y; mask = drop_mask())),
        "missing_in_Y" => entry(base, attempt(f, with_missing(Y))),
    )
    m, x = res["mask"], res["missing_in_Y"]
    mask_eq_missing = m["loglik"] !== nothing && x["loglik"] !== nothing ?
        abs(m["loglik"] - x["loglik"]) <= 1e-8 * max(1.0, abs(m["loglik"])) : nothing
    push!(paths_out, Dict{String,Any}(
        "path" => label, "fixture" => string(kind),
        "baseline" => Dict{String,Any}("ok" => base.ok, "loglik" => base.loglik, "fit_type" => base.fit_type,
                                       "error_type" => base.error_type, "error" => base.error),
        "surfaces" => res, "mask_loglik_equals_missing_loglik" => mask_eq_missing))
    println(rpad(label, 56), " weights=", res["weights"]["status"], " offset=", res["offset"]["status"],
            " mask=", res["mask"]["status"], " missing=", res["missing_in_Y"]["status"])
end

# ---------------------------------------------------------------------------
# Minimal JSON writer (no external dependency; mirrors tools/core070_data_batch.jl).
# Floats are written with repr() so they round-trip exactly.
# ---------------------------------------------------------------------------
json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
to_json(x::Bool) = x ? "true" : "false"
to_json(x::Nothing) = "null"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(x[k]) for k in sort(collect(keys(x)))), ",") * "}"

receipt = Dict{String,Any}(
    "schema" => "core070-data-surface-probe/v1",
    "scope" => "CORE070_DATA_FIT_TIME_SURFACE_BEHAVIOUR",
    "reference_commit" => REFERENCE_COMMIT,
    "julia_version" => string(VERSION),
    "gllvm_package_uuid" => "2dc8e01c-4f48-4476-aaae-e919b4a30df7",
    "fixture" => Dict{String,Any}("p" => P, "n" => N, "K" => K, "dropped_cells" => [[i, j] for (i, j) in DROPPED],
                                  "offset" => "0.3 * randn(MersenneTwister(20260927), 4, 24), non-constant",
                                  "weights" => "0.5 .+ rand(same stream), 4 x 24"),
    "loglik_effect_tolerance" => LOGLIK_EFFECT_TOL,
    "status_meaning" => Dict{String,Any}(
        "refused" => "the call threw; error_type and the first line of the message are recorded",
        "accepted_changes_loglik" => "the call returned a fit whose logLik differs from the no-surface fit by more than loglik_effect_tolerance",
        "accepted_no_loglik_change" => "the call returned a fit with the no-surface logLik (the keyword may be ignored)",
        "accepted_baseline_failed" => "the call returned a fit but the no-surface baseline threw"),
    "paths" => paths_out,
)

output_path = length(ARGS) >= 1 ? ARGS[1] : nothing
if output_path !== nothing
    mkpath(dirname(abspath(output_path)))
    open(output_path, "w") do io
        print(io, to_json(receipt))
    end
end
println("CORE070_DATA_SURFACE_PROBE_DONE ", length(paths_out), " paths")
