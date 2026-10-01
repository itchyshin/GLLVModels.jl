# Runtime evidence that no Julia surface exists yet for any of the 28
# planned data-area cases in docs/dev-log/core070/data-batch-contract.json.
# Loads the actual GLLVModels module and checks, for every planned_surface_group
# in that contract (candidate_exported_symbols / candidate_kwargs, duplicated
# here rather than JSON-parsed, to avoid adding a JSON dependency -- see
# tools/core070_masks_known.jl for the same no-external-dependency convention
# in this repo), that none of the candidate exported symbols is defined on
# the module and that none of the candidate keywords appears in the keyword
# list of gllvm()/fit_gllvm() (via Base.kwarg_decl). This is retained runtime
# evidence, not an assumption: no R call, no fit, no frozen source involved.
#
# Limit: it is a NAME check. Both entry points end in `kwargs...`, so
# Base.kwarg_decl lists only the dispatcher's own keywords, not what it forwards
# to the family fitters; it is not a keyword census. The fit-time offset= and
# mask=/missing-Y surfaces the non-Gaussian fitters do have are exercised
# behaviourally by tools/core070_data_surface_probe.jl.
#
# Usage: julia --project=<repo> tools/core070_data_batch.jl [output.json]

using GLLVModels

# GLLVM_PARITY_PIN=P1 names the P1-regenerated contract and the P1 reference
# commit (tools/core070_data_p1_contract.py; planned surfaces verbatim);
# unset/P0 is unchanged. Any other value stops here.
const PARITY_PIN = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P1")))
PARITY_PIN in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(PARITY_PIN))")
const CONTRACT_PATH = PARITY_PIN == "P1" ?
    joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "true-parity-latest", "data-batch-contract-p1.json") :
    joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "data-batch-contract.json")
const REFERENCE_COMMIT = PARITY_PIN == "P1" ?
    "9539352f66f2db2cc26b1c393e67212a359b60c9" : "b4d5fee64def88bc768dda1f1f77c29b295edd86"

# Must match docs/dev-log/core070/data-batch-contract.json's
# `julia_planned_surfaces`; the generator that wrote that contract derived
# these same lists (see the contract's julia_surface_status/julia_planned_surfaces).
const PLANNED_SURFACES = [
    ("miss_control", ["MissingDataControl", "miss_control", "MissControl"], ["miss"]),
    ("offset_preparation", ["prepare_offset", "gll_prepare_offset", "OffsetSpec"], ["offset"]),
    ("stored_or_predict_offset_accessor", ["offset", "training_offset", "offset_vec", "offset_newdata"], String[]),
    ("weight_normalisation", ["normalise_weights", "normalize_weights", "WeightSpec"], ["weights"]),
    ("weight_shape_adapter", ["normalise_weights", "normalize_weights", "wide_matrix_weights", "wide_df_weights"], ["weights"]),
]

# ---------------------------------------------------------------------------
# Minimal JSON writer (no external dependency; mirrors tools/core070_masks_known.jl).
# ---------------------------------------------------------------------------
json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"")
to_json(x::Bool) = x ? "true" : "false"
to_json(x::Nothing) = "null"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(v) for (k, v) in x), ",") * "}"
to_json(x::Vector{Pair{String,Any}}) = "{" * join(("\"$(json_escape(k))\":" * to_json(v) for (k, v) in x), ",") * "}"

function kwarg_names(f)
    names_seen = Set{Symbol}()
    for m in methods(f)
        try
            for k in Base.kwarg_decl(m)
                ks = string(k)
                endswith(ks, "...") && continue
                push!(names_seen, Symbol(ks))
            end
        catch
            continue
        end
    end
    names_seen
end

exported = Set(string.(names(GLLVModels)))
entry_point_kwargs = Dict{String,Set{Symbol}}(
    "gllvm" => isdefined(GLLVModels, :gllvm) ? kwarg_names(GLLVModels.gllvm) : Set{Symbol}(),
    "fit_gllvm" => isdefined(GLLVModels, :fit_gllvm) ? kwarg_names(GLLVModels.fit_gllvm) : Set{Symbol}(),
)

surfaces_out = Dict{String,Any}()
all_absent = true
for (group_name, candidate_symbols, candidate_kwargs) in PLANNED_SURFACES
    found_symbols = [s for s in candidate_symbols if isdefined(GLLVModels, Symbol(s))]
    found_kwargs = String[]
    for (entry, kws) in entry_point_kwargs
        for ck in candidate_kwargs
            Symbol(ck) in kws && push!(found_kwargs, "$(entry).$(ck)")
        end
    end
    absent = isempty(found_symbols) && isempty(found_kwargs)
    global all_absent &= absent
    surfaces_out[group_name] = Dict(
        "candidate_exported_symbols" => candidate_symbols,
        "candidate_kwargs" => candidate_kwargs,
        "found_symbols" => found_symbols,
        "found_kwargs" => found_kwargs,
        "surface_absent" => absent,
    )
end

receipt = Dict(
    "schema" => "core070-data-batch-julia-introspection/v1",
    "scope" => "CORE070_DATA_BATCH_JULIA_SURFACE_ABSENCE",
    "reference_commit" => REFERENCE_COMMIT,
    "contract_path" => relpath(CONTRACT_PATH, joinpath(@__DIR__, "..")),  # repo-relative, no host path
    "julia_version" => string(VERSION),
    "gllvm_package_uuid" => "2dc8e01c-4f48-4476-aaae-e919b4a30df7",
    "gllvm_exported_symbol_count" => length(exported),
    "entry_point_kwargs" => Dict(k => sort(string.(collect(v))) for (k, v) in entry_point_kwargs),
    "surfaces" => surfaces_out,
    "all_planned_surfaces_absent" => all_absent,
)

output_path = length(ARGS) >= 1 ? ARGS[1] : nothing
if output_path !== nothing
    mkpath(dirname(abspath(output_path)))
    open(output_path, "w") do io
        print(io, to_json(receipt))
    end
end

println("CORE070_DATA_BATCH_JULIA_INTROSPECTION_", all_absent ? "ALL_SURFACES_ABSENT" : "SURFACE_FOUND_REVIEW_CONTRACT")
exit(all_absent ? 0 : 1)
