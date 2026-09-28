# Shared readers for the gllvmTMB P1 temporal receipts in this directory.
# Included by test/test_temporal_*.jl and julia_optima.jl; not a test file.

using TOML, SHA

# Included by several test files in one session: define once.
if !isdefined(@__MODULE__, :TEMPORAL_P1_DIR)

const TEMPORAL_P1_DIR = @__DIR__

# sha256 of each committed receipt file. A regenerated receipt changes these;
# update them only together with the regenerated file and its generator run.
const TEMPORAL_P1_SHA256 = Dict(
    "oracle.toml" => "ab41f55069d46b4cac23519cdba039872809f04020b1dad2ec129a057bc9aea0",
    "fits.toml" => "4c2ea4d92ad5abd137fdaf23fbc372f46694f742a7116b012d08d78e5242678c",
    "forecast.toml" => "41fc640a5892e7952cb4df331f84a799b91facf0cda4ae92a73dce55fe0b72fd",
    "profile.toml" => "6e442a7edb7867aa871a0b664deade557666e95e44cf33bed19cde950ceb0a64",
    "compare.toml" => "0209f6f2836465fbf6069540223406a7505651ec1c68113afbb9f583951975e8",
    "bootstrap.toml" => "a83f83bec3efa0791f83151193eb6e2797bbe4b4c6527a2be6dd9d58d919fa0e",
    "cross_objective.toml" => "680bfefdd6c0d5de7e142eb537363f3a2953c80d1f54c9af7babb1132e5cfb72",
    # Slice 2 (generate_temporal_p1_slice2.R).
    "composed.toml" => "bf3cd27d73de9593857be6c6d996266a04c6088875b9a50220a0b2c5051124b5",
    "composed_cross.toml" => "e5c6e803bfb8d66524bc86465ae9163596fb940033d48d9627b7b166355feb87",
)

temporal_p1_path(name) = joinpath(TEMPORAL_P1_DIR, name)
temporal_p1_sha(name) = bytes2hex(sha256(read(temporal_p1_path(name))))
temporal_p1_load(name) = TOML.parsefile(temporal_p1_path(name))

# NA in a receipt is the string "NA"; numbers may be parsed as Int or Float64.
temporal_p1_num(x) = x == "NA" ? missing : Float64(x)
temporal_p1_vec(v) = Float64[Float64(x) for x in v]

function temporal_p1_table(ds::AbstractDict)
    pairs = Pair{Symbol,Any}[:series => String.(ds["series"]), :trait => String.(ds["trait"])]
    for k in ("occasion", "elapsed")
        haskey(ds, k) && push!(pairs, Symbol(k) => temporal_p1_vec(ds[k]))
    end
    haskey(ds, "measurement") && push!(pairs, :measurement => String.(ds["measurement"]))
    push!(pairs, :value => temporal_p1_vec(ds["value"]))
    return (; pairs...)
end

temporal_p1_value_sha(tbl) = bytes2hex(sha256(reinterpret(UInt8, htol.(tbl.value))))

function temporal_p1_term(c::AbstractDict)
    rep = get(c, "replicate", "")
    rep = isempty(rep) ? nothing : Symbol(rep)
    t = Symbol(c["time"]); s = Symbol(c["structure"])
    mode = c["mode"]
    mode == "indep" && return temporal_indep(:(0 + trait | series), t; structure=s, replicate=rep)
    mode == "dep" && return temporal_dep(:(0 + trait | series), t; structure=s, replicate=rep)
    return temporal_latent(:(0 + trait | series), t; structure=s, replicate=rep, unique=c["unique"])
end

# Fit settings of the between-optima receipts (julia_optima.jl and
# test_temporal_fit_receipts.jl use the same ones).
const TEMPORAL_P1_GTOL = 1e-8
const TEMPORAL_P1_ITERATIONS = 2000

function temporal_p1_fit(F::AbstractDict, c::AbstractDict; kwargs...)
    tbl = temporal_p1_table(F["datasets"][c["dataset"]])
    return fit_temporal_gllvm(tbl; formula=@formula(value ~ 0 + trait),
        temporal=temporal_p1_term(c), kwargs...)
end

# ---- slice 2: temporal + ordinary unit / unit_obs terms (composed.toml) ----
const TEMPORAL_P1_STRING_COLS = ("series", "trait", "measurement", "unit_obs", "within_unit")

function temporal_p1_composed_table(ds::AbstractDict)
    pairs = Pair{Symbol,Any}[]
    for k in sort(collect(keys(ds)))
        k in ("value", "value_sha256") && continue
        push!(pairs, Symbol(k) => (k in TEMPORAL_P1_STRING_COLS ? String.(ds[k]) : temporal_p1_vec(ds[k])))
    end
    push!(pairs, :value => temporal_p1_vec(ds["value"]))
    return (; pairs...)
end

# The receipt stores the temporal marker and the ordinary terms in Julia syntax.
temporal_p1_eval(s::AbstractString) = Core.eval(GLLVModels, Meta.parse(s))
temporal_p1_terms(c::AbstractDict) = Any[Meta.parse(t) for t in c["julia_terms"]]

function temporal_p1_composed_fit(C::AbstractDict, c::AbstractDict; kwargs...)
    tbl = temporal_p1_composed_table(C["datasets"][c["dataset"]])
    uo = get(c, "unit_obs", "")
    return fit_temporal_gllvm(tbl; formula=@formula(value ~ 0 + trait),
        temporal=temporal_p1_eval(c["julia_temporal"]), structure=temporal_p1_terms(c),
        unit=:series, unit_obs=isempty(uo) ? nothing : Symbol(uo), kwargs...)
end

end # if !isdefined
