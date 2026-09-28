# Readers for the iSDM P1 fixtures (test/fixtures/isdm/), shared by
# test/test_isdm.jl and test/parity/isdm_cases.jl. No CSV package: the files
# are R `write.csv` output (quoted strings, unquoted numbers, NA for missing).
using SHA, TOML

const ISDM_FIXTURE_DIR = @__DIR__

# SHA-256 of each fixture file, recorded when the fixtures were exported from R
# at P1 (export_p1_fixtures.R, stage `fixtures`); checked on every load.
const ISDM_FIXTURE_SHA256 = Dict(
    "isdm_predict.csv" => "cacb721676436f9751ab147ba26900d5fac8147bea96163b99517fa13e688d44",
    "isdm_ms3.csv" => "29f721f32ee93d8a7926f81481d40027da6539ff4d2b7f6d8cae1aaf3a3358df",
    "isdm_srcform_pois.csv" => "adf10a0f800b45aa8d47b64888dbd8dbb130150d2bf6a33c98213e861d232197",
    "isdm_srcform_mixed.csv" => "afacde410707af181e653e76e4f3478696e06dce738936eb821775272feee382",
    "cloglog_grid_p1.csv" => "481dddc374098a96ef847e8e9611477ba43f42155a0ef18a8d26cb6496523a46",
)

function isdm_fixture_path(name::AbstractString; check::Bool = true)
    path = joinpath(ISDM_FIXTURE_DIR, name)
    if check && haskey(ISDM_FIXTURE_SHA256, name)
        got = bytes2hex(open(sha256, path))
        got == ISDM_FIXTURE_SHA256[name] ||
            error("fixture $name sha256 $got does not match the pinned $(ISDM_FIXTURE_SHA256[name])")
    end
    return path
end

function _isdm_split_csv(line::AbstractString)
    out = String[]; buf = IOBuffer(); inq = false
    for c in line
        if c == '"'
            inq = !inq
        elseif c == ',' && !inq
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
    end
    push!(out, String(take!(buf)))
    return out
end

# Read a fixture into a NamedTuple of columns. A column whose raw fields were
# quoted is a String column (NA -> missing); otherwise numeric (Float64).
function read_isdm_csv(name::AbstractString)
    lines = readlines(isdm_fixture_path(name))
    header = Symbol.(_isdm_split_csv(lines[1]))
    rawq = [Vector{Bool}() for _ in header]
    fields = [String[] for _ in header]
    for ln in lines[2:end]
        isempty(ln) && continue
        parts = _isdm_split_csv(ln)
        for (j, f) in enumerate(parts)
            push!(fields[j], f)
        end
        # was each raw field quoted?
        rq = Bool[]
        inq = false; start = true
        for c in ln
            if start
                push!(rq, c == '"'); start = false
            end
            if c == '"'
                inq = !inq
            elseif c == ',' && !inq
                start = true
            end
        end
        length(rq) < length(parts) && push!(rq, false)
        for j in eachindex(parts)
            push!(rawq[j], rq[j])
        end
    end
    cols = Any[]
    for j in eachindex(header)
        f = fields[j]
        if any(rawq[j])
            v = [x == "NA" ? missing : x for x in f]
            push!(cols, any(ismissing, v) ? v : String.(v))
        else
            v = [x == "NA" ? missing : parse(Float64, x) for x in f]
            push!(cols, any(ismissing, v) ? v : Float64.(v))
        end
    end
    return NamedTuple{Tuple(header)}(Tuple(cols))
end

isdm_r_values() = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "r_values_p1.toml"))

# The four paired cases, as the Julia door spells them (the R spelling is in
# export_p1_fixtures.R). Requires `using GLLVModels, Distributions`.
function isdm_case(name::AbstractString)
    cl = (Binomial(), CLogLogLink())
    if name == "predict"
        return (csv = "isdm_predict.csv",
                formula = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) +
                            latent(0 + trait | cell_id, d = 1, unique = FALSE)),
                family = isdm_sources(gbif = Poisson(), survey = cl))
    elseif name == "ms3"
        return (csv = "isdm_ms3.csv",
                formula = :(value ~ 0 + trait + trait & env + trait & src + offset(log_support) +
                            latent(0 + trait | cell_id, d = 1, unique = FALSE)),
                family = isdm_sources(gbif = Poisson(), literature = Poisson(), survey = cl))
    elseif name == "srcform_pois"
        return (csv = "isdm_srcform_pois.csv",
                formula = :(value ~ 0 + trait + trait & env + offset(log_support)),
                family = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ access)),
                                      inat = Poisson(),
                                      survey = isdm_source(Poisson(); observation = :(~ observer + method))))
    elseif name == "srcform_mixed"
        return (csv = "isdm_srcform_mixed.csv",
                formula = :(value ~ 0 + trait + trait & env + offset(log_support)),
                family = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ access)),
                                      inat = Poisson(),
                                      survey = isdm_source(cl; observation = :(~ observer + method))))
    end
    error("unknown iSDM case $name")
end

const ISDM_CASES = ("predict", "ms3", "srcform_pois", "srcform_mixed")
