using Test
using TOML

# Fixture index test (#276). test/fixtures/comparator_targets.toml is a machine-readable index of
# serialized fixtures that already live in this repo. It is an index only: it names the fixture, the
# estimator target and a claim boundary, and it asserts no external comparator. This test checks the
# index is well formed and points at files that exist; it flips no capability row.

const _CT_ROOT = dirname(@__DIR__)
const _CT_INDEX = joinpath(@__DIR__, "fixtures", "comparator_targets.toml")
const _CT_README = joinpath(@__DIR__, "fixtures", "README_comparator_targets.md")
const _CT_REQUIRED_KEYS = ("id", "path", "target", "evidence_type", "comparator_status", "claim_boundary")
const _CT_EVIDENCE_TYPES = ("julia_target",)
const _CT_COMPARATOR_STATUS = ("not asserted",)

# Returns a list of human-readable problems; empty means the index is well formed.
function _comparator_target_problems(doc::AbstractDict, root::AbstractString)
    problems = String[]
    get(doc, "schema_version", nothing) == 1 || push!(problems, "schema_version must be 1")
    rows = get(doc, "target", nothing)
    if !(rows isa AbstractVector) || isempty(rows)
        push!(problems, "no [[target]] rows")
        return problems
    end
    seen = Set{String}()
    for (i, row) in enumerate(rows)
        label = "target[$i]"
        for k in _CT_REQUIRED_KEYS
            if !haskey(row, k)
                push!(problems, "$label: missing key '$k'")
            elseif !(row[k] isa AbstractString) || isempty(strip(row[k]))
                push!(problems, "$label: key '$k' must be a non-empty string")
            end
        end
        if haskey(row, "id") && row["id"] isa AbstractString
            row["id"] in seen && push!(problems, "$label: duplicate id '$(row["id"])'")
            push!(seen, row["id"])
        end
        if haskey(row, "path") && row["path"] isa AbstractString && !isempty(row["path"])
            isfile(joinpath(root, row["path"])) ||
                push!(problems, "$label: path '$(row["path"])' does not exist")
        end
        if haskey(row, "evidence_type") && !(row["evidence_type"] in _CT_EVIDENCE_TYPES)
            push!(problems, "$label: evidence_type '$(row["evidence_type"])' not in $(_CT_EVIDENCE_TYPES)")
        end
        if haskey(row, "comparator_status") && !(row["comparator_status"] in _CT_COMPARATOR_STATUS)
            push!(problems, "$label: comparator_status '$(row["comparator_status"])' not in $(_CT_COMPARATOR_STATUS)")
        end
    end
    return problems
end

@testset "comparator_targets fixture index (#276)" begin
    @test isfile(_CT_INDEX)
    doc = TOML.parsefile(_CT_INDEX)

    @testset "the shipped index is well formed" begin
        @test isempty(_comparator_target_problems(doc, _CT_ROOT))
        @test length(doc["target"]) >= 5
    end

    @testset "the guard fails on a broken index" begin
        # Deleted fixture path.
        bad = deepcopy(doc)
        bad["target"][1]["path"] = "test/fixtures/__does_not_exist__.toml"
        @test any(occursin("does not exist", p) for p in _comparator_target_problems(bad, _CT_ROOT))

        # Each required key removed in turn.
        for k in _CT_REQUIRED_KEYS
            bad = deepcopy(doc)
            delete!(bad["target"][1], k)
            @test any(occursin("missing key '$k'", p) for p in _comparator_target_problems(bad, _CT_ROOT))
        end

        # Empty claim boundary.
        bad = deepcopy(doc)
        bad["target"][1]["claim_boundary"] = "  "
        @test any(occursin("non-empty", p) for p in _comparator_target_problems(bad, _CT_ROOT))

        # A comparator claim the index may not make.
        bad = deepcopy(doc)
        bad["target"][1]["comparator_status"] = "matched"
        @test any(occursin("comparator_status", p) for p in _comparator_target_problems(bad, _CT_ROOT))

        # Wrong schema version, duplicate id.
        bad = deepcopy(doc)
        bad["schema_version"] = 2
        @test any(occursin("schema_version", p) for p in _comparator_target_problems(bad, _CT_ROOT))
        bad = deepcopy(doc)
        bad["target"][2]["id"] = bad["target"][1]["id"]
        @test any(occursin("duplicate id", p) for p in _comparator_target_problems(bad, _CT_ROOT))
    end

    @testset "README states the boundary" begin
        @test isfile(_CT_README)
        readme = read(_CT_README, String)
        @test occursin("fixture index only, not comparator evidence", readme)
        @test occursin("flips no covered row", readme)
    end
end
