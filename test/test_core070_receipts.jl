using Test
using SHA
using TOML

include(joinpath(@__DIR__, "parity", "core070_receipts.jl"))
using .Core070Receipts

# Execute the actual group helper in a stdlib-only module; importing RCall is
# unnecessary for this receipt-accounting regression (no model fit is run).
module GroupReceiptHarness
using Test
import ..Core070Receipts: record_case!, testset_counts
const _CORE070_RUN = Ref{Any}(nothing)
const _CORE070_ACTIVE_CELLS = Ref{Vector{String}}(String[])
const _RECEIPT_DIR = Ref("")
_core070_receipt_dir() = _RECEIPT_DIR[]
_core070_required() = true
core070_case_requested(id) = id in _CORE070_RUN[].requested_case_ids
using TOML
end
helper_ast = Meta.parseall(read(joinpath(@__DIR__, "parity", "parity_helpers.jl"), String))
for name in (:core070_execute_group!, :core070_record_values!)
    definition = only(filter(helper_ast.args) do node
        node isa Expr && node.head == :function && node.args[1] isa Expr &&
            node.args[1].head == :call && node.args[1].args[1] == name
    end)
    Core.eval(GroupReceiptHarness, definition)
end

_sha(path) = bytes2hex(sha256(read(path)))

@testset "CORE-070 receipt kernel" begin
    nested = @testset "outer fixture" begin
        @testset "inner fixture" begin
            @test true
        end
    end
    @test testset_counts(nested)["passed"] == 1

    mktempdir() do tmp
        expected = joinpath(tmp, "expected")
        other = joinpath(tmp, "other")
        mkpath(joinpath(expected, "src")); mkpath(joinpath(other, "src"))
        write(joinpath(expected, "src", "GLLVModels.jl"), "module GLLVModels end")
        write(joinpath(other, "src", "GLLVModels.jl"), "module GLLVModels end")
        @test verify_loaded_source(expected, expected, joinpath(expected, "src", "GLLVModels.jl"))
        @test_throws ArgumentError verify_loaded_source(expected, other, joinpath(other, "src", "GLLVModels.jl"))
        @test_throws ArgumentError verify_loaded_source(expected, expected, joinpath(other, "src", "GLLVModels.jl"))
        fixture = joinpath(tmp, "fixture.jl")
        write(fixture, "# fixture\n")
        inventory_file = joinpath(tmp, "runner.jl")
        write(inventory_file, "# runner\n")
        inventory = execution_inventory(tmp, ["runner.jl"])
        source = Dict("reference_commit" => "pin")

        @test_throws ArgumentError start_run!(tmp;
            requested_case_ids = ["case-a", "case-a"], source, inventory,
            contract_sha256 = "contract")

        run = start_run!(joinpath(tmp, "missing");
            requested_case_ids = ["case-a", "case-b"], source, inventory,
            contract_sha256 = "contract")
        record_case!(run, "case-a", fixture; passed = 2)
        @test_throws ArgumentError finish_run!(run)
        started = TOML.parsefile(joinpath(tmp, "missing", "run.toml"))
        @test started["status"] == "failed"
        @test started["requested_case_ids"] == ["case-a", "case-b"]
        @test started["completed_case_ids"] == ["case-a"]

        run = start_run!(joinpath(tmp, "zero");
            requested_case_ids = ["case-a"], source, inventory,
            contract_sha256 = "contract")
        record_case!(run, "case-a", fixture; passed = 0)
        @test_throws ArgumentError finish_run!(run)
        cell = TOML.parsefile(joinpath(tmp, "zero", "cell-case-a.toml"))
        @test cell["status"] == "failed"
        @test cell["assertions"]["passed"] == 0

        run = start_run!(joinpath(tmp, "broken");
            requested_case_ids = ["case-a"], source, inventory,
            contract_sha256 = "contract")
        record_case!(run, "case-a", fixture; passed = 1, broken = 1)
        @test_throws ArgumentError finish_run!(run)
        cell = TOML.parsefile(joinpath(tmp, "broken", "cell-case-a.toml"))
        @test cell["status"] == "failed"
        @test cell["assertions"]["broken"] == 1

        run = start_run!(joinpath(tmp, "success");
            requested_case_ids = ["case-a"], source, inventory,
            contract_sha256 = "contract")
        record_case!(run, "case-a", fixture; passed = 2)
        @test_throws ArgumentError record_case!(run, "case-a", fixture; passed = 2)
        finish_run!(run)
        complete = TOML.parsefile(joinpath(tmp, "success", "run.toml"))
        @test complete["status"] == "success"
        @test complete["scope"] == "subset"
        @test complete["completed_case_ids"] == ["case-a"]
        @test complete["actual_assertions"] == 2
        @test complete["execution"]["manifest_sha256"] == inventory["manifest_sha256"]
        @test complete["contract_sha256"] == "contract"

        @test _sha(fixture) == complete["cells"]["case-a"]["fixture_sha256"]

        @testset "one execution can cover several required IDs" begin
            run = start_run!(joinpath(tmp, "grouped");
                requested_case_ids = ["a", "b", "c"], source, inventory,
                contract_sha256 = "contract")
            for id in ["a", "b", "c"]
                record_case!(run, id, fixture; passed = 28, execution_case_ids = ["c", "a", "b"])
            end
            finish_run!(run)
            receipt = TOML.parsefile(joinpath(run.dir, "run.toml"))
            @test receipt["actual_assertions"] == 28
            @test receipt["assertion_counting"] == "execution_groups_v1"
            @test length(receipt["completed_case_ids"]) == 3
            @test receipt["cells"]["a"]["execution_case_ids"] == ["a", "b", "c"]
        end

        @testset "independent executions of same file both count" begin
            run = start_run!(joinpath(tmp, "independent");
                requested_case_ids = ["a", "b"], source, inventory,
                contract_sha256 = "contract")
            record_case!(run, "a", fixture; passed = 2)
            record_case!(run, "b", fixture; passed = 3)
            finish_run!(run)
            @test TOML.parsefile(joinpath(run.dir, "run.toml"))["actual_assertions"] == 5
        end

        @testset "actual shared fixture helper supplies execution membership" begin
            run = start_run!(joinpath(tmp, "actual-helper");
                requested_case_ids = ["a", "b", "c"], source, inventory,
                contract_sha256 = "contract")
            GroupReceiptHarness._CORE070_RUN[] = run
            invocations = Ref(0)
            GroupReceiptHarness.core070_execute_group!(["a", "b", "c"], fixture, () -> begin
                invocations[] += 1
                @test 2 + 2 == 4
                @test true
            end)
            finish_run!(run)
            @test invocations[] == 1
            @test TOML.parsefile(joinpath(run.dir, "run.toml"))["actual_assertions"] == 2
            @test all(cell["execution_case_ids"] == ["a", "b", "c"] for cell in values(run.cells))
        end

        @testset "invalid groups fail without erasing earlier cells" begin
            run = start_run!(joinpath(tmp, "bad-group");
                requested_case_ids = ["a", "b", "c"], source, inventory,
                contract_sha256 = "contract")
            @test_throws ArgumentError record_case!(run, "a", fixture; passed = 2, execution_case_ids = ["b"])
            @test_throws ArgumentError record_case!(run, "a", fixture; passed = 2, execution_case_ids = ["a", "a"])
            @test_throws ArgumentError record_case!(run, "a", fixture; passed = 2, execution_case_ids = ["a", "unknown"])
            @test_throws ArgumentError record_case!(run, "a", fixture; passed = true)
            record_case!(run, "a", fixture; passed = 2, execution_case_ids = ["a", "b"])
            @test_throws ArgumentError record_case!(run, "b", fixture; passed = 3, execution_case_ids = ["a", "b"])
            @test_throws ArgumentError record_case!(run, "b", inventory_file; passed = 2, execution_case_ids = ["a", "b"])
            @test_throws ArgumentError record_case!(run, "b", fixture; passed = 2)
            @test_throws ArgumentError record_case!(run, "c", fixture; passed = 2, execution_case_ids = ["b", "c"])
            @test_throws ArgumentError finish_run!(run)
            @test TOML.parsefile(joinpath(run.dir, "run.toml"))["status"] == "failed"
            @test length(run.cells) == 1
        end
    end
end
