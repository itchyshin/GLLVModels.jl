module Family11P1Boundary
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RECEIPT_TOOL = joinpath(ROOT, "tools", "core070_family_p1_receipts.py")

if get(ENV, "GLLVM_P1_RECEIPT_CONTROLS", "0") == "1"
@testset "FAMILY-11 public bridge boundary receipt controls" begin
    python = get(ENV, "GLLVM_P1_RECEIPT_PYTHON", "python3")
    output = read(`$python $RECEIPT_TOOL --family11-boundary-self-test`, String)
    @test occursin("CORE070_FAMILY11_BOUNDARY_SELF_TEST_OK", output)
end
else
    @testset "FAMILY-11 receipt controls require opt-in Python/Git tooling" begin
        @test_skip false
    end
end

end
