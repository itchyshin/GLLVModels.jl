using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RECEIPT_TOOL = joinpath(ROOT, "tools", "core070_family_p1_receipts.py")

@testset "FAMILY-11 public bridge boundary receipt controls" begin
    output = read(`python3 $RECEIPT_TOOL --family11-boundary-self-test`, String)
    @test occursin("CORE070_FAMILY11_BOUNDARY_SELF_TEST_OK", output)
end
