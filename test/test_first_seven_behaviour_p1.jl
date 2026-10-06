module FirstSevenP1ReceiptControls
using Test
if get(ENV, "GLLVM_P1_RECEIPT_CONTROLS", "0") == "1"
    tool = joinpath(@__DIR__, "..", "tools", "test_core070_behaviour_receipts.py")
    @testset "P1 behavioural receipt controls (fit-free)" begin
        python = get(ENV, "GLLVM_P1_RECEIPT_PYTHON", "python3")
    output = read(`$python $tool`, String)
        @test occursin("controls passed", output)
    end
else
    @testset "P1 behavioural receipt controls require opt-in R/Python tooling" begin
        @test_skip false
    end
end
end
