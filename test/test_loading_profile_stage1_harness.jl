using Test

include(joinpath(@__DIR__, "..", "tools", "loading_profile", "d3_stage1_harness.jl"))

@testset "D3 loading_profile Stage 1 harness (paste-gated scaffold)" begin
    @test D3_LOADING_PROFILE_STAGE1_PASTE_EXACT == "G0 Stage 1"
    @test !d3_loading_profile_stage1_paste_authorized()
    @test_throws ArgumentError d3_loading_profile_stage1_require_paste()

    usage = d3_loading_profile_stage1_scaffold_usage()
    @test occursin("GLLVM_STAGE1_PASTE", usage)
    @test occursin("loading-profile-confirmatory-substrate.json", usage)
    @test occursin(D3_STAGE1_RUNBOOK_REL, usage)

    root = abspath(joinpath(@__DIR__, ".."))
    cfg = d3_loading_profile_stage1_config(gllvm_root = root)
    plan = d3_loading_profile_stage1_executor_plan(cfg)
    @test occursin("test_loading_profile_confirmatory", plan)
    @test occursin(D3_STAGE1_LEDGER_ROW, plan)

    @test_throws ArgumentError d3_loading_profile_stage1_scaffold_validate!(cfg)
end
