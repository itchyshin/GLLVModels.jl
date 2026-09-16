using Test

include(joinpath(@__DIR__, "..", "tools", "totoro323", "totoro_323_track_a_harness.jl"))

@testset "Totoro #323 Track A harness (paste-gated scaffold)" begin
    @test TOTORO_323_TRACK_A_PASTE_EXACT == "ack Totoro D-139 #323 Track A"
    @test !totoro_323_track_a_paste_authorized()
    @test_throws ArgumentError totoro_323_track_a_require_paste()

    usage = totoro_323_track_a_scaffold_usage()
    @test occursin("GLLVM_TOTORO_PASTE", usage)
    @test occursin(TOTORO_323_FROZEN_GLLVMTMB_REF_SHORT, usage)
    @test occursin("2026-09-14-issue-323-totoro-launch-pack", usage)

    root = abspath(joinpath(@__DIR__, ".."))
    cfg = totoro_323_track_a_config(gllvm_root = root)
    @test cfg.track == "A"
    @test occursin("runparity.jl", totoro_323_track_a_executor_plan(cfg))

    @test_throws ArgumentError totoro_323_track_a_config(
        gllvm_root = root,
        track = "B",
    )
end
