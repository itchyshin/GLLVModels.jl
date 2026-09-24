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
    @test occursin("--dry-run", usage)
    @test occursin("TEMPLATE-totoro-323-track-a-receipt", usage)

    root = abspath(joinpath(@__DIR__, ".."))
    cfg = totoro_323_track_a_config(gllvm_root = root)
    @test cfg.track == "A"
    @test occursin("runparity.jl", totoro_323_track_a_executor_plan(cfg))

    @test_throws ArgumentError totoro_323_track_a_config(
        gllvm_root = root,
        track = "B",
    )

    template = totoro_323_track_a_after_task_receipt_template(root)
    @test isfile(template)
    @test occursin("TEMPLATE", basename(template))

    result = totoro_323_track_a_preflight!(cfg; dry_run = true)
    @test result.report.dry_run
    @test result.report.gllvm_root == root
    @test isfile(result.report.frozen_contract_path)

    summary = totoro_323_track_a_preflight_summary(result.report)
    @test occursin("DRY-RUN", summary)
    @test occursin(TOTORO_323_FROZEN_GLLVMTMB_REF, summary)

    err = nothing
    try
        totoro_323_track_a_require_paste()
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("GLLVM_TOTORO_PASTE", string(err))
end
