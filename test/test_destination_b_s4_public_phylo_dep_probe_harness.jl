using Test

include(joinpath(@__DIR__, "..", "tools", "destination_b",
    "s4_public_phylo_dep_probe_harness.jl"))

@testset "S4 public phylo_dep probe harness (paste-gated scaffold)" begin
    @test S4_PUBLIC_PHYLO_DEP_PASTE_EXACT == "S4 probe yes"
    @test !s4_public_phylo_dep_paste_authorized()
    @test_throws ArgumentError s4_public_phylo_dep_require_paste()

    usage = s4_public_phylo_dep_scaffold_usage()
    @test occursin("97214679c", usage)
    @test occursin("GLLVM_S4_PROBE_PASTE", usage)

    @test_throws ArgumentError s4_public_phylo_dep_verify_recorder_tip(
        joinpath(@__DIR__, "nonexistent_gllvmtmb_root"))

    gllvmtmb = get(ENV, "GLLVM_TEST_GLLVMTMB_ROOT", "")
    if !isempty(gllvmtmb) && isdir(gllvmtmb)
        try
            verified = s4_public_phylo_dep_verify_recorder_tip(gllvmtmb)
            @test isfile(verified.runner)
            env = s4_public_phylo_dep_r_environment(s4_public_phylo_dep_probe_config(
                gllvmtmb_root = gllvmtmb,
                julia_project = abspath(joinpath(@__DIR__, "..")),
                julia_executable = get(ENV, "JULIA_EXECUTABLE", joinpath(Sys.BINDIR, "julia")),
                receipt_path = joinpath(tempdir(), "s4-probe-harness-test-receipt.json"),
            ))
            @test env["GLLVM_S4_LIVE_FORMULA_TESTS"] == "1"
            @test haskey(env, "GLLVM_DESTINATION_B_PROJECT")
        catch err
            @test err isa ArgumentError
            @test occursin("recorder", lowercase(string(err)))
        end
    end

    @test_throws ArgumentError s4_public_phylo_dep_probe_config(
        gllvmtmb_root = @__DIR__,
        julia_project = abspath(joinpath(@__DIR__, "..")),
        julia_executable = joinpath(Sys.BINDIR, "julia"),
        receipt_path = joinpath(tempdir(), "s4-probe-harness-test-receipt.json"),
    )
end
