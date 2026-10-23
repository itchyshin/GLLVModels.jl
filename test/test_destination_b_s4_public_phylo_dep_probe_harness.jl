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

    # GLLVM_S4_JULIA_HOME must be the *directory* holding the `julia`
    # executable, not the executable file itself: the frozen recorder's
    # `s4_public_phylo_dep_clean_julia_probe()` in
    # run-destination-b-s4-public-phylo-dep-isolated.R resolves it as
    # `file.path(normalizePath(julia_home, mustWork = TRUE), "julia")`
    # (falling back to `.../bin/julia`) — passing the executable file itself
    # makes that `file.path()` call append a second `julia` segment onto a
    # file, which never exists.
    let executable = joinpath(Sys.BINDIR, Base.julia_exename()),
        cfg = S4PublicPhyloDepProbeConfig(
            @__DIR__,
            abspath(joinpath(@__DIR__, "..")),
            executable,
            abspath(joinpath(@__DIR__, "..")),
            joinpath(tempdir(), "s4-probe-harness-unused-receipt.json"),
            "Rscript",
        )
        env = s4_public_phylo_dep_r_environment(cfg)
        @test env["GLLVM_S4_JULIA_HOME"] == dirname(executable)
        @test env["GLLVM_S4_JULIA_HOME"] != executable
        @test isfile(joinpath(env["GLLVM_S4_JULIA_HOME"], basename(executable)))

        # The R child command must build on Julia 1.10: `Cmd(::Vector; dir=)`
        # has no such method (first live run, 2026-09-24), so the working
        # directory goes through `setenv(cmd, env; dir=)`.
        cmd = s4_public_phylo_dep_r_command(cfg)
        @test cmd isa Cmd
        @test cmd.dir == cfg.gllvmtmb_root
        @test cmd.exec == ["Rscript", "--vanilla",
            joinpath(cfg.gllvmtmb_root, S4_PUBLIC_PHYLO_DEP_RUNNER_REL)]
        @test "GLLVM_S4_LIVE_FORMULA_TESTS=1" in cmd.env
    end

    # GLLVM_S4_JULIA_ENV defaults to the committed probe-only environment
    # (develops the repo root; carries LogExpFunctions as a direct dep) when
    # the caller does not pass julia_env explicitly.
    let julia_project = abspath(joinpath(@__DIR__, ".."))
        @test s4_public_phylo_dep_probe_env(julia_project) ==
              joinpath(julia_project, "tools", "destination_b", "probe_env")
        @test isfile(joinpath(s4_public_phylo_dep_probe_env(julia_project), "Project.toml"))
    end

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

    gllvm_root = abspath(joinpath(@__DIR__, ".."))
    template = s4_public_phylo_dep_after_task_receipt_template(gllvm_root)
    @test isfile(template)
    @test occursin("TEMPLATE", basename(template))

    err = nothing
    try
        s4_public_phylo_dep_require_paste()
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("GLLVM_S4_PROBE_PASTE", string(err))

    summary = s4_public_phylo_dep_preflight_summary(S4PublicPhyloDepPreflightReport(
        "97214679cdeadbeef",
        "97214679c0000000",
        S4_FROZEN_ORACLE_PIN,
        template,
        gllvm_root,
        joinpath(tempdir(), "s4-probe-dry-run-receipt.json"),
        true,
    ))
    @test occursin("DRY-RUN", summary)
    @test occursin(S4_FROZEN_ORACLE_PIN, summary)
end
