#!/usr/bin/env julia
# Paste-gated launcher for gllvmTMB PR #1283 isolated S4 public phylo_dep probe.
# Does not edit gllvmTMB; does not run without ENV["GLLVM_S4_PROBE_PASTE"] == "S4 probe yes"
# unless --dry-run (preflight only; no Rscript).

include(joinpath(@__DIR__, "s4_public_phylo_dep_probe_harness.jl"))

function _s4_parse_args(args::Vector{String})
    gllvmtmb = nothing
    project = nothing
    julia_exe = nothing
    julia_env = nothing
    receipt = nothing
    rscript = "Rscript"
    dry_run = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--gllvmtmb-root"
            i += 1
            gllvmtmb = args[i]
        elseif arg == "--julia-project"
            i += 1
            project = args[i]
        elseif arg == "--julia"
            i += 1
            julia_exe = args[i]
        elseif arg == "--julia-env"
            i += 1
            julia_env = args[i]
        elseif arg == "--receipt"
            i += 1
            receipt = args[i]
        elseif arg == "--rscript"
            i += 1
            rscript = args[i]
        elseif arg == "--dry-run"
            dry_run = true
        elseif arg in ("-h", "--help")
            print(s4_public_phylo_dep_scaffold_usage())
            exit(0)
        else
            error("unknown argument: $arg (try --help)")
        end
        i += 1
    end
    for (name, val) in (
        ("--gllvmtmb-root", gllvmtmb),
        ("--julia-project", project),
        ("--julia", julia_exe),
        ("--receipt", receipt),
    )
        val === nothing && error("missing required argument $name")
    end
    julia_env = something(julia_env, s4_public_phylo_dep_probe_env(project))
    gllvm_root = abspath(String(project))
    while !isfile(joinpath(gllvm_root, "Project.toml")) && gllvm_root != dirname(gllvm_root)
        gllvm_root = dirname(gllvm_root)
    end
    return (
        gllvmtmb_root = gllvmtmb,
        julia_project = project,
        julia_executable = julia_exe,
        julia_env = julia_env,
        receipt_path = receipt,
        rscript_executable = rscript,
        dry_run = dry_run,
        gllvm_root = gllvm_root,
    )
end

function main(args = ARGS)
    str_args = collect(String, args)
    dry_run = "--dry-run" in str_args
    if !dry_run && !s4_public_phylo_dep_paste_authorized()
        print(s4_public_phylo_dep_scaffold_usage())
        exit(2)
    end
    parsed = _s4_parse_args(str_args)
    dry_run = parsed.dry_run
    result = s4_public_phylo_dep_preflight!(;
        gllvmtmb_root = parsed.gllvmtmb_root,
        julia_project = parsed.julia_project,
        julia_executable = parsed.julia_executable,
        julia_env = parsed.julia_env,
        receipt_path = parsed.receipt_path,
        rscript_executable = parsed.rscript_executable,
        dry_run = dry_run,
        gllvm_root = parsed.gllvm_root,
    )
    print(s4_public_phylo_dep_preflight_summary(result.report))
    if dry_run
        rscript = parsed.rscript_executable
        if !isfile(abspath(String(rscript)))
            println("note: Rscript not found at $(rscript); required for live probe after paste")
        end
        println("S4_PUBLIC_PHYLO_DEP_PREFLIGHT_DRY_RUN_OK")
        exit(0)
    end
    s4_public_phylo_dep_run!(result.cfg)
    println("S4_PUBLIC_PHYLO_DEP_PROBE_DONE receipt=", result.cfg.receipt_path)
end

main()
