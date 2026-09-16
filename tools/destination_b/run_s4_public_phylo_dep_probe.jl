#!/usr/bin/env julia
# Paste-gated launcher for gllvmTMB PR #1283 isolated S4 public phylo_dep probe.
# Does not edit gllvmTMB; does not run without ENV["GLLVM_S4_PROBE_PASTE"] == "S4 probe yes".

include(joinpath(@__DIR__, "s4_public_phylo_dep_probe_harness.jl"))

function _s4_parse_args(args::Vector{String})
    gllvmtmb = nothing
    project = nothing
    julia_exe = nothing
    julia_env = nothing
    receipt = nothing
    rscript = "Rscript"
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
    julia_env = something(julia_env, project)
    return (
        gllvmtmb_root = gllvmtmb,
        julia_project = project,
        julia_executable = julia_exe,
        julia_env = julia_env,
        receipt_path = receipt,
        rscript_executable = rscript,
    )
end

function main(args = ARGS)
    if !s4_public_phylo_dep_paste_authorized()
        print(s4_public_phylo_dep_scaffold_usage())
        exit(2)
    end
    kw = _s4_parse_args(collect(String, args))
    cfg = s4_public_phylo_dep_probe_config(; kw...)
    s4_public_phylo_dep_run!(cfg)
    println("S4_PUBLIC_PHYLO_DEP_PROBE_DONE receipt=", cfg.receipt_path)
end

main()
