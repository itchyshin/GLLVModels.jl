#!/usr/bin/env julia
# Paste-gated scaffold for D3 confirmatory `loading_profile` Stage 1.
# Does not export `loading_profile` or run confirmatory grids.

include(joinpath(@__DIR__, "d3_stage1_harness.jl"))

function _stage1_parse_args(args::Vector{String})
    gllvm = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--gllvm-root"
            i += 1
            gllvm = args[i]
        elseif arg in ("-h", "--help")
            print(d3_loading_profile_stage1_scaffold_usage())
            exit(0)
        else
            error("unknown argument: $arg (try --help)")
        end
        i += 1
    end
    gllvm === nothing && error("missing required argument --gllvm-root")
    return (gllvm_root = gllvm,)
end

function main(args = ARGS)
    if !d3_loading_profile_stage1_paste_authorized()
        print(d3_loading_profile_stage1_scaffold_usage())
        exit(2)
    end
    kw = _stage1_parse_args(collect(String, args))
    cfg = d3_loading_profile_stage1_config(; kw...)
    d3_loading_profile_stage1_scaffold_validate!(cfg)
    println("D3_LOADING_PROFILE_STAGE1_SCAFFOLD_OK gllvm_root=", cfg.gllvm_root)
end

main()
