#!/usr/bin/env julia
# Paste-gated scaffold for Totoro #323 Track A (Frozen R smoke, D-139).
# Does not SSH; does not run runparity.jl without Codex on Totoro.

include(joinpath(@__DIR__, "totoro_323_track_a_harness.jl"))

function _track_a_parse_args(args::Vector{String})
    gllvm = nothing
    track = "A"
    stamp = ""
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--gllvm-root"
            i += 1
            gllvm = args[i]
        elseif arg == "--track"
            i += 1
            track = args[i]
        elseif arg == "--receipt-stamp"
            i += 1
            stamp = args[i]
        elseif arg in ("-h", "--help")
            print(totoro_323_track_a_scaffold_usage())
            exit(0)
        else
            error("unknown argument: $arg (try --help)")
        end
        i += 1
    end
    gllvm === nothing && error("missing required argument --gllvm-root")
    return (gllvm_root = gllvm, track = track, receipt_stamp = stamp)
end

function main(args = ARGS)
    if !totoro_323_track_a_paste_authorized()
        print(totoro_323_track_a_scaffold_usage())
        exit(2)
    end
    kw = _track_a_parse_args(collect(String, args))
    cfg = totoro_323_track_a_config(; kw...)
    totoro_323_track_a_scaffold_validate!(cfg)
    println("TOTORO_323_TRACK_A_SCAFFOLD_OK gllvm_root=", cfg.gllvm_root)
end

main()
