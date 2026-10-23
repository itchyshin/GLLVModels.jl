#!/usr/bin/env julia
# Paste-gated scaffold for Totoro #323 Track A (Frozen R smoke, D-139).
# Does not SSH; does not run runparity.jl without Codex on Totoro.
# --dry-run: preflight only (no paste, no executor plan).

include(joinpath(@__DIR__, "totoro_323_track_a_harness.jl"))

function _track_a_parse_args(args::Vector{String})
    gllvm = nothing
    track = "A"
    stamp = ""
    dry_run = false
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
        elseif arg == "--dry-run"
            dry_run = true
        elseif arg in ("-h", "--help")
            print(totoro_323_track_a_scaffold_usage())
            exit(0)
        else
            error("unknown argument: $arg (try --help)")
        end
        i += 1
    end
    gllvm === nothing && error("missing required argument --gllvm-root")
    return (
        gllvm_root = gllvm,
        track = track,
        receipt_stamp = stamp,
        dry_run = dry_run,
    )
end

function main(args = ARGS)
    str_args = collect(String, args)
    dry_run = "--dry-run" in str_args
    if !dry_run && !totoro_323_track_a_paste_authorized()
        print(totoro_323_track_a_scaffold_usage())
        exit(2)
    end
    kw = _track_a_parse_args(str_args)
    dry_run = kw.dry_run
    cfg = totoro_323_track_a_config(;
        gllvm_root = kw.gllvm_root,
        track = kw.track,
        receipt_stamp = kw.receipt_stamp,
    )
    result = totoro_323_track_a_preflight!(cfg; dry_run = dry_run)
    print(totoro_323_track_a_preflight_summary(result.report))
    if dry_run
        println("TOTORO_323_TRACK_A_PREFLIGHT_DRY_RUN_OK")
        exit(0)
    end
    totoro_323_track_a_scaffold_validate!(cfg)
    println("TOTORO_323_TRACK_A_SCAFFOLD_OK gllvm_root=", cfg.gllvm_root)
end

main()
