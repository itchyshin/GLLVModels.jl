"""
    Totoro #323 Track A — paste-gated launcher harness (DRAFT)

Runbook: `docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md`.
Canonical commands: `docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md`.

This harness does **not** SSH to Totoro or run `runparity.jl`. Codex executes the launch
pack on Totoro after maintainer paste. The driver only enforces the paste gate and prints
the executor checklist.
"""

const TOTORO_323_TRACK_A_PASTE_EXACT = "ack Totoro D-139 #323 Track A"
const TOTORO_323_FROZEN_GLLVMTMB_REF =
    "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const TOTORO_323_FROZEN_GLLVMTMB_REF_SHORT = "b4d5fee6"
const TOTORO_323_LAUNCH_PACK_REL =
    "docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md"
const TOTORO_323_RUNBOOK_REL =
    "docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md"
const TOTORO_323_TRACK_A_PARITY_RUNNER = "test/parity/runparity.jl"

_track_a_fail(message) = throw(ArgumentError("Totoro #323 Track A harness: " * message))

"""
    totoro_323_track_a_paste_authorized() -> Bool

True only when `ENV["GLLVM_TOTORO_PASTE"]` equals the maintainer paste string exactly.
"""
function totoro_323_track_a_paste_authorized()
    return get(ENV, "GLLVM_TOTORO_PASTE", "") == TOTORO_323_TRACK_A_PASTE_EXACT
end

"""
    totoro_323_track_a_require_paste()

Fail closed unless the maintainer paste is present in the environment.
"""
function totoro_323_track_a_require_paste()
    totoro_323_track_a_paste_authorized() ||
        _track_a_fail("refusing Track A launcher without paste " *
            repr(TOTORO_323_TRACK_A_PASTE_EXACT) *
            " in ENV[\"GLLVM_TOTORO_PASTE\"]")
    return nothing
end

"""
    Totoro323TrackAConfig

Paths for validating a local GLLVM.jl workspace before Codex runs on Totoro.
"""
struct Totoro323TrackAConfig
    gllvm_root::String
    track::String
    receipt_stamp::String
end

function totoro_323_track_a_config(;
    gllvm_root::AbstractString,
    track::AbstractString = "A",
    receipt_stamp::AbstractString = "",
)
    root = abspath(String(gllvm_root))
    isdir(root) || _track_a_fail("GLLVM root is not a directory: $root")
    isfile(joinpath(root, "Project.toml")) ||
        _track_a_fail("GLLVM root lacks Project.toml: $root")
    isfile(joinpath(root, TOTORO_323_TRACK_A_PARITY_RUNNER)) ||
        _track_a_fail("missing parity runner at $(TOTORO_323_TRACK_A_PARITY_RUNNER)")
    track_u = uppercase(strip(String(track)))
    track_u == "A" || _track_a_fail("this harness is Track A only (got track=$(repr(track)))")
    stamp = String(receipt_stamp)
    return Totoro323TrackAConfig(root, track_u, stamp)
end

"""
    totoro_323_track_a_scaffold_usage() -> String

Plain-language instructions when the launcher is invoked without paste.
"""
function totoro_323_track_a_scaffold_usage()
    return """
    Totoro #323 Track A launcher (DRAFT — paste-gated, not merge-ready)

    Frozen gllvmTMB pin: $(TOTORO_323_FROZEN_GLLVMTMB_REF_SHORT)
    Runbook: $(TOTORO_323_RUNBOOK_REL)
    Launch pack: $(TOTORO_323_LAUNCH_PACK_REL)

    1. Maintainer paste in chat (exact):
       $(TOTORO_323_TRACK_A_PASTE_EXACT)
    2. Export for Codex executor (does not authorize Cursor SSH from this repo):
       export GLLVM_TOTORO_PASTE='$(TOTORO_323_TRACK_A_PASTE_EXACT)'
    3. Validate workspace (no Totoro spend from this driver):
       julia --project=. tools/totoro323/run_totoro_323_track_a_launcher.jl \\
         --gllvm-root $(pwd())
    4. Codex runs launch pack Runner block on Totoro (steps 0–5a).

    Without the paste this launcher exits non-zero and does not print an execution plan.
    """
end

"""
    totoro_323_track_a_executor_plan(cfg::Totoro323TrackAConfig) -> String

Checklist text for Codex after paste; not executed by this harness.
"""
function totoro_323_track_a_executor_plan(cfg::Totoro323TrackAConfig)
    stamp = cfg.receipt_stamp
    if isempty(stamp)
        stamp = "<set RECEIPT_STAMP on Totoro>"
    end
    return """
    Totoro #323 Track A — Codex executor plan (paste acknowledged; no SSH from GLLVM.jl lane)

    GLLVM_ROOT=$(cfg.gllvm_root)
    TRACK=A
    RECEIPT_STAMP=$(stamp)
    gllvmTMB pin=$(TOTORO_323_FROZEN_GLLVMTMB_REF)
    Parity entry: julia --project=test/parity $(TOTORO_323_TRACK_A_PARITY_RUNNER)

    Follow steps 0–5a in $(TOTORO_323_LAUNCH_PACK_REL).
    D-139 band: ~90–150 min wall; single Julia process; OPENBLAS/JULIA threads = 1.
    """
end

"""
    totoro_323_track_a_scaffold_validate!(cfg::Totoro323TrackAConfig)

Requires maintainer paste; prints executor plan only (no oracle build, no parity run).
"""
function totoro_323_track_a_scaffold_validate!(cfg::Totoro323TrackAConfig)
    totoro_323_track_a_require_paste()
    print(totoro_323_track_a_executor_plan(cfg))
    return cfg
end
