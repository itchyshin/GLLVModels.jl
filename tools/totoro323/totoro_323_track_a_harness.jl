"""
    Totoro #323 Track A — paste-gated launcher harness (DRAFT)

Runbook: `docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md`.
Canonical commands: `docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md`.

This harness does **not** SSH to Totoro or run `runparity.jl`. Codex executes the launch
pack on Totoro after maintainer paste. The driver enforces the paste gate and prints
the executor checklist.
"""

const TOTORO_323_TRACK_A_PASTE_EXACT = "ack Totoro D-139 #323 Track A"
const TOTORO_323_FROZEN_GLLVMTMB_REF =
    "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const TOTORO_323_FROZEN_GLLVMTMB_REF_SHORT = "b4d5fee6"
const TOTORO_323_FROZEN_CONTRACT_REL =
    "docs/dev-log/core070/frozen-r070-contract.toml"
const TOTORO_323_LAUNCH_PACK_REL =
    "docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md"
const TOTORO_323_RUNBOOK_REL =
    "docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md"
const TOTORO_323_TRACK_A_PARITY_RUNNER = "test/parity/runparity.jl"
const TOTORO_323_AFTER_TASK_RECEIPT_TEMPLATE_REL =
    "docs/dev-log/after-task/TEMPLATE-totoro-323-track-a-receipt.md"

function _track_a_fail(message::AbstractString, hint::AbstractString = "")
    body = "Totoro #323 Track A harness: " * message
    if !isempty(hint)
        body *= "\n  → " * hint
    end
    return throw(ArgumentError(body))
end

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
    if totoro_323_track_a_paste_authorized()
        return nothing
    end
    got = get(ENV, "GLLVM_TOTORO_PASTE", "")
    if isempty(got)
        _track_a_fail(
            "refusing Track A launcher: ENV[\"GLLVM_TOTORO_PASTE\"] is unset",
            "after maintainer paste, export GLLVM_TOTORO_PASTE='$(TOTORO_323_TRACK_A_PASTE_EXACT)'",
        )
    else
        _track_a_fail(
            "refusing Track A launcher: paste mismatch (got $(repr(got)))",
            "exact paste required: $(repr(TOTORO_323_TRACK_A_PASTE_EXACT))",
        )
    end
end

"""
    totoro_323_track_a_after_task_receipt_template(gllvm_root::AbstractString) -> String

Absolute path to the markdown receipt template (for post-Totoro after-task).
"""
function totoro_323_track_a_after_task_receipt_template(gllvm_root::AbstractString)
    return abspath(joinpath(gllvm_root, TOTORO_323_AFTER_TASK_RECEIPT_TEMPLATE_REL))
end

function _track_a_git_head(gllvm_root::AbstractString)
    try
        return strip(read(Cmd(["git", "-C", gllvm_root, "rev-parse", "HEAD"]), String))
    catch
        return nothing
    end
end

"""
    Totoro323TrackAConfig

Paths for validating a local GLLVModels.jl workspace before Codex runs on Totoro.
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
    isdir(root) ||
        _track_a_fail(
            "GLLVM root is not a directory: $root",
            "pass --gllvm-root to a GLLVModels.jl checkout (directory containing Project.toml)",
        )
    isfile(joinpath(root, "Project.toml")) ||
        _track_a_fail(
            "GLLVM root lacks Project.toml: $root",
            "point --gllvm-root at the repository root, not a subdirectory",
        )
    isfile(joinpath(root, TOTORO_323_TRACK_A_PARITY_RUNNER)) ||
        _track_a_fail(
            "missing parity runner at $(TOTORO_323_TRACK_A_PARITY_RUNNER)",
            "use a full GLLVModels.jl tree with test/parity/runparity.jl",
        )
    track_u = uppercase(strip(String(track)))
    track_u == "A" ||
        _track_a_fail(
            "this harness is Track A only (got track=$(repr(track)))",
            "use Track B only via launch pack on Totoro (paste ack Totoro D-139 #323 Track B)",
        )
    stamp = String(receipt_stamp)
    return Totoro323TrackAConfig(root, track_u, stamp)
end

"""
    Totoro323TrackAPreflightReport

Checklist fields populated by [`totoro_323_track_a_preflight!`](@ref).
"""
struct Totoro323TrackAPreflightReport
    gllvm_root::String
    gllvm_head::Union{String,Nothing}
    frozen_gllvmtmb_pin::String
    frozen_contract_path::String
    launch_pack_path::String
    receipt_template::String
    dry_run::Bool
    paste_ready::Bool
end

"""
    totoro_323_track_a_preflight!(cfg::Totoro323TrackAConfig; dry_run=false)
        -> (cfg=Totoro323TrackAConfig, report=Totoro323TrackAPreflightReport)

Validate workspace paths and frozen contract docs. Does not SSH or run parity.
Safe without paste when `dry_run=true`.
"""
function totoro_323_track_a_preflight!(cfg::Totoro323TrackAConfig; dry_run::Bool = false)
    root = cfg.gllvm_root
    contract = joinpath(root, TOTORO_323_FROZEN_CONTRACT_REL)
    isfile(contract) ||
        _track_a_fail(
            "frozen contract missing at $(TOTORO_323_FROZEN_CONTRACT_REL)",
            "run from a GLLVModels.jl checkout that includes the frozen-R 0.7.0 contract",
        )
    launch = joinpath(root, TOTORO_323_LAUNCH_PACK_REL)
    isfile(launch) ||
        _track_a_fail(
            "launch pack missing at $(TOTORO_323_LAUNCH_PACK_REL)",
            "see runbook $(TOTORO_323_RUNBOOK_REL)",
        )
    template = totoro_323_track_a_after_task_receipt_template(root)
    isfile(template) ||
        _track_a_fail(
            "after-task receipt template missing: $(TOTORO_323_AFTER_TASK_RECEIPT_TEMPLATE_REL)",
            "run from a checkout that includes DRAFT #410 harness docs",
        )
    head = _track_a_git_head(root)
    report = Totoro323TrackAPreflightReport(
        root,
        head,
        TOTORO_323_FROZEN_GLLVMTMB_REF,
        contract,
        launch,
        template,
        dry_run,
        totoro_323_track_a_paste_authorized(),
    )
    return (cfg = cfg, report = report)
end

"""
    totoro_323_track_a_preflight_summary(report::Totoro323TrackAPreflightReport) -> String
"""
function totoro_323_track_a_preflight_summary(report::Totoro323TrackAPreflightReport)
    head_line = report.gllvm_head === nothing ?
        "GLLVModels.jl git HEAD: (skipped — not a git checkout or git unavailable)" :
        "GLLVModels.jl git HEAD: $(report.gllvm_head[1:min(end, 12)])…"
    mode = report.dry_run ? "DRY-RUN (no Totoro / no parity)" : "EXECUTE (requires paste)"
    paste_line = report.paste_ready ?
        "paste: GLLVM_TOTORO_PASTE set OK" :
        "paste: not set (export after maintainer ack before live launcher)"
    return """
    Totoro #323 Track A preflight — $(mode)

    $(head_line)
    gllvmTMB pin: $(TOTORO_323_FROZEN_GLLVMTMB_REF_SHORT) ($(TOTORO_323_FROZEN_GLLVMTMB_REF))
    frozen contract: $(report.frozen_contract_path)
    launch pack: $(report.launch_pack_path)
    parity entry: julia --project=test/parity $(TOTORO_323_TRACK_A_PARITY_RUNNER)
    after-task template: $(report.receipt_template)
    $(paste_line)

    Runbook: $(TOTORO_323_RUNBOOK_REL)
    D-139 band (Track A): ~90–150 min wall; single Julia process; OPENBLAS/JULIA threads = 1.
    """
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

    Preflight (no paste, no SSH, no parity):
       julia --project=. tools/totoro323/run_totoro_323_track_a_launcher.jl --dry-run \\
         --gllvm-root $(pwd())

    4. Codex runs launch pack Runner block on Totoro (steps 0–5a).

    Without the paste this launcher exits 2 and does not print an execution plan.
    After Totoro run, fill: $(TOTORO_323_AFTER_TASK_RECEIPT_TEMPLATE_REL)
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
    Totoro #323 Track A — Codex executor plan (paste acknowledged; no SSH from GLLVModels.jl lane)

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

Requires maintainer paste; runs preflight and prints executor plan (no oracle build, no parity run).
"""
function totoro_323_track_a_scaffold_validate!(cfg::Totoro323TrackAConfig)
    totoro_323_track_a_require_paste()
    totoro_323_track_a_preflight!(cfg; dry_run = false)
    print(totoro_323_track_a_executor_plan(cfg))
    return cfg
end
