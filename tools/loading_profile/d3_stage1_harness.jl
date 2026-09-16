"""
    D3 confirmatory `loading_profile` — Stage 1 paste-gated scaffold (DRAFT)

Runbook: `docs/dev-log/plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`.
Stage 0 substrate: PR #345 (`test/test_loading_profile_stage0.jl`, frozen JSON fixtures).

This harness does **not** export `loading_profile`, wire fitters, or rebind the ledger.
After maintainer paste it prints the bounded implementation checklist only.
"""

const D3_LOADING_PROFILE_STAGE1_PASTE_EXACT = "G0 Stage 1"
const D3_STAGE1_RUNBOOK_REL =
    "docs/dev-log/plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md"
const D3_STAGE0_SUBSTRATE_JSON_REL =
    "docs/dev-log/core070/loading-profile-confirmatory-substrate.json"
const D3_STAGE0_TEST_REL = "test/test_loading_profile_stage0.jl"
const D3_STAGE0_PARITY_REL = "test/parity/loading_profile_confirmatory_substrate.jl"
const D3_STAGE1_LEDGER_ROW = "namespace/export/loading_profile"

_stage1_fail(message) =
    throw(ArgumentError("D3 loading_profile Stage 1 harness: " * message))

"""
    d3_loading_profile_stage1_paste_authorized() -> Bool

True only when `ENV["GLLVM_STAGE1_PASTE"]` equals the maintainer paste string exactly.
"""
function d3_loading_profile_stage1_paste_authorized()
    return get(ENV, "GLLVM_STAGE1_PASTE", "") == D3_LOADING_PROFILE_STAGE1_PASTE_EXACT
end

"""
    d3_loading_profile_stage1_require_paste()

Fail closed unless the maintainer paste is present in the environment.
"""
function d3_loading_profile_stage1_require_paste()
    d3_loading_profile_stage1_paste_authorized() ||
        _stage1_fail("refusing Stage 1 scaffold without paste " *
            repr(D3_LOADING_PROFILE_STAGE1_PASTE_EXACT) *
            " in ENV[\"GLLVM_STAGE1_PASTE\"]")
    return nothing
end

"""
    D3LoadingProfileStage1Config

Local workspace paths validated before any Stage 1 engine work (post-paste).
"""
struct D3LoadingProfileStage1Config
    gllvm_root::String
end

function d3_loading_profile_stage1_config(;
    gllvm_root::AbstractString,
)
    root = abspath(String(gllvm_root))
    isdir(root) || _stage1_fail("GLLVM root is not a directory: $root")
    isfile(joinpath(root, "Project.toml")) ||
        _stage1_fail("GLLVM root lacks Project.toml: $root")
    for rel in (
        D3_STAGE0_SUBSTRATE_JSON_REL,
        D3_STAGE0_TEST_REL,
        D3_STAGE0_PARITY_REL,
        D3_STAGE1_RUNBOOK_REL,
    )
        path = joinpath(root, rel)
        isfile(path) || _stage1_fail("missing Stage 0 / runbook artifact: $rel")
    end
    return D3LoadingProfileStage1Config(root)
end

"""
    d3_loading_profile_stage1_scaffold_usage() -> String

Plain-language instructions when the driver is invoked without paste.
"""
function d3_loading_profile_stage1_scaffold_usage()
    return """
    D3 `loading_profile` Stage 1 scaffold (DRAFT — paste-gated, not merge-ready)

    Runbook: $(D3_STAGE1_RUNBOOK_REL)
    Stage 0: $(D3_STAGE0_TEST_REL) + $(D3_STAGE0_SUBSTRATE_JSON_REL)
    Ledger row (still blocked): $(D3_STAGE1_LEDGER_ROW)

    1. Maintainer paste in chat (exact):
       $(D3_LOADING_PROFILE_STAGE1_PASTE_EXACT)
    2. Export for executor (does not authorize public export by itself):
       export GLLVM_STAGE1_PASTE='$(D3_LOADING_PROFILE_STAGE1_PASTE_EXACT)'
    3. Validate workspace (no confirmatory grid spend from this driver):
       julia --project=. tools/loading_profile/run_d3_stage1_scaffold.jl \\
         --gllvm-root $(pwd())
    4. Implement bounded slice in this DRAFT PR (or follow-on) per runbook; Rose receipt after evidence.

    Without the paste this driver exits non-zero and does not print an execution plan.
    """
end

"""
    d3_loading_profile_stage1_executor_plan(cfg::D3LoadingProfileStage1Config) -> String

Checklist text after paste; not executed by this harness.
"""
function d3_loading_profile_stage1_executor_plan(cfg::D3LoadingProfileStage1Config)
    return """
    D3 `loading_profile` Stage 1 — executor plan (paste acknowledged; no public export yet)

    GLLVM_ROOT=$(cfg.gllvm_root)
    Runbook: $(D3_STAGE1_RUNBOOK_REL)

    Bounded slice (maintainer decision set 2026-09-03):
    1. Wire `lambda_constraint` (or equivalent) on Gaussian + ordinary latent fitters using Stage 0 fixtures (MASK-B-PINS / UPPER / ALLFIXED).
    2. Export confirmatory `loading_profile(fit; level, entries, n_grid, grid_extent, conf_level, y)` — not `loading_profile_exploratory`.
    3. Add `test/test_loading_profile_confirmatory.jl` (unit + one R-aligned pin-and-refit grid cell on frozen fixtures).
    4. Rebind ledger row $(D3_STAGE1_LEDGER_ROW) only after paired fixture evidence.
    5. Docs cascade: docstrings, reference, check-log, after-task.

    Fences: no `Project.toml` bump; do not remove deprecation shim in the same PR as first export.
    Heavy grid / Totoro: separate `ack Totoro D-139 #323 Track A` paste.
    """
end

"""
    d3_loading_profile_stage1_scaffold_validate!(cfg::D3LoadingProfileStage1Config)

Requires maintainer paste; prints executor plan only (no fit, no grid, no ledger edit).
"""
function d3_loading_profile_stage1_scaffold_validate!(cfg::D3LoadingProfileStage1Config)
    d3_loading_profile_stage1_require_paste()
    print(d3_loading_profile_stage1_executor_plan(cfg))
    return cfg
end
