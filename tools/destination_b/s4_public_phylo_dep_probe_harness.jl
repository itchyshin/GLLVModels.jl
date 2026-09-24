"""
    S4 public `phylo_dep` isolated probe — Julia-side harness (paste-gated)

Runbook: `docs/dev-log/plans/2026-09-16-s4-probe-julia-checklist-paste-gated.md`.

Twin recorder (read-only, no gllvmTMB `src/` edits from this repo):
gllvmTMB [PR #1283](https://github.com/itchyshin/gllvmTMB/pull/1283), commit
`97214679c94cc4a6b9e02d3c2b03ccce516027d8` on branch
`codex/destination-b-s4-phylo-dep-formula-20260910`.

Execution requires maintainer paste **`S4 probe yes`** in
`ENV["GLLVM_S4_PROBE_PASTE"]`. This harness does not substitute for that paste.
"""

const S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT =
    "97214679c94cc4a6b9e02d3c2b03ccce516027d8"
const S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT = "97214679c"
const S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH =
    "codex/destination-b-s4-phylo-dep-formula-20260910"
const S4_PUBLIC_PHYLO_DEP_RECORDER_PR = 1283
const S4_PUBLIC_PHYLO_DEP_RUNNER_REL =
    "tests/testthat/run-destination-b-s4-public-phylo-dep-isolated.R"
const S4_PUBLIC_PHYLO_DEP_PASTE_EXACT = "S4 probe yes"
const S4_FROZEN_ORACLE_PIN =
    "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const S4_AFTER_TASK_RECEIPT_TEMPLATE_REL =
    "docs/dev-log/after-task/TEMPLATE-s4-public-phylo-dep-probe-receipt.md"
const S4_PUBLIC_PHYLO_DEP_PROBE_ENV_REL =
    "tools/destination_b/probe_env"

"""
    s4_public_phylo_dep_probe_env(julia_project::AbstractString) -> String

Default `GLLVM_S4_JULIA_ENV`: the committed probe-only environment that
`develop`s `julia_project` and adds `LogExpFunctions` as a direct dependency
(the recorder's clean-Julia-probe step does `using LogExpFunctions` before
`using GLLVM`; that fails against the main `Project.toml`, which does not list
`LogExpFunctions` directly). Callers may still override via `--julia-env` /
the `julia_env` keyword.
"""
function s4_public_phylo_dep_probe_env(julia_project::AbstractString)
    return joinpath(julia_project, S4_PUBLIC_PHYLO_DEP_PROBE_ENV_REL)
end

function _s4_probe_fail(message::AbstractString, hint::AbstractString = "")
    body = "S4 public phylo_dep probe harness: " * message
    if !isempty(hint)
        body *= "\n  → " * hint
    end
    return throw(ArgumentError(body))
end

"""
    s4_public_phylo_dep_paste_authorized() -> Bool

True only when `ENV["GLLVM_S4_PROBE_PASTE"]` equals the maintainer paste string
exactly (`S4 probe yes`).
"""
function s4_public_phylo_dep_paste_authorized()
    return get(ENV, "GLLVM_S4_PROBE_PASTE", "") == S4_PUBLIC_PHYLO_DEP_PASTE_EXACT
end

"""
    s4_public_phylo_dep_require_paste()

Fail closed unless the maintainer paste is present in the environment.
"""
function s4_public_phylo_dep_require_paste()
    if s4_public_phylo_dep_paste_authorized()
        return nothing
    end
    got = get(ENV, "GLLVM_S4_PROBE_PASTE", "")
    if isempty(got)
        _s4_probe_fail(
            "refusing probe execution: ENV[\"GLLVM_S4_PROBE_PASTE\"] is unset",
            "after maintainer paste, export GLLVM_S4_PROBE_PASTE='$(S4_PUBLIC_PHYLO_DEP_PASTE_EXACT)'",
        )
    else
        _s4_probe_fail(
            "refusing probe execution: paste mismatch (got $(repr(got)))",
            "exact paste required: $(repr(S4_PUBLIC_PHYLO_DEP_PASTE_EXACT))",
        )
    end
end

function _s4_git_read(cmd::Cmd)
    try
        return strip(read(cmd, String))
    catch err
        _s4_probe_fail("git command failed ($(cmd)): $(err)")
    end
end

"""
    s4_public_phylo_dep_verify_recorder_tip(gllvmtmb_root::AbstractString)

Confirm `gllvmtmb_root` is a git checkout whose `HEAD` matches the pinned
recorder commit (full or short prefix).
"""
function s4_public_phylo_dep_verify_recorder_tip(gllvmtmb_root::AbstractString)
    root = abspath(gllvmtmb_root)
    isdir(root) ||
        _s4_probe_fail(
            "gllvmTMB root is not a directory: $root",
            "check out gllvmTMB at $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT) " *
            "(branch $(S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH); PR #$(S4_PUBLIC_PHYLO_DEP_RECORDER_PR))",
        )
    git_dir = joinpath(root, ".git")
    isdir(git_dir) || isfile(git_dir) ||
        _s4_probe_fail(
            "gllvmTMB root is not a git checkout: $root",
            "use a full clone/worktree, not a source tarball",
        )
    head = _s4_git_read(Cmd(["git", "-C", root, "rev-parse", "HEAD"]))
    if !(startswith(head, S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT) ||
         startswith(head, S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT))
        _s4_probe_fail(
            "gllvmTMB HEAD $(head) is not recorder $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT)",
            "git -C \"$root\" fetch origin $(S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH) && " *
            "git -C \"$root\" checkout $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT)",
        )
    end
    runner = joinpath(root, S4_PUBLIC_PHYLO_DEP_RUNNER_REL)
    isfile(runner) ||
        _s4_probe_fail(
            "recorder runner missing at $(S4_PUBLIC_PHYLO_DEP_RUNNER_REL)",
            "confirm checkout matches gllvmTMB PR #$(S4_PUBLIC_PHYLO_DEP_RECORDER_PR)",
        )
    return (root = root, head = head, runner = runner)
end

"""
    s4_public_phylo_dep_recorder_remote_tip() -> Union{String,Nothing}

Best-effort `git ls-remote` for the pinned recorder branch. Returns tip SHA or
`nothing` if network/git remote is unavailable (dry-run still proceeds).
"""
function s4_public_phylo_dep_recorder_remote_tip()
    try
        out = strip(read(
            Cmd([
                "git",
                "ls-remote",
                "origin",
                "refs/heads/$(S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH)",
            ]),
            String,
        ))
        isempty(out) && return nothing
        sha = first(split(out))
        return strip(sha)
    catch
        return nothing
    end
end

"""
    s4_public_phylo_dep_after_task_receipt_template(gllvm_root::AbstractString) -> String

Absolute path to the markdown receipt template (for post-probe after-task).
"""
function s4_public_phylo_dep_after_task_receipt_template(gllvm_root::AbstractString)
    return abspath(joinpath(gllvm_root, S4_AFTER_TASK_RECEIPT_TEMPLATE_REL))
end

"""
    S4PublicPhyloDepProbeConfig

Paths and executables for one isolated probe invocation. Does not run R.
"""
struct S4PublicPhyloDepProbeConfig
    gllvmtmb_root::String
    julia_project::String
    julia_executable::String
    julia_env::String
    receipt_path::String
    rscript_executable::String
end

function _s4_require_executable(path::AbstractString, label::AbstractString)
    p = abspath(String(path))
    isfile(p) || _s4_probe_fail("$label is not a file: $p")
    return p
end

function _s4_require_project(path::AbstractString)
    p = abspath(String(path))
    isfile(joinpath(p, "Project.toml")) ||
        _s4_probe_fail("Julia project lacks Project.toml: $p")
    return p
end

"""
    s4_public_phylo_dep_probe_config(;
        gllvmtmb_root,
        julia_project,
        julia_executable,
        julia_env = s4_public_phylo_dep_probe_env(julia_project),
        receipt_path,
        rscript_executable = "Rscript",
    ) -> S4PublicPhyloDepProbeConfig

Validate configuration for the gllvmTMB isolated runner. Caller must still
pass [`s4_public_phylo_dep_require_paste`](@ref) before [`s4_public_phylo_dep_run!`](@ref).
"""
function s4_public_phylo_dep_probe_config(;
    gllvmtmb_root::AbstractString,
    julia_project::AbstractString,
    julia_executable::AbstractString,
    julia_env::AbstractString = s4_public_phylo_dep_probe_env(julia_project),
    receipt_path::AbstractString,
    rscript_executable::AbstractString = "Rscript",
    dry_run::Bool = false,
)
    verified = s4_public_phylo_dep_verify_recorder_tip(gllvmtmb_root)
    receipt = abspath(String(receipt_path))
    parent = dirname(receipt)
    if !isdir(parent)
        try
            mkpath(parent)
        catch err
            _s4_probe_fail(
                "receipt parent is not creatable: $parent ($err)",
                "choose --receipt under a writable directory",
            )
        end
    end
    isdir(parent) || _s4_probe_fail("receipt parent is not a directory: $parent")
    if ispath(receipt) && !dry_run
        _s4_probe_fail(
            "refusing to overwrite existing receipt: $receipt",
            "pick a fresh path or archive the prior receipt first",
        )
    end
    julia_proj = _s4_require_project(julia_project)
    julia_env_path = _s4_require_project(julia_env)
    julia_exe = _s4_require_executable(julia_executable, "julia_executable")
    rscript = if dry_run && !isfile(abspath(String(rscript_executable)))
        # Dry-run may run on hosts without R; probe execution still requires Rscript.
        abspath(String(rscript_executable))
    else
        _s4_require_executable(rscript_executable, "rscript_executable")
    end
    return S4PublicPhyloDepProbeConfig(
        verified.root,
        julia_proj,
        julia_exe,
        julia_env_path,
        receipt,
        rscript,
    )
end

"""
    S4PublicPhyloDepPreflightReport

Checklist fields populated by [`s4_public_phylo_dep_preflight!`](@ref).
"""
struct S4PublicPhyloDepPreflightReport
    gllvmtmb_head::String
    recorder_remote_tip::Union{String,Nothing}
    frozen_oracle_pin::String
    receipt_template::String
    julia_project::String
    receipt_path::String
    dry_run::Bool
end

"""
    s4_public_phylo_dep_preflight!(; kwargs..., dry_run=false, gllvm_root=julia_project)
        -> (cfg=S4PublicPhyloDepProbeConfig, report=S4PublicPhyloDepPreflightReport)

Validate paths and recorder pin. Does not invoke R or RCall. Safe without paste
when `dry_run=true`.
"""
function s4_public_phylo_dep_preflight!(;
    gllvmtmb_root::AbstractString,
    julia_project::AbstractString,
    julia_executable::AbstractString,
    julia_env::AbstractString = s4_public_phylo_dep_probe_env(julia_project),
    receipt_path::AbstractString,
    rscript_executable::AbstractString = "Rscript",
    dry_run::Bool = false,
    gllvm_root::AbstractString = julia_project,
)
    cfg = s4_public_phylo_dep_probe_config(;
        gllvmtmb_root,
        julia_project,
        julia_executable,
        julia_env,
        receipt_path,
        rscript_executable,
        dry_run,
    )
    remote = s4_public_phylo_dep_recorder_remote_tip()
    if remote !== nothing &&
       !(startswith(remote, S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT) ||
         startswith(remote, S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT))
        _s4_probe_fail(
            "origin/$(S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH) tip $(remote) " *
            "drifted from pinned recorder $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT)",
            "fetch/recheck gllvmTMB PR #$(S4_PUBLIC_PHYLO_DEP_RECORDER_PR) before probing",
        )
    end
    template = s4_public_phylo_dep_after_task_receipt_template(gllvm_root)
    isfile(template) ||
        _s4_probe_fail(
            "after-task receipt template missing: $(S4_AFTER_TASK_RECEIPT_TEMPLATE_REL)",
            "run from a GLLVModels.jl checkout that includes DRAFT #409 harness docs",
        )
    head = _s4_git_read(Cmd(["git", "-C", cfg.gllvmtmb_root, "rev-parse", "HEAD"]))
    report = S4PublicPhyloDepPreflightReport(
        head,
        remote,
        S4_FROZEN_ORACLE_PIN,
        template,
        cfg.julia_project,
        cfg.receipt_path,
        dry_run,
    )
    return (cfg = cfg, report = report)
end

"""
    s4_public_phylo_dep_preflight_summary(report::S4PublicPhyloDepPreflightReport) -> String
"""
function s4_public_phylo_dep_preflight_summary(report::S4PublicPhyloDepPreflightReport)
    remote_line = report.recorder_remote_tip === nothing ?
        "recorder remote tip: (skipped — git ls-remote unavailable)" :
        "recorder remote tip: $(report.recorder_remote_tip[1:min(end, 12)])… OK"
    mode = report.dry_run ? "DRY-RUN (no Rscript probe)" : "EXECUTE (requires paste)"
    return """
    S4 public phylo_dep preflight — $(mode)

    gllvmTMB HEAD: $(report.gllvmtmb_head[1:min(end, 12)])… (pinned $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT))
    $(remote_line)
    frozen oracle pin (manual): $(report.frozen_oracle_pin)
    Julia project: $(report.julia_project)
    receipt path: $(report.receipt_path)
    after-task template: $(report.receipt_template)

    Runbook: docs/dev-log/plans/2026-09-16-s4-probe-julia-checklist-paste-gated.md
    """
end

"""
    s4_public_phylo_dep_r_environment(cfg::S4PublicPhyloDepProbeConfig) -> Dict{String,String}

Environment variables expected by gllvmTMB
`run-destination-b-s4-public-phylo-dep-isolated.R` at the recorder commit.

`GLLVM_S4_JULIA_HOME` is the *directory* containing the `julia` executable, not
the executable file itself: the recorder's `s4_public_phylo_dep_clean_julia_probe()`
resolves it as `file.path(normalizePath(julia_home, mustWork = TRUE), "julia")`
(falling back to `.../bin/julia`), so passing the file itself makes that
`file.path()` call append a second `julia` path segment onto a file and always
miss. `cfg.julia_executable` is still validated as a file by
[`s4_public_phylo_dep_probe_config`](@ref); only the value handed to the R
child process changes.
"""
function s4_public_phylo_dep_r_environment(cfg::S4PublicPhyloDepProbeConfig)
    return Dict{String,String}(
        "GLLVM_S4_LIVE_FORMULA_TESTS" => "1",
        "GLLVM_DESTINATION_B_PROJECT" => cfg.julia_project,
        "GLLVM_S4_JULIA_HOME" => dirname(cfg.julia_executable),
        "GLLVM_S4_JULIA_ENV" => cfg.julia_env,
        "GLLVM_S4_RECEIPT_PATH" => cfg.receipt_path,
    )
end

"""
    s4_public_phylo_dep_r_command(cfg::S4PublicPhyloDepProbeConfig) -> Cmd

The `Rscript --vanilla <runner>` command, run from the gllvmTMB root with the
recorder's environment. Does not run it. The working directory is set through
`setenv(cmd, env; dir=)` because `Cmd(::Vector; dir=)` has no method on
Julia 1.10.
"""
function s4_public_phylo_dep_r_command(cfg::S4PublicPhyloDepProbeConfig)
    runner = joinpath(cfg.gllvmtmb_root, S4_PUBLIC_PHYLO_DEP_RUNNER_REL)
    env = merge(Dict{String,String}(ENV), s4_public_phylo_dep_r_environment(cfg))
    return setenv(Cmd([cfg.rscript_executable, "--vanilla", runner]), env;
        dir = cfg.gllvmtmb_root)
end

"""
    s4_public_phylo_dep_run!(cfg::S4PublicPhyloDepProbeConfig)

Invoke the gllvmTMB isolated R runner (read-only w.r.t. gllvmTMB source from
this repo). Requires maintainer paste; overwrites nothing if receipt exists.
"""
function s4_public_phylo_dep_run!(cfg::S4PublicPhyloDepProbeConfig)
    s4_public_phylo_dep_require_paste()
    return run(s4_public_phylo_dep_r_command(cfg))
end

"""
    s4_public_phylo_dep_scaffold_usage() -> String

Plain-language instructions printed when the driver is invoked without paste.
"""
function s4_public_phylo_dep_scaffold_usage()
    return """
    S4 public phylo_dep probe harness (DRAFT — paste-gated, not merge-ready)

    Recorder: gllvmTMB PR $(S4_PUBLIC_PHYLO_DEP_RECORDER_PR) @ $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT)
    Runbook: docs/dev-log/plans/2026-09-16-s4-probe-julia-checklist-paste-gated.md

    1. Check out gllvmTMB at $(S4_PUBLIC_PHYLO_DEP_RECORDER_COMMIT_SHORT) (branch $(S4_PUBLIC_PHYLO_DEP_RECORDER_BRANCH)).
    2. After maintainer paste, export:
       export GLLVM_S4_PROBE_PASTE='$(S4_PUBLIC_PHYLO_DEP_PASTE_EXACT)'
    3. Run (single Julia process):
       julia --project=. tools/destination_b/run_s4_public_phylo_dep_probe.jl \\
         --gllvmtmb-root /path/to/gllvmTMB \\
         --julia-project $(pwd()) \\
         --julia /path/to/julia \\
         --receipt /tmp/s4-public-phylo-dep-receipt.json

    Preflight (no paste, no R):
       julia --project=. tools/destination_b/run_s4_public_phylo_dep_probe.jl --dry-run \\
         --gllvmtmb-root /path/to/gllvmTMB \\
         --julia-project $(pwd()) \\
         --julia /path/to/julia \\
         --receipt /tmp/s4-public-phylo-dep-receipt.json

    Without the paste the driver exits 2 and does not call R.
    After probe, fill: docs/dev-log/after-task/TEMPLATE-s4-public-phylo-dep-probe-receipt.md
    """
end
