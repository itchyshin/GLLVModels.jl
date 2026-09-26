# parity_helpers.jl — shared RCall / gllvmTMB helpers for opt-in parity cells.
#
# Included by family test files under test/parity/. Never included by runtests.jl.
# Twin call shape: gllvmTMB (not CRAN gllvm), latent(..., unique = FALSE),
# extractors via stats::logLik / -opt$objective.
# Source: docs/dev-log/plans/scratch/2026-08-01-gaussian-rcall-shape.md

using RCall
using SHA
using TOML
using LinearAlgebra

include(joinpath(@__DIR__, "parity_trial_inputs.jl"))
include(joinpath(@__DIR__, "core070_receipts.jl"))
using .Core070Receipts
include(joinpath(@__DIR__, "core070_case_registry.jl"))
include(joinpath(@__DIR__, "..", "..", "tools", "core070_second_order", "r_lib.jl"))

const _CORE070_REFERENCE_COMMIT = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const _CORE070_NAMESPACE_SHA256 = "9094613610789faab69c43195d3cfdafb2c7dfef284e6646b10dababa4fa132c"
const _CORE070_SOURCE_TREE_SHA256 = "f83545faa6543dbb1f64d64bbf5a9498adcdf036cc3da5851f269912698b1cc7"
const _CORE070_ARCHIVE_SHA256 = "0c2f4323eb9fb19acccf039b8d57b4dd6bda82e2aa8b4a7bb712f36a64b022bc"
const _CORE070_FAMILY_SMOKE_IDS = Core070CaseRegistry.FAMILY_IDS
const _CORE070_SOURCE = Ref{Dict{String, Any}}()
const _CORE070_RUN = Ref{Any}(nothing)
_core070_required() = get(ENV, "CORE070_PARITY_REQUIRED", "0") == "1"
const _CORE070_ORACLE_BUILD_RECEIPT = ".unlazy/core070-aghq/oracle-receipts/build.json"
const _CORE070_ORACLE_SOURCE_RECEIPT = ".unlazy/core070-aghq/oracle-source/source.json"

const _CORE070_FIXTURES = Core070CaseRegistry.FIXTURES

_core070_root() = normpath(joinpath(@__DIR__, "..", ".."))

function _core070_copy_oracle_receipts!(receipt_dir::AbstractString)
    root = _core070_root()
    for rel in (_CORE070_ORACLE_BUILD_RECEIPT, _CORE070_ORACLE_SOURCE_RECEIPT)
        source = joinpath(root, rel)
        isfile(source) || throw(ArgumentError("required oracle receipt is missing: $rel"))
        cp(source, joinpath(receipt_dir, basename(source)); force = false)
    end
    return nothing
end

core070_requested_case_ids() = Core070CaseRegistry.requested_ids(get(ENV, "CORE070_PARITY_CASE_IDS", ""))

function core070_case_requested(id::AbstractString)
    run = _CORE070_RUN[]
    return run !== nothing && String(id) in run.requested_case_ids
end

function _core070_execution_paths(requested::AbstractVector{<:AbstractString})
    paths = String[
        "src", "test/parity/core070_receipts.jl", "test/parity/core070_case_registry.jl", "test/parity/parity_helpers.jl",
        "test/parity/parity_trial_inputs.jl", "test/parity/test_negbin_parity.jl", "test/parity/truncnb2_policy.jl", "test/parity/nb2_health.jl",
        "test/parity/family_formula_cases.jl", "test/parity/test_truncated_nbinom2_parity.jl", "docs/dev-log/core070/family-formulas-contract.json",
        "test/parity/poisson_beta_health.jl", "test/parity/test_poisson_parity.jl", "test/parity/test_beta_parity.jl", "docs/dev-log/core070/poisson-beta-required-contract.json", "test/parity/runparity.jl", "test/parity/r_health.R",
        "tools/core070_delta_matched.jl", "test/parity/test_delta_lognormal_parity.jl",
        "test/parity/test_delta_gamma_parity.jl", "test/parity/fixtures/core070_gaussian_original.toml",
    "test/parity/covariance_formula_cases.jl", "docs/dev-log/core070/covariance-formula-programme-contract.json", "tools/core070_covariance_mode_fits.jl", "tools/core070_source_fixed_residual_pair.jl", "test/parity/fixtures/core070_covariance_modes.R", "test/parity/fixtures/core070_covariance_fits.R", "docs/dev-log/core070/covariance-programme-contract.json",
    "test/parity/fixtures/core070_gaussian_reference.R", "Project.toml", "test/Project.toml",
        "test/parity/Project.toml", "docs/dev-log/core070/frozen-r070-contract.toml",
    ]
    append!(paths, (_CORE070_FIXTURES[id] for id in requested))
    for manifest in ("Manifest.toml", "test/Manifest.toml", "test/parity/Manifest.toml")
        isfile(joinpath(_core070_root(), manifest)) && push!(paths, manifest)
    end
    return unique(paths)
end

function _core070_receipt_dir()
    raw = get(ENV, "GLLVM_PARITY_RECEIPT_DIR", "")
    isempty(raw) && throw(ArgumentError(
        "required parity evidence needs GLLVM_PARITY_RECEIPT_DIR; no receipt means no pass"))
    dir = abspath(raw)
    mkpath(dir)
    return dir
end

_core070_sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

function _core070_tree_sha256(root::AbstractString; ignore::Function = _ -> false)
    entries = String[]
    for (dir, _, files) in walkdir(root)
        for file in sort(files)
            path = joinpath(dir, file)
            rel = relpath(path, root)
            (islink(path) || ignore(rel)) && continue
            push!(entries, rel * "\0" * _core070_sha256_file(path))
        end
    end
    return bytes2hex(sha256(join(sort(entries), "\n")))
end

function _core070_write_toml(name::AbstractString, receipt::Dict{String, Any})
    path = joinpath(_core070_receipt_dir(), name)
    open(path, "w") do io
        TOML.print(io, receipt)
    end
    return path
end

function _core070_source_pin!()
    Core070Receipts.verify_loaded_source(_core070_root(), Base.pkgdir(GLLVModels), pathof(GLLVModels))
    marker = get(ENV, "GLLVM_PARITY_R_SOURCE_PIN", "")
    isempty(marker) && throw(ArgumentError(
        "required parity evidence needs GLLVM_PARITY_R_SOURCE_PIN inside the installed gllvmTMB library"))
    isfile(marker) || throw(ArgumentError("R source-pin marker is missing: $marker"))
    pin = TOML.parsefile(marker)
    get(pin, "reference_commit", nothing) == _CORE070_REFERENCE_COMMIT ||
        throw(ArgumentError("R source-pin reference_commit does not match frozen CORE-070 commit"))
    get(pin, "namespace_sha256", nothing) == _CORE070_NAMESPACE_SHA256 ||
        throw(ArgumentError("R source-pin namespace_sha256 does not match frozen CORE-070 source"))
    source_tree = get(pin, "source_tree_sha256", "")
    source_tree == _CORE070_SOURCE_TREE_SHA256 ||
        throw(ArgumentError("R source-pin source_tree_sha256 does not match the pinned archive tree"))
    get(pin, "archive_sha256", nothing) == _CORE070_ARCHIVE_SHA256 ||
        throw(ArgumentError("R source-pin archive_sha256 does not match the pinned archive"))
    pkg_root = rcopy(String, R"find.package('gllvmTMB')")
    realpath(marker) == joinpath(realpath(pkg_root), "CORE070_SOURCE_PIN.toml") ||
        throw(ArgumentError("R source-pin marker must be CORE070_SOURCE_PIN.toml in the installed gllvmTMB library"))
    _core070_sha256_file(joinpath(pkg_root, "NAMESPACE")) == _CORE070_NAMESPACE_SHA256 ||
        throw(ArgumentError("installed gllvmTMB NAMESPACE is not the frozen R reference"))
    installed_tree = _core070_tree_sha256(pkg_root; ignore = rel -> rel == "CORE070_SOURCE_PIN.toml")
    get(pin, "installed_tree_sha256", nothing) == installed_tree ||
        throw(ArgumentError("installed gllvmTMB bytes differ from the exact-build source-pin receipt"))
    source = Dict{String, Any}(
        "reference_commit" => _CORE070_REFERENCE_COMMIT,
        "archive_sha256" => _CORE070_ARCHIVE_SHA256,
        "namespace_sha256" => _CORE070_NAMESPACE_SHA256,
        "source_marker_sha256" => _core070_sha256_file(marker),
        "source_tree_sha256" => source_tree,
        "installed_tree_sha256" => installed_tree,
        "oracle_build_receipt_sha256" => _core070_sha256_file(joinpath(_core070_root(), _CORE070_ORACLE_BUILD_RECEIPT)),
        "oracle_source_receipt_sha256" => _core070_sha256_file(joinpath(_core070_root(), _CORE070_ORACLE_SOURCE_RECEIPT)),
        "julia_source_tree_sha256" => _core070_tree_sha256(joinpath(@__DIR__, "..", "..", "src")),
        "julia_version" => string(VERSION),
        "julia_machine" => Sys.MACHINE,
        "julia_package_path" => realpath(pathof(GLLVModels)),
        "julia_package_root" => realpath(Base.pkgdir(GLLVModels)),
        "julia_project_path" => realpath(Base.active_project()),
        "julia_project_sha256" => _core070_sha256_file(Base.active_project()),
        "julia_manifest_sha256" => begin
            manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
            isfile(manifest) ? _core070_sha256_file(manifest) : "ABSENT"
        end,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "rcall_version" => string(Base.pkgversion(RCall)),
        "r_version" => rcopy(String, R"R.version.string"),
        "r_home" => rcopy(String, R"R.home()"),
        "r_library_path" => realpath(pkg_root),
        "tmb_version" => rcopy(String, R"as.character(packageVersion('TMB'))"),
        "matrix_version" => rcopy(String, R"as.character(packageVersion('Matrix'))"),
    )
    _CORE070_SOURCE[] = source
    return source
end

function core070_start_run!()
    _core070_required() || return nothing
    _CORE070_RUN[] === nothing || throw(ArgumentError("CORE-070 run was already started in this Julia process"))
    Core070CaseRegistry.validate_manifest(TOML.parsefile(joinpath(_core070_root(), "docs/dev-log/core070/frozen-r070-contract.toml")))
    requested = core070_requested_case_ids()
    source = _core070_source_pin!()
    root = _core070_root()
    inventory = execution_inventory(root, _core070_execution_paths(requested))
    run = start_run!(_core070_receipt_dir();
        requested_case_ids = requested, family_smoke_case_ids = _CORE070_FAMILY_SMOKE_IDS,
        source = source, inventory = inventory,
        contract_sha256 = _core070_sha256_file(joinpath(root, "docs/dev-log/core070/frozen-r070-contract.toml")))
    try
        _core070_copy_oracle_receipts!(run.dir)
    catch err
        abort_run!(run, err)
        rethrow()
    end
    _CORE070_RUN[] = run
    return nothing
end

function core070_execute_case!(id::AbstractString, fixture::AbstractString, thunk::Function)
    _core070_required() || return thunk()
    core070_case_requested(id) || return nothing
    run = _CORE070_RUN[]
    run === nothing && throw(ArgumentError("CORE-070 run provenance was not verified"))
    testset = @testset "CORE-070 required cell: $id" begin
        thunk()
    end
    counts = testset_counts(testset)
    return record_case!(run, id, fixture;
        passed = counts["passed"], failed = counts["failed"],
        errored = counts["errored"], broken = counts["broken"])
end

function core070_execute_group!(ids::AbstractVector{<:AbstractString}, fixture::AbstractString, thunk::Function)
    _core070_required() || return thunk()
    active = [String(id) for id in ids if core070_case_requested(id)]
    isempty(active) && return nothing
    length(active) == length(ids) || throw(ArgumentError(
        "a shared-fixture CORE-070 group must be requested as one complete scope"))
    run = _CORE070_RUN[]
    run === nothing && throw(ArgumentError("CORE-070 run provenance was not verified"))
    testset = @testset "CORE-070 required fixture group: $(join(active, ", "))" begin
        thunk()
    end
    counts = testset_counts(testset)
    return [record_case!(run, id, fixture;
                         execution_case_ids = active,
                         passed = counts["passed"], failed = counts["failed"],
                         errored = counts["errored"], broken = counts["broken"]) for id in active]
end

function core070_abort_run!(reason)
    _core070_required() || return nothing
    run = _CORE070_RUN[]
    run === nothing || abort_run!(run, reason)
    return nothing
end

function core070_finish_run!()
    _core070_required() || return nothing
    run = _CORE070_RUN[]
    run === nothing && throw(ArgumentError("CORE-070 run provenance was not verified"))
    finish_run!(run)
    return nothing
end

# Prefer the lane twin install when present (gllvmTMB @ origin/main SHA recorded
# in LOOP / after-task). Override with ENV["GLLVM_PARITY_R_LIBS"]; a library
# named there that does not hold gllvmTMB is refused rather than silently
# falling back to R's default library -- see second_order_r_lib() in
# tools/core070_second_order/r_lib.jl (included above), which owns that check.
const _PARITY_TWIN_RLIB_DEFAULT = "/tmp/R-gllvmtmb-x-parity-20260802"

# Returns the twin library path it loaded gllvmTMB from, or `nothing` when no
# twin library was found (unset GLLVM_PARITY_R_LIBS and no historical default).
function _parity_prepend_twin_lib!()
    # Required runs may never fall back to the historical developer library
    # after validating a different oracle at startup.
    twin = if _core070_required()
        marker = get(ENV, "GLLVM_PARITY_R_SOURCE_PIN", "")
        isfile(marker) || throw(ArgumentError("required R source marker is missing"))
        dirname(dirname(realpath(marker)))
    else
        something(second_order_r_lib(), _PARITY_TWIN_RLIB_DEFAULT)
    end
    isdir(joinpath(twin, "gllvmTMB")) || return nothing
    @rput twin
    R"""
    local({
      expected <- normalizePath(file.path(twin, "gllvmTMB"), mustWork = TRUE)
      loaded_from <- function() normalizePath(getNamespaceInfo("gllvmTMB", "path"), mustWork = TRUE)
      if ("gllvmTMB" %in% loadedNamespaces() && loaded_from() != expected) {
        stop("gllvmTMB is already loaded from ", loaded_from(),
             ", not the parity twin library (", twin, "); start a fresh process")
      }
      .libPaths(c(twin, .libPaths()))
      suppressPackageStartupMessages(library(gllvmTMB, lib.loc = twin))
      if (loaded_from() != expected) {
        stop("gllvmTMB loaded from ", loaded_from(), ", not the parity twin library (", twin, ")")
      }
    })
    """
    return twin
end

function _parity_require_gllvmtmb!()
    twin = _parity_prepend_twin_lib!()
    if twin === nothing
        R"""
        if (!requireNamespace("gllvmTMB", quietly = TRUE)) {
            stop("R package 'gllvmTMB' is not installed. ",
                 "Install from the twin checkout or GitHub (itchyshin/gllvmTMB).")
        }
        suppressPackageStartupMessages(library(gllvmTMB))
        invisible(TRUE)
        """
    end
    _core070_required() && _core070_source_pin!()
    return nothing
end

"""
    parity_site_design(x, p) -> Array{Float64,3}

Build `(p, n, 1)` design with shared site covariate: `X[t,s,1] = x[s]`.
"""
function parity_site_design(x::AbstractVector{<:Real}, p::Integer)
    n = length(x)
    X = zeros(Float64, p, n, 1)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = Float64(x[s])
    end
    return X
end

"""
    fit_gllvmtmb_parity_loglik(y, K; family, N=nothing, binomial_link=:logit) -> NamedTuple

Fit `gllvmTMB` on a Julia `p × n` response matrix with the twin-aligned
no-X formula:

```
value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE)
```

`family` ∈ `(:gaussian, :binomial, :poisson, :lognormal, :negbinomial, :beta,
:truncated_poisson)`. Returns `(logLik, objective, converged)`.

For `:binomial`, `N` defaults to ones (Bernoulli); otherwise pass a `p×n`
trial-count matrix. `binomial_link` admits `:logit`, `:probit`, or `:cloglog`.
Both count families forward supplied trials as R weights, preserving site/trait
order. Omitted binomial N retains the original R weights=NULL Bernoulli call.
This complete-data harness requires positive integer trials and integer successes
in `[0,N]`, exactly representable as Float64. Noncount families reject `N`;
non-binomial families reject a nondefault `binomial_link`. These are oracle
fixture constraints, not a claim about all frozen-R missing/data policies.

For `:negbinomial` / `:beta`, R defaults estimate per-trait dispersion; pair
with Julia grouped fitters (`disp_group=:species`), not shared-dispersion defaults.

`:lognormal` (twin fid 3) is the one no-X family where per-trait dispersion would
be WRONG: the twin ties a **shared scalar** `sigma_eps` across traits, so pair it
with `fit_lognormal_gllvm` (scalar `σ`), never a grouped fitter. Its reported
log-likelihood is on the **y scale** and must include the change-of-variables
Jacobian `−Σ log y` on both sides (Identity
`docs/dev-log/decisions/2026-08-15-lognormal-identity.md`).

`:truncated_nbinom2` (twin fid 11) carries **per-trait** dispersion
`log_phi_truncnb2` (`src/gllvmTMB.cpp:1187-1190`), so pair it with
`fit_truncated_nbinom2_gllvm_pertrait`, never the shared-scalar
`fit_truncated_nbinom2_gllvm`. Log link only; support `y ≥ 1`; η on the untruncated
mean. Its Laplace log-det must use `hessian = :observed` to match TMB — the NB2
curvature is y-dependent, unlike fid 10 (Identity
`docs/dev-log/decisions/2026-08-15-truncated-nbinom2-identity.md`).

`:truncated_poisson` (twin fid 10) has no dispersion. η is on the **untruncated**
mean `μ = exp(η)`; the twin's `linkinv` returns the truncated mean
`λ/(1−e^{−λ})` for GLM display only — never compare a mean-scale quantity, only
the log-likelihood (Identity
`docs/dev-log/decisions/2026-08-15-truncated-poisson-identity.md`).
"""
function fit_gllvmtmb_parity_loglik(y::AbstractMatrix, K::Integer; family::Symbol,
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        binomial_link::Symbol = :logit)
    family in (:gaussian, :binomial, :poisson, :lognormal, :gamma, :negbinomial,
               :nb1, :beta, :betabinomial, :truncated_poisson, :truncated_nbinom2) ||
        throw(ArgumentError("unsupported parity family: $family"))
    p, n = size(y)
    trials, binomial_link = parity_trial_inputs(y, family, N, binomial_link)
    trials_provided = N !== nothing
    fam = String(family)
    _parity_require_gllvmtmb!()
    @rput y K p n fam trials binomial_link trials_provided

    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)   # column-major on p×n ⇒ site blocks
    )
    fam_obj <- switch(fam,
        gaussian          = stats::gaussian(),
        binomial          = stats::binomial(link = binomial_link),
        poisson           = stats::poisson(),
        lognormal         = gllvmTMB::lognormal(),
        gamma             = stats::Gamma(link = "log"),
        negbinomial       = gllvmTMB::nbinom2(),
        nb1               = gllvmTMB::nbinom1(),
        beta              = gllvmTMB::Beta(),
        betabinomial      = gllvmTMB::betabinomial(),
        truncated_poisson = gllvmTMB::truncated_poisson(),
        truncated_nbinom2 = gllvmTMB::truncated_nbinom2(),
        stop(sprintf("unknown family: %s", fam))
    )
    # betabinomial/binomial rows: `weights` = per-row trial count (twin API B);
    # NULL for every other family (lme4-style per-observation multiplier).
    weights_vec <- if (identical(fam, "betabinomial") || (identical(fam, "binomial") && trials_provided)) as.vector(trials) else NULL
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        weights = weights_vec,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    .gllvm_parity_last <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L)
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_last$logL"),
        objective = rcopy(Float64, R".gllvm_parity_last$objective"),
        converged = rcopy(Bool, R".gllvm_parity_last$converged"),
    )
end

"""
    fit_gllvmtmb_parity_loglik_x(y, x_site, K; family, N=nothing, binomial_link=:logit) -> NamedTuple

Shared-site-X twin oracle. Trial/link controls match the no-X oracle contract.
`x_site` is length-`n` (one value per site). R formula
uses a **shared** slope `+ x` (not `(0 + trait):x`):

```
value ~ 0 + trait + x + latent(0 + trait | site, d = K, unique = FALSE)
```

`family` ∈ `(:gaussian, :binomial, :poisson, :gamma, :negbinomial, :nb1, :beta,
:ordinal, :betabinomial)`.
Pair with Julia `fit_gaussian_gllvm(; X=)` / `fit_gllvm_cov` (shared γ) for
G/Bin/Pois; `fit_gamma_gllvm_grouped_cov` (per-trait shape α + shared γ) for
Gamma; `fit_nb_gllvm_grouped_cov` / `fit_nb1_gllvm_grouped_cov` /
`fit_beta_gllvm_grouped_cov` / `fit_beta_binomial_gllvm_grouped_cov` (per-trait
dispersion + shared γ) for NB2/NB1/Beta/BetaBinomial; or
`fit_ordinal_gllvm_pertrait_cov` (per-trait cutpoints τ₁=0 / K−2 + shared γ,
`ProbitLink`) for Ordinal — R defaults match twin API B under X (Gamma identity
`2026-08-03-gamma-x-dispersion-identity.md`; NB2/Beta
`2026-08-02-nb2-beta-x-dispersion-identity.md`; NB1
`2026-08-05-nb1-x-dispersion-identity.md`; Ordinal cutpoint identity
`2026-08-03-ordinal-x-cutpoint-identity.md`; BetaBinomial
`2026-08-05-betabinomial-x-dispersion-identity.md`). `:ordinal` uses
`gllvmTMB::ordinal_probit()`; `:nb1` uses `gllvmTMB::nbinom1()`; `:betabinomial`
uses `gllvmTMB::betabinomial()`.

`N` (`p×n` trial counts, required for `:betabinomial`) is threaded to R as the
`weights` argument to `gllvmTMB()` — gllvmTMB's beta-binomial/binomial rows
(fid 8/1) interpret a numeric `weights` vector of length `nrow(data)` as the
per-row trial count (`R/fit-multi.R:2031–2045`), the "API (B)" alternative to
`cbind(successes, failures)` on the LHS. Ignored for all other families.
"""
function fit_gllvmtmb_parity_loglik_x(
    y::AbstractMatrix,
    x_site::AbstractVector{<:Real},
    K::Integer;
    family::Symbol,
    N::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    binomial_link::Symbol = :logit,
)
    family in (:gaussian, :binomial, :poisson, :gamma, :negbinomial, :nb1, :beta, :ordinal, :betabinomial) ||
        throw(ArgumentError("unsupported shared-X parity family: $family"))
    p, n = size(y)
    length(x_site) == n ||
        throw(DimensionMismatch("x_site length ($(length(x_site))) must equal n ($n)"))
    trials, binomial_link = parity_trial_inputs(y, family, N, binomial_link)
    trials_provided = N !== nothing
    fam = String(family)
    x = collect(Float64, x_site)
    _parity_require_gllvmtmb!()
    @rput y K p n fam x trials binomial_link trials_provided

    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y),   # column-major on p×n ⇒ site blocks
        x     = rep(as.numeric(x), each = p)
    )
    fam_obj <- switch(fam,
        gaussian     = stats::gaussian(),
        binomial     = stats::binomial(link = binomial_link),
        poisson      = stats::poisson(),
        gamma        = stats::Gamma(link = "log"),
        negbinomial  = gllvmTMB::nbinom2(),
        nb1          = gllvmTMB::nbinom1(),
        beta         = gllvmTMB::Beta(),
        ordinal      = gllvmTMB::ordinal_probit(),
        betabinomial = gllvmTMB::betabinomial(),
        stop(sprintf("unknown family: %s", fam))
    )
    # betabinomial/binomial rows: `weights` = per-row trial count (API B);
    # NULL for every other family (lme4-style per-observation multiplier).
    weights_vec <- if (identical(fam, "betabinomial") || (identical(fam, "binomial") && trials_provided)) as.vector(trials) else NULL
    # Shared site slope: bare `x`, NOT `(0 + trait):x` (per-trait slopes).
    fit_r <- gllvmTMB(
        value ~ 0 + trait + x + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        weights = weights_vec,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    .gllvm_parity_last_x <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L)
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_last_x$logL"),
        objective = rcopy(Float64, R".gllvm_parity_last_x$objective"),
        converged = rcopy(Bool, R".gllvm_parity_last_x$converged"),
    )
end

"""
    fit_gllvmtmb_parity_loglik_species_x(y, x_site, K; family) -> NamedTuple

Species-specific site-X twin oracle. R formula uses **per-trait** slopes
`(0 + trait):x` (not bare `+ x`):

```
value ~ 0 + trait + (0 + trait):x + latent(0 + trait | site, d = K, unique = FALSE)
```

Pair with Julia [`fit_gllvm_speciescov`](@ref) (`B` is `p×q`). Supported
families: `:poisson` (Arc 0 / #190) and `:binomial` (Bernoulli N=1; capacity
programme S1). Other families may need a separate dispersion identity.
`:binomial` matches the shared-X helper: `stats::binomial()` with no `weights`
(N=1). Do not narrate as a full species-B cohort.
"""
function fit_gllvmtmb_parity_loglik_species_x(
    y::AbstractMatrix,
    x_site::AbstractVector{<:Real},
    K::Integer;
    family::Symbol,
)
    family in (:poisson, :binomial) ||
        throw(ArgumentError("unsupported species-XB parity family: $family"))
    p, n = size(y)
    length(x_site) == n ||
        throw(DimensionMismatch("x_site length ($(length(x_site))) must equal n ($n)"))
    fam = String(family)
    x = collect(Float64, x_site)
    _parity_require_gllvmtmb!()
    @rput y K p n fam x

    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y),
        x     = rep(as.numeric(x), each = p)
    )
    fam_obj <- switch(fam,
        poisson  = stats::poisson(),
        binomial = stats::binomial(),
        stop(sprintf("unknown family: %s", fam))
    )
    # Per-trait slopes: (0 + trait):x — NOT bare shared `x`.
    fit_r <- gllvmTMB(
        value ~ 0 + trait + (0 + trait):x + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    .gllvm_parity_last_species_x <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L)
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_last_species_x$logL"),
        objective = rcopy(Float64, R".gllvm_parity_last_species_x$objective"),
        converged = rcopy(Bool, R".gllvm_parity_last_species_x$converged"),
    )
end

function print_parity_loglik(label::AbstractString; jl_logL, r_logL, r_obj)
    println()
    println("── ", label, " ──")
    println("  Julia logLik          = ", jl_logL)
    println("  gllvmTMB logLik       = ", r_logL)
    println("  gllvmTMB -objective   = ", -r_obj)
    println("  Δ logLik (jl − r)     = ", jl_logL - r_logL)
    println()
    return nothing
end

"""Tiny LT loadings fixture used across parity DGPs."""
function parity_loadings_p5k2()
    return [
        0.8   0.0
        0.5   0.6
        0.3  -0.4
       -0.2   0.5
        0.1   0.3
    ]
end

"""
    fit_gllvmtmb_parity_loglik_multinomial(y, ncat) -> NamedTuple

Twin oracle for **multinomial (twin fid 16)**. Deliberately NOT part of
[`fit_gllvmtmb_parity_loglik`](@ref): every other cell reshapes a numeric `p×n`
matrix and fits `value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE)`.
Multinomial differs on both counts —

* the response is a **single categorical (factor) column** on the formula LHS, not a
  numeric cell value; the twin expands it internally into `K−1` one-hot pseudo-trait
  rows (`R/gllvmTMB.R` `expand_multinomial_response()`), and
* there is **no `latent(...)` term**, because GLLVModels.jl's v1 multinomial is
  fixed-effects softmax only (no LV — `fit_multinomial_gllvm` throws on `K`/`num_lv`).
  The twin supports a no-covstruct multinomial fit, so the FE-only shape is a genuine
  same-model comparison rather than a concession.

`y` is a length-`n` integer vector of category codes `1..ncat` (`ncat ≥ 3`; a
2-category response is binomial and the twin rejects it). Returns
`(logLik, objective, converged)`.

Two footguns are handled here rather than left to callers:

1. **Explicit factor levels.** `factor(y)` sorts levels as *strings*, so with
   `ncat ≥ 10` the baseline would silently permute ("10" sorts before "2"). Levels are
   pinned to `as.character(1:ncat)`.
2. **No `baseline=` argument.** The twin's default reference is the first level, which
   under those pinned levels is category 1 — exactly Julia's `η₁ ≡ 0`. Passing
   `baseline` would risk disagreeing with the Julia convention.
"""
function fit_gllvmtmb_parity_loglik_multinomial(y::AbstractVector{<:Integer},
        ncat::Integer)
    ncat >= 3 || throw(ArgumentError(
        "multinomial parity needs ncat ≥ 3 (a 2-category response is binomial)"))
    all(v -> 1 <= v <= ncat, y) ||
        throw(ArgumentError("y must hold category codes in 1..$ncat"))
    n = length(y)
    yv = collect(Int, y)
    _parity_require_gllvmtmb!()
    @rput yv ncat n

    R"""
    lev <- as.character(seq_len(ncat))
    df_long <- data.frame(
        unit  = factor(seq_len(n)),
        trait = factor(rep("t1", n)),
        value = factor(as.character(yv), levels = lev)
    )
    # No latent(...) term: GLLVModels.jl v1 multinomial is fixed-effects softmax only.
    fit_r <- gllvmTMB(
        value ~ 0 + trait,
        data = df_long,
        unit = "unit",
        trait = "trait",
        family = multinomial(),
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    .gllvm_parity_multinom <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L)
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_multinom$logL"),
        objective = rcopy(Float64, R".gllvm_parity_multinom$objective"),
        converged = rcopy(Bool, R".gllvm_parity_multinom$converged"),
    )
end

"""
    fit_gllvmtmb_parity_delta(y, K; family) -> NamedTuple

Twin oracle for the **delta (hurdle) families**, `:delta_lognormal` (fid 12) /
`:delta_gamma` (fid 13). Unlike [`fit_gllvmtmb_parity_loglik`](@ref), the twin's
`delta_lognormal()` / `delta_gamma()` share ONE linear predictor across occurrence
and the positive part (`gllvmTMB.cpp:2816-2844`), matching Julia's
`predictor = :shared` mode on `fit_delta_lognormal_gllvm` /
`fit_delta_gamma_gllvm` (Identity
`docs/dev-log/decisions/2026-08-28-delta-shared-predictor-identity.md`).

**Dispersion is PER-TRAIT on the twin side** (`log_sigma_lognormal_delta` /
`log_phi_gamma_delta`, `n_traits`-length TMB parameter vectors; see
`R/dispersion-trait-map.R`). Julia's delta fitters now expose `disp_group`, so
the caller must choose `:species` for a same-grouping comparison or explicitly
record a deliberate shared-dispersion mismatch. This helper reports the twin
vector so the selected parameterisation is auditable.

`family` ∈ `(:delta_lognormal, :delta_gamma)`. `y` is `p×n` with `0` for absences.
Returns `(logLik, objective, converged, b_fix, disp_vec)`: `b_fix` is the twin's
trait-intercept vector (length `p`, ONE per trait since the shared predictor has
no separate occurrence/positive intercepts); `disp_vec` is the reported per-trait
`sigma_lognormal_delta` / `phi_gamma_delta` vector (length `p`; for `:delta_gamma`
this is the **CV**, `phi = 1/sqrt(shape)`, NOT the shape — map before comparing to
Julia's `α` = shape via `α ≈ 1/phi^2`).
"""
function fit_gllvmtmb_parity_delta(y::AbstractMatrix, K::Integer; family::Symbol)
    family in (:delta_lognormal, :delta_gamma) ||
        throw(ArgumentError("unsupported delta parity family: $family"))
    p, n = size(y)
    fam = String(family)
    _parity_require_gllvmtmb!()
    @rput y K p n fam

    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)   # column-major on p×n ⇒ site blocks
    )
    fam_obj <- switch(fam,
        delta_lognormal = gllvmTMB::delta_lognormal(),
        delta_gamma     = gllvmTMB::delta_gamma(),
        stop(sprintf("unknown family: %s", fam))
    )
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    pl <- fit_r$tmb_obj$env$parList(fit_r$opt$par)
    disp_vec <- if (identical(fam, "delta_lognormal")) {
        as.numeric(fit_r$report$sigma_lognormal_delta)
    } else {
        as.numeric(fit_r$report$phi_gamma_delta)
    }
    .gllvm_parity_delta <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L),
        b_fix     = as.numeric(pl$b_fix),
        disp_vec  = disp_vec
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_delta$logL"),
        objective = rcopy(Float64, R".gllvm_parity_delta$objective"),
        converged = rcopy(Bool, R".gllvm_parity_delta$converged"),
        b_fix = rcopy(Vector{Float64}, R".gllvm_parity_delta$b_fix"),
        disp_vec = rcopy(Vector{Float64}, R".gllvm_parity_delta$disp_vec"),
    )
end

"""
    fit_gllvmtmb_parity_student(y, K; df_fixed) -> NamedTuple

Twin oracle for the Student-t family (`gllvmTMB::student()`, fid 9,
identity link). `df_fixed` is passed straight to `student(df = df_fixed)`
so BOTH sides hold degrees of freedom fixed at the same value — the twin's
own default is to ESTIMATE df per trait (`student(df = NULL)`). Julia now
also supports estimated `nu` and `disp_group = :species`; this fixed-`df`
helper remains useful for the original target, while the estimated route must
be compared under its own declared parameter map.

Returns `(logLik, objective, converged, optimizer_code, optimizer_message,
optimizer_iterations, b_fix, sigma_vec, df_vec)`. `b_fix` is the twin's
trait-intercept vector (length `p`); `sigma_vec` /`df_vec` are the reported
per-trait `sigma_student` / `df_student` vectors (length `p`). Per the
parameterisation note, `df_vec` should equal `df_fixed` on every trait
(fixed, not estimated) — assert that in the caller before trusting a logLik
Δ as dispersion-only.
"""
function fit_gllvmtmb_parity_student(y::AbstractMatrix, K::Integer; df_fixed::Union{Nothing, Real} = nothing)
    df_fixed !== nothing && (df_fixed > 1 || throw(ArgumentError("student(): df_fixed must be > 1; got $df_fixed")))
    p, n = size(y)
    _parity_require_gllvmtmb!()
    dfv = df_fixed === nothing ? nothing : Float64(df_fixed)
    @rput y K p n dfv

    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)   # column-major on p×n ⇒ site blocks
    )
    fam_obj <- if (is.null(dfv)) {
        gllvmTMB::student(link = "identity")
    } else {
        gllvmTMB::student(link = "identity", df = dfv)
    }
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    pl <- fit_r$tmb_obj$env$parList(fit_r$opt$par)
    .gllvm_parity_student <<- list(
        logL      = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L),
        optimizer_code = as.integer(fit_r$opt$convergence),
        optimizer_message = as.character(if (is.null(fit_r$opt$message)) "" else fit_r$opt$message),
        optimizer_iterations = as.integer(if (is.null(fit_r$opt$iterations)) NA_integer_ else fit_r$opt$iterations),
        b_fix     = as.numeric(pl$b_fix),
        sigma_vec = as.numeric(fit_r$report$sigma_student),
        df_vec    = as.numeric(fit_r$report$df_student)
    )
    invisible(NULL)
    """

    return (
        logLik = rcopy(Float64, R".gllvm_parity_student$logL"),
        objective = rcopy(Float64, R".gllvm_parity_student$objective"),
        converged = rcopy(Bool, R".gllvm_parity_student$converged"),
        optimizer_code = rcopy(Int, R".gllvm_parity_student$optimizer_code"),
        optimizer_message = rcopy(String, R".gllvm_parity_student$optimizer_message"),
        optimizer_iterations = rcopy(Int, R".gllvm_parity_student$optimizer_iterations"),
        b_fix = rcopy(Vector{Float64}, R".gllvm_parity_student$b_fix"),
        sigma_vec = rcopy(Vector{Float64}, R".gllvm_parity_student$sigma_vec"),
        df_vec = rcopy(Vector{Float64}, R".gllvm_parity_student$df_vec"),
    )
end

"""
    fit_gllvmtmb_parity_tweedie(y, K; p_fixed, power_group=:species) -> NamedTuple

Twin oracle for the Tweedie family (`gllvmTMB::tweedie()`, fid 6,
log link). `p_fixed` is passed to `tweedie(link = "log", p = p_fixed)`
if specified. With the default `power_group = :species`, an unpinned power is
estimated per trait (`p_tweedie`), exactly as the public frozen-R call does.

`power_group = :shared` is a **reference-engine constraint adapter**, not a
public `gllvmTMB()` argument. It first fits the ordinary frozen-R model, then
rebuilds that exact fit's retained `tmb_data`, parameter list, map, and
`random = "z_B"` declaration with a single `logit_p_tweedie` map level. It
refuses mixed-family, AGHQ, non-ML, or non-`z_B` fits; preserves every other
map; checks that exactly one power coordinate remains; and checks the report
has one equal power for every trait. This is the only admissible shared-power
oracle until R exposes a public control. `p_fixed` and `power_group = :shared`
are intentionally mutually exclusive.

Returns `(logLik, objective, converged, b_fix, phi_vec, p_vec)`.
"""
function fit_gllvmtmb_parity_tweedie(y::AbstractMatrix, K::Integer;
        p_fixed::Union{Nothing, Real} = nothing, power_group::Symbol = :species)
    p_fixed !== nothing && (1.0 < p_fixed < 2.0 || throw(ArgumentError("tweedie(): p_fixed must be in (1, 2); got $p_fixed")))
    power_group in (:species, :shared) || throw(ArgumentError(
        "tweedie power_group must be :species or :shared; got :$power_group"))
    p_fixed !== nothing && power_group === :shared && throw(ArgumentError(
        "p_fixed already defines the Tweedie power; power_group=:shared is only for estimated power"))
    p, n = size(y)
    _parity_require_gllvmtmb!()
    pv = p_fixed === nothing ? nothing : Float64(p_fixed)
    pg = String(power_group)
    health_source = joinpath(@__DIR__, "r_health.R")
    @rput y K p n pv pg health_source

    R"""
    source(health_source, local = FALSE)
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)   # column-major on p×n ⇒ site blocks
    )
    fam_obj <- if (is.null(pv)) {
        gllvmTMB::tweedie(link = "log")
    } else {
        gllvmTMB::tweedie(link = "log", p = pv)
    }
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = fam_obj,
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    adapter <- FALSE
    if (identical(pg, "shared")) {
      if (!is.null(pv) || !identical(fit_r$estimator, "ML") ||
          isTRUE(fit_r$aghq$used) || !identical(fit_r$random, "z_B") ||
          !all(fit_r$tmb_data$family_id == 6L)) {
        stop("shared Tweedie power adapter requires an ordinary ML, all-Tweedie, z_B-only Laplace fit", call. = FALSE)
      }
      ## Rebuild only the power map.  Data, all non-power maps, the compiled
      ## template, and the Laplace random block come from the fitted frozen R
      ## object; no R engine source is changed or reimplemented here.
      shared_map <- fit_r$tmb_map
      shared_params <- fit_r$tmb_params
      fitted_params <- fit_r$tmb_obj$env$parList(fit_r$opt$par)
      for (nm in intersect(names(shared_params), names(fitted_params))) {
        shared_params[[nm]] <- fitted_params[[nm]]
      }
      length(shared_params$logit_p_tweedie) == p ||
        stop("shared Tweedie adapter expected one power entry per trait", call. = FALSE)
      shared_params$logit_p_tweedie[] <- mean(fitted_params$logit_p_tweedie)
      shared_map$logit_p_tweedie <- factor(rep(1L, p))
      obj_shared <- TMB::MakeADFun(
        data = fit_r$tmb_data, parameters = shared_params, map = shared_map,
        random = fit_r$random, DLL = "gllvmTMB", silent = TRUE
      )
      sum(grepl("^logit_p_tweedie", names(obj_shared$par))) == 1L ||
        stop("shared Tweedie adapter did not produce exactly one free power coordinate", call. = FALSE)
      opt_shared <- nlminb(start = obj_shared$par, objective = obj_shared$fn,
                           gradient = obj_shared$gr)
      fit_r$opt <- opt_shared
      fit_r$tmb_obj <- obj_shared
      fit_r$tmb_params <- shared_params
      fit_r$tmb_map <- shared_map
      ## report() consumes the full vector (including latent modes), whereas
      ## nlminb returns only the free outer vector. Re-evaluate fn at the
      ## selected optimum to refresh the conditional modes, then use the
      ## object's full best vector; passing opt_shared$par is a length error.
      obj_shared$fn(opt_shared$par)
      fit_r$report <- obj_shared$report(obj_shared$env$last.par.best)
      adapter <- TRUE
    }
    pl <- fit_r$tmb_obj$env$parList(fit_r$opt$par)
    p_report <- as.numeric(fit_r$report$p_tweedie)
    if (isTRUE(adapter) && (!all(is.finite(p_report)) ||
        length(unique(round(p_report, 12))) != 1L)) {
      stop("shared Tweedie adapter report does not carry one common power", call. = FALSE)
    }
    gradient_error <- ""
    gradient <- tryCatch(as.numeric(fit_r$tmb_obj$gr(fit_r$opt$par)),
      error = function(e) {
        gradient_error <<- conditionMessage(e)
        rep(NA_real_, length(fit_r$opt$par))
      })
    health <- core070_tweedie_health(fit_r$opt, gradient,
      as.numeric(fit_r$report$phi_tweedie), p_report,
      hessian_pd = core070_hessian_pd(fit_r))
    health$gradient_error <- gradient_error
    .gllvm_parity_tweedie <<- list(
        health = health,
        ## The constraint adapter changes the objective. The public fit's
        ## cached objective_components belongs to the pre-adapter fit and
        ## must not be reused as logLik evidence for the tied-power model.
        logL      = if (adapter) -as.numeric(fit_r$opt$objective) else as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L),
        b_fix     = as.numeric(pl$b_fix),
        phi_vec   = as.numeric(fit_r$report$phi_tweedie),
        p_vec     = p_report,
        power_group = pg,
        reference_constraint_adapter = adapter
    )
    invisible(NULL)
    """

    return (
        health = (
            passed = rcopy(Bool, R".gllvm_parity_tweedie$health$healthy"),
            optimizer_code = rcopy(Int, R".gllvm_parity_tweedie$health$optimizer_code"),
            optimizer_message = rcopy(String, R"paste(.gllvm_parity_tweedie$health$optimizer_message, collapse='; ')"),
            gradient_max_scaled = rcopy(Float64, R".gllvm_parity_tweedie$health$gradient_max_scaled"),
            finite_parameters = rcopy(Bool, R".gllvm_parity_tweedie$health$finite_parameters"),
            finite_report = rcopy(Bool, R".gllvm_parity_tweedie$health$finite_report"),
            n_free = rcopy(Int, R".gllvm_parity_tweedie$health$n_free"),
            n_power_free = rcopy(Int, R".gllvm_parity_tweedie$health$n_power_free"),
            hessian_diagnostic = rcopy(String, R".gllvm_parity_tweedie$health$hessian_diagnostic"),
            gradient_error = rcopy(String, R".gllvm_parity_tweedie$health$gradient_error"),
        ),
        logLik = rcopy(Float64, R".gllvm_parity_tweedie$logL"),
        objective = rcopy(Float64, R".gllvm_parity_tweedie$objective"),
        converged = rcopy(Bool, R".gllvm_parity_tweedie$converged"),
        b_fix = rcopy(Vector{Float64}, R".gllvm_parity_tweedie$b_fix"),
        phi_vec = rcopy(Vector{Float64}, R".gllvm_parity_tweedie$phi_vec"),
        p_vec = rcopy(Vector{Float64}, R".gllvm_parity_tweedie$p_vec"),
        power_group = rcopy(String, R".gllvm_parity_tweedie$power_group"),
        reference_constraint_adapter = rcopy(Bool, R".gllvm_parity_tweedie$reference_constraint_adapter"),
    )
end

# ---------------------------------------------------------------------------
# Cross-objective identity delta (panel 2026-09-01 generalization), usable by
# parity fixtures. This only adds the shared evaluator; wiring it into any
# individual fixture (Gaussian-sources, Binomial, Poisson, ...) is a separate,
# reviewed slice and is deliberately NOT done here.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "tools", "core070_cross_objective.jl"))

"""
    core070_cross_objective_delta(family_kind, Y, r; rank,
        beta_key = "beta", crossprod_key = "crossprod",
        dispersion_key = nothing, loglik_key = "loglik",
        N = nothing, link = nothing, mask = nothing, offset = nothing,
        source = nothing) -> Float64

Cross-objective identity delta for one retained-coordinate route `r` (e.g. one
table of a `TOML.parsefile` fixture, keyed like
`test/fixtures/core070_latent_bare_retained.toml`'s `[native_julia]` /
`[r_reference]` tables): evaluate `family_kind`'s objective (via
`cross_objective_at`, defined in `tools/core070_cross_objective.jl`) at `r`'s
serialized fitted coordinates and return
`abs(loglik_at_coordinates - r[loglik_key])`. A near-zero delta (<= 1e-8, the
known-answer gate's tolerance — see `test/test_cross_objective_known_answer.jl`)
is the likelihood-function identity claim itself.
"""
function core070_cross_objective_delta(family_kind::Symbol, Y::AbstractMatrix, r::AbstractDict;
        rank::Integer, beta_key::AbstractString = "beta",
        crossprod_key::AbstractString = "crossprod",
        dispersion_key::Union{Nothing, AbstractString} = nothing,
        loglik_key::AbstractString = "loglik",
        N = nothing, link = nothing, mask = nothing, offset = nothing,
        source = nothing)
    beta = Vector{Float64}(r[beta_key])
    crossprod = Matrix{Float64}(reduce(hcat,
        [Vector{Float64}(c) for c in r[crossprod_key]]))
    dispersion = dispersion_key === nothing ? nothing : Float64(r[dispersion_key])
    ll = cross_objective_at(family_kind, Y; beta = beta,
        crossprod_or_loadings = crossprod, rank = rank, dispersion = dispersion,
        N = N, link = link, mask = mask, offset = offset, source = source)
    return abs(ll - Float64(r[loglik_key]))
end
