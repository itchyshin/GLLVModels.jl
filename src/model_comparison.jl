# Nested-model comparison — a twin of gllvmTMB's `anova.gllvmTMB_multi()` /
# `print.anova.gllvmTMB_multi()` S3 methods (R/aghq-report.R, pin
# 9539352f66f2db2cc26b1c393e67212a359b60c9, "P1"). `AIC.gllvmTMB_multi` and
# `BIC.gllvmTMB_multi` are NOT re-implemented here: they wrap `stats::AIC`/
# `stats::BIC` with `k = 2` and `attr(logLik(fit), "df")`/`nobs(fit)`, and
# GLLVModels.jl's own `aic`/`bic`/`dof`/`nobs` (src/postfit.jl) already use
# the identical conventions once the SAME model is compared:
# `dof(fit)` matches R's `length(opt$par)` exactly (both count loadings
# modulo the K(K−1)/2 rotational df) and `nobs(fit, Y)` is documented as R's
# p·n observed-cell count, not the number of sites (src/postfit.jl:604-620).
#
# One care point verified while building the twin test
# (test/test_model_comparison.jl): `fit_gaussian_gllvm(Y; K)` with no `X` is
# a ZERO-MEAN factor model (src/likelihood.jl: `X === nothing && β ===
# nothing` sets `resid = y` directly, no per-species intercept of any kind),
# whereas gllvmTMB's `y ~ 0 + trait + latent(...)` fixture model has an
# explicit non-zero per-trait fixed intercept. These are genuinely different
# models, so directly comparing their dof/AIC/BIC is not a convention check
# at all — the ~p-parameter dof gap it produces is a modelling difference,
# not a df-counting bug. The apples-to-apples GLLVModels.jl twin of R's
# fixture model supplies the per-species intercepts explicitly via a (p, n,
# p) indicator `X` to `fit_gaussian_gllvm(Y; X, K)` (still the same closed-
# form `GllvmFit` engine); with that, `dof`/`loglikelihood`/`aic`/`bic`
# reproduce R's recorded values to ~1e-8 at K = 1, 2, 3 (test/fixtures/
# gllvmtmb_anova_fixture.json) — full agreement, not merely a shared
# convention. `aic`/`bic` are twinned by test only; no new code was needed
# for them, and none is added here, per AGENTS.md "do not change existing
# behaviour".
#
# `update.gllvmTMB_multi` (R/methods-gllvmTMB.R) has NO twin in this file.
# Reading its R definition shows it is not a general "re-fit with changed
# arguments" method: for a non-`temporal_latent` fit (the only case
# GLLVModels.jl could possibly match — temporal covariance providers are an
# R-only capability, gllvmTMB-exports-recon.md) it does nothing but delegate
# to `stats::update.default(object, ...)`, which replays a *stored call*.
# GLLVModels.jl fit structs (`GllvmFit` and every other `AnyGllvmFit` member,
# src/postfit.jl:9) do not retain the original `gllvm(formula, Y, data; ...)`
# call or the `data`/`Y` arguments — only the fitted parameters and design
# metadata. There is therefore nothing to replay: an `update()` twin would
# have to either silently refit from scratch on caller-supplied arguments
# (not what R's `update()` contracts to do) or fail on every call. Per this
# PR's task brief, this is a documented STOP rather than a fabricated
# partial implementation — see the PR description.

# Best-effort latent rank of a fit, used only to classify anova() steps
# (see `gllvm_anova` below). Reuses `_loadings` (src/postfit.jl), which
# already dispatches Λ/Λc/pars.Λ across the AnyGllvmFit union; returns
# `missing` for the few fit types `_loadings` does not cover (e.g. the
# two-level Λ_B/Λ_W split) rather than throwing, since an unknown rank must
# not abort the whole comparison — it only prevents classifying that fit's
# adjacent step as a rank step.
function _anova_latent_rank(fit)
    try
        return size(_loadings(fit), 2)
    catch
        return missing
    end
end

"""
    GllvmAnovaTable

Result of [`gllvm_anova`](@ref): one row per compared fit, ordered by
increasing free-parameter count (`npar`). Fields (all vectors, aligned by
row):

- `model::Vector{String}` — `"Model 1"`, `"Model 2"`, ...
- `d::Vector{Union{Missing,Int}}` — latent rank if determinable, else `missing`.
- `npar::Vector{Int}` — free-parameter count ([`dof`](@ref)).
- `loglik::Vector{Float64}` — maximised marginal log-likelihood ([`loglikelihood`](@ref)).
- `deviance::Vector{Float64}` — `-2 * loglik`.
- `df::Vector{Union{Missing,Int}}` — parameter-count change from the previous row (`missing` on row 1).
- `LRT::Vector{Union{Missing,Float64}}` — likelihood-ratio statistic vs the previous row (`missing` on row 1).
- `test::Vector{String}` — the test actually used for that row's step (`"chisq"`, `"chibar (q=...)"`, `"none"`, or `"refused"`); `""` on row 1.
- `pvalue::Vector{Union{Missing,Float64}}` — p-value for that step, or `missing`.
- `note::Vector{Union{Missing,String}}` — a per-row explanation (caveat or refusal reason), or `missing`.
"""
struct GllvmAnovaTable
    model::Vector{String}
    d::Vector{Union{Missing,Int}}
    npar::Vector{Int}
    loglik::Vector{Float64}
    deviance::Vector{Float64}
    df::Vector{Union{Missing,Int}}
    LRT::Vector{Union{Missing,Float64}}
    test::Vector{String}
    pvalue::Vector{Union{Missing,Float64}}
    note::Vector{Union{Missing,String}}
end

# The chi-bar caveat / interior / refusal note text below intentionally
# mirrors the substance of R's anova.gllvmTMB_multi() notes (R/aghq-report.R)
# so a reader who already knows the R package recognises the same claims;
# wording is adapted for GLLVModels.jl.
const _ANOVA_NOTE_FIXED = "Interior (regular) fixed-effect comparison; ordinary Wilks chi-square applies. GLLVModels.jl cannot verify that the smaller model's fixed effects are a strict subset of the larger's (fit structs do not retain the design), so nesting is ASSUMED, not checked."
const _ANOVA_NOTE_CHISQ_AT_RANK = "User-requested plain chi-square at a boundary (rank) comparison: this p-value is too LARGE (the naive test is conservative, i.e. understates the evidence) because it ignores the positive probability mass the null places exactly at the boundary of the parameter space. Prefer test = :chibar."
const _ANOVA_NOTE_CHIBAR = "Self-Liang chi-bar-square mixture with q independent boundary components (Self & Liang 1987), via chibar2_pvalue. Independence of these q new loading parameters is a documented APPROXIMATION for a rank/dimension test, not a proven fact for this parameterisation (gllvmTMB R/aghq-report.R carries the identical caveat for its anova.gllvmTMB_multi)."
const _ANOVA_NOTE_UNCLASSIFIED = "Could not classify this step as a single fixed-effect change or a single (+1) latent-rank change from free-parameter counts and latent rank alone. Compare adjacent fits that differ in exactly one of those two ways, or use select_lv() for a criterion-based choice across many K."

# Core comparison logic, decoupled from `AnyGllvmFit` so it can be exercised
# directly against externally supplied (e.g. R-recorded) npar/loglik/d
# triples in tests, independent of GLLVModels.jl's own fitting path.
function _anova_core(npar_v::AbstractVector{<:Integer}, loglik_v::AbstractVector{<:Real},
                      d_v::AbstractVector; test::Symbol = :chibar)
    test in (:chibar, :chisq, :none) ||
        throw(ArgumentError("test must be :chibar, :chisq, or :none; got :$test"))
    n = length(npar_v)
    n == length(loglik_v) == length(d_v) ||
        throw(ArgumentError("npar_v, loglik_v, and d_v must have the same length"))
    n >= 2 ||
        throw(ArgumentError("gllvm_anova needs at least two fits to compare; got $n"))
    allunique(npar_v) ||
        throw(ArgumentError("two or more fits have the same free-parameter count ($(npar_v)); gllvm_anova cannot order them for a sequential comparison"))

    ord = sortperm(collect(npar_v))
    npar_s = collect(Int, npar_v)[ord]
    loglik_s = collect(Float64, loglik_v)[ord]
    d_s = collect(Union{Missing,Int}, d_v)[ord]
    deviance_s = -2 .* loglik_s

    df_v      = Vector{Union{Missing,Int}}(missing, n)
    lrt_v     = Vector{Union{Missing,Float64}}(missing, n)
    method_v  = fill("", n)
    pvalue_v  = Vector{Union{Missing,Float64}}(missing, n)
    note_v    = Vector{Union{Missing,String}}(missing, n)

    for i in 2:n
        df_v[i]  = npar_s[i] - npar_s[i - 1]
        lrt_v[i] = deviance_s[i - 1] - deviance_s[i]

        d_prev, d_curr = d_s[i - 1], d_s[i]
        is_rank_step = d_prev !== missing && d_curr !== missing && (d_curr - d_prev) == 1
        is_fixed_step = d_prev !== missing && d_curr !== missing && d_curr == d_prev

        if is_rank_step
            q = df_v[i]
            if test === :none
                method_v[i] = "none"
            elseif test === :chisq
                pvalue_v[i] = ccdf(Chisq(df_v[i]), lrt_v[i])
                method_v[i] = "chisq (requested)"
                note_v[i] = _ANOVA_NOTE_CHISQ_AT_RANK
            else
                pvalue_v[i] = chibar2_pvalue(lrt_v[i], q)
                method_v[i] = "chibar (q=$q)"
                note_v[i] = _ANOVA_NOTE_CHIBAR
            end
        elseif is_fixed_step
            if test === :none
                method_v[i] = "none"
            else
                pvalue_v[i] = ccdf(Chisq(df_v[i]), lrt_v[i])
                method_v[i] = "chisq"
                note_v[i] = _ANOVA_NOTE_FIXED
            end
        else
            method_v[i] = "refused"
            note_v[i] = _ANOVA_NOTE_UNCLASSIFIED
        end
    end

    return GllvmAnovaTable(
        ["Model $i" for i in 1:n], d_s, npar_s, loglik_s, deviance_s,
        df_v, lrt_v, method_v, pvalue_v, note_v,
    )
end

"""
    gllvm_anova(fits::AnyGllvmFit...; test = :chibar) -> GllvmAnovaTable

Sequential nested likelihood-ratio comparison of two or more fitted GLLVMs,
sorted by increasing free-parameter count ([`dof`](@ref)). A twin of
gllvmTMB's `anova.gllvmTMB_multi()` (R/aghq-report.R, pin
`9539352f66f2db2cc26b1c393e67212a359b60c9`): each adjacent pair is classified
as a **rank (boundary) step** (latent dimension `K` increases by exactly 1;
the Self–Liang chi-bar-square mixture [`chibar2_pvalue`](@ref) is used, as an
explicit approximation — see Details), an **interior fixed-effect step**
(latent rank unchanged; plain Wilks chi-square), or left **unclassified**
("refused" in the returned table, with a `note` explaining why) when neither
pattern is detected from free-parameter counts and latent rank alone.

`test` is one of `:chibar` (default), `:chisq` (plain chi-square everywhere —
conservative at a rank step, flagged in the note), or `:none` (report the
table with no p-values).

# Details

R's `anova.gllvmTMB_multi()` additionally verifies that the fixed-effect
design of the smaller model is a strict subset of the larger's, that every
fit was made to the same data, family, and estimator, and refuses outright
if any of those fail. GLLVModels.jl fit structs do not retain the design,
data, or call they were fitted from (the same reason gllvmTMB's `update()`
has no twin here — see the file-level note in `src/model_comparison.jl`), so
`gllvm_anova` **cannot perform those checks**: nesting and identical
data/family are ASSUMED of the caller, not verified. Compare only fits you
know are genuinely nested and fitted to the same data.

The rank-step chi-bar-square mixture is the identical Self & Liang (1987)
approximation R's anova documents for its own newly-added-loading-column
case (`q` new free parameters treated as `q` independent boundary variances,
which they are not — see [`chibar2_pvalue`](@ref)); it is not a proven exact
result for this parameterisation, on either side of the twin.

```julia
fit1 = fit_gaussian_gllvm(Y; K = 1)
fit2 = fit_gaussian_gllvm(Y; K = 2)
fit3 = fit_gaussian_gllvm(Y; K = 3)
tab = gllvm_anova(fit1, fit2, fit3)   # test = :chibar by default
tab.pvalue                            # p-value per step (missing on row 1)
```

See also [`select_lv`](@ref) for a criterion-based (AIC/BIC) choice across a
`K` sweep instead of pairwise hypothesis tests.
"""
function gllvm_anova(fits::AnyGllvmFit...; test::Symbol = :chibar)
    length(fits) >= 2 ||
        throw(ArgumentError("gllvm_anova needs at least two fits to compare; got $(length(fits))"))
    any(f isa GaussianREMLFit for f in fits) &&
        throw(ArgumentError("gllvm_anova (likelihood-ratio comparison) is not defined for REML fits; refit with the ML fitter before comparing"))

    npar_v   = [StatsAPI.dof(f) for f in fits]
    loglik_v = [StatsAPI.loglikelihood(f) for f in fits]
    d_v      = Union{Missing,Int}[_anova_latent_rank(f) for f in fits]
    return _anova_core(npar_v, loglik_v, d_v; test = test)
end

# print.anova.gllvmTMB_multi twin — Julia's multiple-dispatch equivalent of
# an S3 print method. Table layout follows R's print.anova.gllvmTMB_multi
# (model / d / npar / logLik / deviance / df / LRT / test / p.value), then a
# "Notes:" section for any row carrying one.
function Base.show(io::IO, ::MIME"text/plain", x::GllvmAnovaTable)
    println(io, "Likelihood-ratio comparison of GLLVModels.jl fits")
    println(io)
    header = ("model", "d", "npar", "logLik", "deviance", "df", "LRT", "test", "p.value")
    println(io, join(lpad.(header, 10), " "))
    n = length(x.model)
    fmt_missing(v; digits = 4) = v === missing ? "NA" : string(round(v; digits = digits))
    for i in 1:n
        row = (
            x.model[i],
            x.d[i] === missing ? "NA" : string(x.d[i]),
            string(x.npar[i]),
            string(round(x.loglik[i]; digits = 4)),
            string(round(x.deviance[i]; digits = 4)),
            fmt_missing(x.df[i]; digits = 0),
            fmt_missing(x.LRT[i]),
            x.test[i],
            fmt_missing(x.pvalue[i]),
        )
        println(io, join(lpad.(row, 10), " "))
    end
    has_notes = any(v -> v !== missing, x.note)
    if has_notes
        println(io)
        println(io, "Notes:")
        for i in 1:n
            x.note[i] === missing && continue
            println(io, "  Model $i: ", x.note[i])
        end
    end
end

Base.show(io::IO, x::GllvmAnovaTable) =
    print(io, "GllvmAnovaTable(", length(x.model), " models)")
