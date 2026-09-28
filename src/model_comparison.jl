# Nested-model comparison — a twin of gllvmTMB's `anova.gllvmTMB_multi()` /
# `print.anova.gllvmTMB_multi()` S3 methods (R/aghq-report.R, pin
# 9539352f66f2db2cc26b1c393e67212a359b60c9, "P1"). `AIC.gllvmTMB_multi` and
# `BIC.gllvmTMB_multi` are NOT re-implemented here: they wrap `stats::AIC`/
# `stats::BIC` with `k = 2` and `attr(logLik(fit), "df")`/`nobs(fit)`, and
# GLLVModels.jl's own `aic`/`bic`/`dof`/`nobs` (src/postfit.jl) already use
# the identical conventions once the SAME model is compared: `dof(fit)`
# matches R's `length(opt$par)` exactly (both count loadings modulo the
# K(K−1)/2 rotational df) and `nobs(fit, Y)` is documented as R's p·n
# observed-cell count, not the number of sites (src/postfit.jl:604-620).
#
# SCOPE OF THAT VERIFICATION: numerically checked only for the Gaussian
# closed-form route (`GllvmFit`) with a single shared residual variance
# (test/fixtures/gllvmtmb_anova_fixture.toml, K = 1, 2, 3). The other
# Laplace-approximated families and the REML/diagonal-variance/masked-data
# routes are NOT independently re-verified here; their `dof`/`nobs`
# docstrings carry the same R-convention claim, but this PR did not fit each
# one against a live gllvmTMB oracle. For data with missing cells, pass the
# fit's own `mask` through explicitly (`nobs(fit, Y; mask = mask)` /
# `bic(fit, Y; mask = mask)`, src/postfit.jl) — a fit struct does not retain
# the mask it was fitted with, so the default (`mask = nothing`, "complete
# data") silently overcounts `nobs` for a masked fit if `mask` is omitted.
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
# gllvmtmb_anova_fixture.toml) — full agreement, not merely a shared
# convention. `aic`/`bic` are twinned by test only; no new code was needed
# for them, and none is added here, per AGENTS.md "do not change existing
# behaviour".
#
# `update.gllvmTMB_multi` (R/methods-gllvmTMB.R) has NO twin in this file.
# Reading its R definition shows it is not a general "re-fit with changed
# arguments" method: it delegates to `stats::update.default(object, ...)`,
# which replays the fit's *stored call* (`getCall(object)` / `object$call`).
# That call is stored ONLY for two cases (R/gllvmTMB.R:1467-1472): a
# `gllvmTMB_va` (variational) fit, or an active `temporal_latent` fit. Every
# OTHER ordinary ML/REML gllvmTMB fit (the great majority: plain Gaussian,
# Poisson, Binomial, ... fits) has NO stored call at all, so R's OWN
# `update()` is not merely narrower on those fits, it errors ("need an
# object with a call component") — VA and temporal are the only cases where
# it actually works, and both are R-only capabilities with no Julia
# equivalent at all (temporal per gllvmTMB-exports-recon.md; GLLVModels.jl
# has no variational-fit class that retains a call either). GLLVModels.jl
# fit structs (`GllvmFit` and every other `AnyGllvmFit` member,
# src/postfit.jl:9) do not retain the original `gllvm(formula, Y, data; ...)`
# call or the `data`/`Y` arguments for ANY fit — only the fitted parameters
# and design metadata. There is therefore nothing to replay for any
# GLLVModels.jl fit: an `update()` twin would have to either silently refit
# from scratch on caller-supplied arguments (not what R's `update()`
# contracts to do) or fail on every call, the same as R's own `update()`
# does outside its two narrow working cases. Per this PR's task brief, this
# is a documented STOP rather than a fabricated partial implementation, see
# the PR description.

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

# Best-effort species/trait count (p) of a fit: the loadings' ROW count.
# Used to VERIFY, not just assume, a rank-step's q = p - d_prev (see
# `_anova_core`) — the review finding this file fixes was a step reported as
# a clean rank test with q taken straight from the free-parameter delta,
# which silently absorbs any other change (e.g. fixed effects added) between
# the two fits. `missing` when `_loadings` does not dispatch for this type.
function _anova_p(fit)
    try
        return size(_loadings(fit), 1)
    catch
        return missing
    end
end

# A fit's family marker when its type carries one explicitly (e.g.
# `GllvmCovFit`, which is reused across several families — `typeof` alone
# does not distinguish a Gamma `GllvmCovFit` from a NegativeBinomial one).
# `nothing` when the type has no `family` field (most fit types ARE
# themselves family-specific, e.g. `PoissonFit`, `NBGroupedFit`).
_anova_family_key(fit) = hasfield(typeof(fit), :family) ? string(fit.family) : nothing

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
- `test::Vector{String}` — the test actually used for that row's step (`"chisq"`, `"chisq (requested)"`, `"chibar (q=...)"`, `"none"`, or `"refused"`); `""` on row 1.
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
#
# `p` is the (single, assumed-common) species/trait count, used ONLY to
# VERIFY a would-be rank step's free-parameter delta against the expected
# q = p - d_prev (the package's rr() lower-triangular loading-column cost,
# the same formula R's anova.gllvmTMB_multi uses for its chi-bar-square q).
# `missing` when the caller could not determine it (e.g. a fit type with no
# `_loadings` dispatch) — every would-be rank step is then left unclassified
# ("refused") rather than trusting an unverified q, because a step's
# free-parameter delta alone cannot distinguish "the rank went up by exactly
# one, nothing else changed" from "the rank went up AND something else also
# changed" (the defect this function's caller, `gllvm_anova`, was fixed for:
# comparing a no-X fit at K to an X-added fit at K+1 silently absorbed the
# X's parameters into a wrong q).
function _anova_core(npar_v::AbstractVector{<:Integer}, loglik_v::AbstractVector{<:Real},
                      d_v::AbstractVector, p::Union{Missing,Integer} = missing;
                      test::Symbol = :chibar)
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
        d_up_by_one = d_prev !== missing && d_curr !== missing && (d_curr - d_prev) == 1
        is_fixed_step = d_prev !== missing && d_curr !== missing && d_curr == d_prev

        # A rank step is trusted CLEAN only when the observed free-parameter
        # change equals the expected q exactly; otherwise something ELSE also
        # changed (fixed effects, family, structure) and this is a compound
        # step, refused per-row exactly as R refuses a compound step.
        q_expected = (d_up_by_one && p !== missing) ? p - d_prev : missing
        is_clean_rank_step = q_expected !== missing && df_v[i] == q_expected

        if is_clean_rank_step
            q = q_expected
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
            note_v[i] = if d_up_by_one && p === missing
                "Latent rank increased by 1, but the species/trait count p could not be determined for this fit type (no _loadings dispatch), so the expected rank-step free-parameter count could not be verified. " * _ANOVA_NOTE_UNCLASSIFIED
            elseif d_up_by_one && p !== missing
                "Latent rank increased by 1 (from $d_prev to $d_curr), but the free-parameter count changed by $(df_v[i]) instead of the $(q_expected) expected for a clean rank-only step at p = $p species/traits; something else (fixed effects, family, or structure) also changed between these two fits. " * _ANOVA_NOTE_UNCLASSIFIED
            else
                _ANOVA_NOTE_UNCLASSIFIED
            end
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
if any of those fail. `gllvm_anova` verifies what it mechanically can from
the fit structs themselves, and refuses the WHOLE call (an `ArgumentError`,
not a per-row note) when:

- the fits are not all the same concrete type, or (for a type reused across
  families, e.g. `GllvmCovFit`) do not all carry the same `family`;
- a determinable species/trait count (`p`, from `_loadings`) differs across
  fits;
- any fit is a `GaussianREMLFit` (R's ML-only restriction).

It does NOT verify identical data (GLLVModels.jl fit structs do not retain
the number of sites/rows `n`, or the response `Y` itself; see the file-level
note in `src/model_comparison.jl` for why gllvmTMB's `update()` has no twin
here, for the same underlying reason) or that a fixed-effect step's design is
actually nested (fit structs do not retain the design at all). **Those two
remain assumptions about the caller, not checks.** A rank step's `q` IS
checked, not assumed: it is trusted only when the observed free-parameter
change matches `p - d_prev` exactly, and a step where the rank went up by one
but the free-parameter delta does not match that (something else, such as
added fixed effects, also changed) is refused per-row rather than silently
mislabelled as a clean rank step.

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

    # Twin of R's family/engine-mismatch refusal
    # (.gllvmTMB_anova_global_check aborts outright on differing
    # family_id_vec or engine): a GLOBAL refusal of the whole call, not a
    # per-step note, because incomparable families make every step's test
    # meaningless, not just one row. Each GLLVModels.jl family has its own
    # concrete fit type EXCEPT the generic covariate fits (`GllvmCovFit` and
    # similar), which are reused across families via a `family` field — a
    # bare `typeof` check misses a Gamma-vs-NegativeBinomial `GllvmCovFit`
    # pair, so both `typeof` and, when present, `family` must agree.
    compat_label(f) = _anova_family_key(f) === nothing ? string(typeof(f)) :
                      string(typeof(f)) * " (" * _anova_family_key(f) * ")"
    compat_keys = [(typeof(f), _anova_family_key(f)) for f in fits]
    length(unique(compat_keys)) == 1 ||
        throw(ArgumentError("gllvm_anova compares only fits of the same family/engine; got " *
                             join(unique(compat_label.(fits)), ", ")))

    npar_v   = [StatsAPI.dof(f) for f in fits]
    loglik_v = [StatsAPI.loglikelihood(f) for f in fits]
    d_v      = Union{Missing,Int}[_anova_latent_rank(f) for f in fits]
    p_v      = Union{Missing,Int}[_anova_p(f) for f in fits]

    # Twin of R's "same data" refusal, restricted to what a fit struct
    # actually retains: the response dimension p. GLLVModels.jl fit structs
    # do not store the number of sites/rows n, so identical n stays a
    # DOCUMENTED ASSUMPTION (see the docstring), not a checked one.
    known_p = collect(skipmissing(p_v))
    if !isempty(known_p) && !all(==(first(known_p)), known_p)
        throw(ArgumentError("gllvm_anova requires every fit to have the same number of species/traits; got $(known_p)"))
    end
    p = isempty(known_p) ? missing : first(known_p)

    return _anova_core(npar_v, loglik_v, d_v, p; test = test)
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
