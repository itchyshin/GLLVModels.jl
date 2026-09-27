# Julia twin of gllvmTMB's extract_latent_scores() (R/extract-latent-scores.R,
# pinned P1 9539352f66f2db2cc26b1c393e67212a359b60c9, gllvmTMB 0.7.1).
#
# R's generic dispatches on four S3 methods: .default, .gllvmTMB_multi,
# .gllvmTMB_site_trait_sim, .gllvmTMB_va. Per the P1 case map
# (GLLVModels.jl PR #526), .gllvmTMB_site_trait_sim and .gllvmTMB_va are
# EXCLUDED from this twin: Julia has neither a site-trait-simulation class
# (simulate_site_trait()'s truth$z_B/z_W generating draws) nor a
# variational-approximation fit class to extract scores from. This file
# twins only .default and .gllvmTMB_multi.
#
# Dispatch note: not every `getLV` method in this package accepts a
# `component` keyword. Only the seven fit types with a predictor-informed
# latent-score mean (`X_lv`) do — GllvmFit, BinomialFit, PoissonFit, NBFit,
# BetaFit, OrdinalFit, GammaFit (src/postfit.jl) — and only for those does
# `component = :innovation` differ from the plain, no-`component` call.
# `_ComponentAwareGllvmFit` below is exactly this set.
#
# Every fit type in `_PlainGllvmFit` has no `X_lv`/predictor-informed mean at
# all, so it has no separate "mean" layer to add to or subtract from:
# whatever its plain `getLV(fit, y; rotate=false, ...)` (no `component`
# keyword — passing one would raise a plain `MethodError`) returns already IS
# the (only) zero-mean latent score.
#
# `_PositionalArgGllvmFit` is a further, disjoint group of fit types whose
# `getLV` needs an extra required positional argument beyond `(fit, y)` — `X`
# (a 3-D covariate array, or a 2-D design matrix for
# `ConstrainedOrdinationFit`), `(Xenv, TR)` for `FourthCornerFit`, or `locs`
# for `SPDELatentFit`. This wrapper's signature is `(fit, y; kwargs...)`, so
# it cannot route a required positional argument through `kwargs...`; rather
# than silently misrouting it as a keyword (which would raise an unhelpful
# `MethodError`), these types get a named `ArgumentError` pointing at the
# `getLV` call to make directly. `RRRFit` is in this Union too, but for a
# different reason (see its own, more specific method below, which Julia
# dispatches to in preference to the Union-typed one): its
# `getLV(fit, X; rotate)` is a plain 2-argument call with no missing
# argument, but `X` there is a deterministic, fully predictor-driven
# reduced-rank-regression projection with no latent innovation at all — there
# is no meaningful "innovation" score to return, and treating this wrapper's
# `y` (a response matrix) as RRRFit's `X` (a predictor design matrix) would
# raise a raw, unhelpful `DimensionMismatch` instead.
#
# `_ComponentAwareGllvmFit`, `_PlainGllvmFit`, and `_PositionalArgGllvmFit`
# are meant to jointly, disjointly cover every `AnyGllvmFit` member that has
# a `getLV` method at all (test/test_extract_latent_scores.jl asserts this
# via `methods(getLV)`, so a newly added fit type with a `getLV` method that
# nobody sorts into one of these three `Union`s turns that test red instead
# of silently falling through to a wrong bucket). `AnyGllvmFit` members with
# no `getLV` method at all (e.g. `MultinomialFit`, `StudentTFit`,
# `TruncatedPoissonFit`, the phylo/spatial/temporal-only fits) are outside
# this accounting on purpose: they already fail with Julia's plain
# `MethodError` on `_extract_latent_scores_unit` (same failure mode as
# calling `getLV` on them directly today), which is the correct behaviour —
# there is nothing to twin for a fit type this package cannot ordinate at
# all. Likewise, `QuadraticFit`, `OrderedBetaFit`, and `MixedFamilyFit` have
# `getLV` methods but are not part of `AnyGllvmFit`, so `extract_latent_scores`
# cannot reach them either (see the PR body's "not covered" list).

const _ComponentAwareGllvmFit = Union{
    GllvmFit, BinomialFit, PoissonFit, NBFit, BetaFit, OrdinalFit, GammaFit,
}

const _PlainGllvmFit = Union{
    NB1Fit, GP1Fit, ExponentialFit, DeltaLogNormalFit, HurdlePoissonFit,
    HurdleNBFit, DeltaGammaFit, ZIPFit, ZINBFit, ZIBFit, TweedieFit,
    COMPoissonFit, BetaBinomialFit, BetaBinomialGroupedFit, BetaHurdleFit,
    GammaGroupedFit, NB1GroupedFit, NBGroupedFit, BetaGroupedFit,
    OrdinalPerTraitFit, RowEffectFit, RowRandomFit,
}

const _PositionalArgGllvmFit = Union{
    GllvmCovFit, GllvmSpeciesCovFit, ZIPCovFit, ZINBCovFit, ZIBCovFit,
    BetaBinomialGroupedCovFit, GammaGroupedCovFit, NB1GroupedCovFit,
    NBGroupedCovFit, BetaGroupedCovFit, OrdinalPerTraitCovFit,
    ConstrainedOrdinationFit, FourthCornerFit, SPDELatentFit, RRRFit,
}

"""
    extract_latent_scores(fit, y; level=:unit, kwargs...) -> Matrix{Float64} or Nothing

Twin of gllvmTMB's `extract_latent_scores(x, level)` for a fitted model
(`R/extract-latent-scores.R`, `extract_latent_scores.gllvmTMB_multi`). R
documents the point estimate as matching `extract_ordination(x, level, component
= "innovation")`, `getLV(x, level)` at `rotate = "none"`, and
`ordination_uncertainty(x, level)\$scores`. In this package, `level = :unit`
returns `getLV(fit, y; rotate = false, kwargs...)` — passing `component =
:innovation` too on the seven fit types whose `getLV` accepts that keyword
(see the dispatch note at the top of `src/extract_latent_scores.jl`) — the
Gaussian posterior mean / Laplace mode of the between-unit latent scores in
native (unrotated) orientation. The seven `component`-aware types are the
*only* ones with a predictor-informed latent-score mean (`X_lv`) at all;
every other fit type this method supports has no such mean layer, so its
plain (no-`component`) `getLV` call already returns the zero-mean score.

This package's own [`extract_ordination`](@ref) (`src/extractors.jl`) already
exists and forwards to [`ordination`](@ref), whose `sites` field is
`getLV(fit, Y; rotate = false)` called with `getLV`'s *default* `component`
(`:total` on the seven `X_lv`-capable types, not `:innovation`) and, notably,
with no fixed-effect `X` forwarded at all — `ordination()` always evaluates
`getLV` as if `X = nothing`. So `extract_ordination(fit, Y; rotate =
false).sites == extract_latent_scores(fit, Y)` holds exactly only on a fit
with **neither** a fixed-effect `X` **nor** an `X_lv` (both `:mean` layers
zero, so `:total` and `:innovation` coincide and the omitted `X` costs
nothing); on a fit with either, the two differ and `extract_ordination`'s
result is not a meaningful ordination for that fit. This mirrors R's own
`extract_ordination(x, component = "innovation")` identity, restricted here
to the fits `extract_ordination`'s (fixed-effect-blind) call to `getLV`
already supports correctly.

`level = :unit_obs` always returns `nothing`. R's `unit_obs` (`z_W`) tier is
the within-unit block of gllvmTMB's two-tier `latent(0 + trait | site, d)`
"multi" formula grammar — a nested between-unit/within-unit reduced-rank
structure fit on a long-format trait table, entirely separate from the
single-tier wide-format `Y` (species/trait × site) GLLVM this package
implements. This package cannot fit a within-unit tier at all — no fit type
here has one — so, unlike R (where `level = "unit"` on an indep-only fit with
no reduced-rank term also returns `NULL`, a case that cannot arise in this
package since it has no such formula grammar to fit that model in the first
place), `level = :unit_obs` returning `nothing` is not fit-dependent: it is
categorically true of every fit this package can produce, not a per-fit
"this model happens to lack that tier" check.

`y` and `kwargs...` (e.g. `X_lv`, `N`, `mask`, `offset`) forward to the
fit-type-specific [`getLV`](@ref) method exactly as `getLV` itself requires
them for that fit type. Fit types whose `getLV` needs an extra *required
positional* argument beyond `(fit, y)` (`X`, `(Xenv, TR)`, or `locs`) raise a
named `ArgumentError` here instead — call `getLV` directly for those.
`RRRFit` raises a different named `ArgumentError`: it is a pure
reduced-rank-regression fit with a deterministic, fully predictor-driven
projection and no latent innovation score at all (see the dispatch note at
the top of `src/extract_latent_scores.jl` for the full list and reasoning).

# Differences from R (documented, not twinned)
- **Explicit `y`**: R's fitted object retains its TMB environment and needs
  only `(x, level)`; this package's fits do not store the data (see
  [`getLV`](@ref)), so `y` (and any of `X_lv`/`N`/`mask`/`offset` the fit
  used) must be supplied here.
- **No deprecated aliases**: R accepts `level = "B"`/`"W"` (old names for
  `"unit"`/`"unit_obs"`) with a one-time warning. This method only accepts
  `:unit`/`:unit_obs` and throws `ArgumentError` on anything else, including
  `:B`/`:W` — no warn-and-continue path is provided.
- **No row/column names**: R returns unit row names and `"LV1"`, `"LV2"`, ...
  column names. This package's `getLV` already returns a plain
  `Matrix{Float64}` with no names — a pre-existing, package-wide convention,
  not introduced here.
- **`gllvmTMB_multi`'s trait-table model family**: beyond the ordinary
  single-tier abundance/trait GLLVM twinned here, R's `gllvmTMB_multi` class
  also covers the `latent(0 + trait | site, d)` two-tier trait-table grammar
  (unbalanced replication per unit, an optional `unit_obs` tier). That model
  family is a documented Julia gap, not part of this twin.

See also [`getLV`](@ref) and [`extract_ordination`](@ref).
"""
function extract_latent_scores(fit::AnyGllvmFit, y::AbstractMatrix;
                                level::Symbol = :unit, kwargs...)
    level === :unit || level === :unit_obs ||
        throw(ArgumentError("level must be :unit or :unit_obs; got :$level"))
    level === :unit_obs && return nothing
    return _extract_latent_scores_unit(fit, y; kwargs...)
end

# level = :unit dispatch. The seven X_lv-capable types get component =
# :innovation explicitly; every "plain" fit type has no separate mean layer,
# so the no-component call already returns the zero-mean score.
_extract_latent_scores_unit(fit::_ComponentAwareGllvmFit, y::AbstractMatrix; kwargs...) =
    getLV(fit, y; component = :innovation, rotate = false, kwargs...)

_extract_latent_scores_unit(fit::_PlainGllvmFit, y::AbstractMatrix; kwargs...) =
    getLV(fit, y; rotate = false, kwargs...)

# RRRFit: more specific than the _PositionalArgGllvmFit method below (RRRFit
# is one of that Union's members, but Julia always prefers a method typed on
# the concrete type over one typed on a Union containing it), so this is the
# method RRRFit actually dispatches to.
function _extract_latent_scores_unit(fit::RRRFit, y::AbstractMatrix; kwargs...)
    throw(ArgumentError(
        "extract_latent_scores has no innovation score for RRRFit: it is a " *
        "pure reduced-rank-regression fit, a deterministic, fully " *
        "predictor-driven projection z_s = B' x_s with no latent innovation " *
        "at all; call getLV(fit, X) directly for the constrained ordination axes"))
end

function _extract_latent_scores_unit(fit::_PositionalArgGllvmFit, y::AbstractMatrix; kwargs...)
    throw(ArgumentError(
        "extract_latent_scores does not route $(typeof(fit))'s extra required " *
        "getLV argument (X, (Xenv, TR), or locs — see the dispatch note at the " *
        "top of src/extract_latent_scores.jl); call getLV(fit, y, <that argument>; " *
        "rotate = false, ...) directly instead"))
end

"""
    extract_latent_scores(x, args...; kwargs...)

Fallback twin of `extract_latent_scores.default` (`R/extract-latent-scores.R`),
which aborts with a named error for any object without a specific method.
Without this method, calling `extract_latent_scores` on an unsupported type
would raise Julia's generic `MethodError`; this gives the same kind of
informative, named failure R raises instead.
"""
function extract_latent_scores(x, args...; kwargs...)
    throw(ArgumentError(
        "no extract_latent_scores method for $(typeof(x)); expected a fitted " *
        "GLLVModels fit object (see AnyGllvmFit in src/postfit.jl)"))
end
