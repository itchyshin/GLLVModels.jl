# Unified GLLVM fit entry point — dispatches on the response family.

"""
    fit_gllvm(Y; family = Normal(), K, num_lv = nothing,
              row_eff = :none, disp_group = nothing, pervar = false, kwargs...)

Fit a GLLVM, dispatching on the response `family` — a Distributions.jl
distribution used as a marker (the GLM.jl convention):

- `Normal()`   → [`fit_gaussian_gllvm`](@ref) — closed-form Gaussian marginal
- `Binomial()` → [`fit_binomial_gllvm`](@ref) — Laplace marginal (binary / binomial)
- `Poisson()`  → [`fit_poisson_gllvm`](@ref) — Laplace marginal (counts)
- `TruncatedPoisson()` → [`fit_truncated_poisson_gllvm`](@ref) — zero-truncated Poisson
- `CensoredPoisson()` → [`fit_censored_poisson_gllvm`](@ref) — right-censored Poisson (Julia-forward)
- `Lognormal()` → [`fit_lognormal_gllvm`](@ref) — one-part lognormal (twin fid 3)
- `TruncatedNegBin2()` → [`fit_truncated_nbinom2_gllvm`](@ref) — zero-truncated NB2 (shared `r`)
- `NegativeBinomial()` → [`fit_nb_gllvm_grouped`](@ref) with per-species `r`
  (twin-aligned default; shared-`r` via [`fit_nb_gllvm`](@ref))
- `Beta()`     → [`fit_beta_gllvm_grouped`](@ref) with per-species `φ`
  (twin-aligned default; shared-`φ` via [`fit_beta_gllvm`](@ref))
- `NB1()`      → [`fit_nb1_gllvm_grouped`](@ref) with per-species linear-variance `φ`
  (twin-aligned default; shared-`φ` via [`fit_nb1_gllvm`](@ref)). The marker's `φ`
  field is a tag payload — it is never read here; `φ` is always estimated.
- `BetaBinom()` → [`fit_beta_binomial_gllvm_grouped`](@ref) with per-species Beta
  precision `φ` (twin-aligned default; shared-`φ` via
  [`fit_beta_binomial_gllvm`](@ref)). The marker's `φ` field is a tag payload, as for
  `NB1`. The p×n trial counts `N` are **required** on this route: at `N = 1` the
  beta-binomial collapses to `Bernoulli(μ)` and `φ` is unidentifiable, so the named
  fitter's all-ones default is not inherited here and a missing `N` errors.
- `Ordinal()`  → [`fit_ordinal_gllvm_pertrait`](@ref) — Laplace marginal (ordered categories)
- `Gamma()`    → [`fit_gamma_gllvm`](@ref) — Laplace marginal (positive continuous; shared shape)
- `Exponential()` → [`fit_exponential_gllvm`](@ref) — Laplace marginal
- `StudentTFamily(ν)` → [`fit_studentt_gllvm`](@ref) — Laplace marginal (heavy-tailed
  continuous; location–scale t, identity link). Like `ZIB`, this is not a zero-payload
  marker: the FIXED degrees of freedom `ν` is structural and travels on the family
  instance, forwarded here as the fitter's `nu`, so passing `nu` as a separate keyword
  is an error. The marker's `σ` field is a tag payload — the scale is always estimated.
- `DeltaLogNormal()` → [`fit_delta_lognormal_gllvm`](@ref) — two-part Laplace (Bernoulli
  occurrence × positive lognormal). The marker's `σ` is a tag payload — always estimated.
- `DeltaGamma()` → [`fit_delta_gamma_gllvm`](@ref) — two-part Laplace (Bernoulli occurrence
  × positive Gamma). The marker's `α` is a tag payload — always estimated.
- `HurdlePoisson()` → [`fit_hurdle_poisson_gllvm`](@ref) — two-part Laplace (Bernoulli
  occurrence × zero-truncated Poisson). Empty marker; no payload.
- `HurdleNB()` → [`fit_hurdle_nb_gllvm`](@ref) — two-part Laplace (Bernoulli
  occurrence × zero-truncated NB2). The marker's `r` field is a tag payload —
  it is never read here; `r` is always estimated. There is no `r_init`.
- `BetaHurdle()` → [`fit_beta_hurdle_gllvm`](@ref) — two-part Laplace (Bernoulli
  occurrence × positive Beta). The marker's `φ` field is a tag payload — it is
  never read here; `φ` is always estimated. There is no `φ_init`.
- `COMPoisson()` → [`fit_compoisson_gllvm`](@ref) — Laplace marginal (CMP counts;
  under- or over-dispersion). The marker's `ν` field is a tag payload — it is
  never read here; `ν` is always estimated. This is the opposite of
  `StudentTFamily(ν)`, where numeric degrees of freedom are fixed and `nothing` requests estimation.
- `OrderedBeta()` → [`fit_ordered_beta_gllvm`](@ref) — Laplace marginal
  (proportions / cover with point masses at 0 and 1). The marker's `c0`, `c1`,
  and `φ` fields are tag payloads — they are never read here; all three are
  always estimated. They are not used as `c0_init` / `c1_init` / `φ_init`.
  This is the opposite of `StudentTFamily(ν)` (structural pin) and of Ordinal's
  `τ₁ = 0` pin.
- `Multinomial()` → [`fit_multinomial_gllvm`](@ref) — fixed-effect softmax
  (twin fid 16; v1, no LV). `K` / `num_lv` must be omitted or 0. Not a
  capability-table admit; ledger stays `missing`.
- `GeneralizedPoisson1(α)` → [`fit_gp1_gllvm`](@ref) — Laplace marginal (GP-1 counts, signed dispersion)
- `ZIB(N)` → [`fit_zib_gllvm`](@ref) — zero-inflated binomial (Julia-forward). Unlike the
  other markers, `ZIB` carries the shared trials count, so `N` travels on the family
  instance rather than as a keyword argument.

`K` is the latent dimension; the gllvm-style alias `num_lv` is accepted as a synonym
for `K` (gllvm uses `num.lv`). Family-specific keyword arguments (`link`, `N`,
`Σ_phy`, …) pass through to the underlying fitter.

# Structural / dispersion variants (gllvm-style keyword routing)

These keyword arguments route to the corresponding specialised fitter while keeping
the plain-call behaviour when they are at their defaults (regression safe):

- `row_eff::Symbol = :none` — community / random row effect (gllvm's `row.eff`):
  - `:none`   → no row effect (the standard dispatch above).
  - `:fixed`  → [`fit_roweffect_gllvm`](@ref) — per-site fixed intercept `ρ_s`.
  - `:random` → [`fit_row_random_gllvm`](@ref) — per-site random intercept `ρ_s ~ N(0, σ_row²)`.

- `disp_group` — grouped / species-specific dispersion (gllvm's `disp.group`).
  For `NegativeBinomial`, `Beta`, `NB1`, and `BetaBinom` only, `disp_group = nothing`
  is coerced to `:species` (per-trait φ / `r`, matching gllvmTMB). Pass a length-`p`
  integer vector of group ids for custom grouping, or `:species` explicitly. Routes to:
  - `NegativeBinomial` → [`fit_nb_gllvm_grouped`](@ref) (per-group `r`)
  - `Beta`             → [`fit_beta_gllvm_grouped`](@ref) (per-group `φ`)
  - `Gamma`            → [`fit_gamma_gllvm_grouped`](@ref) (per-species shape `α`; opt-in only)
  - `NB1`              → [`fit_nb1_gllvm_grouped`](@ref) (per-species linear-variance `φ`)
  - `BetaBinom`        → [`fit_beta_binomial_gllvm_grouped`](@ref) (per-group Beta
    precision `φ`; the p×n `N` keyword is required)
  - `GLLVModels.TweedieED`  → [`fit_tweedie_gllvm_grouped`](@ref) (per-species `φ`, shared power)
  - `TruncatedNegBin2` → [`fit_truncated_nbinom2_gllvm_pertrait`](@ref) with explicit
    `:species` or distinct positive group IDs; partial grouping is unsupported.
    Omitting `disp_group` preserves the shared-`r` default for this family.
  Families without a grouped fitter throw a clear `ArgumentError`. Shared
  dispersion for NB/Beta/NB1/BetaBinom remains available via the named fitters
  [`fit_nb_gllvm`](@ref) / [`fit_beta_gllvm`](@ref) / [`fit_nb1_gllvm`](@ref) /
  [`fit_beta_binomial_gllvm`](@ref).

- `pervar::Bool = false` — heteroscedastic (per-species variance) Gaussian. Only valid
  for `family = Normal()`; `true` routes to [`fit_gaussian_pervar_gllvm`](@ref).

## Precedence and unsupported combinations

`grouping=[GroupingTerm(...), ...]` selects jointly fitted named
random effects. Supply `unit`, `unit_obs`, `cluster`, and/or `cluster2` labels
for selected terms; labels alone never add a random effect. Do not also supply
`K`, `num_lv`, `row_eff`, `disp_group`, or `pervar`. Each term owns its covariance
and rank. Gaussian models have one shared residual variance and return
`GroupedGaussianFit`. Poisson-log, Binomial-logit, Beta-logit and NB2-log return
`GroupedNonGaussianFit` from one joint Laplace objective. Beta and NB2 estimate
trait-specific dispersion by default; use `dispersion=:shared` for a shared
value. Binomial requires a response-shaped trials matrix `N`. These development
routes have explicit fit/interval diagnostics; full parity qualification remains
in progress, and unsupported family/link/grouping combinations are not implied.

`phylo=precision` selects an explicit `PrecisionPhy` source for Gaussian data.
Use `phylo_rank` (default one), `phylo_mode` (default `:barelowrank`) and
`species_id` (observation-to-tip indices). Without grouping this returns a
`PrecisionMultivariateFit`; with grouping it returns a
`JointPhyloGroupedGaussianFit` from one joint marginal likelihood. The first
joint ordinary-term scope is `mode=:indep, common=false`. Precision-only fitting
accepts `residual_mode=:trait` (default) or `:shared`; joint grouping retains
trait-specific observation residual variances. Phylogenetic scale is fixed
at one and is not separately estimated alongside free trait loadings. Precision
options without `phylo`, non-Gaussian families and the legacy shortcuts below
are rejected. Native scaling and public R bridge admission are unchanged.

The legacy variants below route to single specialised fitters; no single underlying fitter combines
two of them. Therefore at most one of `row_eff != :none`, effective
`disp_group !== nothing` (including the NB/Beta default coerce), and
`pervar == true` may be active. Any other combination throws an `ArgumentError`
("combination not yet supported") rather than silently ignoring a request.

```julia
fit_gllvm(Y; family = Normal(),   K = 2)                          # Gaussian
fit_gllvm(Y; family = Binomial(), K = 2, link = LogitLink())      # binary
fit_gllvm(Y; family = Poisson(),  K = 2, row_eff = :random)       # random row effect
fit_gllvm(Y; family = NegativeBinomial(1.0, 0.5), K = 2)          # per-species r (default)
fit_gllvm(Y; family = Beta(), K = 2)                              # per-species φ (default)
fit_gllvm(Y; family = NB1(),  K = 2)                              # per-species linear-variance φ
fit_gllvm(Y; family = BetaBinom(), K = 2, N = trials)             # per-species φ; N is p×n, required
fit_gllvm(Y; family = StudentTFamily(4.0), K = 2)                 # heavy-tailed continuous, ν fixed
fit_gllvm(Y; family = COMPoisson(), K = 2)                        # CMP counts; ν estimated (tag payload)
fit_gllvm(Y; family = OrderedBeta(), K = 2)                       # 0/1 masses + Beta interior; tags estimated
fit_gllvm(Y; family = DeltaLogNormal(), K = 2)                    # two-part: occurrence × lognormal
fit_gllvm(Y; family = DeltaGamma(), K = 2)                        # two-part: occurrence × Gamma
fit_gllvm(Y; family = Normal(), K = 2, pervar = true)             # per-species variance
```
"""
function fit_gllvm(Y::AbstractMatrix; family = Normal(), K = nothing,
                   num_lv = nothing, row_eff::Symbol = :none,
                   disp_group = nothing, pervar::Bool = false,
                   grouping=nothing, unit=nothing, unit_obs=nothing,
                   cluster=nothing, cluster2=nothing,
                   phylo=nothing, phylo_rank=nothing, phylo_mode=nothing,
                   species_id=nothing, kwargs...)
    if phylo !== nothing
        phylo isa PrecisionPhy || throw(ArgumentError("phylo must be a PrecisionPhy"))
        family isa Normal || throw(ArgumentError("explicit precision fitting currently requires Gaussian responses"))
        K === nothing && num_lv === nothing || throw(ArgumentError(
            "use phylo_rank for the precision source; do not also supply K or num_lv"))
        row_eff === :none && disp_group === nothing && !pervar || throw(ArgumentError(
            "explicit precision fitting cannot be combined with row_eff, disp_group or pervar"))
        rank = phylo_rank === nothing ? 1 : phylo_rank
        mode = phylo_mode === nothing ? :barelowrank : phylo_mode
        rank isa Integer || throw(ArgumentError("phylo_rank must be an integer"))
        mode isa Symbol || throw(ArgumentError("phylo_mode must be a Symbol"))
        mapping = species_id === nothing ? collect(1:phylo.n_leaves) : species_id
        mapping isa AbstractVector{<:Integer} || throw(ArgumentError("species_id must be an integer vector"))
        if grouping === nothing
            all(isnothing, (unit, unit_obs, cluster, cluster2)) || throw(ArgumentError(
                "group identifiers do not add random effects; supply explicit grouping terms"))
            return fit_precision_multivariate(Y, phylo; rank=rank, mode=mode,
                species_id=mapping, kwargs...)
        end
        return fit_joint_phylo_grouped_gaussian(Y, phylo; rank=rank, phylo_mode=mode,
            species_id=mapping, terms=grouping, unit=unit, unit_obs=unit_obs,
            cluster=cluster, cluster2=cluster2, kwargs...)
    end
    all(isnothing, (phylo_rank, phylo_mode, species_id)) || throw(ArgumentError(
        "phylo_rank, phylo_mode and species_id require an explicit phylo=PrecisionPhy"))
    if grouping !== nothing
        (K === nothing && num_lv === nothing) || throw(ArgumentError(
            "explicit grouping terms own their ranks; do not also supply K or num_lv"))
        (row_eff === :none && disp_group === nothing && !pervar) || throw(ArgumentError(
            "explicit grouping cannot be combined with row_eff, disp_group or pervar"))
        if family isa Normal
            return fit_grouped_gaussian(Y; terms=grouping, unit=unit, unit_obs=unit_obs,
                cluster=cluster, cluster2=cluster2, kwargs...)
        elseif family isa Union{Poisson,Binomial,Beta,NegativeBinomial}
            return fit_grouped_nongaussian(Y; family=family, terms=grouping,
                unit=unit, unit_obs=unit_obs, cluster=cluster, cluster2=cluster2, kwargs...)
        end
        throw(ArgumentError("explicit grouping supports Gaussian, Poisson-log, Binomial-logit, Beta-logit and NB2-log only"))
    end
    all(isnothing, (unit, unit_obs, cluster, cluster2)) || throw(ArgumentError(
        "group identifiers do not add random effects; supply explicit grouping=[GroupingTerm(...)]"))
    # gllvm's `num.lv` alias for K. If both given they must agree.
    if num_lv !== nothing
        if K !== nothing && K != num_lv
            throw(ArgumentError("fit_gllvm: K=$K and num_lv=$num_lv disagree; pass only one"))
        end
        K = num_lv
    end

    # API B (Curie): NB/Beta/NB1/BetaBinom public default matches gllvmTMB per-trait
    # φ — the estimand already shipped on the R bridge and on `@formula` with X, so
    # the unified entry point must not default to a shared scalar instead. Shared-φ
    # engines remain `fit_nb_gllvm` / `fit_beta_gllvm` / `fit_nb1_gllvm` /
    # `fit_beta_binomial_gllvm`. The NB1 and BetaBinom markers' `φ` fields are never
    # read: φ is always estimated. Gamma unchanged.
    if disp_group === nothing &&
       (family isa NegativeBinomial || family isa Beta || family isa NB1 ||
        family isa BetaBinom || (family isa StudentTFamily && family.ν === nothing))
        disp_group = :species
    end
    # PASTE `accept delta dispersion A`: add DeltaLogNormal / DeltaGamma to the coerce
    # block above (disp_group = :species when nothing), then forward disp_group into
    # fit_delta_* via kwargs below — public twin default must not flip before paste.

    # --- Multinomial v1: FE softmax only (no LV). ----------------------------
    # Must run before row_eff / disp_group / pervar routes (those require K).
    if family isa Multinomial
        row_eff === :none || throw(ArgumentError(
            "fit_gllvm: Multinomial v1 is fixed-effects softmax only — " *
            "row_eff is not supported"))
        disp_group === nothing || throw(ArgumentError(
            "fit_gllvm: Multinomial has no dispersion — disp_group is not supported"))
        pervar && throw(ArgumentError(
            "fit_gllvm: Multinomial v1 is fixed-effects softmax only — pervar is not supported"))
        (K !== nothing && K != 0) && throw(ArgumentError(
            "fit_gllvm: Multinomial v1 is fixed-effects softmax only — no LV " *
            "(got K=$K). Leave K / num_lv unset."))
        return fit_multinomial_gllvm(Y; kwargs...)
    end

    # Count how many structural/dispersion variants are active. At most one is
    # supported, since each routes to a distinct single-purpose fitter.
    nvariants = (row_eff !== :none) + (disp_group !== nothing) + pervar
    if nvariants > 1
        throw(ArgumentError(
            "fit_gllvm: combination of row_eff=:$(row_eff), " *
            "disp_group=$(disp_group === nothing ? "nothing" : "set"), pervar=$pervar " *
            "is not yet supported — at most one of row_eff / disp_group / pervar may be active"))
    end

    # --- pervar: heteroscedastic Gaussian. -----------------------------------
    if pervar
        family isa Normal || throw(ArgumentError(
            "fit_gllvm: pervar=true is only supported for family=Normal() " *
            "(got $(nameof(typeof(family))))"))
        K === nothing && throw(ArgumentError("fit_gllvm: K (or num_lv) is required"))
        return fit_gaussian_pervar_gllvm(Y; K = K, kwargs...)
    end

    # --- row_eff: community (fixed) or random row effect. --------------------
    if row_eff !== :none
        K === nothing && throw(ArgumentError("fit_gllvm: K (or num_lv) is required"))
        if row_eff === :fixed
            return fit_roweffect_gllvm(Y; family = family, K = K, kwargs...)
        elseif row_eff === :random
            return fit_row_random_gllvm(Y; family = family, K = K, kwargs...)
        else
            throw(ArgumentError(
                "fit_gllvm: row_eff must be :none, :fixed, or :random (got :$(row_eff))"))
        end
    end

    # --- Delta two-part: disp_group is shared vs per-trait σ/α (not NB grouped routing). ---
    if disp_group !== nothing && (family isa DeltaLogNormal || family isa DeltaGamma)
        K === nothing && throw(ArgumentError("fit_gllvm: K (or num_lv) is required"))
        disp_group in (:shared, :species) || throw(ArgumentError(
            "fit_gllvm: Delta disp_group must be :shared or :species (got :$disp_group)"))
        return _fit_gllvm(family, Y; K = K, disp_group = disp_group, kwargs...)
    end

    # --- disp_group: grouped / species-specific dispersion. ------------------
    if disp_group !== nothing
        K === nothing && throw(ArgumentError("fit_gllvm: K (or num_lv) is required"))
        p = size(Y, 1)
        group = if disp_group === :species
            collect(1:p)
        elseif disp_group isa AbstractVector{<:Integer}
            collect(disp_group)
        else
            throw(ArgumentError(
                "fit_gllvm: disp_group must be :species or a length-p Int vector " *
                "(got $(typeof(disp_group)))"))
        end
        return _fit_gllvm_grouped(family, Y; K = K, group = group, kwargs...)
    end

    # --- default: family dispatch (unchanged behaviour). ---------------------
    K === nothing ? _fit_gllvm(family, Y; kwargs...) :
                    _fit_gllvm(family, Y; K = K, kwargs...)
end

_fit_gllvm(::Normal,   Y::AbstractMatrix; kwargs...) = fit_gaussian_gllvm(Y; kwargs...)
_fit_gllvm(::Binomial, Y::AbstractMatrix; kwargs...) = fit_binomial_gllvm(Y; kwargs...)
_fit_gllvm(::Poisson,  Y::AbstractMatrix; kwargs...) = fit_poisson_gllvm(Y; kwargs...)
_fit_gllvm(::TruncatedPoisson, Y::AbstractMatrix; kwargs...) =
    fit_truncated_poisson_gllvm(Y; kwargs...)
_fit_gllvm(::CensoredPoisson, Y::AbstractMatrix; kwargs...) =
    fit_censored_poisson_gllvm(Y; kwargs...)
_fit_gllvm(::Lognormal, Y::AbstractMatrix; kwargs...) =
    fit_lognormal_gllvm(Y; kwargs...)
_fit_gllvm(::TruncatedNegBin2, Y::AbstractMatrix; kwargs...) =
    fit_truncated_nbinom2_gllvm(Y; kwargs...)
_fit_gllvm(::NegativeBinomial, Y::AbstractMatrix; kwargs...) = fit_nb_gllvm(Y; kwargs...)
_fit_gllvm(::Beta,     Y::AbstractMatrix; kwargs...) = fit_beta_gllvm(Y; kwargs...)
_fit_gllvm(::Ordinal,  Y::AbstractMatrix; kwargs...) = fit_ordinal_gllvm_pertrait(Y; kwargs...)
_fit_gllvm(::Gamma,    Y::AbstractMatrix; kwargs...) = fit_gamma_gllvm(Y; kwargs...)
_fit_gllvm(::Exponential, Y::AbstractMatrix; kwargs...) = fit_exponential_gllvm(Y; kwargs...)
# `StudentTFamily` carries the degrees-of-freedom policy: numeric ν is fixed;
# nothing requests estimation. This model control travels on
# the family instance (as `ZIB`'s trials count does) and is forwarded as `nu`. A
# separate `nu` keyword would silently win over the marker (Julia resolves a
# duplicated keyword in favour of the splatted one), so it is rejected rather than
# left to contradict `family` in silence. The marker's `σ` is a tag payload: the
# scale is always estimated, seeded only by `σ_init`.
function _fit_gllvm(family::StudentTFamily, Y::AbstractMatrix; kwargs...)
    haskey(kwargs, :nu) && throw(ArgumentError(
        "fit_gllvm: the Student-t degrees of freedom travels on the family marker — " *
        "pass family = StudentTFamily($(kwargs[:nu])) rather than a separate nu keyword"))
    return fit_studentt_gllvm(Y; nu = family.ν, kwargs...)
end
# Delta markers are tag-payload markers: σ / α are always estimated by the named
# fitters and are never read from the family instance (same pattern as NB1(φ)).
_fit_gllvm(::DeltaLogNormal, Y::AbstractMatrix; disp_group = :shared, kwargs...) =
    fit_delta_lognormal_gllvm(Y; disp_group = disp_group, kwargs...)
_fit_gllvm(::DeltaGamma, Y::AbstractMatrix; disp_group = :shared, kwargs...) =
    fit_delta_gamma_gllvm(Y; disp_group = disp_group, kwargs...)
_fit_gllvm(::GeneralizedPoisson1, Y::AbstractMatrix; kwargs...) = fit_gp1_gllvm(Y; kwargs...)
_fit_gllvm(::ZIPoisson, Y::AbstractMatrix; kwargs...) = fit_zip_gllvm(Y; kwargs...)
_fit_gllvm(::ZINegBin, Y::AbstractMatrix; kwargs...) = fit_zinb_gllvm(Y; kwargs...)
# `ZIB` is not a zero-arg marker: the shared scalar trials count travels on the
# family instance (`ZIB(N)`), so it is forwarded here rather than taken as a kwarg.
_fit_gllvm(family::ZIB, Y::AbstractMatrix; kwargs...) =
    fit_zib_gllvm(Y; N = family.N, kwargs...)
# Empty two-part marker (same shape as `ZIPoisson`): no payload to forward.
_fit_gllvm(::HurdlePoisson, Y::AbstractMatrix; kwargs...) =
    fit_hurdle_poisson_gllvm(Y; kwargs...)
# Hurdle-NB marker is a tag-payload marker: r is always estimated by the
# named fitter and is never read from the family instance (same pattern as
# COMPoisson(ν) / DeltaGamma(α); opposite of StudentTFamily(ν)).
_fit_gllvm(::HurdleNB, Y::AbstractMatrix; kwargs...) =
    fit_hurdle_nb_gllvm(Y; kwargs...)
# Beta-hurdle marker is a tag-payload marker: φ is always estimated by the
# named fitter and is never read from the family instance (same pattern as
# HurdleNB(r) / COMPoisson(ν); opposite of StudentTFamily(ν)).
_fit_gllvm(::BetaHurdle, Y::AbstractMatrix; kwargs...) =
    fit_beta_hurdle_gllvm(Y; kwargs...)
# COM-Poisson marker is a tag-payload marker: ν is always estimated by the
# named fitter and is never read from the family instance (same pattern as
# NB1(φ) / DeltaGamma(α); opposite of StudentTFamily(ν), which is structural).
_fit_gllvm(::COMPoisson, Y::AbstractMatrix; kwargs...) =
    fit_compoisson_gllvm(Y; kwargs...)
# Ordered-beta marker is a three-field tag-payload marker: c0, c1, and φ are
# always estimated by the named fitter and are never read from the family
# instance (same pattern as COMPoisson(ν) / BetaHurdle(φ); opposite of
# StudentTFamily(ν) and of Ordinal's τ₁ = 0 pin).
_fit_gllvm(::OrderedBeta, Y::AbstractMatrix; kwargs...) =
    fit_ordered_beta_gllvm(Y; kwargs...)
_fit_gllvm(::Multinomial, Y::AbstractMatrix; kwargs...) =
    fit_multinomial_gllvm(Y; kwargs...)

# No `_fit_gllvm(::NB1, …)` / `_fit_gllvm(::BetaBinom, …)` arms: the per-trait
# coerce above always sets `disp_group`, so both reach `_fit_gllvm_grouped`
# instead. A bare arm here would be unreachable and would advertise the shared-φ
# estimand, which is available only through the named `fit_nb1_gllvm` /
# `fit_beta_binomial_gllvm`.

# Clear error for families not yet implemented.
_fit_gllvm(family, Y::AbstractMatrix; kwargs...) = throw(ArgumentError(
    "fit_gllvm: family $(nameof(typeof(family))) is not implemented yet " *
    "(available: Normal, Binomial, Poisson, TruncatedPoisson, CensoredPoisson, TruncatedNegBin2, Lognormal, Multinomial, NegativeBinomial, NB1, Beta, BetaBinom, Ordinal, Gamma, Exponential, StudentTFamily, DeltaLogNormal, DeltaGamma, HurdlePoisson, HurdleNB, BetaHurdle, COMPoisson, OrderedBeta, GeneralizedPoisson1, ZIPoisson, ZINegBin, ZIB)"))

# --- grouped-dispersion routing keyed on the family marker. ------------------
function _fit_gllvm_grouped(::TruncatedNegBin2, Y::AbstractMatrix; group, kwargs...)
    p = size(Y, 1)
    length(group) == p || throw(ArgumentError(
        "fit_gllvm: TruncatedNegBin2 needs one dispersion group ID per trait (length $p)"))
    all(>(0), group) || throw(ArgumentError(
        "fit_gllvm: TruncatedNegBin2 dispersion group IDs must be positive"))
    length(unique(group)) == p || throw(ArgumentError(
        "fit_gllvm: TruncatedNegBin2 supports only distinct per-trait dispersion groups; " *
        "use disp_group = :species, or omit disp_group for shared dispersion"))
    return fit_truncated_nbinom2_gllvm_pertrait(Y; kwargs...)
end

_fit_gllvm_grouped(::NegativeBinomial, Y::AbstractMatrix; kwargs...) =
    fit_nb_gllvm_grouped(Y; kwargs...)
_fit_gllvm_grouped(::Beta,  Y::AbstractMatrix; kwargs...) = fit_beta_gllvm_grouped(Y; kwargs...)
_fit_gllvm_grouped(::Gamma, Y::AbstractMatrix; kwargs...) = fit_gamma_gllvm_grouped(Y; kwargs...)
_fit_gllvm_grouped(::NB1,   Y::AbstractMatrix; kwargs...) = fit_nb1_gllvm_grouped(Y; kwargs...)
_fit_gllvm_grouped(::TweedieED, Y::AbstractMatrix; kwargs...) =
    fit_tweedie_gllvm_grouped(Y; kwargs...)
_fit_gllvm_grouped(family::StudentTFamily, Y::AbstractMatrix; group = nothing, kwargs...) =
    fit_studentt_gllvm(Y; nu = family.ν, disp_group = :species, kwargs...)

# Beta-binomial: the trial counts are required here, unlike in the named fitter.
# `fit_beta_binomial_gllvm_grouped` defaults `N === nothing` to all-ones, but at
# N = 1 the beta-binomial collapses to `Bernoulli(μ)` and φ is unidentifiable —
# the log-density is flat in φ to roundoff. Inheriting that default at a public
# entry point would hand back a per-trait φ vector the likelihood cannot inform,
# with no warning, so a missing `N` is an error naming the keyword. A scalar `N`
# is likewise rejected rather than broadcast: shaping data is the family file's
# job, not the dispatcher's.
function _fit_gllvm_grouped(::BetaBinom, Y::AbstractMatrix; N = nothing, kwargs...)
    p, n = size(Y)
    N === nothing && throw(ArgumentError(
        "fit_gllvm: family BetaBinom requires the trial counts `N` as a $(p)×$(n) " *
        "matrix (φ is unidentifiable at N = 1, so there is no safe default) — " *
        "call fit_gllvm(Y; family = BetaBinom(), K = …, N = N)"))
    N isa AbstractMatrix || throw(ArgumentError(
        "fit_gllvm: family BetaBinom needs `N` as a $(p)×$(n) matrix, got " *
        "$(typeof(N)); a scalar is not broadcast here — pass fill(N, $p, $n)"))
    return fit_beta_binomial_gllvm_grouped(Y; N = N, kwargs...)
end

# Families without a grouped-dispersion fitter.
_fit_gllvm_grouped(family, Y::AbstractMatrix; kwargs...) = throw(ArgumentError(
    "fit_gllvm: disp_group (grouped dispersion) is not supported for family " *
    "$(nameof(typeof(family))) — available: NegativeBinomial, Beta, Gamma, NB1, BetaBinom, Tweedie (TweedieED), StudentTFamily, TruncatedNegBin2 (per-trait only)"))
