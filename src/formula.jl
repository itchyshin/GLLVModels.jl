# @formula front-end — a pre-processor mapping gllvmTMB-style syntax and StatsModels
# formulas onto the matrix-level engine. Slices 1–2 of the formula-front-end design spec
# (docs/superpowers/specs/2026-05-31-formula-frontend-random-slopes-design.md).
#
#     gllvm(@formula(y ~ 1 + temp + depth), Y, site_data; family = Poisson(), K = 2)
#     gllvm(@formula(y ~ 1 + temp + habitat), Y, site_data; contrasts = Dict(:habitat => DummyCoding()))
#
# Mapping (verified against the engine contract):
#  - The intercept `1` is the engine's BUILT-IN per-species intercept (the Gaussian
#    path profiles out the per-trait row mean — src/likelihood.jl:39; fit_gllvm_cov
#    carries explicit per-species β). So the `1` term is dropped here, not put into X.
#  - Site-level covariates (continuous, categorical contrasts via StatsModels, function
#    terms, interactions) become columns of the engine's (p, n, q) design X, broadcast
#    across species (X[t,s,k] = covariate[s,k]) ⇒ a coefficient SHARED across species.
#  - Dispatch: Normal() → fit_gaussian_gllvm(Y; X); other families → fit_gllvm_cov /
#    specialised grouped-cov fitters.
#
# StatsModels integration supports continuous covariates, categorical contrasts
# (DummyCoding, EffectsCoding, HelmertCoding, etc.), function transformations (e.g. `log(x)`),
# and interaction terms across site-level variables.
#
# StatsModels is imported SELECTIVELY so it does not bring StatsAPI's
# `predict`/`residuals`/`fit` into the module and clash with GLLVModels's own post-fit generics.

import StatsModels
using StatsModels: @formula, FormulaTerm, Term, ConstantTerm, FunctionTerm, InteractionTerm, schema, apply_schema, modelmatrix, coefnames
import Tables

struct _FormulaKUnset end
const _FORMULA_K_UNSET = _FormulaKUnset()

function _extract_formula_symbols(term)
    syms = Symbol[]
    if term isa Term
        push!(syms, term.sym)
    elseif term isa FunctionTerm
        for arg in term.args
            append!(syms, _extract_formula_symbols(arg))
        end
    elseif term isa InteractionTerm
        for t in term.terms
            append!(syms, _extract_formula_symbols(t))
        end
    elseif term isa Tuple
        for t in term
            append!(syms, _extract_formula_symbols(t))
        end
    end
    return syms
end

# Extract site-level model matrix mm (n × q) and coefficient names from formula RHS.
function _build_site_modelmatrix(rhs, data; contrasts::AbstractDict = Dict{Symbol, Any}())
    cols = Tables.columntable(data)
    ts = rhs isa Tuple ? rhs : (rhs,)
    non_const = [t for t in ts if !(t isa ConstantTerm)]
    if isempty(non_const)
        n = length(Tables.rows(cols))
        return zeros(Float64, n, 0), String[]
    end

    # Validate that required symbols exist in data
    for t in non_const
        syms = _extract_formula_symbols(t)
        for s in syms
            haskey(cols, s) || throw(ArgumentError(
                "covariate `$s` in the formula is not a column of `data`"))
        end
    end

    rhs_term = length(non_const) == 1 ? non_const[1] : Tuple(non_const)
    f_cov = FormulaTerm(ConstantTerm(0), rhs_term)
    sch = StatsModels.schema(f_cov, cols, contrasts)
    applied = StatsModels.apply_schema(f_cov, sch)
    mm = StatsModels.modelmatrix(applied.rhs, cols)
    cnames = StatsModels.coefnames(applied.rhs)
    cnames_vec = cnames isa AbstractVector ? string.(cnames) : [string(cnames)]
    return Matrix{Float64}(mm), cnames_vec
end

# The per-variance Gaussian fitter treats X as the complete mean design. Use
# StatsModels' intercept/rank rules before expanding a site intercept to one
# coefficient per trait. Keep the legacy shared-variance route separate.
function _pervar_formula_design(rhs, cols, p, n; contrasts, names::Bool=false)
    intercept = !StatsModels.omitsintercept(rhs)
    site_names = String[]
    terms = rhs isa Tuple ? rhs : (rhs,)
    if all(t -> t isa ConstantTerm, terms)
        site = zeros(n, 0)
    else
        f = FormulaTerm(ConstantTerm(0), rhs)
        sch = StatsModels.schema(f, cols, contrasts)
        applied = StatsModels.apply_schema(f, sch, StatsModels.StatisticalModel)
        mm = Matrix{Float64}(StatsModels.modelmatrix(applied.rhs, cols))
        model_names = StatsModels.coefnames(applied.rhs)
        model_names = model_names isa AbstractVector ? string.(model_names) : [string(model_names)]
        intercept = StatsModels.hasintercept(applied.rhs)
        site_idx = findall(!=("(Intercept)"), model_names)
        site = mm[:, site_idx]
        site_names = model_names[site_idx]
    end
    size(site, 1) == n || throw(DimensionMismatch("formula design must have one row per site"))
    q0 = intercept ? p : 0
    X = zeros(p, n, q0 + size(site, 2))
    if intercept
        for t in 1:p
            X[t, :, t] .= 1
        end
    end
    for k in axes(site, 2), s in 1:n, t in 1:p
        X[t, s, q0 + k] = site[s, k]
    end
    if names
        coefficient_names = String[]
        append!(coefficient_names, ("trait_$(trait)" for trait in 1:p if intercept))
        append!(coefficient_names, site_names)
        return X, coefficient_names
    end
    return X
end

"""
    gllvm(formula, Y, data; family = Normal(), K, sources = nothing,
          contrasts = Dict(), kwargs...)

Fit a GLLVM from an R-`gllvmTMB`-style `@formula` over a wide species×site response
matrix `Y` (`p × n`) and a `Tables`-compatible `data` of **site-level** covariates
(one row per site = per column of `Y`).

```julia
using GLLVModels, Distributions, StatsModels
gllvm(@formula(y ~ 1 + temp + depth), Y, site_data; family = Normal(),  K = 2)
gllvm(@formula(y ~ 1 + temp + habitat), Y, site_data; family = Poisson(), K = 2,
      contrasts = Dict(:habitat => DummyCoding()))
```

The response symbol on the formula LHS (`y`) names the matrix `Y` and is otherwise
ignored. The intercept (`1`) is the engine's built-in per-species intercept; each
covariate column on the RHS (continuous, categorical via `contrasts`, interactions)
becomes a coefficient **shared across species** (the engine's `(p,n,q)` design).
Dispatches to [`fit_gaussian_gllvm`](@ref) for `Normal()`, to [`fit_nb_gllvm_grouped_cov`](@ref) /
[`fit_beta_gllvm_grouped_cov`](@ref) / [`fit_gamma_gllvm_grouped_cov`](@ref) /
[`fit_nb1_gllvm_grouped_cov`](@ref) / `fit_beta_binomial_gllvm_grouped_cov` for
NB2/NB1/Beta/Gamma/beta-binomial (per-trait φ/α + shared site-X; twin API B; the
beta-binomial route also threads a binomial-style `N` trial-count kwarg), to
[`fit_ordinal_gllvm_pertrait_cov`](@ref) for `Ordinal()` (per-trait cutpoints +
shared site-X), to [`fit_zip_gllvm_cov`](@ref) for `ZIPoisson()` (separate
`γz`/`γc`, `Λz=0`; Julia-forward), to [`fit_zinb_gllvm_cov`](@ref) for
`ZINegBin()` (separate `γz`/`γc`, `Λz=0`, shared scalar `r`; Julia-forward),
and to [`fit_gllvm_cov`](@ref) for the other non-Gaussian families (shared
dispersion + X). Returns that fitter's result.
For `Normal()`, `pervar=true` instead selects
[`fit_gaussian_pervar_gllvm`](@ref). This route constructs the complete mean:
`y ~ 1 + x` (or `y ~ x`) has trait-specific intercepts and a shared slope;
`y ~ 0 + x` or `y ~ -1 + x` removes those intercepts. `y ~ 0` is zero mean.
`fixed_residual_sd=c` passes an explicit fixed residual scale to this route;
it does not choose R's data-dependent scale automatically. Categorical contrasts
follow StatsModels' rank rules. Do not also supply `X` with a formula.
The default shared-variance route keeps its existing behavior.
With no covariates it reduces to the intercept-only fit. Supplied table columns
must still have one entry per site; an empty table is allowed for an
intercept-only formula because `Y` supplies the site count.
With explicit `sources`, only `Normal()` is admitted and `K`, `pervar=true`, and
an explicit `X` are rejected. The formula then supplies the complete source-model
mean design: trait-specific intercept columns (when present) and shared
site-level coefficients, using StatsModels' contrast, interaction, and rank
rules. This does not parse formula source terms, trees, pedigrees, meshes, or
random slopes.
With `grouping=[GroupingTerm(...), ...]`, the formula defines the complete mean.
Grouping identifiers may be vectors or symbols naming data columns, for example
`unit=:plot`. Grouped support covers Gaussian, Poisson-log, Binomial-logit,
Beta-logit and NB2-log responses; do not also supply `K`, `sources`, `X`, or
`pervar`. Unsupported links and covariance combinations fail explicitly.
With `phylo=precision`, the same complete-design route supports Gaussian
precision-only or precision-plus-grouping fits; `species_id=:tip_column`
resolves the observation-to-tip map from `data`. Use `phylo_rank` for the
phylogenetic rank, not `K`; ordinary joint terms currently require independent,
non-common trait variances. This does not open public R bridge admission.
`ZIB` through `@formula` is **no-X only** for now (bridge still OWED; ZIB+X formula
is fenced).
"""
function gllvm(formula::FormulaTerm, Y::AbstractMatrix, data;
               family = Normal(), K::Union{Integer, _FormulaKUnset} = _FORMULA_K_UNSET,
               sources = nothing, grouping=nothing,
               pervar::Bool = false,
               contrasts::AbstractDict = Dict{Symbol, Any}(), kwargs...)
    p, n = size(Y)
    cols = Tables.columntable(data)
    for (name, column) in pairs(cols)
        length(column) == n || throw(DimensionMismatch(
            "`data` column `$name` has $(length(column)) rows but Y has $n sites (columns)"))
    end
    if grouping !== nothing || get(kwargs, :phylo, nothing) !== nothing
        sources === nothing || throw(ArgumentError("do not combine grouping with explicit sources"))
        K === _FORMULA_K_UNSET || throw(ArgumentError("grouping terms own their ranks; do not supply K"))
        pervar && throw(ArgumentError("pervar=true is incompatible with explicit grouping"))
        haskey(kwargs, :X) && throw(ArgumentError("the grouping formula supplies the complete mean; do not supply X"))
        haskey(kwargs, :coefficient_names) && throw(ArgumentError("the formula supplies coefficient names"))
        X, coefficient_names = _pervar_formula_design(
            formula.rhs, cols, p, n; contrasts=contrasts, names=true)
        resolved = map(collect(pairs(kwargs))) do (key, value)
            if key in (:unit, :unit_obs, :cluster, :cluster2, :species_id) && value isa Symbol
                haskey(cols, value) || throw(ArgumentError("group column `$value` is absent from data"))
                key => getproperty(cols, value)
            else
                key => value
            end
        end
        return fit_gllvm(Y; family=family, grouping=grouping, X=X,
            coefficient_names=coefficient_names, resolved...)
    end
    if sources !== nothing
        family isa Normal || throw(ArgumentError("formula source models require family=Normal()"))
        K === _FORMULA_K_UNSET || throw(ArgumentError("do not supply K with explicit sources; each source owns its rank"))
        pervar && throw(ArgumentError("pervar=true is incompatible with explicit sources"))
        haskey(kwargs, :X) && throw(ArgumentError("do not supply X with a source formula; the formula defines the complete mean"))
        haskey(kwargs, :coefficient_names) && throw(ArgumentError("do not supply coefficient_names with a source formula; the formula supplies them"))
        X, coefficient_names = _pervar_formula_design(
            formula.rhs, cols, p, n; contrasts=contrasts, names=true)
        return fit_gaussian_sources(Y; sources=sources, X=X,
            coefficient_names=coefficient_names, kwargs...)
    end
    K === _FORMULA_K_UNSET && throw(UndefKeywordError(:K))
    if pervar
        family isa Normal || throw(ArgumentError("pervar=true requires family=Normal()"))
        haskey(kwargs, :X) && throw(ArgumentError("do not supply X with a pervar formula; the formula defines the complete mean"))
        X = _pervar_formula_design(formula.rhs, cols, p, n; contrasts=contrasts)
        return fit_gllvm(Y; family=family, K=K, pervar=true, X=X, kwargs...)
    end
    mm, cnames = _build_site_modelmatrix(formula.rhs, cols; contrasts = contrasts)
    q = size(mm, 2)

    if q == 0
        return family isa Normal ? fit_gaussian_gllvm(Y; K = K, kwargs...) :
               family isa ZIPoisson ? fit_zip_gllvm(Y; K = K, kwargs...) :
               family isa ZINegBin ? fit_zinb_gllvm(Y; K = K, kwargs...) :
               family isa ZIB ? fit_gllvm(Y; family = family, K = K, kwargs...) :
                                fit_gllvm(Y; family = family, K = K, kwargs...)
    end

    size(mm, 1) == n || throw(DimensionMismatch(
        "`data` has $(size(mm, 1)) rows but Y has $n sites (columns)"))

    X = Array{Float64, 3}(undef, p, n, q)
    @inbounds for k in 1:q, s in 1:n, t in 1:p
        X[t, s, k] = mm[s, k]
    end

    if family isa Normal
        return fit_gaussian_gllvm(Y; X = X, K = K, kwargs...)
    elseif family isa NegativeBinomial
        return fit_nb_gllvm_grouped_cov(Y; X = X, K = K, kwargs...)
    elseif family isa Beta
        return fit_beta_gllvm_grouped_cov(Y; X = X, K = K, kwargs...)
    elseif family isa Gamma
        return fit_gamma_gllvm_grouped_cov(Y; X = X, K = K, kwargs...)
    elseif family isa NB1
        return fit_nb1_gllvm_grouped_cov(Y; X = X, K = K, kwargs...)
    elseif family isa BetaBinom
        return fit_beta_binomial_gllvm_grouped_cov(Y; X = X, K = K, kwargs...)
    elseif family isa Ordinal
        return fit_ordinal_gllvm_pertrait_cov(Y; X = X, K = K, kwargs...)
    elseif family isa ZIPoisson
        return fit_zip_gllvm_cov(Y; X = X, K = K, kwargs...)
    elseif family isa ZINegBin
        return fit_zinb_gllvm_cov(Y; X = X, K = K, kwargs...)
    elseif family isa ZIB
        throw(ArgumentError(
            "ZIB through @formula is no-X only for now (pass `@formula(y ~ 1)`); " *
            "ZIB+X formula / bridge admission is still OWED"))
    else
        return fit_gllvm_cov(Y; family = family, X = X, K = K, kwargs...)
    end
end

"""
    gllvm(formula, long_data; family = Normal(), K, species = :species, site = :site,
          contrasts = Dict(), kwargs...)

Long-format (melted) front door: one row per `(species, site)` observation. Pivots
`long_data` to the wide `(Y, site_data)` representation and calls the wide
[`gllvm`](@ref) — so the two data shapes share one engine path.

```julia
# long_data has columns y, species, site, temp, habitat
gllvm(@formula(y ~ 1 + temp + habitat), long_data; family = Poisson(), K = 2,
      species = :species, site = :site, contrasts = Dict(:habitat => DummyCoding()))
```

The formula LHS names the response column; `species`/`site` name the grouping
keys (default `:species`/`:site`). `Y` is built in sorted species×site order, so
`gllvm(f, long)` and `gllvm(f, Y, site_data)` give **identical** fits (a tested
round-trip identity). Requires a **complete** species×site grid (no missing
cells) and site covariates that are **constant within site** (both validated with
a clear error), matching the wide-mode contract.
For explicit Gaussian `sources`, source projection rows must correspond to this
same sorted site order; the long route passes that order through to the wide
source fitter unchanged.
"""
function gllvm(formula::FormulaTerm, long_data; family = Normal(),
               K::Union{Integer, _FormulaKUnset} = _FORMULA_K_UNSET,
               species::Symbol = :species, site::Symbol = :site,
               contrasts::AbstractDict = Dict{Symbol, Any}(), kwargs...)
    cols = Tables.columntable(long_data)
    formula.lhs isa Term || throw(ArgumentError(
        "long-format gllvm needs a single response column on the formula LHS; got $(formula.lhs)"))
    rsym = formula.lhs.sym
    for key in (rsym, species, site)
        haskey(cols, key) || throw(ArgumentError("column `$key` not found in long data"))
    end
    spcol = getproperty(cols, species)
    stcol = getproperty(cols, site)
    ycol  = getproperty(cols, rsym)
    nrow = length(ycol)
    (length(spcol) == nrow && length(stcol) == nrow) ||
        throw(DimensionMismatch("response, species, and site columns must have equal length"))

    splevels = sort(unique(spcol)); stlevels = sort(unique(stcol))
    p = length(splevels); n = length(stlevels)
    spidx = Dict(v => i for (i, v) in enumerate(splevels))
    stidx = Dict(v => j for (j, v) in enumerate(stlevels))

    Y = Matrix{eltype(ycol)}(undef, p, n)
    filled = falses(p, n)
    @inbounds for r in 1:nrow
        i = spidx[spcol[r]]; j = stidx[stcol[r]]
        filled[i, j] && throw(ArgumentError(
            "duplicate (species, site) = ($(spcol[r]), $(stcol[r])) in long data"))
        Y[i, j] = ycol[r]; filled[i, j] = true
    end
    all(filled) || throw(ArgumentError(
        "long data is not a complete species×site grid (v1 requires every cell present; " *
        "missing-response handling is a separate capability)"))

    # Repeated terms in interactions share one site-level column. `unique`
    # preserves first occurrence, which keeps the formula's covariate order.
    syms = unique(_extract_formula_symbols(formula.rhs))
    site_data = if isempty(syms)
        NamedTuple()
    else
        vecs = map(syms) do cv
            haskey(cols, cv) || throw(ArgumentError("covariate `$cv` not found in long data"))
            col = getproperty(cols, cv)
            vals = Vector{eltype(col)}(undef, n)
            seen = falses(n)
            for r in 1:nrow
                j = stidx[stcol[r]]
                if seen[j]
                    isequal(vals[j], col[r]) || throw(ArgumentError(
                        "covariate `$cv` is not constant within site `$(stcol[r])` " *
                        "(a site-level covariate must repeat identically down the species axis)"))
                else
                    vals[j] = col[r]; seen[j] = true
                end
            end
            vals
        end
        NamedTuple{Tuple(syms)}(Tuple(vecs))
    end

    if K === _FORMULA_K_UNSET
        return gllvm(formula, Y, site_data; family=family, contrasts=contrasts, kwargs...)
    end
    return gllvm(formula, Y, site_data; family=family, K=K,
        contrasts=contrasts, kwargs...)
end

# ===========================================================================
# Structured-term recognizer (core070 formula-recognizer-spec, §2 Steps 0-6)
#
# STATUS: SPEC + lane implementation only. Formula grammar changes are
# maintainer-approval-required to merge (AGENTS.md merge authority: "any ...
# formula grammar change"; see docs/dev-log/core070/formula-recognizer-spec.md
# and docs/dev-log/core070/formula-recognizer-impl-notes.md). Nothing below
# is wired into the public `gllvm(formula, ...)` front door, and none of it
# is authorized to reach `main` without Shinichi's explicit approval.
#
# StatsModels' `@formula` macro does not understand the R `lhs | group` bar
# sugar (`@formula(y ~ indep(0 + trait | g))` errors at macro-expansion —
# verified against StatsModels 0.7; `|` has no Term method). This recognizer
# therefore walks *raw, unevaluated* Julia `Expr` trees — e.g.
# `:(indep(0 + trait | g, common = true))` — rather than `@formula`-produced
# `FormulaTerm`s. Callers assemble those Exprs themselves (`Meta.parse` or
# `:(...)` quoting).
# ===========================================================================

const _STRUCTURED_TERM_KINDS = (:dep, :indep, :scalar,
    :kernel_indep, :kernel_dep, :kernel_scalar, :kernel_latent, :kernel_unique)

"""
    SourceTermSpec

Parsed, not-yet-materialized structured source term recognized from a raw
formula `Expr` by [`GLLVModels._recognize_source_term`](@ref). `kind` is one of
`:dep, :indep, :scalar, :kernel_indep, :kernel_dep, :kernel_scalar,
:kernel_latent, :kernel_unique`. `group` names the grouping column (the bar
RHS). `common`/`unique` are literal booleans, `nothing` when not applicable
to `kind`. `name` defaults to `:source` (non-kernel kinds) or `:kernel`
(kernel kinds). `K` carries the *raw, unresolved* kernel-matrix expression
for `kernel_*` kinds (`nothing` otherwise); `d` is the requested rank for
`kernel_latent` (`nothing` otherwise). Internal and not exported.
"""
struct SourceTermSpec
    kind::Symbol
    group::Symbol
    common::Union{Bool,Nothing}
    unique::Union{Bool,Nothing}
    name::Symbol
    K::Union{Nothing,Expr,Symbol,QuoteNode}
    d::Union{Nothing,Int}
end

_is_structured_call(expr) = expr isa Expr && expr.head === :call &&
    !isempty(expr.args) && expr.args[1] isa Symbol && expr.args[1] in _STRUCTURED_TERM_KINDS

"""Port of R `.assert_no_augmented_lhs` (brms-sugar.R:2172-2215): the bar LHS
of a structured source term must be exactly `0 + trait` or `1`; anything else
(e.g. an augmented `1 + x | g`) aborts."""
function _assert_no_augmented_lhs(lhs, kind::Symbol)
    (lhs isa Integer && !(lhs isa Bool) && lhs == 1) && return nothing
    if lhs isa Expr && lhs.head === :call && length(lhs.args) == 3 &&
            lhs.args[1] === :+ && lhs.args[2] == 0 && lhs.args[3] === :trait
        return nothing
    end
    throw(ArgumentError("$(kind)(...) does not support an augmented random-effect " *
        "LHS (only `0 + trait | g` or `1 | g` are recognized); got `$lhs`"))
end

"""Port of R `.read_common_flag` / the kernel `unique=` gate
(brms-sugar.R:2464-2483, :3364-3372): the flag must be a literal `true`/
`false`, never an expression or symbol."""
function _read_literal_flag(expr, key::Symbol)
    expr isa Bool || throw(ArgumentError("`$key` must be a literal `true` or `false`"))
    return expr
end

_read_optional_flag(kwargs::AbstractDict, key::Symbol, default::Bool) =
    haskey(kwargs, key) ? _read_literal_flag(kwargs[key], key) : default

function _literal_symbol(expr, key::Symbol)
    expr isa Symbol && return expr
    expr isa QuoteNode && expr.value isa Symbol && return expr.value
    expr isa String && return Symbol(expr)
    throw(ArgumentError("`$key` must be a literal name"))
end

function _literal_positive_int(expr, key::Symbol)
    expr isa Integer && !(expr isa Bool) && expr > 0 && return Int(expr)
    throw(ArgumentError("`$key` must be a positive integer literal"))
end

function _structured_call_kwargs(args)
    kwargs = Dict{Symbol,Any}()
    for a in args
        (a isa Expr && a.head === :kw && length(a.args) == 2) ||
            throw(ArgumentError("structured term arguments after the bar formula must be named (key = value); got `$a`"))
        key = a.args[1]::Symbol
        haskey(kwargs, key) && throw(ArgumentError("duplicate keyword `$key`"))
        kwargs[key] = a.args[2]
    end
    return kwargs
end

function _reject_unknown_kwargs(kwargs::AbstractDict, allowed, kind::Symbol)
    extra = setdiff(keys(kwargs), allowed)
    isempty(extra) || throw(ArgumentError("$(kind)(...) does not accept keyword(s) " *
        join(sort(String.(collect(extra))), ", ")))
end

const _KERNEL_TERM_KINDS = (:kernel_indep, :kernel_dep, :kernel_scalar, :kernel_latent, :kernel_unique)

"""
    _recognize_source_term(expr::Expr) -> SourceTermSpec

Recognize one structured source term from a raw `Expr`. `dep(...)`,
`indep(...)`, and `scalar(...)` take an R-style bar formula as their first
argument (`dep(0 + trait | g)`, matching R signatures `dep(formula)` /
`indep(form, common=FALSE)`); the bar is split off here and validated. The
`kernel_*(...)` siblings instead take a **bare grouping symbol** as their
first argument (R signature `kernel_latent(unit, K, d=1, name="kernel",
unique=FALSE)` — kernel-keywords.R:55-57 — no bar). Throws `ArgumentError` on
an unrecognized shape, an augmented LHS, or a non-literal flag. This helper is
internal and not exported.
"""
function _recognize_source_term(expr::Expr)
    _is_structured_call(expr) || throw(ArgumentError("not a recognized structured source term: `$expr`"))
    kind = expr.args[1]::Symbol
    rest = expr.args[2:end]
    isempty(rest) && throw(ArgumentError("$(kind)(...) requires a grouping argument"))
    if kind in _KERNEL_TERM_KINDS
        group_expr = rest[1]
        group_expr isa Symbol || throw(ArgumentError("$(kind)(...) requires a bare grouping symbol as its first argument; got `$group_expr`"))
        kwargs = _structured_call_kwargs(rest[2:end])
        return _build_source_term_spec(kind, group_expr::Symbol, kwargs)
    end
    bar = rest[1]
    (bar isa Expr && bar.head === :call && length(bar.args) == 3 && bar.args[1] === :(|)) ||
        throw(ArgumentError("$(kind)(...) requires a `lhs | group` bar expression as its first argument"))
    lhs, group_expr = bar.args[2], bar.args[3]
    _assert_no_augmented_lhs(lhs, kind)
    group_expr isa Symbol || throw(ArgumentError("$(kind)(...) group must be a bare column symbol; got `$group_expr`"))
    group = group_expr::Symbol
    kwargs = _structured_call_kwargs(rest[2:end])
    return _build_source_term_spec(kind, group, kwargs)
end

function _build_source_term_spec(kind::Symbol, group::Symbol, kwargs::AbstractDict)
    if kind === :dep
        _reject_unknown_kwargs(kwargs, (), kind)
        return SourceTermSpec(:dep, group, nothing, nothing, :source, nothing, nothing)
    elseif kind === :indep
        _reject_unknown_kwargs(kwargs, (:common,), kind)
        common = _read_optional_flag(kwargs, :common, false)
        return SourceTermSpec(:indep, group, common, nothing, :source, nothing, nothing)
    elseif kind === :scalar
        _reject_unknown_kwargs(kwargs, (), kind)
        _warn_scalar_deprecated_once()
        return SourceTermSpec(:scalar, group, true, nothing, :source, nothing, nothing)
    elseif kind in (:kernel_indep, :kernel_dep, :kernel_scalar, :kernel_latent, :kernel_unique)
        haskey(kwargs, :K) || throw(ArgumentError("$(kind)(...) requires a named `K` matrix"))
        K = kwargs[:K]
        name = haskey(kwargs, :name) ? _literal_symbol(kwargs[:name], :name) : :kernel
        if kind === :kernel_latent
            _reject_unknown_kwargs(kwargs, (:K, :d, :name, :unique), kind)
            d = haskey(kwargs, :d) ? _literal_positive_int(kwargs[:d], :d) : 1
            unique = _read_optional_flag(kwargs, :unique, false)
            return SourceTermSpec(:kernel_latent, group, nothing, unique, name, K, d)
        elseif kind === :kernel_indep
            _reject_unknown_kwargs(kwargs, (:K, :name, :common), kind)
            common = _read_optional_flag(kwargs, :common, false)
            return SourceTermSpec(:kernel_indep, group, common, nothing, name, K, nothing)
        elseif kind === :kernel_scalar
            _reject_unknown_kwargs(kwargs, (:K, :name), kind)
            return SourceTermSpec(:kernel_scalar, group, true, nothing, name, K, nothing)
        elseif kind === :kernel_dep
            _reject_unknown_kwargs(kwargs, (:K, :name), kind)
            return SourceTermSpec(:kernel_dep, group, nothing, nothing, name, K, nothing)
        else # :kernel_unique
            _reject_unknown_kwargs(kwargs, (:K, :name), kind)
            return SourceTermSpec(:kernel_unique, group, nothing, nothing, name, K, nothing)
        end
    end
    throw(ArgumentError("unrecognized structured term kind `$kind`"))
end

# One-shot deprecation warning mirroring R `.gllvmTMB_warn_scalar_family_deprecated`
# (brms-sugar.R:150-167, fired at :4170). Session-scoped, matching the R helper's
# once-per-session behaviour (no persistent option store on the Julia side).
const _SCALAR_DEPRECATION_WARNED = Ref(false)

function _warn_scalar_deprecated_once()
    if !_SCALAR_DEPRECATION_WARNED[]
        _SCALAR_DEPRECATION_WARNED[] = true
        @warn "scalar(...) is deprecated; use indep(..., common = true) instead"
    end
    return nothing
end

"""Port of the fit-multi.R exclusion-gate quartet (dep+latent same grouping
fit-multi.R:1642-1656; dep+unique :1657-1671; dep+indep :1672-1681;
indep+latent :1682-1695), generalized over this recognizer's kind vocabulary
(`kernel_indep`/`kernel_scalar` count as `indep`-family; `kernel_latent`
counts as `latent`; a `kernel_latent(unique=true)` term is the Julia
unique-folded analogue of R's separate `unique` term). Throws `ArgumentError`
naming both terms and the shared group on any forbidden pairing within one
grouping column. This helper is internal."""
function _check_source_term_exclusions(specs::Vector{SourceTermSpec})
    by_group = Dict{Symbol,Vector{Symbol}}()
    for s in specs
        push!(get!(by_group, s.group, Symbol[]), s.kind)
    end
    indep_family = (:indep, :scalar, :kernel_indep, :kernel_scalar)
    for (g, kinds) in by_group
        has_dep = :dep in kinds || :kernel_dep in kinds
        has_indep = any(k -> k in indep_family, kinds)
        has_latent = :kernel_latent in kinds
        has_unique = :kernel_unique in kinds ||
            any(s -> s.group === g && s.kind === :kernel_latent && s.unique === true, specs)
        has_dep && has_indep && throw(ArgumentError(
            "group `$g`: dep(...) and indep/scalar(...) on the same grouping are mutually exclusive (redundant covariance structure)"))
        has_dep && has_latent && throw(ArgumentError(
            "group `$g`: dep(...) and latent(...) (kernel_latent) on the same grouping are over-parameterised"))
        has_dep && has_unique && throw(ArgumentError(
            "group `$g`: dep(...) and a unique(-folded latent) term on the same grouping are mutually exclusive"))
        # indep + latent on one grouping is deliberately ALLOWED: it is the
        # diag + reduced-rank model (Sigma = K.*(LL') + diag), which the
        # frozen R oracle ACCEPTS (wave6-conversion6 receipt: the R fit for
        # indep + kernel_latent on `species` succeeded). The spec's original
        # Step-4 over-parameterisation guess contradicted the oracle and was
        # removed — observed oracle behavior is the contract.
    end
    return nothing
end

"""Resolve a recognized `K=` reference (a bare `Symbol`/`QuoteNode` captured
by [`GLLVModels._recognize_source_term`](@ref)) against `kernel_env` — a
`NamedTuple`/`Dict`-like environment supplied by the caller, mirroring R's
calling-environment lookup for `K=A`. Internal helper."""
function _resolve_kernel(K, kernel_env)
    key = K isa Symbol ? K :
          (K isa QuoteNode && K.value isa Symbol) ? K.value :
          throw(ArgumentError("K= must be a bare symbol naming a matrix in `kernel_env`; got `$K`"))
    env = kernel_env isa NamedTuple ? kernel_env : NamedTuple(kernel_env)
    haskey(env, key) || throw(ArgumentError("K=$key not found in kernel_env"))
    M = getproperty(env, key)
    M isa AbstractMatrix || throw(ArgumentError("K=$key must resolve to a matrix; got $(typeof(M))"))
    return M
end

"""
    _source_term_covariance(spec::SourceTermSpec, data; kernel_env=NamedTuple())
        -> SourceCovariance

Materialize a [`SourceCovariance`](@ref) (src/source_fit.jl) from a recognized
[`SourceTermSpec`](@ref). `data` supplies the grouping column named by
`spec.group`; its levels (sorted, matching `SourceCovariance(C; groups=...)`'s
one-based convention) define an identity/kernel covariance over the group
nodes with a one-hot projection built by unit row. `kernel_env` resolves
`K=` for `kernel_*` kinds. `:kernel_unique` has no standalone
`SourceCovariance` mode; use `kernel_latent(..., unique=true)` instead. This
helper is internal.
"""
function _source_term_covariance(spec::SourceTermSpec, data; kernel_env=NamedTuple())
    cols = Tables.columntable(data)
    haskey(cols, spec.group) || throw(ArgumentError("group column `$(spec.group)` not found in data"))
    gcol = getproperty(cols, spec.group)
    levels = sort(unique(gcol))
    level_index = Dict(v => i for (i, v) in enumerate(levels))
    ids = [level_index[v] for v in gcol]
    if spec.kind in (:indep, :dep, :scalar)
        C = Matrix{Float64}(I, length(levels), length(levels))
        mode = spec.kind === :dep ? :dep : :indep
        common = spec.kind === :scalar ? true : (spec.common === nothing ? false : spec.common)
        return mode === :dep ? SourceCovariance(C; groups=ids, name=spec.name, mode=:dep) :
            SourceCovariance(C; groups=ids, name=spec.name, mode=:indep, common=common)
    elseif spec.kind === :kernel_unique
        throw(ArgumentError("kernel_unique(...) has no standalone SourceCovariance mode; " *
            "fold it via kernel_latent(..., unique=true) (formula-recognizer-spec.md §1.4)"))
    else
        K = _resolve_kernel(spec.K, kernel_env)
        size(K, 1) == length(levels) || throw(DimensionMismatch(
            "$(spec.kind)(...) K has $(size(K,1)) rows but group `$(spec.group)` has $(length(levels)) levels"))
        if spec.kind === :kernel_latent
            return SourceCovariance(K; groups=ids, name=spec.name, mode=:latent,
                rank=spec.d, unique=(spec.unique === true))
        elseif spec.kind === :kernel_indep
            common = spec.common === nothing ? false : spec.common
            return SourceCovariance(K; groups=ids, name=spec.name, mode=:indep, common=common)
        elseif spec.kind === :kernel_scalar
            return SourceCovariance(K; groups=ids, name=spec.name, mode=:indep, common=true)
        else # :kernel_dep
            return SourceCovariance(K; groups=ids, name=spec.name, mode=:dep)
        end
    end
end

"""
    _fit_gaussian_structured_sources(Y, data, term_exprs; kernel_env=NamedTuple(), kwargs...)

Internal recognizer entry point: recognizes each raw structured-term `Expr` in `term_exprs` against
`data`, runs the Step 4 mutual-exclusion gates over the full set, materializes
their `SourceCovariance`s, and fits them via [`fit_gaussian_sources`](@ref).
**Not wired into the public `gllvm(formula, ...)` front door** — StatsModels'
`@formula` macro does not parse the `lhs | group` bar syntax these terms use
(see module header note above); surface integration is a separate grammar
extension. `kwargs` forward to `fit_gaussian_sources` (e.g. `X`,
`sigma_eps_fixed`, `start`). Not exported.
"""
function _fit_gaussian_structured_sources(Y::AbstractMatrix{<:Real}, data, term_exprs;
        kernel_env=NamedTuple(), kwargs...)
    specs = SourceTermSpec[_recognize_source_term(e) for e in term_exprs]
    _check_source_term_exclusions(specs)
    sources = SourceCovariance[_source_term_covariance(s, data; kernel_env=kernel_env) for s in specs]
    return fit_gaussian_sources(Y; sources=sources, kwargs...)
end

"""
    fit_gaussian_structured(Y, data; structure::Vector{Expr}, family=Normal(),
                             kernel_env=NamedTuple(), kwargs...)

Public entry point for the structured-term source grammar:
exposes the recognizers driven by `_fit_gaussian_structured_sources` through an
explicit `structure=` keyword, **not** a macro. This is a thin, documented
wrapper — it runs exactly the same recognizer/gate/fit pipeline as the internal
function: parse each `Expr` in `structure` with [`_recognize_source_term`](@ref),
run the Step-4 mutual-exclusion gates via [`_check_source_term_exclusions`](@ref),
materialize each spec's [`SourceCovariance`](@ref) via [`_source_term_covariance`](@ref),
and fit with [`fit_gaussian_sources`](@ref). Every named error raised inside that
pipeline (augmented-LHS rejection, non-literal flag rejection, unknown term name,
mutual-exclusion violations, PD-strictness on kernel matrices, ...) surfaces
unchanged through this wrapper.

# Why `Expr`, not `@formula`

StatsModels' `@formula` macro parses its right-hand side into `Term`s at
macro-expansion time and rejects the `lhs | group` bar syntax these structured
terms use (`indep(0 + trait | g)`, `dep(1 | grp)`, `kernel_latent(g, K=K, d=2)`,
...) before any GLLVModels code ever runs — the macro has no hook to recognize a
`|`-headed call as anything but a parse error. Passing raw, unevaluated `Expr`s
via `structure=` sidesteps `@formula` entirely: each `Expr` is quoted by the
caller with `:(...)` and walked by GLLVModels's own recognizer, so the grammar can
support the bar syntax without teaching StatsModels a new dialect. A macro
front door that lets users write `structure(indep(0 + trait | g))` directly
(instead of quoting) is a separate, later grammar decision — this wrapper does
not attempt it.

# Family support

Gaussian only, for now: `family` must be `Normal()` (`Distributions.Normal`),
matching every fitter this wrapper delegates to (`fit_gaussian_sources`, which
takes no `family` argument at all). Any other `family` value throws a named
`ArgumentError` rather than silently ignoring it.

# Arguments

- `Y::AbstractMatrix{<:Real}`: `p × n` response matrix (species × units).
- `data`: a Tables.jl-compatible table (or `NamedTuple` of vectors) holding the
  grouping columns referenced inside `structure`.
- `structure::Vector{Expr}`: one or more raw, quoted structured-term calls —
  `indep(...)`, `dep(...)`, `scalar(...)`, `kernel_indep(...)`,
  `kernel_scalar(...)`, `kernel_dep(...)`, or `kernel_latent(...)`.
- `family`: response family; only `Normal()` is accepted (default).
- `kernel_env=NamedTuple()`: named kernel matrices referenced by
  `kernel_*(...)` terms' `K=` keyword (e.g. `(K = phylogenetic_kernel,)`).
- `kwargs...`: forwarded to [`fit_gaussian_sources`](@ref) (e.g. `X`,
  `sigma_eps_fixed`, `start`, `g_tol`, `iterations`).

# Example

```jldoctest
julia> using GLLVModels, Random, LinearAlgebra

julia> rng = MersenneTwister(70100); p, n = 3, 24;

julia> g = repeat(1:6; inner = 4); data = (g = g,);

julia> Y = randn(rng, p, n) .+ [1.0, -0.5, 0.2];

julia> L = randn(rng, 6, 6); K = L * L' + 6.0 * Matrix(I, 6, 6);

julia> kernel_env = (K = K,);

julia> fit = fit_gaussian_structured(Y, data;
           structure = [:(indep(0 + trait | g)), :(kernel_latent(g, K = K, d = 1, name = "k1"))],
           kernel_env = kernel_env, sigma_eps_fixed = 0.5, g_tol = 1e-7);

julia> isfinite(fit.loglik)
true
```

See also [`_fit_gaussian_structured_sources`](@ref) (the internal function this
wraps), [`SourceCovariance`](@ref), [`fit_gaussian_sources`](@ref).

This is a direct matrix-and-data interface. It is not currently called by
`gllvm(formula, ...)`; use the documented `structure=` form shown above.
"""
function fit_gaussian_structured(Y::AbstractMatrix{<:Real}, data;
        structure::Vector{Expr}, family::Distribution=Normal(),
        kernel_env=NamedTuple(), kwargs...)
    family isa Normal || throw(ArgumentError(
        "fit_gaussian_structured supports family=Normal() only; got $(family)"))
    return _fit_gaussian_structured_sources(Y, data, structure; kernel_env=kernel_env, kwargs...)
end
