# Response-domain validation shared by the Laplace fitters (#730).
#
# Without it an out-of-support cell makes the objective -Inf and the fitter returns a
# fit with `loglik = -Inf, converged = false`; a non-integer or NaN count fails late with
# an `InexactError` that does not mention the data. Only observed cells are checked, so
# `missing` entries and `mask = false` cells (the documented NA route) are untouched.

# Does the response `y` lie in the support named by `kind`?
function _in_support(kind::Symbol, y, Nti)
    y isa Real && isfinite(y) || return false
    kind === :beta     && return 0 < y < 1
    kind === :positive && return y > 0
    kind === :finite   && return true
    kind === :nonneg_real && return y >= 0
    kind === :unit_closed && return 0 <= y <= 1
    (y >= 0 && y == round(y)) || return false      # :count and :binomial
    return kind === :binomial ? y <= Nti : true
end

function _support_text(kind::Symbol)
    kind === :beta     && return "Beta support, the open interval (0, 1)"
    kind === :positive && return "positive support, y > 0"
    kind === :finite   && return "real support, a finite number"
    kind === :nonneg_real && return "non-negative real support, y >= 0"
    kind === :unit_closed && return "closed unit interval [0, 1]"
    kind === :binomial && return "Binomial support, an integer 0 <= y <= N"
    return "Poisson support, a non-negative integer"
end

# _check_response_support(kind, Y, mask; caller, N = nothing)
#
# Throw an `ArgumentError` naming `caller`, the first offending cell, the number of
# offending cells and the allowed support when an observed cell of `Y` is outside the
# support for `kind` (`:count`, `:binomial` with trials `N`, `:beta`, `:positive`,
# `:nonneg_real`, `:unit_closed` or `:finite`).
# Non-finite cells count as offending; mask them or use `missing` for unobserved cells.
# `mask = nothing` means every non-`missing` cell is observed.
function _check_response_support(kind::Symbol, Y::AbstractMatrix, mask;
        caller::AbstractString, N = nothing, row0::Integer = 0)
    p, n = size(Y)
    bad = 0
    first_bad = (0, 0)
    @inbounds for i in 1:n, t in 1:p
        (mask === nothing || mask[t, i]) || continue
        y = Y[t, i]
        ismissing(y) && continue
        if !_in_support(kind, y, N === nothing ? 1 : N[t, i])
            bad += 1
            bad == 1 && (first_bad = (t, i))
        end
    end
    bad == 0 && return nothing
    t, i = first_bad
    extra = kind === :binomial ? " with N[$(t + row0), $i] = $(N === nothing ? 1 : N[t, i])" : ""
    throw(ArgumentError("$caller: Y[$(t + row0), $i] = $(Y[t, i]) is outside the " *
        _support_text(kind) * extra * " ($bad observed cell(s) in total). " *
        "Fix the data, or mark unobserved cells `missing` / `mask = false`; " *
        "NaN and Inf are not valid responses."))
end

# Support kind for the families the model-structure fitters share (`_cov_default_link`
# lists them); `nothing` for a family this table does not know, which is left unchecked.
function _family_support_kind(family)
    (family isa Poisson || family isa NegativeBinomial || family isa NB1) && return :count
    family isa Binomial && return :binomial
    family isa Beta && return :beta
    (family isa Gamma || family isa Exponential) && return :positive
    return nothing
end

# `_family_support_kind` plus `Normal` (any finite real), for the fitters whose family set
# includes a Gaussian response (SPDE latent, coevolution, mixed-family).
_family_support_kind_normal(family) =
    family isa Normal ? :finite : _family_support_kind(family)

# `_check_response_support` for a fitter that takes `family`; `N` is the trials matrix
# (or `nothing`, meaning one trial) and is used for `Binomial` only.
function _check_family_response_support(family, Y::AbstractMatrix, mask;
        caller::AbstractString, N = nothing)
    kind = _family_support_kind(family)
    kind === nothing && return nothing
    return _check_response_support(kind, Y, mask; caller = caller, N = N)
end

# Same as `_check_family_response_support` but a `Normal` family is also checked (finite).
function _check_family_response_support_normal(family, Y::AbstractMatrix, mask;
        caller::AbstractString, N = nothing)
    kind = _family_support_kind_normal(family)
    kind === nothing && return nothing
    return _check_response_support(kind, Y, mask; caller = caller, N = N)
end

# Mixed-family fitter: one family per row of `Y`, checked row by row so the message names
# the true trait row. `N` is the p x n trials matrix (or `nothing`).
function _check_mixed_response_support(families, Y::AbstractMatrix, N;
        caller::AbstractString)
    for t in axes(Y, 1)
        kind = _family_support_kind_normal(families[t])
        kind === nothing && continue
        _check_response_support(kind, view(Y, t:t, :), nothing; caller = caller,
                                N = N === nothing ? nothing : view(N, t:t, :), row0 = t - 1)
    end
    return nothing
end

# Like `_sanitize_missing`, but also replaces the cells that `mask` marks unobserved, so a
# `NaN` (or any out-of-support) value in a masked-out cell never reaches an
# `Integer.(...)` conversion, the warm start or the objective. The result always has a
# concrete element type: `Int` data with a `Float64` placeholder, or a `Matrix{Any}`
# holding `missing` (e.g. from `readdlm`), are narrowed to the promoted value type, as the
# objectives cannot work on an abstract element type.
function _sanitize_unobserved(Y::AbstractMatrix, mask, placeholder)
    mask === nothing && return _sanitize_missing(Y, placeholder)
    out = [(mask[t, i] && !ismissing(Y[t, i])) ? Y[t, i] : placeholder
           for t in axes(Y, 1), i in axes(Y, 2)]
    isconcretetype(eltype(out)) && return out
    return convert(Matrix{mapreduce(typeof, promote_type, out; init = typeof(placeholder))}, out)
end

# _check_finite_covariates(X, caller; obs = nothing, unit = :cell, name = "X")
#
# Covariates must be finite where they are used. A NaN (or `missing`, or Inf) in `X` used
# to give an objective of NaN or -Inf, so the fitter returned `loglik = -Inf` and
# `converged = false` with no message. This throws an `ArgumentError` naming `caller`, the
# array, the first offending entry and the number of offending entries.
#
# `obs` is the observation mask (`nothing` = every cell observed). With `unit = :cell` the
# array is `p x n x q` and only the entries at observed cells are checked, so a covariate
# value hidden by `mask = false` / `missing` stays allowed. With `unit = :site` the array
# is `n x q` and a row is checked when its site has an observed cell; with `unit = :trait`
# it is `p x r` and a row is checked when its trait has one. A mask whose size does not
# match the array (3 x 29 against a 5 x 24 response, say) is a `DimensionMismatch`, raised
# before the scan reads it; the scan is bounds-checked as well.
function _check_finite_covariates(X::AbstractArray, caller::AbstractString;
        obs = nothing, unit::Symbol = :cell, name::AbstractString = "X")
    if obs !== nothing
        want = unit === :cell ? (size(X, 1), size(X, 2)) :
               unit === :site ? (nothing, size(X, 1)) : (size(X, 1), nothing)
        ok = ndims(obs) == 2 && (unit === :cell ? size(obs) == want :
             unit === :site ? size(obs, 2) == want[2] : size(obs, 1) == want[1])
        ok || throw(DimensionMismatch(
            "$caller: mask is $(size(obs)) but $name is $(size(X)); `mask` must have the shape " *
            "of Y (traits × units, true = observed) and match $name " *
            (unit === :cell ? "in its first two dimensions." :
             unit === :site ? "in its rows (units)." : "in its rows (traits).")))
    end
    bad = 0
    first_bad = nothing
    for I in CartesianIndices(X)
        x = X[I]
        (x isa Real && isfinite(x)) && continue
        if obs !== nothing
            seen = unit === :cell ? obs[I[1], I[2]] :
                   unit === :site ? any(view(obs, :, I[1])) : any(view(obs, I[1], :))
            seen || continue
        end
        bad += 1
        first_bad === nothing && (first_bad = I)
    end
    bad == 0 && return nothing
    idx = join(Tuple(first_bad), ", ")
    loc = obs === nothing ? "" : " at observed cells"
    hint = obs === nothing ? "Fix or impute the covariate." :
        "Fix or impute the covariate, or mark the response cell unobserved (`missing` / `mask = false`)."
    throw(ArgumentError("$caller: $name[$idx] = $(X[first_bad]) is not a finite number " *
        "($bad non-finite entr$(bad == 1 ? "y" : "ies")$loc). " *
        "Covariates must be finite; NaN, `missing` and Inf are not accepted. " * hint))
end

# All-zero count columns (ruling on constant columns, second sentence). A species whose
# observed cells are all zero has no finite maximum-likelihood intercept: the fit runs to a
# large negative value and still reports `converged = true`. The Poisson and ZIP fitters call
# this once per public call; it only warns (no number changes). A column with even one non-zero
# observed cell, a column with no observed cell at all, and sparse inputs are left alone.
# `maxlog = 1` as the NB2 Poisson-limit warning, but the logger's repeat counter is keyed by
# `_id`, which here is the fitter name and the exact column list written out (a Symbol made
# from that text: equal only when the text is equal, so two different column sets can never
# share a counter, and the same text gives the same id in every session and Julia version). A console therefore shows
# one message per distinct (fitter, columns) per session, not one per session.
# Species (row) indices whose observed cells are all zero counts.
function _all_zero_count_column_indices(Y::AbstractMatrix, mask)
    p, n = size(Y)
    cols = Int[]
    @inbounds for t in 1:p
        nobs = 0
        allzero = true
        for i in 1:n
            (mask === nothing || mask[t, i]) || continue
            y = Y[t, i]
            ismissing(y) && continue
            nobs += 1
            if y != 0
                allzero = false
                break
            end
        end
        (allzero && nobs > 0) && push!(cols, t)
    end
    return cols
end

# True when every species row has only zero counts in observed cells (degenerate intercept).
_all_species_all_zero_counts(Y::AbstractMatrix, mask) =
    (p = size(Y, 1); p > 0 && length(_all_zero_count_column_indices(Y, mask)) == p)

function _warn_all_zero_count_columns(Y::AbstractMatrix, mask; caller::AbstractString)
    cols = _all_zero_count_column_indices(Y, mask)
    isempty(cols) && return nothing
    # ZIP only: the zero-inflation logit can also absorb the column, so it runs off too (observed on 24 fits:
    # logit +13 to +1600 and intercept -12 to -1500, with `converged = true`).
    zinote = caller == "fit_zip_gllvm" ? " The zero-inflation term of such a species also runs off to a large value, so it is not identified either." : ""
    @warn "$caller: species column(s) $cols have no non-zero count in any observed cell. The fit still runs, but the intercept of such a species has no finite maximum-likelihood value (it ends at a large negative number set by where the optimiser stops), so `converged = true` does not mean it is identified.$zinote Drop the column or treat its estimates as not identified." maxlog=1 _id=Symbol("all_zero_count_columns:", caller, ":", join(cols, ","))
    return nothing
end
