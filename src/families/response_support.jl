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
        caller::AbstractString, N = nothing)
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
    extra = kind === :binomial ? " with N[$t, $i] = $(N === nothing ? 1 : N[t, i])" : ""
    throw(ArgumentError("$caller: Y[$t, $i] = $(Y[t, i]) is outside the " *
        _support_text(kind) * extra * " ($bad observed cell(s) in total). " *
        "Fix the data, or mark unobserved cells `missing` / `mask = false`; " *
        "NaN and Inf are not valid responses."))
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

# All-zero count columns (ruling on constant columns, second sentence). A species whose
# observed cells are all zero has no finite maximum-likelihood intercept: the fit runs to a
# large negative value and still reports `converged = true`. The Poisson and ZIP fitters call
# this once per fit; it only warns (no number changes). A column with even one non-zero
# observed cell, a column with no observed cell at all, and sparse inputs are left alone.
# `maxlog = 1` as the NB2 Poisson-limit warning.
function _warn_all_zero_count_columns(Y::AbstractMatrix, mask; caller::AbstractString)
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
    isempty(cols) && return nothing
    @warn "$caller: species column(s) $cols have no non-zero count in any observed cell. The fit still runs, but the intercept of such a species has no finite maximum-likelihood value (it ends at a large negative number set by where the optimiser stops), so `converged = true` does not mean it is identified. Drop the column or treat its estimates as not identified." maxlog=1
    return nothing
end
