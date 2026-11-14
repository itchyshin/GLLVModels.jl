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
