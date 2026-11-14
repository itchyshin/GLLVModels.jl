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

# Gamma and Beta only: a dispersion group whose members are ALL constant (every member has
# zero variance over its observed cells) has no finite maximum in its dispersion (the log
# density of a constant column grows like (1/2) log(shape) per cell), so the fit runs the
# dispersion off to infinity and reports a meaningless log-likelihood (P82 CONST-COLUMN,
# issue #1143). A species with its own dispersion is a group of one. Refused with the
# species named. A constant species that shares its group with at least one species that
# varies is NOT refused: that case is outside the ruling (CONST-COLUMN-SCOPE) and the
# varying member does not guarantee a bounded dispersion (the latent variable can absorb
# it; the existing large-shape warning is the only guard there). A species with fewer than
# two observed cells does not count as constant. `gidx[t]` is the dispersion-group index
# of species `t`.
function _check_constant_species(Y::AbstractMatrix, mask, gidx::AbstractVector{<:Integer};
        caller::AbstractString)
    p, n = size(Y)
    const_val = Vector{Any}(nothing, p)   # the constant value of a constant species, else nothing
    nobs_t = zeros(Int, p)
    for t in 1:p
        first_y = nothing
        nobs = 0
        constant = true
        for i in 1:n
            (mask === nothing || mask[t, i]) || continue
            y = Y[t, i]
            ismissing(y) && continue
            nobs += 1
            if first_y === nothing
                first_y = y
            elseif y != first_y
                constant = false
                break
            end
        end
        nobs_t[t] = nobs
        (constant && nobs >= 2) && (const_val[t] = first_y)
    end
    for g in unique(gidx)
        members = findall(==(g), gidx)
        all(t -> const_val[t] !== nothing, members) || continue
        if length(members) == 1
            t = members[1]
            throw(ArgumentError(
                "$caller: species $t (row $t of Y) is constant at $(const_val[t]) over its $(nobs_t[t]) " *
                "observed cells and has its own dispersion, whose likelihood has no finite " *
                "maximum. Drop the species, or put it in a dispersion group with a species that varies."))
        else
            throw(ArgumentError(
                "$caller: species $(join(members, ", ")) (rows of Y) share one dispersion group and " *
                "every one of them is constant over its observed cells, so the group's dispersion " *
                "likelihood has no finite maximum. Drop these species, or put one of them in a " *
                "group with a species that varies."))
        end
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
