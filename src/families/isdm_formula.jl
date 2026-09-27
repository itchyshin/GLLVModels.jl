# iSDM formula reader (docs/design/isdm-port-spec.md section 3.2).
#
# The integrated door takes the model formula as a quoted expression, because
# StatsModels' `@formula` cannot parse the keyword arguments inside
# `latent(0 + trait | cell_id, d = 1, unique = FALSE)`:
#
#   :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) +
#       latent(0 + trait | cell_id, d = 1, unique = FALSE))
#
# Interactions are written with `&` (R's `trait:env` is `trait & env`; a bare
# `:` binds more loosely than `+` in Julia, so it is accepted only inside
# parentheses, `(trait:env)`).
# The reader pulls out `offset(expr)` (R/offset.R:14-16) and zero or one
# `latent(0 + trait | unit, d = K, unique = FALSE)` (R/parse-multi-formula.R),
# refuses every other structured term by name, and hands the remaining fixed
# right-hand side to StatsModels, whose implicit-intercept and full-rank rules
# follow R's `model.matrix` (so `0 + trait` gives full dummy coding).
#
# R's `latent()` defaults to `unique = TRUE` (R/brms-sugar.R:607), which adds a
# per-trait unit-level diagonal variance (TMB parameter `theta_diag_B`). The P1
# port spec (section 1.2) describes the loadings-only model, so the Julia door
# fits that model and requires `unique = FALSE` to be written explicitly: a
# call that relies on R's default would otherwise fit a different model from
# the one R fits under the same text.

import StatsModels
import Tables

const _ISDM_REFUSED_CALLS = (:indep, :dep, :unique, :phylo, :animal, :kernel, :spatial,
                             :temporal, :propto, :diag, :rr, :re_int)

_isdm_flatten_plus(ex) =
    ex isa Expr && ex.head === :call && ex.args[1] === :+ ?
        reduce(vcat, (_isdm_flatten_plus(a) for a in ex.args[2:end]); init = Any[]) : Any[ex]

function _isdm_refused_structured(ex)
    ex isa Expr && ex.head === :call || return false
    f = ex.args[1]
    f === :| && return true
    f isa Symbol || return false
    s = String(f)
    return any(r -> s == String(r) || startswith(s, String(r) * "_"), _ISDM_REFUSED_CALLS)
end

# Parse `latent(0 + trait | unit, d = K, unique = FALSE)`; returns K.
function _isdm_parse_latent(ex::Expr, trait::Symbol, unit::Symbol)
    bar = length(ex.args) >= 2 ? ex.args[2] : nothing
    (bar isa Expr && bar.head === :call && bar.args[1] === :|) || throw(ArgumentError(
        "latent() in the integrated door must be written latent(0 + $trait | $unit, d = K, unique = FALSE)."))
    lhs, grp = bar.args[2], bar.args[3]
    lhs_terms = _isdm_flatten_plus(lhs)
    (length(lhs_terms) == 2 && lhs_terms[1] == 0 && lhs_terms[2] === trait) || throw(ArgumentError(
        "latent() in the integrated door needs the left-hand side `0 + $trait`; got `$(lhs)`."))
    grp === unit || throw(ArgumentError(
        "latent() grouping `$grp` must be the unit column `$unit`: the integrated door fits " *
        "one between-unit reduced-rank block (R's rr_B) and nothing else."))
    d = nothing; uniq = nothing
    for a in ex.args[3:end]
        (a isa Expr && a.head === :kw) || throw(ArgumentError(
            "latent(): unexpected argument `$a`; admitted are d = K and unique = FALSE."))
        if a.args[1] === :d
            d = a.args[2]
        elseif a.args[1] === :unique
            uniq = a.args[2]
        else
            throw(ArgumentError("latent(): argument `$(a.args[1])` is not admitted on the integrated door."))
        end
    end
    (d isa Integer && d >= 1) || throw(ArgumentError(
        "latent(): `d` must be a positive integer literal; got `$(repr(d))`."))
    if uniq !== false && uniq !== :FALSE
        throw(ArgumentError(
            "latent(..., unique = TRUE) is R's default and adds a per-trait unit-level unique " *
            "variance (theta_diag_B) that the Julia integrated door does not fit. Write " *
            "latent(0 + $trait | $unit, d = $d, unique = FALSE) for the loadings-only model; " *
            "R fits the same model under that text."))
    end
    return Int(d)
end

# Expr -> StatsModels term, for the admitted fixed-effect grammar.
function _isdm_term(ex)
    if ex isa Symbol
        return StatsModels.Term(ex)
    elseif ex isa Integer && (ex == 0 || ex == 1)
        return StatsModels.ConstantTerm(ex)
    elseif ex isa Expr && ex.head === :call && ex.args[1] in (:&, :(:)) && length(ex.args) >= 3
        return reduce(&, (_isdm_term(a) for a in ex.args[2:end]))
    elseif ex isa Expr && ex.head === :call && ex.args[1] === :* && length(ex.args) >= 3
        return reduce(*, (_isdm_term(a) for a in ex.args[2:end]))
    end
    throw(ArgumentError(
        "Unsupported term `$ex` in the integrated formula: admitted fixed terms are column " *
        "names, 0/1, and their interactions written with `&` or `*` (R's `trait:env` is " *
        "`trait & env` in Julia)."))
end

_isdm_term_symbols(ex) = ex isa Symbol ? Symbol[ex] :
    ex isa Expr && ex.head === :call ? reduce(vcat, (_isdm_term_symbols(a) for a in ex.args[2:end]); init = Symbol[]) :
    Symbol[]

"""
    _isdm_parse_formula(f::Expr; trait, unit) -> NamedTuple

Split an integrated formula into `(response, fixed, fixed_symbols, offset, K)`:
`fixed` is the StatsModels right-hand side (a tuple of terms), `offset` the inner
expression of `offset(...)` (or `nothing`), and `K` the latent rank (0 without a
`latent()` term, in which case the fit is a GLM through the same kernel).
"""
function _isdm_parse_formula(f::Expr; trait::Symbol, unit::Symbol)
    (f.head === :call && f.args[1] === :~ && length(f.args) == 3) || throw(ArgumentError(
        "The integrated formula must be two-sided, e.g. :(value ~ 0 + trait + trait & env + offset(log_support))."))
    resp = f.args[2]
    resp isa Symbol || throw(ArgumentError(
        "The integrated formula needs a single response column on the left-hand side; got `$resp`. " *
        "Multi-trial `cbind(successes, failures)` responses are not admitted: give each visit its own row."))
    offset = nothing
    K = 0; nlatent = 0
    fixed = Any[]
    for t in _isdm_flatten_plus(f.args[3])
        if t isa Expr && t.head === :call && t.args[1] === :offset
            offset === nothing || throw(ArgumentError("Only one offset() term is admitted."))
            length(t.args) == 2 || throw(ArgumentError("offset() takes one expression."))
            offset = t.args[2]
        elseif t isa Expr && t.head === :call && t.args[1] === :latent
            nlatent += 1
            nlatent == 1 || throw(ArgumentError(
                "The integrated door admits zero or one latent() term; got more than one."))
            K = _isdm_parse_latent(t, trait, unit)
        elseif _isdm_refused_structured(t)
            throw(ArgumentError(
                "Structured term `$t` is not admitted on the integrated door (P1 scope: fixed " *
                "effects, offset(), and zero or one latent(0 + $trait | $unit, d = K, unique = FALSE))."))
        else
            push!(fixed, t)
        end
    end
    isempty(fixed) && throw(ArgumentError("The integrated formula has no fixed-effect terms."))
    terms = Tuple(_isdm_term(t) for t in fixed)
    syms = unique(reduce(vcat, (_isdm_term_symbols(t) for t in fixed); init = Symbol[]))
    return (response = resp, fixed = terms, fixed_symbols = syms, offset = offset, K = K)
end

# StatsModels coefficient name -> R model.matrix name ("trait: sp1 & env" ->
# "traitsp1:env"), so b_fix pairs with R's X_fix_names by name (spec Q3).
_isdm_r_name(s::AbstractString) = replace(replace(s, r"([A-Za-z0-9_.]+): " => s"\1"), " & " => ":")

# Evaluate an offset expression against the table: column names, numeric
# literals, and a small set of arithmetic and log/exp functions, broadcast.
const _ISDM_OFFSET_FUNS = (:log, :exp, :log1p, :log10, :log2, :sqrt, :abs, :+, :-, :*, :/, :^)

function _isdm_eval_expr(ex, cols, n::Int)
    if ex isa Symbol
        haskey(cols, ex) || throw(ArgumentError(
            "`offset()` could not evaluate `$ex`: column not found in `data`."))
        col = getproperty(cols, ex)
        any(ismissing, col) && throw(ArgumentError(
            "`offset()` has $(count(ismissing, col)) non-finite value(s)."))
        return Float64.(col)
    elseif ex isa Real
        return fill(Float64(ex), n)
    elseif ex isa Expr && ex.head === :call && ex.args[1] in _ISDM_OFFSET_FUNS
        fn = getfield(Base, ex.args[1])
        args = (_isdm_eval_expr(a, cols, n) for a in ex.args[2:end])
        return Float64.(broadcast(fn, args...))
    end
    throw(ArgumentError("`offset()` could not evaluate `$ex`."))
end
