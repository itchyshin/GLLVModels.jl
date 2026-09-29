# Integrated species distribution model (iSDM): source declarations.
#
# Twin of gllvmTMB's public door `gllvmTMB(..., family = isdm_sources(...))` at
# pin P1 (gllvmTMB 9539352f6), R/isdm-sources.R. A source is a named
# observation stream with one admitted observation law. Only two laws are
# admitted, because both observe a thinning of one shared intensity:
#
#   count stream      Poisson(), log link        R (family_id, link_id) = (2, 0)
#   detection stream  Bernoulli, cloglog link    R (family_id, link_id) = (1, 2)
#
# (R/isdm-sources.R:15-26; ids from R/fit-multi.R:1193-1213.) Design spec:
# docs/design/isdm-port-spec.md (sections 1.1, 2.1, 3.2).
#
# Julia spellings of a law: `Poisson()` (log link implied), `(Binomial(),
# CLogLogLink())`, or `Binomial() => CLogLogLink()`. A bare `Binomial()` means
# the logit link and is refused, exactly as R refuses `binomial()`.

"""
    IsdmSource

One declared iSDM source: an observation law (`family`, `link`) plus an optional
source-specific observation formula (`observation`, a one-sided `Expr` such as
`:(~ access + popdens)`, stored as `~(rhs)`). Built by [`isdm_source`](@ref) and
consumed by [`isdm_sources`](@ref).
"""
struct IsdmSource
    family::Any
    link::Link
    observation::Union{Nothing, Expr}
end

"""
    IsdmSources

A validated multi-source iSDM declaration, the Julia twin of the list that R's
`isdm_sources()` returns (with `family_var = "isdm_source"`). Fields: `names`
(source labels, in declaration order), `families` and `links` (the law of each
source), `fid` and `lid` (R's `(family_id, link_id)` of each law), and
`observation`, a `Dict{Symbol, Any}` mapping each source name to its
observation formula (`nothing` for a bare law). `observation` is empty when no
source carries a formula. Built by [`isdm_sources`](@ref).
"""
struct IsdmSources
    names::Vector{Symbol}
    families::Vector{Any}
    links::Vector{Link}
    fid::Vector{Int}
    lid::Vector{Int}
    observation::Dict{Symbol, Any}
end

Base.show(io::IO, s::IsdmSources) = print(io, "IsdmSources(",
    join(("$(n) = $(_isdm_law_label(s.fid[i], s.lid[i]))" for (i, n) in enumerate(s.names)), ", "),
    ")")

_isdm_law_label(fid, lid) = (fid, lid) == (2, 0) ? "poisson(log)" :
                            (fid, lid) == (1, 2) ? "binomial(cloglog)" : "fid=$fid,lid=$lid"

# Normalise a law spelling to (family, link); `nothing` if it is not a family
# marker at all (the R check `inherits(family, "family")`).
function _isdm_family_link(x)
    if x isa Distribution
        return (x, default_link(x))
    elseif x isa Tuple && length(x) == 2 && x[1] isa Distribution && x[2] isa Link
        return (x[1], x[2])
    elseif x isa Pair && x.first isa Distribution && x.second isa Link
        return (x.first, x.second)
    end
    return nothing
end

"""
    _isdm_admitted_law_id(family, link) -> Union{Tuple{Int,Int}, Nothing}

R's `(family_id, link_id)` of an admitted iSDM law, or `nothing` when the law
is not admitted (`R/isdm-sources.R:15-26`).
"""
_isdm_admitted_law_id(family, link) =
    family isa Poisson && link isa LogLink ? (2, 0) :
    family isa Binomial && link isa CLogLogLink ? (1, 2) : nothing

"""
    isdm_source(family; observation) -> IsdmSource

Declare one iSDM source with a source-specific observation formula, the twin
of R's `isdm_source(family, observation)`. `family` is a family marker
(`Poisson()`, `(Binomial(), CLogLogLink())`, ...); whether the law is admitted
is checked later by [`isdm_sources`](@ref), as in R. `observation` is a
one-sided formula written as a quoted expression, e.g. `:(~ access + popdens)`
or `:(~ observer + method)`. Its columns enter the fixed design only on rows
of this source, prefixed `isdm_source:<source>:`.

Refusals (first line of R's message): a non-family `family` throws
"`family` must be an R <family> object."; a non-formula `observation` throws
"`observation` must be a one-sided formula.".
"""
function isdm_source(family; observation = nothing)
    fl = _isdm_family_link(family)
    fl === nothing && throw(ArgumentError(
        "`family` must be an R <family> object. In Julia: pass an admitted family " *
        "marker such as Poisson() or (Binomial(), CLogLogLink())."))
    rhs = _isdm_one_sided_rhs(observation)
    rhs === nothing && throw(ArgumentError(
        "`observation` must be a one-sided formula. Use observation = :(~ access + popdens) " *
        "or observation = :(~ observer + method)."))
    return IsdmSource(fl[1], fl[2], Expr(:call, :~, rhs))
end

# Right-hand side of a one-sided formula expression, or `nothing`. Julia parses
# `:(~ observer + method)` as `(~observer) + method` (unary `~` binds tightly),
# so the leading `~` is stripped from the leftmost leaf and the whole
# expression is taken as the right-hand side.
function _isdm_one_sided_rhs(ex)
    ex isa Expr && ex.head === :call || return nothing
    if ex.args[1] === :~
        return length(ex.args) == 2 ? ex.args[2] : nothing
    end
    length(ex.args) >= 2 || return nothing
    inner = _isdm_one_sided_rhs(ex.args[2])
    inner === nothing && return nothing
    out = copy(ex)
    out.args[2] = inner
    return out
end

"""
    isdm_sources(; name = law, ...) -> IsdmSources
    isdm_sources(:name => law, ...) -> IsdmSources

Declare the sources of an integrated species-distribution model, the twin of
R's `isdm_sources(...)` (gllvmTMB P1). Each argument is named for a source and
gives either a bare law (`Poisson()` for a count stream, `(Binomial(),
CLogLogLink())` for a detection/non-detection stream) or an
[`isdm_source`](@ref) with its observation formula.

```julia
fam = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))
```

Checks, in R's order, each an `ArgumentError` whose message starts with R's
first line: fewer than two sources, or any unnamed ("`isdm_sources()` needs at
least two named sources."); a duplicate name ("Source names must be unique;
... is declared twice."); a law other than Poisson-log or Bernoulli-cloglog
("... declare(s) an observation law that is not admitted."); no count source
("An integrated declaration needs at least one count arm."). An all-count
declaration is accepted.

Everything an integrated fit reports is relative intensity: presence-only data
cannot identify absolute abundance, occupancy or detectability.
"""
function isdm_sources(args...; kwargs...)
    decls = Pair{Any, Any}[]
    named = true
    for a in args
        if a isa Pair && (a.first isa Symbol || a.first isa AbstractString)
            push!(decls, Symbol(a.first) => a.second)
        else
            named = false
            push!(decls, nothing => a)
        end
    end
    for (k, v) in kwargs
        push!(decls, k => v)
    end
    (length(decls) < 2 || !named) && throw(ArgumentError(
        "`isdm_sources()` needs at least two named sources. Each argument is named for a " *
        "source and gives its observation law, e.g. " *
        "isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))."))
    nms = Symbol[first(d) for d in decls]
    seen = Set{Symbol}()
    for n in nms
        n in seen && throw(ArgumentError(
            "Source names must be unique; \"$(n)\" is declared twice. Rename one of the " *
            "`isdm_sources()` arguments so every source has a distinct name."))
        push!(seen, n)
    end
    fams = Any[]; lnks = Link[]; obs = Dict{Symbol, Any}(); has_obs = false
    bad = Symbol[]
    ids = Tuple{Int, Int}[]
    for (n, v) in decls
        law = v isa IsdmSource ? (v.family, v.link) : _isdm_family_link(v)
        if v isa IsdmSource
            obs[n] = v.observation; has_obs = true
        else
            obs[n] = nothing
        end
        id = law === nothing ? nothing : _isdm_admitted_law_id(law[1], law[2])
        if id === nothing
            push!(bad, n)
            continue
        end
        push!(fams, law[1]); push!(lnks, law[2]); push!(ids, id)
    end
    if !isempty(bad)
        lab = join(("\"$b\"" for b in bad), ", ")
        verb = length(bad) == 1 ? "Source $lab declares" : "Sources $lab declare"
        throw(ArgumentError(
            "$verb an observation law that is not admitted. The integrated model admits " *
            "Poisson() (count stream) and (Binomial(), CLogLogLink()) (detection/non-detection), " *
            "because both observe a thinning of one shared intensity; logit or probit " *
            "detection models, and dispersion-carrying families, are refused."))
    end
    any(id -> id[1] == 2, ids) || throw(ArgumentError(
        "An integrated declaration needs at least one count arm. Every declared source is a " *
        "detection/non-detection stream; the detection arm's offset is admitted as a " *
        "change-of-support term only alongside a count arm sharing the same intensity. " *
        "Declare at least one Poisson() source, or fit the surveys separately."))
    return IsdmSources(nms, fams, lnks, [id[1] for id in ids], [id[2] for id in ids],
                       has_obs ? obs : Dict{Symbol, Any}())
end
