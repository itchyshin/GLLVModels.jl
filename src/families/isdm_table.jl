# iSDM long-table assembly and contract checks (docs/design/isdm-port-spec.md
# sections 1.4, 1.5, 2.2, 3.2). One row per (unit, trait, source[, visit]);
# the checks run in R's order (gllvmTMB P1):
#
#   1. selector alignment by name                 R/fit-multi.R:1432-1452
#   2. the declared-contract core predicate       R/isdm-sources.R:412-439
#   3. the within-trait family/link scale rule    R/fit-multi.R:293-318, 1497-1501
#   4. fixed design, then source-masked columns   R/fit-multi.R:3329-3345; R/isdm-sources.R:178-277
#   5. the offset gate (cloglog exception)        R/offset.R:108-195
#   6. weights and multi-trial refusals           R/fit-multi.R:3462-3480
#   7. every source x trait arm observed          R/isdm-sources.R:445-477
#
# Missing responses are refused up front (scope decision D-296: R's
# `miss_control(response = "include")` masking is a separate missing-data row).

"""
    IsdmTable

The validated long table of an integrated species-distribution model, built by
[`isdm_table`](@ref). Per-row fields: `y`, `n_trials` (1 on every detection
row), `trait_id`, `unit_id`, `source_id` (index into `sources.names`), `fid`,
`lid` (R's per-row family and link ids), `offset`, and the fixed design `X`
with R-style column names `X_names` (e.g. `"traitsp1:env"`,
`"isdm_source:gbif:access"`). Also `rows_by_unit` (row indices per unit), the
level vectors, the latent rank `K` read from the formula, whether the
mixed-law contract was `admitted`, and the frozen design basis used by
`predict(...; newdata)`.
"""
struct IsdmTable
    y::Vector{Float64}
    n_trials::Vector{Int}
    trait_id::Vector{Int}
    unit_id::Vector{Int}
    source_id::Vector{Int}
    fid::Vector{Int}
    lid::Vector{Int}
    offset::Vector{Float64}
    X::Matrix{Float64}
    X_names::Vector{String}
    rows_by_unit::Vector{Vector{Int}}
    trait_levels::Vector{String}
    unit_levels::Vector{String}
    sources::IsdmSources
    admitted::Bool
    K::Int
    response::Symbol
    trait::Symbol
    unit::Symbol
    offset_expr::Any
    fixed_rhs::Any                 # schema-applied StatsModels right-hand side
    fixed_names::Vector{String}    # R names of the ecological (formula) columns
    obs_basis::Dict{Symbol, Any}   # source => (rhs, raw R names) for predict
    formula::Expr
end

Base.show(io::IO, t::IsdmTable) = print(io, "IsdmTable(", length(t.y), " rows, ",
    length(t.trait_levels), " traits, ", length(t.unit_levels), " units, sources = ",
    join(t.sources.names, ", "), ", K = ", t.K, ")")

# A column with no missing values, narrowed to its concrete element type.
_isdm_narrow(col) = any(ismissing, col) ? col : [v for v in col]

_isdm_labels(col) = [ismissing(v) ? missing : string(v) for v in col]

"""
    _isdm_declared_core(sources, selector, fid, lid, trait_labels, n) -> Bool

The shared admission predicate, twin of `.gllvmTMB_isdm_declared_core`
(`R/isdm-sources.R:412-439`). `true` only when the selector has one value per
row, every value is a declared name and every declared name occurs, at least
one count and one detection law are declared, each row's `(fid, lid)` is its
source's declared law, and every trait carries every declared source (row
presence, not balance).
"""
function _isdm_declared_core(sources::IsdmSources, selector::AbstractVector,
        fid::AbstractVector, lid::AbstractVector, trait_labels, n::Integer)
    nms = String.(sources.names)
    sel = [ismissing(s) ? missing : string(s) for s in selector]
    ok = length(sel) == n && all(s -> !ismissing(s) && s in nms, sel) &&
         all(nm -> any(isequal(nm), sel), nms) &&
         any(==(2), sources.fid) && any(==(1), sources.fid)
    ok || return false
    idx = [findfirst(==(s), nms) for s in sel]
    (collect(Int.(fid)) == sources.fid[idx] && collect(Int.(lid)) == sources.lid[idx]) || return false
    (trait_labels === nothing || length(trait_labels) != n) && return false
    tl = string.(trait_labels)
    for t in unique(tl)
        present = Set(sel[tl .== t])
        all(nm -> nm in present, nms) || return false
    end
    return !isempty(tl)
end

# The within-trait scale rule, with the admitted exception
# (R/fit-multi.R:293-318; class gllvmTMB_family_within_trait_unsupported).
function _isdm_assert_trait_scale(fid, lid, trait_labels; allow_isdm_mixed::Bool)
    allow_isdm_mixed && return nothing
    keys_by_trait = Dict{String, Set{Tuple{Int, Int}}}()
    for i in eachindex(fid)
        push!(get!(keys_by_trait, string(trait_labels[i]), Set{Tuple{Int, Int}}()), (fid[i], lid[i]))
    end
    bad = sort([t for (t, s) in keys_by_trait if length(s) > 1])
    isempty(bad) || throw(ArgumentError(
        "Response family/link cannot currently vary across rows within a trait. Multiple " *
        "family/link scales were requested within: $(join(bad, ", ")). The one admitted " *
        "exception is the integrated multi-source model, which needs every trait to be " *
        "observed by every declared source."))
    return nothing
end

"""
    _isdm_prepare_offset(off, fid, lid; allow_isdm_cloglog) -> off

The count-family offset gate, twin of `gll_prepare_offset()` (`R/offset.R:108-195`)
after evaluation: non-finite values abort; a non-zero offset is refused on any row
whose family is not a count family (R ids 2, 5, 10, 11, 15), except a
Bernoulli-cloglog row (`fid = 1`, `lid = 2`) when `allow_isdm_cloglog` is set,
which happens only once the mixed-law contract has been admitted. Zero offsets
always pass.
"""
function _isdm_prepare_offset(off::AbstractVector{<:Real}, fid::AbstractVector,
        lid::AbstractVector; allow_isdm_cloglog::Bool = false)
    bad = findall(!isfinite, off)
    isempty(bad) || throw(ArgumentError(
        "`offset()` has $(length(bad)) non-finite value(s). First at row $(first(bad))."))
    count_ids = (2, 5, 10, 11, 15)
    for i in eachindex(off)
        off[i] == 0 && continue
        fid[i] in count_ids && continue
        allow_isdm_cloglog && fid[i] == 1 && lid[i] == 2 && continue
        throw(ArgumentError(
            "offsets are supported for count families (poisson, nbinom) only; row $i uses " *
            "family id $(fid[i]) with link id $(lid[i]). Set the offset to 0 on the rows of a " *
            "non-count family."))
    end
    return off
end

"""
    _isdm_assert_observed_arms(source, trait, is_observed, declared) -> true

Every declared source x trait arm must contribute at least one observed
response, twin of `.gllvmTMB_assert_isdm_observed_arms()`
(`R/isdm-sources.R:445-477`).
"""
function _isdm_assert_observed_arms(source::AbstractVector, trait::AbstractVector,
        is_observed::AbstractVector, declared)
    src = string.(source); tr = string.(trait)
    (length(src) == length(tr) == length(is_observed)) || throw(ArgumentError(
        "Internal: integrated-source observation mask is misaligned."))
    bad = String[]
    for t in unique(tr), s in string.(declared)
        any(i -> src[i] == s && tr[i] == t && is_observed[i] != 0, eachindex(src)) ||
            push!(bad, "$s x $t")
    end
    isempty(bad) || throw(ArgumentError(
        "Every declared integrated source-trait arm needs an observed response. No observed " *
        "response in: $(join(bad, ", "))."))
    return true
end

# Rank as R's qr()$rank reports it: pivoted QR, diagonal entries above a 1e-7
# relative tolerance (LINPACK dqrdc2's default tol; spec risk R4).
function _isdm_qr_rank(M::AbstractMatrix)
    size(M, 2) == 0 && return 0
    F = qr(Matrix{Float64}(M), ColumnNorm())
    d = abs.(diag(F.R))
    isempty(d) && return 0
    return count(>(1e-7 * d[1]), d)
end

# Levels of a categorical column over the FULL column (non-missing), as R's
# factor levels are, so a source subset carries the same dummy basis.
_isdm_full_levels(col) = sort(unique(string(v) for v in col if !ismissing(v)))

function _isdm_contrasts_for(syms, cols)
    c = Dict{Symbol, Any}()
    for s in syms
        haskey(cols, s) || continue
        col = getproperty(cols, s)
        nonmiss = [v for v in col if !ismissing(v)]
        (isempty(nonmiss) || all(v -> v isa Real, nonmiss)) && continue
        c[s] = StatsModels.DummyCoding(levels = _isdm_full_levels(col))
    end
    return c
end

# Build the rhs model matrix of a one-sided or two-sided term tuple on `cols`.
function _isdm_model_matrix(rhs_terms, cols, contrasts)
    dummy = :__isdm_response__
    n = length(first(values(cols)))
    cols2 = merge(cols, NamedTuple{(dummy,)}((zeros(n),)))
    f = StatsModels.FormulaTerm(StatsModels.Term(dummy), rhs_terms)
    sch = StatsModels.schema(f, cols2, contrasts)
    applied = StatsModels.apply_schema(f, sch, StatsModels.StatisticalModel)
    X = Matrix{Float64}(StatsModels.modelmatrix(applied.rhs, cols2))
    names = StatsModels.coefnames(applied.rhs)
    names = names isa AbstractString ? [names] : collect(names)
    return X, _isdm_r_name.(names), applied.rhs
end

"""
    _isdm_observation_design(X, X_names, cols, source, sources) -> (X, names, basis)

Source-masked observation columns, twin of `.gll_isdm_observation_design()`
(`R/isdm-sources.R:178-277`): each source formula is evaluated on that source's
rows only, its columns are named `isdm_source:<src>:<term>` and zero elsewhere,
and a column is kept only if it raises the QR rank of the design built so far.
"""
function _isdm_observation_design(X::Matrix{Float64}, X_names::Vector{String}, cols,
        source::AbstractVector, sources::IsdmSources)
    basis = Dict{Symbol, Any}()
    isempty(sources.observation) && return X, X_names, basis
    any(nm -> occursin(r"(^|:)isdm_source", nm), X_names) && throw(ArgumentError(
        "Top-level `isdm_source` fixed effects duplicate an `isdm_source()` observation formula. " *
        "Remove `isdm_source` from the main ecological formula, or use bare laws in " *
        "`isdm_sources()` and write the source effects manually."))
    n = size(X, 1)
    src_chr = string.(source)
    blocks = Matrix{Float64}[]; bnames = String[]
    for s in sources.names
        form = get(sources.observation, s, nothing)
        form === nothing && continue
        rows = findall(==(String(s)), src_chr)
        isempty(rows) && throw(ArgumentError(
            "Internal: declared iSDM source \"$s\" has no rows after filtering."))
        rhs = form.args[2]
        vars = unique(_isdm_term_symbols(rhs))
        missing_vars = [v for v in vars if !haskey(cols, v)]
        isempty(missing_vars) || throw(ArgumentError(
            "Observation formula for source \"$s\" uses variable(s) not found in `data`. Missing: " *
            join(("\"$v\"" for v in missing_vars), ", ") * "."))
        for v in vars
            any(ismissing, view(getproperty(cols, v), rows)) && throw(ArgumentError(
                "Observation formula for source \"$s\" has missing values after source filtering. " *
                "Remove or impute missing observation covariates for that source before fitting."))
        end
        sub = NamedTuple{Tuple(vars)}(Tuple(collect(skipmissing(getproperty(cols, v)[rows])) for v in vars))
        terms = Tuple(_isdm_term(t) for t in _isdm_flatten_plus(rhs))
        mm, raw, applied = _isdm_model_matrix(terms, sub, _isdm_contrasts_for(vars, cols))
        cn = ["isdm_source:$(s):" * r for r in raw]
        full = zeros(n, size(mm, 2)); full[rows, :] = mm
        push!(blocks, full); append!(bnames, cn)
        basis[s] = (rhs = applied, columns = cn, vars = vars, contrasts = _isdm_contrasts_for(vars, cols))
    end
    dup = [nm for nm in unique(bnames) if count(==(nm), bnames) > 1]
    isempty(dup) || throw(ArgumentError(
        "Source labels and observation terms produce ambiguous fixed-effect columns. Colliding: " *
        join(dup, ", ") * "."))
    S = isempty(blocks) ? zeros(n, 0) : reduce(hcat, blocks)
    keep = falses(size(S, 2))
    current = X
    rk = _isdm_qr_rank(current)
    for j in axes(S, 2)
        cand = hcat(current, view(S, :, j))
        rc = _isdm_qr_rank(cand)
        if rc > rk
            keep[j] = true; current = cand; rk = rc
        end
    end
    if !all(keep)
        @info "Dropped aliased source-observation column(s): $(join(bnames[.!keep], ", ")). " *
              "The retained columns are source-specific effects relative to the ecological `0 + trait` intercepts."
    end
    return hcat(X, S[:, keep]), vcat(X_names, bnames[keep]), basis
end

"""
    isdm_table(formula::Expr, data; family::IsdmSources, trait = :trait,
               unit = :cell_id, weights = nothing, n_trials = nothing) -> IsdmTable

Validate and assemble the long table of an integrated species-distribution
model. `data` is any Tables.jl table with one row per (unit, trait, source[,
visit]): the response, the `trait` and `unit` columns, an `isdm_source` column
whose values are the declared source names, and the offset variable. `formula`
is a quoted expression (see [`fit_isdm_gllvm`](@ref)). `weights` exists only to
be refused; `n_trials` (per row, default all 1) exists only so a multi-trial
detection row can be refused.

The checks run in R's order and each throws an `ArgumentError` whose message
starts with R's first line (gllvmTMB P1): selector alignment ("length(family)
must match the number of distinct levels in `isdm_source`."), the
within-trait scale rule, the observation-formula checks, the offset gate
("offsets are supported for count families (poisson, nbinom) only"), weights
("`weights` is not admitted for the integrated multi-source model."), multi-trial
detection rows, and the observed-arm check. Missing responses are refused.
"""
function isdm_table(formula::Expr, data; family::IsdmSources, trait::Symbol = :trait,
        unit::Symbol = :cell_id, weights = nothing, n_trials = nothing)
    pf = _isdm_parse_formula(formula; trait = trait, unit = unit)
    cols = Tables.columntable(data)
    for key in (pf.response, trait, unit)
        haskey(cols, key) || throw(ArgumentError("column `$key` not found in `data`."))
    end
    n = length(getproperty(cols, pf.response))
    ycol = getproperty(cols, pf.response)
    any(ismissing, ycol) && throw(ArgumentError(
        "The integrated door refuses missing responses in P1 ($(count(ismissing, ycol)) " *
        "missing in `$(pf.response)`); drop those rows. Masked-response fitting is a separate " *
        "missing-data capability."))

    # 1. selector alignment by name (R/fit-multi.R:1432-1452).
    haskey(cols, :isdm_source) || throw(ArgumentError(
        "Mixed-family fit needs a `isdm_source` column in `data`."))
    sel = _isdm_labels(getproperty(cols, :isdm_source))
    any(ismissing, sel) && throw(ArgumentError(
        "`isdm_source` has missing values; every row must name its declared source."))
    fam_levels = sort(unique(sel))
    length(fam_levels) == length(family.names) || throw(ArgumentError(
        "length(family) must match the number of distinct levels in `isdm_source`. Got " *
        "$(length(family.names)) families for $(length(fam_levels)) levels."))
    nms = String.(family.names)
    missing_lv = setdiff(fam_levels, nms); extra = setdiff(nms, fam_levels)
    (isempty(missing_lv) && isempty(extra)) || throw(ArgumentError(
        "Mixed-family `family` list names must match the levels of `isdm_source`. Missing " *
        "family entries for: $(join(missing_lv, ", ")). Unused family entries for: $(join(extra, ", "))."))
    source_id = [findfirst(==(s), nms) for s in sel]
    fid = family.fid[source_id]; lid = family.lid[source_id]
    trait_lab = string.(getproperty(cols, trait))
    unit_lab = string.(getproperty(cols, unit))

    # 2-3. admission, then the scale rule.
    admitted = _isdm_declared_core(family, sel, fid, lid, trait_lab, n)
    _isdm_assert_trait_scale(fid, lid, trait_lab; allow_isdm_mixed = admitted)

    # 4. fixed design, then source-masked observation columns.
    for s in pf.fixed_symbols
        haskey(cols, s) || throw(ArgumentError("column `$s` not found in `data`."))
    end
    fcols = NamedTuple{Tuple(pf.fixed_symbols)}(Tuple(_isdm_narrow(getproperty(cols, s)) for s in pf.fixed_symbols))
    X, Xn, fixed_rhs = _isdm_model_matrix(pf.fixed, fcols, _isdm_contrasts_for(pf.fixed_symbols, cols))
    fixed_names = copy(Xn)
    X, Xn, basis = _isdm_observation_design(X, Xn, cols, sel, family)

    # 5. the offset gate.
    off = pf.offset === nothing ? zeros(n) : _isdm_eval_expr(pf.offset, cols, n)
    length(off) == n || throw(ArgumentError("`offset()` has length $(length(off)) but the model has $n rows."))
    _isdm_prepare_offset(off, fid, lid; allow_isdm_cloglog = admitted)

    # 6. weights and multi-trial detection rows.
    nt = n_trials === nothing ? ones(Int, n) : Int.(collect(n_trials))
    length(nt) == n || throw(ArgumentError("`n_trials` must have one value per row."))
    if admitted
        weights === nothing || throw(ArgumentError(
            "`weights` is not admitted for the integrated multi-source model. Across this " *
            "model's arms `weights` would mean two different things: a binomial trial count on " *
            "the detection rows and a likelihood multiplier on the count rows. Drop `weights` " *
            "and give each visit its own row."))
        nbad = count(i -> fid[i] == 1 && nt[i] != 1, 1:n)
        nbad == 0 || throw(ArgumentError(
            "The integrated multi-source model admits only single-trial detection rows. " *
            "$nbad detection row(s) carry more than one trial; give each visit its own row " *
            "with its own support."))
    elseif weights !== nothing
        throw(ArgumentError(
            "`weights` is not available on the Julia integrated door in P1. R keeps `weights` " *
            "as likelihood multipliers on an all-count declaration; that route is fenced here."))
    end

    # 7. every declared source x trait arm observed (all rows are observed:
    # missing responses were refused above).
    _isdm_assert_observed_arms(sel, trait_lab, trues(n), family.names)

    # Julia-side value checks (the densities below assume them).
    y = Float64.(ycol)
    for i in 1:n
        if fid[i] == 2
            (y[i] >= 0 && isinteger(y[i])) || throw(ArgumentError(
                "Count rows need non-negative integer responses; row $i has $(y[i])."))
        else
            (y[i] == 0 || y[i] == 1) || throw(ArgumentError(
                "Detection rows need 0/1 responses; row $i has $(y[i])."))
        end
    end

    trait_levels = sort(unique(trait_lab)); unit_levels = sort(unique(unit_lab))
    tmap = Dict(v => i for (i, v) in enumerate(trait_levels))
    umap = Dict(v => i for (i, v) in enumerate(unit_levels))
    trait_id = [tmap[v] for v in trait_lab]; unit_id = [umap[v] for v in unit_lab]
    rows_by_unit = [Int[] for _ in unit_levels]
    for i in 1:n
        push!(rows_by_unit[unit_id[i]], i)
    end
    return IsdmTable(y, nt, trait_id, unit_id, source_id, fid, lid, off, X, Xn, rows_by_unit,
                     trait_levels, unit_levels, family, admitted, pf.K, pf.response, trait, unit,
                     pf.offset, fixed_rhs, fixed_names, basis, formula)
end
