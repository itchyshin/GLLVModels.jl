# Private Destination-B grouped non-Gaussian outer fit candidate.  The parent
# integration owner controls inclusion, exports, and any public surface.

"""
    GroupedNonGaussianFit

Joint-Laplace fit returned by `fit_gllvm` for explicit non-Gaussian grouping.
Stores fixed effects, per-term trait covariances, complete marginal parameter
coordinates, convergence/curvature diagnostics, and owned response/design data.
Beta precision and NB2 size default to one value per trait; `dispersion=:shared`
selects one scalar. Point convergence does not establish interval feasibility.
"""
struct GroupedNonGaussianFit
    beta::Vector{Float64}
    family::Any
    dispersion::Union{Nothing,Float64,Vector{Float64}}
    term_covariances::Vector{Matrix{Float64}}
    terms::Vector{GroupingTerm}
    parameters::Vector{Float64}
    loglik::Float64
    converged::Bool
    gradient_norm::Float64
    hessian_min_eigenvalue::Float64
    hessian_positive_definite::Bool
    iterations::Int
    stopping_reason::Symbol
    mean_design::Matrix{Float64}
    coefficient_names::Vector{Union{String,Symbol}}
    parameter_labels::Vector{String}
    response_shape::Tuple{Int,Int}
    response::Matrix{Float64}
    trials::Matrix{Float64}
    incidences::Vector{SparseMatrixCSC{Float64,Int}}
    family_kind::Symbol
    dispersion_mode::Symbol
    inner_status::Symbol
end

function _grouped_nongaussian_kind(family)
    family isa Poisson && return :poisson
    family isa Binomial && return :binomial
    if family isa Beta
        family.β == 1.0 ||
            throw(ArgumentError("Beta grouped Laplace family must be Beta(phi, 1.0)"))
        isfinite(family.α) && family.α > 0 ||
            throw(ArgumentError("Beta precision phi must be finite and positive"))
        return :beta
    end
    if family isa NegativeBinomial
        family.p == 0.5 ||
            throw(ArgumentError("NB2 grouped Laplace family must be NegativeBinomial(r, 0.5)"))
        isfinite(family.r) && family.r > 0 ||
            throw(ArgumentError("NB2 size r must be finite and positive"))
        return :nb2
    end
    throw(ArgumentError("grouped non-Gaussian Laplace supports Poisson, Binomial, Beta(phi, 1.0), and NegativeBinomial(r, 0.5)"))
end

_grouped_nongaussian_link(::Val{:poisson}) = LogLink()
_grouped_nongaussian_link(::Val{:binomial}) = LogitLink()
_grouped_nongaussian_link(::Val{:beta}) = LogitLink()
_grouped_nongaussian_link(::Val{:nb2}) = LogLink()

function _grouped_nongaussian_dispersion_mode(kind::Symbol, dispersion::Symbol)
    dispersion in (:shared, :trait) ||
        throw(ArgumentError("dispersion must be :shared or :trait"))
    return kind in (:beta, :nb2) ? dispersion : :none
end

function _grouped_nongaussian_internal_dispersion_mode(kind::Symbol, mode::Symbol)
    if kind in (:beta, :nb2)
        mode in (:shared, :trait) ||
            throw(ArgumentError("Beta/NB2 internal dispersion mode must be :shared or :trait"))
        return mode
    end
    # The standalone objective historically defaulted to :trait.  Poisson and
    # Binomial have no dispersion coordinate, so normalize both public modes
    # (and the explicit internal spelling) to the no-op representation.
    mode in (:none, :shared, :trait) ||
        throw(ArgumentError("Poisson/Binomial internal dispersion mode must be :none, :shared, or :trait"))
    return :none
end

_grouped_nongaussian_dispersion_count(kind::Symbol, mode::Symbol, p::Integer) =
    kind in (:beta, :nb2) ? (mode === :shared ? 1 : p) : 0

function _grouped_nongaussian_dispersion_indices(q::Integer, source_coordinates::Integer,
        kind::Symbol, mode::Symbol, p::Integer)
    count = _grouped_nongaussian_dispersion_count(kind, mode, p)
    return (q + source_coordinates + 1):(q + source_coordinates + count)
end

function _grouped_nongaussian_family(kind::Symbol, value::AbstractVector,
        dispersion_indices::UnitRange{Int}, p::Integer, n::Integer, mode::Symbol)
    kind === :poisson && return Poisson()
    kind === :binomial && return Binomial()
    length(dispersion_indices) == _grouped_nongaussian_dispersion_count(kind, mode, p) ||
        throw(ArgumentError("dispersion coordinate layout is invalid"))
    values = exp.(view(value, dispersion_indices))
    all(isfinite, values) && all(>(0.0), values) || return nothing
    marker = kind === :beta ? (d -> Beta(d, 1.0)) :
        kind === :nb2 ? (d -> NegativeBinomial(d, 0.5)) : nothing
    marker === nothing && throw(ArgumentError("unsupported grouped non-Gaussian family kind"))
    mode === :shared && return marker(only(values))
    trait_markers = marker.(values)
    return [trait_markers[trait] for trait in 1:p, observation in 1:n] |> vec
end

function _grouped_nongaussian_dispersion(value::AbstractVector,
        dispersion_indices::UnitRange{Int}, kind::Symbol, mode::Symbol)
    kind in (:beta, :nb2) || return nothing
    natural = exp.(view(value, dispersion_indices))
    return mode === :shared ? only(natural) : collect(natural)
end

function _grouped_nongaussian_marker_start(family, kind::Symbol, p::Integer,
        mode::Symbol)
    value = kind === :beta ? family.α : kind === :nb2 ? family.r : nothing
    value === nothing && return Float64[]
    return fill(log(value), _grouped_nongaussian_dispersion_count(kind, mode, p))
end

function _grouped_nongaussian_parameter_labels(coefficient_names, p::Integer,
        terms::Vector{GroupingTerm}, kind::Symbol, mode::Symbol)
    labels = _grouped_parameter_labels(coefficient_names, p, terms)
    pop!(labels) # Gaussian residual scale is absent in the conditional families.
    if kind === :beta
        append!(labels, mode === :shared ? ["log_phi"] : ["log_phi[$trait]" for trait in 1:p])
    elseif kind === :nb2
        append!(labels, mode === :shared ? ["log_r"] : ["log_r[$trait]" for trait in 1:p])
    end
    return labels
end

function _grouped_laplace_trait_factors(load::AbstractMatrix, unique)
    L = Matrix{Float64}(load)
    columns = Matrix{Float64}[]
    all(iszero, L) || push!(columns, L)
    if unique !== nothing
        d = Float64.(unique)
        all(isfinite, d) && all(>=(0.0), d) ||
            throw(ArgumentError("grouped unique variances must be finite and non-negative"))
        positive = findall(>(0.0), d)
        isempty(positive) || begin
            U = zeros(Float64, length(d), length(positive))
            for (column, trait) in enumerate(positive)
                U[trait, column] = sqrt(d[trait])
            end
            push!(columns, U)
        end
    end
    isempty(columns) && return zeros(Float64, size(L, 1), 0)
    return reduce(hcat, columns)
end

"""
    _grouped_laplace_design(incidences, loads; uniques=nothing)

Build one global sparse random-effect design for the joint Laplace kernel. Its
response rows follow `vec(Y)` and its column blocks follow selected grouping
terms. This is the same `Z_s \\otimes L_s^*` representation as the grouped
Gaussian factor kernel, not scalar row effects.
"""
function _grouped_laplace_design(incidences::Vector{SparseMatrixCSC{Float64,Int}},
        loads::Vector{<:AbstractMatrix}; uniques=nothing)
    length(incidences) == length(loads) ||
        throw(DimensionMismatch("one grouped factor matrix is required per incidence"))
    source_uniques = uniques === nothing ? fill(nothing, length(loads)) : uniques
    length(source_uniques) == length(loads) ||
        throw(DimensionMismatch("one unique-variance vector is required per grouped factor"))
    blocks = SparseMatrixCSC{Float64,Int}[]
    for source in eachindex(loads)
        Lstar = _grouped_laplace_trait_factors(loads[source], source_uniques[source])
        size(Lstar, 2) == 0 && continue
        push!(blocks, grouped_trait_design(incidences[source], Lstar))
    end
    isempty(blocks) && return spzeros(Float64, size(incidences[1], 1) * size(loads[1], 1), 0)
    return sparse(reduce(hcat, blocks))
end

# ---------------------------------------------------------------------------
# S8: analytic outer gradient -- the design Jacobian dk W = dW/dtheta_k for
# psi coordinates (docs/design/grouped-analytic-gradient.md sections 3, 6).
# ---------------------------------------------------------------------------

"""
    _grouped_term_lstar_jacobian(term, p, local_index, Lstar, load_ncols) -> Matrix{Float64}

`d Lstar / d theta_local`, `theta_local` the `local_index`-th (1-based) raw
coordinate WITHIN this one term's own packed sub-vector (the same indexing
`_grouped_term_unpack` walks). `Lstar` is this term's OWN `p`-by-width block
as built by `_grouped_laplace_trait_factors`; `load_ncols` is how many of
its leading columns are the loading block `L` (`0` for `:indep`, or for a
`:latent`/`:dep` term whose current `L` is exactly the zero matrix and so
was dropped by `_grouped_laplace_trait_factors` -- see the caveat raised at
this function's use site in `_grouped_laplace_design_jacobian`).

Two cases, both linear in `theta` so the derivative needs no clamp/domain
logic:
  - a LOADING coordinate (`unpack_lambda`, src/packing.jl): the packing is a
    raw coordinate-selecting map, so its derivative is the same map applied
    to a unit vector -- exactly one entry of `L` is `1`, the rest `0`.
  - a UNIQUE-VARIANCE coordinate (`:indep`, or a `:latent` term's own unique
    tail): the entry that actually enters the design is `exp(theta_j)`
    (docs/design/grouped-analytic-gradient.md section 1.2), so
    `d(entry)/d(theta_j) = entry` -- the derivative of that column (or, for
    `common=true`, every column, since one raw coordinate then drives ALL
    `p` diagonal entries identically) is THE SAME BLOCK'S OWN VALUES.
"""
function _grouped_term_lstar_jacobian(term::GroupingTerm, p::Integer, local_index::Integer,
        Lstar::AbstractMatrix{Float64}, load_ncols::Integer)
    width = size(Lstar, 2)
    dLstar = zeros(Float64, p, width)
    if term.mode === :dep
        e = zeros(Float64, rr_theta_len(p, p)); e[local_index] = 1.0
        dLstar .= unpack_lambda(e, p, p)
        return dLstar
    end
    if term.mode === :latent
        nload = rr_theta_len(p, term.rank)
        if local_index <= nload
            load_ncols == 0 && throw(ArgumentError(
                "psi coordinate $local_index drives a loading column that " *
                "_grouped_laplace_trait_factors dropped as exactly zero -- " *
                "the analytic design jacobian does not cover this boundary case"))
            e = zeros(Float64, nload); e[local_index] = 1.0
            dL = unpack_lambda(e, p, term.rank)
            dLstar[:, 1:load_ncols] .= view(dL, :, 1:load_ncols)
            return dLstar
        end
        local_index -= nload
    end
    # :indep, or the unique tail of a :latent term.
    ucols = width - load_ncols
    ucols == 0 && throw(ArgumentError(
        "psi coordinate has no unique-variance columns to drive (local_index=$local_index)"))
    if term.common
        dLstar[:, (load_ncols + 1):width] .= view(Lstar, :, (load_ncols + 1):width)
    else
        col = load_ncols + local_index
        dLstar[local_index, col] = Lstar[local_index, col]
    end
    return dLstar
end

"""
    _grouped_laplace_design_jacobian(incidences, loads, uniques, terms, psi_index) -> SparseMatrixCSC

`dk W = dW/dtheta_k` for the psi coordinate at 1-based index `psi_index`
within the packed psi block (`theta[q+1:q+source_coordinates]`, the layout
`_grouped_term_unpack` consumes). Same shape as
`_grouped_laplace_design(incidences, loads; uniques=uniques)`; nonzero ONLY
in the block-columns owned by the one grouping term `psi_index` belongs to
-- `dk W` never introduces a new block column (docs/design/grouped-analytic-
gradient.md section 3's `pattern(dk W) ⊆ pattern(W)` argument).
"""
function _grouped_laplace_design_jacobian(incidences::Vector{SparseMatrixCSC{Float64,Int}},
        loads::Vector{<:AbstractMatrix}, uniques, terms::Vector{GroupingTerm},
        psi_index::Integer)
    p = size(loads[1], 1)
    n = size(incidences[1], 1)
    N = p * n
    blocks = SparseMatrixCSC{Float64,Int}[]
    remaining = Int(psi_index)
    found = false
    for s in eachindex(loads)
        term = terms[s]
        nparams = _grouping_term_nparams(term, p)
        Lstar = _grouped_laplace_trait_factors(loads[s], uniques[s])
        width = size(Lstar, 2)
        if !found && remaining <= nparams
            found = true
            width == 0 && throw(ArgumentError("psi coordinate $psi_index drives a zero-width term block"))
            load_ncols = all(iszero, loads[s]) ? 0 : size(loads[s], 2)
            dLstar = _grouped_term_lstar_jacobian(term, p, remaining, Lstar, load_ncols)
            push!(blocks, grouped_trait_design(incidences[s], dLstar))
        else
            found || (remaining -= nparams)
            width == 0 || push!(blocks, spzeros(Float64, N, width))
        end
    end
    found || throw(ArgumentError("psi_index $psi_index out of range"))
    isempty(blocks) && return spzeros(Float64, N, 0)
    return sparse(reduce(hcat, blocks))
end

function _grouped_nongaussian_initial_parameters(data::Matrix{Float64},
        trials::Matrix{Float64}, D::Matrix{Float64}, terms::Vector{GroupingTerm},
        kind::Symbol, family, mode::Symbol)
    p, n = size(data)
    q = size(D, 2)
    theta = zeros(Float64, q)
    if D == _trait_mean_design(p, n)
        for trait in 1:p
            values = view(data, trait, :)
            if kind === :poisson || kind === :nb2
                theta[trait] = log(max(mean(values), 0.1))
            elseif kind === :binomial
                probability = (sum(values) + 0.5) / (sum(view(trials, trait, :)) + 1.0)
                theta[trait] = log(probability / (1.0 - probability))
            else
                probability = clamp(mean(values), 0.01, 0.99)
                theta[trait] = log(probability / (1.0 - probability))
            end
        end
    end
    for term in terms
        if term.mode === :latent
            append!(theta, init_theta_rr(p, term.rank))
            term.unique && append!(theta, fill(log(0.25), term.common ? 1 : p))
        elseif term.mode === :indep
            append!(theta, fill(log(0.25), term.common ? 1 : p))
        else
            append!(theta, init_theta_rr(p, p))
        end
    end
    append!(theta, _grouped_nongaussian_marker_start(family, kind, p, mode))
    return theta
end

"""
    _grouped_nongaussian_objective(...; warm_start_inner=false)

`warm_start_inner` (S7c, default `false` — opt-in): when `true`, the closure
keeps the last successfully-converged inner Laplace mode `b` (across calls to
the SAME closure instance) and seeds the next call's `joint_grouped_laplace_loglik`
with it via `b_init`, instead of starting cold from `zeros(m)` every time. One
outer optimisation (Nelder-Mead simplex evaluations plus FD-gradient/-Hessian
stencils) calls this closure with many nearby `value`s, so the previous mode
is typically a very short walk from the next one's. This changes ONLY the
Newton starting point, never the converged answer (see
`joint_grouped_laplace_loglik`'s docstring and
`test/test_grouped_laplace_identity.jl --gate warm_identity`). A non-`:ok`
result leaves the cache untouched (never poisons the next call with a bad
guess); `destination_b_fixed_effects.jl`'s call site does not pass this
keyword and stays cold, unaffected.
"""
function _grouped_nongaussian_objective(data::Matrix{Float64}, trials::Matrix{Float64},
        D::Matrix{Float64}, terms::Vector{GroupingTerm},
        incidences::Vector{SparseMatrixCSC{Float64,Int}}, kind::Symbol;
        dispersion_mode::Symbol=:trait, inner_maxiter::Integer, inner_tol::Float64,
        warm_start_inner::Bool=false)
    p, n = size(data)
    q = size(D, 2)
    mode = _grouped_nongaussian_internal_dispersion_mode(kind, dispersion_mode)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), terms; init=0)
    dispersion_indices = _grouped_nongaussian_dispersion_indices(q, source_coordinates,
        kind, mode, p)
    expected = q + source_coordinates + length(dispersion_indices)
    b_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    return function (value)
        length(value) == expected && all(isfinite, value) || return _NLL_SENTINEL
        try
            gamma = collect(view(value, 1:q))
            loads, uniques, _, used = _grouped_term_unpack(
                view(value, q + 1:q + source_coordinates), p, terms)
            used == source_coordinates || return _NLL_SENTINEL
            family = _grouped_nongaussian_family(kind, value, dispersion_indices, p, n, mode)
            family === nothing && return _NLL_SENTINEL
            W = _grouped_laplace_design(incidences, loads; uniques=uniques)
            b_init = warm_start_inner ? b_cache[] : nothing
            result = joint_grouped_laplace_loglik(family, vec(data), vec(trials), D, gamma, W;
                link=_grouped_nongaussian_link(Val(kind)), maxiter=inner_maxiter, tol=inner_tol,
                b_init=b_init)
            if warm_start_inner && result.status === :ok
                b_cache[] = copy(result.mode)
            end
            return result.status === :ok && result.converged && isfinite(result.loglik) ?
                -result.loglik : _NLL_SENTINEL
        catch
            return _NLL_SENTINEL
        end
    end
end

# ---------------------------------------------------------------------------
# S8: analytic outer gradient of the grouped non-Gaussian Laplace objective
# (docs/design/grouped-analytic-gradient.md, whose alignment table section 6
# names every function below). Replaces the ~2*nθ inner Laplace-fit calls a
# central-difference gradient needs with exactly ONE inner Newton solve plus
# nθ sparse triangular solves against the already-factorised precision `Fo`
# and one `takahashi_selinv` call -- section 4's O(nnz(L)) log-det trace.
# ---------------------------------------------------------------------------

"""
    _grouped_analytic_loglik_gradient(theta, data, trials, D, terms, incidences, kind;
        dispersion_mode, inner_maxiter, inner_tol) -> Vector{Float64} | nothing

`grad L`, the gradient of the Laplace marginal `L(theta)` itself (NOT the
minimised objective `F = -L`; see [`_grouped_analytic_gradient`](@ref)),
computed through the implicit function theorem exactly as derived in
docs/design/grouped-analytic-gradient.md sections 2-5. Always solves the
inner mode COLD (`b_init = nothing`), matching `objective_cold`'s own
convention (section 8.1) -- Newton converges to the same fixed point
regardless of the starting `b`, so this is never a different answer than a
warm-started evaluation at the same `theta`, only a possibly slower one.

Returns `nothing` on ANY failure (non-finite `theta`, invalid packing, a
non-`:ok`/non-converged inner fit, a `nothing` factor) so the caller can
fall back to `_grouped_fd_gradient` -- mirroring `_grouped_nongaussian_
objective`'s own `_NLL_SENTINEL` convention for the value path.
"""
function _grouped_analytic_loglik_gradient(theta::AbstractVector{<:Real},
        data::Matrix{Float64}, trials::Matrix{Float64}, D::Matrix{Float64},
        terms::Vector{GroupingTerm}, incidences::Vector{SparseMatrixCSC{Float64,Int}},
        kind::Symbol; dispersion_mode::Symbol, inner_maxiter::Integer, inner_tol::Real)
    p, n = size(data)
    q = size(D, 2)
    mode = _grouped_nongaussian_internal_dispersion_mode(kind, dispersion_mode)
    source_coordinates = sum(t -> _grouping_term_nparams(t, p), terms; init=0)
    dispersion_indices = _grouped_nongaussian_dispersion_indices(q, source_coordinates, kind, mode, p)
    disp_count = length(dispersion_indices)
    ntheta = q + source_coordinates + disp_count
    (length(theta) == ntheta && all(isfinite, theta)) || return nothing

    thetaf = Float64.(theta)
    gamma = collect(view(thetaf, 1:q))
    loads, uniques, _, used = _grouped_term_unpack(view(thetaf, q + 1:q + source_coordinates), p, terms)
    used == source_coordinates || return nothing
    family = _grouped_nongaussian_family(kind, thetaf, dispersion_indices, p, n, mode)
    family === nothing && return nothing
    W = _grouped_laplace_design(incidences, loads; uniques=uniques)
    y = vec(data); ntrial = vec(trials)
    link = _grouped_nongaussian_link(Val(kind))

    result = try
        joint_grouped_laplace_loglik(family, y, ntrial, D, gamma, W;
            link=link, maxiter=Int(inner_maxiter), tol=Float64(inner_tol), b_init=nothing)
    catch
        return nothing
    end
    (result.status === :ok && result.converged) || return nothing
    Fo = result.factor
    Fo === nothing && return nothing
    bhat = result.mode

    state = _joint_grouped_state(family, D, gamma, W, link, bhat)
    state[1] === :ok || return nothing
    _, eta, mu, me = state

    Npts = length(y)
    s = Vector{Float64}(undef, Npts)
    w = Vector{Float64}(undef, Npts)
    kappa = Vector{Float64}(undef, Npts)
    @inbounds for i in 1:Npts
        rf = _joint_grouped_family_at(family, i)
        s[i] = _glm_score(rf, mu[i], ntrial[i], me[i], y[i])
        w[i] = _glm_obs_weight(rf, mu[i], ntrial[i], me[i], y[i], link, eta[i])
        kappa[i] = _glm_obs_weight_deta(rf, mu[i], ntrial[i], me[i], y[i], link, eta[i])
    end
    all(isfinite, s) && all(isfinite, w) && all(isfinite, kappa) || return nothing

    Sigma = try
        takahashi_selinv(Fo)
    catch
        return nothing
    end
    Wt = SparseMatrixCSC(transpose(W))
    t = _grouped_selinv_row_quadform(Wt, Sigma)

    trait_of(i) = mod1(i, p)
    disp_local = disp_count == 0 ? Int[] : (mode === :shared ? fill(1, p) : collect(1:p))

    gradL = Vector{Float64}(undef, ntheta)

    @inbounds for k in 1:q
        e_expl = view(D, :, k)
        rhs = -(W' * (w .* e_expl))
        u = Fo \ rhs
        edot = e_expl .+ W * u
        wdot = kappa .* edot
        gradL[k] = dot(s, e_expl) - 0.5 * dot(wdot, t)
    end

    @inbounds for local_k in 1:source_coordinates
        k = q + local_k
        dW = _grouped_laplace_design_jacobian(incidences, loads, uniques, terms, local_k)
        e_expl = dW * bhat
        rhs = dW' * s .- (W' * (w .* e_expl))
        u = Fo \ rhs
        edot = e_expl .+ W * u
        wdot = kappa .* edot
        dWt = SparseMatrixCSC(transpose(dW))
        r = _grouped_selinv_row_crossform(Wt, dWt, Sigma)
        gradL[k] = dot(s, e_expl) - dot(w, r) - 0.5 * dot(wdot, t)
    end

    @inbounds for local_k in 1:disp_count
        k = q + source_coordinates + local_k
        ds_drho = zeros(Float64, Npts)
        wdot = zeros(Float64, Npts)
        dl_drho_total = 0.0
        for i in 1:Npts
            disp_local[trait_of(i)] == local_k || continue
            rf = _joint_grouped_family_at(family, i)
            phi = kind === :beta ? rf.α : rf.r
            ds_drho[i] = _glm_score_dphi(rf, mu[i], ntrial[i], me[i], y[i]) * phi
            wdot[i] = _glm_obs_weight_dphi(rf, mu[i], ntrial[i], me[i], y[i], link, eta[i]) * phi
            dl_drho_total += _glm_logpdf_dphi(rf, mu[i], ntrial[i], y[i]) * phi
        end
        rhs = W' * ds_drho
        u = Fo \ rhs
        edot = W * u
        wdot .+= kappa .* edot
        gradL[k] = dl_drho_total - 0.5 * dot(wdot, t)
    end

    all(isfinite, gradL) || return nothing
    return gradL
end

"""
    _grouped_analytic_gradient(theta, data, trials, D, terms, incidences, kind;
        dispersion_mode, inner_maxiter, inner_tol) -> Vector{Float64} | nothing

`grad F = -grad L`, the gradient of the MINIMISED objective (`F(theta) =
-L(theta)`, `src/grouped_nongaussian_fit.jl:263`; docs/design/grouped-
analytic-gradient.md section 1.4). `nothing` on any failure -- the caller
falls back to `_grouped_fd_gradient`.
"""
function _grouped_analytic_gradient(theta::AbstractVector{<:Real},
        data::Matrix{Float64}, trials::Matrix{Float64}, D::Matrix{Float64},
        terms::Vector{GroupingTerm}, incidences::Vector{SparseMatrixCSC{Float64,Int}},
        kind::Symbol; dispersion_mode::Symbol, inner_maxiter::Integer, inner_tol::Real)
    gradL = _grouped_analytic_loglik_gradient(theta, data, trials, D, terms, incidences, kind;
        dispersion_mode=dispersion_mode, inner_maxiter=inner_maxiter, inner_tol=inner_tol)
    gradL === nothing && return nothing
    return -gradL
end

function _grouped_nongaussian_trials(Y::AbstractMatrix, N, kind::Symbol)
    if kind === :binomial
        N === nothing && throw(ArgumentError("Binomial grouped Laplace fitting requires N trials"))
        size(N) == size(Y) || throw(DimensionMismatch("N must have the same shape as Y"))
        trials = try
            Matrix{Float64}(N)
        catch
            throw(ArgumentError("N must be representable as Float64"))
        end
        all(isfinite, trials) || throw(ArgumentError("N must be finite"))
        return trials
    end
    N === nothing || throw(ArgumentError("N trials are only used with Binomial grouped Laplace fitting"))
    return ones(Float64, size(Y))
end

"""
    fit_grouped_nongaussian(Y; family, terms, unit=nothing, unit_obs=nothing,
                             cluster=nothing, cluster2=nothing, N=nothing, ...)

Internal joint outer fit for grouped Poisson, Binomial, Beta, and NB2 models.
It reuses `GroupingTerm` packing and one global joint-Laplace random-effect
mode. Public fitting uses `fit_gllvm` or the grouping formula route; marginal
interval diagnostics use `grouped_nongaussian_intervals`. The development route
does not establish frozen-R parity, recovery, or coverage qualification.

`warm_start_inner` (S7c, default `true`): each of the many inner Laplace-fit
calls the outer optimiser makes (Nelder-Mead simplex evaluations plus the FD
gradient/Hessian stencils) seeds its Newton solve from the previous call's
converged mode instead of `zeros(m)`, since nearby outer-parameter values
share a nearby mode. This changes only how fast each inner solve converges,
never the converged answer (see `joint_grouped_laplace_loglik`'s `b_init`
docstring); pass `warm_start_inner=false` to recover the pre-S7c cold-start
behaviour.
"""
function fit_grouped_nongaussian(Y::AbstractMatrix{<:Real}; family, terms,
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        N=nothing, X=nothing, coefficient_names=nothing, start=nothing,
        dispersion::Symbol=:trait,
        g_tol::Real=1e-4, iterations::Integer=100,
        inner_maxiter::Integer=100, inner_tol::Real=1e-8,
        warm_start_inner::Bool=true, analytic_gradient::Bool=true)
    p, n = size(Y)
    p > 0 && n >= 2 || throw(ArgumentError("grouped fitting needs at least one trait and two observations"))
    all(isfinite, Y) || throw(ArgumentError("grouped fitting requires finite complete responses"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    inner_maxiter >= 0 || throw(ArgumentError("inner_maxiter must be non-negative"))
    isfinite(inner_tol) && inner_tol > 0 || throw(ArgumentError("inner_tol must be finite and positive"))
    kind = _grouped_nongaussian_kind(family)
    mode = _grouped_nongaussian_dispersion_mode(kind, dispersion)
    all(term -> term isa GroupingTerm, terms) ||
        throw(ArgumentError("terms must contain GroupingTerm objects"))
    termvec = GroupingTerm[terms...]
    _grouped_warn_identification(termvec, p)
    labels = _grouped_labels(n, termvec; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in labels]
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses exceed Float64 range"))
    trials = _grouped_nongaussian_trials(data, N, kind)
    default_means = X === nothing
    D = default_means ? _trait_mean_design(p, n) : _source_mean_design(X, p, n)
    q = size(D, 2)
    names = _source_coefficient_names(coefficient_names, q; default_trait_names=default_means)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), termvec; init=0)
    dispersion_indices = _grouped_nongaussian_dispersion_indices(q, source_coordinates,
        kind, mode, p)
    total = q + source_coordinates + length(dispersion_indices)
    theta = if start === nothing
        _grouped_nongaussian_initial_parameters(data, trials, D, termvec, kind, family, mode)
    else
        length(start) == total || throw(DimensionMismatch("start has $(length(start)) coordinates; expected $total"))
        all(x -> x isa Real && isfinite(x), start) || throw(ArgumentError("start must be finite and real"))
        Float64.(start)
    end
    # S7c: warm-starting is confined to the Nelder-Mead VALUE-ONLY search
    # below. FD-gradient/-Hessian differencing (`_grouped_fd_gradient`/
    # `_grouped_fd_hessian`) needs the objective to vary SMOOTHLY and
    # CONSISTENTLY between the +h/-h stencil points it differences — cached
    # warm starts break that: which mode a stencil point's inner Newton
    # solve lands on (to within inner_tol, not to full machine precision)
    # depends on the arbitrary cache state left by whichever theta was
    # evaluated most recently, so the tiny (~inner_tol-scale) residual
    # noise floor is INCONSISTENT across nearby stencil points instead of
    # both referencing the same b=0 start — and dividing that inconsistent
    # noise by the small FD step size amplifies it by ~1/h. MEASURED: on a
    # 4-source common=true Destination B fixture (test/test_destination_b_
    # joint_other_families.jl) this took the fitter's own FD gradient norm
    # from ~1e-7 (cold) to ~1e-4-2e-4 (warm-throughout) — enough to flip
    # `converged` under the 1e-4 g_tol. So `objective_cold` (always cold,
    # `warm_start_inner=false`) is used for every FD-differenced quantity
    # and for BFGS refinement (which needs a gradient at every line-search
    # trial), reproducing origin/main's numerics there exactly;
    # `objective_warm` (the caller's actual `warm_start_inner`) is used
    # ONLY for the plain value-only Nelder-Mead search, which does not
    # difference nearby evaluations and tolerates inner_tol-scale noise.
    objective_cold = _grouped_nongaussian_objective(data, trials, D, termvec, incidences, kind;
        dispersion_mode=mode, inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol),
        warm_start_inner=false)
    objective_warm = warm_start_inner ?
        _grouped_nongaussian_objective(data, trials, D, termvec, incidences, kind;
            dispersion_mode=mode, inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol),
            warm_start_inner=true) :
        objective_cold
    objective = objective_cold   # used below for the final, reported diagnostics
    initial_value = objective_cold(theta)
    isfinite(initial_value) && !_nll_failed(initial_value) ||
        throw(ArgumentError("start produces an invalid grouped Laplace objective"))
    # The joint Laplace domain can invalidate a finite-difference neighbour.
    # `_grouped_fd_gradient` correctly marks such a stencil NaN; feeding it to
    # a line-search method would turn a rejected stencil into an Optim error.
    # Use a value-only outer search, then require a valid FD gradient/Hessian
    # for convergence and observed-marginal inference below.
    result = Optim.optimize(objective_warm, theta, Optim.NelderMead(),
        Optim.Options(g_tol=Float64(g_tol), iterations=Int(iterations)))
    # A valid simplex endpoint can still be non-stationary because Nelder-Mead
    # stops on objective/simplex geometry, not this fitter's FD gradient norm.
    # Refine only when every initial BFGS stencil is valid. Invalid stencils
    # retain the value-only result and cannot be smuggled into a line search.
    candidate = collect(Optim.minimizer(result))
    # S8: analytic outer gradient (implicit function theorem + selected
    # inverse, docs/design/grouped-analytic-gradient.md), used as the
    # `gradient!` Optim.BFGS refines against. `analytic_gradient=false`
    # recovers the exact pre-S8 all-FD path -- kept reachable as both a
    # deliberate fallback (any theta where the analytic value comes back
    # non-finite falls through to the SAME `_grouped_fd_gradient` call the
    # pre-S8 code always used) and a test oracle
    # (test/test_grouped_analytic_grad.jl).
    grad_fn = value -> begin
        analytic_gradient || return _grouped_fd_gradient(objective_cold, value)
        g = _grouped_analytic_gradient(value, data, trials, D, termvec, incidences, kind;
            dispersion_mode=mode, inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol))
        (g === nothing || !all(isfinite, g)) ? _grouped_fd_gradient(objective_cold, value) : g
    end
    candidate_gradient = grad_fn(candidate)
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= grad_fn(value))
        refined = try
            Optim.optimize(objective_cold, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol=Float64(g_tol), iterations=Int(iterations)))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = objective_cold(refined_estimate)
            if isfinite(refined_value) && !_nll_failed(refined_value) &&
                    refined_value <= objective_cold(candidate)
                result = refined
            end
        end
    end
    estimate = collect(Optim.minimizer(result))
    value = objective(estimate)
    valid = isfinite(value) && !_nll_failed(value)
    gradient = valid ? _grouped_fd_gradient(objective, estimate) : fill(Inf, length(estimate))
    gradient_norm = all(isfinite, gradient) ? maximum(abs, gradient) : Inf
    H = valid ? _grouped_fd_hessian(objective, estimate) : fill(NaN, length(estimate), length(estimate))
    min_eigenvalue = all(isfinite, H) ? eigmin(Symmetric(H)) : NaN
    pd_hessian = isfinite(min_eigenvalue) && min_eigenvalue > 0
    converged = Optim.converged(result) && valid && gradient_norm <= g_tol
    reason = _grouped_stopping_reason(converged, valid, Optim.iterations(result),
        iterations, gradient_norm, g_tol)
    loads, uniques, covariances, _ = _grouped_term_unpack(
        view(estimate, q + 1:q + source_coordinates), p, termvec)
    final_family = _grouped_nongaussian_family(kind, estimate, dispersion_indices, p, n, mode)
    final_W = _grouped_laplace_design(incidences, loads; uniques=uniques)
    inner = final_family === nothing ? nothing : joint_grouped_laplace_loglik(
        final_family, vec(data), vec(trials), D, collect(view(estimate, 1:q)), final_W;
        link=_grouped_nongaussian_link(Val(kind)), maxiter=Int(inner_maxiter), tol=Float64(inner_tol))
    inner_status = inner === nothing ? :invalid_family : inner.status
    natural_dispersion = _grouped_nongaussian_dispersion(estimate, dispersion_indices, kind, mode)
    return GroupedNonGaussianFit(collect(estimate[1:q]), final_family, natural_dispersion, covariances,
        termvec, estimate, valid ? -value : -Inf, converged, gradient_norm,
        min_eigenvalue, pd_hessian, Optim.iterations(result), reason, D, names,
        _grouped_nongaussian_parameter_labels(names, p, termvec, kind, mode), (p, n),
        copy(data), copy(trials), copy.(incidences), kind, mode, inner_status)
end

function _grouped_nongaussian_primary_targets(fit::GroupedNonGaussianFit)
    p, _ = fit.response_shape
    q = size(fit.mean_design, 2)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), fit.terms; init=0)
    unpack = theta -> _grouped_term_unpack(view(theta, q + 1:q + source_coordinates),
        p, fit.terms)
    targets = NamedTuple[]
    for j in 1:q
        index = j
        push!(targets, (name=Symbol("fixed.$(fit.parameter_labels[index])"),
            value=theta -> theta[index], transform=:identity))
    end
    for term_index in eachindex(fit.terms)
        prefix = String(fit.terms[term_index].name)
        covariance = theta -> unpack(theta)[3][term_index]
        for j in 1:p
            index = j
            push!(targets, (name=Symbol("$(prefix).variance[$index]"),
                value=theta -> covariance(theta)[index, index], transform=:log))
            push!(targets, (name=Symbol("$(prefix).sd[$index]"),
                value=theta -> sqrt(covariance(theta)[index, index]), transform=:log))
        end
        if fit.terms[term_index].mode !== :indep
            for column in 2:p, row in 1:(column - 1)
                i, j = row, column
                push!(targets, (name=Symbol("$(prefix).covariance[$i,$j]"),
                    value=theta -> covariance(theta)[i, j], transform=:identity))
            end
        end
        if fit.terms[term_index].mode === :latent && fit.terms[term_index].unique
            for j in 1:p
                index = j
                push!(targets, (name=Symbol("$(prefix).unique_variance[$index]"),
                    value=theta -> unpack(theta)[2][term_index][index], transform=:log))
            end
        end
    end
    dispersion_count = _grouped_nongaussian_dispersion_count(fit.family_kind,
        fit.dispersion_mode, p)
    if fit.family_kind === :beta
        for trait in 1:dispersion_count
            index = length(fit.parameters) - dispersion_count + trait
            name = fit.dispersion_mode === :shared ? :beta_precision :
                Symbol("beta_precision[$trait]")
            push!(targets, (name=name, value=theta -> exp(theta[index]), transform=:log))
        end
    elseif fit.family_kind === :nb2
        for trait in 1:dispersion_count
            index = length(fit.parameters) - dispersion_count + trait
            name = fit.dispersion_mode === :shared ? :nb2_size :
                Symbol("nb2_size[$trait]")
            push!(targets, (name=name, value=theta -> exp(theta[index]), transform=:log))
        end
    end
    return targets
end

"""
    grouped_nongaussian_intervals(Y, fit; unit=nothing, unit_obs=nothing,
                                  cluster=nothing, cluster2=nothing, ...)

Observed-marginal Wald intervals for a `GroupedNonGaussianFit`, including all
nuisance coordinates. Targets include fixed effects, group covariance entries,
and fitted Beta precision or NB2 size. Suitable positive interior targets use
log-scale intervals. No profile fallback is implemented here.

Supplied response, trials and labels must reproduce the fit provenance exactly.
Omitted `dispersion` inherits the fitted mode; an explicit mismatch is rejected.
Inspect overall and per-target statuses: nonconvergence, structural redundancy
or boundaries can make some or all intervals unavailable.
"""
function grouped_nongaussian_intervals(Y::AbstractMatrix{<:Real}, fit::GroupedNonGaussianFit;
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        N=nothing, dispersion::Union{Nothing,Symbol}=nothing,
        level::Real=0.95, gradient_tolerance::Real=1e-4,
        inner_maxiter::Integer=100, inner_tol::Real=1e-8)
    p, n = fit.response_shape
    size(Y) == (p, n) || throw(DimensionMismatch("Y shape must match the fitted response shape"))
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses must be finite"))
    requested_mode = dispersion === nothing ? fit.dispersion_mode :
        _grouped_nongaussian_dispersion_mode(fit.family_kind, dispersion)
    requested_mode == fit.dispersion_mode ||
        throw(ArgumentError("dispersion mode must match the grouped fit"))
    trials = _grouped_nongaussian_trials(data, N, fit.family_kind)
    labels = _grouped_labels(n, fit.terms; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in labels]
    data == fit.response ||
        throw(ArgumentError("Y must exactly match the response used to fit grouped uncertainty"))
    trials == fit.trials ||
        throw(ArgumentError("N must exactly match the trials used to fit grouped uncertainty"))
    length(incidences) == length(fit.incidences) &&
        all(incidences[i] == fit.incidences[i] for i in eachindex(incidences)) ||
        throw(ArgumentError("grouping labels must induce the same incidences used to fit grouped uncertainty"))
    objective = _grouped_nongaussian_objective(fit.response, fit.trials, fit.mean_design,
        fit.terms, fit.incidences, fit.family_kind;
        dispersion_mode=fit.dispersion_mode,
        inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol))
    return _marginal_target_intervals(objective, fit.parameters,
        _grouped_nongaussian_primary_targets(fit);
        structural_redundancy=!isempty(_grouped_identification_diagnostics(fit.terms, p)),
        converged=fit.converged, level=level, gradient_tolerance=gradient_tolerance)
end
