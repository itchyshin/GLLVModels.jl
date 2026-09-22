# Joint (not per-unit) Laplace objective for grouped non-Gaussian effects.
#
# This file is deliberately private until the Destination B dispatcher owns a
# user-facing contract.  It accepts a fixed design and fixed family parameters;
# outer optimisation and interval extraction belong to later slices.

"""
    JointGroupedLaplaceResult

Diagnostic result from [`joint_grouped_laplace_loglik`](@ref).  `status == :ok`
is the only successful result.  Invalid data, invalid trials, exhausted Newton
iterations, non-finite conditional evaluations, and a non-positive-definite
observed Laplace precision return `converged == false` and `loglik == -Inf`.
The `precision` field is the observed-curvature joint precision on success.
"""
struct JointGroupedLaplaceResult
    loglik::Float64
    mode::Vector{Float64}
    precision::SparseMatrixCSC{Float64, Int}
    logdet_precision::Float64
    converged::Bool
    status::Symbol
    iterations::Int
    gradient_norm::Float64
    # S8 (analytic outer gradient): the CHOLMOD factor of `precision` at
    # convergence, exposed rather than discarded so the implicit-function-
    # theorem gradient can reuse it directly (nθ triangular solves for
    # dbhat/dtheta plus one `takahashi_selinv` call) instead of
    # refactorising `precision` from scratch. `nothing` on any failure
    # result (no factor was ever computed, or the factorisation itself
    # failed) -- see docs/design/grouped-analytic-gradient.md section 6.
    factor::Union{Nothing,SparseArrays.CHOLMOD.Factor{Float64}}
end

function _joint_grouped_failure(status::Symbol, m::Integer; iterations::Integer = 0)
    return JointGroupedLaplaceResult(-Inf, zeros(Float64, m),
        spdiagm(0 => ones(Float64, m)), NaN, false, status, iterations, Inf, nothing)
end

# ---------------------------------------------------------------------------
# CHOLMOD symbolic-reuse (S7b, leaf-S7b G7b.1/G7b.2). `joint_grouped_laplace_
# loglik`'s inner Newton loop previously called plain `cholesky(Symmetric(...))`
# on `Hf` (Fisher precision) and `Ho` (observed precision) fresh EVERY
# iteration; each such call re-runs CHOLMOD's symbolic analysis (AMD ordering
# + supernodal tree) even though `Hf`/`Ho` share the SAME sparsity pattern for
# the entire lifetime of one `joint_grouped_laplace_loglik` call (`W` is fixed
# for that call; only the diagonal weight VALUES change between Newton
# iterations, never which entries of `W' * diag * W + I` are structurally
# nonzero). `cholesky!(F, A)` reuses `F`'s symbolic factorization and performs
# only the numeric refactorization for `A`'s new values — internally this is
# EXACTLY what `cholesky(A)` already does after its own (redundant, repeated)
# symbolic step, so the numeric result is bit-identical; only the repeated
# AMD/tree work is eliminated. `_grouped_cached_cholesky!` asserts the
# sparsity pattern on every call and falls back to a fresh `cholesky` (with a
# counted fallback) if it ever differs, so a genuine pattern change is never
# silently misfactorized.
#
# Scope is ONE `joint_grouped_laplace_loglik` call (`cache` is a fresh
# `Ref{Any}(nothing)` local to that call, per `_grouped_cached_cholesky!`'s
# call sites below) — not across the ~100 inner-Laplace-fit calls a full
# outer optimisation makes, since each of those calls gets a freshly built
# `W` (the caller's business, not this file's) and this file makes no claim
# about whether that W's PATTERN also stays fixed across outer evaluations.
# `_GROUPED_CHOL_STATS` counts calls/fresh/reused/fallback so
# `test/test_grouped_laplace_identity.jl` can assert the "2 fresh per call,
# 0 thereafter" behaviour without external instrumentation.
mutable struct _GroupedCholStats
    calls::Int      # joint_grouped_laplace_loglik invocations counted
    fresh::Int      # fresh cholesky(...) symbolic+numeric factorizations
    reused::Int     # cholesky!(...) numeric-only refactorizations (cache hit)
    fallback::Int   # a cached factor existed but its sparsity pattern changed
end

const _GROUPED_CHOL_STATS = _GroupedCholStats(0, 0, 0, 0)

_grouped_chol_stats_reset!() = begin
    _GROUPED_CHOL_STATS.calls = 0
    _GROUPED_CHOL_STATS.fresh = 0
    _GROUPED_CHOL_STATS.reused = 0
    _GROUPED_CHOL_STATS.fallback = 0
    nothing
end

_grouped_chol_stats() = (calls = _GROUPED_CHOL_STATS.calls, fresh = _GROUPED_CHOL_STATS.fresh,
    reused = _GROUPED_CHOL_STATS.reused, fallback = _GROUPED_CHOL_STATS.fallback)

# `cache[]` holds `nothing` (no factor cached yet for this call) or
# `(factor, rowval, colptr)` — the exact CSC pattern the factor was built
# for. Returns the (possibly freshly built) CHOLMOD factor for `A`; `check`
# is forwarded so callers keep inspecting `issuccess` for a non-PD matrix
# exactly as before (never throws for that reason).
function _grouped_cached_cholesky!(cache::Base.RefValue,
        A::Symmetric{Float64, <:SparseMatrixCSC{Float64, Int}}; check::Bool = false)
    S = parent(A)
    cached = cache[]
    if cached !== nothing
        factor, rowval, colptr = cached
        if size(factor) == size(S) && rowval == S.rowval && colptr == S.colptr
            _GROUPED_CHOL_STATS.reused += 1
            return cholesky!(factor, A; check = check)
        end
        _GROUPED_CHOL_STATS.fallback += 1
    end
    factor = cholesky(A; check = check)
    _GROUPED_CHOL_STATS.fresh += 1
    cache[] = (factor, copy(S.rowval), copy(S.colptr))
    return factor
end

"""
    grouped_trait_design(incidence, trait_factors) -> SparseMatrixCSC

Build the sparse random-effect design for responses stacked as `vec(Y)`, where
`Y` is traits-by-units.  `incidence` is units-by-groups and `trait_factors` is
traits-by-random-effect factors; the result is `kron(incidence,
trait_factors)`.  A row of `incidence` may contain several nonzeros (crossed
groups): those entries remain in the *same* global design matrix.
"""
function grouped_trait_design(incidence::AbstractMatrix, trait_factors::AbstractMatrix)
    all(isfinite, incidence) || throw(ArgumentError("incidence must be finite"))
    all(isfinite, trait_factors) || throw(ArgumentError("trait_factors must be finite"))
    return sparse(kron(sparse(Float64.(incidence)), Float64.(trait_factors)))
end

_joint_grouped_link_ok(::Poisson, ::LogLink) = true
_joint_grouped_link_ok(::Binomial, ::LogitLink) = true
_joint_grouped_link_ok(::Beta, ::LogitLink) = true
_joint_grouped_link_ok(::NegativeBinomial, ::LogLink) = true
_joint_grouped_link_ok(family, link) = false

_joint_grouped_family_at(family, i) = family
_joint_grouped_family_at(family::AbstractVector, i) = family[i]
function _joint_grouped_link_ok(families::AbstractVector, link)
    isempty(families) && return false
    homogeneous = all(f -> f isa Beta, families) || all(f -> f isa NegativeBinomial, families)
    return homogeneous && all(f -> _joint_grouped_link_ok(f, link), families)
end

function _joint_grouped_data_status(families::AbstractVector, y::Vector{Float64}, n::Vector{Float64})
    length(families) == length(y) == length(n) || return :invalid_dimensions
    for i in eachindex(y)
        status = _joint_grouped_data_status(families[i], [y[i]], [n[i]])
        status == :ok || return status
    end
    return :ok
end

function _joint_grouped_data_status(family, y::Vector{Float64}, n::Vector{Float64})
    all(isfinite, y) || return :invalid_data
    all(isfinite, n) || return :invalid_trials
    if family isa Poisson || family isa NegativeBinomial
        all(v -> v >= 0 && isinteger(v), y) || return :invalid_data
        if family isa NegativeBinomial
            (isfinite(family.r) && family.r > 0 && family.p == 0.5) || return :invalid_family
        end
    elseif family isa Binomial
        all(v -> v >= 1 && isinteger(v), n) || return :invalid_trials
        all(i -> y[i] >= 0 && isinteger(y[i]) && y[i] <= n[i], eachindex(y)) ||
            return :invalid_data
    elseif family isa Beta
        all(v -> 0 < v < 1, y) || return :invalid_data
        (isfinite(family.α) && family.α > 0 && family.β == 1.0) || return :invalid_family
    else
        return :unsupported_family
    end
    return :ok
end

"""
    _joint_grouped_state(...) -> (:ok, eta, mu, mu_eta) | (status, nothing, nothing, nothing)

Return the differentiable conditional state.  The generic family core clamps
`eta` and family means for its legacy fitters; differentiating an *unclamped*
link through that altered value would be wrong.  This fixed-parameter kernel
therefore rejects that non-differentiable saturation domain explicitly rather
than reporting a false Newton mode or curvature.
"""
function _joint_grouped_state(family, X::Matrix{Float64}, beta::Vector{Float64},
        W::SparseMatrixCSC{Float64, Int}, link::Link, b::Vector{Float64})
    eta = X * beta + W * b
    all(isfinite, eta) || return (:nonfinite_linear_predictor, nothing, nothing, nothing)
    all(e -> -30.0 < e < 30.0, eta) || return (:saturated_domain, nothing, nothing, nothing)
    raw_mu = linkinv.(Ref(link), eta)
    all(isfinite, raw_mu) || return (:nonfinite_mean, nothing, nothing, nothing)
    mu = [_clamp_mu(_joint_grouped_family_at(family,i), raw_mu[i]) for i in eachindex(raw_mu)]
    all(i -> raw_mu[i] == mu[i], eachindex(mu)) ||
        return (:saturated_domain, nothing, nothing, nothing)
    me = mu_eta.(Ref(link), eta)
    all(isfinite, me) || return (:nonfinite_link_derivative, nothing, nothing, nothing)
    return (:ok, eta, mu, me)
end

function _joint_grouped_logpost(family, y::Vector{Float64}, n::Vector{Float64},
        X::Matrix{Float64}, beta::Vector{Float64}, W::SparseMatrixCSC{Float64, Int},
        link::Link, b::Vector{Float64}; state = nothing)
    current = state === nothing ? _joint_grouped_state(family, X, beta, W, link, b) : state
    current[1] === :ok || return NaN
    _, _, mu, _ = current
    ll = -0.5 * dot(b, b)
    @inbounds for i in eachindex(y)
        term = _glm_logpdf(_joint_grouped_family_at(family,i), mu[i], n[i], y[i])
        isfinite(term) || return -Inf
        ll += term
    end
    return ll
end

function _joint_grouped_components(family, y::Vector{Float64}, n::Vector{Float64},
        X::Matrix{Float64}, beta::Vector{Float64}, W::SparseMatrixCSC{Float64, Int},
        link::Link, b::Vector{Float64}; state = nothing)
    current = state === nothing ? _joint_grouped_state(family, X, beta, W, link, b) : state
    current[1] === :ok || return nothing
    _, eta, mu, me = current
    score = Vector{Float64}(undef, length(y))
    fisher = Vector{Float64}(undef, length(y))
    observed = Vector{Float64}(undef, length(y))
    @inbounds for i in eachindex(y)
        row_family = _joint_grouped_family_at(family,i)
        score[i] = _glm_score(row_family, mu[i], n[i], me[i], y[i])
        fisher[i] = _glm_weight(row_family, mu[i], n[i], me[i])
        observed[i] = _glm_obs_weight(row_family, mu[i], n[i], me[i], y[i], link, eta[i])
    end
    all(isfinite, score) && all(isfinite, fisher) && all(isfinite, observed) || return nothing
    Hf = sparse(W' * spdiagm(0 => fisher) * W + spdiagm(0 => ones(Float64, size(W, 2))))
    Ho = sparse(W' * spdiagm(0 => observed) * W + spdiagm(0 => ones(Float64, size(W, 2))))
    return score, Hf, Ho
end

# ---------------------------------------------------------------------------
# S8: analytic outer gradient -- family-generic curvature/dispersion
# derivatives (docs/design/grouped-analytic-gradient.md section 6 alignment
# table). Every function here differentiates the SAME function the objective
# already calls (`_glm_obs_weight`, `_glm_logpdf`, `_glm_score`), never a
# re-derivation of it, so a family whose curvature is a hand-coded analytic
# override (NegativeBinomial's `_glm_obs_weight` at src/families/negbin.jl:34)
# gets its kappa/dispersion derivatives from differentiating THAT override,
# not from a generic third-order-AD reconstruction that would silently
# target a different function than the one the log-det actually uses (the
# hazard docs/design/grouped-analytic-gradient.md section 5 names explicitly
# by pointing at src/laplace_grad.jl:308-316's warning).
# ---------------------------------------------------------------------------

"""
    _glm_obs_weight_deta(family, μ, n, me, y, link, η) -> κ = dw/dη

`κ_i = dw_i/dη_i = -d³ℓ_i/dη_i³`, the derivative of the observed conditional
curvature `w = _glm_obs_weight(...)` (src/families/laplace.jl:260, or a
family's own override) with respect to η. Implemented by nesting one more
`ForwardDiff.derivative` around whatever `_glm_obs_weight` method actually
dispatches for this family -- the generic two-nested-derivative default for
Poisson/Binomial/Beta (three nested derivatives total here), or a single
derivative of NegativeBinomial's closed-form override. This is the "third
nested ForwardDiff" of docs/design/grouped-analytic-gradient.md section 7.6,
generalised to dispatch correctly rather than re-deriving from `_glm_logpdf`
directly (see the file header note above).
"""
function _glm_obs_weight_deta(family, μ, n, me, y, link::Link, η)
    w = ηv -> _glm_obs_weight(family, _clamp_mu(family, linkinv(link, ηv)), n,
        mu_eta(link, ηv), y, link, ηv)
    return ForwardDiff.derivative(w, η)
end

# `_glm_logpdf_dphi`, `_glm_obs_weight_dphi`, `_glm_score_dphi`: sensitivity
# of the log-density, the observed weight, and the score to the family's
# NATURAL dispersion parameter (Beta's φ, NB2's r) -- the outer parameter
# `rho` is the LOG of this natural parameter, so the caller multiplies by
# the natural value itself (d(exp(rho))/d(rho) = exp(rho)) to get the
# derivative wrt `rho`, exactly the `exp(theta_j)` scale already used for
# the unique-variance block (section 1.2). `_glm_score_dphi` is not named in
# the design doc's section 6 alignment table, but section 3's `v_k` formula
# needs `ds/drho` for any `theta_k` in the dispersion block (a Beta/NB2
# score is an explicit function of φ/r, so the mode shifts with dispersion
# too) -- added here to keep that term from being silently dropped; see the
# implementation report for this note made explicit.
_glm_logpdf_dphi(f::Beta, μ, n, y) =
    ForwardDiff.derivative(φ -> _glm_logpdf(_with_dispersion(f, φ), μ, n, y), f.α)
_glm_logpdf_dphi(f::NegativeBinomial, μ, n, y) =
    ForwardDiff.derivative(r -> _glm_logpdf(_with_dispersion(f, r), μ, n, y), f.r)

_glm_obs_weight_dphi(f::Beta, μ, n, me, y, link::Link, η) =
    ForwardDiff.derivative(φ -> _glm_obs_weight(_with_dispersion(f, φ), μ, n, me, y, link, η), f.α)
_glm_obs_weight_dphi(f::NegativeBinomial, μ, n, me, y, link::Link, η) =
    ForwardDiff.derivative(r -> _glm_obs_weight(_with_dispersion(f, r), μ, n, me, y, link, η), f.r)

_glm_score_dphi(f::Beta, μ, n, me, y) =
    ForwardDiff.derivative(φ -> _glm_score(_with_dispersion(f, φ), μ, n, me, y), f.α)
_glm_score_dphi(f::NegativeBinomial, μ, n, me, y) =
    ForwardDiff.derivative(r -> _glm_score(_with_dispersion(f, r), μ, n, me, y), f.r)

# `_selinv_get`: a STRUCTURAL lookup into a `takahashi_selinv` output --
# throws rather than silently returning 0.0 for an absent entry, per
# docs/design/grouped-analytic-gradient.md section 4's own recommendation
# ("the implementation should assert that every (j,l) it reads from Sigma
# is structurally present rather than silently reading zero"). Reuses
# `_csc_rowidx` from src/takahashi_selinv.jl (not modified by this slice).
function _selinv_get(Sigma::SparseMatrixCSC{Float64,Int}, j::Integer, l::Integer)
    idx = _csc_rowidx(Sigma.colptr, Sigma.rowval, l, j)
    idx == -1 && throw(ArgumentError(
        "selected inverse missing entry ($j,$l): pattern(A) not covered by " *
        "pattern(L+Lᵀ) -- see docs/design/grouped-analytic-gradient.md section 4"))
    return Sigma.nzval[idx]
end

"""
    _grouped_selinv_row_quadform(Wt, Sigma) -> Vector{Float64}

`t_i = (W Σ Wᵀ)_{ii}` for every response row `i`, where `Σ` is a selected
inverse (`takahashi_selinv` output) at (a superset of) `pattern(A)`. `Wt` is
`sparse(transpose(W))` (row `i` of `W` = column `i` of `Wt`), passed in so a
caller computing several row-forms against the SAME `W` builds the
transpose once. Computed once per gradient call and reused across every
`theta_k` (docs/design/grouped-analytic-gradient.md section 4).
"""
function _grouped_selinv_row_quadform(Wt::SparseMatrixCSC{Float64,Int}, Sigma::SparseMatrixCSC{Float64,Int})
    N = size(Wt, 2)
    t = zeros(Float64, N)
    @inbounds for i in 1:N
        rng = Wt.colptr[i]:(Wt.colptr[i + 1] - 1)
        cols = view(Wt.rowval, rng)
        vals = view(Wt.nzval, rng)
        acc = 0.0
        for a in eachindex(cols)
            j = cols[a]; wij = vals[a]
            for b in eachindex(cols)
                l = cols[b]; wil = vals[b]
                acc += wij * wil * _selinv_get(Sigma, j, l)
            end
        end
        t[i] = acc
    end
    return t
end

"""
    _grouped_selinv_row_crossform(Wt, dWt, Sigma) -> Vector{Float64}

`r_i(k) = (W Σ (dk W)ᵀ)_{ii}` for every response row `i`. `dWt` is
`sparse(transpose(dk W))`; rows of `dk W` with no nonzero entries (every
response outside the one grouping term `theta_k` touches) contribute `0.0`
without touching `Σ` at all.
"""
function _grouped_selinv_row_crossform(Wt::SparseMatrixCSC{Float64,Int},
        dWt::SparseMatrixCSC{Float64,Int}, Sigma::SparseMatrixCSC{Float64,Int})
    N = size(Wt, 2)
    r = zeros(Float64, N)
    @inbounds for i in 1:N
        rngW = Wt.colptr[i]:(Wt.colptr[i + 1] - 1)
        rngD = dWt.colptr[i]:(dWt.colptr[i + 1] - 1)
        isempty(rngD) && continue
        acc = 0.0
        for a in rngW
            j = Wt.rowval[a]; wij = Wt.nzval[a]
            for b in rngD
                l = dWt.rowval[b]; dwil = dWt.nzval[b]
                acc += wij * dwil * _selinv_get(Sigma, j, l)
            end
        end
        r[i] = acc
    end
    return r
end

"""
    joint_grouped_laplace_loglik(family, y, n, X, beta, W;
        link, maxiter=100, tol=1e-8, b_init=nothing) -> JointGroupedLaplaceResult

Compute a fixed-parameter Laplace approximation for `eta = X * beta + W * b`
with one global `b ~ N(0, I)`.  `W` must already contain all grouped effects;
`family` may also be a homogeneous Beta or NB2 marker vector, one element per
response, for trait-specific dispersion. The caller supplies trait-fast ordering.
use [`grouped_trait_design`](@ref) for an incidence-by-trait-factor design.
The joint mode is located by observed Newton steps with Fisher-scoring fallback,
while the reported
log-determinant is assembled from observed conditional curvature.  No
independent-per-unit mode solve is performed.

`b_init` (S7c), when supplied, seeds the Newton iteration's starting `b`
instead of the cold `zeros(m)` default — e.g. the converged mode from a
nearby outer-parameter evaluation (an FD stencil point or a neighbouring
Nelder-Mead simplex vertex), so the walk to the mode is short instead of
starting over. Newton's iteration converges to the SAME fixed point (the
unique interior mode, to within `tol`) regardless of the starting `b`, so
this is an identity-preserving performance lever, never a different answer —
see `test/test_grouped_laplace_identity.jl --gate warm_identity`. A
length mismatch against `size(W, 2)` returns `_joint_grouped_failure(:invalid_warm_start, ...)`
rather than silently truncating or padding.
"""
function joint_grouped_laplace_loglik(family, y::AbstractVector, n::AbstractVector,
        X::AbstractMatrix, beta::AbstractVector, W::AbstractMatrix;
        link::Link, maxiter::Integer = 100, tol::Real = 1e-8,
        b_init::Union{Nothing,AbstractVector{<:Real}} = nothing)
    m = size(W, 2)
    maxiter >= 0 || return _joint_grouped_failure(:invalid_control, m)
    isfinite(tol) && tol > 0 || return _joint_grouped_failure(:invalid_control, m)
    _joint_grouped_link_ok(family, link) || return _joint_grouped_failure(:unsupported_link, m)
    length(y) == length(n) == size(X, 1) == size(W, 1) ||
        return _joint_grouped_failure(:invalid_dimensions, m)
    size(X, 2) == length(beta) || return _joint_grouped_failure(:invalid_dimensions, m)

    yf = try
        Float64.(y)
    catch
        return _joint_grouped_failure(:invalid_data, m)
    end
    nf = try
        Float64.(n)
    catch
        return _joint_grouped_failure(:invalid_trials, m)
    end
    status = _joint_grouped_data_status(family, yf, nf)
    status === :ok || return _joint_grouped_failure(status, m)

    Xf = try
        Matrix{Float64}(X)
    catch
        return _joint_grouped_failure(:invalid_design, m)
    end
    betaf = try
        Vector{Float64}(beta)
    catch
        return _joint_grouped_failure(:invalid_design, m)
    end
    Wf = try
        sparse(Float64.(W))
    catch
        return _joint_grouped_failure(:invalid_design, m)
    end
    all(isfinite, Xf) && all(isfinite, betaf) && all(isfinite, nonzeros(Wf)) ||
        return _joint_grouped_failure(:invalid_design, m)

    _GROUPED_CHOL_STATS.calls += 1
    ff_cache = Ref{Any}(nothing)   # Fisher-precision (Hf) factor, reused across iterations
    ho_cache = Ref{Any}(nothing)   # observed-precision (Ho) factor, reused across iterations
                                   # (shared by Fn mid-loop and Fo at convergence — same formula)

    b = if b_init === nothing
        zeros(Float64, m)
    else
        length(b_init) == m || return _joint_grouped_failure(:invalid_warm_start, m)
        all(isfinite, b_init) || return _joint_grouped_failure(:invalid_warm_start, m)
        Vector{Float64}(b_init)
    end
    for iter in 1:maxiter
        state = _joint_grouped_state(family, Xf, betaf, Wf, link, b)
        state[1] === :ok || return _joint_grouped_failure(state[1], m; iterations = iter - 1)
        q0 = _joint_grouped_logpost(family, yf, nf, Xf, betaf, Wf, link, b; state = state)
        isfinite(q0) || return _joint_grouped_failure(:nonfinite_objective, m; iterations = iter - 1)
        parts = _joint_grouped_components(family, yf, nf, Xf, betaf, Wf, link, b; state = state)
        parts === nothing && return _joint_grouped_failure(:nonfinite_curvature, m; iterations = iter - 1)
        score, Hf, Ho = parts
        g = Wf' * score - b
        maximum(abs, g; init = 0.0) <= tol * (1 + norm(b)) && begin
            Fo = try
                _grouped_cached_cholesky!(ho_cache, Symmetric(Ho); check = false)
            catch
                return _joint_grouped_failure(:observed_precision_factorization_failed, m;
                    iterations = iter - 1)
            end
            issuccess(Fo) || return _joint_grouped_failure(:observed_precision_not_pd, m; iterations = iter - 1)
            ld = logdet(Fo)
            isfinite(ld) || return _joint_grouped_failure(:nonfinite_logdet, m; iterations = iter - 1)
            return JointGroupedLaplaceResult(q0 - 0.5 * ld, b, Ho, ld, true, :ok,
                iter - 1, maximum(abs, g; init=0.0), Fo)
        end
        Ff = try
            _grouped_cached_cholesky!(ff_cache, Symmetric(Hf); check = false)
        catch
            return _joint_grouped_failure(:fisher_precision_factorization_failed, m;
                iterations = iter - 1)
        end
        issuccess(Ff) || return _joint_grouped_failure(:fisher_precision_not_pd, m; iterations = iter - 1)
        # Observed Newton steps converge quadratically near the mode. Fisher
        # scoring alone can stall on a likelihood rounding plateau even when
        # a neighbouring outer-parameter value has a perfectly regular mode.
        # Keep Fisher scoring as the positive-definite fallback away from it.
        Fn = try
            _grouped_cached_cholesky!(ho_cache, Symmetric(Ho); check = false)
        catch
            nothing
        end
        step_direction = Fn !== nothing && issuccess(Fn) ? Fn \ g : Ff \ g
        all(isfinite, step_direction) || return _joint_grouped_failure(:nonfinite_step, m; iterations = iter - 1)

        accepted = false
        saturated_trial = false
        interior_trial = false
        step = 1.0
        for _ in 1:30
            trial = b .+ step .* step_direction
            trial_state = _joint_grouped_state(family, Xf, betaf, Wf, link, trial)
            if trial_state[1] === :saturated_domain
                saturated_trial = true
                step *= 0.5
                continue
            elseif trial_state[1] !== :ok
                step *= 0.5
                continue
            end
            interior_trial = true
            qtrial = _joint_grouped_logpost(family, yf, nf, Xf, betaf, Wf, link, trial;
                state = trial_state)
            # Near a mode the true ascent can be smaller than summation
            # roundoff. Admit only a roundoff-sized objective decrease AND
            # a strict reduction of the actual joint score, never a looser
            # convergence threshold or a clamped surrogate derivative.
            roundoff_step = false
            if isfinite(qtrial) && qtrial < q0 &&
                    q0 - qtrial <= 32eps(Float64) * (1 + abs(q0))
                trial_parts = _joint_grouped_components(family, yf, nf, Xf,
                    betaf, Wf, link, trial; state=trial_state)
                if trial_parts !== nothing
                    trial_g = Wf' * trial_parts[1] - trial
                    roundoff_step = norm(trial_g) < norm(g)
                end
            end
            if isfinite(qtrial) && (qtrial >= q0 || roundoff_step)
                b = trial
                accepted = true
                break
            end
            step *= 0.5
        end
        accepted || return _joint_grouped_failure(
            saturated_trial && !interior_trial ? :saturated_domain : :line_search_failed, m; iterations = iter)
    end
    return _joint_grouped_failure(:nonconvergence, m; iterations = maxiter)
end
