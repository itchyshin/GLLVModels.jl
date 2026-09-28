# One-step implicit gradient of the iSDM Laplace marginal
# (docs/design/isdm-port-spec.md, Q5; the pattern of src/laplace_grad.jl:1-26).
#
# Per cell: find the mode zhat concretely at the primal theta, then form the
# single differentiable Newton step
#     z(theta) = zhat + A_obs(zhat, theta)^-1 g(zhat; theta),
#     g(z; theta) = sum_o s_o(z; theta) lambda_t(o) - z,
# which equals zhat at the primal theta but whose theta-derivative is the
# implicit dzhat/dtheta, and take ForwardDiff of the marginal evaluated at
# z(theta). Three details the spec review fixed:
#   * A_obs uses the OBSERVED row weights -d2 l_o / d eta2, because the implicit
#     function theorem needs the true Jacobian of g (on cloglog rows Fisher is
#     not the observed curvature; on Poisson-log rows the two agree);
#   * the offset is carried explicitly inside eta_o(theta) = x_o'b + offset_o +
#     lambda_t(o)'z, so an offset never forces a finite-difference fallback;
#   * Poisson rows use the AD-friendly `_pois_logpmf`.
# A cell whose mode search did not converge makes the gradient `nothing`, and
# `_optimize_with_analytic` falls back to central finite differences for that
# theta (spec risk R6).

# theta = [b; pack_lambda(Λ)] (the lower-triangular layout of src/packing.jl,
# diagonal first, which is also gllvmTMB's theta_rr_B layout).
function _isdm_unpack(θ::AbstractVector, pX::Int, p::Int, K::Int)
    b = θ[1:pX]
    Λ = K == 0 ? zeros(eltype(θ), p, 0) : unpack_lambda(θ[(pX + 1):end], p, K)
    return b, Λ
end

_isdm_pack(b::AbstractVector, Λ::AbstractMatrix) =
    size(Λ, 2) == 0 ? collect(float.(b)) : vcat(float.(b), pack_lambda(Λ))

# Differentiable one-step value of one cell at the concrete mode zhat.
function _isdm_cell_onestep(y, fid, tr, eta0, Λ, ẑ::AbstractVector)
    K = size(Λ, 2)
    T = promote_type(eltype(Λ), eltype(eta0))
    g = Vector{T}(undef, K)
    A = Matrix{T}(undef, K, K)
    @inbounds for k in 1:K
        g[k] = -ẑ[k]
        for j in 1:K
            A[j, k] = j == k ? one(T) : zero(T)
        end
    end
    @inbounds for o in eachindex(y)
        t = tr[o]
        η = _isdm_eta(eta0[o], Λ, t, ẑ)
        s = _isdm_row_score(fid[o], y[o], η)
        W = _isdm_row_obs_weight(fid[o], y[o], η)
        for k in 1:K
            g[k] += s * Λ[t, k]
            for j in 1:K
                A[j, k] += W * Λ[t, j] * Λ[t, k]
            end
        end
    end
    z = ẑ .+ Symmetric(A) \ g
    return _isdm_cell_value_at(y, fid, tr, eta0, Λ, z)
end

"""
    isdm_laplace_grad(table::IsdmTable, θ; K = table.K, maxiter = 100, tol = 1e-9)
        -> Union{Vector{Float64}, Nothing}

Gradient of [`isdm_marginal_loglik_laplace`](@ref) with respect to
`θ = [b; pack_lambda(Λ)]` by the one-step implicit method. Returns `nothing`
when any cell's mode search fails at `θ`.
"""
function isdm_laplace_grad(table::IsdmTable, θ::AbstractVector; K::Integer = table.K,
        maxiter::Integer = 100, tol::Real = 1e-9)
    pX = size(table.X, 2); p = length(table.trait_levels)
    b, Λ = _isdm_unpack(θ, pX, p, K)
    if K == 0
        f0(θd) = begin
            bd = θd[1:pX]
            eta0 = table.X * bd .+ table.offset
            acc = zero(eltype(θd))
            @inbounds for o in eachindex(table.y)
                acc += _isdm_row_logpdf(table.fid[o], table.y[o], eta0[o])
            end
            acc
        end
        return ForwardDiff.gradient(f0, θ)
    end
    eta0 = table.X * b .+ table.offset
    Ẑ = Matrix{Float64}(undef, K, length(table.rows_by_unit))
    for (u, rows) in enumerate(table.rows_by_unit)
        z, ok = _isdm_cell_mode_retry(view(table.y, rows), view(table.fid, rows),
                                      view(table.trait_id, rows), view(eta0, rows), Λ;
                                      maxiter = maxiter, tol = tol)
        ok || return nothing
        Ẑ[:, u] = z
    end
    function marg(θd)
        bd, Λd = _isdm_unpack(θd, pX, p, K)
        e0 = table.X * bd .+ table.offset
        acc = zero(eltype(θd))
        for (u, rows) in enumerate(table.rows_by_unit)
            acc += _isdm_cell_onestep(view(table.y, rows), view(table.fid, rows),
                                      view(table.trait_id, rows), view(e0, rows), Λd,
                                      view(Ẑ, :, u))
        end
        return acc
    end
    return ForwardDiff.gradient(marg, θ)
end
