# Profile-likelihood Gaussian GLLVM NLL — MixedModels.jl-style speed pass.
#
# Inspired by Bates et al.'s `LinearMixedModel.objective!` (MixedModels.jl,
# JuliaStats): the residual variance σ²_eps is profiled out analytically,
# β is profiled out via GLS, and both are reconstructed from the optimum
# of the remaining (scaled) parameters. This shrinks the optimisation
# parameter vector and removes the σ_eps log-scale axis from the surface
# the optimiser has to walk.
#
# Reparameterisation: all variance components are expressed in σ²_eps units.
#   Λ_B = σ_eps · L_B
#   Λ_W = σ_eps · L_W
#   σ²_B = σ²_eps · τ_B
#   σ²_W = σ²_eps · τ_W
#   Λ_phy = σ_eps · L_phy
#   σ_phy = σ_eps · ρ_phy   (identity link — ρ_phy is signed)
# Then per-site site covariance Σ_y_site = σ²_eps · Ã with
#   Ã = L_B L_B' + diag(d̃)
#   d̃[t] = (L_W L_W')[t,t] + τ_B[t] + τ_W[t] + 1
#
# Phylogenetic block (J3) — same trick: Σ_y_full = σ²_eps · (I_n ⊗ Ã + J_n ⊗ B̃)
# with B̃ = (L_phy_aug L_phy_aug') .* Σ_phy, L_phy_aug = hcat(L_phy, ρ_phy).
# The joint sign flip (ρ_phy → -ρ_phy, φ → -φ) leaves the likelihood unchanged;
# when K_phy = 0 and Σ_phy has zero blocks (always so for a tree-derived Σ_phy,
# whose root edge is dropped), a flip of one block does too. fit.jl applies a global sign anchor post-hoc.
#
# Profile σ²_eps: -2ℓ has the form
#   -2ℓ = n·p·log(2π) + n·p·log(σ²_eps) + (logdet pieces in Ã)
#        + (1/σ²_eps) · Q̃
# where Q̃ is the quadratic form computed with Ã (no σ²_eps factor).
# ∂(-2ℓ)/∂σ²_eps = 0 gives σ̂²_eps = Q̃ / (n·p), so
#   -2ℓ_profile = n·p·(log(2π) + 1) + n·p·log(Q̃/(n·p)) + (logdet pieces)
#
# Profile β (non-phy path): given Ã, Q̃ is a quadratic form in (y − Xβ).
# The GLS minimiser is β̂ = (Σ_s X_s' Ã⁻¹ X_s)⁻¹ (Σ_s X_s' Ã⁻¹ y_s),
# computed via Woodbury through the same A_K factor used for the data
# quadratic form.

"""
    gaussian_profile_nll(params, y; spec, X=nothing, Σ_phy=nothing,
                         profile_beta=true) -> Real

Profile negative log-likelihood. σ²_eps is profiled out analytically;
β is profiled out via GLS when `profile_beta=true` and there is no
phylogenetic block.

Parameter layout (length = `profile_nparams(spec; profile_beta)`):
- β (spec.q entries if !profile_beta or has_phy_block)
- log_τ_B (p entries if spec.has_diag) — τ_B = exp(2·log_τ_B), i.e. ratios
  σ²_B / σ²_eps on the log-SD scale.
- log_τ_W (p entries if spec.has_diag)
- θ_rr_B for L_B (rr_theta_len(p, K_B) entries)
- θ_rr_W for L_W (rr_theta_len(p, K_W) entries if K_W > 0)
- ρ_phy (p entries if has_phy_unique) — identity link, signed.
- θ_rr_phy for L_phy (rr_theta_len(p, K_phy) entries if K_phy > 0)
"""
function gaussian_profile_nll(params::AbstractVector, y::AbstractMatrix;
                              spec::NamedTuple,
                              X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                              Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                              profile_beta::Bool = true)
    p = spec.p
    q = spec.q
    K_B = spec.K_B
    K_W = spec.K_W
    has_diag = spec.has_diag
    K_phy = hasproperty(spec, :K_phy) ? spec.K_phy : 0
    has_phy_unique = hasproperty(spec, :has_phy_unique) ? spec.has_phy_unique : false
    has_phy = (K_phy > 0) || has_phy_unique

    # β profiling is only supported on the non-phy site-stacked path.
    # For phy, β stays in the parameter vector (still no σ_eps axis).
    do_profile_beta = profile_beta && !has_phy && q > 0

    n = size(y, 2)
    size(y, 1) == p ||
        throw(ArgumentError("y first dim ($(size(y, 1))) must equal spec.p ($p)"))

    rr_B = rr_theta_len(p, K_B)
    rr_W = K_W > 0 ? rr_theta_len(p, K_W) : 0
    rr_phy = K_phy > 0 ? rr_theta_len(p, K_phy) : 0
    diag_count = has_diag ? 2 * p : 0
    phy_diag_count = has_phy_unique ? p : 0
    q_in_params = do_profile_beta ? 0 : q
    n_expected = q_in_params + diag_count + rr_B + rr_W + phy_diag_count + rr_phy
    length(params) == n_expected || throw(ArgumentError(
        "params length ($(length(params))) must equal $n_expected " *
        "(q_in_params=$q_in_params + diag=$diag_count + rr_B=$rr_B + rr_W=$rr_W " *
        "+ phy_diag=$phy_diag_count + rr_phy=$rr_phy)"))

    if has_phy && Σ_phy === nothing
        throw(ArgumentError("Σ_phy is required when K_phy > 0 or has_phy_unique"))
    end

    cursor = 0
    β = if do_profile_beta
        nothing  # computed below from GLS
    elseif q > 0
        X === nothing && throw(ArgumentError("spec.q = $q > 0 requires X"))
        b = @view params[(cursor + 1):(cursor + q)]
        cursor += q
        b
    else
        nothing
    end

    if has_diag
        log_τ_B = @view params[(cursor + 1):(cursor + p)]
        cursor += p
        log_τ_W = @view params[(cursor + 1):(cursor + p)]
        cursor += p
    end

    θ_rr_B = @view params[(cursor + 1):(cursor + rr_B)]
    cursor += rr_B
    L_B = unpack_lambda(θ_rr_B, p, K_B)

    if K_W > 0
        θ_rr_W = @view params[(cursor + 1):(cursor + rr_W)]
        cursor += rr_W
        L_W = unpack_lambda(θ_rr_W, p, K_W)
    end

    if has_phy_unique
        ρ_phy_raw = @view params[(cursor + 1):(cursor + p)]
        cursor += p
    end

    if K_phy > 0
        θ_rr_phy = @view params[(cursor + 1):(cursor + rr_phy)]
        cursor += rr_phy
        L_phy = unpack_lambda(θ_rr_phy, p, K_phy)
    end

    # ----- Build d̃ (per-trait diag) — all variance components in σ²_eps units.
    # d̃[t] = (L_W L_W')[t,t] + τ_B[t] + τ_W[t] + 1
    Td = promote_type(eltype(y), eltype(L_B))
    if has_diag
        Td = promote_type(Td, eltype(log_τ_B), eltype(log_τ_W))
    end
    if K_W > 0
        Td = promote_type(Td, eltype(L_W))
    end
    if Σ_phy !== nothing
        Td = promote_type(Td, eltype(Σ_phy))
    end
    if has_phy_unique
        Td = promote_type(Td, eltype(ρ_phy_raw))
    end
    if K_phy > 0
        Td = promote_type(Td, eltype(L_phy))
    end

    d_tilde = Vector{Td}(undef, p)
    @inbounds for t in 1:p
        v = one(Td)
        if K_W > 0
            for k in 1:K_W
                v += L_W[t, k]^2
            end
        end
        if has_diag
            v += exp(2 * log_τ_B[t])
            v += exp(2 * log_τ_W[t])
        end
        d_tilde[t] = v
    end

    # ----- The two paths split here.
    if !has_phy
        # ----- Non-phy path (J1 / J2). Site-stacked Woodbury on Ã.
        d_inv = Vector{Td}(undef, p)
        @inbounds for t in 1:p
            d_inv[t] = one(Td) / d_tilde[t]
        end

        DinvL = d_inv .* L_B                         # p × K_B
        A_K = I + L_B' * DinvL                       # K_B × K_B
        cA = cholesky(Symmetric(A_K))

        logdet_A_tilde = sum(log, d_tilde) + logdet(cA)

        if do_profile_beta
            # ----- β profile-out via GLS on the Woodbury Ã⁻¹.
            # M β = v with
            #   M = Σ_s X_s' Ã⁻¹ X_s,    v = Σ_s X_s' Ã⁻¹ y_s
            # Apply Ã⁻¹ to each y_s and to each column of stacked X to build
            # both M (q × q) and v (q-vector). Same Woodbury cost as the
            # data path; we just do (q+1) right-hand sides instead of 1.
            # Stack residuals/data: layout (p, n) for y, (p, n, q) for X.
            # Apply Ã⁻¹ column-by-column using the Woodbury Σ⁻¹ r formula.
            # Cheaper as one big batched solve:
            #   X_flat[:, j] := vec of X[:, :, j_col_idx] for j = q+1 stacked cols
            # We instead loop over q+1 RHSs and reuse the K-dim solve.
            #
            # Build M and v using batched matrix ops to minimise allocs:
            #   Ã⁻¹ Y_full where Y_full = hcat(y, X[:, :, 1], ..., X[:, :, q])
            # Build the augmented RHS as a 2D matrix of width n*(q+1).
            # Then split back into y-block and per-X blocks.
            #
            # Compact implementation: do all the Ã⁻¹ solves for y and for
            # each X-column block, accumulate M and v as we go.
            Ty = promote_type(Td, eltype(X), eltype(y))
            M = zeros(Ty, q, q)
            vvec = zeros(Ty, q)

            # Ã⁻¹ y (p × n)
            Dinv_y = d_inv .* y                       # p × n
            LtDy = L_B' * Dinv_y                      # K_B × n
            zy = cA \ LtDy
            Ainv_y = Dinv_y .- DinvL * zy             # p × n

            # Per-q apply: Ã⁻¹ X[:, :, j] (p × n), accumulate into M and v.
            # We accumulate M[j, k] = sum_s X[:, s, j]' (Ã⁻¹ X[:, s, k])_p.
            # Doing the q solves once and then double-summing is the cheap path.
            Ainv_X = Array{Ty}(undef, p, n, q)
            for j in 1:q
                Xj = @view X[:, :, j]
                Dinv_Xj = d_inv .* Xj
                LtDXj = L_B' * Dinv_Xj
                zXj = cA \ LtDXj
                @inbounds Ainv_X[:, :, j] = Dinv_Xj .- DinvL * zXj
            end
            for j in 1:q
                Xj = @view X[:, :, j]
                # v[j] = sum_s X[:, s, j]' Ã⁻¹ y[:, s]
                vvec[j] = sum(Xj .* Ainv_y)
                for k in j:q
                    Xk_ainv = @view Ainv_X[:, :, k]
                    M[j, k] = sum(Xj .* Xk_ainv)
                    if k > j
                        M[k, j] = M[j, k]
                    end
                end
            end
            β_hat = M \ vvec

            # Now r_s = y_s - X_s β̂. Quad = y' Ã⁻¹ y − 2 β̂' v + β̂' M β̂.
            # By the GLS normal eqns M β̂ = v, this collapses to:
            #   Q̃ = y' Ã⁻¹ y − β̂' v
            quad_y = sum(y .* Ainv_y)
            quad = quad_y - dot(β_hat, vvec)
        else
            # Pass-through residual computation (β provided or no X)
            if X === nothing || q == 0
                resid = y
            else
                Tres = promote_type(Td, eltype(X), eltype(y), eltype(β))
                resid = Matrix{Tres}(undef, p, n)
                @inbounds for s in 1:n, t in 1:p
                    μ_ts = zero(Tres)
                    for k in 1:q
                        μ_ts += X[t, s, k] * β[k]
                    end
                    resid[t, s] = y[t, s] - μ_ts
                end
            end
            Dinv_r = d_inv .* resid
            LtDr = L_B' * Dinv_r
            z = cA \ LtDr
            Ainv_r = Dinv_r .- DinvL * z
            quad = sum(resid .* Ainv_r)
        end

        # Profile σ²_eps from Q̃; build profile NLL.
        np = n * p
        Tq = promote_type(typeof(quad), Td)
        σ²_eps_hat = quad / np

        nll = convert(Tq, 0.5) * (
            np * log(convert(Tq, 2π))
            + np
            + np * log(σ²_eps_hat)
            + n * logdet_A_tilde
        )

        return nll
    else
        # ----- Phy path (J3). β stays as a parameter; σ²_eps still profiled.
        # Build Ã and B̃ in σ²_eps units.
        A_tilde = L_B * L_B'
        @inbounds for t in 1:p
            A_tilde[t, t] += d_tilde[t]
        end

        L_phy_aug = if K_phy > 0 && has_phy_unique
            hcat(L_phy, collect(ρ_phy_raw))
        elseif K_phy > 0
            L_phy
        else
            reshape(collect(ρ_phy_raw), p, 1)
        end
        B_tilde = (L_phy_aug * L_phy_aug') .* Σ_phy

        # Residuals
        if X === nothing || q == 0
            resid = y
        else
            Tres = promote_type(Td, eltype(X), eltype(y), eltype(β))
            resid = Matrix{Tres}(undef, p, n)
            @inbounds for s in 1:n, t in 1:p
                μ_ts = zero(Tres)
                for k in 1:q
                    μ_ts += X[t, s, k] * β[k]
                end
                resid[t, s] = y[t, s] - μ_ts
            end
        end

        m = vec(sum(resid, dims = 2)) ./ n
        Y_centered = resid .- reshape(m, p, 1)

        cA_sym = cholesky(Symmetric((A_tilde + A_tilde') ./ 2))
        AnB = A_tilde .+ n .* B_tilde
        cAnB = cholesky(Symmetric((AnB + AnB') ./ 2))

        v_mean = cAnB \ m
        quad_mean = n * dot(m, v_mean)
        V_c = cA_sym \ Y_centered
        quad_centered = sum(Y_centered .* V_c)
        quad = quad_mean + quad_centered

        logdet_total = logdet(cAnB) + (n - 1) * logdet(cA_sym)

        np = n * p
        Tq = promote_type(typeof(quad), Td)
        σ²_eps_hat = quad / np

        nll = convert(Tq, 0.5) * (
            np * log(convert(Tq, 2π))
            + np
            + np * log(σ²_eps_hat)
            + logdet_total
        )

        return nll
    end
end

"""
    profile_nparams(spec; profile_beta=true) -> Int

Number of parameters the profile NLL optimises over. Drops σ_eps (always)
and β (when `profile_beta && !has_phy`).
"""
function profile_nparams(spec::NamedTuple; profile_beta::Bool = true)
    p = spec.p
    q = spec.q
    K_B = spec.K_B
    K_W = spec.K_W
    has_diag = spec.has_diag
    K_phy = hasproperty(spec, :K_phy) ? spec.K_phy : 0
    has_phy_unique = hasproperty(spec, :has_phy_unique) ? spec.has_phy_unique : false
    has_phy = (K_phy > 0) || has_phy_unique
    do_profile_beta = profile_beta && !has_phy && q > 0
    q_in = do_profile_beta ? 0 : q

    rr_B = rr_theta_len(p, K_B)
    rr_W = K_W > 0 ? rr_theta_len(p, K_W) : 0
    rr_phy = K_phy > 0 ? rr_theta_len(p, K_phy) : 0
    diag = has_diag ? 2 * p : 0
    phy_diag = has_phy_unique ? p : 0
    return q_in + diag + rr_B + rr_W + phy_diag + rr_phy
end

"""
    profile_recover(params, y; spec, X=nothing, Σ_phy=nothing,
                    profile_beta=true) -> NamedTuple

Run one final NLL pass at `params` and return everything needed to
build the user-facing fit:
  (logLik, σ_eps, β, Λ_B, Λ_W, σ²_B, σ²_W, Λ_phy, σ_phy)
"""
function profile_recover(params::AbstractVector, y::AbstractMatrix;
                         spec::NamedTuple,
                         X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                         Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                         profile_beta::Bool = true)
    p = spec.p
    q = spec.q
    K_B = spec.K_B
    K_W = spec.K_W
    has_diag = spec.has_diag
    K_phy = hasproperty(spec, :K_phy) ? spec.K_phy : 0
    has_phy_unique = hasproperty(spec, :has_phy_unique) ? spec.has_phy_unique : false
    has_phy = (K_phy > 0) || has_phy_unique
    do_profile_beta = profile_beta && !has_phy && q > 0

    rr_B = rr_theta_len(p, K_B)
    rr_W = K_W > 0 ? rr_theta_len(p, K_W) : 0
    rr_phy = K_phy > 0 ? rr_theta_len(p, K_phy) : 0

    cursor = 0
    if do_profile_beta
        # β placeholder; computed below.
        β_in = nothing
    elseif q > 0
        β_in = collect(params[(cursor + 1):(cursor + q)])
        cursor += q
    else
        β_in = Float64[]
    end

    if has_diag
        log_τ_B = collect(params[(cursor + 1):(cursor + p)])
        cursor += p
        log_τ_W = collect(params[(cursor + 1):(cursor + p)])
        cursor += p
    end
    θ_rr_B = collect(params[(cursor + 1):(cursor + rr_B)])
    cursor += rr_B
    L_B = unpack_lambda(θ_rr_B, p, K_B)
    if K_W > 0
        θ_rr_W = collect(params[(cursor + 1):(cursor + rr_W)])
        cursor += rr_W
        L_W = unpack_lambda(θ_rr_W, p, K_W)
    end
    if has_phy_unique
        # Identity link: ρ_phy is signed (a joint flip with φ, and with zero
        # blocks in Σ_phy a per-block flip, leaves the likelihood unchanged).
        ρ_phy = collect(params[(cursor + 1):(cursor + p)])
        cursor += p
    end
    if K_phy > 0
        θ_rr_phy = collect(params[(cursor + 1):(cursor + rr_phy)])
        cursor += rr_phy
        L_phy = unpack_lambda(θ_rr_phy, p, K_phy)
    end

    n = size(y, 2)
    d_tilde = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        v = 1.0
        if K_W > 0
            for k in 1:K_W
                v += L_W[t, k]^2
            end
        end
        if has_diag
            v += exp(2 * log_τ_B[t])
            v += exp(2 * log_τ_W[t])
        end
        d_tilde[t] = v
    end

    if !has_phy
        d_inv = 1 ./ d_tilde
        DinvL = d_inv .* L_B
        A_K = I + L_B' * DinvL
        cA = cholesky(Symmetric(A_K))
        logdet_A_tilde = sum(log, d_tilde) + logdet(cA)

        if do_profile_beta
            M = zeros(Float64, q, q)
            vvec = zeros(Float64, q)
            Dinv_y = d_inv .* y
            LtDy = L_B' * Dinv_y
            zy = cA \ LtDy
            Ainv_y = Dinv_y .- DinvL * zy
            Ainv_X = Array{Float64}(undef, p, n, q)
            for j in 1:q
                Xj = @view X[:, :, j]
                Dinv_Xj = d_inv .* Xj
                LtDXj = L_B' * Dinv_Xj
                zXj = cA \ LtDXj
                @inbounds Ainv_X[:, :, j] = Dinv_Xj .- DinvL * zXj
            end
            for j in 1:q
                Xj = @view X[:, :, j]
                vvec[j] = sum(Xj .* Ainv_y)
                for k in j:q
                    Xk_ainv = @view Ainv_X[:, :, k]
                    M[j, k] = sum(Xj .* Xk_ainv)
                    if k > j
                        M[k, j] = M[j, k]
                    end
                end
            end
            β_hat = M \ vvec
            quad_y = sum(y .* Ainv_y)
            quad = quad_y - dot(β_hat, vvec)
            β_out = collect(β_hat)
        else
            if X === nothing || q == 0
                resid = y
            else
                resid = similar(y, Float64)
                @inbounds for s in 1:n, t in 1:p
                    μ_ts = 0.0
                    for k in 1:q
                        μ_ts += X[t, s, k] * β_in[k]
                    end
                    resid[t, s] = y[t, s] - μ_ts
                end
            end
            Dinv_r = d_inv .* resid
            LtDr = L_B' * Dinv_r
            z = cA \ LtDr
            Ainv_r = Dinv_r .- DinvL * z
            quad = sum(resid .* Ainv_r)
            β_out = β_in === nothing ? Float64[] : collect(β_in)
        end

        np = n * p
        σ²_eps_hat = quad / np
        σ_eps_hat = sqrt(σ²_eps_hat)
        # Recover unscaled loadings/variances
        Λ_B_hat = σ_eps_hat .* L_B
        Λ_W_hat = K_W > 0 ? (σ_eps_hat .* L_W) : nothing
        σ²_B_hat = has_diag ? (exp.(2 .* log_τ_B) .* σ²_eps_hat) : nothing
        σ²_W_hat = has_diag ? (exp.(2 .* log_τ_W) .* σ²_eps_hat) : nothing
        # Profile NLL → logLik
        ll = -0.5 * (np * log(2π) + np + np * log(σ²_eps_hat) + n * logdet_A_tilde)
        return (logLik = ll, σ_eps = σ_eps_hat, β = β_out,
                Λ_B = Λ_B_hat, Λ_W = Λ_W_hat,
                σ²_B = σ²_B_hat, σ²_W = σ²_W_hat,
                Λ_phy = nothing, σ_phy = nothing)
    else
        # Phy path — β is just whatever was passed (not profiled).
        A_tilde = L_B * L_B'
        @inbounds for t in 1:p
            A_tilde[t, t] += d_tilde[t]
        end
        L_phy_aug = if K_phy > 0 && has_phy_unique
            hcat(L_phy, ρ_phy)
        elseif K_phy > 0
            L_phy
        else
            reshape(ρ_phy, p, 1)
        end
        B_tilde = (L_phy_aug * L_phy_aug') .* Σ_phy

        if X === nothing || q == 0
            resid = y
        else
            resid = similar(y, Float64)
            @inbounds for s in 1:n, t in 1:p
                μ_ts = 0.0
                for k in 1:q
                    μ_ts += X[t, s, k] * β_in[k]
                end
                resid[t, s] = y[t, s] - μ_ts
            end
        end
        m = vec(sum(resid, dims = 2)) ./ n
        Y_centered = resid .- reshape(m, p, 1)
        cA_sym = cholesky(Symmetric((A_tilde + A_tilde') ./ 2))
        AnB = A_tilde .+ n .* B_tilde
        cAnB = cholesky(Symmetric((AnB + AnB') ./ 2))
        v_mean = cAnB \ m
        quad_mean = n * dot(m, v_mean)
        V_c = cA_sym \ Y_centered
        quad_centered = sum(Y_centered .* V_c)
        quad = quad_mean + quad_centered
        logdet_total = logdet(cAnB) + (n - 1) * logdet(cA_sym)
        np = n * p
        σ²_eps_hat = quad / np
        σ_eps_hat = sqrt(σ²_eps_hat)
        Λ_B_hat = σ_eps_hat .* L_B
        Λ_W_hat = K_W > 0 ? (σ_eps_hat .* L_W) : nothing
        σ²_B_hat = has_diag ? (exp.(2 .* log_τ_B) .* σ²_eps_hat) : nothing
        σ²_W_hat = has_diag ? (exp.(2 .* log_τ_W) .* σ²_eps_hat) : nothing
        Λ_phy_hat = K_phy > 0 ? (σ_eps_hat .* L_phy) : nothing
        σ_phy_hat = has_phy_unique ? (σ_eps_hat .* ρ_phy) : nothing
        β_out = β_in === nothing ? Float64[] : collect(β_in)
        ll = -0.5 * (np * log(2π) + np + np * log(σ²_eps_hat) + logdet_total)
        return (logLik = ll, σ_eps = σ_eps_hat, β = β_out,
                Λ_B = Λ_B_hat, Λ_W = Λ_W_hat,
                σ²_B = σ²_B_hat, σ²_W = σ²_W_hat,
                Λ_phy = Λ_phy_hat, σ_phy = σ_phy_hat)
    end
end
