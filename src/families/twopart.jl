# Two-part / mixture family substrate for GLLVModels.jl (shared-z, Option A — one
# latent z drives both parts via part-specific loadings Λ_z, Λ_c). Two-part
# observations depend on TWO linear predictors η^z (occurrence/zero) and η^c
# (positive/count), so they do not fit the scalar-μ generic core in
# families/laplace.jl. Each two-part family instead provides, dispatched on its
# marker:
#     _tp_pieces(family, y, η^z, η^c) -> (s^z, s^c, W^z, W^c, logf)
# the per-observation block scores s = ∂logf/∂η, the expected-information Fisher
# weights W = −E[∂²logf/∂η²] (cross term is 0 — the two parts are conditionally
# independent), and the two-part log-density logf. The shared-z mode-finder then
# assembles (spec §2.0):
#     A(z) = Λ_z'diag(W^z)Λ_z + Λ_c'diag(W^c)Λ_c + I       (SPD)
#     g(z) = Λ_z's^z + Λ_c's^c − z
#     z ← z + A(z)⁻¹ g(z)                                  (Fisher scoring)
#     log p(y_s) ≈ ℓ_s(ẑ) − ½ẑ'ẑ − ½logdet A(ẑ).
# `_clamp_eta`/`_safe_solve` are reused from families/laplace.jl. With the v1
# default Λ_z = 0 the occurrence block drops out of A and g (the integral is
# genuinely K-dimensional; β^z carries a per-species occurrence intercept).
#
# Per-trait dispersion (2026-08-28, gllvmTMB.cpp:1195-1196 `log_sigma_lognormal_delta`
# / `log_phi_gamma_delta`, both length n_traits). `family` is broadcast over species
# by `_twopart_mode`/`twopart_loglik_site` via `_tp_pieces_at`/`_tp_observed_Wc_at`
# below: the scalar-marker method is unchanged (bit-identical), and a NEW
# `AbstractVector` method dispatches `fams[t]` per species — the same pattern as
# `grouped_dispersion.jl`'s per-species `fams::AbstractVector`, kept local here since
# the two-part kernel threads `family` through 3 call sites rather than 1.
@inline _tp_pieces_at(family, t, y, ηz, ηc) = _tp_pieces(family, y, ηz, ηc)
@inline _tp_pieces_at(fams::AbstractVector, t, y, ηz, ηc) = _tp_pieces(fams[t], y, ηz, ηc)
@inline _tp_observed_Wc_at(family, t, y, ηc, Wc) = _tp_observed_Wc(family, y, ηc, Wc)
@inline _tp_observed_Wc_at(fams::AbstractVector, t, y, ηc, Wc) = _tp_observed_Wc(fams[t], y, ηc, Wc)

# Per-site joint mode ẑ over the shared latent z (Fisher-scoring Newton).
function _twopart_mode(family, y::AbstractVector,
        Λz::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector;
        offsetz = nothing, offsetc = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λc)
    offz = offsetz === nothing ? false : offsetz    # additive identity ⇒ no-offset path unchanged
    offc = offsetc === nothing ? false : offsetc
    z = zeros(K)
    sz = Vector{Float64}(undef, p); sc = Vector{Float64}(undef, p)
    Wz = Vector{Float64}(undef, p); Wc = Vector{Float64}(undef, p)
    # Per-call buffers, reused across Newton iterations. Each is written in place
    # with the SAME broadcast / BLAS expression as the allocating version, so the
    # computed values and FP-operation order are bit-identical.
    Λzz = Vector{Float64}(undef, p)    # Λz*z (occurrence linear-predictor contribution)
    Λcz = Vector{Float64}(undef, p)    # Λc*z (positive-part linear-predictor contribution)
    ηz  = Vector{Float64}(undef, p)    # clamped occurrence predictor
    ηc  = Vector{Float64}(undef, p)    # clamped positive-part predictor
    WzΛz = Matrix{Float64}(undef, p, K)  # Wz .* Λz
    WcΛc = Matrix{Float64}(undef, p, K)  # Wc .* Λc
    Amat = Matrix{Float64}(undef, K, K)  # Λz'(Wz.*Λz) (+ Λc'(Wc.*Λc) and + I in place)
    Atmp = Matrix{Float64}(undef, K, K)  # Λc'(Wc.*Λc) temp before accumulation
    g  = Vector{Float64}(undef, K)     # rhs Λz'sz + Λc'sc − z
    gc = Vector{Float64}(undef, K)     # Λc'sc temp before accumulation
    for _ in 1:maxiter
        mul!(Λzz, Λz, z)
        mul!(Λcz, Λc, z)
        ηz .= _clamp_eta.(βz .+ offz .+ Λzz)
        ηc .= _clamp_eta.(βc .+ offc .+ Λcz)
        @inbounds for t in 1:p
            s_z, s_c, W_z, W_c, _ = _tp_pieces_at(family, t, y[t], ηz[t], ηc[t])
            sz[t] = s_z; sc[t] = s_c; Wz[t] = W_z; Wc[t] = W_c
        end
        WzΛz .= Wz .* Λz                       # = Wz .* Λz (p×K)
        WcΛc .= Wc .* Λc                       # = Wc .* Λc (p×K)
        mul!(Amat, Λz', WzΛz)                  # = Λz' * (Wz .* Λz)
        mul!(Atmp, Λc', WcΛc)                  # = Λc' * (Wc .* Λc)
        Amat .+= Atmp                          # = Λz'(Wz.*Λz) + Λc'(Wc.*Λc)
        @inbounds for d in 1:K
            Amat[d, d] += 1.0                  # + I (adds 1.0 to each diagonal entry)
        end
        A = Symmetric(Amat)
        mul!(g, Λz', sz)                        # = Λz' * sz
        mul!(gc, Λc', sc)                       # = Λc' * sc
        g .= g .+ gc .- z                       # rhs = Λz'sz + Λc'sc − z
        Δ = _safe_solve(A, g)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

"""
    twopart_loglik_site(family, y, Λz, Λc, βz, βc; offsetz=nothing, offsetc=nothing,
                        maxiter=100, tol=1e-9) -> Float64

Two-part Laplace log-marginal for one site: `ℓ_s(ẑ) − ½ẑ'ẑ − ½logdet A(ẑ)`. Optional
`offsetz` / `offsetc` are known additive terms on the occurrence / positive-part
predictors (`η^z = β^z + offsetz + Λ^z z`, similarly `η^c`).
"""
# ---------------------------------------------------------------------------
# Observed-curvature override for the POSITIVE-part weight (2026-08-24).
#
# `_tp_pieces` returns the Fisher (expected-information) `Wc`, which the two-part
# substrate then uses for BOTH roles: the Fisher-scoring mode search and the Laplace
# log-det. Those roles have different requirements. The mode search may use expected
# information — it solves the same score equation, so the mode is unchanged. The
# log-det must use the OBSERVED curvature to match TMB, which obtains it structurally
# by AD-ing the joint nll (`TMB::MakeADFun(..., random=)`) and so never faces this
# choice at all.
#
# This hook is applied ONLY in `twopart_loglik_site`'s A-matrix, never in
# `_twopart_mode`. Keeping the mode solver untouched is deliberate: substituting a
# curvature into a mode search that was tuned for a different one is exactly how the
# Exponential fix first went wrong (‖Λ‖ ran away to ~960 against a true 0.38).
#
# Default is identity, so every family not given a method here is bit-for-bit
# unchanged.
_tp_observed_Wc(::Any, y, ηc, Wc) = Wc

function twopart_loglik_site(family, y::AbstractVector,
        Λz::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector;
        offsetz = nothing, offsetc = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Λc, 1)
    offz = offsetz === nothing ? false : offsetz
    offc = offsetc === nothing ? false : offsetc
    K = size(Λc, 2)
    ẑ = _twopart_mode(family, y, Λz, Λc, βz, βc;
                      offsetz = offsetz, offsetc = offsetc, maxiter = maxiter, tol = tol)
    ηz = _clamp_eta.(βz .+ offz .+ Λz * ẑ)
    ηc = _clamp_eta.(βc .+ offc .+ Λc * ẑ)
    Wz = Vector{Float64}(undef, p); Wc = Vector{Float64}(undef, p)
    ℓ = 0.0
    @inbounds for t in 1:p
        _, _, W_z, W_c, logf = _tp_pieces_at(family, t, y[t], ηz[t], ηc[t])
        # Observed curvature enters HERE ONLY (the log-det), never the mode solve.
        Wz[t] = W_z
        Wc[t] = hessian === :observed ? _tp_observed_Wc_at(family, t, y[t], ηc[t], W_c) : W_c
        ℓ += logf
    end
    # Per-call buffers (written in place with the SAME broadcast / BLAS expressions
    # as before ⇒ bit-identical values and FP-operation order).
    WzΛz = Wz .* Λz                           # = Wz .* Λz (p×K)
    WcΛc = Wc .* Λc                           # = Wc .* Λc (p×K)
    Amat = Λz' * WzΛz                          # = Λz' * (Wz .* Λz) (K×K)
    Atmp = Λc' * WcΛc                          # = Λc' * (Wc .* Λc) (K×K)
    Amat .+= Atmp                              # = Λz'(Wz.*Λz) + Λc'(Wc.*Λc)
    @inbounds for d in 1:K
        Amat[d, d] += 1.0                      # + I (adds 1.0 to each diagonal entry)
    end
    A = Symmetric(Amat)
    return ℓ - 0.5 * dot(ẑ, ẑ) - 0.5 * logdet(A)
end

"""
    twopart_marginal_loglik_laplace(family, Y, Λz, Λc, βz, βc;
                                    offsetz=nothing, offsetc=nothing, kwargs...) -> Float64

Total two-part Laplace log-marginal over the `n` sites (columns of `Y`). `offsetz` /
`offsetc` (p×n, or `nothing`) are known additive offsets on the occurrence /
positive-part predictors; a constant per-species `offsetc` is equivalent to shifting
`βc` (the offset-absorption identity).
"""
function twopart_marginal_loglik_laplace(family, Y::AbstractMatrix,
        Λz::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector;
        offsetz = nothing, offsetc = nothing, kwargs...)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        ozs = offsetz === nothing ? nothing : view(offsetz, :, s)
        ocs = offsetc === nothing ? nothing : view(offsetc, :, s)
        acc += twopart_loglik_site(family, view(Y, :, s), Λz, Λc, βz, βc;
                                   offsetz = ozs, offsetc = ocs, kwargs...)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Delta-lognormal family — occurrence Bernoulli × positive lognormal.
# P(y=0)=1−π, density for y>0 = π·LogNormal(y; meanlog=η^c, sdlog=σ),
# π = logistic(η^z). The positive part is Gaussian in log y, so W^c=1/σ² is the
# exact Hessian and the Laplace marginal is exact (the cleanest substrate check).
# ---------------------------------------------------------------------------

"""
    DeltaLogNormal(σ = 1.0)

Marker for the Delta-lognormal two-part family (Bernoulli occurrence × positive
lognormal). Use with the unified entry point:

```julia
fit_gllvm(Y; family = DeltaLogNormal(), K = 2)
fit_gllvm(Y; family = DeltaLogNormal(0.5), K = 2)   # same fit — σ is a tag payload
gllvm(@formula(y ~ 1), Y, site_data; family = DeltaLogNormal(), K = 2)
```

The marker's `σ` field is a **tag payload** — it is never read by
[`fit_gllvm`](@ref) / [`fit_delta_lognormal_gllvm`](@ref); the shared log-scale
SD is always estimated. The value is only used when the marker is passed into
the Laplace marginal / score kernels during fitting (with the current iterate's
`σ`). Pass `DeltaLogNormal()` so the public call need not invent a scale that
is never read from the family instance.

Named fitter [`fit_delta_lognormal_gllvm`](@ref) remains available. +X,
`disp_group`, and `row_eff` are not admitted on this surface.
"""
struct DeltaLogNormal
    σ::Float64
end

DeltaLogNormal() = DeltaLogNormal(1.0)

function _tp_pieces(f::DeltaLogNormal, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))                 # occurrence prob = logistic(η^z)
    Wz = π * (one(π) - π)
    if y > 0
        σ = f.σ
        sc = (log(y) - ηc) / σ^2                # ∂logf/∂η^c, θ = η^c (meanlog)
        return (one(π) - π, sc, Wz, inv(σ^2),
                log(π) + logpdf(LogNormal(ηc, σ), y))
    else
        return (-π, zero(ηc), Wz, zero(ηc), log1p(-π))
    end
end

"""
    delta_lognormal_marginal_loglik_laplace(Y, Λc, βz, βc, σ; Λz=nothing, kwargs...) -> Float64

Total two-part Laplace log-marginal for a Delta-lognormal GLLVM: occurrence
probability `π = logistic(β^z + Λ_z z)` (intercept-only by default, `Λ_z = 0`)
times a positive lognormal with meanlog `η^c = β^c + Λ_c z` and sdlog `σ`. `Y` is
p×n with `0` for absences and positive reals for the positive part. With `Λ_c = 0`
(and `Λ_z = 0`) this reduces exactly to the independent two-part-regression
log-likelihood.
"""
function delta_lognormal_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, σ::Real;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(DeltaLogNormal(float(σ)), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    delta_lognormal_marginal_loglik_laplace(Y, Λc, βz, βc, σ::AbstractVector; Λz=nothing, kwargs...) -> Float64

Per-trait-dispersion variant (`length(σ) == p`, gllvmTMB's `log_sigma_lognormal_delta`,
`gllvmTMB.cpp:1195-1196`, length `n_traits`): species `t` gets its own sdlog `σ[t]`
rather than one shared scalar. With a constant `σ` this equals the shared-scalar
method above to machine precision (same `_tp_pieces` density per species, only the
dispersion lookup changes).
"""
function delta_lognormal_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, σ::AbstractVector;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    length(σ) == p || throw(ArgumentError("length(σ)=$(length(σ)) must equal p=$p"))
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    fams = DeltaLogNormal.(float.(σ))
    return twopart_marginal_loglik_laplace(fams, Y, Λz_, Λc, βz, βc; kwargs...)
end

# ---------------------------------------------------------------------------
# Fit driver (Delta-lognormal slice 2).
# ---------------------------------------------------------------------------

"""
    DeltaLogNormalFit

Result of [`fit_delta_lognormal_gllvm`](@ref): occurrence logits `βz` (length p),
positive-part meanlog intercepts `βc` (length p), positive-part loadings `Λc`
(p×K), the log-scale SD `σ` (a `Float64` under `disp_group == :shared`, or a
length-p `Vector{Float64}` under `disp_group == :species`), the maximised
`loglik`, `converged`, `iterations`, `predictor` (`:separate` default or
`:shared`), and `disp_group` (`:species` default, matching R gllvmTMB's
per-trait dispersion — `:shared` remains available as an explicit opt-in).
(`Λz = 0` — occurrence is intercept-only — under `predictor == :separate`.
Under `predictor == :shared` (the gllvmTMB twin-identity mode: one linear
predictor drives both parts, `gllvmTMB.cpp:2816-2830`), `βz === βc` and `Λc`
IS the shared loadings matrix (also equal to `Λz`) — read `f.βc`/`f.Λc` as
"the one shared predictor" in that mode, `f.βz` is the identical array, not
an independent estimate.)
"""
struct DeltaLogNormalFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    σ::Union{Float64, Vector{Float64}}
    loglik::Float64
    converged::Bool
    iterations::Int
    predictor::Symbol
    disp_group::Symbol
end

# Positional-compat constructors (hessian::Symbol precedent, e.g. binomial.jl):
# old 7-/8-arg call sites default the newly-appended trailing field(s).
DeltaLogNormalFit(βz, βc, Λc, σ, loglik, converged, iterations) =
    DeltaLogNormalFit(βz, βc, Λc, σ, loglik, converged, iterations, :separate, :shared)
DeltaLogNormalFit(βz, βc, Λc, σ, loglik, converged, iterations, predictor) =
    DeltaLogNormalFit(βz, βc, Λc, σ, loglik, converged, iterations, predictor, :shared)

function Base.show(io::IO, f::DeltaLogNormalFit)
    p, K = size(f.Λc)
    σstr = f.σ isa Real ? string(round(f.σ; sigdigits = 4)) : "per-trait"
    print(io, "DeltaLogNormalFit(p=", p, ", K=", K, ", σ=", σstr,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_delta_lognormal_gllvm(Y; K, …) -> DeltaLogNormalFit

Fit a Delta-lognormal two-part GLLVM by L-BFGS on the two-part Laplace marginal
([`delta_lognormal_marginal_loglik_laplace`](@ref)). `Y` is p×n with `0` for
absences and positive reals otherwise. Finite-difference gradient; warm start =
`logit(empirical P(y>0))` occurrence intercepts + mean / SVD of the positive-part
log-responses + `σ₀ = sd(log y_{>0})`.

`predictor` selects the parameterisation (`:separate` default, or `:shared`):
- `:separate` (default, previous/only behaviour before this kwarg existed):
  independent occurrence (`βz`, `Λz = 0`) and positive-part (`βc`, `Λc`)
  predictors; optimises over `[βz; βc; vec(Λc); log σ]`.
- `:shared`: the counterpart's tied-predictor mode — ONE linear predictor
  `η = β + Λz` drives both
  parts (`βz ≡ βc ≡ β`, `Λz ≡ Λc ≡ Λ`), optimising over the smaller
  `[β; vec(Λ); log σ]`. This is a twin-parity-oriented, restrictive
  parameterisation (occurrence log-odds and log-abundance move together by
  construction), not a general-purpose recommendation over `:separate`. A
  supplied `offset` is threaded into BOTH `offsetz` and `offsetc` under
  `:shared` so the tie `ηz ≡ ηc` is preserved.

`disp_group` selects the dispersion parameterisation (`:species` default, or
`:shared`) for grouped or species-specific dispersion:
- `:species` (default since `accept delta dispersion A`, 2026-09-24): one
  sdlog per species (`length(σ) == p`), matching gllvmTMB's per-trait
  `log_sigma_lognormal_delta`. Adds `p − 1` free parameters relative to
  `:shared`; the two nest (`:shared` is the `:species` model with all p
  sdlogs tied), so `:species` logLik ≥ `:shared` logLik on the same data up
  to optimiser noise. Composes with either `predictor` mode. **Existing code
  that called this fitter without an explicit `disp_group` will see σ change
  from a scalar to a length-p vector** — pass `disp_group = :shared`
  explicitly to keep the old one-scalar-per-model behaviour.
- `:shared` (previous default, before 2026-09-24; still available as an
  explicit opt-in): one scalar sdlog `σ` for every species.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap. This
holds under both `predictor` modes.
"""
function fit_delta_lognormal_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        predictor::Symbol = :separate,
        disp_group::Symbol = :species,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_delta_lognormal_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    predictor in (:separate, :shared) || throw(ArgumentError(
        "fit_delta_lognormal_gllvm: predictor must be :separate or :shared; got :$predictor"))
    disp_group in (:shared, :species) || throw(ArgumentError(
        "fit_delta_lognormal_gllvm: disp_group must be :shared or :species; got :$disp_group"))
    rr = rr_theta_len(p, K)

    βz0 = Vector{Float64}(undef, p)
    βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += log(Y[t, j]); c += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : s / c
    end
    sumsq = 0.0; nres = 0
    sumsq_t = zeros(p); nres_t = zeros(Int, p)
    @inbounds for t in 1:p, j in 1:n
        if Y[t, j] > 0
            r = log(Y[t, j]) - βc0[t]
            sumsq += r^2; nres += 1
            sumsq_t[t] += r^2; nres_t[t] += 1
        end
    end
    σ0 = nres > 1 ? max(sqrt(sumsq / (nres - 1)), 0.1) : 0.5
    # Per-trait warm start (disp_group == :species): per-species sd(log y_{>0}),
    # falling back to the pooled σ0 for species with < 2 positive observations.
    σ0vec = [nres_t[t] > 1 ? max(sqrt(sumsq_t[t] / (nres_t[t] - 1)), 0.1) : σ0 for t in 1:p]
    ndisp = disp_group === :shared ? 1 : p
    Zc = [Y[t, j] > 0 ? log(Y[t, j]) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    logσ0 = disp_group === :shared ? [log(σ0)] : log.(σ0vec)

    if predictor === :separate
        θ0 = vcat(βz0, βc0, pack_lambda(Λc0), logσ0)
        negll = θ -> begin
            βz = θ[1:p]
            βc = θ[(p + 1):(2p)]
            Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
            σ = disp_group === :shared ? exp(θ[2p + rr + 1]) : exp.(θ[(2p + rr + 1):(2p + rr + ndisp)])
            v = try
                -delta_lognormal_marginal_loglik_laplace(Y, Λc, βz, βc, σ; offsetc = offset, hessian = hessian,
                                                         maxiter = newton_maxiter, tol = newton_tol)
            catch
                1e12
            end
            isfinite(v) ? v : 1e12
        end
    else # :shared — one η = β + Λz drives both parts; offset threads to both parts symmetrically.
        β0 = 0.5 .* (βz0 .+ βc0)
        θ0 = vcat(β0, pack_lambda(Λc0), logσ0)
        negll = θ -> begin
            β = θ[1:p]
            Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
            σ = disp_group === :shared ? exp(θ[p + rr + 1]) : exp.(θ[(p + rr + 1):(p + rr + ndisp)])
            v = try
                -delta_lognormal_marginal_loglik_laplace(Y, Λ, β, β, σ; Λz = Λ,
                        offsetz = offset, offsetc = offset, hessian = hessian,
                        maxiter = newton_maxiter, tol = newton_tol)
            catch
                1e12
            end
            isfinite(v) ? v : 1e12
        end
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    if predictor === :separate
        βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
        Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
        σ = disp_group === :shared ? exp(θ̂[2p + rr + 1]) : exp.(θ̂[(2p + rr + 1):(2p + rr + ndisp)])
    else
        β = θ̂[1:p]
        Λc = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        σ = disp_group === :shared ? exp(θ̂[p + rr + 1]) : exp.(θ̂[(p + rr + 1):(p + rr + ndisp)])
        βz = β; βc = β
    end
    return DeltaLogNormalFit(βz, βc, Λc, σ, _fit_verdict(res)..., predictor, disp_group)
end

# ---------------------------------------------------------------------------
# Hurdle-Poisson — occurrence Bernoulli × ZERO-TRUNCATED Poisson count.
# P(y=0)=1−π, P(y=k)=π·Poisson(k;μ)/(1−e^{−μ}) for k≥1, π=logistic(η^z), μ=exp(η^c).
# Positive-block score/weight use the truncated mean μ_tr=μ/(1−e^{−μ}) and its
# variance Var_tr = μ_tr(1+μ−μ_tr): s^c = y−μ_tr, W^c = Var_tr (y>0; 0 for y=0).
# ---------------------------------------------------------------------------

"""
    HurdlePoisson()

Marker for the Hurdle-Poisson two-part family (Bernoulli occurrence × zero-truncated
Poisson count).
"""
struct HurdlePoisson end

function _tp_pieces(::HurdlePoisson, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))
    Wz = π * (one(π) - π)
    if y > 0
        μ = exp(ηc)
        p0 = exp(-μ)
        μtr = μ / (1 - p0)                       # zero-truncated mean
        Wc = μtr * (1 + μ - μtr)                 # zero-truncated variance ≥ 0
        logf = log(π) + logpdf(Poisson(μ), Int(y)) - log1p(-p0)
        return (one(π) - π, y - μtr, Wz, Wc, logf)
    else
        return (-π, zero(ηc), Wz, zero(ηc), log1p(-π))
    end
end

"""
    hurdle_poisson_marginal_loglik_laplace(Y, Λc, βz, βc; Λz=nothing, kwargs...) -> Float64

Total two-part Laplace log-marginal for a Hurdle-Poisson GLLVM (occurrence
`π=logistic(β^z)`, intercept-only by default; zero-truncated Poisson count with
`μ=exp(β^c+Λ_c z)`). `Y` is p×n integer counts (`0`=absence). `Λc=0` ⇒ exact
independent hurdle-Poisson loglik.
"""
function hurdle_poisson_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(HurdlePoisson(), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    HurdlePoissonFit

Result of [`fit_hurdle_poisson_gllvm`](@ref): occurrence logits `βz`, count log-mean
intercepts `βc`, count loadings `Λc`, `loglik`, `converged`, `iterations`.
"""
struct HurdlePoissonFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::HurdlePoissonFit)
    p, K = size(f.Λc)
    print(io, "HurdlePoissonFit(p=", p, ", K=", K,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_hurdle_poisson_gllvm(Y; K, …) -> HurdlePoissonFit

Fit a Hurdle-Poisson two-part GLLVM by L-BFGS over `[βz; βc; vec(Λc)]` (Λz=0).
`Y` p×n integer counts. Finite-difference gradient; warm start =
`logit(empirical P(y>0))` + `log` mean positive count + SVD loadings.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_hurdle_poisson_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_hurdle_poisson_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)
    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j]; c += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : log(max(s / c, 1.0))
    end
    Zc = [Y[t, j] > 0 ? log(max(Y[t, j], 0.5)) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = vcat(βz0, βc0, pack_lambda(Λc0))
    function negll(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        v = try
            -hurdle_poisson_marginal_loglik_laplace(Y, Λc, βz, βc; offsetc = offset, hessian = hessian,
                                                    maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    return HurdlePoissonFit(βz, βc, Λc, _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Hurdle-NB — occurrence Bernoulli × zero-truncated negative-binomial (NB2) count
# with shared dispersion r (Var = μ + μ²/r). p0=(r/(r+μ))^r; μ_tr=μ/(1−p0);
# s^c=y−μ_tr; W^c=(V+μ²)/(1−p0)−μ_tr², V=μ+μ²/r (y>0; 0 for y=0). r→∞ ⇒ Hurdle-Poisson.
# ---------------------------------------------------------------------------

"""
    HurdleNB(r = 10.0)

Marker for the Hurdle-NB family (Bernoulli occurrence × zero-truncated NB2 count,
shared dispersion `r`). Use with the unified entry point:

```julia
fit_gllvm(Y; family = HurdleNB(), K = 2)
fit_gllvm(Y; family = HurdleNB(99.0), K = 2)   # same fit — r is a tag payload
gllvm(@formula(y ~ 1), Y, site_data; family = HurdleNB(), K = 2)
```

The marker's `r` field is a **tag payload** — it is never read by
[`fit_gllvm`](@ref) / [`fit_hurdle_nb_gllvm`](@ref); the shared dispersion is
always estimated. There is no `r_init` keyword. `HurdleNB()` is the public
convenience (`10.0` matches the named fitter's hardcoded start). Named fitter
[`fit_hurdle_nb_gllvm`](@ref) remains available. +X, `disp_group`, and
`row_eff` are not admitted on this surface.
"""
struct HurdleNB
    r::Float64
end

# Public-call convenience (mirrors `NB1()` / `HurdlePoisson()` / `COMPoisson()`):
# r is never read on the `fit_gllvm` route. 10.0 matches the fitter start.
HurdleNB() = HurdleNB(10.0)

function _tp_pieces(f::HurdleNB, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))
    Wz = π * (one(π) - π)
    if y > 0
        μ = exp(ηc); r = f.r
        p0 = (r / (r + μ))^r
        μtr = μ / (1 - p0)
        V = μ + μ^2 / r
        Wc = (V + μ^2) / (1 - p0) - μtr^2
        logf = log(π) + logpdf(NegativeBinomial(r, r / (r + μ)), Int(y)) - log1p(-p0)
        return (one(π) - π, y - μtr, Wz, Wc, logf)
    else
        return (-π, zero(ηc), Wz, zero(ηc), log1p(-π))
    end
end

"""
    hurdle_nb_marginal_loglik_laplace(Y, Λc, βz, βc, r; Λz=nothing, kwargs...) -> Float64

Two-part Laplace log-marginal for a Hurdle-NB GLLVM. `Λc=0` ⇒ exact independent
hurdle-NB loglik; as `r→∞` tends to the Hurdle-Poisson marginal.
"""
function hurdle_nb_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, r::Real;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(HurdleNB(float(r)), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    HurdleNBFit

Result of [`fit_hurdle_nb_gllvm`](@ref): `βz`, `βc`, `Λc`, dispersion `r`, `loglik`,
`converged`, `iterations`.
"""
struct HurdleNBFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    r::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::HurdleNBFit)
    p, K = size(f.Λc)
    print(io, "HurdleNBFit(p=", p, ", K=", K, ", r=", round(f.r; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_hurdle_nb_gllvm(Y; K, …) -> HurdleNBFit

Fit a Hurdle-NB two-part GLLVM by L-BFGS over `[βz; βc; vec(Λc); log r]` (Λz=0).

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_hurdle_nb_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_hurdle_nb_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)
    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j]; c += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : log(max(s / c, 1.0))
    end
    Zc = [Y[t, j] > 0 ? log(max(Y[t, j], 0.5)) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(βz0, βc0, pack_lambda(Λc0), log(10.0))
    function negll(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        r = exp(θ[2p + rr + 1])
        v = try
            -hurdle_nb_marginal_loglik_laplace(Y, Λc, βz, βc, r; offsetc = offset, hessian = hessian,
                                               maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    r = exp(θ̂[2p + rr + 1])
    return HurdleNBFit(βz, βc, Λc, r, _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Delta-Gamma family — occurrence Bernoulli × positive Gamma (log-link mean).
# P(y=0)=1−π, density for y>0 = π·Gamma(y; shape α, scale μ/α), so E[y|y>0]=μ,
# Var[y|y>0]=μ²/α, μ=exp(η^c), π=logistic(η^z). The positive-block score/weight
# are the Gamma GLM pieces (log link, V(μ)=μ²/α): s^c=α(y−μ)/μ, W^c=α (y>0; 0 for
# y=0) — the expected-information weight, exactly as in families/gamma.jl. This is
# the second Delta family: same occurrence block as Delta-lognormal, Gamma swapped
# in for the positive part.
# ---------------------------------------------------------------------------

"""
    DeltaGamma(α = 1.0)

Marker for the Delta-Gamma two-part family (Bernoulli occurrence × positive
Gamma with log-link mean). Use with the unified entry point:

```julia
fit_gllvm(Y; family = DeltaGamma(), K = 2)
fit_gllvm(Y; family = DeltaGamma(4.0), K = 2)   # same fit — α is a tag payload
gllvm(@formula(y ~ 1), Y, site_data; family = DeltaGamma(), K = 2)
```

The marker's `α` field is a **tag payload** — it is never read by
[`fit_gllvm`](@ref) / [`fit_delta_gamma_gllvm`](@ref); the shared shape is
always estimated (`Var = μ²/α`, `μ = exp(η^c)`). Pass `DeltaGamma()` so the
public call need not invent a shape that is never read from the family
instance.

Named fitter [`fit_delta_gamma_gllvm`](@ref) remains available. +X,
`disp_group`, and `row_eff` are not admitted on this surface.
"""
struct DeltaGamma
    α::Float64
end

DeltaGamma() = DeltaGamma(1.0)

function _tp_pieces(f::DeltaGamma, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))                 # occurrence prob = logistic(η^z)
    Wz = π * (one(π) - π)
    if y > 0
        α = f.α
        μ = exp(ηc)                             # mean (log link)
        sc = α * (y - μ) / μ                    # ∂logf/∂η^c (Gamma GLM, log link)
        return (one(π) - π, sc, Wz, α,
                log(π) + logpdf(Gamma(α, μ / α), y))
    else
        return (-π, zero(ηc), Wz, zero(ηc), log1p(-π))
    end
end

# DeltaGamma: the positive part is Gamma(α, μ/α) at the log link, whose observed
# curvature is α·y/μ. Rather than re-derive it, reuse the already-verified
# implementation in `grouped_dispersion.jl` (`_gamma_grouped_laplace_weight`), which
# is the same formula under test elsewhere in the package.
function _tp_observed_Wc(f::DeltaGamma, y, ηc, Wc)
    y > 0 || return Wc          # absence cells contribute nothing to the positive part
    μ = exp(ηc)
    return _gamma_grouped_laplace_weight(:observed, Gamma(f.α, 1.0), μ, μ, y, LogLink())
end


"""
    delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α; Λz=nothing, kwargs...) -> Float64

Total two-part Laplace log-marginal for a Delta-Gamma GLLVM: occurrence probability
`π = logistic(β^z + Λ_z z)` (intercept-only by default, `Λ_z = 0`) times a positive
Gamma with mean `μ = exp(β^c + Λ_c z)` and shape `α` (`Var = μ²/α`). `Y` is p×n with
`0` for absences and positive reals for the positive part. With `Λ_c = 0` (and
`Λ_z = 0`) this reduces exactly to the independent two-part-regression log-likelihood.
"""
function delta_gamma_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, α::Real;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(DeltaGamma(float(α)), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α::AbstractVector; Λz=nothing, kwargs...) -> Float64

Per-trait-dispersion variant (`length(α) == p`, gllvmTMB's `log_phi_gamma_delta`,
`gllvmTMB.cpp:1195-1196`, length `n_traits`): species `t` gets its own shape `α[t]`
rather than one shared scalar. With a constant `α` this equals the shared-scalar
method above to machine precision (same `_tp_pieces` density per species, only the
dispersion lookup changes).
"""
function delta_gamma_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, α::AbstractVector;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    length(α) == p || throw(ArgumentError("length(α)=$(length(α)) must equal p=$p"))
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    fams = DeltaGamma.(float.(α))
    return twopart_marginal_loglik_laplace(fams, Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    DeltaGammaFit

Result of [`fit_delta_gamma_gllvm`](@ref): occurrence logits `βz` (length p),
positive-part log-mean intercepts `βc` (length p), positive-part loadings `Λc`
(p×K), the shape `α` (`Var = μ²/α`; a `Float64` under `disp_group == :shared`,
or a length-p `Vector{Float64}` under `disp_group == :species`), the maximised
`loglik`, `converged`, `iterations`, `predictor` (`:separate` default or
`:shared`), and `disp_group` (`:species` default, matching R gllvmTMB's
per-trait dispersion — `:shared` remains available as an explicit opt-in).
(`Λz = 0` —
occurrence is intercept-only — under `predictor == :separate`. Under
`predictor == :shared` (the gllvmTMB twin-identity mode: one linear predictor
drives both parts, `gllvmTMB.cpp:2831-2844`), `βz === βc` and `Λc` IS the
shared loadings matrix (also equal to `Λz`) — read `f.βc`/`f.Λc` as "the one
shared predictor" in that mode, `f.βz` is the identical array, not an
independent estimate.)
"""
struct DeltaGammaFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    α::Union{Float64, Vector{Float64}}
    loglik::Float64
    converged::Bool
    iterations::Int
    predictor::Symbol
    disp_group::Symbol
end

# Positional-compat constructors (hessian::Symbol precedent, e.g. binomial.jl):
# old 7-/8-arg call sites default the newly-appended trailing field(s).
DeltaGammaFit(βz, βc, Λc, α, loglik, converged, iterations) =
    DeltaGammaFit(βz, βc, Λc, α, loglik, converged, iterations, :separate, :shared)
DeltaGammaFit(βz, βc, Λc, α, loglik, converged, iterations, predictor) =
    DeltaGammaFit(βz, βc, Λc, α, loglik, converged, iterations, predictor, :shared)

function Base.show(io::IO, f::DeltaGammaFit)
    p, K = size(f.Λc)
    αstr = f.α isa Real ? string(round(f.α; sigdigits = 4)) : "per-trait"
    print(io, "DeltaGammaFit(p=", p, ", K=", K, ", α=", αstr,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_delta_gamma_gllvm(Y; K, …) -> DeltaGammaFit

Fit a Delta-Gamma two-part GLLVM by L-BFGS on the two-part Laplace marginal
([`delta_gamma_marginal_loglik_laplace`](@ref)), jointly estimating the shape
`α`. `Y` is p×n with `0` for absences and positive reals otherwise.
Finite-difference gradient; warm start = `logit(empirical P(y>0))` occurrence
intercepts + `log` mean positive value as log-mean intercepts + SVD of
positive-part log-residuals as loadings + a method-of-moments `α₀` from the
standardised positives.

`predictor` selects the parameterisation (`:separate` default, or `:shared`):
- `:separate` (default, previous/only behaviour before this kwarg existed):
  independent occurrence (`βz`, `Λz = 0`) and positive-part (`βc`, `Λc`)
  predictors; optimises over `[βz; βc; vec(Λc); log α]`.
- `:shared`: the counterpart's tied-predictor mode — ONE linear predictor
  `η = β + Λz` drives both
  parts (`βz ≡ βc ≡ β`, `Λz ≡ Λc ≡ Λ`), optimising over the smaller
  `[β; vec(Λ); log α]`. This is a twin-parity-oriented, restrictive
  parameterisation (occurrence log-odds and log-abundance move together by
  construction), not a general-purpose recommendation over `:separate`. A
  supplied `offset` is threaded into BOTH `offsetz` and `offsetc` under
  `:shared` so the tie `ηz ≡ ηc` is preserved.

`disp_group` selects the dispersion parameterisation (`:species` default, or
`:shared`) for grouped or species-specific dispersion:
- `:species` (default since `accept delta dispersion A`, 2026-09-24): one
  shape per species (`length(α) == p`), matching gllvmTMB's per-trait
  `log_phi_gamma_delta`. Adds `p − 1` free parameters relative to `:shared`;
  the two nest (`:shared` is the `:species` model with all p shapes tied), so
  `:species` logLik ≥ `:shared` logLik on the same data up to optimiser
  noise. Composes with either `predictor` mode. **Existing code that called
  this fitter without an explicit `disp_group` will see α change from a
  scalar to a length-p vector** — pass `disp_group = :shared` explicitly to
  keep the old one-scalar-per-model behaviour.
- `:shared` (previous default, before 2026-09-24; still available as an
  explicit opt-in): one scalar shape `α` for every species.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. DeltaGamma is the one
two-part family whose observed count-part weight is implemented, so the two
selectors genuinely differ here. This holds under both `predictor` modes.
"""
function fit_delta_gamma_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing, hessian::Symbol = :observed,
        predictor::Symbol = :separate,
        disp_group::Symbol = :species,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    # Validated up front: the objective wraps its body in a try/catch that converts any
    # throw into a large penalty, so a typo'd symbol would otherwise be laundered into
    # a converged-looking garbage fit.
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_delta_gamma_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    predictor in (:separate, :shared) || throw(ArgumentError(
        "fit_delta_gamma_gllvm: predictor must be :separate or :shared; got :$predictor"))
    disp_group in (:shared, :species) || throw(ArgumentError(
        "fit_delta_gamma_gllvm: disp_group must be :shared or :species; got :$disp_group"))
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j]; c += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : log(max(s / c, 1e-6))
    end
    # method-of-moments shape from standardised positives r = y/μ̂ (mean≈1, Var≈1/α)
    sumsq = 0.0; nres = 0
    sumsq_t = zeros(p); nres_t = zeros(Int, p)
    @inbounds for t in 1:p
        μt = exp(βc0[t])
        for j in 1:n
            if Y[t, j] > 0
                r = Y[t, j] / μt - 1.0
                sumsq += r^2; nres += 1
                sumsq_t[t] += r^2; nres_t[t] += 1
            end
        end
    end
    α0 = nres > 1 ? clamp((nres - 1) / sumsq, 0.1, 100.0) : 1.0
    # Per-trait warm start (disp_group == :species): per-species method-of-moments
    # shape, falling back to the pooled α0 for species with < 2 positive observations.
    α0vec = [nres_t[t] > 1 ? clamp((nres_t[t] - 1) / sumsq_t[t], 0.1, 100.0) : α0 for t in 1:p]
    ndisp = disp_group === :shared ? 1 : p
    Zc = [Y[t, j] > 0 ? log(max(Y[t, j], 1e-6)) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    # Offset (on the positive-part predictor η^c = β^c + offset + Λ^c z): remove it
    # from the loadings warm start so the SVD sees the offset-free residual.
    offset === nothing || (@inbounds for t in 1:p, j in 1:n
        Y[t, j] > 0 && (Zc[t, j] -= offset[t, j])
    end)
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    logα0 = disp_group === :shared ? [log(α0)] : log.(α0vec)

    if predictor === :separate
        θ0 = vcat(βz0, βc0, pack_lambda(Λc0), logα0)
        negll = θ -> begin
            βz = θ[1:p]; βc = θ[(p + 1):(2p)]
            Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
            α = disp_group === :shared ? exp(θ[2p + rr + 1]) : exp.(θ[(2p + rr + 1):(2p + rr + ndisp)])
            v = try
                -delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α; offsetc = offset,
                                                     hessian = hessian,
                                                     maxiter = newton_maxiter, tol = newton_tol)
            catch
                1e12
            end
            isfinite(v) ? v : 1e12
        end
    else # :shared — one η = β + Λz drives both parts; offset threads to both parts symmetrically.
        β0 = 0.5 .* (βz0 .+ βc0)
        θ0 = vcat(β0, pack_lambda(Λc0), logα0)
        negll = θ -> begin
            β = θ[1:p]
            Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
            α = disp_group === :shared ? exp(θ[p + rr + 1]) : exp.(θ[(p + rr + 1):(p + rr + ndisp)])
            v = try
                -delta_gamma_marginal_loglik_laplace(Y, Λ, β, β, α; Λz = Λ,
                        offsetz = offset, offsetc = offset, hessian = hessian,
                        maxiter = newton_maxiter, tol = newton_tol)
            catch
                1e12
            end
            isfinite(v) ? v : 1e12
        end
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    if predictor === :separate
        βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
        Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
        α = disp_group === :shared ? exp(θ̂[2p + rr + 1]) : exp.(θ̂[(2p + rr + 1):(2p + rr + ndisp)])
    else
        β = θ̂[1:p]
        Λc = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        α = disp_group === :shared ? exp(θ̂[p + rr + 1]) : exp.(θ̂[(p + rr + 1):(p + rr + ndisp)])
        βz = β; βc = β
    end
    return DeltaGammaFit(βz, βc, Λc, α, _fit_verdict(res)..., predictor, disp_group)
end

# ===========================================================================
# Zero-inflated families (ZIP / ZINB) — MIXTURE, not hurdle.
#
# A zero is produced by EITHER a structural-zero process (prob π) OR the count
# process (prob 1−π times the count's own P(0)):
#     P(y=0) = π + (1−π)·p₀,   P(y=k) = (1−π)·count(k)   (k ≥ 1)
# with π = logistic(η^z) and the count mean μ = exp(η^c). Unlike the hurdle
# families the count process is "active" at every observation, so a y=0 carries
# count-part information — its score s^c and Fisher weight W^cc are non-zero.
#
# These DO couple η^z and η^c at y=0 (∂²logf/∂η^z∂η^c ≠ 0). In the v1 convention
# the zero-inflation is per-species intercept-only (Λ_z = 0 — only β^z), so the
# latent z enters ONLY through η^c; the cross-term is multiplied by Λ_z = 0 in
# the shared-z mode-finder and drops out. The integral over z is therefore the
# same K-dimensional Laplace as the hurdle path, and these slot straight onto the
# existing `_tp_pieces` / `_twopart_mode` substrate — provided we supply the
# count-part score s^c, the *expected* Fisher information W^cc (≥ 0 ⇒ SPD), and
# the zero-inflated log-density. (Letting Λ_z load on z would need the 2×2
# cross-term machinery; that is a deliberate future extension.)
#
# W^cc is the expected information E[(s^c)²] in closed form (verified: ZIP → the
# Poisson weight μ as π → 0, ZINB → ZIP as r → ∞).
# ---------------------------------------------------------------------------

# Count-part expected information E[(∂logf/∂η^c)²] for the zero-inflated Poisson.
function _zi_Icc_pois(π, μ)
    e = exp(-μ); P0 = π + (one(π) - π) * e
    Icc = (one(π) - π) * (μ - e * μ^2) + (one(π) - π)^2 * e^2 * μ^2 / P0
    return max(Icc, 1e-12)
end

# Count-part expected information for the zero-inflated NB2 (dispersion r).
function _zi_Icc_nb(π, μ, r)
    p0 = (r / (r + μ))^r
    P0 = π + (one(π) - π) * p0
    Inb = μ * r / (r + μ)                    # = μ / (1 + μ/r), the NB2 info
    c = (r * μ / (r + μ))^2
    Icc = (one(π) - π) * (Inb - π * p0 * c / P0)
    return max(Icc, 1e-12)
end

"""
    ZIPoisson()

Marker for the zero-inflated Poisson family (structural zero prob `π=logistic(η^z)`
mixed with a Poisson count, mean `μ=exp(η^c)`).
"""
struct ZIPoisson end

function _tp_pieces(::ZIPoisson, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))
    μ = exp(ηc)
    Wz = π * (one(π) - π)                     # zero-inflation Fisher weight (unused: Λ_z = 0)
    Wcc = _zi_Icc_pois(π, μ)
    if y > 0
        return (-π, y - μ, Wz, Wcc, log1p(-π) + logpdf(Poisson(μ), Int(y)))
    else
        e = exp(-μ)
        P0 = π + (one(π) - π) * e
        g = (one(π) - π) * e / P0             # posterior P(count-zero | y=0)
        sz = π * (one(π) - π) * (one(π) - e) / P0
        return (sz, -g * μ, Wz, Wcc, log(P0))
    end
end

"""
    zip_marginal_loglik_laplace(Y, Λc, βz, βc; Λz=nothing, kwargs...) -> Float64

Two-part Laplace log-marginal for a zero-inflated Poisson GLLVM (structural-zero
`π=logistic(β^z)`, intercept-only by default; Poisson count with `μ=exp(β^c+Λ_c z)`).
`Y` is p×n integer counts. `Λc=0` ⇒ exact independent ZIP loglik; `β^z→−∞` ⇒ the
Poisson marginal.
"""
function zip_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(ZIPoisson(), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    ZIPFit

Result of [`fit_zip_gllvm`](@ref): structural-zero logits `βz`, count log-mean
intercepts `βc`, count loadings `Λc`, `loglik`, `converged`, `iterations`.
"""
struct ZIPFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZIPFit)
    p, K = size(f.Λc)
    print(io, "ZIPFit(p=", p, ", K=", K,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Shared warm start for zero-inflated count fits: structural-zero logits from the
# excess-zero fraction, count log-mean from the positive counts, SVD loadings.
function _zi_warmstart(Y::AbstractMatrix, K::Integer)
    p, n = size(Y)
    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        nz = count(==(0), view(Y, t, :))
        propzero = nz / n
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j]; c += 1
            end
        end
        μ̂ = c == 0 ? 1.0 : max(s / c, 1.0)
        βc0[t] = log(μ̂)
        excess = clamp(propzero - exp(-μ̂), 1e-3, 0.8)   # structural-zero share
        βz0[t] = log(excess / (1 - excess))
    end
    Zc = [Y[t, j] > 0 ? log(max(Y[t, j], 0.5)) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    return βz0, βc0, Λc0
end

"""
    fit_zip_gllvm(Y; K, …) -> ZIPFit

Fit a zero-inflated Poisson GLLVM by L-BFGS over `[βz; βc; vec(Λc)]` (Λz=0).
`Y` p×n integer counts. Finite-difference gradient; warm start from the
excess-zero fraction + positive-count log-means + SVD loadings.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_zip_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zip_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)
    βz0, βc0, Λc0 = _zi_warmstart(Y, K)
    θ0 = vcat(βz0, βc0, pack_lambda(Λc0))
    function negll(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        v = try
            -zip_marginal_loglik_laplace(Y, Λc, βz, βc; offsetc = offset, hessian = hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    return ZIPFit(βz, βc, Λc, _fit_verdict(res)...)
end

"""
    ZIPCovFit

Result of [`fit_zip_gllvm_cov`](@ref): ZIP under shared site-X with separate
structural-zero slopes `γz` and count slopes `γc` (Identity 2026-08-09;
`Λ_z = 0`). Fields: `βz`, `γz`, `βc`, `γc`, `γ_fixed` (Bool mask of fixed-zero
covariate columns applied to both parts), count loadings `Λc`, `loglik`,
`converged`, `iterations`.
"""
struct ZIPCovFit
    βz::Vector{Float64}
    γz::Vector{Float64}
    βc::Vector{Float64}
    γc::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λc::Matrix{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZIPCovFit)
    p, K = size(f.Λc); q = length(f.γc)
    print(io, "ZIPCovFit(p=", p, ", q=", q, ", K=", K,
          any(f.γ_fixed) ? ", fixed γ=$(count(f.γ_fixed))" : "",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_zip_gllvm_cov(Y; X, K, γ_fixed=nothing, …) -> ZIPCovFit

Fit a zero-inflated Poisson GLLVM **with shared site covariates**. It uses
separate occurrence and count coefficients:

- structural-zero logit: `η^z_{ts} = β^z_t + Σ_k X[t,s,k]·γ^z_k` (`Λ_z = 0`)
- count log-mean: `η^c_{ts} = β^c_t + Σ_k X[t,s,k]·γ^c_k + (Λ_c z_s)_t`

Packing is `[βz; γz; βc; γc; pack(Λc)]`. Offsets
`Oz = _build_offset(X, γz)` / `Oc = _build_offset(X, γc)` feed the existing
ZIP Laplace marginal. Finite-difference gradient; warm start from
[`fit_zip_gllvm`](@ref)'s excess-zero / positive-count / SVD start with `γ=0`.

`X` is `(p, n, q)`. `γ_fixed` optionally zeros selected covariate columns for
**both** parts (same contract as [`fit_gllvm_cov`](@ref)). Twin light RCall Δ
is out of scope (twin ZIP cut).

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. As for the no-covariate
fitter, the observed count-part weight is not yet specialised for this family,
so both selectors currently produce the identical objective (the
`TWOPART_KNOWN_OPEN` census gap).
"""
function fit_zip_gllvm_cov(Y::AbstractMatrix{<:Real}; X::AbstractArray{<:Real, 3},
        K::Integer, γ_fixed = nothing, hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zip_gllvm_cov: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    βz0, βc0, Λc0 = _zi_warmstart(Y, K)
    θ0 = vcat(βz0, zeros(q), βc0, zeros(q), pack_lambda(Λc0))
    function negll(θ)
        βz = θ[1:p]
        γz = θ[(p + 1):(p + q)]
        βc = θ[(p + q + 1):(2p + q)]
        γc = θ[(2p + q + 1):(2p + 2q)]
        Λc = unpack_lambda(θ[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        Oz = _build_offset(X_fit, γz)
        Oc = _build_offset(X_fit, γc)
        v = try
            -zip_marginal_loglik_laplace(Y, Λc, βz, βc;
                                         offsetz = Oz, offsetc = Oc, hessian = hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βẑ = θ̂[1:p]
    γz_free = θ̂[(p + 1):(p + q)]
    βĉ = θ̂[(p + q + 1):(2p + q)]
    γc_free = θ̂[(2p + q + 1):(2p + 2q)]
    Λĉ = unpack_lambda(θ̂[(2p + 2q + 1):(2p + 2q + rr)], p, K)
    γẑ = collect(Float64, _expand_fixed_zero(γz_free, γ_fixed_mask))
    γĉ = collect(Float64, _expand_fixed_zero(γc_free, γ_fixed_mask))
    return ZIPCovFit(βẑ, γẑ, βĉ, γĉ, collect(Bool, γ_fixed_mask), Λĉ,
                     _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Zero-inflated NB (ZINB) — structural zero × NB2 count with shared dispersion r.
# ---------------------------------------------------------------------------

"""
    ZINB(r)

Marker for the zero-inflated NB2 family (structural zero prob `π=logistic(η^z)`
mixed with an NB2 count, mean `μ=exp(η^c)`, dispersion `r`). `r→∞ ⇒ ZIP`.
"""
struct ZINB
    r::Float64
end

function _tp_pieces(f::ZINB, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))
    μ = exp(ηc); r = f.r
    Wz = π * (one(π) - π)
    Wcc = _zi_Icc_nb(π, μ, r)
    if y > 0
        sc = r * (y - μ) / (r + μ)
        logf = log1p(-π) + logpdf(NegativeBinomial(r, r / (r + μ)), Int(y))
        return (-π, sc, Wz, Wcc, logf)
    else
        p0 = (r / (r + μ))^r
        P0 = π + (one(π) - π) * p0
        g = (one(π) - π) * p0 / P0
        sz = π * (one(π) - π) * (one(π) - p0) / P0
        return (sz, -g * r * μ / (r + μ), Wz, Wcc, log(P0))
    end
end

"""
    zinb_marginal_loglik_laplace(Y, Λc, βz, βc, r; Λz=nothing, kwargs...) -> Float64

Two-part Laplace log-marginal for a zero-inflated NB2 GLLVM. `Λc=0` ⇒ exact
independent ZINB loglik; `r→∞` ⇒ the ZIP marginal.
"""
function zinb_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, r::Real;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(ZINB(float(r)), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    ZINBFit

Result of [`fit_zinb_gllvm`](@ref): `βz`, `βc`, `Λc`, dispersion `r`, `loglik`,
`converged`, `iterations`.
"""
struct ZINBFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    r::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZINBFit)
    p, K = size(f.Λc)
    print(io, "ZINBFit(p=", p, ", K=", K, ", r=", round(f.r; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_zinb_gllvm(Y; K, …) -> ZINBFit

Fit a zero-inflated NB2 GLLVM by L-BFGS over `[βz; βc; vec(Λc); log r]` (Λz=0).

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_zinb_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zinb_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)
    βz0, βc0, Λc0 = _zi_warmstart(Y, K)
    θ0 = vcat(βz0, βc0, pack_lambda(Λc0), log(10.0))
    function negll(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        r = exp(θ[2p + rr + 1])
        v = try
            -zinb_marginal_loglik_laplace(Y, Λc, βz, βc, r; offsetc = offset, hessian = hessian,
                                          maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    r = exp(θ̂[2p + rr + 1])
    return ZINBFit(βz, βc, Λc, r, _fit_verdict(res)...)
end

"""
    ZINegBin()

Public marker for the zero-inflated NB2 family (ZIP clone of [`ZIPoisson`](@ref)).
The Laplace kernel remains the internal `ZINB(r)` with one shared scalar `r`.
"""
struct ZINegBin end

"""
    ZINBCovFit

Result of [`fit_zinb_gllvm_cov`](@ref): ZINB under shared site-X with separate
structural-zero slopes `γz` and count slopes `γc` (Identity 2026-08-13;
`Λ_z = 0`) and **one shared scalar** NB2 dispersion `r` (packed as `log r`).
Fields: `βz`, `γz`, `βc`, `γc`, `γ_fixed`, count loadings `Λc`, `r`, `loglik`,
`converged`, `iterations`.
"""
struct ZINBCovFit
    βz::Vector{Float64}
    γz::Vector{Float64}
    βc::Vector{Float64}
    γc::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λc::Matrix{Float64}
    r::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZINBCovFit)
    p, K = size(f.Λc); q = length(f.γc)
    print(io, "ZINBCovFit(p=", p, ", q=", q, ", K=", K,
          ", r=", round(f.r; sigdigits = 4),
          any(f.γ_fixed) ? ", fixed γ=$(count(f.γ_fixed))" : "",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_zinb_gllvm_cov(Y; X, K, γ_fixed=nothing, …) -> ZINBCovFit

Fit a zero-inflated NB2 GLLVM **with shared site covariates**. It uses
separate occurrence and count coefficients plus one shared NB2 size:

- structural-zero logit: `η^z_{ts} = β^z_t + Σ_k X[t,s,k]·γ^z_k` (`Λ_z = 0`)
- count log-mean: `η^c_{ts} = β^c_t + Σ_k X[t,s,k]·γ^c_k + (Λ_c z_s)_t`
- **one shared scalar `r`** for all traits, optimised as `log r`

Packing is `[βz; γz; βc; γc; pack(Λc); log r]`. Offsets
`Oz = _build_offset(X, γz)` / `Oc = _build_offset(X, γc)` feed the existing
ZINB Laplace marginal (`r = exp(θ_tail)`). Finite-difference gradient; warm
start from [`fit_zinb_gllvm`](@ref)'s excess-zero / positive-count / SVD start
with `γ=0` and `log r = log(10)`.

`X` is `(p, n, q)`. `γ_fixed` optionally zeros selected covariate columns for
**both** parts (same contract as [`fit_zip_gllvm_cov`](@ref)). `θ_init` is an
optional packed start `[βz; γz; βc; γc; pack(Λc); log r]` (same length as the
free packed vector). Twin light RCall Δ is out of scope (twin ZINB cut).
Per-trait `r` is **not** the default.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. As for the no-covariate
fitter, the observed count-part weight is not yet specialised for this family,
so both selectors currently produce the identical objective (the
`TWOPART_KNOWN_OPEN` census gap).
"""
function fit_zinb_gllvm_cov(Y::AbstractMatrix{<:Real}; X::AbstractArray{<:Real, 3},
        K::Integer, γ_fixed = nothing, θ_init = nothing, hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zinb_gllvm_cov: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    nθ = 2p + 2q + rr + 1
    θ0 = if θ_init === nothing
        βz0, βc0, Λc0 = _zi_warmstart(Y, K)
        vcat(βz0, zeros(q), βc0, zeros(q), pack_lambda(Λc0), log(10.0))
    else
        θin = collect(Float64, θ_init)
        length(θin) == nθ || throw(ArgumentError(
            "θ_init length ($(length(θin))) must equal $nθ = 2p+2q+rr+1"))
        θin
    end
    function negll(θ)
        βz = θ[1:p]
        γz = θ[(p + 1):(p + q)]
        βc = θ[(p + q + 1):(2p + q)]
        γc = θ[(2p + q + 1):(2p + 2q)]
        Λc = unpack_lambda(θ[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        r = exp(θ[2p + 2q + rr + 1])
        Oz = _build_offset(X_fit, γz)
        Oc = _build_offset(X_fit, γc)
        v = try
            -zinb_marginal_loglik_laplace(Y, Λc, βz, βc, r; hessian = hessian,
                                         offsetz = Oz, offsetc = Oc,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    # iterations ≤ 0: evaluate at θ0 only (shared-start / transport checks; no
    # second LBFGS wander on the shared-r ridge).
    if iterations <= 0
        θ̂ = θ0
        nll0 = negll(θ̂)
        βẑ = θ̂[1:p]
        γz_free = θ̂[(p + 1):(p + q)]
        βĉ = θ̂[(p + q + 1):(2p + q)]
        γc_free = θ̂[(2p + q + 1):(2p + 2q)]
        Λĉ = unpack_lambda(θ̂[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        r̂ = exp(θ̂[2p + 2q + rr + 1])
        γẑ = collect(Float64, _expand_fixed_zero(γz_free, γ_fixed_mask))
        γĉ = collect(Float64, _expand_fixed_zero(γc_free, γ_fixed_mask))
        return ZINBCovFit(βẑ, γẑ, βĉ, γĉ, collect(Bool, γ_fixed_mask), Λĉ, r̂,
                          -nll0, true, 0)
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βẑ = θ̂[1:p]
    γz_free = θ̂[(p + 1):(p + q)]
    βĉ = θ̂[(p + q + 1):(2p + q)]
    γc_free = θ̂[(2p + q + 1):(2p + 2q)]
    Λĉ = unpack_lambda(θ̂[(2p + 2q + 1):(2p + 2q + rr)], p, K)
    r̂ = exp(θ̂[2p + 2q + rr + 1])
    γẑ = collect(Float64, _expand_fixed_zero(γz_free, γ_fixed_mask))
    γĉ = collect(Float64, _expand_fixed_zero(γc_free, γ_fixed_mask))
    return ZINBCovFit(βẑ, γẑ, βĉ, γĉ, collect(Bool, γ_fixed_mask), Λĉ, r̂,
                      _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Zero-inflated binomial (ZIB) — structural zero × Binomial(N, μ) count, with
# μ = logistic(η^c) and a shared scalar number of trials N. π → 0 ⇒ plain
# Binomial; N = 1 is the zero-inflated Bernoulli. Mirrors the ZINB substrate:
# the count-zero score magnitude rμ/(r+μ) is replaced by Nμ, and the NB2 count
# info μr/(r+μ) by the binomial-logit info Nμ(1−μ).
# ---------------------------------------------------------------------------

# Count-part expected information E[(∂logf/∂η^c)²] for the zero-inflated binomial
# (N trials, μ = success prob). As π → 0 this → Nμ(1−μ), the binomial-logit info.
function _zi_Icc_binom(π, μ, N)
    p0 = (one(μ) - μ)^N
    P0 = π + (one(π) - π) * p0
    Ibin = N * μ * (one(μ) - μ)              # = Nμ(1−μ), the binomial-logit info
    c = (N * μ)^2
    Icc = (one(π) - π) * (Ibin - π * p0 * c / P0)
    return max(Icc, 1e-12)
end

"""
    ZIB(N)

Marker for the zero-inflated binomial family: structural zero prob
`π = logistic(η^z)` mixed with a `Binomial(N, μ)` count, success probability
`μ = logistic(η^c)`, shared number of trials `N`. `π → 0 ⇒` plain Binomial;
`N = 1` is the zero-inflated Bernoulli.
"""
struct ZIB
    N::Int
end

function _tp_pieces(f::ZIB, y, ηz, ηc)
    π = inv(one(ηz) + exp(-ηz))
    μ = inv(one(ηc) + exp(-ηc))              # logit link for the count part
    N = f.N
    Wz = π * (one(π) - π)
    Wcc = _zi_Icc_binom(π, μ, N)
    if y > 0
        sc = y - N * μ
        logf = log1p(-π) + logpdf(Binomial(N, μ), Int(y))
        return (-π, sc, Wz, Wcc, logf)
    else
        p0 = (one(μ) - μ)^N
        P0 = π + (one(π) - π) * p0
        g = (one(π) - π) * p0 / P0           # posterior P(binomial-zero | y=0)
        sz = π * (one(π) - π) * (one(π) - p0) / P0
        return (sz, -g * N * μ, Wz, Wcc, log(P0))
    end
end

"""
    zib_marginal_loglik_laplace(Y, Λc, βz, βc, N; Λz=nothing, kwargs...) -> Float64

Two-part Laplace log-marginal for a zero-inflated binomial GLLVM (`N` trials).
`Y` is p×n with counts in `0:N`. `Λc = 0` ⇒ exact independent ZIB loglik;
`β^z → −∞` ⇒ the plain Binomial marginal.
"""
function zib_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, N::Integer;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(ZIB(Int(N)), Y, Λz_, Λc, βz, βc; kwargs...)
end

"""
    ZIBFit

Result of [`fit_zib_gllvm`](@ref): structural-zero logits `βz`, count success-logit
intercepts `βc`, count loadings `Λc`, the shared number of trials `N`, `loglik`,
`converged`, `iterations`.
"""
struct ZIBFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    N::Int
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZIBFit)
    p, K = size(f.Λc)
    print(io, "ZIBFit(p=", p, ", K=", K, ", N=", f.N,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Warm start for the zero-inflated binomial fit: success-logit intercept from the
# positive-part success fraction, structural-zero logit from the excess-zero share
# (over the binomial-zero rate), SVD loadings of the logit residuals.
function _zib_warmstart(Y::AbstractMatrix, N::Integer, K::Integer)
    p, n = size(Y)
    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    _logit(x) = (xx = clamp(x, 1e-3, 1 - 1e-3); log(xx / (1 - xx)))
    @inbounds for t in 1:p
        propzero = count(==(0), view(Y, t, :)) / n
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j] / N; c += 1
            end
        end
        μ̂ = c == 0 ? 1.0 / N : clamp(s / c, 1e-3, 1 - 1e-3)
        βc0[t] = _logit(μ̂)
        excess = clamp(propzero - (1 - μ̂)^N, 1e-3, 0.8)
        βz0[t] = log(excess / (1 - excess))
    end
    Zc = [Y[t, j] > 0 ? _logit(Y[t, j] / N) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    return βz0, βc0, Λc0
end

"""
    fit_zib_gllvm(Y; K, N, …) -> ZIBFit

Fit a zero-inflated binomial GLLVM by L-BFGS over `[βz; βc; vec(Λc)]` (Λz=0),
with a shared number of trials `N`. `Y` p×n with counts in `0:N`. Finite-difference
gradient; warm start from the excess-zero share + positive-part success logits +
SVD loadings.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_zib_gllvm(Y::AbstractMatrix{<:Real}; K::Integer, N::Integer,
        offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zib_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)
    βz0, βc0, Λc0 = _zib_warmstart(Y, N, K)
    θ0 = vcat(βz0, βc0, pack_lambda(Λc0))
    function negll(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        v = try
            -zib_marginal_loglik_laplace(Y, Λc, βz, βc, N; offsetc = offset, hessian = hessian,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    return ZIBFit(βz, βc, Λc, Int(N), _fit_verdict(res)...)
end

"""
    ZIBCovFit

Result of [`fit_zib_gllvm_cov`](@ref): ZIB under shared site-X with separate
structural-zero slopes `γz` and count slopes `γc` (Identity 2026-08-15;
`Λ_z = 0`) and fixed trial count `N`. Fields: `βz`, `γz`, `βc`, `γc`,
`γ_fixed` (Bool mask of fixed-zero covariate columns applied to both parts),
count loadings `Λc`, `N`, `loglik`, `converged`, `iterations`.
"""
struct ZIBCovFit
    βz::Vector{Float64}
    γz::Vector{Float64}
    βc::Vector{Float64}
    γc::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λc::Matrix{Float64}
    N::Int
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::ZIBCovFit)
    p, K = size(f.Λc); q = length(f.γc)
    print(io, "ZIBCovFit(p=", p, ", q=", q, ", K=", K, ", N=", f.N,
          any(f.γ_fixed) ? ", fixed γ=$(count(f.γ_fixed))" : "",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_zib_gllvm_cov(Y; X, K, N, γ_fixed=nothing, …) -> ZIBCovFit

Fit a zero-inflated binomial GLLVM **with shared site covariates**. It uses
separate occurrence and success coefficients:

- structural-zero logit: `η^z_{ts} = β^z_t + Σ_k X[t,s,k]·γ^z_k` (`Λ_z = 0`)
- count success logit: `η^c_{ts} = β^c_t + Σ_k X[t,s,k]·γ^c_k + (Λ_c z_s)_t`
- shared fixed trials `N` (same contract as [`fit_zib_gllvm`](@ref))

Packing is `[βz; γz; βc; γc; pack(Λc)]`. Offsets
`Oz = _build_offset(X, γz)` / `Oc = _build_offset(X, γc)` feed the existing
ZIB Laplace marginal. Finite-difference gradient; warm start from
[`fit_zib_gllvm`](@ref)'s excess-zero / positive-success / SVD start with `γ=0`.

`X` is `(p, n, q)`. `γ_fixed` optionally zeros selected covariate columns for
**both** parts (same contract as [`fit_zip_gllvm_cov`](@ref)). Twin light
RCall Δ is out of scope (twin ZIP/ZINB cut; no live twin ZIB).

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search is always Fisher-scored. As for the no-covariate
fitter, the observed count-part weight is not yet specialised for this family,
so both selectors currently produce the identical objective (the
`TWOPART_KNOWN_OPEN` census gap).
"""
function fit_zib_gllvm_cov(Y::AbstractMatrix{<:Real}; X::AbstractArray{<:Real, 3},
        K::Integer, N::Integer, γ_fixed = nothing, hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_zib_gllvm_cov: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    βz0, βc0, Λc0 = _zib_warmstart(Y, N, K)
    θ0 = vcat(βz0, zeros(q), βc0, zeros(q), pack_lambda(Λc0))
    function negll(θ)
        βz = θ[1:p]
        γz = θ[(p + 1):(p + q)]
        βc = θ[(p + q + 1):(2p + q)]
        γc = θ[(2p + q + 1):(2p + 2q)]
        Λc = unpack_lambda(θ[(2p + 2q + 1):(2p + 2q + rr)], p, K)
        Oz = _build_offset(X_fit, γz)
        Oc = _build_offset(X_fit, γc)
        v = try
            -zib_marginal_loglik_laplace(Y, Λc, βz, βc, N; hessian = hessian,
                                         offsetz = Oz, offsetc = Oc,
                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βẑ = θ̂[1:p]
    γz_free = θ̂[(p + 1):(p + q)]
    βĉ = θ̂[(p + q + 1):(2p + q)]
    γc_free = θ̂[(2p + q + 1):(2p + 2q)]
    Λĉ = unpack_lambda(θ̂[(2p + 2q + 1):(2p + 2q + rr)], p, K)
    γẑ = collect(Float64, _expand_fixed_zero(γz_free, γ_fixed_mask))
    γĉ = collect(Float64, _expand_fixed_zero(γc_free, γ_fixed_mask))
    return ZIBCovFit(βẑ, γẑ, βĉ, γĉ, collect(Bool, γ_fixed_mask), Λĉ, Int(N),
                     _fit_verdict(res)...)
end
