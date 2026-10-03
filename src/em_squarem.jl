# SQUAREM acceleration for the gradient-free EM fit of the Gaussian phylo
# GLLVM (`em_phylo.jl`).
#
# ---------------------------------------------------------------------------
# Why
# ---------------------------------------------------------------------------
# `em_fit_phylo` (in `em_phylo.jl`) converges to the dense MLE but, like every
# plain EM, crawls along a slow linear tail near the optimum — thousands of
# iterations at p = 100–500. SQUAREM (Varadhan & Roland 2008, Scand. J. Statist.
# 35: 335–353) is a deterministic *squared-extrapolation* accelerator for ANY
# EM map G: θ ↦ G(θ). It needs ONLY the update map — no gradients, no Hessian,
# no model change — and keeps the SAME fixed point (hence the same MLE), while
# typically cutting the iteration count 3–10×.
#
# One SQUAREM (order-3, "S3") step, given the current θ0:
#       θ1 = G(θ0)
#       θ2 = G(θ1)
#       r  = θ1 − θ0
#       v  = (θ2 − θ1) − r          # = θ2 − 2θ1 + θ0
#       α  = −‖r‖ / ‖v‖             # the SqS3 steplength
#       θ' = θ0 − 2α r + α² v
#       θ_new = G(θ')               # one stabilising EM step
# The stabilising EM step restores the EM monotonicity/feasibility guarantees
# the raw extrapolation can break. We additionally GLOBALISE with backtracking:
# if the candidate θ_new lowers the marginal log-lik (vs. the plain-EM value at
# θ2) or leaves the domain (σ²_eps must stay > 0), we halve the distance of α
# from the safe value −1 and retry; α = −1 reduces SQUAREM to two plain EM
# steps, so the fallback can never do worse than plain EM. This is exactly the
# Varadhan–Roland recommendation (their §6: project α back toward −1).
#
# ---------------------------------------------------------------------------
# How it wraps em_phylo.jl
# ---------------------------------------------------------------------------
# The EM map of `em_phylo.jl` is `_mstep_dense ∘ _estep_dense`: one E-step
# (closed-form Gaussian posterior) followed by one M-step (per-trait WLS +
# residual-trace σ²_eps). We DO NOT modify `em_phylo.jl`; we `include` it and
# build the one-step closure here. The parameter is packed as a flat vector
# θ = [vec(Λ_B); σ_eps; σ_phy] so the SQUAREM vector algebra (r, v, norms) is
# trivial. The log-lik is scored with the SAME dense closed form
# (`GLLVModels.gaussian_marginal_loglik`) the plain EM and the gradient fit use, so
# the trajectories are directly comparable.
#
# New file only — not wired into the GLLVModels module (mirrors the em_phylo.jl
# convention). `include` this file AFTER the GLLVModels module is loaded; it pulls
# in `em_phylo.jl` itself (guarded so a double-include is harmless).

if !isdefined(@__MODULE__, :_estep_dense)
    include(joinpath(@__DIR__, "em_phylo.jl"))
end

using LinearAlgebra
using Statistics

# ---------------------------------------------------------------------------
# Pack / unpack and the one-step EM map as a closure over the flat parameter.
# ---------------------------------------------------------------------------

# θ = [vec(Λ_B) (p·K_B); σ_eps (1); σ_phy (p)]
@inline _pack_phylo(Λ_B::AbstractMatrix, σ_eps::Real, σ_phy::AbstractVector) =
    vcat(vec(Λ_B), float(σ_eps), Vector{Float64}(σ_phy))

@inline function _unpack_phylo(θ::AbstractVector, p::Integer, K_B::Integer)
    Λ_B   = reshape(θ[1:(p * K_B)], p, K_B)
    σ_eps = θ[p * K_B + 1]
    σ_phy = θ[(p * K_B + 2):end]
    return Λ_B, σ_eps, σ_phy
end

# Domain guard: σ_eps must be a positive real (the M-step floors σ²_eps at eps()
# so a finite map output is always in-domain; this catches NaN/Inf/≤0 that a raw
# extrapolation θ' could produce before the stabilising EM step).
@inline _phylo_in_domain(θ::AbstractVector, p::Integer, K_B::Integer) = begin
    σ_eps = θ[p * K_B + 1]
    all(isfinite, θ) && isfinite(σ_eps) && σ_eps > 0
end

"""
    _em_map_phylo(θ, y, Σ_phy, p, K_B) -> θ'

One plain-EM update (E-step then M-step) of `em_phylo.jl`, on the packed
parameter `θ = [vec(Λ_B); σ_eps; σ_phy]`. This is the map G whose fixed point
is the (dense) MLE; SQUAREM accelerates iteration of exactly this map.
"""
function _em_map_phylo(θ::AbstractVector, y::AbstractMatrix,
                       Σ_phy::AbstractMatrix, p::Integer, K_B::Integer)
    Λ_B, σ_eps, σ_phy = _unpack_phylo(θ, p, K_B)
    ss = _estep_dense(y, Λ_B, σ_eps, σ_phy, Σ_phy)
    Λ_B′, σ_eps′, σ_phy′ = _mstep_dense(y, ss)
    return _pack_phylo(Λ_B′, σ_eps′, σ_phy′)
end

@inline function _loglik_phylo(θ::AbstractVector, y::AbstractMatrix,
                               Σ_phy::AbstractMatrix, p::Integer, K_B::Integer)
    Λ_B, σ_eps, σ_phy = _unpack_phylo(θ, p, K_B)
    return GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; σ_phy = σ_phy,
                                          Σ_phy = Σ_phy)
end

# ---------------------------------------------------------------------------
# SQUAREM driver
# ---------------------------------------------------------------------------

"""
    em_fit_phylo_squarem(y, K_B, Σ_phy;
                         λ_init=nothing, σ_eps_init=nothing, σ_phy_init=nothing,
                         tol=1e-9, max_iter=1000, max_backtrack=20,
                         assert_monotone=true,
                         safety_check=true, safety_polish_iters=50,
                         safety_tol=1e-3) -> EMPhyloFit

SQUAREM-accelerated version of `em_fit_phylo` (`em_phylo.jl`). Identical model,
identical warm start, identical convergence criterion (|Δ log-lik| < `tol`) and
identical dense log-lik scoring — only the iteration is accelerated by Varadhan
& Roland's (2008) order-3 squared extrapolation with a stabilising EM step and
steplength backtracking toward the safe value α = −1.

Returns the same `EMPhyloFit` struct as `em_fit_phylo`; `n_iter` counts SQUAREM
*cycles* (each cycle invokes the EM map three times: θ1, θ2, and the stabilising
step). The SQUAREM fixed point is a fixed point of the SAME EM map as plain EM
— but it need not be the SAME fixed point. SQUAREM takes much larger steps, and
from the shared warm start those steps can cross a flat ridge into a DIFFERENT,
inferior basin: the per-cycle log-lik stays monotone (it lands on a genuine EM
stationary point), yet the converged optimum is below the one plain EM reaches
from the identical start. This is path dependence, not premature stopping, and
it is observed at larger p (e.g. the seed-17 p=500 fixture in
`bench/em_squarem_bench.jl` / `test/test_em_squarem_safety.jl`, where raw
SQUAREM lands ≈24 log-lik units below plain EM).

INFERIOR-BASIN SAFETY CHECK (`safety_check`, default ON). Two failure modes,
detected by two screens:

* `safety_polish_iters` plain-EM polish steps from `θ_sq` (cheap). EM cannot
  lower the log-lik, so a polish GAIN beyond `safety_tol` proves `θ_sq` was not
  a maximum — a PREMATURE STOP on a slow ridge. Caught here for ~free.
* `safety_warmstart` (default ON): a full plain-EM run from the IDENTICAL warm
  start. This is the ONLY reliable detector of the PATH-DEPENDENT inferior
  basin, because there `θ_sq` is a *genuine* EM fixed point (`G(θ_sq) = θ_sq`)
  that the polish cannot escape — empirically, 3000 polish steps gain < 1e-5 on
  the seed-17 p=500 fixture even though `θ_sq` is ≈24 log-lik units low.

Whenever the better of {SQUAREM, plain-EM-from-warm-start} is the plain-EM
optimum (by more than `safety_tol`), that fit is returned with
`fallback_used = true`; otherwise the SQUAREM result is returned unchanged with
`fallback_used = false`. The warm-start screen costs a plain-EM run, so the safe
DEFAULT trades SQUAREM's iteration speedup for a guarantee of the plain-EM
optimum. Use `safety_warmstart = false` to keep only the cheap polish screen
(fast, but blind to path-dependent basins), or `safety_check = false` for the
raw SQUAREM result regardless (the original behaviour).

Each SQUAREM cycle is itself guaranteed no worse than two plain-EM steps: when
the extrapolation would lower the log-lik or leave the domain, α is repeatedly
moved halfway toward −1, and α = −1 makes θ' = θ2 (two EM steps) exactly. That
per-cycle monotonicity does NOT rule out the global path-dependence above, which
is what the safety check guards.

Note on `σ_phy` (#136; see the gllvmTMB parity page): here `σ_phy` is the
signed row-model parameter of `fit_gaussian_gllvm` with no `Λ_phy` columns,
so it enters the likelihood only through `B = (σ_phy * σ_phy') .* Σ_phy`.
Only a global flip `σ_phy → -σ_phy` leaves the likelihood unchanged: the
relative signs are identified, because they set the sign of each `B[t, t′]`,
so read `σ_phy` up to one overall sign. EM can stop in a worse sign pattern;
compare its log-likelihood with `fit_gaussian_gllvm` on the same data, or
warm-start EM from that fit. gllvmTMB's `phylo_unique` (now `phylo_indep()`,
or `phylo_latent(..., unique = TRUE)`) is a different model term, an
independent phylogenetic field per trait with its own scale, so compare
fitted models, not these parameters, and do not expect them to agree, even in
absolute value.
"""
function em_fit_phylo_squarem(y::AbstractMatrix, K_B::Integer,
                              Σ_phy::AbstractMatrix;
                              λ_init = nothing, σ_eps_init = nothing,
                              σ_phy_init = nothing,
                              tol = 1e-9, max_iter = 1000, max_backtrack = 20,
                              assert_monotone = true,
                              safety_check = true, safety_polish_iters = 50,
                              safety_tol = 1e-3, safety_warmstart = true)
    p, n = size(y)
    K_B ≥ 1 || throw(ArgumentError("K_B must be ≥ 1"))
    K_B < p || throw(ArgumentError("EM requires K_B < p; got K_B=$K_B, p=$p"))
    size(Σ_phy) == (p, p) ||
        throw(ArgumentError("Σ_phy must be p × p; got $(size(Σ_phy)) for p=$p"))

    yf = Matrix{Float64}(y)

    # ----- Warm start: IDENTICAL to em_fit_phylo (PPCA Λ_B, σ_eps; phylo SD
    #       from the marginal scale) so the two fitters start from one point. --
    if λ_init === nothing || σ_eps_init === nothing
        Λ0, σ0 = GLLVModels.ppca_init(yf, K_B)
        Λ_B   = λ_init === nothing ? Matrix{Float64}(Λ0) : Matrix{Float64}(λ_init)
        σ_eps = σ_eps_init === nothing ? float(σ0) : float(σ_eps_init)
    else
        Λ_B   = Matrix{Float64}(λ_init)
        σ_eps = float(σ_eps_init)
    end
    σ_phy = if σ_phy_init === nothing
        fill(0.1 * sqrt(mean(abs2, yf)), p)
    else
        Vector{Float64}(σ_phy_init)
    end

    # Stash the warm start so the inferior-basin safety check below can re-run
    # PLAIN EM from the IDENTICAL starting point (its escape hatch).
    Λ_B_warm = copy(Λ_B)
    σ_eps_warm = σ_eps
    σ_phy_warm = copy(σ_phy)

    θ = _pack_phylo(Λ_B, σ_eps, σ_phy)

    loglik_trace = Float64[]
    converged    = false
    cycles_run   = 0

    # Score and record the starting point (matches em_fit_phylo's first trace
    # entry: the log-lik at the warm-start parameters).
    ll_prev = _loglik_phylo(θ, yf, Σ_phy, p, K_B)
    push!(loglik_trace, ll_prev)

    for cycle in 1:max_iter
        cycles_run = cycle

        # --- two base EM steps -------------------------------------------------
        θ1 = _em_map_phylo(θ,  yf, Σ_phy, p, K_B)
        θ2 = _em_map_phylo(θ1, yf, Σ_phy, p, K_B)
        ll2 = _loglik_phylo(θ2, yf, Σ_phy, p, K_B)   # plain-EM log-lik this cycle

        # --- SQUAREM (order-3) extrapolation ----------------------------------
        r = θ1 .- θ
        v = (θ2 .- θ1) .- r                          # = θ2 − 2θ1 + θ0
        rn = norm(r)
        vn = norm(v)

        if vn < eps() || rn < eps()
            # No curvature / no movement: the two EM steps are the whole update.
            θ_cand = θ2
            ll_cand = ll2
        else
            α0 = -rn / vn                            # SqS3 steplength (≤ −1)
            α  = min(α0, -1.0)                       # never extrapolate "inward"
            θ_cand = θ2
            ll_cand = ll2
            # Backtrack α halfway toward −1 until the stabilised candidate is
            # in-domain AND does not lower the log-lik below the plain-EM value.
            for _ in 0:max_backtrack
                θ′ = θ .- (2α) .* r .+ (α^2) .* v    # squared extrapolation
                if _phylo_in_domain(θ′, p, K_B)
                    θ_new = _em_map_phylo(θ′, yf, Σ_phy, p, K_B)  # stabilising EM
                    ll_new = _loglik_phylo(θ_new, yf, Σ_phy, p, K_B)
                    if isfinite(ll_new) && ll_new ≥ ll2 - 1e-9
                        θ_cand  = θ_new
                        ll_cand = ll_new
                        break
                    end
                end
                α ≈ -1.0 && break                    # already at the safe value
                α = (α - 1.0) / 2                     # move halfway toward −1
                if α > -1.0                           # don't overshoot past −1
                    α = -1.0
                end
            end
        end

        ll = ll_cand
        push!(loglik_trace, ll)

        inc = ll - ll_prev
        if assert_monotone && inc < -1e-7
            error("SQUAREM-EM log-lik decreased by $(abs(inc)) at cycle " *
                  "$cycle (was $ll_prev, now $ll) — monotonicity violated.")
        end

        θ = θ_cand
        if abs(inc) < tol
            converged = true
            break
        end
        ll_prev = ll
    end

    ll_sq = _loglik_phylo(θ, yf, Σ_phy, p, K_B)   # SQUAREM converged log-lik

    # ----- Inferior-basin safety check ------------------------------------
    # SQUAREM's per-cycle monotonicity guarantees it lands on an EM STATIONARY
    # point, but NOT the same (or as good) a basin as plain EM from the shared
    # warm start. Two distinct failure modes, with different detectability:
    #
    #   (1) PREMATURE STOP. The |Δ log-lik| < tol test fires on a slow ridge
    #       where the iterate is NOT yet a fixed point — a short plain-EM polish
    #       from θ_sq then climbs. Detected cheaply by the polish below.
    #
    #   (2) PATH-DEPENDENT INFERIOR BASIN. SQUAREM's large extrapolation steps
    #       cross a flat ridge and converge to a GENUINE EM fixed point in a
    #       worse basin. A plain-EM polish from θ_sq CANNOT escape it (θ_sq is a
    #       true fixed point: G(θ_sq) = θ_sq), so the polish gains ~0 and is
    #       blind to this mode. Verified on the seed-17 p=500 fixture, where the
    #       EM-map residual at θ_sq is ~1e-6 yet θ_sq is ~24 log-lik units below
    #       the plain-EM optimum: 3000 polish steps gain < 1e-5. (See
    #       bench/em_squarem_characterise.jl.)
    #
    # The ONLY reliable detector of mode (2) is to compare against plain EM from
    # the identical warm start (`safety_warmstart`, default ON). That costs a
    # plain-EM run, so the safe DEFAULT trades SQUAREM's iteration speedup for a
    # correctness guarantee: it returns whichever of {SQUAREM, plain-EM-from-
    # warm-start} has the higher log-lik. Set `safety_warmstart = false` to keep
    # only the cheap polish screen, or `safety_check = false` for raw SQUAREM.
    fallback_used = false
    if safety_check
        needs_fallback = false

        # -- cheap screen: polish from θ_sq (catches mode (1) only) ----------
        if safety_polish_iters > 0
            θ_polish = θ
            for _ in 1:safety_polish_iters
                θ_polish = _em_map_phylo(θ_polish, yf, Σ_phy, p, K_B)
            end
            ll_polish = _loglik_phylo(θ_polish, yf, Σ_phy, p, K_B)
            needs_fallback = isfinite(ll_polish) && ll_polish > ll_sq + safety_tol
        end

        # -- authoritative screen: plain EM from the warm start (mode (2)) ---
        # Skip if the cheap screen already flagged a fallback. The plain-EM run
        # IS the fallback result, so this does no redundant work in that case.
        if safety_warmstart && !needs_fallback
            fb = em_fit_phylo(yf, K_B, Σ_phy;
                              λ_init = Λ_B_warm, σ_eps_init = σ_eps_warm,
                              σ_phy_init = σ_phy_warm,
                              tol = tol, max_iter = max_iter,
                              assert_monotone = assert_monotone)
            if isfinite(fb.logLik) && fb.logLik > ll_sq + safety_tol
                return EMPhyloFit(fb.Λ_B, fb.σ_eps, fb.σ_phy, fb.logLik,
                                  fb.n_iter, fb.converged, fb.loglik_trace,
                                  fb.blup_phy, fb.blup_phi, true)
            end
        elseif needs_fallback
            fb = em_fit_phylo(yf, K_B, Σ_phy;
                              λ_init = Λ_B_warm, σ_eps_init = σ_eps_warm,
                              σ_phy_init = σ_phy_warm,
                              tol = tol, max_iter = max_iter,
                              assert_monotone = assert_monotone)
            # The polish proved θ_sq is not a maximum; return the best of
            # {θ_polish, plain-EM-from-warm-start}, never the known-worse θ_sq (#146).
            if isfinite(fb.logLik) && fb.logLik > ll_polish
                return EMPhyloFit(fb.Λ_B, fb.σ_eps, fb.σ_phy, fb.logLik,
                                  fb.n_iter, fb.converged, fb.loglik_trace,
                                  fb.blup_phy, fb.blup_phi, true)
            end
            # θ_polish is the best candidate: fall through to the common tail,
            # which recomputes BLUPs and log-lik at θ.
            θ = θ_polish
            fallback_used = true
        end
    end

    Λ_B, σ_eps, σ_phy = _unpack_phylo(θ, p, K_B)

    # Refresh the ancestral-state BLUPs at the converged parameters.
    ss = _estep_dense(yf, Λ_B, σ_eps, σ_phy, Σ_phy)
    blup_phy = copy(ss.μ_z)
    blup_phi = copy(ss.μ_φ)

    ll_final = GLLVModels.gaussian_marginal_loglik(yf, Λ_B, σ_eps;
                                              σ_phy = σ_phy, Σ_phy = Σ_phy)
    if !isempty(loglik_trace) && ll_final > loglik_trace[end]
        push!(loglik_trace, ll_final)
    end

    # Global φ-orientation convention (IDENTICAL to em_fit_phylo): anchor the
    # sign so the dominant-magnitude trait's σ_phy is ≥ 0. Flipping ALL σ_phy
    # signs leaves every B[t,t'] = σ_phy[t] σ_phy[t'] Σ_phy[t,t'] unchanged.
    t_anchor = argmax(abs.(σ_phy))
    if σ_phy[t_anchor] < 0
        σ_phy = -σ_phy
        blup_phi = -blup_phi
    end

    return EMPhyloFit(Matrix{Float64}(Λ_B), σ_eps, Vector{Float64}(σ_phy),
                      ll_final, cycles_run, converged, loglik_trace,
                      blup_phy, blup_phi, fallback_used)
end

const fit_phylo_squarem = em_fit_phylo_squarem
