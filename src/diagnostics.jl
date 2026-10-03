# Read-only diagnostics / model-comparison layer over already-fitted GLLVM
# objects. Ports of the gllvmTMB R diagnostics cluster — see each
# docstring for the exact R source file this mirrors
# (`.unlazy/core070-aghq/oracle-source/readback/R/…`). These are
# check/compare/inspect functions only: none of them fit a new model.
#
# Ground rules (see docs/dev-log/core070/diagnostics-slice-notes.md):
#   - loadings comparisons use ΛΛᵀ (tcrossprod) invariants ONLY — never a
#     signed entrywise Λ comparison (rotation/sign are not identified).
#   - where the R diagnostic needs a quantity GLLVModels.jl does not compute
#     (TMB's joint/marginal Laplace random-effect split, a native
#     mixed-family-per-trait surface, a "two-ψ" alternative
#     parameterisation, sdreport SEs for non-Gaussian fits, …) this file
#     implements the subset that exists on GLLVModels.jl fit objects and
#     documents the gap in the docstring. No stub silently claims a
#     result it did not compute.

using Random: AbstractRNG, default_rng

# ---------------------------------------------------------------------
# gllvmTMB_check_consistency — check-consistency.R:131-240
# ---------------------------------------------------------------------

"""
    gllvmTMB_check_consistency(fit::GllvmFit, y::AbstractMatrix;
                                n_sim=100, seed=nothing, X=nothing,
                                Σ_phy=nothing) -> NamedTuple

Port of R's `gllvmTMB_check_consistency()`
(`check-consistency.R:131-240`). R simulates `n_sim` datasets at the
fitted parameters and calls `TMB::checkConsistency()`, which tests
whether the mean *score* (gradient of the negative log-likelihood) at
those parameters is centred at zero — a Laplace/score-non-centring
diagnostic, not an accuracy certificate.

**Gap vs R**: GLLVModels.jl has no TMB joint/marginal Laplace random-effect
split, so this port re-simulates the single-tier (`K_W == 0 &&
!has_diag && K_phy == 0`, `β === nothing`) Gaussian generative model
directly from `fit.pars.Λ` / `fit.pars.σ_eps`, and tests centring of
the packed-NLL score via a Hotelling T² test on the `n_sim` score
vectors (rather than TMB's internal joint/marginal split). `estimate =
TRUE` (the `joint_p_value` re-fit path) is not implemented — that
field is always `missing`. Any other structure (`K_W>0`, `has_diag`,
`K_phy>0`, `has_phy_unique`, fixed-effect `β`) throws `ArgumentError` rather than
silently simulating the wrong generative model. The per-trait intercepts
estimated by `fit_gllvm(Y; family = Normal(), K)` are supported.

Returns a `NamedTuple` with fields mirroring the R object:
`marginal_p_value`, `marginal_bias` (`Dict{String,Float64}` per
parameter), `joint_p_value` (`missing`), `flagged_parameters`
(`Vector{String}`, `|t-stat| > 2` on the per-parameter bias), `n_sim`,
`threshold`, `diagnostics` (`Vector{String}`), `scores` (the raw
`n_sim × n_par` score matrix).
"""
function gllvmTMB_check_consistency(fit::GllvmFit, y::AbstractMatrix;
                                     n_sim::Integer = 100,
                                     seed::Union{Nothing,Integer} = nothing,
                                     X = nothing, Σ_phy = nothing)
    n_sim >= 2 || throw(ArgumentError("n_sim must be >= 2; got $n_sim"))
    m = fit.model
    (m.K_W == 0 && !m.has_diag && m.K_phy == 0 && !m.has_phy_unique) || throw(ArgumentError(
        "gllvmTMB_check_consistency only supports the single-tier Gaussian model " *
        "(K_W == 0, has_diag == false, K_phy == 0, has_phy_unique == false); the " *
        "fitted model has structure " *
        "GLLVModels.jl does not yet re-simulate for this check"))
    isempty(fit.pars.β) || _has_intercept_design(fit) || throw(ArgumentError(
        "gllvmTMB_check_consistency does not support fixed-effect design X yet"))

    rng = seed === nothing ? default_rng() : MersenneTwister(Int(seed))
    Λ = fit.pars.Λ
    σ_eps = fit.pars.σ_eps
    p, K = size(Λ)
    n = size(y, 2)
    θ̂ = fit.pars.θ_packed
    n_par = length(θ̂)
    terms, _ = _confint_all_term_names(fit)

    scores = Matrix{Float64}(undef, n_sim, n_par)
    for s in 1:n_sim
        Z = randn(rng, K, n)
        ysim = Λ * Z .+ σ_eps .* randn(rng, p, n)
        _has_intercept_design(fit) && (ysim .+= _intercept_mean(fit))
        nll_s = _confint_reconstruct_nll(fit, ysim, X, Σ_phy)
        scores[s, :] = ForwardDiff.gradient(nll_s, θ̂)
    end

    marginal_b = vec(Statistics.mean(scores; dims = 1))
    marginal_se = vec(Statistics.std(scores; dims = 1)) ./ sqrt(n_sim)
    tstat = [marginal_se[i] > 0 ? marginal_b[i] / marginal_se[i] : NaN for i in 1:n_par]

    threshold = 2.0
    flagged = String[terms[i] for i in 1:n_par if isfinite(tstat[i]) && abs(tstat[i]) > threshold]

    # Hotelling T² omnibus test that the mean score vector is zero,
    # using the sample covariance of the score rows (drop near-singular
    # directions rather than fail outright — mirrors R's "information
    # matrix singular" defensive path).
    Σ_scores = Statistics.cov(scores)
    marginal_p = NaN
    flags = String[]
    try
        Σreg = Σ_scores + 1e-10 * I
        T2 = n_sim * (marginal_b' * (Σreg \ marginal_b))
        F = T2 * (n_sim - n_par) / (n_par * (n_sim - 1))
        if n_sim > n_par && isfinite(F) && F >= 0
            fd = Distributions.FDist(n_par, n_sim - n_par)
            marginal_p = 1 - cdf(fd, F)
        else
            push!(flags, "information_matrix_singular")
        end
    catch
        push!(flags, "information_matrix_singular")
    end

    if !isnan(marginal_p) && marginal_p <= 0.05
        push!(flags, "marginal_score_non_centred")
    end
    if !isempty(flagged) && !("marginal_score_non_centred" in flags)
        push!(flags, "marginal_score_non_centred")
    end
    if isnan(marginal_p) && isempty(flags)
        push!(flags, "marginal_p_value_unavailable")
    end
    if isempty(flags)
        push!(flags, "no_centring_warning")
    end

    return (marginal_p_value = marginal_p,
            marginal_bias = Dict(terms[i] => marginal_b[i] for i in 1:n_par),
            joint_p_value = missing,
            flagged_parameters = flagged,
            n_sim = Int(n_sim),
            threshold = threshold,
            diagnostics = flags,
            scores = scores)
end

# ---------------------------------------------------------------------
# check_auto_residual — check-auto-residual.R
# ---------------------------------------------------------------------

"""
    check_auto_residual(fit) -> NamedTuple

Port of R's `check_auto_residual()` (`check-auto-residual.R`). R flags
two configurations that make `link_residual = "auto"` incoherent:
within-trait family mixing (error) and ordinal-probit traits (warning
— the latent residual is already standardised at 1, so `"auto"` would
over-count).

**Gap vs R**: GLLVModels.jl's fit types are one family per whole model (no
native per-trait mixed-family surface yet), so the family-mixing branch
never fires on the current family surface — `mixed_family = false`
always, documented rather than silently verified. The ordinal-probit
branch is fully checked against `fit.link`.

Returns `(coherent::Bool, mixed_family::Bool, ordinal_probit::Bool,
messages::Vector{String})`.
"""
function check_auto_residual(fit)
    messages = String[]
    mixed_family = false  # gap: no per-trait mixed-family GLLVModels.jl surface yet
    ordinal_probit = false
    if hasfield(typeof(fit), :link) && hasfield(typeof(fit), :C)
        link = fit.link
        if link isa ProbitLink
            ordinal_probit = true
            push!(messages,
                  "ordinal trait uses ProbitLink: link_residual = \"auto\" would add a " *
                  "second unit-variance term over the already-standardised latent residual")
        end
    end
    coherent = !mixed_family && !ordinal_probit
    return (coherent = coherent, mixed_family = mixed_family,
            ordinal_probit = ordinal_probit, messages = messages)
end

# ---------------------------------------------------------------------
# sanity_multi — postfit-2 required surface
# ---------------------------------------------------------------------

"""
    sanity_multi(fit; y=nothing, X=nothing, Σ_phy=nothing, grad_tol=1e-3) -> NamedTuple

Structural / convergence sanity checks over a fitted GLLVM, in the
spirit of R's `sanity()` adapted to the multi-response fit. Checks that
compose from what GLLVModels.jl already computes on the fit object:

  - `converged` — `fit.converged` when present.
  - `loadings_finite` — every entry of the loadings (`_loadings(fit)`)
    is finite.
  - `pd_hessian` — for `GllvmFit` with `y` supplied, whether the
    observed-information Hessian at the MLE is positive definite
    (reuses the same Hessian path as [`confint`](@ref)); `missing`
    otherwise (no generic Hessian path for the non-Gaussian families).
  - `gradient_norm` — for `GllvmFit` with `y` supplied, the Euclidean
    norm of the packed-NLL gradient at `θ̂`; `missing` otherwise.
  - `gradient_ok` — `gradient_norm < grad_tol` (or `missing`).

**Gap vs R**: R's `sanity()` also inspects TMB-specific boundary
conditions (random-effect variance component reports, `sdreport`
convergence codes) that have no GLLVModels.jl equivalent; those are not
checked here.

Returns `(pass::Bool, converged, loadings_finite::Bool, pd_hessian,
gradient_norm, gradient_ok, messages::Vector{String})`.
"""
function sanity_multi(fit; y = nothing, X = nothing, Σ_phy = nothing, grad_tol::Real = 1e-3)
    messages = String[]
    converged = hasfield(typeof(fit), :converged) ? fit.converged : missing
    if converged === false
        push!(messages, "optimizer did not report convergence")
    end

    Λ = _loadings(fit)
    loadings_finite = all(isfinite, Λ)
    loadings_finite || push!(messages, "non-finite loadings")

    pd_hessian = missing
    gradient_norm = missing
    gradient_ok = missing
    if fit isa GllvmFit && y !== nothing
        θ̂ = fit.pars.θ_packed
        nll = _confint_reconstruct_nll(fit, y, X, Σ_phy)
        g = ForwardDiff.gradient(nll, θ̂)
        gradient_norm = LinearAlgebra.norm(g)
        gradient_ok = gradient_norm < grad_tol
        gradient_ok || push!(messages, "gradient norm $(round(gradient_norm, digits = 6)) exceeds grad_tol")
        try
            H = ForwardDiff.hessian(nll, θ̂)
            Hsym = (H .+ H') ./ 2
            pd_hessian = isposdef(Hsym)
        catch
            pd_hessian = false
        end
        pd_hessian === false && push!(messages, "observed-information Hessian is not positive definite")
    end

    pass = loadings_finite && (converged !== false) &&
           (pd_hessian === missing || pd_hessian) &&
           (gradient_ok === missing || gradient_ok)

    return (pass = pass, converged = converged, loadings_finite = loadings_finite,
            pd_hessian = pd_hessian, gradient_norm = gradient_norm,
            gradient_ok = gradient_ok, messages = messages)
end

# ---------------------------------------------------------------------
# gllvmTMB_diagnose / check_gllvmTMB — diagnose.R:1548-2213
# ---------------------------------------------------------------------

"""
    gllvmTMB_diagnose(fit; y=nothing, X=nothing, Σ_phy=nothing,
                       var_tol=1e-4, corr_tol=0.995) -> NamedTuple

Port of R's `gllvmTMB_diagnose()` (`diagnose.R:2005-2213`), the
holistic fit-health summary that wraps `check_gllvmTMB()`'s boundary
scan. This port composes [`sanity_multi`](@ref) with a boundary-flag
scan over the implied `Σ_y`: for a `GllvmFit`, [`sigma_y_site`](@ref)
(all non-phylo tiers — Λ_B, Λ_W, σ²_B, σ²_W, σ_eps); for any other fit
type, `ΛΛᵀ + diag(σ_eps²)` (the only tier those fit types carry):

  - `variance_near_zero` — any of `σ_eps`, `σ_B`, `σ_W`, `σ_phy` within
    `var_tol` of the zero boundary **on the variance scale**: `σ_eps` and
    `σ_phy` are SD-parameterised (`fit.pars`), so they are SQUARED before
    comparison against `var_tol`; `σ²_B` and `σ²_W` are already
    variance-parameterised and compared directly. Mixing an SD-scale and a
    variance-scale threshold under one tolerance would flag inconsistently
    across parameterisations by orders of magnitude; comparing everything
    on the variance scale avoids that.
  - `correlation_near_boundary` — any off-diagonal entry of the
    Σ_y-implied correlation matrix with `|ρ| > corr_tol`.

**Gap vs R**: R's `.gllvmTMB_build_fit_health()` (`diagnose.R:15-127`)
also inspects TMB `sdreport` estimability flags, per-family boundary
patterns (binomial-prevalence loading rows, multinomial degeneracy,
ordinal cutpoint span, spatial-domain diameter — `diagnose.R:417-1548`)
that have no GLLVModels.jl equivalent surface yet; those checks are not
ported.

Returns `(pass::Bool, sanity::NamedTuple, boundary_flags::Vector{String},
messages::Vector{String})`.
"""
function gllvmTMB_diagnose(fit; y = nothing, X = nothing, Σ_phy = nothing,
                            var_tol::Real = 1e-4, corr_tol::Real = 0.995)
    sanity = sanity_multi(fit; y = y, X = X, Σ_phy = Σ_phy)
    boundary_flags = String[]
    messages = String[]

    # σ_eps and σ_phy are SD-parameterised; σ²_B and σ²_W are already
    # variance-parameterised. Compare all four on the VARIANCE scale so one
    # var_tol threshold means the same thing for every parameterisation.
    for nm in (:σ_eps, :σ²_B, :σ²_W, :σ_phy)
        if hasfield(typeof(fit), :pars) && haskey(fit.pars, nm) && fit.pars[nm] !== nothing
            is_sd = nm === :σ_eps || nm === :σ_phy
            v = fit.pars[nm]
            vals = v isa AbstractArray ? v : (v,)
            for x in vals
                xvar = is_sd ? x^2 : x
                if isfinite(xvar) && abs(xvar) < var_tol
                    push!(boundary_flags, "variance_near_zero:$(nm)")
                    scale_label = is_sd ? "SD, variance = $(round(xvar, digits = 10))" :
                                           "variance = $(round(xvar, digits = 10))"
                    push!(messages, "$(nm) ($(scale_label)) is within var_tol=$(var_tol) " *
                                     "of the zero boundary (variance scale)")
                    break
                end
            end
        end
    end

    Σy = _implied_Sigma_y(fit)
    d = sqrt.(diag(Σy))
    if all(isfinite, d) && all(d .> 0)
        R = Σy ./ (d * d')
        pcount = size(R, 1)
        for i in 1:pcount, j in (i + 1):pcount
            if abs(R[i, j]) > corr_tol
                push!(boundary_flags, "correlation_near_boundary:$(i),$(j)")
                push!(messages, "|corr($(i),$(j))| = $(round(R[i, j], digits = 4)) exceeds corr_tol")
            end
        end
    end

    pass = sanity.pass && isempty(boundary_flags)
    return (pass = pass, sanity = sanity, boundary_flags = boundary_flags, messages = messages)
end

"""
    check_gllvmTMB(fit; y=nothing, X=nothing, Σ_phy=nothing, kwargs...) -> NamedTuple

Port of R's `check_gllvmTMB()` (`diagnose.R:1548-2005`), the
umbrella pass/fail gate. This port aggregates
[`sanity_multi`](@ref), [`check_auto_residual`](@ref), and the
boundary scan from [`gllvmTMB_diagnose`](@ref) into a single verdict.

**Gap vs R**: see the gaps documented on `sanity_multi`,
`check_auto_residual`, and `gllvmTMB_diagnose` — this umbrella
inherits all of them (no TMB `sdreport` estimability scan, no
per-family boundary rows beyond the generic variance/correlation
scan, `mixed_family` always `false` on the current GLLVModels.jl family
surface).

Returns `(pass::Bool, sanity::NamedTuple, residual::NamedTuple,
diagnose::NamedTuple, messages::Vector{String})`.
"""
function check_gllvmTMB(fit; y = nothing, X = nothing, Σ_phy = nothing, kwargs...)
    diag = gllvmTMB_diagnose(fit; y = y, X = X, Σ_phy = Σ_phy, kwargs...)
    residual = check_auto_residual(fit)
    messages = vcat(diag.messages, residual.messages)
    pass = diag.pass && residual.coherent
    return (pass = pass, sanity = diag.sanity, residual = residual, diagnose = diag,
            messages = messages)
end

# ---------------------------------------------------------------------
# fit_diagnostic_table (renamed from diagnostic_table) — diagnostic-tables.R:53-151
# ---------------------------------------------------------------------

"""
    fit_diagnostic_table(fit; y=nothing, X=nothing, Σ_phy=nothing, kwargs...) -> NamedTuple

Port of R's `diagnostic_table()` (`diagnostic-tables.R:53-151`), a flat
row-per-check table over [`check_gllvmTMB`](@ref)'s components — the
Julia-idiomatic column-vectors-of-a-NamedTuple shape rather than a
`data.frame`. Renamed from `diagnostic_table` because
R's `diagnostic_table(x, table=)` requires `x` to already carry
`gllvmTMB_diagnostic` metadata attached by a prior call to
`predictive_check()`/`residuals()` — a different call shape from this
function, which takes the raw fit directly.

Returns `(check::Vector{String}, status::Vector{Symbol},
message::Vector{String})` with one row per named check
(`:converged`, `:loadings_finite`, `:pd_hessian`, `:gradient_ok`,
`:variance_boundary`, `:correlation_boundary`, `:auto_residual`); each
`status` is `:pass`, `:fail`, or `:unavailable` (the GLLVModels.jl-gap
cases documented as `missing` upstream).
"""
function fit_diagnostic_table(fit; y = nothing, X = nothing, Σ_phy = nothing, kwargs...)
    verdict = check_gllvmTMB(fit; y = y, X = X, Σ_phy = Σ_phy, kwargs...)
    s = verdict.sanity
    _status(x) = x === missing ? :unavailable : (x ? :pass : :fail)

    check = String[]
    status = Symbol[]
    message = String[]

    push!(check, "converged"); push!(status, _status(s.converged))
    push!(message, s.converged === missing ? "not reported on this fit type" : "")

    push!(check, "loadings_finite"); push!(status, _status(s.loadings_finite))
    push!(message, "")

    push!(check, "pd_hessian"); push!(status, _status(s.pd_hessian))
    push!(message, s.pd_hessian === missing ? "no generic Hessian path for this fit type" : "")

    push!(check, "gradient_ok"); push!(status, _status(s.gradient_ok))
    push!(message, s.gradient_ok === missing ? "no generic gradient path for this fit type" : "")

    vboundary = any(f -> startswith(f, "variance_near_zero"), verdict.diagnose.boundary_flags)
    push!(check, "variance_boundary"); push!(status, vboundary ? :fail : :pass)
    push!(message, join(filter(f -> startswith(f, "variance_near_zero"), verdict.diagnose.boundary_flags), "; "))

    cboundary = any(f -> startswith(f, "correlation_near_boundary"), verdict.diagnose.boundary_flags)
    push!(check, "correlation_boundary"); push!(status, cboundary ? :fail : :pass)
    push!(message, join(filter(f -> startswith(f, "correlation_near_boundary"), verdict.diagnose.boundary_flags), "; "))

    push!(check, "auto_residual"); push!(status, verdict.residual.coherent ? :pass : :fail)
    push!(message, join(verdict.residual.messages, "; "))

    return (check = check, status = status, message = message)
end

# ---------------------------------------------------------------------
# Principal angles between two column spaces. The naive `svd(A'B).S` on
# non-orthonormal A, B is NOT cos(angle) — it conflates the columns'
# norms/skew with the subspace geometry (identical subspaces spanned by
# differently-scaled bases can report a spuriously small max singular
# value, i.e. "separable" when they are the same space). Orthonormalize
# each column space first (thin QR basis), then take the SVD of the
# product of the two orthonormal bases — the textbook principal-angle
# construction (Björck & Golub 1973).
# ---------------------------------------------------------------------
function _principal_angles(A::AbstractMatrix, B::AbstractMatrix)
    QA = Matrix(qr(A).Q)[:, 1:size(A, 2)]
    QB = Matrix(qr(B).Q)[:, 1:size(B, 2)]
    σ = svd(QA' * QB).S
    return acos.(clamp.(σ, 0.0, 1.0))
end

# ---------------------------------------------------------------------
# diagnose_kernel_separability — kernel-helpers.R / kernel-keywords.R
# ---------------------------------------------------------------------

"""
    diagnose_kernel_separability(fit; angle_tol=1e-3) -> NamedTuple

Port of the kernel-separability check in `kernel-helpers.R` /
`kernel-keywords.R`: whether a multi-tier fit's B-tier and W-tier
loading column spaces are identifiably separate. Implemented here as
the smallest principal angle between `range(Λ_B)` and `range(Λ_W)`,
via the textbook principal-angle construction — orthonormalize each
column space first (thin QR basis), then `cos(angle) = σ_max` of the
SVD of the product of the two orthonormal bases (see
[`_principal_angles`](@ref); the naive `svd(Λ_B'Λ_W)` on the raw,
non-orthonormal loadings is NOT `cos(angle)` and can report identical
column spaces as separable). An angle near zero means the two tiers'
latent axes are not separable from the data.

**Gap vs R**: R's kernel-keyword machinery covers a broader family of
named structured-covariance kernels (spatial, phylogenetic,
`ar1`, …); this port only checks the two loading-tier case GLLVModels.jl
currently fits (`Λ_B` vs `Λ_W`). Single-tier fits (`K_W == 0`) return
`separable = missing` — there is nothing to separate.

**Caveat for `GllvmFit`:** each column of `y` is one unit observed once, so
the data identify only `Λ_B Λ_Bᵀ + Λ_W Λ_Wᵀ`, never the split between the
two tiers (#135). The angle reported for such a fit reflects the
starting values, and `separable = true` does not mean the tiers are
identified.
"""
function diagnose_kernel_separability(fit; angle_tol::Real = 1e-3)
    if !(fit isa GllvmFit) || fit.model.K_W == 0 || fit.pars.Λ_W === nothing
        return (separable = missing, min_principal_angle = missing,
                message = "no W-tier loadings on this fit; nothing to separate")
    end
    Λ_B = fit.pars.Λ
    Λ_W = fit.pars.Λ_W
    angles = _principal_angles(Λ_B, Λ_W)
    angle = isempty(angles) ? 0.0 : minimum(angles)
    separable = angle > angle_tol
    msg = separable ? "" : "smallest principal angle $(round(angle, digits = 6)) rad is below angle_tol"
    return (separable = separable, min_principal_angle = angle, message = msg)
end

# ---------------------------------------------------------------------
# compare_* — extract-sigma-table.R / extract-two-psi-cross-check.R /
# rotate-loadings.R (compare_loadings)
# ---------------------------------------------------------------------

# Generic σ_eps accessor across the fit types this file touches; returns
# 0.0 (no idiosyncratic-error tier) when the fit has none.
_sigma_eps_or_zero(fit) = (hasfield(typeof(fit), :pars) && haskey(fit.pars, :σ_eps)) ?
                          fit.pars.σ_eps :
                          (hasfield(typeof(fit), :σ_eps) ? fit.σ_eps : 0.0)

# For a GllvmFit, use the full-tier sigma_y_site (Λ_B, Λ_W, σ²_B, σ²_W,
# σ_eps — every non-phylo tier the fit carries), not just Λ (== Λ_B) and
# σ_eps: the naive ΛΛᵀ + diag(σ_eps²) silently drops the W-tier's
# contribution, which can report a spuriously inflated implied correlation
# on a genuinely well-separated multi-tier fit. Other fit types (single-Λ,
# single-σ_eps by construction) keep the exact ΛΛᵀ + diag(σ_eps²) formula.
_implied_Sigma_y(fit::GllvmFit) = sigma_y_site(fit)
_implied_Sigma_y(fit) = begin
    Λ = _loadings(fit)
    σ = _sigma_eps_or_zero(fit)
    Λ * Λ' .+ σ^2 .* Matrix(I, size(Λ, 1), size(Λ, 1))
end

"""
    compare_fits_Sigma_table(fit1, fit2) -> NamedTuple

Port of R's `extract-sigma-table.R` comparison surface. Compares the
implied `Σ_y = ΛΛᵀ + diag(σ_eps²)` of two fits of the same number of
traits `p`. Rotation/sign of Λ are not identified, so this compares
`Σ_y` itself (a tcrossprod invariant), never signed loading entries.

Renamed from `compare_Sigma_table` because
R's `compare_Sigma_table(x, truth, ...)` compares a fit's implied Σ_y
against a supplied GROUND-TRUTH matrix, not two fits against each other —
a different signature and purpose from this two-fit bridge.

**Gap vs R**: R's `extract_sigma_table()` also reports per-entry
bootstrap or profile SEs for the comparison; GLLVModels.jl's derived-CI
machinery (`confint_derived.jl`) covers single-fit Σ_y CIs but not a
paired-fit comparison SE, so only the point-estimate comparison is
returned here.

Returns `(Sigma1, Sigma2, diff, frobenius_norm::Float64,
max_abs_diff::Float64)`.
"""
function compare_fits_Sigma_table(fit1, fit2)
    Σ1 = _implied_Sigma_y(fit1)
    Σ2 = _implied_Sigma_y(fit2)
    size(Σ1) == size(Σ2) || throw(ArgumentError(
        "compare_fits_Sigma_table requires fits with the same number of traits; got $(size(Σ1)) vs $(size(Σ2))"))
    d = Σ1 .- Σ2
    return (Sigma1 = Σ1, Sigma2 = Σ2, diff = d,
            frobenius_norm = LinearAlgebra.norm(d), max_abs_diff = maximum(abs.(d)))
end

"""
    compare_loadings(fit1, fit2) -> NamedTuple

Port of R's loading-comparison surface (`rotate-loadings.R`). Per the
repo-wide rule, this NEVER compares signed Λ entries (rotation and
sign are not identified). It compares:

  - `frobenius_norm_LLt` — `‖Λ1Λ1ᵀ − Λ2Λ2ᵀ‖_F`, the rotation/sign-free
    tcrossprod invariant.
  - `principal_angles` — proper principal angles (radians) between
    `range(Λ1)` and `range(Λ2)` (see [`_principal_angles`](@ref):
    orthonormalize each column space first, then the SVD of the
    orthonormal-basis product), `0` meaning identical column spaces;
    only defined when both fits share the same latent rank `K`, else
    `missing`.
"""
function compare_loadings(fit1, fit2)
    Λ1 = _loadings(fit1)
    Λ2 = _loadings(fit2)
    size(Λ1, 1) == size(Λ2, 1) || throw(ArgumentError(
        "compare_loadings requires fits with the same number of traits; got $(size(Λ1,1)) vs $(size(Λ2,1))"))
    LLt1 = Λ1 * Λ1'
    LLt2 = Λ2 * Λ2'
    fro = LinearAlgebra.norm(LLt1 .- LLt2)
    angles = missing
    if size(Λ1, 2) == size(Λ2, 2)
        angles = _principal_angles(Λ1, Λ2)
    end
    return (frobenius_norm_LLt = fro, principal_angles = angles)
end

"""
    compare_fits_dep_vs_two_psi(fit_dep, fit_alt, n::Integer) -> NamedTuple

Port of R's `extract-two-psi-cross-check.R` dependent-vs-alternative
model-comparison bridge. `fit_dep` is the fit under test; `fit_alt` is
a fitted alternative on the same data; `n` is the shared number of
sites (BIC's sample size — passed explicitly because
`StatsAPI.nobs(fit)` requires the response matrix, which this
comparison does not otherwise need).

Renamed from `compare_dep_vs_two_psi` because
R's `compare_dep_vs_two_psi(fit_two_psi, ...)` takes a SINGLE fitted
phylogenetic "two-ψ" model and internally refits an alternative "dep"
model itself — a phylo-specific identifiability cross-check for a named
reparameterisation GLLVModels.jl does not implement, unlike this generic
two-fit bridge.

**Gap vs R**: R's "two-ψ" alternative is a specific named
reparameterisation (two independent latent-variable blocks) that
GLLVModels.jl does not implement as a distinct family; this port is the
generic bridge — Σ_y comparison via [`compare_fits_Sigma_table`](@ref) plus
an information-criterion delta — applicable to any two fits of the
same `p`, not specifically the R "two-ψ" family.

Returns `(sigma_comparison::NamedTuple, loglik_dep::Float64,
loglik_alt::Float64, aic_delta::Float64, bic_delta::Float64)` where
the deltas are `alt − dep` (negative favours `fit_dep`).
"""
function compare_fits_dep_vs_two_psi(fit_dep, fit_alt, n::Integer)
    sc = compare_fits_Sigma_table(fit_dep, fit_alt)
    ll_dep = _loglik(fit_dep)
    ll_alt = _loglik(fit_alt)
    k_dep = _nparams(fit_dep)
    k_alt = _nparams(fit_alt)
    aic_dep = 2 * k_dep - 2 * ll_dep
    aic_alt = 2 * k_alt - 2 * ll_alt
    bic_dep = k_dep * log(n) - 2 * ll_dep
    bic_alt = k_alt * log(n) - 2 * ll_alt
    return (sigma_comparison = sc, loglik_dep = ll_dep, loglik_alt = ll_alt,
            aic_delta = aic_alt - aic_dep, bic_delta = bic_alt - bic_dep)
end

"""
    compare_fits_indep_vs_two_psi(fit_indep, fit_alt, n::Integer) -> NamedTuple

Independent-vs-alternative counterpart of
[`compare_fits_dep_vs_two_psi`](@ref); same generic Σ_y + information-
criterion bridge and the same "two-ψ" naming gap documented there.
"""
compare_fits_indep_vs_two_psi(fit_indep, fit_alt, n::Integer) = compare_fits_dep_vs_two_psi(fit_indep, fit_alt, n)

# ---------------------------------------------------------------------
# predictive_check — predictive-diagnostics.R:78-229
# ---------------------------------------------------------------------

"""
    predictive_check(fit, y::AbstractMatrix; nsim=100, rng=Random.default_rng(),
                      stats=(mean=Statistics.mean, sd=Statistics.std)) -> NamedTuple

Port of R's `predictive_check()` (`predictive-diagnostics.R:78-229`):
simulate-and-summarise posterior/parametric predictive check. Draws
`nsim` replicate response matrices from the fitted model via the
existing per-family `simulate(fit, n; rng)` (`src/simulate_fit.jl`)
and compares each observed per-trait summary statistic in `stats`
against its simulated reference distribution with a two-sided
Bayesian/parametric p-value `2·min(mean(sim ≤ obs), mean(sim ≥ obs))`.

**Gap vs R**: works on any fit type with a `simulate(fit, n)` method
(the non-Gaussian families in `src/simulate_fit.jl`, plus the
loadings-only `GllvmFit` Gaussian simulator in
`src/families/aghq_gaussian_fit.jl`); other fit shapes throw
`ArgumentError` naming the gap. R's rootogram / rank-residual /
density-overlay *plots* (`predictive-diagnostics.R:933-1253`) are not
ported — this returns the underlying numeric summary only.

Returns `(stat::Vector{String}, trait::Vector{Int}, observed::Vector{Float64},
p_value::Vector{Float64}, sim_mean::Vector{Float64}, sim_sd::Vector{Float64})`.
"""
function predictive_check(fit, y::AbstractMatrix; nsim::Integer = 100,
                           rng::AbstractRNG = default_rng(),
                           stats::NamedTuple = (mean = Statistics.mean, sd = Statistics.std))
    nsim >= 2 || throw(ArgumentError("nsim must be >= 2; got $nsim"))
    hasmethod(simulate, Tuple{typeof(fit), Int}) || throw(ArgumentError(
        "predictive_check has no simulate(fit, n) method for $(typeof(fit)); " *
        "see the docstring gap note"))

    p, n = size(y)
    draws = Array{Float64}(undef, nsim, p, length(stats))
    for s in 1:nsim
        ysim = simulate(fit, n; rng = rng)
        for (k, f) in enumerate(stats)
            for t in 1:p
                draws[s, t, k] = f(view(ysim, t, :))
            end
        end
    end

    stat_out = String[]
    trait_out = Int[]
    observed_out = Float64[]
    p_out = Float64[]
    sim_mean_out = Float64[]
    sim_sd_out = Float64[]
    stat_names = collect(keys(stats))
    for (k, nm) in enumerate(stat_names)
        f = stats[nm]
        for t in 1:p
            obs = f(view(y, t, :))
            sim_col = view(draws, :, t, k)
            lo_p = Statistics.mean(sim_col .<= obs)
            hi_p = Statistics.mean(sim_col .>= obs)
            pv = 2 * min(lo_p, hi_p)
            push!(stat_out, String(nm))
            push!(trait_out, t)
            push!(observed_out, obs)
            push!(p_out, min(pv, 1.0))
            push!(sim_mean_out, Statistics.mean(sim_col))
            push!(sim_sd_out, Statistics.std(sim_col))
        end
    end

    return (stat = stat_out, trait = trait_out, observed = observed_out,
            p_value = p_out, sim_mean = sim_mean_out, sim_sd = sim_sd_out)
end

# ---------------------------------------------------------------------
# confint_inspect — confint-inspect.R:128-388
# ---------------------------------------------------------------------

"""
    confint_inspect(fit::GllvmFit, y::AbstractMatrix; level=0.95, parm=nothing,
                     X=nothing, Σ_phy=nothing, disagree_frac=0.25) -> NamedTuple

Port of R's `confint_inspect()` (`confint-inspect.R:128-388`): compares
Wald ([`confint`](@ref)) and profile-likelihood
([`profile_ci`](@ref)) intervals side by side and flags parameters
where the two disagree by more than `disagree_frac` of the Wald
half-width (R's `disagree_flag()`, `confint-inspect.R:297-`).

**Gap vs R**: R's version also supports bootstrap CIs
(`confint_bootstrap.jl` exists in this repo but is not wired into this
comparison to keep the check bounded) and renders a comparison plot
(`.confint_inspect_plot()`, `confint-inspect.R:388-490`); neither is
ported here.

Returns `(term::Vector{String}, wald_lower, wald_upper, profile_lower,
profile_upper, disagree::Vector{Bool})`.
"""
function confint_inspect(fit::GllvmFit, y::AbstractMatrix; level::Real = 0.95,
                          parm = nothing, X = nothing, Σ_phy = nothing,
                          disagree_frac::Real = 0.25)
    wald = confint(fit; level = level, parm = parm, y = y, X = X, Σ_phy = Σ_phy)
    terms, _ = _confint_all_term_names(fit)
    idx = [findfirst(==(t), terms) for t in wald.term]

    prof_lower = Vector{Float64}(undef, length(idx))
    prof_upper = Vector{Float64}(undef, length(idx))
    disagree = Vector{Bool}(undef, length(idx))
    for (k, i) in enumerate(idx)
        r = profile_ci(fit, i; level = level, y = y, X = X, Σ_phy = Σ_phy)
        prof_lower[k] = r.lower
        prof_upper[k] = r.upper
        hw = wald.upper[k] - wald.estimate[k]
        if hw > 0 && isfinite(r.lower) && isfinite(r.upper)
            disagree[k] = abs(r.lower - wald.lower[k]) > disagree_frac * hw ||
                          abs(r.upper - wald.upper[k]) > disagree_frac * hw
        else
            disagree[k] = false
        end
    end

    return (term = wald.term, wald_lower = wald.lower, wald_upper = wald.upper,
            profile_lower = prof_lower, profile_upper = prof_upper, disagree = disagree)
end

# ---------------------------------------------------------------------
# Deprecated forwarding shims. Renamed per maintainer decision
# docs/dev-log/decisions/2026-09-01-maintainer-decisions-round2-3.md #5:
# each old name shadowed an R function with different semantics (see the
# "Gap vs R" / rename notes on the corresponding new-name docstring above);
# freeing the old name for a future true mirror of R's own surface.
# ---------------------------------------------------------------------

function diagnostic_table(args...; kwargs...)
    Base.depwarn(
        "diagnostic_table is deprecated: renamed to fit_diagnostic_table; the name " *
        "diagnostic_table is reserved for a future R-mirroring surface",
        :diagnostic_table)
    return fit_diagnostic_table(args...; kwargs...)
end

function compare_Sigma_table(args...; kwargs...)
    Base.depwarn(
        "compare_Sigma_table is deprecated: renamed to compare_fits_Sigma_table; the " *
        "name compare_Sigma_table is reserved for a future R-mirroring " *
        "surface", :compare_Sigma_table)
    return compare_fits_Sigma_table(args...; kwargs...)
end

function compare_dep_vs_two_psi(args...; kwargs...)
    Base.depwarn(
        "compare_dep_vs_two_psi is deprecated: renamed to compare_fits_dep_vs_two_psi; " *
        "the name compare_dep_vs_two_psi is reserved for a future " *
        "R-mirroring surface", :compare_dep_vs_two_psi)
    return compare_fits_dep_vs_two_psi(args...; kwargs...)
end

function compare_indep_vs_two_psi(args...; kwargs...)
    Base.depwarn(
        "compare_indep_vs_two_psi is deprecated: renamed to compare_fits_indep_vs_two_psi; " *
        "the name compare_indep_vs_two_psi is reserved for a future " *
        "R-mirroring surface", :compare_indep_vs_two_psi)
    return compare_fits_indep_vs_two_psi(args...; kwargs...)
end
