# Draft collector only. The caller must establish source provenance, the exact
# fitted objective, the data/design, free indices and actual prior configuration.
using LinearAlgebra

# Declared derivative-error radii (consumed by tools/core070_postcoef_fit_health.py
# certify_hessian / certify_gradient). They are DECLARATIONS, not proofs, and
# are recorded only when the caller names the derivative method as exact
# forward-mode AD ("forwarddiff_exact"); any other method records no radius,
# so the certifier then reports INCOMPLETE rather than trusting an invented one.
#
# Rationale. Forward-mode AD evaluates the exact derivative of the evaluated
# floating-point program: there is no step size and no truncation error. The
# only error is binary64 rounding in the primal and tangent recurrences, bounded
# by gamma_m * (sum of absolute magnitudes of the contributing terms) with
# gamma_m ~ m * 2^-53 for m sequential roundings. We declare
#   radius = 2^10 * 2^-52 * max(1, S),
# i.e. a budget of 1024 ulps (m up to ~2000 roundings) times a scale S. S is
# max|H| for the Hessian, and max(|objective|, max|g|) for the gradient. S is a
# PROXY for the term-magnitude sum: cancellation can make the true sum larger,
# so the radius is not a theorem. The factor 2^10 and the proxy are the
# declared assumptions a reviewer must accept or replace. The Python floor
# k * 2^-52 * max|H| (k <= 32) is always below this radius because 2^10 > 32.
const POSTCOEF_AD_ULP_BUDGET = 1024.0
const POSTCOEF_AD_METHOD = "forwarddiff_exact"
const POSTCOEF_HESSIAN_COORDINATE = "packed_free_index"
# Every recorded radius is stamped with this status. There is no derivation of the
# radii for the Laplace objective (nor an independent one for the closed form), so
# the certifier carries the status through to a PASS as a visible qualifier.
const POSTCOEF_RADIUS_STATUS = "declared_unproven"
const POSTCOEF_LAMBDA_MAP_LAYOUT = "beta_free,sigma_eps,lambda_b_packed"
postcoef_ad_radius(scale) = POSTCOEF_AD_ULP_BUDGET * 2.0^-52 * max(1.0, scale)
const POSTCOEF_AD_HESSIAN_BASIS =
    "declared: 2^10 * 2^-52 * max(1, max|H|); forward-mode AD exact, rounding-only; scale is a proxy, not a bound"
const POSTCOEF_AD_GRADIENT_BASIS =
    "declared: 2^10 * 2^-52 * max(1, |objective|, max|g|); forward-mode AD exact, rounding-only; scale is a proxy, not a bound"

# penalty_state from the keyword arguments actually passed to the Gaussian
# `gllvm(@formula(...), Y, data; kwargs...)` call. That route has no prior,
# ridge or penalty argument (src/formula.jl, src/fit.jl at the audited tree),
# so only the default family and `K` prove "none". A keyword naming a prior,
# ridge or penalty is "active"; anything else is unproven and stays `nothing`
# (unrecorded), which the certifier refuses.
function postcoef_julia_penalty_state(call_kwargs)
    call_kwargs isa NamedTuple || return nothing
    names = keys(call_kwargs)
    any(k -> occursin(r"prior|ridge|penal"i, String(k)), names) && return "active"
    all(==(:K), names) || return nothing
    return "none"
end

function postcoef_julia_fit_health(case, objective, theta, optimizer_objective;
                                   penalty_state, free_indices,
                                   gradient_eval, hessian_eval,
                                   hessian_coordinate=nothing, derivative_method=nothing,
                                   lambda_constraint_map=nothing)
    out = Dict{String,Any}(
        "case" => case, "pd_hessian" => nothing, "max_abs_gradient" => nothing,
        "objective_verified" => false, "penalty_configuration_recorded" => false,
        "error" => nothing, "parameters" => nothing, "gradient" => nothing, "hessian" => nothing,
        "hessian_raw" => nothing, "hessian_asymmetry" => nothing, "hessian_asymmetry_limit" => nothing,
        "optimizer_objective" => nothing, "recomputed_objective" => nothing, "objective_difference" => nothing,
        "objective_roundoff_limit" => nothing,
        "free_indices" => nothing, "penalty_state" => penalty_state,
        "pinned_indices" => nothing, "hessian_coordinate" => nothing,
        "hessian_error_radius" => nothing, "hessian_error_radius_basis" => nothing,
        "packed_gradient_error_radius" => nothing, "radius_status" => nothing,
        "lambda_constraint_map" => lambda_constraint_map)
    fail(message) = (out["error"] = message; out)
    if !(theta isa AbstractVector{<:Real}) || isempty(theta) || !all(isfinite, theta)
        return fail("fit parameters unavailable or non-finite")
    end
    if !(free_indices isa AbstractVector) || isempty(free_indices) ||
       !all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= length(theta), free_indices) ||
       length(unique(free_indices)) != length(free_indices)
        return fail("free parameter indices unavailable or invalid")
    end
    out["parameters"] = collect(Float64, theta)
    out["free_indices"] = collect(Int, free_indices)
    # Complement of the free indices in 1..length(parameters), as the certifier requires.
    out["pinned_indices"] = collect(Int, setdiff(1:length(theta), free_indices))
    out["penalty_configuration_recorded"] = penalty_state isa AbstractString &&
                                               penalty_state in ("none", "active")
    if !(optimizer_objective isa Real) || optimizer_objective isa Bool || !isfinite(optimizer_objective)
        return fail("optimizer objective unavailable or non-finite")
    end
    out["optimizer_objective"] = Float64(optimizer_objective)
    try
        value = objective(theta)
        if !(value isa Real) || value isa Bool || !isfinite(value)
            return fail("recomputed objective unavailable or non-finite")
        end
        out["recomputed_objective"] = Float64(value)
        # Diagnostic equality only: an explicit Float64 roundoff bound,
        # independent of all signed parity and gradient thresholds.
        difference = abs(out["optimizer_objective"] - out["recomputed_objective"])
        limit = 8 * eps(Float64) * max(1.0, abs(out["optimizer_objective"]), abs(out["recomputed_objective"]))
        out["objective_difference"] = difference
        out["objective_roundoff_limit"] = limit
        out["objective_verified"] = isfinite(difference) && isfinite(limit) && difference <= limit
        gradient = gradient_eval(theta)
        if !(gradient isa AbstractVector{<:Real}) || length(gradient) != length(theta) ||
           !all(isfinite, gradient)
            return fail("gradient unavailable, malformed or non-finite")
        end
        out["gradient"] = collect(Float64, gradient)
        out["max_abs_gradient"] = maximum(abs, gradient[free_indices])
        if derivative_method == POSTCOEF_AD_METHOD
            # Radius of the packed gradient at theta (a separate bound from the optimizer-coordinate one).
            out["packed_gradient_error_radius"] = postcoef_ad_radius(
                max(abs(out["recomputed_objective"]), out["max_abs_gradient"]))
            out["radius_status"] = POSTCOEF_RADIUS_STATUS
        end
        hessian = hessian_eval(theta)
        if !(hessian isa AbstractMatrix{<:Real}) || size(hessian) != (length(theta), length(theta)) ||
           !all(isfinite, hessian)
            return fail("Hessian unavailable, malformed or non-finite")
        end
        # Match the package's existing symmetric-Hessian convention, but fail
        # closed beyond a documented floating-point roundoff bound.
        asymmetry = maximum(abs, hessian - hessian')
        limit = 32 * eps(Float64) * length(theta) * max(1.0, maximum(abs, hessian))
        out["hessian_raw"] = [collect(Float64, hessian[i, :]) for i in axes(hessian, 1)]
        out["hessian_asymmetry"] = asymmetry
        out["hessian_asymmetry_limit"] = limit
        asymmetry <= limit || return fail("Hessian asymmetry exceeds roundoff bound")
        hessian_sym = (hessian + hessian') ./ 2
        out["hessian"] = [collect(Float64, hessian_sym[i, :]) for i in axes(hessian, 1)]
        out["pd_hessian"] = isposdef(Symmetric(hessian_sym[free_indices, free_indices]))
        # Coordinate label is the caller's statement about `theta` and `objective`
        # (the adapter passes the fit's own packed theta); never inferred here.
        out["hessian_coordinate"] = hessian_coordinate isa AbstractString ? String(hessian_coordinate) : nothing
        if derivative_method == POSTCOEF_AD_METHOD
            out["hessian_error_radius"] = postcoef_ad_radius(maximum(abs, hessian))
            out["hessian_error_radius_basis"] = POSTCOEF_AD_HESSIAN_BASIS
            out["radius_status"] = POSTCOEF_RADIUS_STATUS
        end
    catch err
        err isa InterruptException && rethrow()
        return fail("diagnostic evaluation failed: " * sprint(showerror, err))
    end
    out
end

# The supplied factories must be the source-pinned model's own reconstruction
# and automatic derivatives. This adapter alone does not authenticate them.
function postcoef_julia_health_from_fit(case, fit, Y, X;
                                       penalty_state, nll_factory, pinned_indices,
                                       gradient_eval, hessian_eval, derivative_method=nothing,
                                       lambda_map=nothing)
    try
        theta = fit.pars.θ_packed
        pinned = pinned_indices(fit)
        if !(theta isa AbstractVector) || !(pinned isa AbstractVector) ||
           !all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= length(theta), pinned) ||
           length(unique(pinned)) != length(pinned)
            throw(ArgumentError("pinned parameter indices unavailable or invalid"))
        end
        free = setdiff(collect(eachindex(theta)), pinned)
        # The fit's own lambda_constraint map, from which the certifier recomputes
        # the pinned indices; the declared `pinned` above is never trusted alone.
        lcmap = lambda_map === nothing ? nothing : lambda_map(fit)
        objective = nll_factory(fit, Y, X, nothing)
        return postcoef_julia_fit_health(case, objective, theta, -fit.logLik;
            penalty_state=penalty_state, free_indices=free,
            gradient_eval=x -> gradient_eval(objective, x),
            hessian_eval=x -> hessian_eval(objective, x),
            # theta is fit.pars.θ_packed and the objective is the package's own
            # packed-theta NLL, so the Hessian is in the packed coordinate.
            hessian_coordinate=POSTCOEF_HESSIAN_COORDINATE, derivative_method=derivative_method,
            lambda_constraint_map=lcmap)
    catch err
        err isa InterruptException && rethrow()
        return Dict{String,Any}("case" => case, "pd_hessian" => nothing,
            "max_abs_gradient" => nothing, "objective_verified" => false,
            "penalty_configuration_recorded" => false,
            "error" => "fit adapter failed: " * sprint(showerror, err))
    end
end

# Recompute the gradient in the optimizer's own coordinates. Provenance of the
# supplied profile objective and equality to the fitted route must be reviewed.
function postcoef_julia_optimizer_health(result, objective;
                                         minimizer_eval, minimum_eval, gradient_eval,
                                         derivative_method=nothing)
    out = Dict{String,Any}("optimizer_parameters" => nothing,
        "gradient_error_radius" => nothing, "gradient_error_radius_basis" => nothing,
        "optimizer_gradient" => nothing, "optimizer_max_abs_gradient" => nothing,
        "optimizer_minimum" => nothing, "optimizer_profile_value" => nothing,
        "optimizer_profile_verified" => false, "optimizer_error" => nothing,
        "radius_status" => nothing)
    try
        x = minimizer_eval(result)
        x isa AbstractVector{<:Real} && !isempty(x) && all(isfinite, x) ||
            throw(ArgumentError("invalid optimizer parameters"))
        minimum = minimum_eval(result)
        value = objective(x)
        minimum isa Real && isfinite(minimum) && value isa Real && isfinite(value) ||
            throw(ArgumentError("invalid profile objective"))
        out["optimizer_parameters"] = collect(Float64, x)
        out["optimizer_minimum"] = Float64(minimum)
        out["optimizer_profile_value"] = Float64(value)
        out["optimizer_profile_verified"] = out["optimizer_minimum"] == out["optimizer_profile_value"]
        gradient = gradient_eval(objective, x)
        gradient isa AbstractVector{<:Real} && length(gradient) == length(x) && all(isfinite, gradient) ||
            throw(ArgumentError("invalid optimizer gradient"))
        out["optimizer_gradient"] = collect(Float64, gradient)
        out["optimizer_max_abs_gradient"] = maximum(abs, gradient)
        if derivative_method == POSTCOEF_AD_METHOD
            out["gradient_error_radius"] = postcoef_ad_radius(max(abs(out["optimizer_profile_value"]),
                                                                  out["optimizer_max_abs_gradient"]))
            out["gradient_error_radius_basis"] = POSTCOEF_AD_GRADIENT_BASIS
            out["radius_status"] = POSTCOEF_RADIUS_STATUS
        end
    catch err
        err isa InterruptException && rethrow()
        out["optimizer_error"] = sprint(showerror, err)
    end
    out
end

# Diagnostic reconstruction binding; admission still uses the existing gate.
function postcoef_julia_reconstruction_health(optimizer_parameters, fitted_parameters, fitted_loglik; recover_eval)
    out = Dict{String,Any}("recovered_parameters" => nothing, "recovered_loglik" => nothing,
        "parameters_match" => false, "loglik_matches" => false, "error" => nothing,
        "optimizer_parameters" => nothing)
    try
        rec = recover_eval(optimizer_parameters)
        pars = rec.parameters
        pars isa AbstractVector{<:Real} && fitted_parameters isa AbstractVector{<:Real} &&
            length(pars) == length(fitted_parameters) && all(isfinite, pars) &&
            all(isfinite, fitted_parameters) && isfinite(rec.loglik) && isfinite(fitted_loglik) ||
            throw(ArgumentError("invalid reconstruction"))
        # The point this reconstruction was recovered from; the certifier requires it to be
        # the optimizer block's own point.
        out["optimizer_parameters"] = collect(Float64, optimizer_parameters)
        out["recovered_parameters"] = collect(Float64, pars)
        out["recovered_loglik"] = Float64(rec.loglik)
        out["parameters_match"] = pars == fitted_parameters
        out["loglik_matches"] = rec.loglik == fitted_loglik
    catch err
        err isa InterruptException && rethrow()
        out["error"] = sprint(showerror, err)
    end
    out
end

# The fit's own lambda_constraint map for the certifier's pin binding (consumed by
# tools/core070_postcoef_fit_health.py _lambda_pin_binding). Layout: q_free betas,
# one log sigma, then the packed loadings (src/packing.jl _lower_index;
# src/confint_profile.jl _profile_all_term_names). Only the plain route (no W block,
# no diagonal, no phylogenetic term) has this layout: any other route yields
# `nothing`, which the certifier refuses. null = free entry, number = pinned value.
function postcoef_lambda_constraint_map(fit; q_free)
    model = fit.model
    (model.K_W == 0 && !model.has_diag && model.K_phy == 0 && !model.has_phy_unique) || return nothing
    p, K = Int(model.p), Int(model.K)
    M = hasproperty(fit.pars, :lambda_constraint) ? fit.pars.lambda_constraint : nothing
    free(v) = v isa Real && isnan(v)  # the package's own free marker
    entries = [Any[(M === nothing || free(M[i, k])) ? nothing : Float64(M[i, k]) for k in 1:K] for i in 1:p]
    Dict{String,Any}("layout" => POSTCOEF_LAMBDA_MAP_LAYOUT, "p" => p, "K" => K,
                     "q_free" => Int(q_free), "entries" => entries)
end
