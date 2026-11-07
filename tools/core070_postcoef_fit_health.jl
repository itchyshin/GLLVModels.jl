# Draft collector only. The caller must establish source provenance, the exact
# fitted objective, the data/design, free indices and actual prior configuration.
using LinearAlgebra

function postcoef_julia_fit_health(case, objective, theta, optimizer_objective;
                                   penalty_state, free_indices,
                                   gradient_eval, hessian_eval)
    out = Dict{String,Any}(
        "case" => case, "pd_hessian" => nothing, "max_abs_gradient" => nothing,
        "objective_verified" => false, "penalty_configuration_recorded" => false,
        "error" => nothing, "parameters" => nothing, "gradient" => nothing, "hessian" => nothing,
        "hessian_raw" => nothing, "hessian_asymmetry" => nothing, "hessian_asymmetry_limit" => nothing,
        "optimizer_objective" => nothing, "recomputed_objective" => nothing, "objective_difference" => nothing,
        "objective_roundoff_limit" => nothing,
        "free_indices" => nothing, "penalty_state" => penalty_state)
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
                                       gradient_eval, hessian_eval)
    try
        theta = fit.pars.θ_packed
        pinned = pinned_indices(fit)
        if !(theta isa AbstractVector) || !(pinned isa AbstractVector) ||
           !all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= length(theta), pinned) ||
           length(unique(pinned)) != length(pinned)
            throw(ArgumentError("pinned parameter indices unavailable or invalid"))
        end
        free = setdiff(collect(eachindex(theta)), pinned)
        objective = nll_factory(fit, Y, X, nothing)
        return postcoef_julia_fit_health(case, objective, theta, -fit.logLik;
            penalty_state=penalty_state, free_indices=free,
            gradient_eval=x -> gradient_eval(objective, x),
            hessian_eval=x -> hessian_eval(objective, x))
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
                                         minimizer_eval, minimum_eval, gradient_eval)
    out = Dict{String,Any}("optimizer_parameters" => nothing,
        "optimizer_gradient" => nothing, "optimizer_max_abs_gradient" => nothing,
        "optimizer_minimum" => nothing, "optimizer_profile_value" => nothing,
        "optimizer_profile_verified" => false, "optimizer_error" => nothing)
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
    catch err
        err isa InterruptException && rethrow()
        out["optimizer_error"] = sprint(showerror, err)
    end
    out
end

# Diagnostic reconstruction binding; admission still uses the existing gate.
function postcoef_julia_reconstruction_health(optimizer_parameters, fitted_parameters, fitted_loglik; recover_eval)
    out = Dict{String,Any}("recovered_parameters" => nothing, "recovered_loglik" => nothing,
        "parameters_match" => false, "loglik_matches" => false, "error" => nothing)
    try
        rec = recover_eval(optimizer_parameters)
        pars = rec.parameters
        pars isa AbstractVector{<:Real} && fitted_parameters isa AbstractVector{<:Real} &&
            length(pars) == length(fitted_parameters) && all(isfinite, pars) &&
            all(isfinite, fitted_parameters) && isfinite(rec.loglik) && isfinite(fitted_loglik) ||
            throw(ArgumentError("invalid reconstruction"))
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
