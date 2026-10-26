# True-parity campaign (C3), Julia-side convergence polish. W1-11 harness build; DRAFT rule in
# docs/dev-log/w1-11-convergence-parity-rule-DRAFT.md (needs maintainer signature, N9).
#
# Lives under tools/, not src/: it is measurement scaffolding, not package API, and changes no fit. It
#   (1) records the gradient max-abs of the objective a fitter minimised, at the point the fitter returned;
#   (2) applies ONE Newton step (gradient and Hessian both by central finite differences of that same objective);
#   (3) records the gradient max-abs, the objective and any derived quantity again, so Julia and R carry the
#       same before/after record (R side: run_R.R, CAMPAIGN_POLISH=1).
# It never feeds anything back into a fit, a tolerance or a receipt. run_J.jl includes it when CAMPAIGN_POLISH=1 and
# writes the record to a SEPARATE <cell>_J_polish.toml.
#
# Usage from run_J.jl (ordinal cell): include(polish_J.jl); rec = polish_ordinal_pertrait(fit, Y; link = ...)
using LinearAlgebra, GLLVModels
const _GM = GLLVModels

"Central-difference gradient of f at θ, step h (absolute; the campaign parameters are O(1) in the optimiser's coordinates)."
function fd_gradient(f, θ::AbstractVector; h::Real = 1e-5)
    g = similar(θ, Float64); x = collect(Float64, θ)
    @inbounds for i in eachindex(x)
        xi = x[i]; x[i] = xi + h; fp = f(x); x[i] = xi - h; fm = f(x); x[i] = xi
        g[i] = (fp - fm) / (2h)
    end
    g
end

"Central second-difference Hessian of f at θ (pairs spread over Threads). Noise floor about eps(f)/h^2."
function fd_hessian(f, θ::AbstractVector; h::Real = 1e-3)
    n = length(θ); x0 = collect(Float64, θ); fc = f(x0); H = zeros(n, n)   # (not `f0`: `2f0` parses as a Float32 literal)
    pairs = [(i, j) for i in 1:n for j in i:n]
    Threads.@threads for k in eachindex(pairs)
        i, j = pairs[k]; x = copy(x0)
        if i == j
            x[i] = x0[i] + h; fp = f(x); x[i] = x0[i] - h; fm = f(x)
            H[i, i] = (fp - 2 * fc + fm) / h^2
        else
            x[i] = x0[i] + h; x[j] = x0[j] + h; fpp = f(x)
            x[j] = x0[j] - h; fpm = f(x)
            x[i] = x0[i] - h; fmm = f(x)
            x[j] = x0[j] + h; fmp = f(x)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4h^2)
        end
    end
    Symmetric((H + H') / 2)
end

"""
    newton_polish(f, θ; hgrad, hhess) -> (θ_polished, record::Dict)

One Newton step `θ - H \\ g` on the objective `f` (a negative log-likelihood, minimised). The step is kept
whatever the objective does; the record holds f and the gradient max-abs before and after, the Hessian's extreme
eigenvalues and the step size. If H is not positive definite the step is NOT taken and the record says so.
"""
function newton_polish(f, θ::AbstractVector; hgrad::Real = 1e-5, hhess::Real = 1e-3)
    θ0 = collect(Float64, θ)
    f0 = f(θ0); g0 = fd_gradient(f, θ0; h = hgrad)
    t = time(); H = fd_hessian(f, θ0; h = hhess); wall_H = time() - t
    ev = eigvals(H)
    rec = Dict{String,Any}("n_par" => length(θ0), "f_before" => f0, "logLik_before" => -f0,
        "grad_max_abs_before" => maximum(abs, g0), "hessian_min_eig" => minimum(ev), "hessian_max_eig" => maximum(ev),
        "gradient_method" => "central finite difference, h = $hgrad", "hessian_method" => "central second differences of f, h = $hhess",
        "wall_hessian_sec" => wall_H)
    if !(minimum(ev) > 0)
        rec["step_taken"] = false; rec["skipped"] = "Hessian not positive definite"
        rec["f_after"] = f0; rec["logLik_after"] = -f0; rec["grad_max_abs_after"] = rec["grad_max_abs_before"]
        return θ0, rec
    end
    step = H \ g0; θ1 = θ0 .- step
    f1 = f(θ1); g1 = fd_gradient(f, θ1; h = hgrad)
    rec["step_taken"] = true; rec["step_max_abs"] = maximum(abs, step)
    rec["f_after"] = f1; rec["logLik_after"] = -f1; rec["grad_max_abs_after"] = maximum(abs, g1)
    rec["f_decreased"] = f1 <= f0
    θ1, rec
end

# ---- the ordinal per-trait cell: the SAME packed objective fit_ordinal_gllvm_pertrait minimises -------------------
# (src/families/ordinal.jl: θ = [β; pack_lambda(Λ); ψ], ψ the log-increments of each trait's cutpoints after τ1 = 0).
# This rebuilds that closure, it does not modify it; Newton-inner tolerances are the fitter's defaults (100, 1e-9).
function ordinal_pertrait_objective(Y::AbstractMatrix, C::AbstractVector{<:Integer}, K::Integer; link = _GM.LogitLink(),
                                    newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p = size(Y, 1); rr = _GM.rr_theta_len(p, K); ncut = sum(C .- 2)
    function negll(θ)
        β = @view θ[1:p]
        Λ = _GM.unpack_lambda(@view(θ[(p + 1):(p + rr)]), p, K)
        τ = _GM._unpack_cutpoints_pertrait(@view(θ[(p + rr + 1):(p + rr + ncut)]), C)
        v = try
            -_GM.ordinal_marginal_loglik_laplace_pertrait(Y, Λ, β, τ, C; link = link, mask = nothing,
                                                         maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll
end

"Pack an OrdinalPerTraitFit back into the optimiser's θ."
function ordinal_pertrait_theta(fit)
    p, K = size(fit.Λ); ψ = Float64[]
    for t in 1:p, c in 2:(fit.C[t] - 1); push!(ψ, log(fit.τ[t, c] - fit.τ[t, c - 1])); end
    vcat(fit.β, _GM.pack_lambda(fit.Λ), ψ)
end

"""
    polish_ordinal_pertrait(fit, Y; link) -> Dict

Before/after record for an `OrdinalPerTraitFit`: gradient max-abs and logLik at the fit, one Newton step, the same
again, `LLt` (ΛΛ') at the polished point and its largest change. Also checks that the rebuilt objective reproduces
`fit.loglik` (the check is recorded, not asserted).
"""
function polish_ordinal_pertrait(fit, Y::AbstractMatrix; link = _GM.LogitLink())
    p, K = size(fit.Λ); negll = ordinal_pertrait_objective(Y, fit.C, K; link = link)
    θ0 = ordinal_pertrait_theta(fit)
    θ1, rec = newton_polish(negll, θ0)
    rr = _GM.rr_theta_len(p, K)
    Λ1 = _GM.unpack_lambda(θ1[(p + 1):(p + rr)], p, K)
    rec["objective_reproduces_fit_loglik_abs_diff"] = abs(-negll(θ0) - fit.loglik)
    rec["LLt_polished"] = [collect(Float64, r) for r in eachrow(Λ1 * Λ1')]
    rec["LLt_polished_vs_default_max_abs_diff"] = maximum(abs, Λ1 * Λ1' .- fit.Λ * fit.Λ')
    rec
end
