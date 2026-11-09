# True-parity campaign (C3 ordinal and iSDM, C4 fungi), Julia-side convergence polish for the N9 rule
# (GATES.md "N9: convergence parity for C3 campaign rows", signed; stopping rule
# docs/dev-log/decisions/2026-10-21-p1-optimizer-stopping-rule.md).
#
# Lives under tools/, not src/: it is measurement scaffolding, not package API, and changes no fit. For one fitted
# cell it
#   (1) rebuilds the objective the fitter minimised (the same packed theta, the same Laplace inner tolerances) and
#       records how well it reproduces the fit's own logLik;
#   (2) runs a pre-registered Newton polish: at most POLISH_MAX_STEPS steps theta - H \ g, stopping at the FIRST
#       point where the gradient max-abs is strictly below 1e-5 AND the Hessian at that same point is positive
#       definite. Gradient and Hessian are evaluated at every point visited, so the Hessian reported for the compared
#       point is the Hessian AT that point. A step that raises the objective is recorded and NOT taken (the loop
#       stops at the previous point); a Hessian that is not positive definite stops the loop with no step;
#   (3) re-evaluates every compared quantity of the cell at the final point with the package's own extractors and
#       records it, plus a self-consistency check that the quantities are those of the final parameter vector.
# Nothing is fed back into a fit, a tolerance or a receipt. run_J.jl includes this file when CAMPAIGN_POLISH=1 and
# writes the record to a SEPARATE <cell>_J_polish.toml; polish_merge.py builds the compared raw file from it.
#
# Gradients: ordinal has no analytic gradient (its fitter uses central differences), so it uses a central difference
# of f (h = 1e-5) and a second-difference Hessian of f (h = 1e-3), as in PR #715. iSDM and fungi use the fitters'
# own analytic gradients (isdm_laplace_grad, _cov_negll_grad) and a Hessian by central differences of that analytic
# gradient (h = 1e-4), symmetrised. The analytic gradient is cross-checked against a central difference of f at the
# start point and the difference is recorded.
using LinearAlgebra, GLLVModels
const _GM = GLLVModels
const POLISH_GRADIENT_BOUND = 1e-5
const POLISH_MAX_STEPS = parse(Int, get(ENV, "CAMPAIGN_POLISH_MAX_STEPS", "3"))

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

"Hessian as the central-difference Jacobian of an analytic gradient g (columns spread over Threads), symmetrised."
function fd_jacobian_hessian(g, θ::AbstractVector; h::Real = 1e-4)
    n = length(θ); x0 = collect(Float64, θ); H = zeros(n, n)
    Threads.@threads for i in 1:n
        x = copy(x0)
        x[i] = x0[i] + h; gp = g(x); x[i] = x0[i] - h; gm = g(x)
        H[:, i] = (gp .- gm) ./ (2h)
    end
    Symmetric((H + H') / 2)
end

"""
    newton_polish_loop(f, gfun, hfun, θ; max_steps, bound) -> (θ_final, record::Dict)

The pre-registered polish: evaluate f, the gradient and the Hessian at the start; while the point does not meet
(gradient max-abs < bound AND Hessian positive definite) and fewer than `max_steps` steps were taken, take one
Newton step θ - H \\ g if H is positive definite and the step does not raise f, and evaluate again at the new point.
Every visited point is recorded (`points`); the final point is the last one recorded.
"""
function newton_polish_loop(f, gfun, hfun, θ0::AbstractVector; max_steps::Integer = POLISH_MAX_STEPS,
                            bound::Real = POLISH_GRADIENT_BOUND)
    evalpt(θ) = begin
        t = time(); g = gfun(θ); H = hfun(θ); wall = time() - t
        ev = eigvals(H)
        (f = f(θ), g = g, H = H, mineig = minimum(ev), maxeig = maximum(ev), wall = wall)
    end
    meets(e) = maximum(abs, e.g) < bound && e.mineig > 0
    θ = collect(Float64, θ0); e = evalpt(θ)
    pt(k, e) = Dict{String,Any}("step" => k, "f" => e.f, "logLik" => -e.f, "max_abs_gradient" => maximum(abs, e.g),
        "hessian_min_eigenvalue" => e.mineig, "hessian_max_eigenvalue" => e.maxeig,
        "hessian_positive_definite" => e.mineig > 0, "wall_gradient_hessian_sec" => e.wall)
    points = [pt(0, e)]; stopped = ""
    k = 0
    while !meets(e)
        if k >= max_steps; stopped = "step cap $max_steps reached before the criterion held"; break; end
        if !(e.mineig > 0); stopped = "Hessian not positive definite at step $k: no Newton step taken"; break; end
        step = e.H \ e.g; θn = θ .- step; fn = f(θn)
        if !(fn <= e.f)
            push!(points, Dict{String,Any}("step" => k + 1, "rejected" => true, "f" => fn, "step_max_abs" => maximum(abs, step),
                                           "reason" => "the Newton step raised the objective; not taken"))
            stopped = "Newton step $(k + 1) raised the objective"; break
        end
        k += 1; θ = θn; e = evalpt(θ)
        p = pt(k, e); p["step_max_abs"] = maximum(abs, step); push!(points, p)
    end
    met = meets(e)
    rec = Dict{String,Any}("n_par" => length(θ), "gradient_bound" => bound, "max_steps" => max_steps,
        "steps_taken" => k, "criterion_met" => met, "stopped" => met ? "criterion held (gradient max-abs < $bound and Hessian positive definite)" : stopped,
        "points" => points,
        "logLik_before" => points[1]["logLik"], "max_abs_gradient_before" => points[1]["max_abs_gradient"],
        "logLik" => -e.f, "max_abs_gradient" => maximum(abs, e.g),
        "hessian_positive_definite" => e.mineig > 0, "hessian_min_eigenvalue" => e.mineig, "hessian_max_eigenvalue" => e.maxeig,
        "theta" => collect(Float64, θ))
    θ, rec
end

maxdiff(a, b) = maximum(abs, collect(Float64, a) .- collect(Float64, b))
rowsof(M) = [collect(Float64, r) for r in eachrow(M)]

# ---- ordinal per-trait cell: the SAME packed objective fit_ordinal_gllvm_pertrait minimises ----------------------
# (src/families/ordinal.jl: θ = [β; pack_lambda(Λ); ψ], ψ the log-increments of each trait's cutpoints after τ1 = 0).
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

"Polish record for an OrdinalPerTraitFit; compared quantities (logLik, LLt, tau) at the final point."
function polish_ordinal_pertrait(fit, Y::AbstractMatrix; link = fit.link)
    p, K = size(fit.Λ); negll = ordinal_pertrait_objective(Y, fit.C, K; link = link)
    θ0 = ordinal_pertrait_theta(fit)
    θ1, rec = newton_polish_loop(negll, θ -> fd_gradient(negll, θ; h = 1e-5), θ -> fd_hessian(negll, θ; h = 1e-3), θ0)
    rec["gradient_method"] = "central finite difference of the rebuilt objective, h = 1e-5"
    rec["hessian_method"] = "central second differences of the rebuilt objective, h = 1e-3"
    rr = _GM.rr_theta_len(p, K); ncut = sum(fit.C .- 2)
    β1 = θ1[1:p]; Λ1 = _GM.unpack_lambda(θ1[(p + 1):(p + rr)], p, K)
    τ1 = _GM._unpack_cutpoints_pertrait(θ1[(p + rr + 1):(p + rr + ncut)], fit.C)
    rec["objective_reproduces_fit_loglik_abs_diff"] = abs(-negll(θ0) - fit.loglik)
    rec["beta"] = collect(Float64, β1); rec["LLt"] = rowsof(Λ1 * Λ1'); rec["tau"] = rowsof(τ1)
    rec["self_consistency_logLik_abs_diff"] = abs(-negll(θ1) - rec["logLik"])
    rec
end

# ---- iSDM cell: the objective fit_isdm_gllvm minimises (latent(..., unique = FALSE): θ = [b_fix; pack_lambda(Λ)]) ----
"""
Polish record for an IsdmFit built from `table` (the same IsdmTable fit_isdm_gllvm builds from the formula and data).
The compared quantities at the final point come from an IsdmFit rebuilt at that point by the package's own fitter
called with `b_init`, `Λ_init` at the final point and `iterations = 0` (so the modes, eta and predict use the package
code); the rebuild is asserted to return exactly the final parameters.
"""
function polish_isdm(ft, table; newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    ft.unique && error("polish_isdm covers latent(..., unique = FALSE) only (the campaign cell)")
    pX = size(table.X, 2); p = length(table.trait_levels); K = size(ft.Λ, 2)
    unpackθ(θ) = _GM._isdm_unpack(θ, pX, p, K)
    function negll(θ)
        b, Λ = unpackθ(θ)
        v = _GM.isdm_marginal_loglik_laplace(table, Λ, b; theta_diag_B = nothing, maxiter = newton_maxiter, tol = newton_tol)
        isfinite(v) ? -v : 1e12
    end
    function grad(θ)
        g = _GM.isdm_laplace_grad(table, collect(Float64, θ); K = K, with_unique = false, maxiter = newton_maxiter, tol = newton_tol)
        g === nothing && error("isdm_laplace_grad returned nothing (a mode search failed)")
        -g
    end
    θ0 = _GM._isdm_pack(ft.b_fix, ft.Λ)
    gcheck = maxdiff(grad(θ0), fd_gradient(negll, θ0; h = 1e-5))
    θ1, rec = newton_polish_loop(negll, grad, θ -> fd_jacobian_hessian(grad, θ; h = 1e-4), θ0)
    rec["gradient_method"] = "analytic: -isdm_laplace_grad (the fitter's own gradient)"
    rec["hessian_method"] = "central differences of the analytic gradient, h = 1e-4, symmetrised"
    rec["analytic_vs_fd_gradient_max_abs_diff_at_start"] = gcheck
    rec["objective_reproduces_fit_loglik_abs_diff"] = abs(-negll(θ0) - ft.loglik)
    b1, Λ1 = unpackθ(θ1)
    f1 = _GM.fit_isdm_gllvm(table; K = K, b_init = b1, Λ_init = Λ1, iterations = 0,
                            newton_maxiter = newton_maxiter, newton_tol = newton_tol)
    (f1.b_fix == b1 && f1.Λ == Λ1) || error("iterations = 0 rebuild moved the parameters (max |db| $(maxdiff(f1.b_fix, b1)))")
    rec["rebuilt_cells_converged"] = all(f1.cell_converged)
    rec["self_consistency_logLik_abs_diff"] = abs(f1.loglik - rec["logLik"])
    rec["beta"] = collect(Float64, f1.b_fix); rec["beta_terms"] = collect(String, f1.b_names); rec["LLt"] = rowsof(f1.Λ * f1.Λ')
    rec["_fit"] = f1   # removed by the caller before writing; used to evaluate predict at the final point
    rec
end

# ---- fungi cell: GllvmCovFit from gllvm(@formula(y ~ 1 + 3 covariates), Y, covdata; family = Binomial(), K = 2) ----
# The objective fit_gllvm_cov minimises, θ = [β; γ; pack_lambda(Λ)] (Binomial has no dispersion), with N = 1 and no mask.
function polish_cov(fit, Y::AbstractMatrix, X::AbstractArray{<:Real, 3}; newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    any(fit.γ_fixed) && error("polish_cov does not cover fixed covariate coefficients")
    _GM._cov_has_disp(fit.family) && error("polish_cov covers families without a dispersion (the Binomial fungi cell) only")
    p, n = size(Y); K = size(fit.Λ, 2); q = size(X, 3); rr = _GM.rr_theta_len(p, K)
    Nm = fill(1, p, n); msk = _GM._resolve_obs_mask(nothing, Y); lk = fit.link
    fam = _GM._cov_family(fit.family, NaN)
    function negll(θ)
        β = θ[1:p]; γ = θ[(p + 1):(p + q)]; Λ = _GM.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        O = _GM._build_offset(X, γ)
        v = try
            -_GM._marginal_loglik_offset(fam, Y, Nm, Λ, β, O, lk; mask = msk, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    function grad(θ)
        g = _GM._cov_negll_grad(fit.family, Y, Nm, X, collect(Float64, θ), p, q, K, rr, false, lk, msk, newton_maxiter, newton_tol)
        g === nothing && error("_cov_negll_grad returned nothing (a mode search failed)")
        collect(Float64, g)
    end
    θ0 = vcat(fit.β, fit.γ, _GM.pack_lambda(fit.Λ))
    gcheck = maxdiff(grad(θ0), fd_gradient(negll, θ0; h = 1e-5))
    θ1, rec = newton_polish_loop(negll, grad, θ -> fd_jacobian_hessian(grad, θ; h = 1e-4), θ0)
    rec["gradient_method"] = "analytic: _cov_negll_grad (the fitter's own gradient)"
    rec["hessian_method"] = "central differences of the analytic gradient, h = 1e-4, symmetrised"
    rec["analytic_vs_fd_gradient_max_abs_diff_at_start"] = gcheck
    rec["objective_reproduces_fit_loglik_abs_diff"] = abs(-negll(θ0) - fit.loglik)
    β1 = θ1[1:p]; γ1 = θ1[(p + 1):(p + q)]; Λ1 = _GM.unpack_lambda(θ1[(p + q + 1):(p + q + rr)], p, K)
    f1 = _GM.GllvmCovFit(fit.family, β1, γ1, fit.γ_fixed, Matrix{Float64}(Λ1), fit.dispersion, lk, -negll(θ1), fit.converged, fit.iterations)
    rec["beta"] = vcat(collect(Float64, β1), collect(Float64, γ1)); rec["LLt"] = rowsof(Λ1 * Λ1')
    rec["eta"] = vec(predict(f1, Y, X; type = :link))
    rec["self_consistency_logLik_abs_diff"] = abs(f1.loglik - rec["logLik"])
    rec
end
