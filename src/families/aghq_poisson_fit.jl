"""
    fit_poisson_gllvm(Y; K, n_init=1, aghq=false, aghq_control=(;), kwargs...) -> PoissonFit

Fit Poisson counts (responses × sites) with an ordinary loadings-only latent
block. Default `aghq=false` preserves the existing Laplace fit and its controls
(`link`, `mask`, `offset`, `X_lv`, starts, tolerances and iteration limits).

Set `aghq=5` for five nodes per latent axis or `aghq=true`/`:auto` for automatic
selection. `aghq=1` uses the Laplace optimizer. Auto declines at 20 traits;
quadrature requires log link, positive latent dimension ≤5 and no `X_lv`.
Ineligible requests retain Laplace with a warning and recorded reason.

`n_init` (default `1`, a single start) fits from `n_init` starts and returns the one with the
highest log-likelihood, preferring converged fits. Start 1 is the usual warm start (or your
`Λ_init`); starts `2:n_init` use lower-triangular `0.3 * randn` loadings from fixed seeds, so the
result is reproducible and the global random stream is untouched. The Laplace likelihood can have
several local optima, and a fit that reports `converged = true` is a local optimum only (issue
#915: on `ade4::mafragh flo table`, `K = 2`, the default start stops 8.4 log-likelihood units below
another converged start). Compare `loglik` across `n_init` values to check. Fit time is roughly
`n_init` times the single-start time. A start that throws is skipped; if all of starts `2:n_init`
fail, the first fit is returned. `n_init > 1` is not available with `aghq`
(`aghq_control.multistart` is its own multistart).

`aghq_control` is a named tuple of `n_adapt`, `iter_cap`, `continuation`,
`escalate_patience`, `shift_tol`, `grad_tol`, `grad_tol_rel`, `f_tol`, `rho_min`,
`optimizer_iterations`, `multistart`, `mode_gradient_tol`, `mode_maxiter`, and `mode_tol`. Defaults match the
unpenalized frozen-reference outer loop; no loading ridge is used. All supplied
starts are retained and converged usable results rank before raw objective.

AGHQ always uses observed adaptation curvature; an explicit `hessian` must be
`:observed`. Warm-start controls are copied and replayed by bootstrap.
`fit.integration` records actual integration, controls, final observed caches,
input identity and diagnostics. AGHQ convergence refers to the **frozen-node
surrogate gradient**, not the total derivative through changing adaptation.
`confint(fit,Y)` uses that fitted objective; prediction retains masks/offsets.
This function covers Poisson; the separate binomial candidate does not establish
the full family or structured-model Stage1a contract.
"""
function fit_poisson_gllvm(Y::AbstractMatrix;K::Integer,n_init::Integer=1,aghq=false,aghq_control=(;),kwargs...)
    n_init>=1 || throw(ArgumentError("fit_poisson_gllvm: n_init must be at least 1; got $n_init"))
    if haskey(kwargs, :optimizer) && kwargs[:optimizer] !== nothing &&
       _aghq_request(aghq) !== :off
        _refuse_optimizer(kwargs[:optimizer], "fit_poisson_gllvm", "an AGHQ request")
    end
    kwargs=_entry_offset_kwargs(kwargs,Y,"fit_poisson_gllvm")
    request=_aghq_request(aghq);c=_aghq_controls(aghq_control)
    n_init>1 && request!==:off && throw(ArgumentError(
        "fit_poisson_gllvm: n_init > 1 is not available with aghq; use aghq_control=(multistart=true,)"))
    request===:off && return _poisson_best_of_starts(Y,K,n_init,kwargs)
    get(kwargs,:weights,nothing)===nothing || _refuse_weights("fit_poisson_gllvm","AGHQ integration (aghq)")
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 5 : request
    link=get(kwargs,:link,LogLink())
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        get(kwargs,:X_lv,nothing)!==nothing ? :predictor_latents :
        !(link isa LogLink) ? :unsupported_link : K>5 ? :unaffordable_dimension :
        request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=_fit_poisson_gllvm_laplace(Y;K=K,kwargs...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _poisson_with_integration(base,info)
    end
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    problem=aghq_poisson_problem(Y,K;k=k,mask=get(kwargs,:mask,nothing),offset=get(kwargs,:offset,nothing),
        mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,
        mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge((;kwargs...),(mask=problem.data.mask,
        offset=get(kwargs,:offset,nothing)===nothing ? nothing : problem.data.offset,
        newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=_fit_poisson_gllvm_laplace(Int.(problem.data.responses);K=K,base_kwargs...)
    theta=vcat(base.β,pack_lambda(base.Λ));starts=[theta]
    if c.multistart
        alt=copy(theta);alt[p+1:end].=.3;push!(starts,alt)
    end
    result=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _poisson_with_integration(base,info)
    end
    selected=result.selected;t=selected.parameters
    prediction_offset=copy(problem.data.offset)
    raw_offset=get(kwargs,:offset,nothing)
    if raw_offset!==nothing
        for j in eachindex(prediction_offset)
            v=raw_offset[j]
            if v isa Real && isfinite(v);prediction_offset[j]=v;end
        end
    end
    data=AGHQPoissonData(problem.data.responses,problem.data.mask,prediction_offset)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    return PoissonFit(copy(t[1:p]),unpack_lambda(t[p+1:end],p,K),link,-selected.objective,
        selected.converged,selected.passes,nothing,copy(t),:observed,info)
end
# Best of `n_init` Laplace fits (issue #915). Start 1 is the unchanged default call; the extra
# starts replace `Λ_init` with a seeded lower-triangular draw. Converged fits rank before
# unconverged ones, then higher `loglik`; ties keep the earlier start, so `n_init = 1` and any
# case where start 1 wins are exactly the single-start fit.
function _poisson_best_of_starts(Y,K,n_init,kwargs)
    best=_fit_poisson_gllvm_laplace(Y;K=K,kwargs...)
    n_init==1 && return best
    p=size(Y,1)
    better(a,b)=a.converged!=b.converged ? a.converged : a.loglik>b.loglik
    for s in 2:n_init
        L=0.3 .* randn(Random.Xoshiro(915+s),p,K)
        for k in 1:K,i in 1:min(k-1,p);L[i,k]=0.0;end
        cand=try
            _fit_poisson_gllvm_laplace(Y;K=K,merge((;kwargs...),(Λ_init=L,))...)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        cand!==nothing && isfinite(cand.loglik) && better(cand,best) && (best=cand)
    end
    return best
end
_poisson_with_integration(f::PoissonFit,i)=PoissonFit(f.β,f.Λ,f.link,f.loglik,f.converged,
    f.iterations,f.alpha_lv,f.theta_packed,f.hessian,i,f.offset,f.weights)
_is_poisson_aghq(f)=f isa PoissonFit && f.integration!==nothing && f.integration.actual===:aghq

function _poisson_aghq_problem(fit::PoissonFit,Y;mask=nothing,offset=nothing,require_identity=false)
    size(Y,1)==size(fit.Λ,1) || throw(DimensionMismatch("AGHQ responses must match fitted trait count"))
    i=fit.integration;d=i.data
    # Compare only observed inputs; masked placeholders cannot alter the model.
    m=mask===nothing && size(Y)==size(d.responses) ? d.mask : mask
    o=offset===nothing && size(Y)==size(d.responses) ? d.offset : offset
    q=aghq_poisson_problem(Y,size(fit.Λ,2);k=i.k,mask=m,offset=o,
        mode_gradient_tol=i.controls.mode_gradient_tol,mode_maxiter=i.controls.mode_maxiter,mode_tol=i.controls.mode_tol)
    data=AGHQPoissonData(q.data.responses,q.data.mask,q.data.offset)
    same=_aghq_data_digest(data)==i.input_digest
    require_identity && !same && throw(ArgumentError("AGHQ inference requires the original observed responses, mask and offset"))
    !same && offset===nothing && any(!iszero,d.offset) &&
        throw(ArgumentError("new-data AGHQ prediction requires an explicit offset"))
    return q,same
end
function _poisson_aghq_scores(fit,Y;rotate=true,mask=nothing,offset=nothing)
    q,same=_poisson_aghq_problem(fit,Y;mask=mask,offset=offset)
    a=same ? fit.integration.caches : q.adapt(vcat(fit.β,pack_lambda(fit.Λ)))
    z=permutedims(reduce(hcat,(cache.mode for cache in a)))
    return rotate ? z*_svd_rotation(fit.Λ) : z
end
function _poisson_aghq_simulate(fit,n;rng=Random.default_rng(),offset=nothing)
    p,K=size(fit.Λ);stored=fit.integration.data.offset
    off=offset===nothing ? (size(stored,2)==n ? stored : any(!iszero,stored) ?
        throw(ArgumentError("AGHQ simulation at a new site count requires offset")) : zeros(p,n)) : offset
    size(off)==(p,n) && all(isfinite,off) || throw(ArgumentError("simulation offset must be finite p by n"))
    Y=Matrix{Int}(undef,p,n)
    for s in 1:n
        eta=fit.β+fit.Λ*randn(rng,K)+off[:,s]
        for t in 1:p;Y[t,s]=rand(rng,Poisson(exp(eta[t])));end
    end
    return Y
end

# Finite supplied offsets still define predictions for cells omitted from fitting.
function _aghq_prediction_offset(fit,Y,offset)
    d=fit.integration.data
    off=offset===nothing ? (size(Y)==size(d.offset) ? d.offset : zeros(size(Y))) : offset
    size(off)==size(Y) || throw(DimensionMismatch("prediction offset must match Y"))
    return map(v->v isa Real && isfinite(v) ? Float64(v) : NaN,off)
end

function _poisson_aghq_refit_kwargs(fit)
    i=fit.integration
    return merge(deepcopy(i.base_controls),(K=size(fit.Λ,2),aghq=i.k,
        aghq_control=i.controls,mask=copy(i.data.mask),offset=copy(i.data.offset)))
end
