"""
    fit_binomial_gllvm(Y; K, N=nothing, aghq=false, aghq_control=(;), kwargs...)

Fit successes in a responses × sites matrix, with matching trials `N` (default
one). Default off and `aghq=1` preserve the Laplace optimizer. Positive node
counts or `true`/`:auto` request ordinary loadings-only AGHQ with logit, probit
or cloglog link. Automatic selection declines at 20 traits; dimensions above
five or predictor-informed scores retain Laplace with warning and provenance.

`aghq_control` uses the controls documented by [`fit_poisson_gllvm`](@ref).
Adaptation uses observed curvature, no ridge, and the frozen-reference outer
loop. Metadata retains trials, masks, offsets, starts, caches and convergence;
a usable nonconverged quadrature result is reported as nonconverged. Convergence
means the frozen-node gradient rule, not total re-adapted stationarity.
`confint(fit,Y)` uses the fitted frozen objective; `predict` returns success
probabilities and `simulate` returns counts using retained trials. At new sites,
provide trials and offsets explicitly when training used nonunit/nonzero values.
The original seed43 five-node R/Julia comparison remains a known failed gate;
this API is not a full parity or calibrated-inference claim.
"""
function fit_binomial_gllvm(Y::AbstractMatrix;K::Integer,aghq=false,aghq_control=(;),kwargs...)
    kwargs=_entry_offset_kwargs(kwargs,Y,"fit_binomial_gllvm")
    request=_aghq_request(aghq);c=_aghq_controls(aghq_control)
    if request!==:off && isfinite(get(kwargs,:loading_ridge,Inf))
        throw(ArgumentError("fit_binomial_gllvm: loading_ridge is not supported together with aghq (aghq != false)"))
    end
    request===:off && return _fit_binomial_gllvm_laplace(Y;K=K,kwargs...)
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 5 : request
    link=get(kwargs,:link,LogitLink())
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        get(kwargs,:X_lv,nothing)!==nothing ? :predictor_latents :
        !(link isa Union{LogitLink,ProbitLink,CLogLogLink}) ? :unsupported_link : K>5 ? :unaffordable_dimension :
        request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=_fit_binomial_gllvm_laplace(Y;K=K,kwargs...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _binomial_with_integration(base,info)
    end
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    problem=aghq_binomial_problem(Y,K;k=k,N=get(kwargs,:N,nothing),link=link,mask=get(kwargs,:mask,nothing),offset=get(kwargs,:offset,nothing),
        mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,
        mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge((;kwargs...),(N=Int.(problem.data.trials),mask=problem.data.mask,
        offset=get(kwargs,:offset,nothing)===nothing ? nothing : problem.data.offset,
        newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=_fit_binomial_gllvm_laplace(Int.(problem.data.responses);K=K,base_kwargs...)
    theta=vcat(base.β,pack_lambda(base.Λ));starts=[theta]
    if c.multistart
        alt=copy(theta);alt[p+1:end].=.3
        for t in 1:p
            # Frozen R fit-multi.R:6461–6469 uses raw mean successes,
            # intentionally not success/trial proportion, for the alternate start.
            obs=view(problem.data.mask,t,:);total=count(obs)
            mu=total>0 ? sum(problem.data.responses[t,obs])/total : .5
            eps=1/(4n);mu=clamp(mu,eps,1-eps)
            alt[t]=log(mu/(1-mu))
        end
        push!(starts,alt)
    end
    result=merge(aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...),
        (starts=deepcopy(starts),))
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _binomial_with_integration(base,info)
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
    prediction_trials=copy(problem.data.trials)
    raw_trials=get(kwargs,:N,nothing)
    for j in eachindex(prediction_trials)
        v=raw_trials===nothing ? 1 : raw_trials[j]
        prediction_trials[j]=v isa Real && isfinite(v) && isinteger(v) && 0<=v<=2.0^53 ? Float64(v) : NaN
    end
    data=AGHQBinomialData(problem.data.responses,prediction_trials,problem.data.mask,prediction_offset)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    return BinomialFit(copy(t[1:p]),unpack_lambda(t[p+1:end],p,K),link,-selected.objective,
        selected.converged,selected.passes,nothing,copy(t),:observed,nothing,info)
end
_binomial_with_integration(f::BinomialFit,i)=BinomialFit(f.β,f.Λ,f.link,f.loglik,f.converged,
    f.iterations,f.alpha_lv,f.theta_packed,f.hessian,f.saturation,i,f.loading_ridge)
_is_binomial_aghq(f)=f isa BinomialFit && f.integration!==nothing && f.integration.actual===:aghq


_is_aghq_fit(f)=_is_poisson_aghq(f) || _is_binomial_aghq(f)
_binomial_aghq_probability(eta,link)=link isa CLogLogLink ? -expm1(-exp(min(eta,700))) : clamp(linkinv(link,eta),1e-12,1-1e-12)

function _binomial_aghq_problem(fit::BinomialFit,Y;N=nothing,mask=nothing,offset=nothing,require_identity=false)
    size(Y,1)==size(fit.Λ,1) || throw(DimensionMismatch("AGHQ responses must match fitted trait count"))
    i=fit.integration;d=i.data;same_shape=size(Y)==size(d.responses)
    if !same_shape
        N===nothing && any(x->x!=1,d.trials) && throw(ArgumentError("new-data AGHQ prediction requires explicit trials N"))
        offset===nothing && any(!iszero,d.offset) && throw(ArgumentError("new-data AGHQ prediction requires an explicit offset"))
    end
    m=mask===nothing && same_shape ? d.mask : mask
    nt=N===nothing && same_shape ? d.trials : N
    o=offset===nothing && same_shape ? d.offset : offset
    q=aghq_binomial_problem(Y,size(fit.Λ,2);k=i.k,N=nt,link=fit.link,mask=m,offset=o,
        mode_gradient_tol=i.controls.mode_gradient_tol,mode_maxiter=i.controls.mode_maxiter,mode_tol=i.controls.mode_tol)
    data=AGHQBinomialData(q.data.responses,q.data.trials,q.data.mask,q.data.offset)
    same=_aghq_data_digest(data)==i.input_digest
    require_identity && !same && throw(ArgumentError("AGHQ inference requires original observed responses, trials, mask and offset"))
    !same && N===nothing && any(x->x!=1,d.trials) && throw(ArgumentError("new-data AGHQ prediction requires explicit trials N"))
    !same && offset===nothing && any(!iszero,d.offset) && throw(ArgumentError("new-data AGHQ prediction requires an explicit offset"))
    return q,same
end
function _binomial_aghq_scores(fit,Y;N=nothing,rotate=true,mask=nothing,offset=nothing)
    q,same=_binomial_aghq_problem(fit,Y;N=N,mask=mask,offset=offset)
    a=same ? fit.integration.caches : q.adapt(fit.theta_packed)
    z=permutedims(reduce(hcat,(cache.mode for cache in a)))
    return rotate ? z*_svd_rotation(fit.Λ) : z
end
function _binomial_aghq_simulate(fit,n;N=nothing,rng=Random.default_rng(),offset=nothing)
    d=fit.integration.data;p,K=size(fit.Λ)
    n>0 || throw(ArgumentError("simulation site count must be positive"))
    nt=N===nothing ? (size(d.trials,2)==n ? d.trials : all(==(1),d.trials) ? ones(p,n) :
        throw(ArgumentError("new-site AGHQ simulation requires trials N"))) : N
    off=offset===nothing ? (size(d.offset,2)==n ? d.offset : all(iszero,d.offset) ? zeros(p,n) :
        throw(ArgumentError("new-site AGHQ simulation requires offset"))) : offset
    size(nt)==(p,n) && all(v->v isa Real && isfinite(v) && isinteger(v) && 0<=v<=2.0^53,nt) ||
        throw(ArgumentError("simulation trials must be finite nonnegative integers p by n"))
    size(off)==(p,n) && all(v->v isa Real && isfinite(v),off) || throw(ArgumentError("simulation offset must be finite p by n"))
    Y=Matrix{Int}(undef,p,n)
    for s in 1:n
        eta=fit.β+fit.Λ*randn(rng,K)+off[:,s]
        for t in 1:p;Y[t,s]=rand(rng,Binomial(Int(nt[t,s]),_binomial_aghq_probability(eta[t],fit.link)));end
    end
    return Y
end
function _binomial_aghq_refit_kwargs(fit)
    i=fit.integration
    return merge(deepcopy(i.base_controls),(K=size(fit.Λ,2),N=map(x->isfinite(x) ? Int(x) : 0,i.data.trials),
        aghq=i.k,aghq_control=i.controls,mask=copy(i.data.mask),offset=copy(i.data.offset)))
end
