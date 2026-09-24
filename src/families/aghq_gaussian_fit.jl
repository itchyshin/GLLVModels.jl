"""
    fit_gaussian_gllvm(Y; K, aghq=false, aghq_control=(;), mask=nothing,
                       offset=nothing, hessian=:observed,
                       lambda_constraint=nothing, kwargs...) -> GllvmFit

Fit a Gaussian GLLVM with shared residual SD. Default `aghq=false` retains the
closed-form Gaussian fitter and its existing keywords. In particular `X=nothing`
means **zero mean**; a complete `X[p,n,q]` defines all fixed effects. `β_fixed`
retains the existing fixed-zero coefficient convention.

`aghq=3` requests three nodes per latent axis; `true`/`:auto` selects five nodes
below 20 traits. The ordinary loadings-only block with K≤5 is eligible; additional
random blocks or predictor-informed latent scores retain exact Gaussian/Laplace
with a warning and recorded reason. `aghq=1` follows exact Gaussian/Laplace.
No loading ridge is added. Controls follow the unpenalized frozen-surrogate
outer optimizer used by the Poisson and binomial candidates.

Masks/missing responses and offsets are admitted for the ordinary Gaussian block.
The masked exact marginal is used for baseline fitting; omitted entries are never
included in the target likelihood. Structured masked/offset routes remain separate.
The fit's `integration` records requested/actual method, node count, starting
vectors, controls, convergence, observed caches and input identity. AGHQ convergence
and inference refer to the **frozen-node surrogate**, not its moving-node derivative.
This route does not establish full R↔Julia parity.

**`lambda_constraint`:** fits a confirmatory model in which specific loadings
are held fixed at given values instead of estimated, mirroring R gllvmTMB's
`lambda_constraint = list(unit = M)`. Pass a `p × K` matrix of raw `Λ` values
(`NaN` = free, a number = pinned at that value); this first fits the ordinary
model with nothing pinned, then re-optimises with the requested entries held
fixed. Available for the ordinary Gaussian latent-variable model only: no
phylogenetic or diagonal random-effect terms, and no fixed-effect covariates
(`X`) or predictor-informed latent scores (`X_lv`). Combining
`lambda_constraint` with `aghq`, `mask`, or `offset` is not yet supported;
each of these combinations raises a clear `ArgumentError` rather than
silently fitting the wrong model. The returned fit's `pars.lambda_constraint`
records the normalised pin matrix, which [`loading_profile`](@ref) reads to
determine which entries are free. No cross-package numeric comparison
against R's own `lambda_constraint` fits has been published yet.
"""
function fit_gaussian_gllvm(Y::AbstractMatrix;K::Integer,aghq=false,aghq_control=(;),
        mask=nothing,offset=nothing,hessian=:observed,lambda_constraint=nothing,kwargs...)
    hessian===:observed || throw(ArgumentError("Gaussian integration uses observed curvature"))
    if lambda_constraint!==nothing
        aghq===false || throw(ArgumentError(
            "lambda_constraint does not yet support aghq in Stage 1; use the default aghq=false"))
        (mask===nothing && offset===nothing) || throw(ArgumentError(
            "lambda_constraint does not yet support mask/offset in Stage 1"))
        get(kwargs,:X,nothing)===nothing || throw(ArgumentError(
            "lambda_constraint currently supports X = nothing (zero-mean) fits only"))
        get(kwargs,:X_lv,nothing)===nothing || throw(ArgumentError(
            "lambda_constraint does not yet support X_lv (predictor-informed latent " *
            "scores) in Stage 1"))
        base=fit_gaussian_gllvm(Y;K=K,hessian=hessian,kwargs...)
        return _fit_confirmatory_lambda_constraint(base,Y,lambda_constraint)
    end
    request=_aghq_request(aghq);c=_aghq_controls(aghq_control)
    if request===:off && mask===nothing && offset===nothing
        return _fit_gaussian_gllvm_exact(Y;K=K,kwargs...)
    end
    p,n=size(Y);1<=K<=p || throw(ArgumentError("K must lie in 1:p"))
    structured=get(kwargs,:K_W,0)>0 || get(kwargs,:has_diag,false) ||
        get(kwargs,:K_phy,0)>0 || get(kwargs,:has_phy_unique,false) || get(kwargs,:Σ_phy,nothing)!==nothing
    predictor=get(kwargs,:X_lv,nothing)!==nothing
    k=request in (:off,:auto) ? 5 : request
    reason=request===:off ? :exact_gaussian : k==1 ? :laplace_rule : structured ? :other_random_blocks :
        predictor ? :predictor_latents : K>5 ? :unaffordable_dimension :
        request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    base_controls=deepcopy((;kwargs...));t0=time()
    if structured || predictor
        (mask===nothing && offset===nothing) || throw(ArgumentError("structured Gaussian masks/offsets require a separate model route"))
        base=_fit_gaussian_gllvm_exact(Y;K=K,kwargs...)
        reason===:laplace_rule || request===:off || @warn "AGHQ request retained exact Gaussian/Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,AGHQAdaptation[],nothing,"",NaN)
        return _gaussian_with_integration(base,info)
    end
    X=get(kwargs,:X,nothing);fullX=X===nothing ? zeros(p,n,0) : X
    # Validation uses k=1 so ineligible node requests can retain Laplace.
    checked=aghq_gaussian_problem(Y,K;k=1,X=fullX,mask=mask,offset=offset,mode_gradient_tol=c.mode_gradient_tol)
    qfull=size(checked.data.design,3)
    fixed=_fixed_zero_mask(get(kwargs,:β_fixed,nothing),qfull,"β_fixed")
    X===nothing && get(kwargs,:β_fixed,nothing)!==nothing && !isempty(get(kwargs,:β_fixed,[])) &&
        throw(ArgumentError("β_fixed requires X"))
    data=AGHQGaussianData(checked.data.responses,checked.data.mask,checked.data.offset,checked.data.design)
    # Keep finite supplied offsets for predictions at omitted cells as well.
    if offset!==nothing
        for j in eachindex(data.offset)
            v=offset[j]
            v isa Real && isfinite(v) && (data.offset[j]=v)
        end
    end
    all(t->any(data.mask[t,:]),1:p) || throw(ArgumentError("every trait requires observed Gaussian responses"))
    free=findall(!,fixed);Xfree=data.design[:,:,free]
    # A complete-data fit supplies starts only for the masked/offset baseline.
    warm=copy(data.responses)-data.offset
    for t in 1:p
        obs=findall(data.mask[t,:]);mu=mean(warm[t,obs])
        warm[t,.!data.mask[t,:]].=mu
    end
    _fixed_init_free(get(kwargs,:β_init,nothing),fixed,"β_init")
    warm_kwargs=isempty(free) ? merge((;kwargs...),(X=nothing,β_fixed=nothing,β_init=nothing)) : (;kwargs...)
    base=_fit_gaussian_gllvm_exact(warm;K=K,warm_kwargs...)
    base=_gaussian_update(base,base.pars.θ_packed,base.logLik,base.n_iter,base.converged,base.optim_result,base.cputime,fixed)
    if any(!,data.mask) || offset!==nothing
        nll=_gaussian_data_nll(data,K,fixed)
        start=base.pars.θ_packed
        res=Optim.optimize(nll,start,Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking(order=3)),
            Optim.Options(iterations=get(kwargs,:iterations,500),g_tol=get(kwargs,:g_tol,1e-6),
                x_abstol=get(kwargs,:x_tol,1e-8),f_reltol=get(kwargs,:f_tol,1e-10));autodiff=:forward)
        t=Optim.minimizer(res)
        base=_gaussian_update(base,t,-nll(t),Optim.iterations(res),_fit_verdict(res)[2],res,time()-t0,fixed)
    end
    problem=aghq_gaussian_problem(Y,K;k=reason===:eligible ? k : 1,X=Xfree,
        mask=data.mask,offset=data.offset,mode_gradient_tol=c.mode_gradient_tol)
    if reason!==:eligible
        reason in (:laplace_rule,:exact_gaussian) || @warn "AGHQ request retained exact Gaussian/Laplace" reason=reason requested=request
        cache=problem.adapt(base.pars.θ_packed)
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,cache,data,_aghq_data_digest(data),
            maximum(d.gradient_max for d in problem.mode_diagnostics(base.pars.θ_packed)))
        return _gaussian_with_integration(base,info)
    end
    start=copy(base.pars.θ_packed);starts=[start]
    if c.multistart
        alt=copy(start);alt[length(free)+2:end].=.3;push!(starts,alt)
    end
    raw=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    result=merge(raw,(starts=deepcopy(starts),))
    if !result.usable
        @warn "AGHQ failed; retained exact Gaussian/Laplace" requested=request
        lap=aghq_gaussian_problem(Y,K;k=1,X=Xfree,mask=data.mask,offset=data.offset)
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            lap.adapt(base.pars.θ_packed),data,_aghq_data_digest(data),NaN)
        return _gaussian_with_integration(base,info)
    end
    sel=result.selected;t=sel.parameters
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),sel.stop_reason,false,Inf,c,base_controls,result,
        deepcopy(sel.adaptation),data,_aghq_data_digest(data),maximum(d.gradient_max for d in problem.mode_diagnostics(t)))
    return _gaussian_with_integration(_gaussian_update(base,t,-sel.objective,sel.passes,sel.converged,result,time()-t0,fixed),info)
end
_gaussian_with_integration(f,i)=GllvmFit(f.model,f.pars,f.logLik,f.n_iter,f.converged,f.optim_result,f.cputime,i)
function _gaussian_update(base,t,ll,it,conv,res,elapsed,fixed)
    q=count(!,fixed);p,K=base.model.p,base.model.K
    pars=merge(base.pars,(β=_expand_fixed_zero(t[1:q],fixed),β_fixed=copy(fixed),σ_eps=exp(t[q+1]),
        Λ=unpack_lambda(t[q+2:end],p,K),θ_packed=copy(t)))
    return GllvmFit(base.model,pars,ll,it,conv,res,elapsed)
end
_has_gaussian_record(f)=f isa GllvmFit && f.integration!==nothing && f.integration.data isa AGHQGaussianData
_is_gaussian_aghq(f)=_has_gaussian_record(f) && f.integration.actual===:aghq
function _gaussian_data_nll(data,K,fixed)
    p,n=size(data.responses);X=data.design[:,:,findall(!,fixed)];q=size(X,3)
    rows=[findall(data.mask[:,s]) for s in 1:n]
    return function(t)
        L=unpack_lambda(t[q+2:end],p,K);variance=exp(2t[q+1]);value=zero(eltype(t))
        for s in 1:n
            obs=rows[s];isempty(obs) && continue
            F=cholesky(Symmetric(L[obs,:]*L[obs,:]'+variance*I))
            e=data.responses[obs,s]-X[obs,s,:]*t[1:q]-data.offset[obs,s]
            value+=(length(obs)*log(2pi)+logdet(F)+dot(e,F\e))/2
        end
        return value
    end
end
function _gaussian_record_problem(f,Y;X=nothing,mask=nothing,offset=nothing,require_identity=false)
    size(Y,1)==f.model.p || throw(DimensionMismatch("responses must match fitted trait count"))
    i=f.integration;d=i.data;same_shape=size(Y)==size(d.responses)
    xx=X===nothing ? (same_shape ? d.design : size(d.design,3)==0 ? zeros(size(Y)...,0) :
        throw(ArgumentError("new-data Gaussian prediction requires X"))) : X
    size(xx,3)==size(d.design,3) || throw(DimensionMismatch("X coefficient count must match fit"))
    m=mask===nothing && same_shape ? d.mask : mask
    o=offset===nothing && same_shape ? d.offset : offset
    full=aghq_gaussian_problem(Y,f.model.K;k=1,X=xx,mask=m,offset=o)
    dd=AGHQGaussianData(full.data.responses,full.data.mask,full.data.offset,full.data.design)
    same=_aghq_data_digest(dd)==i.input_digest
    require_identity && !same && throw(ArgumentError("Gaussian inference requires original observed Y, X, mask and offset"))
    !same && X===nothing && size(d.design,3)>0 && throw(ArgumentError("new-data Gaussian prediction requires explicit X"))
    !same && offset===nothing && any(!iszero,d.offset) && throw(ArgumentError("new-data Gaussian prediction requires explicit offset"))
    free=findall(!,f.pars.β_fixed)
    q=aghq_gaussian_problem(Y,f.model.K;k=i.k,X=full.data.design[:,:,free],mask=full.data.mask,offset=full.data.offset,
        mode_gradient_tol=i.controls.mode_gradient_tol)
    return q,same
end
function _gaussian_record_predict(f,Y;X=nothing,mask=nothing,offset=nothing,component=:total,rotate=false,scores=false)
    q,same=_gaussian_record_problem(f,Y;X=X,mask=mask,offset=offset)
    cache=same ? f.integration.caches : q.adapt(f.pars.θ_packed)
    z=permutedims(reduce(hcat,(c.mode for c in cache)))
    component===:mean && (z.=0)
    scores && return rotate ? z*_svd_rotation(f.pars.Λ) : z
    qfree=size(q.data.design,3);beta=f.pars.θ_packed[1:qfree];p,n=size(Y)
    mean=[dot(q.data.design[t,s,:],beta) for t in 1:p,s in 1:n]
    return mean+f.pars.Λ*z'+_aghq_prediction_offset(f,Y,offset)
end
"""
    simulate(fit::GllvmFit, n; rng=Random.default_rng(), X=nothing, offset=nothing)

Draw ordinary Gaussian observations with the fitted covariance. For fits retaining
integration data, use the original design and offsets at the original site count;
otherwise supply those inputs explicitly. Structured Gaussian simulation uses the
separate model-specific simulation interfaces.
"""
function simulate(f::GllvmFit,n::Integer;rng=Random.default_rng(),X=nothing,offset=nothing)
    n>0 || throw(ArgumentError("n must be positive"))
    (f.model.K_W==0 && !f.model.has_diag && f.model.K_phy==0 && !f.model.has_phy_unique && !_has_lv_predictor(f)) ||
        throw(ArgumentError("this Gaussian simulator requires an ordinary loadings-only block"))
    p,K=f.model.p,f.model.K;q=length(f.pars.β)
    d=_has_gaussian_record(f) ? f.integration.data : nothing
    xx=X===nothing ? (d!==nothing && size(d.responses,2)==n ? d.design : q==0 ? zeros(p,n,0) :
        throw(ArgumentError("Gaussian simulation requires X at the requested site count"))) : X
    off=offset===nothing ? (d!==nothing && size(d.responses,2)==n ? d.offset :
        d!==nothing && any(!iszero,d.offset) ? throw(ArgumentError("simulation at new site count requires offset")) : zeros(p,n)) : offset
    size(xx)==(p,n,q) && all(isfinite,xx) || throw(ArgumentError("simulation X must be finite p by n by q"))
    size(off)==(p,n) && all(isfinite,off) || throw(ArgumentError("simulation offset must be finite p by n"))
    mu=[dot(xx[t,s,:],f.pars.β) for t in 1:p,s in 1:n]
    return mu+off+f.pars.Λ*randn(rng,K,n)+f.pars.σ_eps*randn(rng,p,n)
end

function _gaussian_record_ci(f,Y;X=nothing,mask=nothing,offset=nothing,Σ_phy=nothing,objective=:fit)
    Σ_phy===nothing || throw(ArgumentError("ordinary Gaussian integration has no phylogenetic covariance"))
    expected=_is_gaussian_aghq(f) ? :aghq : :laplace
    objective in (:fit,expected) || throw(ArgumentError("Gaussian inference must use the fitted integration objective"))
    q,_=_gaussian_record_problem(f,Y;X=X,mask=mask,offset=offset,require_identity=true)
    i=f.integration;t=copy(f.pars.θ_packed)
    nll=expected===:aghq ? (v->q.objective(v,i.caches)) : _gaussian_data_nll(i.data,f.model.K,f.pars.β_fixed)
    names,kinds=_confint_all_term_names(f)
    refit=function(Yb)
        fb=_gaussian_record_refit(f,Yb)
        return fb===nothing ? nothing : copy(fb.pars.θ_packed)
    end
    return _FamilyCI(t,nll,names,[k===:log_sd ? :log : :linear for k in kinds],
        rng->simulate(f,size(Y,2);rng=rng),refit)
end
function _gaussian_record_confint(f,Y;method=:wald,level=.95,parm=nothing,X=nothing,mask=nothing,offset=nothing,
        Σ_phy=nothing,objective=:fit,n_boot::Integer=200,seed::Integer=0,parallel::Bool=false,
        profile_iterations::Integer=200,profile_g_tol::Real=1e-4,profile_max_expand::Integer=20,profile_max_bisect::Integer=30)
    0<level<1 || throw(ArgumentError("level must be between zero and one"))
    n_boot>0 || throw(ArgumentError("n_boot must be positive"))
    ad=_gaussian_record_ci(f,Y;X=X,mask=mask,offset=offset,Σ_phy=Σ_phy,objective=objective)
    sel=_confint_select_indices(parm,ad.names)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))
    result=if method===:wald
        _family_wald(ad,sel,level;hessian=ForwardDiff.hessian(ad.nll,ad.θ))
    elseif method===:profile
        _family_profile(ad,sel,level;profile_iterations=profile_iterations,profile_g_tol=profile_g_tol,
            profile_max_expand=profile_max_expand,profile_max_bisect=profile_max_bisect)
    elseif method===:bootstrap
        _family_bootstrap(ad,sel,level,n_boot,seed,parallel;retain_replicates=true)
    else
        throw(ArgumentError("method must be :wald, :profile or :bootstrap"))
    end
    return merge(result,(objective=f.integration.actual,gradient_kind=_is_gaussian_aghq(f) ? :frozen_surrogate : :exact_marginal,))
end
function _gaussian_record_vcov(f,Y;kwargs...)
    ad=_gaussian_record_ci(f,Y;kwargs...)
    H=ForwardDiff.hessian(ad.nll,ad.θ);m=length(ad.θ)
    F=try
        cholesky(Symmetric((H+H')/2))
    catch e
        e isa InterruptException && rethrow()
        nothing
    end
    return F===nothing ? fill(NaN,m,m) : F\Matrix{Float64}(I,m,m)
end

function _gaussian_record_refit(f,Yb)
    i=f.integration
    controls=merge(deepcopy(i.base_controls),(K=f.model.K,aghq=i.requested===:off ? false : i.requested,aghq_control=i.controls,
        X=copy(i.data.design),β_fixed=copy(f.pars.β_fixed),mask=copy(i.data.mask),offset=copy(i.data.offset)))
    fb=try
        fit_gaussian_gllvm(Yb;controls...)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    return fb.converged && fb.integration!==nothing && fb.integration.actual===i.actual ? fb : nothing
end
function _gaussian_record_bootstrap_derived(f,fn;Y,X,Σ_phy,n_sites,n_boot,level,seed)
    n_sites===nothing || n_sites==size(Y,2) || throw(ArgumentError("derived bootstrap preserves the fitted site count"))
    _gaussian_record_ci(f,Y;X=X,Σ_phy=Σ_phy) # bind observed inputs before any draw
    call=function(fb)
        try
            return Float64(fn(fb))
        catch e
            e isa MethodError || rethrow()
            return Float64(fn(fb.pars.θ_packed))
        end
    end
    est=call(f);isfinite(est) || throw(ArgumentError("derived estimate must be finite"))
    reps=fill(NaN,n_boot);ok=falses(n_boot)
    for b in 1:n_boot
        fb=_gaussian_record_refit(f,simulate(f,size(Y,2);rng=MersenneTwister(seed+b)))
        fb===nothing && continue
        ok[b]=true
        try
            value=call(fb);isfinite(value) && (reps[b]=value)
        catch e
            e isa InterruptException && rethrow()
        end
    end
    valid=filter(isfinite,reps);a=(1-level)/2
    lo,hi=length(valid)>=10 ? (quantile(valid,a),quantile(valid,1-a)) : (NaN,NaN)
    return (estimate=est,lower=lo,upper=hi,n_converged=sum(ok),n_valid=length(valid),replicates=reps,
        converged=ok,objective=f.integration.actual)
end
