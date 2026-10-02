# NB2 (negative binomial, log link) AGHQ: problem builder and fitter, mirroring
# aghq_poisson.jl / aghq_poisson_fit.jl. The dispersion is per group of traits
# (`group`); one group per trait is gllvmTMB's `log_phi_nbinom2` (length n_traits).
# Uses the grouped NB2 mode search only as a proposal (it is not modified here).

"""
    aghq_nb2_problem(Y, K; k, group=1:p, mask=nothing, offset=nothing,
        mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)

Internal loadings-only, log-link NB2 AGHQ problem. Parameters are intercepts,
`pack_lambda` loadings, then `log r_g` for each dispersion group (groups are
relabelled to `1..G` in sorted order). Same contract as
[`aghq_poisson_problem`](@ref): `adapt(theta)`, `objective(theta, caches)`,
`mode_diagnostics(theta)`, `grid`, `nparams`; the objective differentiates the
frozen-cache surrogate only; modes are checked against the actual joint gradient
and observed Hessian before any cache is built. The normalized joint is the
full NB2 log density plus a standard-normal latent prior, no loading penalty.
"""
function aghq_nb2_problem(Y::AbstractMatrix, K; k, group=nothing, mask=nothing, offset=nothing,
        mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)
    p,n=size(Y)
    p>0 && n>0 || throw(ArgumentError("AGHQ NB2 needs nonempty responses"))
    K isa Integer && !(K isa Bool) && 1<=K<=p ||
        throw(ArgumentError("AGHQ NB2 K must be an integer in 1:p"))
    k isa Integer && !(k isa Bool) && k>0 ||
        throw(ArgumentError("AGHQ NB2 k must be a positive integer"))
    _aghq_kd_bound(K,k)
    mode_maxiter isa Integer && !(mode_maxiter isa Bool) && mode_maxiter>0 ||
        throw(ArgumentError("AGHQ mode_maxiter must be a positive integer"))
    for (name,value) in ((:mode_tol,mode_tol),(:mode_gradient_tol,mode_gradient_tol))
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    grp=group===nothing ? collect(1:p) : collect(Int,group)
    length(grp)==p || throw(ArgumentError("AGHQ NB2 group must have length p"))
    labels=sort(unique(grp));G=length(labels)
    gidx=[findfirst(==(grp[t]),labels) for t in 1:p]
    if mask !== nothing
        size(mask)==size(Y) || throw(DimensionMismatch("AGHQ mask must match Y"))
        eltype(mask)<:Bool || throw(ArgumentError("AGHQ mask must contain Bool values"))
    end
    offset===nothing || size(offset)==size(Y) ||
        throw(DimensionMismatch("AGHQ offset must match Y"))
    observed=falses(p,n);counts=zeros(p,n);off=zeros(p,n)
    for s in 1:n,t in 1:p
        (mask===nothing || mask[t,s]) && !ismissing(Y[t,s]) || continue
        y=Y[t,s]
        y isa Real && isfinite(y) && y>=0 && isinteger(y) && y<=2.0^53 ||
            throw(ArgumentError("AGHQ NB2 observed response ($t,$s) must be a finite nonnegative count"))
        o=offset===nothing ? 0.0 : offset[t,s]
        o isa Real && isfinite(o) && isfinite(Float64(o)) ||
            throw(ArgumentError("AGHQ NB2 observed offset ($t,$s) must be finite"))
        observed[t,s]=true;counts[t,s]=Float64(y);off[t,s]=Float64(o)
    end
    grid=aghq_grid(K,k);rr=rr_theta_len(p,K);nparams=p+rr+G
    function unpack(theta)
        length(theta)==nparams || throw(DimensionMismatch("AGHQ NB2 parameter length must be $nparams"))
        all(isfinite,theta) || throw(ArgumentError("AGHQ NB2 parameters must be finite"))
        return view(theta,1:p),unpack_lambda(view(theta,p+1:p+rr),p,K),view(theta,p+rr+1:nparams)
    end
    function joint(z,beta,loading,logr,s)
        value=-sum(abs2,z)/2-K*log(2pi)/2
        for t in 1:p
            observed[t,s] || continue
            eta=beta[t]+off[t,s]+dot(view(loading,t,:),z)
            value+=_nb2_logpdf_mean(exp(eta),exp(logr[gidx[t]]),Int(counts[t,s]))
        end
        return value
    end
    function checked_modes(theta)
        beta,loading,logr=unpack(theta)
        fams=[NegativeBinomial(exp(Float64(logr[gidx[t]])),0.5) for t in 1:p]
        caches=AGHQAdaptation[];diagnostics=NamedTuple[]
        for s in 1:n
            z=_grouped_laplace_mode(fams,view(counts,:,s),ones(p),loading,beta,LogLink();
                mask=view(observed,:,s),offset=view(off,:,s),maxiter=mode_maxiter,tol=mode_tol)
            logjoint=v->joint(v,beta,loading,logr,s)
            gradient=ForwardDiff.gradient(logjoint,z)
            g=maximum(abs,gradient)
            all(isfinite,z) && isfinite(logjoint(z)) && isfinite(g) && g<=mode_gradient_tol ||
                error("AGHQ NB2 conditional mode failed at site $s (gradient_max=$g, tolerance=$mode_gradient_tol)")
            H=-ForwardDiff.hessian(logjoint,z)
            all(isfinite,H) && isposdef(Symmetric(H)) ||
                error("AGHQ NB2 observed curvature is invalid at site $s")
            cache=aghq_adaptation(z,H)
            push!(caches,cache)
            push!(diagnostics,(site=s,gradient_max=g,minimum_eigenvalue=cache.minimum_eigenvalue,
                curvature_repaired=cache.curvature_repaired))
        end
        return caches,diagnostics
    end
    function objective(theta,caches)
        beta,loading,logr=unpack(theta)
        length(caches)==n || throw(DimensionMismatch("AGHQ NB2 needs one cache per site"))
        return -sum(aghq_frozen_logintegral(z->joint(z,beta,loading,logr,s),caches[s],grid) for s in 1:n)
    end
    return (adapt=theta->first(checked_modes(theta)),objective=objective,
        mode_diagnostics=theta->last(checked_modes(theta)),grid=grid,nparams=nparams,
        data=(responses=copy(counts),mask=copy(observed),offset=copy(off),group=copy(gidx)),G=G)
end

"""
    NBGroupedAGHQFit

Result of [`fit_nb_gllvm_grouped_aghq`](@ref): the fields of `NBGroupedFit` plus
`theta_packed` (`β`, packed `Λ`, `log r_group`) and `integration`
([`AGHQFitInfo`](@ref)) recording the actual route (`:aghq`, or `:laplace`
with a reason when the request was declined). `loglik` is the AGHQ log
likelihood when `integration.actual === :aghq`. Point estimates only:
`predict`/`confint`/`simulate` have no AGHQ route for this type.
"""
struct NBGroupedAGHQFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    r_group::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
    hessian::Symbol
    integration::AGHQFitInfo
end
function Base.show(io::IO, f::NBGroupedAGHQFit)
    p, K = size(f.Λ)
    print(io, "NBGroupedAGHQFit(p=", p, ", K=", K, ", G=", length(f.r_group),
          ", r_group=", round.(f.r_group; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7), ", route=", f.integration.actual,
          f.converged ? "" : ", NOT CONVERGED", ")")
end
_loadings(fit::NBGroupedAGHQFit) = fit.Λ
_loglik(fit::NBGroupedAGHQFit)   = fit.loglik
_nparams(fit::NBGroupedAGHQFit) = size(fit.Λ,1) + rr_theta_len(size(fit.Λ)...) + length(fit.r_group)

"""
    fit_nb_gllvm_grouped_aghq(Y; K, group=1:p, aghq=:auto, aghq_control=(;), kwargs...)

NB2 (log link) GLLVM with per-group dispersion (default one `r` per trait, the
gllvmTMB `nbinom2` model) fitted with adaptive Gauss-Hermite quadrature. Also
reached as `fit_gllvm(Y; family=NegativeBinomial(), K, disp_group=..., aghq=...)`.
`aghq` follows [`fit_poisson_gllvm`](@ref): `false` (use `fit_nb_gllvm_grouped`
directly), `1` (Laplace), a node count, or `true`/`:auto` (5 nodes, declined at 20
traits or more). Ineligible requests (`K` above 5, non-log link) keep the Laplace
fit with a warning and a recorded reason. The warm start is the Laplace fit;
convergence means the frozen-node gradient rule, not re-adapted stationarity.
`aghq_control` is as for Poisson. `hessian` must be `:observed`.
"""
function fit_nb_gllvm_grouped_aghq(Y::AbstractMatrix;K::Integer,group=nothing,aghq=:auto,aghq_control=(;),kwargs...)
    request=_aghq_request(aghq);request===:off && throw(ArgumentError("fit_nb_gllvm_grouped_aghq needs aghq != false; call fit_nb_gllvm_grouped"))
    c=_aghq_controls(aghq_control)
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 5 : request
    grp=group===nothing ? collect(1:p) : collect(Int,group)
    link=get(kwargs,:link,LogLink())
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    kw=Base.structdiff((;kwargs...),(hessian=nothing,))
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        !(link isa LogLink) ? :unsupported_link : K>5 ? :unaffordable_dimension :
        request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=fit_nb_gllvm_grouped(Y;K=K,group=grp,kw...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _nb_grouped_with_integration(base,info)
    end
    problem=aghq_nb2_problem(Y,K;k=k,group=grp,mask=get(kwargs,:mask,nothing),offset=get(kwargs,:offset,nothing),
        mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge(kw,(mask=problem.data.mask,
        offset=get(kwargs,:offset,nothing)===nothing ? nothing : problem.data.offset,
        newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=fit_nb_gllvm_grouped(Int.(problem.data.responses);K=K,group=grp,base_kwargs...)
    theta=vcat(base.β,pack_lambda(base.Λ),log.(base.r_group));starts=[theta]
    if c.multistart
        alt=copy(theta);alt[p+1:p+rr_theta_len(p,K)].=.3;push!(starts,alt)
    end
    result=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _nb_grouped_with_integration(base,info)
    end
    selected=result.selected;t=selected.parameters;rr=rr_theta_len(p,K)
    prediction_offset=copy(problem.data.offset)
    raw_offset=get(kwargs,:offset,nothing)
    if raw_offset!==nothing
        for j in eachindex(prediction_offset)
            v=raw_offset[j]
            if v isa Real && isfinite(v);prediction_offset[j]=v;end
        end
    end
    data=AGHQNB2Data(problem.data.responses,problem.data.mask,prediction_offset,problem.data.group)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    return NBGroupedAGHQFit(copy(t[1:p]),unpack_lambda(t[p+1:p+rr],p,K),exp.(t[p+rr+1:end]),
        problem.data.group,link,-selected.objective,selected.converged,selected.passes,copy(t),:observed,info)
end
_nb_grouped_with_integration(f::NBGroupedFit,i)=NBGroupedAGHQFit(f.β,f.Λ,f.r_group,f.group,f.link,
    f.loglik,f.converged,f.iterations,vcat(f.β,pack_lambda(f.Λ),log.(f.r_group)),f.hessian,i)
