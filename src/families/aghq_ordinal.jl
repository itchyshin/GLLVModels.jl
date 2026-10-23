# Ordinal (cumulative logit/probit) AGHQ: problem builder and fitter, mirroring
# aghq_nb2.jl. The model is `fit_ordinal_gllvm_pertrait`'s: per-trait intercept beta_t,
# per-trait ordered cutpoints (tau_1 = 0 fixed, C_t - 2 free log-increments), loadings-only
# latent block with a standard-normal prior. gllvmTMB's `ordinal_probit()` (family id 14)
# and `ordinal_logit()` (id 20) are the same model. Uses the Laplace per-trait mode search
# only as a proposal (it is not modified here).

# Per-trait cutpoint vectors from the packed log-increments; AD-friendly (the element type
# follows `psi`), unlike `_unpack_cutpoints_pertrait`, which builds a Float64 matrix.
function _ord_cut_vectors(psi::AbstractVector, C::AbstractVector{<:Integer})
    T=float(eltype(psi));out=Vector{Vector{T}}(undef,length(C));pos=1
    for t in eachindex(C)
        m=C[t]-1;tau=Vector{T}(undef,m);tau[1]=zero(T)
        for c in 2:m
            tau[c]=tau[c-1]+exp(psi[pos]);pos+=1
        end
        out[t]=tau
    end
    return out
end

# Inverse of the packing in `_unpack_cutpoints_pertrait`: log-increments from the padded
# cutpoint matrix.
function _ord_psi_from_tau(tau::AbstractMatrix, C::AbstractVector{<:Integer})
    psi=Float64[]
    for t in eachindex(C),c in 2:(C[t]-1)
        push!(psi,log(tau[t,c]-tau[t,c-1]))
    end
    return psi
end

"""
    aghq_ordinal_problem(Y, K; k, link=LogitLink(), C=nothing, mask=nothing,
        mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)

Internal loadings-only cumulative-link ordinal AGHQ problem (logit or probit). Parameters
are the per-trait intercepts, `pack_lambda` loadings, then the free cutpoint
log-increments: trait `t` has `C[t] - 2` of them (the first cutpoint is fixed at 0), laid
out trait by trait as in [`fit_ordinal_gllvm_pertrait`](@ref). `C` defaults to the largest
observed level of each trait. Same contract as [`aghq_poisson_problem`](@ref):
`adapt(theta)`, `objective(theta, caches)`, `mode_diagnostics(theta)`, `grid`, `nparams`;
the objective differentiates the frozen-cache surrogate only; modes are checked against the
actual joint gradient and observed Hessian before any cache is built. The normalized joint
is the log category probability (floored at `1e-12` as in the Laplace code) plus a
standard-normal latent prior, no loading penalty.
"""
function aghq_ordinal_problem(Y::AbstractMatrix, K; k, link::Link=LogitLink(), C=nothing, mask=nothing,
        mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)
    p,n=size(Y)
    p>0 && n>0 || throw(ArgumentError("AGHQ ordinal needs nonempty responses"))
    _check_ordinal_link(link)
    K isa Integer && !(K isa Bool) && 1<=K<=p ||
        throw(ArgumentError("AGHQ ordinal K must be an integer in 1:p"))
    k isa Integer && !(k isa Bool) && k>0 ||
        throw(ArgumentError("AGHQ ordinal k must be a positive integer"))
    _aghq_kd_bound(K,k)
    mode_maxiter isa Integer && !(mode_maxiter isa Bool) && mode_maxiter>0 ||
        throw(ArgumentError("AGHQ mode_maxiter must be a positive integer"))
    for (name,value) in ((:mode_tol,mode_tol),(:mode_gradient_tol,mode_gradient_tol))
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    if mask !== nothing
        size(mask)==size(Y) || throw(DimensionMismatch("AGHQ mask must match Y"))
        eltype(mask)<:Bool || throw(ArgumentError("AGHQ mask must contain Bool values"))
    end
    observed=falses(p,n);levels=ones(Int,p,n)
    for s in 1:n,t in 1:p
        (mask===nothing || mask[t,s]) && !ismissing(Y[t,s]) || continue
        y=Y[t,s]
        y isa Real && isfinite(y) && isinteger(y) && y>=1 && y<=2.0^31 ||
            throw(ArgumentError("AGHQ ordinal observed response ($t,$s) must be an integer level >= 1"))
        observed[t,s]=true;levels[t,s]=Int(y)
    end
    Cv=if C===nothing
        [maximum(view(levels,t,:)[view(observed,t,:)];init=0) for t in 1:p]
    else
        length(C)==p && all(c->c isa Integer,C) || throw(ArgumentError("AGHQ ordinal C must be p integers"))
        collect(Int,C)
    end
    all(>=(2),Cv) || throw(ArgumentError("AGHQ ordinal needs >= 2 observed categories for every trait; got $Cv"))
    for s in 1:n,t in 1:p
        observed[t,s] && levels[t,s]>Cv[t] &&
            throw(ArgumentError("AGHQ ordinal response ($t,$s) exceeds C[$t] = $(Cv[t])"))
    end
    grid=aghq_grid(K,k);rr=rr_theta_len(p,K);ncut=sum(Cv.-2);nparams=p+rr+ncut
    function unpack(theta)
        length(theta)==nparams || throw(DimensionMismatch("AGHQ ordinal parameter length must be $nparams"))
        all(isfinite,theta) || throw(ArgumentError("AGHQ ordinal parameters must be finite"))
        return view(theta,1:p),unpack_lambda(view(theta,p+1:p+rr),p,K),_ord_cut_vectors(view(theta,p+rr+1:nparams),Cv)
    end
    function joint(z,beta,loading,tau,s)
        value=-sum(abs2,z)/2-K*log(2pi)/2
        for t in 1:p
            observed[t,s] || continue
            eta=_clamp_eta(beta[t]+dot(view(loading,t,:),z))
            value+=log(max(_ord_prob(levels[t,s],eta,tau[t],link),1e-12))
        end
        return value
    end
    function checked_modes(theta)
        beta,loading,tau=unpack(theta)
        taumat=fill(NaN,p,maximum(Cv)-1)
        for t in 1:p,c in eachindex(tau[t]);taumat[t,c]=Float64(tau[t][c]);end
        caches=AGHQAdaptation[];diagnostics=NamedTuple[]
        for s in 1:n
            z=_ordinal_laplace_mode_pertrait(view(levels,:,s),loading,beta,taumat,Cv,link;
                mask=view(observed,:,s),maxiter=mode_maxiter,tol=mode_tol)
            logjoint=v->joint(v,beta,loading,tau,s)
            gradient=ForwardDiff.gradient(logjoint,z)
            g=maximum(abs,gradient)
            all(isfinite,z) && isfinite(logjoint(z)) && isfinite(g) && g<=mode_gradient_tol ||
                error("AGHQ ordinal conditional mode failed at site $s (gradient_max=$g, tolerance=$mode_gradient_tol)")
            H=-ForwardDiff.hessian(logjoint,z)
            all(isfinite,H) && isposdef(Symmetric(H)) ||
                error("AGHQ ordinal observed curvature is invalid at site $s")
            cache=aghq_adaptation(z,H)
            push!(caches,cache)
            push!(diagnostics,(site=s,gradient_max=g,minimum_eigenvalue=cache.minimum_eigenvalue,
                curvature_repaired=cache.curvature_repaired))
        end
        return caches,diagnostics
    end
    function objective(theta,caches)
        beta,loading,tau=unpack(theta)
        length(caches)==n || throw(DimensionMismatch("AGHQ ordinal needs one cache per site"))
        return -sum(aghq_frozen_logintegral(z->joint(z,beta,loading,tau,s),caches[s],grid) for s in 1:n)
    end
    return (adapt=theta->first(checked_modes(theta)),objective=objective,
        mode_diagnostics=theta->last(checked_modes(theta)),grid=grid,nparams=nparams,
        data=(responses=Float64.(levels),mask=copy(observed),C=copy(Cv),link=link),C=Cv)
end

"""
    OrdinalPerTraitAGHQFit

Result of [`fit_ordinal_gllvm_pertrait_aghq`](@ref): the fields of `OrdinalPerTraitFit`
plus `theta_packed` (`β`, packed `Λ`, cutpoint log-increments) and `integration`
(`AGHQFitInfo`) recording the actual route (`:aghq`, or `:laplace` with a reason when the
request was declined). `loglik` is the AGHQ log likelihood when
`integration.actual === :aghq`. Point estimates only: `predict`/`confint`/`simulate` have
no AGHQ route for this type.
"""
struct OrdinalPerTraitAGHQFit
    Λ::Matrix{Float64}
    β::Vector{Float64}
    τ::Matrix{Float64}
    C::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
    hessian::Symbol
    integration::AGHQFitInfo
end
function Base.show(io::IO, f::OrdinalPerTraitAGHQFit)
    p, K = size(f.Λ)
    print(io, "OrdinalPerTraitAGHQFit(p=", p, ", K=", K, ", C=", f.C,
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7), ", route=", f.integration.actual,
          f.converged ? "" : ", NOT CONVERGED", ")")
end
_loadings(fit::OrdinalPerTraitAGHQFit) = fit.Λ
_loglik(fit::OrdinalPerTraitAGHQFit)   = fit.loglik

"""
    fit_ordinal_gllvm_pertrait_aghq(Y; K, link=LogitLink(), aghq=:auto, aghq_control=(;), kwargs...)

Cumulative-link ordinal GLLVM with per-trait intercepts and cutpoints (the gllvmTMB
`ordinal_probit()` / `ordinal_logit()` model; pass `link = ProbitLink()` for the former)
fitted with adaptive Gauss-Hermite quadrature. Also reached as
`fit_gllvm(Y; family=Ordinal(), K, aghq=...)`. `aghq` follows [`fit_poisson_gllvm`](@ref)
(`false` means call `fit_ordinal_gllvm_pertrait` directly; `1` is Laplace; a node count;
`true`/`:auto`) except that `:auto` uses **9 nodes**, the starting rung gllvmTMB's node
ladder gives ordinal families, and is declined at 20 traits or more. Ineligible requests
(`K` above 5) keep the Laplace fit with a warning and a recorded reason. The warm start is
the Laplace fit; convergence means the frozen-node gradient rule, not re-adapted
stationarity. `aghq_control` is as for Poisson. `hessian` must be `:observed`. There is no
`offset` for ordinal fits.
"""
function fit_ordinal_gllvm_pertrait_aghq(Y::AbstractMatrix{<:Integer};K::Integer,aghq=:auto,aghq_control=(;),kwargs...)
    request=_aghq_request(aghq);request===:off && throw(ArgumentError("fit_ordinal_gllvm_pertrait_aghq needs aghq != false; call fit_ordinal_gllvm_pertrait"))
    c=_aghq_controls(aghq_control)
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 9 : request
    link=get(kwargs,:link,LogitLink())
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    kw=Base.structdiff((;kwargs...),(hessian=nothing,))
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        K>5 ? :unaffordable_dimension : request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=fit_ordinal_gllvm_pertrait(Y;K=K,kw...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _ordinal_with_integration(base,info)
    end
    problem=aghq_ordinal_problem(Y,K;k=k,link=link,mask=get(kwargs,:mask,nothing),
        mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge(kw,(newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=fit_ordinal_gllvm_pertrait(Y;K=K,base_kwargs...)
    rr=rr_theta_len(p,K)
    theta=vcat(base.β,pack_lambda(base.Λ),_ord_psi_from_tau(base.τ,base.C));starts=[theta]
    if c.multistart
        alt=copy(theta);alt[p+1:p+rr].=.3;push!(starts,alt)
    end
    result=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _ordinal_with_integration(base,info)
    end
    selected=result.selected;t=selected.parameters
    data=AGHQOrdinalData(problem.data.responses,problem.data.mask,problem.data.C,problem.data.link isa ProbitLink)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    return OrdinalPerTraitAGHQFit(unpack_lambda(t[p+1:p+rr],p,K),copy(t[1:p]),
        _unpack_cutpoints_pertrait(t[p+rr+1:end],problem.C),copy(problem.C),link,-selected.objective,
        selected.converged,selected.passes,copy(t),:observed,info)
end
_ordinal_with_integration(f::OrdinalPerTraitFit,i)=OrdinalPerTraitAGHQFit(f.Λ,f.β,f.τ,f.C,f.link,
    f.loglik,f.converged,f.iterations,vcat(f.β,pack_lambda(f.Λ),_ord_psi_from_tau(f.τ,f.C)),:observed,i)
