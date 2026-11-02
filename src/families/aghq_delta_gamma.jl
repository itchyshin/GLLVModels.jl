# Delta-Gamma (two-part: Bernoulli occurrence x positive Gamma, logit / log links) AGHQ:
# problem builder and fitter, mirroring aghq_tweedie.jl. The model is the one
# `fit_delta_gamma_gllvm` fits and the one gllvmTMB's `delta_gamma()` fits (gllvmTMB.cpp,
# fid 13): ONE latent z per site drives both parts. Under `predictor = :shared` (the
# gllvmTMB model) a single linear predictor eta = beta + Lambda z gives both the occurrence
# logit and the Gamma log-mean, with one shape alpha_t per trait (R's phi_t is the CV,
# alpha_t = 1/phi_t^2). Under `predictor = :separate` (the Laplace fitter's default) the
# occurrence part has its own intercept and no loading, the Gamma part has beta_c and Lambda.
# The Laplace two-part mode search is used only as a proposal (it is not modified here).

# Log-density of one cell at the two linear predictors, generic in the number type.
@inline function _delta_gamma_logf(y::Float64, ηz, ηc, α)
    y > 0 || return -log1p(exp(ηz))                    # log(1 - pi), pi = logistic(eta_z)
    return -log1p(exp(-ηz)) + α * log(α) - loggamma(α) - α * ηc +
           (α - 1) * log(y) - α * y * exp(-ηc)         # log pi + Gamma(shape alpha, mean exp(eta_c))
end

"""
    aghq_delta_gamma_problem(Y, K; k, predictor=:shared, disp_group=:species, offset=nothing,
        mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)

Internal loadings-only Delta-Gamma AGHQ problem (`Y` is p x n, `0` for absence, positive
reals otherwise). Parameters are, for `predictor = :shared`, the intercepts `beta` (p) and
the `pack_lambda` loadings, and for `predictor = :separate` the occurrence intercepts, the
positive-part intercepts and the loadings; then `log alpha` (p values for
`disp_group = :species`, one for `:shared`). Under `:shared` one predictor and one set of
loadings drive both parts (gllvmTMB's `delta_gamma()`); under `:separate` the occurrence
part carries no loading. `offset` (p x n) is added to the Gamma predictor, and to the
occurrence predictor too under `:shared`, as in [`fit_delta_gamma_gllvm`](@ref). Same
contract as [`aghq_poisson_problem`](@ref): `adapt(theta)`, `objective(theta, caches)`,
`mode_diagnostics(theta)`, `grid`, `nparams`; the objective differentiates the frozen-cache
surrogate only; modes are checked against the actual joint gradient and observed Hessian
before any cache is built. The normalized joint is the full two-part log density plus a
standard-normal latent prior, no loading penalty.
"""
function aghq_delta_gamma_problem(Y::AbstractMatrix, K; k, predictor::Symbol=:shared,
        disp_group::Symbol=:species, offset=nothing, mode_maxiter=100, mode_tol=1e-9,
        mode_gradient_tol=1e-7)
    p,n=size(Y)
    p>0 && n>0 || throw(ArgumentError("AGHQ Delta-Gamma needs nonempty responses"))
    K isa Integer && !(K isa Bool) && 1<=K<=p ||
        throw(ArgumentError("AGHQ Delta-Gamma K must be an integer in 1:p"))
    k isa Integer && !(k isa Bool) && k>0 ||
        throw(ArgumentError("AGHQ Delta-Gamma k must be a positive integer"))
    _aghq_kd_bound(K,k)
    mode_maxiter isa Integer && !(mode_maxiter isa Bool) && mode_maxiter>0 ||
        throw(ArgumentError("AGHQ mode_maxiter must be a positive integer"))
    for (name,value) in ((:mode_tol,mode_tol),(:mode_gradient_tol,mode_gradient_tol))
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    predictor in (:separate,:shared) ||
        throw(ArgumentError("AGHQ Delta-Gamma predictor must be :separate or :shared; got :$predictor"))
    disp_group in (:shared,:species) ||
        throw(ArgumentError("AGHQ Delta-Gamma disp_group must be :shared or :species; got :$disp_group"))
    offset===nothing || size(offset)==size(Y) ||
        throw(DimensionMismatch("AGHQ offset must match Y"))
    resp=zeros(p,n);off=zeros(p,n)
    for s in 1:n,t in 1:p
        y=Y[t,s]
        y isa Real && isfinite(y) && y>=0 ||
            throw(ArgumentError("AGHQ Delta-Gamma response ($t,$s) must be finite and nonnegative"))
        o=offset===nothing ? 0.0 : offset[t,s]
        o isa Real && isfinite(Float64(o)) ||
            throw(ArgumentError("AGHQ Delta-Gamma offset ($t,$s) must be finite"))
        resp[t,s]=Float64(y);off[t,s]=Float64(o)
    end
    shared=predictor===:shared;nα=disp_group===:shared ? 1 : p
    grid=aghq_grid(K,k);rr=rr_theta_len(p,K);nb=shared ? p : 2p;nparams=nb+rr+nα
    function unpack(theta)
        length(theta)==nparams || throw(DimensionMismatch("AGHQ Delta-Gamma parameter length must be $nparams"))
        all(isfinite,theta) || throw(ArgumentError("AGHQ Delta-Gamma parameters must be finite"))
        βz=view(theta,1:p);βc=shared ? βz : view(theta,p+1:2p)
        Λ=unpack_lambda(view(theta,nb+1:nb+rr),p,K)
        α=exp.(view(theta,nb+rr+1:nparams));αv=nα==1 ? fill(α[1],p) : α
        return βz,βc,Λ,αv
    end
    # the occurrence predictor carries the loadings and offset only under :shared
    function joint(z,βz,βc,Λ,α,s)
        value=-sum(abs2,z)/2-K*log(2pi)/2
        for t in 1:p
            lz=dot(view(Λ,t,:),z)
            ηz=βz[t]+(shared ? off[t,s]+lz : 0.0)
            ηc=βc[t]+off[t,s]+lz
            value+=_delta_gamma_logf(resp[t,s],ηz,ηc,α[t])
        end
        return value
    end
    function checked_modes(theta)
        βz,βc,Λ,α=unpack(theta)
        fams=DeltaGamma.(Float64.(α));Λf=Matrix{Float64}(Λ);Λz=shared ? Λf : zeros(p,K)
        βzf=Float64.(βz);βcf=Float64.(βc)
        caches=AGHQAdaptation[];diagnostics=NamedTuple[]
        for s in 1:n
            z,ok=_twopart_mode_search(fams,view(resp,:,s),Λz,Λf,βzf,βcf;
                offsetz=shared ? view(off,:,s) : nothing,offsetc=view(off,:,s),
                maxiter=mode_maxiter,tol=mode_tol)
            logjoint=v->joint(v,βz,βc,Λ,α,s)
            gradient=ForwardDiff.gradient(logjoint,z)
            g=maximum(abs,gradient)
            ok && all(isfinite,z) && isfinite(logjoint(z)) && isfinite(g) && g<=mode_gradient_tol ||
                error("AGHQ Delta-Gamma conditional mode failed at site $s (gradient_max=$g, tolerance=$mode_gradient_tol)")
            H=-ForwardDiff.hessian(logjoint,z)
            all(isfinite,H) && isposdef(Symmetric(H)) ||
                error("AGHQ Delta-Gamma observed curvature is invalid at site $s")
            cache=aghq_adaptation(z,H)
            push!(caches,cache)
            push!(diagnostics,(site=s,gradient_max=g,minimum_eigenvalue=cache.minimum_eigenvalue,
                curvature_repaired=cache.curvature_repaired))
        end
        return caches,diagnostics
    end
    function objective(theta,caches)
        βz,βc,Λ,α=unpack(theta)
        length(caches)==n || throw(DimensionMismatch("AGHQ Delta-Gamma needs one cache per site"))
        return -sum(aghq_frozen_logintegral(z->joint(z,βz,βc,Λ,α,s),caches[s],grid) for s in 1:n)
    end
    return (adapt=theta->first(checked_modes(theta)),objective=objective,
        mode_diagnostics=theta->last(checked_modes(theta)),grid=grid,nparams=nparams,
        data=(responses=copy(resp),offset=copy(off)),predictor=predictor,disp_group=disp_group)
end

"""
    DeltaGammaAGHQFit

Result of [`fit_delta_gamma_gllvm_aghq`](@ref): the fields of `DeltaGammaFit` (`βz`, `βc`,
`Λc`, shape `α` as a scalar under `disp_group = :shared` or a length-p vector under
`:species`, `predictor`, `disp_group`; under the `:shared` predictor `βz` equals `βc` and
`Λc` is the one set of loadings) plus `theta_packed` (intercepts, packed `Λ`, `log α`) and
`integration` (`AGHQFitInfo`) recording the actual route (`:aghq`, or `:laplace` with a
reason when the request was declined). `loglik` is the AGHQ log likelihood when
`integration.actual === :aghq`. Point estimates only: `predict`/`confint`/`simulate` have no
AGHQ route for this type.
"""
struct DeltaGammaAGHQFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    α::Union{Float64, Vector{Float64}}
    loglik::Float64
    converged::Bool
    iterations::Int
    predictor::Symbol
    disp_group::Symbol
    theta_packed::Vector{Float64}
    hessian::Symbol
    integration::AGHQFitInfo
end
function Base.show(io::IO, f::DeltaGammaAGHQFit)
    p, K = size(f.Λc)
    αstr = f.α isa Real ? string(round(f.α; sigdigits = 4)) : "per-trait"
    print(io, "DeltaGammaAGHQFit(p=", p, ", K=", K, ", α=", αstr, ", predictor=", f.predictor,
          ", loglik=", round(f.loglik; sigdigits = 7), ", route=", f.integration.actual,
          f.converged ? "" : ", NOT CONVERGED", ")")
end
_loadings(fit::DeltaGammaAGHQFit) = fit.Λc
_loglik(fit::DeltaGammaAGHQFit)   = fit.loglik
_nparams(fit::DeltaGammaAGHQFit) = length(fit.theta_packed)

"""
    fit_delta_gamma_gllvm_aghq(Y; K, predictor=:separate, disp_group=:species, aghq=:auto,
        aghq_control=(;), kwargs...)

Delta-Gamma two-part GLLVM (Bernoulli occurrence x positive Gamma, logit / log links) with
the parameterisations of [`fit_delta_gamma_gllvm`](@ref), fitted with adaptive
Gauss-Hermite quadrature over the one latent z shared by both parts. gllvmTMB's
`delta_gamma()` model is `predictor = :shared, disp_group = :species`. Also reached as
`fit_gllvm(Y; family=DeltaGamma(), K, predictor=..., aghq=...)`. `aghq` follows
[`fit_poisson_gllvm`](@ref) (`false` means call `fit_delta_gamma_gllvm` directly; `1` is
Laplace; a node count; `true`/`:auto`) except that `:auto` uses **5 nodes**, which is what
gllvmTMB's `aghq = "auto"` resolves to for `delta_gamma()` (its delta-family ladder rung of 9
is keyed on a family label that the family object does not match, so the discrete rung 5
applies; measured at the P1 pin), and is declined at 20 traits or more.
Ineligible requests (`K` above 5) keep the Laplace fit with a warning and a recorded
reason. The warm start is the Laplace fit; convergence means the frozen-node gradient rule,
not re-adapted stationarity. `aghq_control` is as for Poisson. `hessian` must be
`:observed`.
"""
function fit_delta_gamma_gllvm_aghq(Y::AbstractMatrix{<:Real};K::Integer,predictor::Symbol=:separate,
        disp_group::Symbol=:species,aghq=:auto,aghq_control=(;),kwargs...)
    kwargs=_entry_offset_kwargs(kwargs,Y,"fit_delta_gamma_gllvm_aghq";maskable=false)
    request=_aghq_request(aghq);request===:off && throw(ArgumentError("fit_delta_gamma_gllvm_aghq needs aghq != false; call fit_delta_gamma_gllvm"))
    predictor in (:separate,:shared) || throw(ArgumentError(
        "fit_delta_gamma_gllvm_aghq: predictor must be :separate or :shared; got :$predictor"))
    disp_group in (:shared,:species) || throw(ArgumentError(
        "fit_delta_gamma_gllvm_aghq: disp_group must be :shared or :species; got :$disp_group"))
    c=_aghq_controls(aghq_control)
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 5 : request
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    kw=Base.structdiff((;kwargs...),(hessian=nothing,))
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        K>5 ? :unaffordable_dimension : request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=fit_delta_gamma_gllvm(Y;K=K,predictor=predictor,disp_group=disp_group,kw...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _delta_gamma_with_integration(base,info)
    end
    problem=aghq_delta_gamma_problem(Y,K;k=k,predictor=predictor,disp_group=disp_group,
        offset=get(kwargs,:offset,nothing),mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,
        mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge(kw,(newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=fit_delta_gamma_gllvm(Y;K=K,predictor=predictor,disp_group=disp_group,base_kwargs...)
    theta=_delta_gamma_theta_from_fit(base);starts=[theta]
    rr=rr_theta_len(p,K);nb=predictor===:shared ? p : 2p
    if c.multistart
        alt=copy(theta);alt[nb+1:nb+rr].=.3;push!(starts,alt)
    end
    result=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _delta_gamma_with_integration(base,info)
    end
    selected=result.selected;t=selected.parameters
    data=AGHQDeltaGammaData(problem.data.responses,problem.data.offset,predictor,disp_group)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    βz=copy(t[1:p]);βc=predictor===:shared ? copy(βz) : copy(t[p+1:2p])
    α=disp_group===:shared ? exp(t[nb+rr+1]) : exp.(t[nb+rr+1:end])
    return DeltaGammaAGHQFit(βz,βc,unpack_lambda(t[nb+1:nb+rr],p,K),α,-selected.objective,
        selected.converged,selected.passes,predictor,disp_group,copy(t),:observed,info)
end
function _delta_gamma_theta_from_fit(f)
    β=f.predictor===:shared ? f.βc : vcat(f.βz,f.βc)
    return vcat(β,pack_lambda(f.Λc),log.(f.α isa Real ? [f.α] : f.α))
end
_delta_gamma_with_integration(f::DeltaGammaFit,i)=DeltaGammaAGHQFit(f.βz,f.βc,f.Λc,f.α,f.loglik,
    f.converged,f.iterations,f.predictor,f.disp_group,_delta_gamma_theta_from_fit(f),:observed,i)
