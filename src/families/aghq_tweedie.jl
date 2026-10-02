# Tweedie (compound Poisson-Gamma, 1 < power < 2, log link) AGHQ: problem builder and
# fitter, mirroring aghq_nb2.jl. Dispersion is per group of traits (`group`; one group
# per trait is gllvmTMB's `log_phi_tweedie`), and the power follows the three contracts of
# `fit_tweedie_gllvm_grouped`: fixed, one shared estimated power, or one estimated power per
# trait (`power_group = :species`, gllvmTMB's `logit_p_tweedie`, power = 1 + plogis(xi)).
# The Laplace grouped mode search is used only as a proposal (it is not modified here).

# log a(y, phi, power), the mu-free Dunn-Smyth series, with the same truncation window as
# `_tweedie_logA` but generic in the number type so ForwardDiff can differentiate it in
# (phi, power). The window [lo, hi] is chosen from the Float64 values (it is piecewise
# constant, so it carries no derivative); the series over that window is then evaluated
# in the caller's number type, i.e. the derivative is the exact derivative of the same
# truncated sum whose value `tweedie_logpdf` reports (terms beyond the window are
# >= 37 nats below the maximum).
function _tweedie_logA_ad(y::Float64, φ::Real, p::Real)
    φv = Float64(ForwardDiff.value(φ)); pv = Float64(ForwardDiff.value(p))
    αv = (2.0 - pv) / (1.0 - pv)
    av = -αv * log(y) + αv * log(pv - 1.0) - (1.0 - αv) * log(φv) - log(2.0 - pv)
    logWv(j) = j * av - loggamma(j + 1.0) - loggamma(-j * αv)
    jstar = max(1, round(Int, y^(2.0 - pv) / (φv * (2.0 - pv))))
    W = 1; lo = 1; hi = 1
    while true
        lo = max(1, jstar - W); hi = jstar + W
        terms = Float64[logWv(float(j)) for j in lo:hi]
        (maximum(terms) - max(terms[1], terms[end])) >= 37.0 || W >= 5000 ? break : (W *= 2)
    end
    α = (2.0 - p) / (1.0 - p)
    a = -α * log(y) + α * log(p - 1.0) - (1.0 - α) * log(φ) - log(2.0 - p)
    terms = [j * a - loggamma(j + 1.0) - loggamma(-j * α) for j in float.(lo:hi)]
    m = maximum(terms)
    return -log(y) + m + log(sum(exp(t - m) for t in terms))
end

# Tweedie log-density with the z-independent series term supplied (`logA`, 0 at y = 0).
@inline function _tweedie_logpdf_given_logA(y::Float64, μ, φ, p, logA)
    μ = max(μ, 1e-12)
    y == 0.0 && return -μ^(2.0 - p) / (φ * (2.0 - p))
    return (y * μ^(1.0 - p) / (1.0 - p) - μ^(2.0 - p) / (2.0 - p)) / φ + logA
end

"""
    aghq_tweedie_problem(Y, K; k, group=1:p, power=nothing, power_group=:shared,
        mask=nothing, offset=nothing, mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)

Internal loadings-only, log-link Tweedie AGHQ problem. Parameters are intercepts,
`pack_lambda` loadings, `log phi_g` for each dispersion group (groups relabelled to
`1..G` in sorted order), then the free power coordinates `xi` (none for fixed `power`,
one for `power_group = :shared`, `p` for `:species`), with `power = 1 + 1/(1+exp(-xi))`.
Same contract as [`aghq_poisson_problem`](@ref): `adapt(theta)`, `objective(theta,
caches)`, `mode_diagnostics(theta)`, `grid`, `nparams`; the objective differentiates the
frozen-cache surrogate only; modes are checked against the actual joint gradient and
observed Hessian before any cache is built. The normalized joint is the full Tweedie log
density plus a standard-normal latent prior, no loading penalty. The series term of the
density does not depend on the latent value, so it is evaluated once per observation
and parameter vector, not once per node.
"""
function aghq_tweedie_problem(Y::AbstractMatrix, K; k, group=nothing, power=nothing, power_group=:shared,
        mask=nothing, offset=nothing, mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)
    p,n=size(Y)
    p>0 && n>0 || throw(ArgumentError("AGHQ Tweedie needs nonempty responses"))
    K isa Integer && !(K isa Bool) && 1<=K<=p ||
        throw(ArgumentError("AGHQ Tweedie K must be an integer in 1:p"))
    k isa Integer && !(k isa Bool) && k>0 ||
        throw(ArgumentError("AGHQ Tweedie k must be a positive integer"))
    _aghq_kd_bound(K,k)
    mode_maxiter isa Integer && !(mode_maxiter isa Bool) && mode_maxiter>0 ||
        throw(ArgumentError("AGHQ mode_maxiter must be a positive integer"))
    for (name,value) in ((:mode_tol,mode_tol),(:mode_gradient_tol,mode_gradient_tol))
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    spec=_tweedie_power_spec(power,power_group,p)
    grp=group===nothing ? collect(1:p) : collect(Int,group)
    length(grp)==p || throw(ArgumentError("AGHQ Tweedie group must have length p"))
    labels=sort(unique(grp));G=length(labels)
    gidx=[findfirst(==(grp[t]),labels) for t in 1:p]
    if mask !== nothing
        size(mask)==size(Y) || throw(DimensionMismatch("AGHQ mask must match Y"))
        eltype(mask)<:Bool || throw(ArgumentError("AGHQ mask must contain Bool values"))
    end
    offset===nothing || size(offset)==size(Y) ||
        throw(DimensionMismatch("AGHQ offset must match Y"))
    observed=falses(p,n);resp=zeros(p,n);off=zeros(p,n)
    for s in 1:n,t in 1:p
        (mask===nothing || mask[t,s]) && !ismissing(Y[t,s]) || continue
        y=Y[t,s]
        y isa Real && isfinite(y) && y>=0 ||
            throw(ArgumentError("AGHQ Tweedie observed response ($t,$s) must be finite and nonnegative"))
        o=offset===nothing ? 0.0 : offset[t,s]
        o isa Real && isfinite(o) && isfinite(Float64(o)) ||
            throw(ArgumentError("AGHQ Tweedie observed offset ($t,$s) must be finite"))
        observed[t,s]=true;resp[t,s]=Float64(y);off[t,s]=Float64(o)
    end
    grid=aghq_grid(K,k);rr=rr_theta_len(p,K);nxi=spec.nfree;nparams=p+rr+G+nxi
    function unpack(theta)
        length(theta)==nparams || throw(DimensionMismatch("AGHQ Tweedie parameter length must be $nparams"))
        all(isfinite,theta) || throw(ArgumentError("AGHQ Tweedie parameters must be finite"))
        xi=view(theta,p+rr+G+1:nparams)
        pw=nxi==0 ? spec.values : spec.mode===:shared ? fill(_tweedie_power(xi[1]),p) : _tweedie_power.(xi)
        return view(theta,1:p),unpack_lambda(view(theta,p+1:p+rr),p,K),exp.(view(theta,p+rr+1:p+rr+G)),pw
    end
    # z-independent series term, one per observation (0 where y = 0 or unobserved)
    function logA_matrix(phig,pw)
        T=promote_type(eltype(phig),eltype(pw))
        A=zeros(T,p,n)
        for s in 1:n,t in 1:p
            observed[t,s] && resp[t,s]>0 || continue
            A[t,s]=_tweedie_logA_ad(resp[t,s],phig[gidx[t]],pw[t])
        end
        return A
    end
    function joint(z,beta,loading,phig,pw,A,s)
        value=-sum(abs2,z)/2-K*log(2pi)/2
        for t in 1:p
            observed[t,s] || continue
            eta=beta[t]+off[t,s]+dot(view(loading,t,:),z)
            value+=_tweedie_logpdf_given_logA(resp[t,s],exp(eta),phig[gidx[t]],pw[t],A[t,s])
        end
        return value
    end
    function checked_modes(theta)
        beta,loading,phig,pw=unpack(theta)
        A=logA_matrix(phig,pw)
        fams=[TweedieED(Float64(phig[gidx[t]]),Float64(pw[t])) for t in 1:p]
        caches=AGHQAdaptation[];diagnostics=NamedTuple[]
        for s in 1:n
            z=_grouped_laplace_mode(fams,view(resp,:,s),ones(p),loading,beta,LogLink();
                mask=view(observed,:,s),offset=view(off,:,s),maxiter=mode_maxiter,tol=mode_tol)
            logjoint=v->joint(v,beta,loading,phig,pw,A,s)
            gradient=ForwardDiff.gradient(logjoint,z)
            g=maximum(abs,gradient)
            all(isfinite,z) && isfinite(logjoint(z)) && isfinite(g) && g<=mode_gradient_tol ||
                error("AGHQ Tweedie conditional mode failed at site $s (gradient_max=$g, tolerance=$mode_gradient_tol)")
            H=-ForwardDiff.hessian(logjoint,z)
            all(isfinite,H) && isposdef(Symmetric(H)) ||
                error("AGHQ Tweedie observed curvature is invalid at site $s")
            cache=aghq_adaptation(z,H)
            push!(caches,cache)
            push!(diagnostics,(site=s,gradient_max=g,minimum_eigenvalue=cache.minimum_eigenvalue,
                curvature_repaired=cache.curvature_repaired))
        end
        return caches,diagnostics
    end
    function objective(theta,caches)
        beta,loading,phig,pw=unpack(theta)
        length(caches)==n || throw(DimensionMismatch("AGHQ Tweedie needs one cache per site"))
        A=logA_matrix(phig,pw)
        return -sum(aghq_frozen_logintegral(z->joint(z,beta,loading,phig,pw,A,s),caches[s],grid) for s in 1:n)
    end
    return (adapt=theta->first(checked_modes(theta)),objective=objective,
        mode_diagnostics=theta->last(checked_modes(theta)),grid=grid,nparams=nparams,
        data=(responses=copy(resp),mask=copy(observed),offset=copy(off),group=copy(gidx)),G=G,
        power_spec=spec)
end

"""
    TweedieGroupedAGHQFit

Result of [`fit_tweedie_gllvm_grouped_aghq`](@ref): `β`, `Λ`, the per-group dispersion
`φ`, the per-trait `power` vector (constant under a fixed or shared power; `power_mode`
is `:fixed`, `:shared` or `:species`), `group`, `link`, `loglik`, `converged`,
`iterations`, `theta_packed` (`β`, packed `Λ`, `log φ_group`, free `ξ`) and `integration`
(`AGHQFitInfo`) recording the actual route (`:aghq`, or `:laplace` with a reason when the
request was declined). `loglik` is the AGHQ log likelihood when `integration.actual ===
:aghq`. Point estimates only: `predict`/`confint`/`simulate` have no AGHQ route for this
type.
"""
struct TweedieGroupedAGHQFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    power::Vector{Float64}
    power_mode::Symbol
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
    hessian::Symbol
    integration::AGHQFitInfo
end
function Base.show(io::IO, f::TweedieGroupedAGHQFit)
    p, K = size(f.Λ)
    print(io, "TweedieGroupedAGHQFit(p=", p, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4), ", power=", round.(f.power; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7), ", route=", f.integration.actual,
          f.converged ? "" : ", NOT CONVERGED", ")")
end
_loadings(fit::TweedieGroupedAGHQFit) = fit.Λ
_loglik(fit::TweedieGroupedAGHQFit)   = fit.loglik
_nparams(fit::TweedieGroupedAGHQFit) = length(fit.theta_packed)

"""
    fit_tweedie_gllvm_grouped_aghq(Y; K, group=1:p, power=nothing, power_group=:shared,
        aghq=:auto, aghq_control=(;), kwargs...)

Tweedie (log link) GLLVM with per-group dispersion and the power contracts of
[`fit_tweedie_gllvm_grouped`](@ref), fitted with adaptive Gauss-Hermite quadrature. The
gllvmTMB `tweedie()` model is `group = 1:p` with `power_group = :species`. Also reached as
`fit_gllvm(Y; family=TweedieED(1.0, 1.5), K, disp_group=..., aghq=...)`. `aghq` follows
[`fit_poisson_gllvm`](@ref) (`false` means call `fit_tweedie_gllvm_grouped` directly; `1`
is Laplace; a node count; `true`/`:auto`) except that `:auto` uses **9 nodes**, the
starting rung gllvmTMB's node ladder gives Tweedie, and is declined at 20 traits or more.
Ineligible requests (`K` above 5, non-log link) keep the Laplace fit with a warning and a
recorded reason. The warm start is the Laplace fit; convergence means the frozen-node
gradient rule, not re-adapted stationarity, and additionally requires every free power to
stay off the closed ends of `(1, 2)`. `aghq_control` is as for Poisson. `hessian` must be
`:observed`.
"""
function fit_tweedie_gllvm_grouped_aghq(Y::AbstractMatrix;K::Integer,group=nothing,power=nothing,
        power_group::Symbol=:shared,aghq=:auto,aghq_control=(;),kwargs...)
    request=_aghq_request(aghq);request===:off && throw(ArgumentError("fit_tweedie_gllvm_grouped_aghq needs aghq != false; call fit_tweedie_gllvm_grouped"))
    c=_aghq_controls(aghq_control)
    c=_aghq_controls(merge(c,(mode_maxiter=get(kwargs,:newton_maxiter,c.mode_maxiter),mode_tol=get(kwargs,:newton_tol,c.mode_tol))))
    base_controls=deepcopy((;kwargs...))
    p,n=size(Y);k=request===:auto ? 9 : request
    grp=group===nothing ? collect(1:p) : collect(Int,group)
    spec=_tweedie_power_spec(power,power_group,p)
    link=get(kwargs,:link,LogLink())
    get(kwargs,:hessian,:observed)===:observed || throw(ArgumentError("AGHQ uses observed curvature; omit hessian or set hessian=:observed"))
    kw=Base.structdiff((;kwargs...),(hessian=nothing,))
    reason=k==1 ? :laplace_rule : K==0 ? :no_latent_block :
        !(link isa LogLink) ? :unsupported_link : K>5 ? :unaffordable_dimension :
        request===:auto && p>=20 ? :auto_trait_cutoff : :eligible
    if reason!==:eligible
        base=fit_tweedie_gllvm_grouped(Y;K=K,group=grp,power=power,power_group=power_group,kw...)
        reason===:laplace_rule || @warn "AGHQ request retained Laplace" reason=reason requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,reason,false,Inf,c,base_controls,nothing,
            AGHQAdaptation[],nothing,"",NaN)
        return _tweedie_grouped_with_integration(base,spec.mode,info)
    end
    problem=aghq_tweedie_problem(Y,K;k=k,group=grp,power=power,power_group=power_group,
        mask=get(kwargs,:mask,nothing),offset=get(kwargs,:offset,nothing),
        mode_maxiter=c.mode_maxiter,mode_tol=c.mode_tol,mode_gradient_tol=c.mode_gradient_tol)
    base_kwargs=merge(kw,(mask=problem.data.mask,
        offset=get(kwargs,:offset,nothing)===nothing ? nothing : problem.data.offset,
        newton_maxiter=c.mode_maxiter,newton_tol=c.mode_tol))
    base=fit_tweedie_gllvm_grouped(problem.data.responses;K=K,group=grp,power=power,power_group=power_group,base_kwargs...)
    theta=_tweedie_theta_from_fit(base,spec,p);starts=[theta]
    rr=rr_theta_len(p,K)
    if c.multistart
        alt=copy(theta);alt[p+1:p+rr].=.3;push!(starts,alt)
    end
    result=aghq_multistart_optimize(starts,problem.adapt,problem.objective;_aghq_outer_controls(c)...)
    if !result.usable
        @warn "AGHQ failed; retained Laplace" requested=request
        info=AGHQFitInfo(request,:laplace,1,k,1,:adaptation_failed,false,Inf,c,base_controls,result,
            AGHQAdaptation[],nothing,"",NaN)
        return _tweedie_grouped_with_integration(base,spec.mode,info)
    end
    selected=result.selected;t=selected.parameters;G=problem.G
    prediction_offset=copy(problem.data.offset)
    raw_offset=get(kwargs,:offset,nothing)
    if raw_offset!==nothing
        for j in eachindex(prediction_offset)
            v=raw_offset[j]
            if v isa Real && isfinite(v);prediction_offset[j]=v;end
        end
    end
    data=AGHQTweedieData(problem.data.responses,problem.data.mask,prediction_offset,problem.data.group)
    mode_gradient=maximum(d.gradient_max for d in problem.mode_diagnostics(t))
    info=AGHQFitInfo(request,:aghq,k,k,length(problem.grid.logw),selected.stop_reason,false,Inf,c,base_controls,
        result,deepcopy(selected.adaptation),data,_aghq_data_digest(data),mode_gradient)
    selected.parameter_shift==0 && @warn "AGHQ returned its warm start without parameter movement" reason=selected.stop_reason
    xi=t[p+rr+G+1:end]
    pw=spec.nfree==0 ? copy(spec.values) : spec.mode===:shared ? fill(_tweedie_power(xi[1]),p) : _tweedie_power.(xi)
    boundary=!isempty(xi) && any(>(_TWEEDIE_XI_MAX),abs.(xi))
    boundary && @warn "fit_tweedie_gllvm_grouped_aghq: a power ran to the boundary of (1, 2); flagged as not converged" power=pw
    return TweedieGroupedAGHQFit(copy(t[1:p]),unpack_lambda(t[p+1:p+rr],p,K),exp.(t[p+rr+1:p+rr+G]),pw,spec.mode,
        problem.data.group,link,-selected.objective,selected.converged && !boundary,selected.passes,copy(t),:observed,info)
end
function _tweedie_theta_from_fit(f,spec,p)
    xi=spec.nfree==0 ? Float64[] : spec.mode===:shared ? [_tweedie_xi(f.power)] : _tweedie_xi.(f.power)
    return vcat(f.β,pack_lambda(f.Λ),log.(f.φ),xi)
end
function _tweedie_grouped_with_integration(f,mode,i)
    pw=f.power isa AbstractVector ? collect(Float64,f.power) : fill(Float64(f.power),size(f.Λ,1))
    spec=_TweediePowerSpec(pw,mode===:fixed ? 0 : mode===:shared ? 1 : length(pw),mode===:fixed,mode)
    return TweedieGroupedAGHQFit(f.β,f.Λ,f.φ,pw,mode,f.group,f.link,f.loglik,f.converged,f.iterations,
        _tweedie_theta_from_fit(f,spec,length(f.β)),f.hessian,i)
end
