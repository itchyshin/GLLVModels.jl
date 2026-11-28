# Optional natural-SD Gamma criterion for named source-tier Psi. This file is
# included by source_fit.jl; it does not extend the general prior-table parser.

function _psi_sd_prior_input(rows)
    rows === nothing && return NamedTuple[]
    rows isa AbstractVector || rows isa Tuple ||
        throw(ArgumentError("psi_sd_priors must be a vector or tuple of named rows"))
    return collect(rows)
end

function _resolve_psi_sd_priors(rows,sources,trait_names,p,q)
    trait_names === nothing && throw(ArgumentError("Psi SD priors require explicit trait_names"))
    length(trait_names)==p || throw(ArgumentError("trait_names must have one name per response trait"))
    all(t->(t isa Symbol || t isa AbstractString) && !isempty(String(t)),trait_names) ||
        throw(ArgumentError("trait_names must be nonempty symbols or strings"))
    traits=String.(trait_names)
    length(unique(traits))==p || throw(ArgumentError("trait_names must be unique"))
    # The first adapter deliberately admits only the independently specified
    # rank-one, trait-specific source decomposition. Other routes stay explicit.
    all(s->s.mode===:latent && s.rank==1 && s.unique && !s.common,sources) ||
        throw(ArgumentError("Psi SD priors currently require rank-one latent sources with unique=true and common=false"))
    isempty(sources) && throw(ArgumentError("Psi SD priors require at least one named latent source"))
    projected=[source.projection*source.covariance*source.projection' for source in sources]
    patterns=hcat((vec(K)/norm(K) for K in projected)...)
    rank(patterns)==length(sources) || throw(ArgumentError(
        "Psi SD prior sources require linearly independent projected kernel patterns"))
    indices=Dict{Tuple{String,String},Int}()
    offset=q
    for source in sources
        offset+=rr_theta_len(p,source.rank)
        for (trait,name) in enumerate(traits)
            indices[(String(source.name),name)]=offset+trait
        end
        offset+=p
    end
    resolved=NamedTuple[]
    seen=Set{Tuple{String,String}}()
    for row in rows
        row isa NamedTuple || throw(ArgumentError("each Psi SD prior must be a named row"))
        Set(keys(row))==Set((:kernel,:trait,:shape,:rate)) ||
            throw(ArgumentError("Psi SD prior rows must contain exactly kernel, trait, shape and rate"))
        row.kernel isa Union{Symbol,AbstractString} && row.trait isa Union{Symbol,AbstractString} ||
            throw(ArgumentError("Psi SD prior kernel and trait must be symbols or strings"))
        target=(String(row.kernel),String(row.trait))
        haskey(indices,target) || throw(ArgumentError("unknown Psi SD target $(target[1]) × $(target[2])"))
        target in seen && throw(ArgumentError("duplicate Psi SD target $(target[1]) × $(target[2])"))
        for (name,value,lower) in ((:shape,row.shape,1.0),(:rate,row.rate,0.0))
            value isa Real && !(value isa Bool) && isfinite(value) && value>lower ||
                throw(ArgumentError("Psi SD Gamma $name must be finite and greater than $lower"))
        end
        a,b=try Float64(row.shape),Float64(row.rate) catch
            throw(ArgumentError("Psi SD Gamma shape and rate must be representable as Float64"))
        end
        isfinite(a) && a>1 && isfinite(b) && b>0 ||
            throw(ArgumentError("Psi SD Gamma shape and rate must be finite representable Float64 values"))
        normalizer=loggamma(a)-a*log(b)
        isfinite(normalizer) || throw(ArgumentError("Psi SD Gamma normalizer is not finite"))
        push!(resolved,(target=:unique_variance,kernel=Symbol(target[1]),trait=Symbol(target[2]),
            parameter_index=indices[target],distribution=:gamma,shape=a,rate=b,mean=a/b,
            measure=:natural_sd,jacobian=false,once_per_target=true,
            negative_log_normalizer=normalizer))
        push!(seen,target)
    end
    return resolved
end

function _psi_sd_penalty(theta,rows)
    value=zero(eltype(theta))
    for row in rows
        eta=theta[row.parameter_index]
        value+=row.rate*exp(eta)-(row.shape-1)*eta+row.negative_log_normalizer
    end
    return value
end

# The shared source-table resolver returns one row per fixed-effect or Psi
# target. Keep the table penalty in optimizer coordinates: fixed-effect rows
# are defined on their selected coefficient scale, while Gamma rows are
# defined on the natural Psi SD and deliberately have no log-SD Jacobian.
function _source_prior_table_penalty(theta, rows)
    value = zero(eltype(theta))
    for row in rows
        if row.target === :fixef
            x = theta[row.parameter_index]
            args = row.args
            logdensity = if row.distribution === :normal
                logpdf(Normal(args[1], args[2]), x)
            elseif row.distribution === :student_t
                logpdf(Distributions.LocationScale(args[1], args[2], TDist(args[3])), x)
            else
                throw(ArgumentError("unsupported Gaussian source fixed-effect prior $(row.distribution)"))
            end
            value -= logdensity
        elseif row.target === :unique_variance
            row.measure === :natural_sd && row.jacobian === false &&
                row.once_per_target === true || throw(ArgumentError(
                    "source Psi Gamma rows must target natural SD once, without a Jacobian"))
            eta = theta[row.parameter_index]
            value += row.rate * exp(eta) - (row.shape - 1) * eta +
                row.negative_log_normalizer
        else
            throw(ArgumentError("unsupported Gaussian source prior target $(row.target)"))
        end
    end
    return value
end

"""
    PenalizedGaussianSourcesFit

A `fit_gaussian_sources` result with optional named source-trait Psi SD priors.
`logLik` and `loglik` are the unpenalized marginal Gaussian likelihood at the
reported estimate; `log_prior` sums normalized natural-SD Gamma densities;
`objective = logLik + log_prior` is the maximized penalized/MAP criterion.
It is not a full posterior. The stored `fit` retains ordinary source-fit data.

Fixed-effect-only, Psi-only, and mixed prior-table fits expose `penalised`,
`prior_targets`, `diagnostics`,
`pd_hessian_unpenalised`, `prior_held_params` and `se_type=:penalised`.
Penalized-curvature standard errors are diagnostics, not calibrated intervals.
The source-table route supports Normal or Student-t fixed-effect targets and
Gamma unique-variance targets. A fixed-effect-only table does not require Psi
coordinates. Psi targets currently require finite complete Gaussian data, a
positive known residual SD, and rank-one named latent sources with
trait-specific Psi.
"""
struct PenalizedGaussianSourcesFit <: StatsAPI.StatisticalModel
    fit::GaussianSourcesFit
    prior_targets::Vector{NamedTuple}
    diagnostics::NamedTuple
end

function Base.getproperty(f::PenalizedGaussianSourcesFit,name::Symbol)
    name in (:fit,:prior_targets,:diagnostics) && return getfield(f,name)
    diagnostic=getfield(f,:diagnostics)
    haskey(diagnostic,name) && return getproperty(diagnostic,name)
    return getproperty(getfield(f,:fit),name)
end
Base.propertynames(f::PenalizedGaussianSourcesFit,private::Bool=false)=
    Tuple(unique((propertynames(getfield(f,:fit),private)...,:fit,:prior_targets,:diagnostics,keys(getfield(f,:diagnostics))...)))
coef(f::PenalizedGaussianSourcesFit)=coef(f.fit)
loglikelihood(f::PenalizedGaussianSourcesFit)=f.fit.loglik
nobs(f::PenalizedGaussianSourcesFit)=nobs(f.fit)
dof(f::PenalizedGaussianSourcesFit)=dof(f.fit)
function StatsAPI.aic(f::PenalizedGaussianSourcesFit)
    @warn "AIC at a penalized estimate has no ordinary likelihood interpretation; use an unpenalized refit."
    return 2dof(f)-2loglikelihood(f)
end
function StatsAPI.bic(f::PenalizedGaussianSourcesFit)
    @warn "BIC at a penalized estimate has no ordinary likelihood interpretation; use an unpenalized refit."
    return log(nobs(f))*dof(f)-2loglikelihood(f)
end
function Base.show(io::IO,f::PenalizedGaussianSourcesFit)
    print(io,"PenalizedGaussianSourcesFit(logLik=",f.logLik,", log_prior=",f.log_prior,
        ", objective=",f.objective,", status=",f.convergence_status,
        ", se_type=penalised)")
end

function _psi_sd_diagnostics(;optimizer_finished,logLik,log_prior,gradient,
        hessian,hessian_unpenalised,theta,parameter_names,prior_targets,
        boundary_params=NamedTuple[],lower_boundary_indices=Int[],
        lower_boundary_values=Dict{Int,Float64}(),g_tol=1e-6,
        optimizer_code=nothing,optimizer_message="")
    objective=logLik+log_prior
    max_gradient=isempty(gradient) || !all(isfinite,gradient) ? Inf : maximum(abs,gradient)
    function curvature(H)
        all(isfinite,H) || return (false,fill(NaN,length(theta)))
        isapprox(H,H';atol=1e-10,rtol=1e-8) || return (false,fill(NaN,length(theta)))
        factor=try cholesky(Symmetric(H)) catch e
            e isa PosDefException || rethrow()
            return (false,fill(NaN,length(theta)))
        end
        se=sqrt.(diag(inv(factor)))
        return (true,se)
    end
    pd,se=curvature(hessian)
    pd0,_=curvature(hessian_unpenalised)
    # Shared-reference inflated-SE screen from drmTMB R/check.R (floor 50,
    # ratio 1000), applied to optimizer-coordinate SEs, not natural-SD errors.
    # Source pin: 67acc18a1cbee478f05cdbcc50ade9b095c2e0a7:1311ff.
    finite_se=filter(isfinite,se)
    ordinary_se=filter(<(50.0),finite_se)
    reference_se=isempty(finite_se) ? NaN : isempty(ordinary_se) ? minimum(finite_se) : median(ordinary_se)
    inflated=findall(s->isfinite(s) && s>=50.0 && s>=1000.0*reference_se,se)
    se_admissible=all(isfinite,se) && isempty(inflated)
    inflated_params=[(name=parameter_names[i],kind=occursin("unique_sd",parameter_names[i]) ?
        :unique_variance : startswith(parameter_names[i],"beta:") ? :fixed_effect :
        occursin("source_sd",parameter_names[i]) ? :source_variance : :loading)
        for i in inflated]
    lower_scores=Dict(index => _fit_natural_positive_score(gradient[index],
        lower_boundary_values[index]) for index in lower_boundary_indices)
    projected_gradient=_fit_projected_gradient(gradient;lower=lower_boundary_indices,
        lower_scores=lower_scores)
    target_names=[row.target === :unique_variance ?
        "unique_variance:$(row.kernel)::$(row.trait)" :
        "fixef:$(row.coefficient)" for row in prior_targets]
    verdict=_fit_convergence(optimizer_finished=Bool(optimizer_finished),
        objective=objective,logLik=logLik,log_prior=log_prior,penalized=true,
        gradient=gradient,projected_gradient=projected_gradient,hessian=hessian,
        hessian_unpenalised=hessian_unpenalised,prior_targets=target_names,
        boundary_params=[row.name for row in boundary_params],
        boundary_check_available=true,relative_gradient_tol=g_tol)
    held=NamedTuple[]
    if pd && !pd0 && all(isfinite,hessian_unpenalised)
        eigen0=eigen(Symmetric(hessian_unpenalised))
        cutoff=sqrt(eps(Float64))*max(1.0,maximum(abs,eigen0.values))
        weak=findall(<=(cutoff),eigen0.values)
        for row in prior_targets
            if sum(abs2,eigen0.vectors[row.parameter_index,weak])>sqrt(eps(Float64))
                base=(name=parameter_names[row.parameter_index],kind=row.target)
                push!(held, row.target === :unique_variance ?
                    merge(base,(kernel=row.kernel,trait=row.trait)) :
                    merge(base,(coefficient=row.coefficient,)))
            end
        end
    end
    status=verdict.status === :converged && !se_admissible ? :degenerate : verdict.status
    return (penalised=true,logLik=Float64(logLik),log_prior=Float64(log_prior),
        objective=Float64(objective),converged=status===:converged,
        convergence_status=status,stopping_reason=status,optimizer_finished=optimizer_finished,
        optimizer_code=optimizer_code,optimizer_message=optimizer_message,
        max_gradient=max_gradient,relative_gradient=verdict.relative_gradient,
        raw_relative_gradient=verdict.raw_relative_gradient,
        pd_hessian=pd,hessian_positive_definite=pd,pd_hessian_unpenalised=pd0,
        prior_held_params=held,boundary_params=boundary_params,se_type=:penalised,
        standard_errors=se,standard_errors_admissible=se_admissible,
        inflated_se_params=inflated_params,standard_error_reference=reference_se,
        hessian=Matrix(hessian),hessian_unpenalised=Matrix(hessian_unpenalised))
end

function _psi_sd_prior_fit(fit,data,likelihood,objective,rows,result,g_tol,trait_names)
    isfinite(fit.loglik) || error("Psi prior fitting did not produce a finite data likelihood")
    theta=fit.parameters;p,n=fit.response_shape;q=length(fit.beta)
    names=String["beta:$(name)" for name in fit.coefficient_names]
    boundaries=NamedTuple[]
    lower_boundary_indices=Int[]
    lower_boundary_values=Dict{Int,Float64}()
    # Match R's fitted kernel-Psi SD screen when Psi coordinates exist:
    # absolute SD < 1e-4 or SD / max(all estimable kernel Psi SDs) < 1e-3.
    # Fixed placeholders and observation residual SD are never part of it.
    all_unique_sd=Float64[]
    cursor=q
    for source in fit.sources
        cursor += source.mode === :dep ? rr_theta_len(p,p) :
            source.mode === :latent ? rr_theta_len(p,source.rank) : 0
        if source.mode === :latent && source.unique
            n_unique=source.common ? 1 : p
            append!(all_unique_sd,exp.(theta[cursor+1:cursor+n_unique]))
            cursor += n_unique
        elseif source.mode === :indep
            cursor += source.common ? 1 : p
        end
    end
    max_unique_sd=isempty(all_unique_sd) ? 0.0 : maximum(all_unique_sd)
    cursor=q
    for source in fit.sources
        if source.mode === :indep
            n_source=source.common ? 1 : p
            append!(names,["$(source.name).source_sd[$trait]" for trait in 1:n_source])
            cursor += n_source
            continue
        end
        n_load=source.mode === :dep ? rr_theta_len(p,p) : rr_theta_len(p,source.rank)
        load=theta[cursor+1:cursor+n_load]
        append!(names,["loading:$(source.name):$j" for j in 1:n_load])
        iszero(norm(load)) && push!(boundaries,
            (name="$(source.name).loading_column",kind=:loading_rank,kernel=source.name,trait=nothing))
        cursor += n_load
        if source.mode === :latent && source.unique
            n_unique=source.common ? 1 : p
            for trait in 1:n_unique
                trait_label=source.common ? "common" :
                    (trait_names === nothing || length(trait_names) != p ?
                        "trait$trait" : String(trait_names[trait]))
                label="$(source.name).unique_sd[$trait_label]"
                push!(names,label)
                sd=exp(theta[cursor+trait])
                if sd<1e-4 || sd/max_unique_sd<1e-3
                    index=cursor+trait
                    push!(boundaries,(name=label,kind=:unique_variance,
                        kernel=source.name,trait=source.common ? nothing : Symbol(trait_label)))
                    push!(lower_boundary_indices,index)
                    lower_boundary_values[index]=sd
                end
            end
            cursor += n_unique
        end
    end
    !fit.residual_fixed && push!(names,"log_sigma_eps")
    gradient,H,H0=if isfinite(objective(theta)) && isfinite(fit.loglik)
        (ForwardDiff.gradient(objective,theta),ForwardDiff.hessian(objective,theta),
            ForwardDiff.hessian(likelihood,theta))
    else
        (fill(NaN,length(theta)),fill(NaN,length(theta),length(theta)),
            fill(NaN,length(theta),length(theta)))
    end
    info=_psi_sd_diagnostics(optimizer_finished=Optim.converged(result),
        logLik=fit.loglik,log_prior=-_source_prior_table_penalty(theta,rows),gradient=gradient,
        hessian=H,hessian_unpenalised=H0,theta=theta,parameter_names=names,
        prior_targets=rows,boundary_params=boundaries,g_tol=g_tol,
        lower_boundary_indices=lower_boundary_indices,
        lower_boundary_values=lower_boundary_values,
        optimizer_code=nothing,
        optimizer_message=sprint(show,result))
    return PenalizedGaussianSourcesFit(fit,deepcopy(rows),info)
end
