"""
    SourceCovariance(C, projection; name=:source, mode=:latent, rank=nothing,
                     unique=false, common=false)
    SourceCovariance(C; groups, kwargs...)

Describe a fixed source covariance `C` on ordered source nodes and a units ×
nodes projection. Integer `groups` builds a one-hot projection. Matrices are
copied, must be finite, and `C` must be exactly symmetric positive definite.
No jitter or loading ridge is added. Unobserved source nodes may be retained.

Trait covariance modes are `:latent` (rank-one by default), `:indep` (separate
trait variances), and `:dep` (full lower-triangular loadings). `unique=true` adds
diagonal covariance to a latent block. `common=true` ties independent variances,
or the unique diagonal of a latent block; it never creates a shared scalar field.
Rank is specified only for `:latent`. Source names must be distinct within a fit.
See [`fit_gaussian_sources`](@ref) for the current fitting domain and limitations.
"""
struct SourceCovariance
    name::Symbol
    covariance::Matrix{Float64}
    projection::Matrix{Float64}
    mode::Symbol
    rank::Int
    unique::Bool
    common::Bool
end

function SourceCovariance(C::AbstractMatrix, P::AbstractMatrix;
        name::Symbol=:source, mode::Symbol=:latent, rank=nothing,
        unique::Bool=false, common::Bool=false)
    size(C,1)==size(C,2) && size(C,1)>0 || throw(DimensionMismatch("source covariance must be nonempty and square"))
    size(P,2)==size(C,1) && size(P,1)>0 || throw(DimensionMismatch("projection must be nonempty units × source nodes"))
    C = _symmetric_admit(C, "source covariance")
    all(x->x isa Real && isfinite(x),P) || throw(ArgumentError("projection must be finite and real"))
    mode in (:latent,:indep,:dep) || throw(ArgumentError("source mode must be :latent, :indep or :dep"))
    unique && mode!==:latent && throw(ArgumentError("unique is allowed only with mode=:latent"))
    common && !(mode===:indep || (mode===:latent && unique)) && throw(ArgumentError("common requires independent or latent-unique variances"))
    if mode===:latent
        rank===nothing && (rank=1)
        rank isa Integer && !(rank isa Bool) && rank>0 || throw(ArgumentError("latent rank must be a positive integer"))
    else
        rank===nothing || throw(ArgumentError("rank is only specified for mode=:latent"))
        rank=0
    end
    A=Matrix{Float64}(C); projection=Matrix{Float64}(P)
    all(isfinite,A) && all(isfinite,projection) || throw(ArgumentError("source matrices exceed Float64 range"))
    isposdef(Symmetric(A)) || throw(ArgumentError("source covariance must be positive definite; no automatic regularization is applied"))
    return SourceCovariance(name,A,projection,mode,Int(rank),unique,common)
end

function SourceCovariance(C::AbstractMatrix; groups, kwargs...)
    ids=collect(groups)
    !isempty(ids) && all(g->g isa Integer && !(g isa Bool) && 1<=g<=size(C,1),ids) ||
        throw(ArgumentError("groups must contain one-based integer source-node indices"))
    P=zeros(length(ids),size(C,1))
    for (i,g) in enumerate(ids); P[i,g]=1; end
    return SourceCovariance(C,P;kwargs...)
end

function _source_nparams(s::SourceCovariance,p::Integer)
    if s.mode===:latent
        s.rank<=p || throw(ArgumentError("source $(s.name) rank exceeds the number of traits"))
        return rr_theta_len(p,s.rank)+(s.unique ? (s.common ? 1 : p) : 0)
    elseif s.mode===:dep
        return rr_theta_len(p,p)
    else
        return s.common ? 1 : p
    end
end

function _source_trait_covariances(sources,p,parameters)
    length(parameters)==sum(s->_source_nparams(s,p),sources;init=0) || throw(DimensionMismatch("source covariance parameter count differs"))
    T=eltype(parameters);result=Matrix{T}[];offset=0
    for s in sources
        if s.mode===:indep
            n=s.common ? 1 : p
            d=s.common ? fill(exp(2parameters[offset+1]),p) : exp.(2 .* parameters[offset+1:offset+n])
            B=Matrix(Diagonal(d));offset+=n
        else
            rank=s.mode===:dep ? p : s.rank;n=rr_theta_len(p,rank)
            L=unpack_lambda(view(parameters,offset+1:offset+n),p,rank)
            B=L*L';offset+=n
            if s.unique
                n=s.common ? 1 : p
                d=s.common ? fill(exp(2parameters[offset+1]),p) : exp.(2 .* parameters[offset+1:offset+n])
                B+=Diagonal(d);offset+=n
            end
        end
        push!(result,B)
    end
    return result
end

"""Validate and convert a user-supplied fixed residual SD to `Float64`."""
function _source_fixed_sigma(value)
    value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
        throw(ArgumentError("sigma_eps_fixed must be a finite positive real representable as Float64"))
    sigma=try
        Float64(value)
    catch
        throw(ArgumentError("sigma_eps_fixed must be a finite positive real representable as Float64"))
    end
    isfinite(sigma) && sigma>0 ||
        throw(ArgumentError("sigma_eps_fixed must be a finite positive real representable as Float64"))
    return sigma
end

function _gaussian_sources_nll(Y,sources,theta; projected=nothing,sigma_eps_fixed=nothing,
        X=nothing)
    p,n=size(Y);m=p*n
    q=X===nothing ? p : size(X,2)
    source_coordinates=sum(s->_source_nparams(s,p),sources;init=0)
    residual_fixed=sigma_eps_fixed!==nothing
    length(theta)==q+source_coordinates+(residual_fixed ? 0 : 1) ||
        throw(DimensionMismatch("source-model parameter count differs"))
    theta_zero=isempty(theta) ? 0.0 : zero(theta[1])
    infinity=oftype(theta_zero,Inf)
    all(isfinite,theta) || return infinity
    # Keep the internal likelihood parameterization in logSD coordinates while
    # differentiating only the user-visible free coordinates.
    full_theta=residual_fixed ? [theta;oftype(theta_zero,log(sigma_eps_fixed))] : theta
    B=_source_trait_covariances(sources,p,view(full_theta,q+1:q+source_coordinates))
    S=projected===nothing ? [s.projection*s.covariance*s.projection' for s in sources] : projected
    V=exp(2full_theta[end])*Matrix{eltype(full_theta)}(I,m,m)
    for r in eachindex(sources);V+=kron(S[r],B[r]);end
    all(isfinite,V) || return infinity
    factor=try
        cholesky(Symmetric(V))
    catch e
        e isa PosDefException || rethrow()
        return infinity
    end
    residual=X===nothing ? vec(Y.-view(full_theta,1:p)) : vec(Y)-X*view(full_theta,1:q)
    return (m*log(2pi)+logdet(factor)+dot(residual,factor\residual))/2
end

function _trait_mean_design(p::Integer,n::Integer)
    D=zeros(Float64,p*n,p)
    for unit in 1:n, trait in 1:p
        D[trait+p*(unit-1),trait]=1.0
    end
    return D
end

_trait_coefficient_names(p::Integer)=Union{String,Symbol}["trait$(j)" for j in 1:p]

function _source_mean_design(X,p::Integer,n::Integer)
    m=p*n
    raw=if X isa AbstractMatrix
        size(X,1)==m || throw(DimensionMismatch("X must have p*n rows"))
        X
    elseif X isa AbstractArray && ndims(X)==3
        size(X,1)==p && size(X,2)==n ||
            throw(DimensionMismatch("three-dimensional X must have shape p × n × q"))
        reshape(X,m,size(X,3))
    else
        throw(ArgumentError("X must be a p*n × q matrix or p × n × q array"))
    end
    all(x->x isa Real && isfinite(x),raw) ||
        throw(ArgumentError("X must be finite and real"))
    D=try
        Matrix{Float64}(raw)
    catch
        throw(ArgumentError("X must be convertible to finite Float64 values"))
    end
    all(isfinite,D) || throw(ArgumentError("X exceeds the Float64 range"))
    rank(D)==size(D,2) || throw(ArgumentError("X must have full column rank"))
    return D
end

function _source_coefficient_names(names,q::Integer; default_trait_names::Bool=false)
    raw=if names===nothing
        default_trait_names ? _trait_coefficient_names(q) :
            Union{String,Symbol}["x$(j)" for j in 1:q]
    else
        length(names)==q || throw(DimensionMismatch("coefficient_names must have one entry per coefficient"))
        collect(names)
    end
    all(name->(name isa String || name isa Symbol) && !isempty(String(name)),raw) ||
        throw(ArgumentError("coefficient_names must be nonempty strings or symbols"))
    length(unique(String.(raw)))==q || throw(ArgumentError("coefficient_names must be unique"))
    return Vector{Union{String,Symbol}}(raw)
end

"""
    GaussianSourcesFit

Result of [`fit_gaussian_sources`](@ref). `beta` contains fixed-effect mean
coefficients, with their complete `mean_design` and `coefficient_names` retained
alongside the response `response_shape`.
`sigma_eps` the independent residual SD, and `trait_covariances` the estimated
covariance for each named source in `sources` order. `parameters` is the exact
optimizer vector (means, source coordinates, and residual logSD when that SD is
free). `residual_fixed` records whether `sigma_eps` was held fixed.

`converged` requires the optimizer verdict and a fresh gradient check.
`gradient_norm`, `hessian_min_eigenvalue`, `hessian_positive_definite`,
`iterations`, and `stopping_reason` expose fit diagnostics. A positive Hessian
or converged optimizer does not establish identification or recovery to truth.
No confidence intervals or new-data prediction method are provided by this layer.
"""
struct GaussianSourcesFit <: StatsAPI.StatisticalModel
    beta::Vector{Float64}
    sigma_eps::Float64
    trait_covariances::Vector{Matrix{Float64}}
    sources::Vector{SourceCovariance}
    parameters::Vector{Float64}
    loglik::Float64
    converged::Bool
    gradient_norm::Float64
    hessian_min_eigenvalue::Float64
    hessian_positive_definite::Bool
    iterations::Int
    stopping_reason::Symbol
    observations::Int
    residual_fixed::Bool
    mean_design::Matrix{Float64}
    response_shape::Tuple{Int,Int}
    coefficient_names::Vector{Union{String,Symbol}}
end

function _legacy_source_mean_metadata(beta,observations)
    p=length(beta)
    p>0 && observations>=p && observations%p==0 ||
        throw(ArgumentError("legacy GaussianSourcesFit metadata cannot infer a valid response shape"))
    n=observations ÷ p
    return _trait_mean_design(p,n),(p,n),_trait_coefficient_names(p)
end

# Stored fits created before complete mean-design metadata used trait intercepts.
function GaussianSourcesFit(beta,sigma_eps,trait_covariances,sources,parameters,loglik,
        converged,gradient_norm,hessian_min_eigenvalue,hessian_positive_definite,iterations,
        stopping_reason,observations,residual_fixed)
    D,shape,names=_legacy_source_mean_metadata(beta,observations)
    return GaussianSourcesFit(beta,sigma_eps,trait_covariances,sources,parameters,loglik,
        converged,gradient_norm,hessian_min_eigenvalue,hessian_positive_definite,iterations,
        stopping_reason,observations,residual_fixed,D,shape,names)
end

# Stored unconstrained fits created before `residual_fixed` was recorded remain
# free-residual trait-intercept fits.
GaussianSourcesFit(beta,sigma_eps,trait_covariances,sources,parameters,loglik,
    converged,gradient_norm,hessian_min_eigenvalue,hessian_positive_definite,iterations,
    stopping_reason,observations)=GaussianSourcesFit(beta,sigma_eps,
        trait_covariances,sources,parameters,loglik,converged,gradient_norm,
        hessian_min_eigenvalue,hessian_positive_definite,iterations,stopping_reason,
        observations,false)

"""
    fit_gaussian_sources(Y; sources, X=nothing, coefficient_names=nothing,
                         start=nothing, sigma_eps_fixed=nothing,
                         g_tol=1e-6, iterations=500)

Fit finite complete Gaussian data `Y` (traits × units) with independent
residual noise and additive [`SourceCovariance`](@ref) blocks. The covariance
of `vec(Y)` is `sigma_eps² I + sum(kron(P*C*P', B))`. `X=nothing` preserves the
default trait-intercept mean model, with coefficients named `trait1`, `trait2`,
and so on. An explicit `X` is either a complete `p*n × q` matrix or a `p × n × q`
array, with traits varying fastest after flattening. It is converted and copied
to `Float64`, must be finite and full column rank, and defaults to names `x1`,
`x2`, and so on. `coefficient_names` may instead supply nonempty, unique string
or symbol names, which are copied into the result.

There must be at least two units. With a free residual SD, the default
trait-intercept model requires within-trait variation; an explicit design must
be nonsaturated and leave a nonzero ordinary-least-squares residual. A finite,
positive, `Float64`-representable `sigma_eps_fixed` holds the residual SD fixed,
allows constant responses and saturated fixed-effect designs, and permits a
zero-column `X` to represent a fixed zero mean. An empty `sources` vector is a
valid independent-residual model.

Fixed source matrices/projections are validated and copied. Trait covariance
parameters are raw packed lower loadings for latent/dependent modes, and
logSDs for independent/unique modes. `common` ties diagonal variances. `start`
contains means, each source's coordinates in order, and residual logSD only when
the residual SD is free; a supplied vector must be finite and have exactly the
required length. The default start is deterministic. No automatic ridge, source
jitter, random restart or estimator selection occurs. Final likelihood,
ForwardDiff gradient and Hessian are recomputed over the free coordinates.

This development interface is limited to fixed Gaussian source matrices. Its
dense covariance has quadratic memory and cubic factorization cost in the
number of observed cells. Fixed slopes and explicit-source Gaussian formula
models use this complete mean-design route. It does not implement random source
slopes, R source-term grammar, the R/Julia bridge, tree/pedigree/mesh parsing,
estimated source kernels, missing responses, loading masks, non-Gaussian
families or intervals. These remain required programme work; this API is not a
full R-parity claim.
"""
function fit_gaussian_sources(Y::AbstractMatrix{<:Real}; sources, X=nothing,
        coefficient_names=nothing,start=nothing,sigma_eps_fixed=nothing,
        g_tol::Real=1e-6,iterations::Integer=500)
    p,n=size(Y)
    p>0 && n>=2 || throw(ArgumentError("source fitting needs at least one trait and two units"))
    all(isfinite,Y) || throw(ArgumentError("source fitting requires finite complete responses"))
    isfinite(g_tol) && g_tol>0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations>=0 || throw(ArgumentError("iterations must be non-negative"))
    all(s->s isa SourceCovariance,sources) || throw(ArgumentError("sources must contain SourceCovariance objects"))
    length(unique(s.name for s in sources))==length(sources) || throw(ArgumentError("source names must be unique"))
    fixed_sigma=sigma_eps_fixed===nothing ? nothing : _source_fixed_sigma(sigma_eps_fixed)
    residual_fixed=fixed_sigma!==nothing
    # Revalidate mutable array contents and keep snapshots independent of the caller.
    snapshots=[SourceCovariance(s.covariance,s.projection;name=s.name,mode=s.mode,
        rank=s.mode===:latent ? s.rank : nothing,unique=s.unique,common=s.common) for s in sources]
    ss=SourceCovariance[snapshots...]
    all(s->size(s.projection,1)==n,ss) || throw(DimensionMismatch("source projection rows must equal Y's unit count"))
    source_coordinates=sum(s->_source_nparams(s,p),ss;init=0)
    data=Matrix{Float64}(Y);all(isfinite,data) || throw(ArgumentError("responses exceed Float64 range"))
    default_means=X===nothing
    D=default_means ? _trait_mean_design(p,n) : _source_mean_design(X,p,n)
    q=size(D,2)
    names=_source_coefficient_names(coefficient_names,q;default_trait_names=default_means)
    total=q+source_coordinates+(residual_fixed ? 0 : 1)
    means=vec(mean(data,dims=2))
    if !residual_fixed
        if default_means
            sum(abs2,data.-means)==0 && throw(ArgumentError("at least one trait must vary across units for a finite residual-variance ML estimate"))
        else
            q==p*n && throw(ArgumentError("a saturated X requires sigma_eps_fixed"))
            beta_ols=D\vec(data)
            sum(abs2,vec(data)-D*beta_ols)==0 &&
                throw(ArgumentError("X exactly fits Y; sigma_eps_fixed is required for a finite residual-variance ML estimate"))
        end
    end
    if start===nothing
        theta=default_means ? copy(means) : collect(D\vec(data))
        for s in ss
            if s.mode===:indep
                append!(theta,fill(log(.25),s.common ? 1 : p))
            else
                rank=s.mode===:dep ? p : s.rank
                L=fill(.05,p,rank)
                for j in 1:rank,i in 1:p;L[i,j]=i<j ? 0.0 : i==j ? .5 : .05;end
                append!(theta,pack_lambda(L))
                s.unique && append!(theta,fill(log(.25),s.common ? 1 : p))
            end
        end
        !residual_fixed && push!(theta,log(max(std(vec(data))/2,.1)))
    else
        length(start)==total || throw(DimensionMismatch("start has $(length(start)) coordinates; expected $total"))
        all(x->x isa Real && isfinite(x),start) || throw(ArgumentError("start must be finite and real"))
        theta=Float64.(start)
    end
    S=[s.projection*s.covariance*s.projection' for s in ss]
    objective(t)=_gaussian_sources_nll(data,ss,t;projected=S,sigma_eps_fixed=fixed_sigma,
        X=default_means ? nothing : D)
    isfinite(objective(theta)) || throw(ArgumentError("start produces an invalid covariance"))
    if total==0
        value=objective(theta)
        B=_source_trait_covariances(ss,p,Float64[])
        return GaussianSourcesFit(Float64[],fixed_sigma,B,ss,Float64[],-value,true,0.0,
            NaN,false,0,:converged,length(data),true,D,(p,n),names)
    end
    # Armijo-only backtracking can stop on objective roundoff before the
    # fresh-gradient gate on either mean parameterization; Hager-Zhang's
    # approximate-Wolfe acceptance reaches that gate on both declared
    # regressions (explicit source design, and the default-mean frozen
    # COV-ORD-LATENT-BARE case).
    optimizer=Optim.LBFGS()
    result=Optim.optimize(objective,theta,optimizer,
        Optim.Options(g_tol=Float64(g_tol),iterations=Int(iterations));autodiff=:forward)
    estimate=Optim.minimizer(result);value=objective(estimate)
    # An invalid final objective cannot yield trustworthy derivative diagnostics.
    # Do not differentiate the constant failure sentinel and report a zero gradient.
    gradient_norm=Inf
    min_eig=NaN
    if isfinite(value)
        gradient=ForwardDiff.gradient(objective,estimate)
        gradient_norm=all(isfinite,gradient) ? maximum(abs,gradient) : Inf
        H=ForwardDiff.hessian(objective,estimate)
        min_eig=all(isfinite,H) ? eigmin(Symmetric(H)) : NaN
    end
    pd=isfinite(min_eig) && min_eig>0
    converged=Optim.converged(result) && isfinite(value) && isfinite(gradient_norm) && gradient_norm<=g_tol
    reason=converged ? :converged : !isfinite(value) ? :invalid_final :
        Optim.iterations(result)>=iterations ? :iteration_limit : :gradient_not_converged
    B=_source_trait_covariances(ss,p,view(estimate,q+1:q+source_coordinates))
    sigma_eps=residual_fixed ? fixed_sigma : exp(estimate[end])
    return GaussianSourcesFit(collect(estimate[1:q]),sigma_eps,B,ss,
        collect(estimate),-value,converged,gradient_norm,min_eig,pd,
        Optim.iterations(result),reason,length(data),residual_fixed,D,(p,n),names)
end

"""Return the fixed-effect mean coefficients of a `GaussianSourcesFit`."""
coef(f::GaussianSourcesFit)=copy(f.beta)
"""Return the normalized marginal log likelihood of a `GaussianSourcesFit`."""
loglikelihood(f::GaussianSourcesFit)=f.loglik
"""Return the number of observed response cells used by a `GaussianSourcesFit`."""
nobs(f::GaussianSourcesFit)=f.observations
"""Return the number of free estimated parameters of a `GaussianSourcesFit`."""
dof(f::GaussianSourcesFit)=length(f.parameters)

function Base.show(io::IO,f::GaussianSourcesFit)
    print(io,"GaussianSourcesFit(",f.response_shape[1]," traits, ",length(f.beta)," coefficients, ",length(f.sources),
        " sources, residual=",(f.residual_fixed ? "fixed" : "free"),
        ", loglik=",f.loglik,", status=",f.stopping_reason,
        ", gradient_norm=",f.gradient_norm,", Hessian_PD=",f.hessian_positive_definite,")")
end
