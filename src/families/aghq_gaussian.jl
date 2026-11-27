"""
    aghq_gaussian_problem(Y, K; k, X=nothing, mask=nothing, offset=nothing,
                         mode_maxiter=100, mode_tol=1e-9, mode_gradient_tol=1e-7)

Internal ordinary Gaussian quadrature problem with a shared residual standard
deviation, matching the frozen R Stage1a model with `unique=false`. Parameters
are `[beta; log_sigma; pack_lambda(Lambda)]`. By default `beta` has one intercept
per trait. A complete `X[p,n,q]` instead defines the entire mean `X_s*beta`;
`q=0` specifies zero mean. Responses are traits × sites and inputs are copied.

Conditional modes and observed curvature are solved exactly. Quadrature uses
those checked caches, retaining the same normalized joint and fixed-cache
objective interface as the other internal adapters. With Gaussian observations,
one node is exact in value at its own adaptation but generally not in frozen
parameter derivatives; gradient agreement requires at least two nodes per axis,
and Hessian agreement at least three. This adapter constructs no public fit.
"""
function aghq_gaussian_problem(Y::AbstractMatrix,K;k,X=nothing,mask=nothing,offset=nothing,
        mode_maxiter=100,mode_tol=1e-9,mode_gradient_tol=1e-7)
    p,n=size(Y)
    p>0 && n>0 || throw(ArgumentError("AGHQ Gaussian requires nonempty responses"))
    K isa Integer && !(K isa Bool) && 1<=K<=p || throw(ArgumentError("AGHQ Gaussian K must be an integer in 1:p"))
    k isa Integer && !(k isa Bool) && k>0 || throw(ArgumentError("AGHQ Gaussian k must be a positive integer"))
    _aghq_kd_bound(K,k)
    for (name,value) in ((:mode_tol,mode_tol),(:mode_gradient_tol,mode_gradient_tol))
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    mode_maxiter isa Integer && !(mode_maxiter isa Bool) && mode_maxiter>0 ||
        throw(ArgumentError("mode_maxiter must be a positive integer"))
    if mask!==nothing
        size(mask)==size(Y) || throw(DimensionMismatch("AGHQ mask must match responses"))
        eltype(mask)<:Bool || throw(ArgumentError("AGHQ mask must contain Bool values"))
    end
    offset===nothing || size(offset)==size(Y) || throw(DimensionMismatch("AGHQ offset must match responses"))
    design=if X===nothing
        nothing
    else
        X isa AbstractArray && ndims(X)==3 && size(X,1)==p && size(X,2)==n ||
            throw(DimensionMismatch("AGHQ Gaussian X must have shape p by n by q"))
        all(v->v isa Real && isfinite(v) && isfinite(Float64(v)),X) || throw(ArgumentError("X must be finite"))
        copy(Float64.(X))
    end
    q=design===nothing ? p : size(design,3)
    observed=falses(p,n);responses=zeros(p,n);off=zeros(p,n)
    for s in 1:n,t in 1:p
        (mask===nothing || mask[t,s]) && !ismissing(Y[t,s]) || continue
        y=Y[t,s];o=offset===nothing ? 0.0 : offset[t,s]
        y isa Real && isfinite(y) && isfinite(Float64(y)) || throw(ArgumentError("observed Gaussian response ($t,$s) must be finite"))
        o isa Real && isfinite(o) && isfinite(Float64(o)) || throw(ArgumentError("observed Gaussian offset ($t,$s) must be finite"))
        observed[t,s]=true;responses[t,s]=y;off[t,s]=o
    end
    rows=[findall(view(observed,:,s)) for s in 1:n]
    grid=aghq_grid(K,k);nparams=q+1+rr_theta_len(p,K)
    function unpack(theta)
        length(theta)==nparams || throw(DimensionMismatch("Gaussian AGHQ parameter length must be $nparams"))
        all(isfinite,theta) || throw(ArgumentError("Gaussian AGHQ parameters must be finite"))
        variance=exp(2theta[q+1])
        isfinite(variance) && variance>0 || throw(ArgumentError("Gaussian AGHQ residual variance must be finite and positive"))
        return view(theta,1:q),theta[q+1],variance,unpack_lambda(view(theta,q+2:nparams),p,K)
    end
    mean_at(beta,t,s)=design===nothing ? beta[t] : dot(view(design,t,s,:),beta)
    function joint(z,beta,log_sigma,variance,loading,s)
        value=-K*log(2pi)/2-sum(abs2,z)/2
        for t in rows[s]
            residual=responses[t,s]-mean_at(beta,t,s)-off[t,s]-dot(view(loading,t,:),z)
            value-=log(2pi)/2+log_sigma+residual^2/(2variance)
        end
        return value
    end
    function checked_modes(theta)
        beta,log_sigma,variance,loading=unpack(theta)
        caches=AGHQAdaptation[];diagnostics=NamedTuple[]
        for s in 1:n
            obs=rows[s];L=loading[obs,:]
            residual=[responses[t,s]-mean_at(beta,t,s)-off[t,s] for t in obs]
            H=Matrix{Float64}(I,K,K)+L'*L/variance
            all(isfinite,H) || error("Gaussian AGHQ observed curvature is nonfinite at site $s")
            F=cholesky(Symmetric(H));z=F\(L'*residual/variance)
            logjoint=v->joint(v,beta,log_sigma,variance,loading,s)
            # Conditional modes are quadratic; the Cholesky solve is the exact mode.
            # The stationarity test uses the normal-equation residual (not ForwardDiff,
            # which can falsely fail the 1e-7 check when loadings and traits interact).
            g=norm(H * z - L' * residual / variance)
            refined=false
            if isfinite(g) && g>mode_gradient_tol && all(isfinite,z)
                nll=v->-logjoint(v)
                grad! = (out,v)->ForwardDiff.gradient!(out,nll,v)
                opt=Optim.optimize(nll,grad!,copy(z),Optim.BFGS(linesearch=Optim.LineSearches.BackTracking(order=3)),
                    Optim.Options(iterations=mode_maxiter,g_tol=mode_gradient_tol/10))
                z=Optim.minimizer(opt)
                g=norm(H * z - L' * residual / variance)
                refined=true
            end
            all(isfinite,z) && isfinite(logjoint(z)) && isfinite(g) && g<=mode_gradient_tol ||
                error("Gaussian AGHQ mode check failed at site $s (gradient_max=$g)")
            cache=aghq_adaptation(z,H)
            push!(caches,cache)
            push!(diagnostics,(site=s,gradient_max=g,minimum_eigenvalue=cache.minimum_eigenvalue,
                curvature_repaired=cache.curvature_repaired,mode_refined=refined))
        end
        return caches,diagnostics
    end
    function objective(theta,caches)
        beta,log_sigma,variance,loading=unpack(theta)
        length(caches)==n || throw(DimensionMismatch("Gaussian AGHQ requires one cache per site"))
        return -sum(aghq_frozen_logintegral(z->joint(z,beta,log_sigma,variance,loading,s),caches[s],grid) for s in 1:n)
    end
    return (adapt=t->first(checked_modes(t)),mode_diagnostics=t->last(checked_modes(t)),
        objective=objective,grid=grid,nparams=nparams,
        data=(responses=copy(responses),mask=copy(observed),offset=copy(off),
              design=design===nothing ? nothing : copy(design)))
end
