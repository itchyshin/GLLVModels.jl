using Test, GLLVModels, LinearAlgebra
_psi_close_entries(x,y,atol,rtol)=all(abs.(x.-y).<=atol.+rtol.*abs.(y))

@testset "Named source Psi SD Gamma prior" begin
    C = [1.0 .2 .1; .2 1.0 .3; .1 .3 1.0]
    ss = [SourceCovariance(C; groups=1:3, name=:oak, rank=1, unique=true)]
    rows = [(kernel=:oak, trait=:t1, shape=2.7, rate=3.1)]
    names = [:t1,:t2,:t3]
    # Target resolution excludes loadings, residual SD and ordinary source SDs.
    resolved = GLLVModels._resolve_psi_sd_priors(rows,ss,names,3,3)
    @test only(resolved).parameter_index == 7
    @test only(resolved).target === :unique_variance
    @test only(resolved).measure === :natural_sd
    @test !only(resolved).jacobian
    theta = [.1,.2,.3,.7,-.3,.4,log(.15),log(.4),log(.2)]
    penalty(t) = GLLVModels._psi_sd_penalty(t,resolved)
    a,b,s = 2.7,3.1,.15
    expected = b*s-(a-1)*log(s)-a*log(b)+GLLVModels.loggamma(a)
    @test penalty(theta) ≈ expected atol=1e-14
    g = GLLVModels.ForwardDiff.gradient(penalty,theta)
    H = GLLVModels.ForwardDiff.hessian(penalty,theta)
    @test g[7] ≈ b*s-(a-1) atol=1e-13
    @test H[7,7] ≈ b*s atol=1e-13
    @test all(iszero,g[[1:6;8:9]])
    @test count(!iszero,H) == 1
    # No empty prior branch changes the ordinary result or diagnostic shape.
    Y = [.8 -.25 .6; -.1 -.8 .5; .35 .6 -.2]
    ordinary = fit_gaussian_sources(Y;sources=ss,start=theta,sigma_eps_fixed=.2,iterations=0)
    empty = fit_gaussian_sources(Y;sources=ss,start=theta,sigma_eps_fixed=.2,iterations=0,
        psi_sd_priors=NamedTuple[],trait_names=names)
    @test typeof(empty) === typeof(ordinary)
    @test propertynames(empty) === propertynames(ordinary)
    @test empty.parameters == ordinary.parameters
    @test empty.loglik == ordinary.loglik
    fit = fit_gaussian_sources(Y;sources=ss,start=theta,sigma_eps_fixed=.2,iterations=0,
        psi_sd_priors=rows,trait_names=names)
    @test fit.penalised
    @test fit.logLik == loglikelihood(fit)
    @test fit.objective ≈ fit.logLik+fit.log_prior atol=1e-13
    @test fit.log_prior ≈ -expected atol=1e-13
    @test fit.se_type === :penalised
    @test fit.convergence_status === :not_converged
    @test !fit.converged
    @test only(fit.prior_targets).parameter_index == 7
    @test_throws ArgumentError GLLVModels._resolve_psi_sd_priors([rows;rows],ss,names,3,3)
    for bad in ((kernel=:absent,trait=:t1,shape=2.,rate=1.),
                (kernel=:oak,trait=:bad,shape=2.,rate=1.),
                (kernel=:oak,trait=:t1,shape=1.,rate=1.),
                (kernel=:oak,trait=:t1,shape=2.,rate=0.),
                (kernel=:oak,trait=:t1,shape=true,rate=1.))
        @test_throws ArgumentError GLLVModels._resolve_psi_sd_priors([bad],ss,names,3,3)
    end
    @test_throws ArgumentError fit_gaussian_sources(Y;sources=ss,start=[theta;log(.2)],
        psi_sd_priors=rows,trait_names=names,iterations=0)
    @test_throws ArgumentError fit_gaussian_sources(Y;sources=ss,start=theta,
        sigma_eps_fixed=.2,psi_sd_priors=rows,iterations=0)
end

@testset "Psi prior-held convergence and status precedence" begin
    target=(target=:unique_variance,parameter_index=2,kernel=:oak,trait=:t1)
    base=(optimizer_finished=true,logLik=-.921586334535191,log_prior=-.102880015114119,
        gradient=[0.,0.],hessian=[1. 0.;0. 2.],hessian_unpenalised=[1. 0.;0. 0.],
        theta=[0.,log(.4)],parameter_names=["beta:t1","oak.unique_sd[t1]"],
        prior_targets=[target])
    d=GLLVModels._psi_sd_diagnostics(;base...)
    @test d.converged && d.pd_hessian && !d.pd_hessian_unpenalised
    @test only(d.prior_held_params).name == "oak.unique_sd[t1]"
    @test only(d.prior_held_params).kind === :unique_variance
    @test d.objective ≈ -1.02446634964931 atol=1e-14
    @test d.relative_gradient == 0
    boundary=[(name="oak.unique_sd[t1]",kind=:unique_variance)]
    @test GLLVModels._psi_sd_diagnostics(;merge(base,(boundary_params=boundary,))...).convergence_status === :boundary
    @test GLLVModels._psi_sd_diagnostics(;merge(base,(optimizer_finished=false,boundary_params=boundary,))...).convergence_status === :not_converged
    @test GLLVModels._psi_sd_diagnostics(;merge(base,(hessian=[1. 0.;0. 0.],))...).convergence_status === :degenerate
    @test GLLVModels._psi_sd_diagnostics(;merge(base,(gradient=[0.,1.],))...).convergence_status === :not_converged
    inflated=GLLVModels._psi_sd_diagnostics(;merge(base,(hessian=[1. 0.;0. 1e-12],))...)
    @test inflated.pd_hessian && !inflated.standard_errors_admissible
    @test inflated.convergence_status===:degenerate
    @test only(inflated.inflated_se_params).name=="oak.unique_sd[t1]"
    rescaled=GLLVModels._psi_sd_diagnostics(;merge(base,(hessian=[1e-6 0.;0. 2.5e-7],))...)
    @test rescaled.pd_hessian && rescaled.standard_errors_admissible
    @test rescaled.converged
    @test isempty(rescaled.inflated_se_params)

    # At a positive lower-bound screening value, project the log-SD score
    # after conversion to the natural SD coordinate. A negative natural score
    # violates the minimization KKT condition and must not be called converged.
    good_kkt = GLLVModels._psi_sd_diagnostics(;merge(base,(
        gradient=[0.0,0.2],boundary_params=boundary,
        lower_boundary_indices=[2],lower_boundary_values=Dict(2=>0.1),))...)
    bad_kkt = GLLVModels._psi_sd_diagnostics(;merge(base,(
        gradient=[0.0,-0.2],boundary_params=boundary,
        lower_boundary_indices=[2],lower_boundary_values=Dict(2=>0.1),))...)
    @test good_kkt.convergence_status === :boundary
    @test good_kkt.relative_gradient == 0
    @test bad_kkt.convergence_status === :not_converged
    underflow = GLLVModels._psi_sd_diagnostics(;merge(base,(
        boundary_params=boundary,lower_boundary_indices=[2],
        lower_boundary_values=Dict(2=>0.0),))...)
    @test underflow.convergence_status === :degenerate
end

# This fixture was frozen before the implementation adapter. Its converter
# checks every original SHA256 and requires no Julia JSON dependency.
include(joinpath(@__DIR__,"..","fixtures","psi_sd_prior","fixture.jl"))
@testset "Frozen independent two-kernel Gaussian oracle" begin
    G=GLLVModels
    f=PSI_SD_ORACLE
    matrix(rows)=permutedims(reduce(hcat,rows))
    Y=reshape(f["response"],3,4)
    sources=[SourceCovariance(matrix(f["kernels"][name]);groups=1:4,
        name=Symbol(name),rank=1,unique=true) for name in ("oak","pine")]
    rows=[(kernel=Symbol(row["kernel"]),trait=Symbol(row["trait"]),
        shape=row["shape"],rate=row["rate"]) for row in f["priors"]]
    resolved=G._resolve_psi_sd_priors(rows,sources,Symbol.(f["traits"]),3,3)
    # Native loadings are raw; oracle first loading is log-positive. Native
    # packing interleaves each source's loadings and Psi; compose the actual
    # package objective before AD, retaining the full second-order chain rule.
    native(t)=[t[1:3];exp(t[4]);t[5:6];t[10:12];exp(t[7]);t[8:9];t[13:15]]
    likelihood(t)=G._gaussian_sources_nll(Y,sources,native(t);sigma_eps_fixed=.2)
    penalty(t)=G._psi_sd_penalty(native(t),resolved)
    objective(t)=likelihood(t)+penalty(t)
    for (point_number,point) in enumerate(f["points"])
        t=point["theta"]
        @test likelihood(t) ≈ point["nll"] atol=1e-8 rtol=1e-8
        @test penalty(t) ≈ point["negative_log_prior"] atol=1e-8 rtol=1e-8
        @test objective(t) ≈ point["criterion"] atol=1e-8 rtol=1e-8
        @test _psi_close_entries(G.ForwardDiff.gradient(likelihood,t),point["unpenalized_gradient"],2e-6,2e-5)
        @test _psi_close_entries(G.ForwardDiff.hessian(likelihood,t),matrix(point["unpenalized_hessian"]),2e-5,2e-5)
        @test _psi_close_entries(G.ForwardDiff.gradient(objective,t),point["gradient"],2e-6,2e-5)
        @test _psi_close_entries(G.ForwardDiff.hessian(objective,t),matrix(point["hessian"]),2e-5,2e-5)
        if point_number<=5
            gradient=G.ForwardDiff.gradient(objective,t)
            H=G.ForwardDiff.hessian(objective,t)
            fd_gradient=similar(t);fd_H=zeros(length(t),length(t))
            for i in eachindex(t)
                step=1e-5*max(1.,abs(t[i]));ei=zeros(length(t));ei[i]=step
                fd_gradient[i]=(objective(t+ei)-objective(t-ei))/(2step)
                hi=2e-4*max(1.,abs(t[i]));ei[i]=hi
                fd_H[i,i]=(objective(t+ei)-2objective(t)+objective(t-ei))/hi^2
                for j in 1:i-1
                    hj=2e-4*max(1.,abs(t[j]));ej=zeros(length(t));ej[j]=hj
                    fd_H[i,j]=(objective(t+ei+ej)-objective(t+ei-ej)-
                        objective(t-ei+ej)+objective(t-ei-ej))/(4hi*hj)
                    fd_H[j,i]=fd_H[i,j]
                end
            end
            @test _psi_close_entries(gradient,fd_gradient,2e-7,2e-6)
            @test _psi_close_entries(H,fd_H,2e-5,2e-5)
            @test _psi_close_entries(H,matrix(point["hessian"]),2e-5,2e-5)
        end
    end
    nominal=f["points"][1]["theta"]
    swapped=reverse(sources)
    reordered=G._resolve_psi_sd_priors(rows,swapped,Symbol.(f["traits"]),3,3)
    swapnative(t)=native(t)[[1:3;10:15;4:9]]
    swapobjective(t)=G._gaussian_sources_nll(Y,swapped,swapnative(t);sigma_eps_fixed=.2)+
        G._psi_sd_penalty(swapnative(t),reordered)
    @test swapobjective(nominal) ≈ objective(nominal) atol=1e-12
    @test _psi_close_entries(G.ForwardDiff.gradient(swapobjective,nominal),G.ForwardDiff.gradient(objective,nominal),2e-12,0.0)
    @test _psi_close_entries(G.ForwardDiff.hessian(swapobjective,nominal),G.ForwardDiff.hessian(objective,nominal),1e-11,0.0)
    for control in f["fixed_sigma_sensitivity"]
        t=control["theta"]
        changed(x)=G._gaussian_sources_nll(Y,sources,native(x);sigma_eps_fixed=control["known_residual_sd"])
        @test changed(t) ≈ control["nll"] atol=1e-8 rtol=1e-8
        @test _psi_close_entries(G.ForwardDiff.gradient(changed,t),control["unpenalized_gradient"],2e-6,2e-5)
        @test _psi_close_entries(G.ForwardDiff.hessian(changed,t),matrix(control["unpenalized_hessian"]),2e-5,2e-5)
    end
    # Deliberate mutations must be visible through the package penalty.
    @test abs(objective(nominal)-(likelihood(nominal)+2penalty(nominal)))>1e-5
    @test abs(objective(nominal)-(likelihood(nominal)+12penalty(nominal)))>1e-5
    @test abs(objective(nominal)-(objective(nominal)-sum(nominal[[10,14]])))>1e-5
    @test_throws ArgumentError G._resolve_psi_sd_priors(rows,[sources[1],
        SourceCovariance(sources[1].covariance;groups=1:4,name=:pine,rank=1,unique=true)],
        Symbol.(f["traits"]),3,3)
    # Exact-zero fixtures are independently valid but not finite log-SD
    # coordinates in this adapter. Do not substitute log(1e-5) for exact zero.
    @test_skip false # exact-zero package adapter is outside finite log-SD coordinates
end

@testset "Psi adapter mutation and scale controls" begin
    G=GLLVModels;f=PSI_SD_ORACLE
    matrix(rows)=permutedims(reduce(hcat,rows))
    Y=reshape(f["response"],3,4); t=f["points"][1]["theta"]
    ss=[SourceCovariance(matrix(f["kernels"][name]);groups=1:4,
        name=Symbol(name),rank=1,unique=true) for name in ("oak","pine")]
    rows=[(kernel=Symbol(r["kernel"]),trait=Symbol(r["trait"]),shape=r["shape"],rate=r["rate"]) for r in f["priors"]]
    resolved=G._resolve_psi_sd_priors(rows,ss,Symbol.(f["traits"]),3,3)
    native(x)=[x[1:3];exp(x[4]);x[5:6];x[10:12];exp(x[7]);x[8:9];x[13:15]]
    nll(x)=G._gaussian_sources_nll(Y,ss,native(x);sigma_eps_fixed=.2)
    penalty(x)=G._psi_sd_penalty(native(x),resolved)
    obj(x)=nll(x)+penalty(x)
    wrong=[merge(resolved[1],(parameter_index=8,)),resolved[2]]
    wrong_target(x)=nll(x)+G._psi_sd_penalty(native(x),wrong)
    extra_jacobian(x)=obj(x)-x[10]-x[14]
    variance_confusion(x)=nll(x)+sum(r.rate*exp(2native(x)[r.parameter_index])-
        2(r.shape-1)*native(x)[r.parameter_index]+r.negative_log_normalizer for r in resolved)
    loading_confusion(x)=nll(x)+G._psi_sd_penalty(native(x),
        [merge(resolved[1],(parameter_index=4,)),merge(resolved[2],(parameter_index=11,))])
    omitted(x)=nll(x)
    for mutated in (wrong_target,extra_jacobian,variance_confusion,loading_confusion,omitted,
                    x->nll(x)+2penalty(x),x->nll(x)+4penalty(x),x->nll(x)+12penalty(x))
        @test abs(mutated(t)-obj(t))>1e-5
        @test maximum(abs.(G.ForwardDiff.gradient(mutated,t)-G.ForwardDiff.gradient(obj,t)))>1e-5 ||
            maximum(abs.(G.ForwardDiff.hessian(mutated,t)-G.ForwardDiff.hessian(obj,t)))>1e-5
    end
    c=3.7
    scaledrows=[merge(r,(rate=r.rate/c,)) for r in rows]
    scaledresolved=G._resolve_psi_sd_priors(scaledrows,ss,Symbol.(f["traits"]),3,3)
    scale(x)=[c.*x[1:3];x[4]+log(c);c.*x[5:6];x[7]+log(c);c.*x[8:9];x[10:15].+log(c)]
    scalednll(x)=G._gaussian_sources_nll(c.*Y,ss,native(scale(x));sigma_eps_fixed=.2c)
    scaledobj(x)=scalednll(x)+G._psi_sd_penalty(native(scale(x)),scaledresolved)
    @test scalednll(t)-nll(t) ≈ 12log(c) atol=1e-12
    @test scaledobj(t)-obj(t) ≈ 14log(c) atol=1e-12
    @test _psi_close_entries(G.ForwardDiff.gradient(scaledobj,t),G.ForwardDiff.gradient(obj,t),2e-12,0.0)
    @test _psi_close_entries(G.ForwardDiff.hessian(scaledobj,t),G.ForwardDiff.hessian(obj,t),1e-11,0.0)
    order=[4,2,1,3]
    reordered=[SourceCovariance(s.covariance;groups=order,name=s.name,rank=1,unique=true) for s in ss]
    orderobj(x)=G._gaussian_sources_nll(Y[:,order],reordered,native(x);sigma_eps_fixed=.2)+penalty(x)
    @test orderobj(t) ≈ obj(t) atol=1e-12
    @test _psi_close_entries(G.ForwardDiff.gradient(orderobj,t),G.ForwardDiff.gradient(obj,t),2e-12,0.0)
    @test _psi_close_entries(G.ForwardDiff.hessian(orderobj,t),G.ForwardDiff.hessian(obj,t),1e-11,0.0)
end

@testset "Psi prior normal-entry optimizer steps" begin
    G=GLLVModels
    # Deterministic smoke data, not a recovery or calibration study.
    n=6;p=3
    Y=[sin(.7j)+.2cos(1.4j) for i in 1:p,j in 1:n]
    Y[2,:].=cos.((1:n).*.8).-.3
    Y[3,:].=sin.((1:n).*1.1).+.2cos.((1:n).*.4)
    Ko=[.55^abs(i-j) for i in 1:n,j in 1:n]
    Kp=[(-.25)^abs(i-j) for i in 1:n,j in 1:n]
    sources=[SourceCovariance(K;groups=1:n,name=name,rank=1,unique=true)
        for (K,name) in ((Ko,:oak),(Kp,:pine))]
    traits=[:t1,:t2,:t3]
    rows=[(kernel=name,trait=trait,shape=4.,rate=5.) for name in (:oak,:pine) for trait in traits]
    fit=fit_gaussian_sources(Y;sources=sources,sigma_eps_fixed=.2,trait_names=traits,
        psi_sd_priors=rows,iterations=50,g_tol=1e-6)
    @test fit.iterations>0
    @test isfinite(fit.logLik) && isfinite(fit.log_prior) && isfinite(fit.objective)
    @test fit.objective ≈ fit.logLik+fit.log_prior atol=1e-12
    dataobjective(t)=G._gaussian_sources_nll(Y,fit.sources,t;sigma_eps_fixed=.2)
    objective(t)=dataobjective(t)+G._psi_sd_penalty(t,fit.prior_targets)
    @test _psi_close_entries(fit.hessian,G.ForwardDiff.hessian(objective,fit.parameters),1e-12,1e-12)
    @test _psi_close_entries(fit.hessian_unpenalised,G.ForwardDiff.hessian(dataobjective,fit.parameters),1e-12,1e-12)
    @test fit.max_gradient ≈ maximum(abs,G.ForwardDiff.gradient(objective,fit.parameters)) atol=1e-12
    @test fit.converged == (fit.convergence_status===:converged)
    @test fit.convergence_status in (:converged,:not_converged,:boundary,:degenerate)
    if fit.converged
        @test fit.pd_hessian && fit.standard_errors_admissible
        @test fit.relative_gradient<=1e-6
        @test isempty(fit.boundary_params)
    end
end

@testset "Gaussian source prior-table penalty combines targets once" begin
    theta = [-0.35, log(0.4), 0.2]
    fixed = (target=:fixef, kind=:fixed_effect, coefficient=:t1,
        parameter_index=1, distribution=:normal, args=[0.0, 1.0],
        measure=:link, jacobian=false, once_per_target=true)
    shape, rate = 2.0, 5.0
    psi = (target=:unique_variance, kernel=:oak, trait=:t1,
        parameter_index=2, distribution=:gamma, shape=shape, rate=rate,
        mean=shape/rate, measure=:natural_sd, jacobian=false,
        once_per_target=true, negative_log_normalizer=loggamma(shape)-shape*log(rate))
    rows = NamedTuple[fixed, psi]

    expected = -GLLVModels.logpdf(GLLVModels.Normal(0.0, 1.0), theta[1]) -
        GLLVModels.logpdf(GLLVModels.Gamma(shape, 1/rate), exp(theta[2]))
    @test GLLVModels._source_prior_table_penalty(theta, rows) ≈ expected atol=1e-13
    @test GLLVModels._source_prior_table_penalty(theta, NamedTuple[]) == 0.0
    @test GLLVModels._source_prior_table_penalty(theta, [fixed]) ≈
        -GLLVModels.logpdf(GLLVModels.Normal(0.0, 1.0), theta[1]) atol=1e-13
    @test GLLVModels._source_prior_table_penalty(theta, [psi]) ≈
        -GLLVModels.logpdf(GLLVModels.Gamma(shape, 1/rate), exp(theta[2])) atol=1e-13
end

@testset "fixed-effect-only source table needs no Psi coordinates" begin
    traits = [:t1, :t2, :t3]
    Y = [0.8 -0.25 0.6 0.15; -0.1 -0.8 0.5 0.3; 0.35 0.6 -0.2 -0.4]
    K = [1.0 0.2 0.0 0.0; 0.2 1.0 0.2 0.0;
         0.0 0.2 1.0 0.2; 0.0 0.0 0.2 1.0]
    sources = [SourceCovariance(K; groups=1:4, name=:oak, rank=1, unique=false)]
    start = [0.1, -0.1, 0.2, 0.7, -0.3, 0.4]
    priors = [(prior="normal(0, 1)", class="fixef", coef="t1",
        dpar="mu", scale="link", trait="", kernel="")]

    fit = fit_gaussian_sources(Y; sources, coefficient_names=traits,
        sigma_eps_fixed=0.2, start, priors, iterations=0)
    @test fit isa GLLVModels.PenalizedGaussianSourcesFit
    @test fit.log_prior ≈ GLLVModels.logpdf(GLLVModels.Normal(0, 1), start[1]) atol=1e-13
    @test fit.objective ≈ fit.logLik + fit.log_prior atol=1e-13
    @test only(fit.prior_targets).target === :fixef
    @test isempty(fit.boundary_params)
end

@testset "source prior table keeps named Psi targets under kernel reordering" begin
    traits = [:t1, :t2, :t3]
    Y = [0.8 -0.25 0.6 0.15; -0.1 -0.8 0.5 0.3; 0.35 0.6 -0.2 -0.4]
    Koak = [1.0 0.2 0.0 0.0; 0.2 1.0 0.2 0.0;
            0.0 0.2 1.0 0.2; 0.0 0.0 0.2 1.0]
    Kpine = [1.0 0.0 0.2 0.0; 0.0 1.0 0.0 0.2;
             0.2 0.0 1.0 0.0; 0.0 0.2 0.0 1.0]
    oak = SourceCovariance(Koak; groups=1:4, name=:oak, rank=1, unique=true)
    pine = SourceCovariance(Kpine; groups=1:4, name=:pine, rank=1, unique=true)
    theta = [0.1, -0.1, 0.2, 0.7, -0.3, 0.4, log(0.15), log(0.4), log(0.2),
        0.5, 0.1, -0.2, log(0.3), log(0.25), log(0.5)]
    rows = [
        (prior="gamma(0.4, 2)", class="unique_variance", coef="", dpar="",
         scale="natural_sd", trait="t1", kernel="oak"),
        (prior="gamma(0.6, 3)", class="unique_variance", coef="", dpar="",
         scale="natural_sd", trait="t2", kernel="pine"),
    ]
    first = fit_gaussian_sources(Y; sources=[oak, pine], coefficient_names=traits,
        sigma_eps_fixed=0.2, trait_names=traits, start=theta,
        priors=rows, iterations=0)
    reversed_start = [theta[1:3]; theta[10:15]; theta[4:9]]
    reversed = fit_gaussian_sources(Y; sources=[pine, oak], coefficient_names=traits,
        sigma_eps_fixed=0.2, trait_names=traits, start=reversed_start,
        priors=rows, iterations=0)
    @test first.logLik ≈ reversed.logLik atol=1e-10 rtol=1e-10
    @test first.log_prior ≈ reversed.log_prior atol=1e-13
    @test first.objective ≈ reversed.objective atol=1e-10 rtol=1e-10
    @test first.prior_targets == reversed.prior_targets
end
