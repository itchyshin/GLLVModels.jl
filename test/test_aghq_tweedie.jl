using GLLVModels,Test,LinearAlgebra,Random,Distributions
# Compound Poisson-Gamma draw with mean mu, dispersion phi, power pw (1 < pw < 2).
function _tw_draw(rng,mu,phi,pw)
    lam=mu^(2-pw)/(phi*(2-pw));shape=(2-pw)/(pw-1);scale=phi*(pw-1)*mu^(pw-1)
    N=rand(rng,Poisson(lam))
    return N==0 ? 0.0 : rand(rng,Gamma(N*shape,scale))
end
@testset "Tweedie AGHQ" begin
    rng=MersenneTwister(20261003)
    p,n=4,60
    β0=[0.3,0.0,0.5,0.1];λ0=[0.8,-0.6,0.5,-0.4]
    φ0=[1.2,0.8,1.5,1.0];pw0=[1.4,1.6,1.5,1.3]
    z0=randn(rng,n)
    Y=[_tw_draw(rng,exp(β0[t]+λ0[t]*z0[s]),φ0[t],pw0[t]) for t in 1:p,s in 1:n]
    @test any(Y.==0) && any(Y.>0)
    K=1;rr=GLLVModels.rr_theta_len(p,K)
    @testset "series derivative and value" begin
        for (y,φ,pw) in ((0.7,1.3,1.4),(5.0,0.8,1.7),(0.05,2.0,1.2))
            @test GLLVModels._tweedie_logA_ad(y,φ,pw) ≈ GLLVModels._tweedie_logA(y,φ,pw) atol=1e-12
            g=GLLVModels.ForwardDiff.gradient(x->GLLVModels._tweedie_logA_ad(y,x[1],x[2]),[φ,pw])
            h=1e-6;f=(a,b)->GLLVModels._tweedie_logA(y,a,b)
            fd=[(f(φ+h,pw)-f(φ-h,pw))/2h,(f(φ,pw+h)-f(φ,pw-h))/2h]
            @test maximum(abs,g-fd)<1e-7
        end
    end
    @testset "objective vs brute force and Laplace (species power)" begin
        xi0=GLLVModels._tweedie_xi.(pw0)
        theta=vcat(β0,GLLVModels.pack_lambda(reshape(λ0,p,1)),log.(φ0),xi0)
        q=GLLVModels.aghq_tweedie_problem(Y,K;k=21,power_group=:species)
        @test q.nparams==length(theta)
        agh=-q.objective(theta,q.adapt(theta))
        zs=range(-12,12;length=4801);h=step(zs)
        brute=sum(begin
            vals=[exp(-z^2/2)/sqrt(2pi)*exp(sum(GLLVModels.tweedie_logpdf(Y[t,s],exp(β0[t]+λ0[t]*z),φ0[t],pw0[t]) for t in 1:p)) for z in zs]
            log(sum(vals)*h)
        end for s in 1:n)
        @test agh ≈ brute atol=1e-8
        q1=GLLVModels.aghq_tweedie_problem(Y,K;k=1,power_group=:species)
        lap=GLLVModels.tweedie_grouped_marginal_loglik_laplace(Y,reshape(λ0,p,1),β0,φ0,pw0)
        # k = 1 is the Laplace rule on the same observed-information curvature
        @test -q1.objective(theta,q1.adapt(theta)) ≈ lap atol=1e-8
        # objective differentiates in (phi, power) as well as (beta, Lambda)
        g=GLLVModels.ForwardDiff.gradient(t->q.objective(t,q.adapt(theta)),theta)
        @test all(isfinite,g)
        # fixed and shared power give the matching parameter counts
        @test GLLVModels.aghq_tweedie_problem(Y,K;k=3,power=1.5).nparams==p+rr+p
        @test GLLVModels.aghq_tweedie_problem(Y,K;k=3).nparams==p+rr+p+1
        @test GLLVModels.aghq_tweedie_problem(Y,K;k=3,group=[1,1,2,2]).nparams==p+rr+2+1
    end
    @testset "routes and policy" begin
        # The Laplace fit is the expensive part (finite-difference gradient), so the route
        # identities use a capped warm start (`iterations`, passed to the Laplace fitter on
        # every path) and a single converged comparison is made for the AGHQ result.
        fam=GLLVModels.TweedieED(1.0,1.5);kw=(power_group=:species,iterations=10)
        base=GLLVModels.fit_tweedie_gllvm_grouped(Y;K=1,group=1:p,kw...)
        k1=fit_gllvm(Y;family=fam,K=1,disp_group=:species,aghq=1,kw...)
        @test k1 isa GLLVModels.TweedieGroupedAGHQFit
        @test abs(k1.loglik-base.loglik)<1e-8 && k1.β==base.β && k1.integration.actual===:laplace
        @test k1.integration.reason===:laplace_rule
        plain=fit_gllvm(Y;family=fam,K=1,disp_group=:species,kw...)
        @test plain isa GLLVModels.TweediePerTraitPowerFit && plain.loglik==base.loglik && plain.β==base.β
        off=fit_gllvm(Y;family=fam,K=1,disp_group=:species,aghq=false,kw...)
        @test off isa GLLVModels.TweediePerTraitPowerFit && off.loglik==base.loglik
        lap=fit_gllvm(Y;family=fam,K=1,disp_group=:species,aghq=1,power_group=:species)
        a=fit_gllvm(Y;family=fam,K=1,disp_group=:species,aghq=:auto,aghq_control=(multistart=false,),power_group=:species)
        @test a.integration.actual===:aghq && a.integration.k==9 && a.integration.node_count==9
        @test length(a.theta_packed)==p+rr+p+p && a.power_mode===:species && length(a.power)==p
        @test all(1 .< a.power .< 2) && all(a.φ.>0)
        @test abs(a.loglik-lap.loglik)<1.0      # Laplace and AGHQ agree to a small approximation error
        # shared and fixed power
        s=GLLVModels.fit_tweedie_gllvm_grouped_aghq(Y;K=1,group=1:p,aghq=3,aghq_control=(multistart=false,),iterations=10)
        @test s.power_mode===:shared && length(unique(s.power))==1 && s.integration.k==3
        f=GLLVModels.fit_tweedie_gllvm_grouped_aghq(Y;K=1,group=1:p,power=1.5,aghq=3,aghq_control=(multistart=false,),iterations=10)
        @test f.power_mode===:fixed && all(f.power.==1.5) && length(f.theta_packed)==p+rr+p
        # auto cutoff: declined at 20 traits, honoured for an explicit k
        Yw=vcat(Y,Y,Y,Y,Y)
        @test_logs (:warn,r"AGHQ.*Laplace") begin
            d=GLLVModels.fit_tweedie_gllvm_grouped_aghq(Yw;K=1,aghq=:auto,iterations=2)
            @test d.integration.actual===:laplace && d.integration.reason===:auto_trait_cutoff
        end
        @test_throws ArgumentError GLLVModels.fit_tweedie_gllvm_grouped_aghq(Y;K=1,aghq=3,hessian=:fisher)
        @test_throws ArgumentError GLLVModels.fit_tweedie_gllvm_grouped_aghq(Y;K=1,aghq=false)
        @test_throws ArgumentError GLLVModels.fit_tweedie_gllvm_grouped_aghq(Y;K=1,aghq=0)
        @test_throws DimensionMismatch GLLVModels.aghq_tweedie_problem(Y,1;k=3,mask=trues(1,1))
        @test_throws ArgumentError GLLVModels.aghq_tweedie_problem(Y,1;k=3,power=2.5)
        @test_throws ArgumentError GLLVModels.aghq_tweedie_problem(Y,1;k=3,power_group=:bad)
        @test_throws ArgumentError GLLVModels.aghq_tweedie_problem(-Y,1;k=3)
        @test_throws ArgumentError GLLVModels.aghq_tweedie_problem(Y,1;k=0)
    end
end
