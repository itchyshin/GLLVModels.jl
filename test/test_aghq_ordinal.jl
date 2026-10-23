using GLLVModels,Test,LinearAlgebra,Random,Distributions
@testset "Ordinal AGHQ" begin
    rng=MersenneTwister(20261002)
    p,n=4,60
    β0=[0.2,-0.3,0.4,0.0];λ0=[0.9,-0.7,0.6,-0.5]
    τ0=[0.0 0.8 1.7;0.0 0.6 1.5;0.0 1.0 2.0;0.0 0.7 1.6]   # C = 4 for every trait
    C0=fill(4,p)
    psi0=Float64[log(τ0[t,c]-τ0[t,c-1]) for t in 1:p for c in 2:3]
    z0=randn(rng,n)
    function draw(link,t,z)
        η=β0[t]+λ0[t]*z
        F=x->link isa ProbitLink ? cdf(Normal(),x) : 1/(1+exp(-x))
        u=rand(rng);cum=[F(τ0[t,c]-η) for c in 1:3]
        return 1+count(<(u),cum)
    end
    Y=[draw(ProbitLink(),t,z0[s]) for t in 1:p,s in 1:n]
    Y[:,1].=1;Y[:,2].=4        # make sure every trait shows levels 1 and 4
    K=1
    for link in (ProbitLink(),LogitLink())
        F=x->link isa ProbitLink ? cdf(Normal(),x) : 1/(1+exp(-x))
        @testset "objective vs brute force and Laplace ($(nameof(typeof(link))))" begin
            theta=vcat(β0,GLLVModels.pack_lambda(reshape(λ0,p,1)),psi0)
            q=GLLVModels.aghq_ordinal_problem(Y,K;k=21,link=link)
            @test q.C==C0 && q.nparams==length(theta)
            caches=q.adapt(theta)
            agh=-q.objective(theta,caches)
            zs=range(-14,14;length=56001);h=step(zs)
            prob(t,c,z)=begin
                η=β0[t]+λ0[t]*z
                hi=c==4 ? 1.0 : F(τ0[t,c]-η)
                lo=c==1 ? 0.0 : F(τ0[t,c-1]-η)
                hi-lo
            end
            brute=sum(begin
                vals=[exp(-z^2/2)/sqrt(2pi)*prod(prob(t,Y[t,s],z) for t in 1:p) for z in zs]
                log(sum(vals)*h)
            end for s in 1:n)
            @test agh ≈ brute atol=1e-8
            q1=GLLVModels.aghq_ordinal_problem(Y,K;k=1,link=link)
            lap=GLLVModels.ordinal_marginal_loglik_laplace_pertrait(Y,reshape(λ0,p,1),β0,τ0,C0;link=link)
            @test -q1.objective(theta,q1.adapt(theta)) ≈ lap atol=1e-8
        end
    end
    @testset "routes and policy" begin
        kw=(link=ProbitLink(),)
        base=fit_ordinal_gllvm_pertrait(Y;K=1,kw...)
        k1=fit_gllvm(Y;family=Ordinal(),K=1,aghq=1,kw...)
        @test k1 isa GLLVModels.OrdinalPerTraitAGHQFit
        @test abs(k1.loglik-base.loglik)<1e-8 && k1.β==base.β && k1.integration.actual===:laplace
        @test k1.integration.reason===:laplace_rule
        plain=fit_gllvm(Y;family=Ordinal(),K=1,kw...)
        @test plain isa GLLVModels.OrdinalPerTraitFit && plain.loglik==base.loglik && plain.β==base.β
        off=fit_gllvm(Y;family=Ordinal(),K=1,aghq=false,kw...)
        @test off isa GLLVModels.OrdinalPerTraitFit && off.loglik==base.loglik
        a=fit_gllvm(Y;family=Ordinal(),K=1,aghq=:auto,aghq_control=(multistart=false,),kw...)
        @test a.integration.actual===:aghq && a.integration.k==9 && a.integration.node_count==9
        @test length(a.theta_packed)==p+GLLVModels.rr_theta_len(p,1)+sum(a.C.-2)
        @test a.C==base.C && size(a.τ)==size(base.τ) && all(a.τ[:,1].==0)
        # Laplace and AGHQ agree to the usual small approximation error
        @test abs(a.loglik-base.loglik)<0.5
        g=fit_gllvm(Y;family=ordinal_logit(),K=1,aghq=3,aghq_control=(multistart=false,))
        @test g.integration.k==3 && g.link isa LogitLink
        @test_throws ArgumentError fit_gllvm(Y;family=ordinal_logit(),K=1,aghq=3,link=ProbitLink())
        # auto cutoff: declined at 20 traits, honoured for an explicit k
        Yw=vcat(Y,Y,Y,Y,Y)
        @test_logs (:warn,r"AGHQ.*Laplace") begin
            d=GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Yw;K=1,aghq=:auto,iterations=2,kw...)
            @test d.integration.actual===:laplace && d.integration.reason===:auto_trait_cutoff
        end
        @test_throws ArgumentError GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y;K=1,aghq=3,hessian=:fisher)
        @test_throws ArgumentError GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y;K=1,aghq=false)
        @test_throws ArgumentError GLLVModels.fit_ordinal_gllvm_pertrait_aghq(Y;K=1,aghq=0)
        @test_throws DimensionMismatch GLLVModels.aghq_ordinal_problem(Y,1;k=3,mask=trues(1,1))
        @test_throws ArgumentError GLLVModels.aghq_ordinal_problem(Y,1;k=3,link=LogLink())
        @test_throws ArgumentError GLLVModels.aghq_ordinal_problem(Y,1;k=3,C=[3,4,4,4])
        @test_throws ArgumentError GLLVModels.aghq_ordinal_problem(Y,1;k=0)
    end
end
