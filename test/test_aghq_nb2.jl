using GLLVModels,Test,LinearAlgebra,Random,Distributions
@testset "NB2 AGHQ" begin
    rng=MersenneTwister(20261001)
    p,n=4,14
    β0=[0.3,0.6,0.1,0.5];λ0=[0.8,-0.6,0.5,-0.4];r0=[3.0,5.0,2.0,8.0]
    z0=randn(rng,n)
    Y=[rand(rng,NegativeBinomial(r0[t],r0[t]/(r0[t]+exp(β0[t]+λ0[t]*z0[s])))) for t in 1:p,s in 1:n]
    K=1
    @testset "objective vs brute force and Laplace" begin
        theta=vcat(β0,GLLVModels.pack_lambda(reshape(λ0,p,1)),log.(r0))
        q=GLLVModels.aghq_nb2_problem(Y,K;k=15)
        caches=q.adapt(theta)
        agh=-q.objective(theta,caches)
        zs=range(-14,14;length=56001);h=step(zs)
        brute=sum(begin
            vals=[exp(-z^2/2)/sqrt(2pi)*prod(pdf(NegativeBinomial(r0[t],r0[t]/(r0[t]+exp(β0[t]+λ0[t]*z))),Y[t,s]) for t in 1:p) for z in zs]
            log(sum(vals)*h)
        end for s in 1:n)
        @test agh ≈ brute atol=1e-8
        q1=GLLVModels.aghq_nb2_problem(Y,K;k=1)
        lap=GLLVModels.nb_grouped_marginal_loglik_laplace(Y,reshape(λ0,p,1),β0,r0;hessian=:observed)
        @test -q1.objective(theta,q1.adapt(theta)) ≈ lap atol=1e-8
    end
    @testset "routes and policy" begin
        base=fit_nb_gllvm_grouped(Y;K=1,group=collect(1:p))
        k1=fit_gllvm(Y;family=NegativeBinomial(),K=1,disp_group=:species,aghq=1)
        @test k1 isa GLLVModels.NBGroupedAGHQFit
        @test abs(k1.loglik-base.loglik)<1e-8 && k1.β==base.β && k1.integration.actual===:laplace
        @test k1.integration.reason===:laplace_rule
        plain=fit_gllvm(Y;family=NegativeBinomial(),K=1,disp_group=:species)
        @test plain isa GLLVModels.NBGroupedFit && plain.loglik==base.loglik
        a=fit_gllvm(Y;family=NegativeBinomial(),K=1,disp_group=:species,aghq=:auto,aghq_control=(multistart=false,))
        @test a.integration.actual===:aghq && a.integration.k==5 && a.integration.node_count==5
        @test length(a.r_group)==p && length(a.theta_packed)==p+GLLVModels.rr_theta_len(p,1)+p
        # Laplace and AGHQ agree to the usual small approximation error at k=5
        @test abs(a.loglik-base.loglik)<0.5
        g=fit_gllvm(Y;family=NegativeBinomial(),K=1,disp_group=[1,1,2,2],aghq=3,aghq_control=(multistart=false,))
        @test length(g.r_group)==2 && g.integration.k==3
        # auto cutoff: declined at 20 traits, honoured for an explicit k
        Yw=vcat(Y,Y,Y,Y,Y)
        @test_logs (:warn,r"AGHQ.*Laplace") begin
            d=GLLVModels.fit_nb_gllvm_grouped_aghq(Yw;K=1,group=collect(1:20),aghq=:auto,iterations=2)
            @test d.integration.actual===:laplace && d.integration.reason===:auto_trait_cutoff
        end
        @test_throws ArgumentError GLLVModels.fit_nb_gllvm_grouped_aghq(Y;K=1,aghq=3,hessian=:fisher)
        @test_throws ArgumentError GLLVModels.fit_nb_gllvm_grouped_aghq(Y;K=1,aghq=false)
        @test_throws DimensionMismatch GLLVModels.aghq_nb2_problem(Y,1;k=3,mask=trues(1,1))
    end
end
