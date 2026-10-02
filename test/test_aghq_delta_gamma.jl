using GLLVModels,Test,LinearAlgebra,Random,Distributions
# Delta-Gamma: one latent z per site drives both parts (shared predictor = gllvmTMB's model).
function _dg_draw(rng,ηz,ηc,α)
    rand(rng)<inv(1+exp(-ηz)) ? rand(rng,Gamma(α,exp(ηc)/α)) : 0.0
end
@testset "Delta-Gamma AGHQ" begin
    rng=MersenneTwister(20261101)
    p,n=3,15
    β0=[0.3,0.0,0.5];λ0=[0.8,-0.6,0.5];α0=[2.0,3.0,1.5]
    z0=randn(rng,n)
    Y=[_dg_draw(rng,β0[t]+λ0[t]*z0[s],β0[t]+λ0[t]*z0[s],α0[t]) for t in 1:p,s in 1:n]
    @test any(Y.==0) && any(Y.>0)
    K=1;rr=GLLVModels.rr_theta_len(p,K)
    # independent brute-force integral, using the package's own two-part density (_tp_pieces)
    function brute(ηzf,ηcf,α)
        zs=range(-9,9;length=1801);h=step(zs)
        sum(begin
            vals=[exp(-z^2/2)/sqrt(2pi)*exp(sum(GLLVModels._tp_pieces(GLLVModels.DeltaGamma(α[t]),Y[t,s],ηzf(t,z),ηcf(t,z))[5] for t in 1:p)) for z in zs]
            log(sum(vals)*h)
        end for s in 1:n)
    end
    @testset "objective vs brute force and Laplace (shared predictor)" begin
        theta=vcat(β0,GLLVModels.pack_lambda(reshape(λ0,p,1)),log.(α0))
        q=GLLVModels.aghq_delta_gamma_problem(Y,K;k=21,predictor=:shared)
        @test q.nparams==length(theta)
        agh=-q.objective(theta,q.adapt(theta))
        eta=(t,z)->β0[t]+λ0[t]*z
        @test agh ≈ brute(eta,eta,α0) atol=1e-8
        q1=GLLVModels.aghq_delta_gamma_problem(Y,K;k=1,predictor=:shared)
        lap=GLLVModels.delta_gamma_marginal_loglik_laplace(Y,reshape(λ0,p,1),β0,β0,α0;
            Λz=reshape(λ0,p,1),hessian=:observed)
        # k = 1 is the Laplace rule on the same observed-information curvature
        @test -q1.objective(theta,q1.adapt(theta)) ≈ lap atol=1e-8
        g=GLLVModels.ForwardDiff.gradient(t->q.objective(t,q.adapt(theta)),theta)
        @test all(isfinite,g)
    end
    @testset "objective vs brute force and Laplace (separate predictor, shared alpha)" begin
        βz0=[0.2,-0.3,0.4]
        theta=vcat(βz0,β0,GLLVModels.pack_lambda(reshape(λ0,p,1)),[log(2.0)])
        q=GLLVModels.aghq_delta_gamma_problem(Y,K;k=21,predictor=:separate,disp_group=:shared)
        @test q.nparams==length(theta)
        agh=-q.objective(theta,q.adapt(theta))
        @test agh ≈ brute((t,z)->βz0[t],(t,z)->β0[t]+λ0[t]*z,fill(2.0,p)) atol=1e-8
        q1=GLLVModels.aghq_delta_gamma_problem(Y,K;k=1,predictor=:separate,disp_group=:shared)
        lap=GLLVModels.delta_gamma_marginal_loglik_laplace(Y,reshape(λ0,p,1),βz0,β0,2.0;hessian=:observed)
        @test -q1.objective(theta,q1.adapt(theta)) ≈ lap atol=1e-8
    end
    @testset "routes and policy" begin
        # Route identities use a capped Laplace warm start (`iterations`, passed on every
        # path); two converged AGHQ fits are made.
        fam=GLLVModels.DeltaGamma();kw=(predictor=:shared,iterations=3)
        base=fit_delta_gamma_gllvm(Y;K=1,disp_group=:species,kw...)
        k1=fit_gllvm(Y;family=fam,K=1,aghq=1,kw...)
        @test k1 isa GLLVModels.DeltaGammaAGHQFit
        @test abs(k1.loglik-base.loglik)<1e-8 && k1.βc==base.βc && k1.integration.actual===:laplace
        @test k1.integration.reason===:laplace_rule
        plain=fit_gllvm(Y;family=fam,K=1,kw...)
        @test plain isa GLLVModels.DeltaGammaFit && plain.loglik==base.loglik && plain.βc==base.βc
        off=fit_gllvm(Y;family=fam,K=1,aghq=false,kw...)
        @test off isa GLLVModels.DeltaGammaFit && off.loglik==base.loglik
        lap=fit_gllvm(Y;family=fam,K=1,aghq=1,predictor=:shared)
        a=fit_gllvm(Y;family=fam,K=1,aghq=:auto,aghq_control=(multistart=false,),predictor=:shared)
        @test a.integration.actual===:aghq && a.integration.k==5 && a.integration.node_count==5
        @test length(a.theta_packed)==p+rr+p && a.predictor===:shared && a.βz==a.βc
        @test length(a.α)==p && all(a.α.>0)
        @test abs(a.loglik-lap.loglik)<1.0      # Laplace and AGHQ agree to a small approximation error
        # separate predictor, shared dispersion: parameter count and route only
        s=GLLVModels.fit_delta_gamma_gllvm_aghq(Y;K=1,disp_group=:shared,aghq=3,aghq_control=(multistart=false,),iterations=3)
        @test s.predictor===:separate && s.α isa Float64 && s.integration.k==3 &&
              length(s.theta_packed)==2p+rr+1
        # auto cutoff: declined at 20 traits, honoured for an explicit k
        Yw=vcat(Y,Y,Y,Y,Y,Y,Y)   # 21 traits
        @test_logs (:warn,r"AGHQ.*Laplace") begin
            d=GLLVModels.fit_delta_gamma_gllvm_aghq(Yw;K=1,aghq=:auto,iterations=1)
            @test d.integration.actual===:laplace && d.integration.reason===:auto_trait_cutoff
        end
        @test_throws ArgumentError GLLVModels.fit_delta_gamma_gllvm_aghq(Y;K=1,aghq=3,hessian=:fisher)
        @test_throws ArgumentError GLLVModels.fit_delta_gamma_gllvm_aghq(Y;K=1,aghq=false)
        @test_throws ArgumentError GLLVModels.fit_delta_gamma_gllvm_aghq(Y;K=1,aghq=0)
        @test_throws ArgumentError GLLVModels.fit_delta_gamma_gllvm_aghq(Y;K=1,aghq=3,predictor=:bad)
        @test_throws DimensionMismatch GLLVModels.aghq_delta_gamma_problem(Y,1;k=3,offset=zeros(1,1))
        @test_throws ArgumentError GLLVModels.aghq_delta_gamma_problem(Y,1;k=3,disp_group=:bad)
        @test_throws ArgumentError GLLVModels.aghq_delta_gamma_problem(-Y,1;k=3)
        @test_throws ArgumentError GLLVModels.aghq_delta_gamma_problem(Y,1;k=0)
    end
end
