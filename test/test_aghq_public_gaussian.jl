using GLLVModels,Test,Random,LinearAlgebra,Distributions
@testset "GU public Gaussian AGHQ" begin
    @test hasfield(GLLVModels.GllvmFit,:integration)
    if hasfield(GLLVModels.GllvmFit,:integration)
        rng=MersenneTwister(714);p=3;n=36;K=1
        L=reshape([.8,.4,-.3],3,1)
        Y=L*randn(rng,K,n)+.7randn(rng,p,n)
        base=fit_gaussian_gllvm(Y;K=K)
        @test base.integration===nothing
        old=GLLVModels.GllvmFit(base.model,base.pars,base.logLik,base.n_iter,base.converged,base.optim_result,base.cputime)
        @test old.integration===nothing
        one=fit_gaussian_gllvm(Y;K=K,aghq=1)
        @test one.integration.actual==:laplace && one.integration.reason==:laplace_rule
        @test one.logLik==base.logLik && one.pars.θ_packed==base.pars.θ_packed
        f=fit_gaussian_gllvm(Y;K=K,aghq=3)
        @test f.integration.actual==:aghq && f.integration.k==3 && !f.integration.penalised
        @test f.converged && f.integration.mode_gradient_max<1e-7
        @test isempty(f.pars.β)
        exact_record=fit_gaussian_gllvm(Y;K=K,mask=trues(p,n),aghq=false)
        replay=GLLVModels._gaussian_record_refit(exact_record,simulate(exact_record,n;rng=MersenneTwister(97)))
        @test replay!==nothing
        if replay!==nothing
            @test replay.integration.requested===:off && replay.integration.actual===:laplace
        end
        allfixed=fit_gaussian_gllvm(Y;K=K,X=ones(p,n,2),β_fixed=[true,true],aghq=3)
        @test allfixed.pars.β==zeros(2) && length(allfixed.pars.θ_packed)==4
        @test abs(allfixed.logLik-f.logLik)<1e-6
        @test abs(f.logLik-base.logLik)<1e-6
        @test length(f.integration.result.starts)==2
        @test isdefined(GLLVModels,:_gaussian_record_ci)
        if isdefined(GLLVModels,:_gaussian_record_ci)
            ci=confint(f,Y)
            @test ci.objective==:aghq && ci.gradient_kind==:frozen_surrogate
            @test ci.pd_hessian && all(isfinite,ci.se)
            ad=GLLVModels._gaussian_record_ci(f,Y)
            probe=copy(ad.θ);probe[1]+=.2
            @test GLLVModels._confint_reconstruct_nll(f,Y,nothing,nothing)(probe)==ad.nll(probe)
            H=GLLVModels.ForwardDiff.hessian(ad.nll,ad.θ)
            @test maximum(abs.(H-GLLVModels._fd_hessian(ad.nll,ad.θ)))<1e-3
            @test Matrix(vcov(f,Y)) ≈ inv(H) atol=1e-6
            @test_throws ArgumentError confint(f,Y;objective=:laplace)
            @test_throws ArgumentError confint(f,Y.+.1)
            pr=confint(f,Y;method=:profile,parm="sigma_eps")
            @test pr.lower[1]<f.pars.σ_eps<pr.upper[1]
            named=profile_ci(f,"sigma_eps";y=Y)
            @test named.lower≈pr.lower[1] && named.upper≈pr.upper[1]
            boot=confint(f,Y;method=:bootstrap,n_boot=2,seed=13)
            @test size(boot.replicates)==(2,length(f.pars.θ_packed))
            @test length(boot.converged)==2 && boot.n_converged==sum(boot.converged)
            println("GU_BOOTSTRAP_ATTEMPTS ",repr((converged=boot.converged,replicates=boot.replicates)))
            @test all(b->boot.converged[b] || all(isnan,boot.replicates[b,:]),1:2)
            @test all(isnan,boot.lower) # functional smoke cannot support calibrated intervals
            bd=GLLVModels.bootstrap_ci_derived(f,fb->fb.integration.actual===:aghq ? fb.pars.σ_eps : NaN;y=Y,n_boot=2,seed=13)
            @test bd.n_valid>=1 && length(bd.replicates)==2
            packed=GLLVModels.bootstrap_ci_derived(f,t->exp(t[1]);y=Y,n_boot=2,seed=13)
            @test packed.replicates≈bd.replicates
            ll,ok,tc,_=GLLVModels._derived_refit_with_fixed(f,t->exp(t[1]),.9f.pars.σ_eps,Y,nothing,nothing)
            @test ok && abs(ll+ad.nll(tc))<1e-8
            @test GLLVModels._tw_sigma_from_hessian(f,Y,nothing,nothing)[1]≈vcov(f,Y)
            nb=bootstrap_ci(f;y=Y,n_boot=2,seed=13,parms="sigma_eps")
            @test size(nb.replicates)==(2,1) && nb.estimate≈[f.pars.σ_eps] # raw scale (#156); replicates stay packed
            @test_throws ArgumentError bootstrap_ci(f;y=Y,n_sites=n+1,n_boot=2)
        end
        @test f.integration.input_digest==GLLVModels._aghq_data_digest(f.integration.data)
        @test getLV(f,Y;rotate=false) ≈ getLV(base,Y;rotate=false) atol=1e-5
        @test predict(f,Y) ≈ predict(base,Y) atol=1e-5
        @test all(iszero,getLV(f,Y;component=:mean))
        @test_throws DimensionMismatch getLV(f,Y[1:2,:];component=:mean)
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,aghq=0)
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,aghq=3,aghq_control=(n_adapt=0,))
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,aghq=3,aghq_control=(ridge=1e-8,))
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,aghq=3,hessian=:fisher)
        # The public route estimates trait intercepts (2026-09-27-gaussian-trait-intercepts.md).
        gen=fit_gllvm(Y;family=Normal(),K=K,aghq=3)
        genX=fit_gaussian_gllvm(Y;K=K,aghq=3,X=GLLVModels._trait_intercept_design(p,n))
        @test gen.integration.actual==:aghq && gen.logLik≈genX.logLik && length(gen.pars.β)==p
        form=gllvm(@formula(y~1),Y,(site=collect(1:n),);family=Normal(),K=K,aghq=3)
        @test form.integration.actual==:aghq
        @test occursin("AGHQ",sprint(show,MIME("text/plain"),f))
        @test_logs (:warn,r"retained exact") begin
            fallback=fit_gaussian_gllvm(Y;K=K,K_W=1,aghq=3,iterations=2)
            @test fallback.integration.actual==:laplace && fallback.integration.reason==:other_random_blocks
        end
        @test_logs (:warn,r"retained exact") begin
            auto=fit_gaussian_gllvm(randn(MersenneTwister(76),20,24);K=1,aghq=:auto,iterations=2)
            @test auto.integration.actual==:laplace && auto.integration.reason==:auto_trait_cutoff
        end
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,K_W=1,aghq=3,mask=trues(p,n))
        X=[j==1 ? 1. : sin(t+s)/3 for t in 1:p,s in 1:n,j in 1:2]
        off=[(t-s/n)/10 for t in 1:p,s in 1:n]
        Yx=Y .+ .4 .+off
        fx=fit_gaussian_gllvm(Yx;K=K,aghq=3,X=X,β_fixed=[false,true],offset=off)
        @test fx.pars.β[2]==0 && length(fx.pars.θ_packed)==5
        @test length(coef(fx))==2
        @test GLLVModels._communality_packed(fx.pars.θ_packed,GLLVModels._derived_spec(fx),1)≈communality(fx)[1]
        @test size(predict(fx,Yx))==size(Y)
        @test size(simulate(fx,n;rng=MersenneTwister(2)))==size(Y)
        @test_throws ArgumentError simulate(fx,n+1)
        ym=Matrix{Union{Missing,Float64}}(Yx);ym[1,2]=missing
        mask=trues(p,n);mask[2,3]=false
        fm=fit_gaussian_gllvm(ym;K=K,aghq=3,X=X,β_fixed=[false,true],offset=off,mask=mask)
        @test fm.integration.actual==:aghq && isfinite(fm.logLik)
        @test fm.integration.data.mask[1,2]==false && fm.integration.data.mask[2,3]==false
        @test size(predict(fm,ym))==size(Y)
        @test isnan(residuals(fm,ym)[1,2]) && isnan(residuals(fm,ym)[2,3])
        q,same=GLLVModels._gaussian_record_problem(fm,ym;require_identity=true)
        @test same && abs(q.objective(fm.pars.θ_packed,fm.integration.caches)+fm.logLik)<1e-9
        changed=copy(ym);changed[1,1]+=1
        @test_throws ArgumentError GLLVModels._gaussian_record_problem(fm,changed;require_identity=true)
        @test_throws ArgumentError predict(fm,changed)
        @test size(predict(fm,changed;X=X,offset=off))==size(Y)
        @test_throws ArgumentError fit_gaussian_gllvm(Y;K=K,aghq=3,X=fill(NaN,p,n,1))
        Yx[1,1]=99;X[1,1,1]=99;off[1,1]=99;mask[:].=false
        @test fm.integration.data.responses[1,1]!=99
        @test fm.integration.data.design[1,1,1]!=99 && fm.integration.data.offset[1,1]!=99
        @test sum(fm.integration.data.mask)==p*n-2
    end
end
