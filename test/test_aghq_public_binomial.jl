using GLLVModels,Test,Random,LinearAlgebra,Distributions
@testset "BU public binomial AGHQ" begin
    @test hasfield(GLLVModels.BinomialFit,:integration)
    if hasfield(GLLVModels.BinomialFit,:integration)
        Y=[0 1 2 3 1 2 0 1;1 2 3 1 0 2 1 3];N=fill(3,size(Y))
        base=fit_binomial_gllvm(Y;K=1,N=N,iterations=2)
        @test base.integration===nothing
        one=fit_binomial_gllvm(Y;K=1,N=N,iterations=2,aghq=1)
        @test one.loglik==base.loglik && one.β==base.β && one.Λ==base.Λ
        @test one.integration.actual===:laplace && one.integration.reason===:laplace_rule
        old=GLLVModels.BinomialFit(base.β,base.Λ,base.link,base.loglik,base.converged,base.iterations,
            base.alpha_lv,base.theta_packed,base.hessian,base.saturation)
        @test old.integration===nothing
        @test_throws ArgumentError fit_binomial_gllvm(Y;K=1,N=N,aghq=0)
        @test_throws ArgumentError fit_binomial_gllvm(Y;K=1,N=N,aghq=3,aghq_control=(typo=1,))
        @test_throws ArgumentError fit_binomial_gllvm(Y;K=1,N=N,aghq=3,hessian=:fisher)
        fallback=@test_logs (:warn,r"AGHQ.*Laplace") fit_binomial_gllvm(Y;K=1,N=N,
            X_lv=reshape(collect(1.:8.),8,1),aghq=3,iterations=2)
        @test fallback.integration.reason===:predictor_latents
        for link in (LogitLink(),ProbitLink(),CLogLogLink())
            off=fill(.2,size(Y));mask=trues(size(Y));mask[1,2]=false;trials=copy(N)
            fit=fit_binomial_gllvm(Y;K=1,N=trials,link=link,offset=off,mask=mask,aghq=3,
                iterations=12,aghq_control=(n_adapt=12,multistart=false))
            @test fit.integration.actual===:aghq && fit.integration.k==3
            @test !fit.integration.penalised && fit.saturation===nothing
            @test fit.converged==fit.integration.result.selected.converged
            @test fit.theta_packed==fit.integration.result.selected.parameters
            @test fit.integration.base_controls.iterations==12
            replay=GLLVModels._binomial_aghq_refit_kwargs(fit)
            @test replay.N==N && replay.N!==trials && replay.aghq==3
            @test replay.aghq_control==fit.integration.controls
            ad=GLLVModels._family_ci(fit,Y;objective=:fit)
            @test ad.nll(ad.θ) ≈ -fit.loglik atol=1e-10
            @test_throws ArgumentError GLLVModels._family_ci(fit,Y;N=N.+1)
            @test_throws ArgumentError confint(fit,Y;objective=:laplace)
            ci=confint(fit,Y;parm="beta[1]")
            @test ci.objective===:aghq && ci.gradient_kind===:frozen_surrogate
            if link isa LogitLink
                H=GLLVModels.ForwardDiff.hessian(ad.nll,ad.θ)
                @test maximum(abs.(H-GLLVModels._fd_hessian(ad.nll,ad.θ)))<1e-3
                profile=confint(fit,Y;method=:profile,parm="beta[1]",profile_iterations=20)
                @test profile.objective===:aghq && profile.gradient_kind===:frozen_surrogate
                @test length(profile.status)==1
                boot=confint(fit,Y;method=:bootstrap,parm="beta[1]",n_boot=2,seed=831)
                @test boot.objective===:aghq && boot.gradient_kind===:frozen_surrogate
                @test size(boot.replicates)==(2,length(ad.θ))
                @test count(boot.converged)==boot.n_converged
                @test all(isnan,boot.replicates[.!boot.converged,:])
                @test all(isfinite,boot.replicates[boot.converged,:])
                @test all(isnan,boot.lower) && all(isnan,boot.upper)
            end
            z=getLV(fit,Y;rotate=false)
            @test z==permutedims(reduce(hcat,(a.mode for a in fit.integration.caches)))
            eta=predict(fit,Y;type=:link)
            @test eta ≈ fit.β .+ fit.Λ*z' .+ off
            prob=predict(fit,Y)
            @test all(0 .<=prob.<=1)
            @test prob ≈ (link isa CLogLogLink ? -expm1.(-exp.(eta)) : clamp.(GLLVModels.linkinv.(Ref(link),eta),1e-12,1-1e-12))
            res=residuals(fit,Y;type=:pearson)
            @test isnan(res[1,2])
            @test res[1,1] ≈ (Y[1,1]-N[1,1]*prob[1,1])/sqrt(N[1,1]*prob[1,1]*(1-prob[1,1]))
            masked=Matrix{Union{Missing,Int}}(Y);masked[1,2]=missing
            @test predict(fit,masked)==prob
            trials[1,1]=10;off[1,1]=10;mask[1,1]=false
            @test predict(fit,Y)==prob
            @test occursin("AGHQ",sprint(show,MIME("text/plain"),fit))
            sim=simulate(fit,8;rng=MersenneTwister(1))
            @test size(sim)==size(Y) && all(0 .<=sim.<=N)
            @test_throws ArgumentError simulate(fit,9)
            @test_throws DimensionMismatch getLV(fit,Y[:,1:4];component=:mean)
            @test_throws DimensionMismatch getLV(fit,Y;component=:mean,mask=trues(1,1))
            @test_throws DimensionMismatch getLV(fit,Y[1:1,:];N=N[1:1,:],offset=zeros(1,8),component=:mean)
            @test size(simulate(fit,9;N=fill(4,2,9),offset=zeros(2,9)))==(2,9)
            @test_throws ArgumentError predict(fit,Y[:,1:4])
            @test size(predict(fit,Y[:,1:4];N=N[:,1:4],offset=zeros(2,4)))==(2,4)
        end
        multi=fit_binomial_gllvm(Y;K=1,N=N,aghq=3,iterations=2,aghq_control=(n_adapt=1,))
        @test hasproperty(multi.integration.result,:starts)
        if hasproperty(multi.integration.result,:starts)
            expected=GLLVModels.linkfun.(Ref(LogitLink()),clamp.(vec(sum(Y;dims=2))./size(Y,2),1/32,31/32))
            @test multi.integration.result.starts[2][1:2] ≈ expected
            @test all(==(.3),multi.integration.result.starts[2][3:end])
        end
        kwargs=(K=1,N=N,aghq=3,iterations=2,aghq_control=(n_adapt=2,multistart=false))
        generic=fit_gllvm(Y;family=Binomial(),kwargs...)
        formula=gllvm(@formula(y~1),Y,(site=collect(1:8),);family=Binomial(),kwargs...)
        @test generic.integration.actual===formula.integration.actual===:aghq
        @test generic.loglik==formula.loglik
    end
end
