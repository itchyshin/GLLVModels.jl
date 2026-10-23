# Same original delta fixtures, now with the frozen R per-trait dispersion map.
using GLLVModels, RCall, Test, Random, SHA, TOML, LinearAlgebra
import Distributions
@assert realpath(Base.pkgdir(GLLVModels))==realpath(pwd())
isdefined(@__MODULE__, :fit_gllvmtmb_parity_delta) || include(joinpath(pwd(),"test/parity/parity_helpers.jl"))
function core070_delta_matched(family::Symbol; tight_r::Bool=false)
family in (:delta_lognormal,:delta_gamma) || error("unknown delta family")
gamma=family===:delta_gamma
fixture="test/parity/test_$(family)_parity.jl"
text=read(fixture,String)
# Execute the historical DGP and sanity assertions verbatim, before either fit.
start=findfirst("    Random.seed!",text).start
stop=findfirst("    jl_fit =",text).start-1
block=text[start:stop]
p,K,n,Y=include_string(@__MODULE__,"const $(gamma ? "_DG_SEED = 62" : "_DLN_SEED = 61")\n"*block*"\n(p,K,n,Y)\n",fixture)
@assert (p,K,n)==(5,1,130)
fitfun=gamma ? fit_delta_gamma_gllvm : fit_delta_lognormal_gllvm
native=fitfun(Y;K=K,predictor=:shared,disp_group=:species,hessian=:observed)
r=fit_gllvmtmb_parity_delta(Y,K;family=family)
@rput tight_r
R"""
 .delta_original_fit <- fit_r
 .delta_original_gradient <- max(abs(fit_r$tmb_obj$gr(fit_r$opt$par)))
 if (tight_r) {
   fit_r <- gllvmTMB(
     value ~ 0 + trait + latent(0 + trait | site,d=K,unique=FALSE),
     data=df_long,unit="site",trait="trait",family=fam_obj,
     control=gllvmTMBcontrol(n_init=1L,se=FALSE,start_from=.delta_original_fit,
       optArgs=list(control=list(rel.tol=1e-12,eval.max=2000,iter.max=1500))))
   stopifnot(identical(names(fit_r$opt$par),names(.delta_original_fit$opt$par)),
     identical(fit_r$tmb_obj$env$data,.delta_original_fit$tmb_obj$env$data),
     identical(fit_r$tmb_obj$env$map,.delta_original_fit$tmb_obj$env$map))
 }
 .delta_gradient <- as.numeric(fit_r$tmb_obj$gr(fit_r$opt$par))
 .delta_nfree <- length(fit_r$opt$par)
 .delta_parameters <- as.numeric(fit_r$opt$par)
"""
r=(logLik=rcopy(Float64,R"as.numeric(logLik(fit_r))"),objective=rcopy(Float64,R"as.numeric(fit_r$opt$objective)"),
 converged=rcopy(Bool,R"identical(as.integer(fit_r$opt$convergence),0L)"),
 disp_vec=rcopy(Vector{Float64},R"if (fam=='delta_gamma') as.numeric(fit_r$report$phi_gamma_delta) else as.numeric(fit_r$report$sigma_lognormal_delta)"))
scale=gamma ? native.α : native.σ
θ=vcat(native.βc,GLLVModels.pack_lambda(native.Λc),log.(scale))
function objective(v)
 β=v[1:p];Λ=GLLVModels.unpack_lambda(v[p+1:2p],p,K);d=exp.(v[2p+1:3p])
 f=gamma ? GLLVModels.delta_gamma_marginal_loglik_laplace : GLLVModels.delta_lognormal_marginal_loglik_laplace
 -f(Y,Λ,β,β,d;Λz=Λ,hessian=:observed,maxiter=100,tol=1e-9)
end
function fd(v,multiplier)
 [begin
 h=multiplier*cbrt(eps(Float64))*max(1.0,abs(v[j]));a=copy(v);b=copy(v)
 a[j]+=h;b[j]-=h;(objective(a)-objective(b))/(2h)
 end for j in eachindex(v)]
end
grad=fd(θ,1.0);grad2=fd(θ,2.0)
rgrad=rcopy(Vector{Float64},R".delta_gradient")
rpars=rcopy(Vector{Float64},R".delta_parameters")
report=Dict("tight_public_r_control"=>tight_r,"original_r_gradient_max"=>rcopy(Float64,R".delta_original_gradient"),"family"=>String(family),"seed"=>(gamma ? 62 : 61),"p"=>p,"K"=>K,"n"=>n,
 "fixture_sha256"=>bytes2hex(sha256(read(fixture))),"dgp_sha256"=>bytes2hex(sha256(block)),
 "data_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(Y)))),"predictor"=>"shared","dispersion"=>"species","hessian"=>"observed",
 "native_loglik"=>native.loglik,"r_loglik"=>r.logLik,"r_objective"=>r.objective,
 "native_converged"=>native.converged,"r_converged"=>r.converged,
 "native_gradient_max"=>maximum(abs,grad),"r_gradient_max"=>maximum(abs,rgrad),
 "native_fd_stability"=>maximum(abs.(grad-grad2)),"native_reevaluation_delta"=>abs(objective(θ)+native.loglik),
 "native_parameters"=>θ,"r_parameters"=>rpars,"native_dispersion"=>scale,"r_dispersion"=>r.disp_vec,
 "native_nfree"=>length(θ),"r_nfree"=>rcopy(Int,R".delta_nfree"),
 "loglik_delta"=>abs(native.loglik-r.logLik),"loglik_rtol"=>1e-6,"gradient_atol"=>1e-4)
mkpath("matched")
open("matched/$(family).toml","w") do io;TOML.print(io,report);end
println("DELTA_MATCHED_RESULT ",report)
@testset "Same-model $(family) required parity" begin
 @test native.converged
 @test r.converged
 @test native.predictor===:shared && native.disp_group===:species
 @test native.βz==native.βc
 @test length(scale)==length(r.disp_vec)==p
 @test length(θ)==report["r_nfree"]==15
 @test all(isfinite,θ) && all(isfinite,rpars)
 @test all(x->isfinite(x)&&x>0,scale) && all(x->isfinite(x)&&x>0,r.disp_vec)
 core070_record_values!("logLik"; julia=native.loglik, r=r.logLik, rtol=1e-6,
                        test="@test native.loglik≈r.logLik rtol=1e-6")
 @test native.loglik≈r.logLik rtol=1e-6
 @test r.logLik≈-r.objective rtol=0 atol=1e-10
 @test report["native_reevaluation_delta"]<=1e-8
 @test report["native_gradient_max"]<=1e-4
 @test report["native_fd_stability"]<=1e-4
 @test report["r_gradient_max"]<=1e-4
end
println("DELTA_MATCHED_PARITY_PASS ",family)

return report
end
if abspath(PROGRAM_FILE)==@__FILE__
 length(ARGS) in (1,2) || error("family [--tight-r] required")
 length(ARGS)==1 || ARGS[2]=="--tight-r" || error("unknown control")
 core070_delta_matched(Symbol(ARGS[1]);tight_r=length(ARGS)==2)
end
