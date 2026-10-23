# Original ordinary NB2 target; stronger acceptance, unchanged fits.
using GLLVModels,RCall,Test,Random,SHA,TOML,LinearAlgebra
startswith(lowercase(readchomp(`hostname`)),"totoro") || error("Totoro only")
get(ENV,"CORE070_PARITY_REQUIRED","")=="1" || error("required mode missing")
@assert realpath(Base.pkgdir(GLLVModels))==realpath(pwd())
ispath("health") && error("fresh output required")
include(joinpath(pwd(),"test/parity/parity_helpers.jl"))
fixture="test/parity/test_negbin_parity.jl";source=read(fixture,String)
helpers=source[findfirst("function _rand_poisson",source).start:findfirst("@testset \"NB2 GLLVModels",source).start-1]
dgp=source[findfirst("    Random.seed!(45)",source).start:findfirst("    jl_fit =",source).start-1]
include_string(Main,helpers*dgp,fixture)
mkpath("health")
datahash=bytes2hex(sha256(reinterpret(UInt8,vec(Float64.(Y)))))
open(io->TOML.print(io,Dict("data_sha256"=>datahash,"Y_column_major"=>vec(Y),"p"=>p,"n"=>n,"K"=>K,"seed"=>45)),"health/data.toml","w")
native=fit_gllvm(Y;family=GLLVModels.NegativeBinomial(),K=K,disp_group=:species,g_tol=1e-7,iterations=800)
r=fit_gllvmtmb_parity_loglik(Y,K;family=:negbinomial)
R"""
r_obj <- fit_r$tmb_obj
r_objective <- as.numeric(r_obj$fn(fit_r$opt$par))
r_gradient <- as.numeric(r_obj$gr(fit_r$opt$par))
r_full <- r_obj$env$last.par
r_par <- r_obj$env$parList(x=fit_r$opt$par,par=r_full)
r_report <- r_obj$report(r_full)
r_beta <- as.numeric(r_par$b_fix)
r_lambda <- as.matrix(r_report$Lambda_B)
r_disp <- exp(as.numeric(r_par$log_phi_nbinom2))
saveRDS(list(opt=fit_r$opt,gradient=r_gradient,parameters=r_par,report=r_report,
    data=r_obj$env$data,objective=r_objective),"health/whole-fit.rds")
"""
rr=GLLVModels.rr_theta_len(p,K)
theta=vcat(native.β,GLLVModels.pack_lambda(native.Λ),log.(native.r_group))
function objective(v)
 -GLLVModels.nb_grouped_marginal_loglik_laplace(Y,GLLVModels.unpack_lambda(v[p+1:p+rr],p,K),
    v[1:p],exp.(v[p+rr+1:end]);hessian=:observed,maxiter=100,tol=1e-9)
end
function fd(v,m)
 [begin h=m*cbrt(eps(Float64))*max(1.0,abs(v[j]));a=copy(v);b=copy(v);a[j]+=h;b[j]-=h
  (objective(a)-objective(b))/(2h)
  end for j in eachindex(v)]
end
g=fd(theta,1.0);g2=fd(theta,2.0)
rtheta=rcopy(Vector{Float64},R"as.numeric(fit_r$opt$par)")
rlambda=rcopy(Matrix{Float64},R"r_lambda");rbeta=rcopy(Vector{Float64},R"r_beta");rdisp=rcopy(Vector{Float64},R"r_disp")
rnative=vcat(rbeta,GLLVModels.pack_lambda(rlambda),log.(rdisp))
rg=rcopy(Vector{Float64},R"r_gradient")
report=Dict("source"=>_core070_source_pin!(),"data_sha256"=>datahash,
 "fixture_sha256"=>bytes2hex(sha256(read(fixture))),"dgp_sha256"=>bytes2hex(sha256(helpers*dgp)),
 "native_converged"=>native.converged,"native_type"=>string(typeof(native)),"hessian"=>string(native.hessian),
 "native_parameters"=>theta,"native_gradient"=>g,"native_gradient_double_step"=>g2,
 "native_loglik"=>native.loglik,"native_gradient_max"=>maximum(abs,g),"fd_stability"=>maximum(abs.(g-g2)),
 "native_objective_delta"=>abs(objective(theta)+native.loglik),"native_r"=>native.r_group,
 "r_loglik"=>r.logLik,"r_cached_objective"=>r.objective,"r_objective"=>rcopy(Float64,R"r_objective"),
 "r_code"=>rcopy(Int,R"as.integer(fit_r$opt$convergence)"),"r_message"=>rcopy(String,R"as.character(fit_r$opt$message)"),
 "r_gradient"=>rg,"r_gradient_max"=>maximum(abs,rg),"r_parameters"=>rtheta,
 "r_native_parameters"=>rnative,"r_packing_delta"=>maximum(abs.(rtheta-rnative)),
 "r_dispersion"=>rdisp,"r_loadings_column_major"=>vec(rlambda),
 "loglik_delta"=>abs(native.loglik-r.logLik),"native_nfree"=>length(theta),"r_nfree"=>length(rtheta),
 "samepoint_native_nll"=>objective(rnative))
report["samepoint_delta"]=report["samepoint_native_nll"]-report["r_objective"]
open(io->TOML.print(io,report),"health/result.toml","w")
println("ORIGINAL_NB2_METRICS ",report)
@testset "Original NB2 required numerical health" begin
 @test native isa NBGroupedFit
 @test native.hessian==:observed
 @test native.converged
 @test report["r_code"]==0
 @test report["native_nfree"]==report["r_nfree"]==19
 @test report["r_packing_delta"]<=1e-12
 @test all(isfinite,theta) && all(isfinite,rtheta)
 @test maximum(abs,g)<=1e-4
 @test report["fd_stability"]<=1e-4
 @test maximum(abs,rg)<=1e-4
 @test report["native_objective_delta"]<=1e-8
 @test abs(report["r_objective"]+r.logLik)<=1e-8
 @test abs(report["samepoint_delta"])<=1e-6
 @test native.loglik≈r.logLik rtol=1e-6
end
println("ORIGINAL_NB2_HEALTH_PASS")
