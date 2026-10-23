using GLLVModels,RCall,Test,LinearAlgebra,SHA,TOML,Statistics
root=normpath(joinpath(@__DIR__,".."))
include(joinpath(root,"test/parity/parity_helpers.jl"));_parity_require_gllvmtmb!()
@testset "AG numerical prerequisites" begin
 include(joinpath(root,"test/test_aghq_gaussian.jl"))
 include(joinpath(root,"test/test_aghq_multistart.jl"))
 include(joinpath(root,"test/test_aghq_outer.jl"))
end
fixture="test/parity/test_gaussian_parity.jl";source=read(joinpath(root,fixture),String)
bytes2hex(sha256(source))=="a1610b34559eeb8c8d37f741432c1ed4603a02fe41747b3d6167b1654db2d884" || error("original Gaussian fixture changed")
a=findfirst("    Random.seed!(",source).start;b=findnext("    # ── 2.",source,a).start
dgp=source[a:b-1]
bytes2hex(sha256(dgp))=="80af1fb0516395facdb411a04c6b26c75b7ef3f7ba77fde6b7286d2faa97f83d" || error("original Gaussian DGP changed")
fixture_module=Module(:AGOriginal)
data=Base.include_string(fixture_module,"using Random,LinearAlgebra,Statistics\n"*dgp*"\n(Y=y,K=K,p=p,n=n)\n",fixture)
Y,K,p,n=data.Y,data.K,data.p,data.n;out=ENV["CORE070_AGHQ_PAIR_OUTPUT"]
open(io->TOML.print(io,Dict("responses"=>vec(Y),"p"=>p,"K"=>K,"n"=>n,
 "fixture_sha256"=>bytes2hex(sha256(source)),"dgp_sha256"=>bytes2hex(sha256(dgp)))),out*".fixture.toml","w")
println("AG_INPUT_SHA256 ",bytes2hex(sha256(read(out*".fixture.toml"))))
X=zeros(p,n,p)
for t in 1:p;X[t,:,t].=1;end
publicfit=fit_gaussian_gllvm(Y;K=K,X=X,aghq=5)
GLLVModels._is_gaussian_aghq(publicfit) || error("public Gaussian fitter did not run AGHQ")
multistart=publicfit.integration.result
start,alt=multistart.starts
problem,_=GLLVModels._gaussian_record_problem(publicfit,Y;require_identity=true)
multistart.usable || error("no usable Gaussian AGHQ start")
fit=multistart.selected
serialize_run(r)=Dict("parameters"=>r.parameters,"objective"=>r.objective,"usable"=>r.usable,
 "converged"=>r.converged,"stop_reason"=>string(r.stop_reason),"passes"=>r.passes,
 "gradient_max"=>r.frozen_gradient_max,"relative_gradient"=>r.relative_gradient,
 "trace"=>[Dict(string(k)=>(v===nothing ? "uncapped" : v) for (k,v) in pairs(row)) for row in r.trace])
open(io->TOML.print(io,Dict("winner"=>multistart.winner,"runs"=>serialize_run.(multistart.runs),
 "starts"=>[start,alt])),out*".julia.toml","w")
println("AG_JULIA_SHA256 ",bytes2hex(sha256(read(out*".julia.toml"))))
@rput Y K p n out
R"""
ag_df <- data.frame(site=factor(rep(seq_len(n),each=p)),
 trait=factor(rep(paste0('t',seq_len(p)),times=n),levels=paste0('t',seq_len(p))),value=as.vector(Y))
ag_warnings <- character()
ag_fit <- withCallingHandlers(gllvmTMB(value~0+trait+latent(0+trait|site,d=K,unique=FALSE),
 data=ag_df,unit='site',trait='trait',family=gaussian(),
 control=gllvmTMBcontrol(aghq=5,aghq_ridge=Inf,aghq_multistart=TRUE,aghq_n_adapt=400L,n_init=1L,se=FALSE)),
 warning=function(w) {ag_warnings <<- c(ag_warnings,conditionMessage(w));invokeRestart('muffleWarning')})
ag_obj <- ag_fit$tmb_obj
ag_value <- as.numeric(ag_obj$fn(ag_fit$opt$par));ag_grad <- as.numeric(ag_obj$gr(ag_fit$opt$par))
ag_full <- ag_obj$env$last.par;ag_params <- ag_obj$env$parList(x=ag_fit$opt$par,par=ag_full)
ag_report <- ag_obj$report(ag_full)
saveRDS(list(opt=ag_fit$opt,aghq=ag_fit$aghq,objective=ag_value,gradient=ag_grad,
 parameters=ag_params,report=ag_report,data=ag_obj$env$data,map=ag_obj$env$map,warnings=ag_warnings),paste0(out,'.rds'))
"""
println("AG_R_SHA256 ",bytes2hex(sha256(read(out*".rds"))))
r_beta=rcopy(Vector{Float64},R"as.numeric(ag_params$b_fix)")
r_loading=rcopy(Matrix{Float64},R"as.matrix(ag_report$Lambda_B)")
r_logsigma=rcopy(Float64,R"as.numeric(ag_params$log_sigma_eps)")
r_theta=vcat(r_beta,r_logsigma,GLLVModels.pack_lambda(r_loading))
r_mode=rcopy(Matrix{Float64},R"as.matrix(ag_obj$env$data$aghq_mode)")
r_B=rcopy(Matrix{Float64},R"as.matrix(ag_obj$env$data$aghq_Lt)")
r_logjac=rcopy(Vector{Float64},R"as.numeric(ag_obj$env$data$aghq_logdet)")
r_caches=[GLLVModels.AGHQAdaptation(vec(r_mode[s,:]),Matrix(reshape(r_B[s,:],K,K)'),r_logjac[s],false,NaN) for s in 1:n]
r_objective=rcopy(Float64,R"ag_value");r_gradient=rcopy(Vector{Float64},R"ag_grad")
theta=fit.parameters
exact=function(t)
 L=GLLVModels.unpack_lambda(t[p+2:end],p,K);M=L*L'+exp(2t[p+1])*I
 F=cholesky(Symmetric(M));e=Y.-t[1:p]
 return (n*p*log(2pi)+n*logdet(F)+sum(e.*(F\e)))/2
end
frozen=t->problem.objective(t,fit.adaptation)
g=GLLVModels.ForwardDiff.gradient(frozen,theta);ge=GLLVModels.ForwardDiff.gradient(exact,theta)
H=GLLVModels.ForwardDiff.hessian(frozen,theta);He=GLLVModels.ForwardDiff.hessian(exact,theta)
L=GLLVModels.unpack_lambda(theta[p+2:end],p,K);sigma=exp(theta[p+1]);rsigma=exp(r_logsigma)
record=Dict("case_id"=>"GU-GAUSSIAN-SEED42-K5","scope"=>"PUBLIC_GAUSSIAN_ORIGINAL_K5",
 "julia_version"=>string(VERSION),"package_root"=>pkgdir(GLLVModels),"winner"=>multistart.winner,
 "native_objective"=>fit.objective,"r_objective"=>r_objective,"delta_loglik"=>abs(fit.objective-r_objective),
 "native_converged"=>fit.converged,"r_converged"=>rcopy(Bool,R"isTRUE(ag_fit$aghq$converged)"),
 "r_used"=>rcopy(Bool,R"isTRUE(ag_fit$aghq$used)"),"r_k"=>rcopy(Int,R"ag_fit$aghq$k"),
 "r_penalised"=>rcopy(Bool,R"isTRUE(ag_fit$aghq$penalised)"),"r_nfree"=>length(r_gradient),
 "r_n_starts"=>rcopy(Int,R"ag_fit$aghq$n_starts"),"r_stop_reason"=>rcopy(String,R"ag_fit$aghq$stop_reason"),
 "native_gradient_max"=>maximum(abs,g),"r_gradient_max"=>maximum(abs,r_gradient),
 "samepoint_delta"=>problem.objective(r_theta,r_caches)-r_objective,
 "exact_objective_delta"=>exact(theta)-fit.objective,
 "exact_gradient_delta"=>maximum(abs.(g-ge)),"exact_hessian_delta"=>maximum(abs.(H-He)),
 "sigma_delta"=>abs(sigma-rsigma),"covariance_delta"=>maximum(abs.(L*L'+sigma^2*I-r_loading*r_loading'-rsigma^2*I)),
 "native_parameters"=>theta,"r_native_parameters"=>r_theta,
 "artifact_sha256"=>Dict(suffix=>bytes2hex(sha256(read(out*suffix))) for suffix in (".fixture.toml",".julia.toml",".rds")))
open(io->TOML.print(io,record),out,"w")
println("AG_RECEIPT_SHA256 ",bytes2hex(sha256(read(out))))
println("AG_METRICS ",repr(Dict(k=>record[k] for k in ("delta_loglik","samepoint_delta","native_gradient_max","r_gradient_max","exact_gradient_delta","exact_hessian_delta","sigma_delta","covariance_delta","r_converged","native_converged"))))
@testset "AG original Gaussian AGHQ pair" begin
 @test record["r_used"] && record["r_k"]==5 && !record["r_penalised"]
 @test record["r_nfree"]==length(theta)==15
 @test length(multistart.runs)==record["r_n_starts"]==2
 @test record["native_converged"] && record["r_converged"]
 @test maximum(abs,g)<1e-4 || maximum(abs,g)/max(1,abs(fit.objective))<1e-6
 @test maximum(abs,r_gradient)<1e-4 || maximum(abs,r_gradient)/max(1,abs(r_objective))<1e-6
 @test record["delta_loglik"]<=1e-3
 @test abs(record["samepoint_delta"])<=1e-6
 @test abs(record["exact_objective_delta"])<=1e-8
 @test record["exact_gradient_delta"]<=1e-7
 @test record["exact_hessian_delta"]<=1e-6
 @test record["sigma_delta"]<=1e-4
 @test record["covariance_delta"]<=1e-4
end
