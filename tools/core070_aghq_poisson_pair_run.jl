using GLLVModels,RCall,Test,LinearAlgebra,SHA,TOML
root=normpath(joinpath(@__DIR__,".."))
include(joinpath(root,"test/parity/parity_helpers.jl"))
_parity_require_gllvmtmb!()
@testset "AGHQ multistart and prerequisites" begin
    include(joinpath(root,"test/test_aghq_multistart.jl"))
    include(joinpath(root,"test/test_aghq_poisson.jl"))
    include(joinpath(root,"test/test_aghq_outer.jl"))
    include(joinpath(root,"test/test_aghq_frozen.jl"))
    include(joinpath(root,"test/test_aghq_grid.jl"))
    include(joinpath(root,"test/test_aghq_adapt.jl"))
    include(joinpath(root,"test/test_aghq_gate.jl"))
    include(joinpath(root,"test/test_aghq_kd_bound.jl"))
end
fixture="test/parity/test_poisson_parity.jl"
source=read(joinpath(root,fixture),String)
bytes2hex(sha256(source))=="d7ca740f8daa303aae730647af53b1db461ba7c0d44a4341db17e7e3495ed204" || error("original fixture changed")
prefix=source[1:findfirst("@testset",source).start-1]
a=findfirst("    Random.seed!(",source).start;b=findnext("    jl_fit =",source,a).start
dgp=source[a:b-1]
bytes2hex(sha256(dgp))=="404e19e607362e4682f7348dec0fc5dd127d06114fe1f9197f24001ddf537100" || error("original DGP changed")
mod=Module(:APPOriginal);Core.eval(mod,:(using Main: parity_loadings_p5k2))
data=Base.include_string(mod,prefix*dgp*"\n(Y=Y,K=K,p=p,n=n)\n",fixture)
Y,K,p,n=data.Y,data.K,data.p,data.n
out=ENV["CORE070_AGHQ_PAIR_OUTPUT"]
# Retain inputs before either fit; exceptions still leave this exact DGP artifact.
open(io->TOML.print(io,Dict("responses"=>vec(Y),"p"=>p,"n"=>n,"K"=>K,
    "fixture_sha256"=>bytes2hex(sha256(source)),"dgp_sha256"=>bytes2hex(sha256(dgp)))),out*".fixture.toml","w")
println("APP_INPUT_SHA256 ",bytes2hex(sha256(read(out*".fixture.toml"))))
public_route=get(ENV,"CORE070_AGHQ_PUBLIC_PAIR","0")=="1"
base=fit_poisson_gllvm(Y;K=K)
start=vcat(base.β,GLLVModels.pack_lambda(base.Λ));alt=copy(start);alt[p+1:end].=.3
problem=GLLVModels.aghq_poisson_problem(Y,K;k=5)
public_fit=public_route ? fit_poisson_gllvm(Y;K=K,aghq=5) : nothing
multistart=public_route ? public_fit.integration.result : GLLVModels.aghq_multistart_optimize([start,alt],problem.adapt,problem.objective;n_adapt=400)
multistart.usable || error("no usable Julia AGHQ start")
runs=multistart.runs;winner=multistart.winner;fit=multistart.selected
serialize_run(r)=Dict("parameters"=>r.parameters,"objective"=>r.objective,"usable"=>r.usable,
    "converged"=>r.converged,"stop_reason"=>string(r.stop_reason),"passes"=>r.passes,
    "gradient_max"=>r.frozen_gradient_max,"relative_gradient"=>r.relative_gradient,
    "trace"=>[Dict(string(k)=>(v===nothing ? "uncapped" : v) for (k,v) in pairs(row)) for row in r.trace])
open(io->TOML.print(io,Dict("winner"=>winner,"runs"=>serialize_run.(runs))),out*".julia.toml","w")
println("APP_JULIA_SHA256 ",bytes2hex(sha256(read(out*".julia.toml"))))
@rput Y K p n out
R"""
app_df <- data.frame(site=factor(rep(seq_len(n),each=p)),
 trait=factor(rep(paste0('t',seq_len(p)),times=n),levels=paste0('t',seq_len(p))),value=as.vector(Y))
app_warnings <- character()
app_fit <- withCallingHandlers(gllvmTMB(value~0+trait+latent(0+trait|site,d=K,unique=FALSE),
 data=app_df,unit='site',trait='trait',family=poisson(),
 control=gllvmTMBcontrol(aghq=5,aghq_ridge=Inf,aghq_multistart=TRUE,aghq_n_adapt=400L,n_init=1L,se=FALSE)),
 warning=function(w) { app_warnings <<- c(app_warnings,conditionMessage(w)); invokeRestart('muffleWarning') })
app_obj <- app_fit$tmb_obj
app_fresh_f <- as.numeric(app_obj$fn(app_fit$opt$par))
app_fresh_g <- as.numeric(app_obj$gr(app_fit$opt$par))
app_full <- app_obj$env$last.par
app_params <- app_obj$env$parList(x=app_fit$opt$par,par=app_full)
app_report <- app_obj$report(app_full)
saveRDS(list(opt=app_fit$opt,aghq=app_fit$aghq,objective=app_fresh_f,gradient=app_fresh_g,
 parameters=app_params,report=app_report,data=app_obj$env$data,map=app_obj$env$map,
 warnings=app_warnings),paste0(out,'.rds'))
"""
println("APP_R_SHA256 ",bytes2hex(sha256(read(out*".rds"))))
r_beta=rcopy(Vector{Float64},R"as.numeric(app_params$b_fix)")
r_loading=rcopy(Matrix{Float64},R"as.matrix(app_report$Lambda_B)")
r_theta=vcat(r_beta,GLLVModels.pack_lambda(r_loading))
r_mode=rcopy(Matrix{Float64},R"as.matrix(app_obj$env$data$aghq_mode)")
r_B=rcopy(Matrix{Float64},R"as.matrix(app_obj$env$data$aghq_Lt)")
r_logjac=rcopy(Vector{Float64},R"as.numeric(app_obj$env$data$aghq_logdet)")
r_caches=[GLLVModels.AGHQAdaptation(vec(r_mode[s,:]),Matrix(reshape(r_B[s,:],K,K)'),r_logjac[s],false,NaN) for s in 1:n]
samepoint=problem.objective(r_theta,r_caches)
r_objective=rcopy(Float64,R"app_fresh_f")
r_gradient=rcopy(Vector{Float64},R"app_fresh_g")
t=fit.parameters
fresh=x->problem.objective(x,problem.adapt(x))
fd(h)=[(fresh(t+h*Matrix{Float64}(I,14,14)[:,i])-fresh(t-h*Matrix{Float64}(I,14,14)[:,i]))/(2h) for i in 1:14]
g1=fd(1e-5);g2=fd(2e-5)
gf=GLLVModels.ForwardDiff.gradient(x->problem.objective(x,fit.adaptation),t)
L=GLLVModels.unpack_lambda(t[p+1:end],p,K)
refinement=[let q=GLLVModels.aghq_poisson_problem(Y,K;k=k);q.objective(t,q.adapt(t));end for k in (5,9,15)]
record=Dict("case_id"=>"APP-POISSON-SEED44-K5","scope"=>(public_route ? "PUBLIC_POISSON_AGHQ_SEED44_ONLY" : "INTERNAL_JULIA_PUBLIC_FROZEN_R_AGHQ_NOT_PUBLIC_JULIA_PARITY"),
 "julia_version"=>string(VERSION),"package_root"=>pkgdir(GLLVModels),"winner"=>winner,
 "native_objective"=>fit.objective,"r_objective"=>r_objective,"delta_loglik"=>abs(fit.objective-r_objective),
 "native_converged"=>fit.converged,"r_converged"=>rcopy(Bool,R"isTRUE(app_fit$aghq$converged)"),
 "r_used"=>rcopy(Bool,R"isTRUE(app_fit$aghq$used)"),"r_k"=>rcopy(Int,R"app_fit$aghq$k"),
 "r_penalised"=>rcopy(Bool,R"isTRUE(app_fit$aghq$penalised)"),"r_nfree"=>length(r_gradient),
 "r_n_starts"=>rcopy(Int,R"app_fit$aghq$n_starts"),"r_start_used"=>rcopy(Int,R"app_fit$aghq$start_used"),
 "r_stop_reason"=>rcopy(String,R"app_fit$aghq$stop_reason"),"r_stored_gradient"=>rcopy(Float64,R"app_fit$aghq$grad_max"),
 "native_gradient"=>gf,"r_gradient"=>r_gradient,"native_gradient_max"=>maximum(abs,gf),"r_gradient_max"=>maximum(abs,r_gradient),
 "samepoint_delta"=>samepoint-r_objective,"readapted_rpoint_delta"=>fresh(r_theta)-r_objective,
 "native_parameters"=>t,"r_native_parameters"=>r_theta,
 "covariance_delta"=>maximum(abs.(L*L'-r_loading*r_loading')),
 "marginal_mean_delta"=>maximum(abs.(exp.(t[1:p]+diag(L*L')/2)-exp.(r_beta+diag(r_loading*r_loading')/2))),
 "total_fd"=>g1,"total_fd_double_step"=>g2,"fd_stability"=>maximum(abs.(g1-g2)),
 "omitted_chain_delta"=>maximum(abs.(g1-gf)),"node_refinement_at_winner"=>refinement,
 "artifact_sha256"=>Dict(suffix=>bytes2hex(sha256(read(out*suffix))) for suffix in (".fixture.toml",".julia.toml",".rds")))
open(io->TOML.print(io,record),out,"w")
println("APP_RECEIPT_SHA256 ",bytes2hex(sha256(read(out))))
println("APP_METRICS ",repr(Dict(k=>record[k] for k in ("delta_loglik","samepoint_delta","native_gradient_max","r_gradient_max","omitted_chain_delta","node_refinement_at_winner","r_converged","native_converged"))))
@testset "APP original Poisson AGHQ pair" begin
 @test record["r_used"] && record["r_k"]==5 && !record["r_penalised"]
 @test record["r_nfree"]==length(t)==14
 @test length(runs)==record["r_n_starts"]==2
 @test record["native_converged"] && record["r_converged"]
 @test maximum(abs,gf)<1e-4 || maximum(abs,gf)/max(1,abs(fit.objective))<1e-6
 @test maximum(abs,r_gradient)<1e-4 || maximum(abs,r_gradient)/max(1,abs(r_objective))<1e-6
 @test record["delta_loglik"]<=1e-3
 @test abs(record["samepoint_delta"])<=1e-6
end
