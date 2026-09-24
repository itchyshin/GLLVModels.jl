"""The original NB2 data, read from `fixtures/nb2_original_data.toml`. Julia 1.10 draws
different numbers from the fixture's seed than Julia 1.12+, where the pinned draw was made,
so the draw is stored and every version fits the same data."""
function parity_nb2_original_Y()
 d=TOML.parsefile(joinpath(@__DIR__,"fixtures","nb2_original_data.toml"))
 Y=reshape(Int.(d["Y_column_major"]),d["p"],d["n"])
 h=bytes2hex(sha256(reinterpret(UInt8,vec(Float64.(Y)))))
 h==d["data_sha256"]=="7abde2731134afe61afee5a7f0c29b58892ad72e550fa41cf8230e9c701a2bf9" || error("stored NB2 data changed")
 return Y
end

"""Record the original NB2 default-oracle health without changing either fit."""
function parity_nb2_health(Y, K, native; artifact_prefix="nb2", receipt_tag="NB2", case_id="NATIVE-06-NB2")
 occursin(r"^[a-z][a-z0-9-]*$", artifact_prefix) || throw(ArgumentError("invalid NB2 artifact prefix"))
 occursin(r"^[A-Z][A-Z0-9_]*$", receipt_tag) || throw(ArgumentError("invalid NB2 receipt tag"))
 p,n=size(Y)
 (p,K,n)==(5,2,80) || error("NB2 health is scoped to the original fixture")
 datahash=bytes2hex(sha256(reinterpret(UInt8,vec(Float64.(Y)))))
 datahash=="7abde2731134afe61afee5a7f0c29b58892ad72e550fa41cf8230e9c701a2bf9" || error("original NB2 data changed")
 fixture=joinpath(@__DIR__,"test_negbin_parity.jl");source=read(fixture,String)
 helpers=source[findfirst("function _rand_poisson",source).start:findfirst("@testset \"NB2 GLLVModels",source).start-1]
 dgp=source[findfirst("    Random.seed!(45)",source).start:findfirst("    jl_fit =",source).start-1]
 output=_core070_required() ? _core070_receipt_dir() : mktempdir()
 rawpath=joinpath(output,artifact_prefix*"-whole-fit.rds")
 ispath(rawpath) && error("NB2 evidence exists; require a fresh run")
 @rput rawpath
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
    data=r_obj$env$data,objective=r_objective),rawpath)
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
 report["policy"]="nb2_original_default_v1"
 report["case_id"]=case_id
 report["raw_fits_sha256"]=_core070_sha256_file(rawpath)
 file=joinpath(output,artifact_prefix*"-health.toml")
 open(io->TOML.print(io,report),file,"w")
 println(receipt_tag*"_HEALTH_SHA256 ",_core070_sha256_file(file))
 println(receipt_tag*"_RAW_FITS_SHA256 ",report["raw_fits_sha256"])
 return merge(r,(health=report,))
end
