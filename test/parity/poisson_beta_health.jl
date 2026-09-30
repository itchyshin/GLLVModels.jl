# Original Poisson/Beta qualification integrated into the required test runner.
# Objective, FD and public-R refinement code retained from the qualified packet.
using LinearAlgebra
function core070_poisson_beta_required(family::Symbol)
    family in (:poisson,:beta) || throw(ArgumentError("unknown original health case"))
    root=_core070_root();dir=_core070_receipt_dir()
    contract="docs/dev-log/core070/poisson-beta-required-contract.json"
    _core070_sha256_file(joinpath(root,contract))=="8b6016cbedcbdc4d20d66f3a4db74f3e0e5bfb4b0fb7d34e742afa499150642c" || error("required model contract changed")
    fixture="test/parity/test_$(family)_parity.jl"
    # Whole-file pins changed only because the package/module spelling changed
    # from GLLVM to GLLVModels. The DGP-region pins below intentionally remain
    # frozen, so this rename cannot silently alter the historical simulation.
    expected=Dict(
        :poisson => ("0faa57a2d346dd85715053c90db35550c614e6f87152956e79d9aaf6f0a296c7","404e19e607362e4682f7348dec0fc5dd127d06114fe1f9197f24001ddf537100"),
        :beta => ("2fda9fcd23ac2bb435cacd7cf9b03bc25b7ccbadacb291a928ed7ac3d1440fa9","45c31fe9b8e846681bdabb7f4a8284bc8a1f04ada25c685a34687c617669e1ab"))
    _core070_sha256_file(joinpath(root,fixture))==expected[family][1] || error("original fixture changed")
    source=read(joinpath(root,fixture),String)
    first=findfirst("    Random.seed!(",source).start;last=findnext("    jl_fit =",source,first).start
    bytes2hex(sha256(source[first:last-1]))==expected[family][2] || error("original DGP changed")
    policy="public_start_from_refinement_v1";refine=true
    id=family===:poisson ? "NATIVE-03-POISSON" : "NATIVE-08-BETA"
    source = read(joinpath(root, fixture), String)
    # Execute exactly the original samplers and DGP, before the original fit.
    prefix = source[1:findfirst("@testset", source).start-1]
    a = findfirst("    Random.seed!(", source).start
    b = findnext("    jl_fit =", source, a).start
    dgp = source[a:b-1]
    mod = Module(Symbol("Original_", family))
    Core.eval(mod, :(using Main: parity_loadings_p5k2))
    data = Base.include_string(mod, prefix * dgp * "\n(Y=Y, K=K, p=p, n=n)\n", fixture)
    Y, K, p, n = data.Y, data.K, data.p, data.n
    (p,n,K) == (5,60,family===:poisson ? 2 : 1) || error("original shape changed")
    datafile = joinpath(dir, string(family)*"-fixture.toml")
    open(io -> TOML.print(io, Dict("p"=>p,"n"=>n,"K"=>K,"Y_column_major"=>vec(Y),
        "fixture_sha256"=>_core070_sha256_file(joinpath(root,fixture)),
        "dgp_sha256"=>bytes2hex(sha256(dgp)))), datafile, "w")
    native = family===:poisson ? fit_poisson_gllvm(Y; K=K) :
        fit_gllvm(Y; family=GLLVModels.Beta(), K=K, g_tol=1e-7, iterations=800)
    r = fit_gllvmtmb_parity_loglik(Y, K; family=family)
    rawpath=joinpath(dir,string(family)*"-whole-fit.rds")
    fam=string(family)
    @rput rawpath fam refine
    R"""
    pb_original <- fit_r
    pb_original_gradient <- as.numeric(pb_original$tmb_obj$gr(pb_original$opt$par))
    if (refine) {
      fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site,d=K,unique=FALSE),
        data=df_long,unit="site",trait="trait",family=fam_obj,
        control=gllvmTMBcontrol(n_init=1L,se=FALSE,start_from=pb_original,
          optArgs=list(control=list(rel.tol=1e-12,eval.max=2000,iter.max=1500))))
    }
    pb_preserved <- identical(names(fit_r$opt$par),names(pb_original$opt$par)) &&
        identical(fit_r$tmb_obj$env$data,pb_original$tmb_obj$env$data) &&
        identical(fit_r$tmb_obj$env$map,pb_original$tmb_obj$env$map)
    """
    r=(logLik=rcopy(Float64,R"as.numeric(logLik(fit_r))"),
       objective=rcopy(Float64,R"as.numeric(fit_r$opt$objective)"),
       converged=rcopy(Bool,R"identical(as.integer(fit_r$opt$convergence),0L)"))
    R"""
    pb_obj <- fit_r$tmb_obj
    pb_objective <- as.numeric(pb_obj$fn(fit_r$opt$par))
    pb_gradient <- as.numeric(pb_obj$gr(fit_r$opt$par))
    pb_full <- pb_obj$env$last.par
    pb_parameters <- pb_obj$env$parList(x=fit_r$opt$par,par=pb_full)
    pb_report <- pb_obj$report(pb_full)
    pb_beta <- as.numeric(pb_parameters$b_fix)
    pb_lambda <- as.matrix(pb_report$Lambda_B)
    pb_disp <- if(fam == "beta") exp(as.numeric(pb_parameters$log_phi_beta)) else numeric(0)
    saveRDS(list(opt=fit_r$opt,gradient=pb_gradient,parameters=pb_parameters,
        report=pb_report,data=pb_obj$env$data,objective=pb_objective,
        map=pb_obj$env$map,random=pb_obj$env$random,
        original_opt=pb_original$opt,original_gradient=pb_original_gradient,
        original_data=pb_original$tmb_obj$env$data,original_map=pb_original$tmb_obj$env$map),rawpath)
    """
    rr=GLLVModels.rr_theta_len(p,K)
    theta=vcat(native.β,GLLVModels.pack_lambda(native.Λ),family===:beta ? log.(native.φ) : Float64[])
    function objective(v)
        beta=v[1:p];lambda=GLLVModels.unpack_lambda(v[p+1:p+rr],p,K)
        family===:poisson ?
            -GLLVModels.poisson_marginal_loglik_laplace(Y,lambda,beta,LogLink();hessian=native.hessian,maxiter=100,tol=1e-9) :
            -GLLVModels.beta_grouped_marginal_loglik_laplace(Y,lambda,beta,exp.(v[p+rr+1:end]);hessian=:observed,maxiter=100,tol=1e-9)
    end
    function fd(v,m)
        [begin
            h=m*cbrt(eps(Float64))*max(1.0,abs(v[j]));a=copy(v);b=copy(v)
            a[j]+=h;b[j]-=h;(objective(a)-objective(b))/(2h)
         end for j in eachindex(v)]
    end
    g1=fd(theta,1.0);g2=fd(theta,2.0)
    rp=rcopy(Vector{Float64},R"as.numeric(fit_r$opt$par)")
    rg=rcopy(Vector{Float64},R"pb_gradient")
    rbeta=rcopy(Vector{Float64},R"pb_beta")
    rlambda=rcopy(Matrix{Float64},R"pb_lambda")
    rdisp=rcopy(Vector{Float64},R"pb_disp")
    rn=vcat(rbeta,GLLVModels.pack_lambda(rlambda),log.(rdisp))
    r_objective=rcopy(Float64,R"pb_objective")
    expected=family===:poisson ? 14 : 15
    report=Dict("id"=>id,"family"=>fam,"scope"=>"ORIGINAL_NATIVE_FIT_HEALTH_NOT_RECOVERY",
        "policy"=>policy,"model_preserved"=>rcopy(Bool,R"pb_preserved"),
        "original_r_gradient"=>rcopy(Vector{Float64},R"pb_original_gradient"),
        "original_r_parameters"=>rcopy(Vector{Float64},R"as.numeric(pb_original$opt$par)"),
        "original_r_code"=>rcopy(Int,R"as.integer(pb_original$opt$convergence)"),
        "original_r_objective"=>rcopy(Float64,R"as.numeric(pb_original$opt$objective)"),"native_loglik"=>native.loglik,"r_loglik"=>r.logLik,
        "native_converged"=>native.converged,"r_converged"=>r.converged,
        "r_code"=>rcopy(Int,R"as.integer(fit_r$opt$convergence)"),
        "r_message"=>rcopy(String,R"as.character(fit_r$opt$message)"),
        "native_parameters"=>theta,"r_parameters"=>rp,"r_native_parameters"=>rn,
        "native_gradient"=>g1,"native_gradient_double_step"=>g2,"r_gradient"=>rg,
        "native_nfree"=>length(theta),"r_nfree"=>length(rp),"expected_nfree"=>expected,
        "native_gradient_max"=>maximum(abs,g1),"r_gradient_max"=>maximum(abs,rg),
        "fd_stability"=>maximum(abs.(g1-g2)),"hessian"=>string(native.hessian),
        "native_objective_delta"=>abs(objective(theta)+native.loglik),
        "r_objective"=>r_objective,"r_cached_objective"=>r.objective,
        "r_packing_delta"=>length(rp)==length(rn) ? maximum(abs.(rp-rn)) : Inf,
        "samepoint_native_nll"=>objective(rn),"samepoint_delta"=>objective(rn)-r_objective,
        "native_dispersion"=>family===:beta ? native.φ : Float64[],"r_dispersion"=>rdisp,
        "data_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(Float64.(Y))))),
        "fixture_sha256"=>_core070_sha256_file(datafile),"raw_fits_sha256"=>_core070_sha256_file(rawpath))
    checks=Dict(
        "native_converged"=>native.converged,"r_converged"=>r.converged,
        "finite"=>all(isfinite,theta)&&all(isfinite,rp)&&isfinite(native.loglik)&&isfinite(r.logLik),
        "likelihood"=>isapprox(native.loglik,r.logLik;rtol=1e-6,atol=0),
        "native_objective"=>report["native_objective_delta"]<=1e-8,
        "r_objective"=>abs(r_objective+r.logLik)<=1e-8&&abs(r.objective+r.logLik)<=1e-10,
        "native_gradient"=>maximum(abs,g1)<=1e-4,
        "r_gradient"=>maximum(abs,rg)<=1e-4,"fd_stability"=>report["fd_stability"]<=1e-4,
        "parameter_count"=>length(theta)==length(rp)==expected,
        "r_packing"=>report["r_packing_delta"]<=1e-12,
        "samepoint"=>abs(report["samepoint_delta"])<=1e-6,
        "link"=>family===:poisson ? native.link isa LogLink : native.link isa LogitLink,
        "curvature"=>family===:poisson ? GLLVModels._glm_weight_matches_observed(GLLVModels.Poisson(),native.link) : native.hessian===:observed,
        "dispersion"=>family===:poisson ? isempty(rdisp) : native.group==collect(1:p)&&length(native.φ)==length(rdisp)==p&&all(>(0),native.φ)&&all(>(0),rdisp))
    checks["model_preserved"] = report["model_preserved"]
    core070_record_values!("logLik"; julia=native.loglik, r=r.logLik, rtol=1e-6,
        test="checks[\"likelihood\"]=isapprox(native.loglik,r.logLik;rtol=1e-6,atol=0)")
    core070_record_values!("objective at the R optimum"; julia=report["samepoint_native_nll"], r=r_objective, atol=1e-6,
        test="checks[\"samepoint\"]=abs(report[\"samepoint_delta\"])<=1e-6")
    report["checks"]=checks
    metric=joinpath(dir,fam*"-health.toml")
    open(io->TOML.print(io,report),metric,"w")
    println(uppercase(fam)*"_HEALTH_SHA256 ",_core070_sha256_file(metric))
    println(uppercase(fam)*"_RAW_FITS_SHA256 ",report["raw_fits_sha256"])
    println(id," checks=",count(values(checks)),"/",length(checks),
        " dLL=",native.loglik-r.logLik," native_gradient=",report["native_gradient_max"],
        " R_gradient=",report["r_gradient_max"]," samepoint=",report["samepoint_delta"])
    return report
end
