# Exact original ordinary independent/common source models; no formula/bridge claim.
using GLLVModels, RCall, LinearAlgebra, TOML, Test, SHA
include(joinpath(pwd(),"test/parity/parity_helpers.jl"))
length(ARGS)==1 || error("expected fresh output path")
output=ARGS[1];ispath(output) && error("fresh output required");mkpath(output)
get(ENV,"CORE070_PARITY_REQUIRED","")=="1" || error("required pinned oracle missing")
_parity_require_gllvmtmb!()
source=_core070_source_pin!()
R"source('test/parity/fixtures/core070_covariance_modes.R')"
Y=permutedims(rcopy(Matrix{Float64},R"matrix(df$value,nrow=18L,ncol=3L)"))
rows=Dict[]
for (index,common) in enumerate((false,true))
    @rput index output
    R"""
    case <- cases[[index]]
    rf <- eval(case$call)
    obj <- rf$tmb_obj
    outer <- rf$opt$par
    objective <- obj$fn(outer)
    gradient <- obj$gr(outer)
    pars <- obj$env$parList(x=outer,par=obj$env$last.par)
    stopifnot(length(pars$log_sigma_eps)==1L,
              !any(names(outer)=='log_sigma_eps'),
              all(is.na(obj$env$map$log_sigma_eps)))
    saveRDS(list(id=case$id,data=rf$tmb_data,map=obj$env$map,parameters=pars,
                outer=outer,gradient=gradient,objective=objective,
                loglik=as.numeric(logLik(rf)),optimizer=rf$opt),
            file.path(output,paste0(case$id,'.rds')))
    """
    id=rcopy(String,R"case$id")
    fixed=rcopy(Float64,R"exp(pars$log_sigma_eps)")
    rbeta=rcopy(Vector{Float64},R"as.numeric(pars$b_fix)")
    rsd=rcopy(Vector{Float64},R"as.numeric(exp(pars$theta_diag_B))")
    rpar=rcopy(Vector{Float64},R"as.numeric(outer)")
    rnames=rcopy(Vector{String},R"names(outer)")
    rgrad=rcopy(Vector{Float64},R"as.numeric(gradient)")
    rll=rcopy(Float64,R"as.numeric(logLik(rf))")
    robj=rcopy(Float64,R"objective")
    rcode=rcopy(Int,R"rf$opt$convergence")
    block=SourceCovariance(Matrix{Float64}(I,18,18);groups=1:18,mode=:indep,common=common,name=Symbol(id))
    native=fit_gaussian_sources(Y;sources=[block],sigma_eps_fixed=fixed,g_tol=1e-7)
    checks=Dict(
      "expected_id"=>id==(common ? "MODE-ORD-COMMON" : "MODE-ORD-INDEP"),
      "residual_fixed"=>native.residual_fixed && native.sigma_eps===fixed,
      "reference_residual_rule"=>isapprox(fixed,rcopy(Float64,R"max(.001*sd(df$value),1e-6)");rtol=1e-12),
      "free_parameters"=>GLLVModels.dof(native)==length(rpar)==(common ? 4 : 6),
      "r_code"=>rcode==0,"r_gradient"=>all(isfinite,rgrad)&&maximum(abs,rgrad)<=1e-4,
      "native_health"=>native.converged && native.gradient_norm<=1e-7,
      "finite_likelihoods"=>all(isfinite,(native.loglik,rll,robj)),
      "likelihood"=>abs(native.loglik-rll)<=1e-6,
      "reported_objective"=>abs(rll+robj)<=1e-8,
      "trait_means"=>isapprox(native.beta,rbeta;atol=1e-5,rtol=1e-5),
      "trait_covariance"=>isapprox(only(native.trait_covariances),Diagonal(rsd.^2);atol=1e-5,rtol=1e-5),
      "independent_offdiagonal"=>all(iszero,only(native.trait_covariances)-Diagonal(diag(only(native.trait_covariances)))))
    row=Dict("id"=>id,"common"=>common,"sigma_eps_fixed"=>fixed,"native_loglik"=>native.loglik,
      "r_loglik"=>rll,"r_objective"=>robj,"r_code"=>rcode,"r_gradient"=>rgrad,
      "r_names"=>rnames,"r_parameters"=>rpar,"r_beta"=>rbeta,"r_source_sd"=>rsd,
      "native_beta"=>native.beta,"native_source_variance"=>diag(only(native.trait_covariances)),
      "native_gradient_max"=>native.gradient_norm,"native_parameters"=>native.parameters,
      "native_dof"=>GLLVModels.dof(native),"checks"=>checks)
    push!(rows,row)
    open(joinpath(output,"result.toml"),"w") do io
        TOML.print(io,Dict("source"=>source,"data_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(Y)))),
          "fixture_sha256"=>bytes2hex(sha256(read("test/parity/fixtures/core070_covariance_modes.R"))),"cases"=>rows))
    end
    println(id," deltaLL=",native.loglik-rll," r_gradient=",maximum(abs,rgrad)," checks=",checks)
end
@testset "Original ordinary fixed-residual R pairs" begin
    @test length(rows)==2
    for row in rows
        @testset "$(row["id"])" begin
            for key in sort(collect(keys(row["checks"])))
                @test row["checks"][key]
            end
        end
    end
end
println("ORIGINAL_ORDINARY_FIXED_RESIDUAL_PAIRS_PASS")
