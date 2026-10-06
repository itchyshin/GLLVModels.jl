# Original retained default-unique fixture; direct and formula cases share one execution.
module Core070GaussianOriginalRequired
using GLLVModels,RCall,Test,LinearAlgebra,Statistics,TOML,SHA
using ..Main: _core070_receipt_dir, _core070_root, _parity_require_gllvmtmb!, core070_record_values!
@assert realpath(pkgdir(GLLVModels))==realpath(_core070_root())
_parity_require_gllvmtmb!()
fixture=joinpath(@__DIR__,"fixtures/core070_gaussian_original.toml")
reference=joinpath(@__DIR__,"fixtures/core070_gaussian_reference.R")
output=_core070_receipt_dir()
row=only(filter(x->x["id"]=="CORE070-AGHQ-K1-GAUSSIAN",TOML.parsefile(fixture)["cases"]))
p=row["p"];n=row["n"];K=row["K"];seed=row["seed"];Y=reshape(row["responses"],p,n)
@rput Y seed reference
R"source($reference); rb <- .admission_run(Y,'gaussian','value ~ 0 + latent(0 + trait | site, d=1)',seed,FALSE)"
rtheta=rcopy(Vector{Float64},R"as.numeric(rb$fit$opt$par)")
names=rcopy(Vector{String},R"names(rb$fit$opt$par)")
lambda_idx=findall(==("theta_rr_B"),names);unique_idx=findall(==("theta_diag_B"),names)
@assert length(lambda_idx)==GLLVModels.rr_theta_len(p,K) && length(unique_idx)==p && length(rtheta)==length(lambda_idx)+p
c=rcopy(Float64,R"exp(rb$params$log_sigma_eps)")
@assert isapprox(c,max(0.001std(vec(Y)),1e-6);atol=1e-14)
function objective(theta)
 L=GLLVModels.unpack_lambda(theta[lambda_idx],p,K)
 -gaussian_pervar_marginal_loglik(Y,L,exp.(2 .* theta[unique_idx]).+c^2)
end
point=copy(rtheta);point[lambda_idx]=[0.8,0.55,-0.45,0.3];point[unique_idx]=log.([0.5,0.6,0.7,0.8])
@rput point
rv=rcopy(Float64,R"as.numeric(rb$fit$tmb_obj$fn(point))");rg=rcopy(Vector{Float64},R"as.numeric(rb$fit$tmb_obj$gr(point))")
point_value_delta=abs(objective(point)-rv);point_gradient_delta=maximum(abs,GLLVModels.ForwardDiff.gradient(objective,point)-rg)
fit=fit_gaussian_pervar_gllvm(Y;K=K,X=zeros(p,n,0),fixed_residual_sd=c,method=:lbfgs,g_tol=1e-8,iterations=3000)
ntheta=copy(rtheta);ntheta[lambda_idx]=GLLVModels.pack_lambda(fit.Λ);ntheta[unique_idx]=log.(fit.ψ²)./2
@rput ntheta rtheta
R"rvalue<-as.numeric(rb$fit$tmb_obj$fn(rtheta)); rgrad<-as.numeric(rb$fit$tmb_obj$gr(rtheta)); cross<-as.numeric(rb$fit$tmb_obj$fn(ntheta))"
rvalue=rcopy(Float64,R"rvalue");rgradient=rcopy(Vector{Float64},R"rgrad")
ngradient=GLLVModels.ForwardDiff.gradient(objective,ntheta)
rconverged=rcopy(Bool,R"as.integer(rb$fit$opt$convergence)==0L")
nh=fit.converged && all(isfinite,ngradient) && (maximum(abs,ngradient)<=1e-4 || maximum(abs,ngradient)/max(1,abs(fit.loglik))<=1e-6)
rh=rconverged && all(isfinite,rgradient) && (maximum(abs,rgradient)<=1e-4 || maximum(abs,rgradient)/max(1,abs(rvalue))<=1e-6)
record=Dict("fixture_sha256"=>bytes2hex(sha256(read(fixture))),"fixed_residual_sd"=>c,"native_unique_variances"=>fit.ψ²,"native_total_variances"=>fit.φ²,"r_parameter_names"=>names,"r_parameters"=>rtheta,"native_parameters_in_r_scale"=>ntheta,"native_loglik"=>fit.loglik,"r_loglik"=>-rvalue,"delta_loglik"=>abs(fit.loglik+rvalue),"point_value_delta"=>point_value_delta,"point_gradient_delta"=>point_gradient_delta,"r_endpoint_value_delta"=>abs(objective(rtheta)-rvalue),"native_endpoint_cross_delta"=>abs(fit.loglik+rcopy(Float64,R"cross")),"native_gradient_max"=>maximum(abs,ngradient),"r_gradient_max"=>maximum(abs,rgradient),"native_health"=>nh,"r_health"=>rh,"r_random"=>rcopy(Vector{String},R"rb$random"),"r_fixed_columns"=>rcopy(Int,R"ncol(rb$fit$tmb_obj$env$data$X_fix)"),"r_sigma_mapped"=>rcopy(Bool,R"!is.null(rb$fit$tmb_obj$env$map$log_sigma_eps)"),"native_dof"=>GLLVModels._nparams(fit))
open(io->TOML.print(io,record),joinpath(output,"gaussian-native.toml"),"w")
println(record)
const NATIVE_ID="CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL"
const FORMULA_ID="CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE"
core070_record_values!("objective at a fixed off-optimum point";julia=objective(point),r=rv,atol=1e-6,case=NATIVE_ID,
    test="@test point_value_delta<=1e-6")
core070_record_values!("objective at the R optimum";julia=objective(rtheta),r=rvalue,atol=1e-6,case=NATIVE_ID,
    test="@test record[\"r_endpoint_value_delta\"]<=1e-6")
core070_record_values!("objective at the Julia optimum";julia=-fit.loglik,r=rcopy(Float64,R"cross"),atol=1e-6,case=NATIVE_ID,
    test="@test record[\"native_endpoint_cross_delta\"]<=1e-6")
core070_record_values!("logLik";julia=fit.loglik,r=-rvalue,atol=1e-3,case=NATIVE_ID,
    test="@test record[\"delta_loglik\"]<=1e-3")
@testset "Original Gaussian default unique native pair" begin
 # Maintainer ruling 2026-10-05 (vault D-319): at P1 gllvmTMB reports only z_B as random
 # (gllvmTMB commit 853daf5f0); the earlier ["z_B","s_B"] expectation was stale.
 @test record["r_random"]==["z_B"]
 @test record["r_fixed_columns"]==0 && record["r_sigma_mapped"]
 @test record["native_dof"]==length(rtheta)
 @test point_value_delta<=1e-6
 @test point_gradient_delta<=1e-6
 @test record["r_endpoint_value_delta"]<=1e-6
 @test record["native_endpoint_cross_delta"]<=1e-6
 @test record["delta_loglik"]<=1e-3
 @test nh
 @test rh
 @test maximum(abs,ngradient)<=1e-4
 @test maximum(abs,rgradient)<=1e-4
end
println("CORE070_DEFAULT_UNIQUE_PAIR_PASS")

formula_fit=gllvm(@formula(y ~ 0),Y,(site=collect(1:n),);family=GLLVModels.Normal(),
    K=K,pervar=true,fixed_residual_sd=c,method=:lbfgs,g_tol=1e-8,iterations=3000)
ftheta=copy(rtheta);ftheta[lambda_idx]=GLLVModels.pack_lambda(formula_fit.Λ)
ftheta[unique_idx]=log.(formula_fit.ψ²)./2
fg=GLLVModels.ForwardDiff.gradient(objective,ftheta)
formula_record=Dict("id"=>"CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE",
    "fixture_sha256"=>bytes2hex(sha256(read(fixture))),
    "formula_parameters"=>ftheta, "formula"=>"y ~ 0", "pervar"=>true,"fixed_residual_sd"=>c,
    "formula_loglik"=>formula_fit.loglik,"r_loglik"=>-rvalue,
    "delta_loglik"=>abs(formula_fit.loglik+rvalue),
    "native_formula_delta"=>abs(formula_fit.loglik-fit.loglik),
    "formula_gradient_max"=>maximum(abs,fg),"converged"=>formula_fit.converged,
    "fixed_effect_count"=>length(formula_fit.β),"dof"=>GLLVModels._nparams(formula_fit),
    "formula_unique_variances"=>formula_fit.ψ²)
open(io->TOML.print(io,formula_record),joinpath(output,"gaussian-formula.toml"),"w")
core070_record_values!("logLik, wide formula";julia=formula_fit.loglik,r=-rvalue,atol=1e-3,case=FORMULA_ID,
    test="@test abs(formula_fit.loglik+rvalue)<=1e-3")
@testset "Original R default unique formula pair" begin
 @test formula_fit isa GaussianPerVarFit
 @test isempty(formula_fit.β)
 @test formula_record["dof"]==length(rtheta)
 @test formula_fit.converged
 @test maximum(abs,fg)<=1e-4
 @test abs(formula_fit.loglik+rvalue)<=1e-3
 @test abs(formula_fit.loglik-fit.loglik)<=1e-7
 @test formula_fit.ψ²≈fit.ψ² atol=1e-6
 @test rh
 @test maximum(abs,ftheta-ntheta)<=1e-7
end
println("CORE070_PERVAR_FORMULA_R_PAIR_PASS")
println(formula_record)

longdata=(y=reverse(vec(Y)),species=reverse(repeat(collect(1:p),n)),
          site=reverse(repeat(collect(1:n);inner=p)))
longfit=gllvm(@formula(y ~ 0),longdata;family=GLLVModels.Normal(),K=K,pervar=true,
    fixed_residual_sd=c,method=:lbfgs,g_tol=1e-8,iterations=3000)
longtheta=copy(rtheta);longtheta[lambda_idx]=GLLVModels.pack_lambda(longfit.Λ)
longtheta[unique_idx]=log.(longfit.ψ²)./2
longgrad=GLLVModels.ForwardDiff.gradient(objective,longtheta)
long_record=Dict("fixture_sha256"=>bytes2hex(sha256(read(fixture))),
    "loglik"=>longfit.loglik,"converged"=>longfit.converged,
    "gradient_max"=>maximum(abs,longgrad),"parameters"=>longtheta,
    "native_parameter_delta"=>maximum(abs,longtheta-ntheta),
    "native_loglik_delta"=>abs(longfit.loglik-fit.loglik))
open(io->TOML.print(io,long_record),joinpath(output,"gaussian-long.toml"),"w")
core070_record_values!("logLik, long formula";julia=longfit.loglik,r=-rvalue,atol=1e-3,case=FORMULA_ID,
    test="@test abs(longfit.loglik+rvalue)<=1e-3")
@testset "Original Gaussian reordered long formula" begin
 @test longfit isa GaussianPerVarFit
 @test longfit.converged
 @test isempty(longfit.β)
 @test maximum(abs,longgrad)<=1e-4
 @test abs(longfit.loglik+rvalue)<=1e-3
 @test long_record["native_parameter_delta"]<=1e-7
 @test long_record["native_loglik_delta"]<=1e-7
 @test_throws ArgumentError gllvm(@formula(y ~ 0),map(x->x[2:end],longdata);
     family=GLLVModels.Normal(),K=K,pervar=true,fixed_residual_sd=c)
 @test_throws ArgumentError gllvm(@formula(y ~ 0),map(x->vcat(x,x[1]),longdata);
     family=GLLVModels.Normal(),K=K,pervar=true,fixed_residual_sd=c)
end
println("GAUSSIAN_ORIGINAL_REQUIRED_PAIR_PASS")

for name in ["gaussian-native.toml","gaussian-formula.toml","gaussian-long.toml"]
 println("GAUSSIAN_ARTIFACT_SHA256 ",name," ",bytes2hex(sha256(read(joinpath(output,name)))))
end

end
