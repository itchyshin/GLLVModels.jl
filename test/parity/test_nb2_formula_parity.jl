# The original model is unchanged; a module isolates fixture RNG helper names.
isdefined(@__MODULE__, :parity_nb2_health) || include(joinpath(@__DIR__, "nb2_health.jl"))
module Core070NB2FormulaCase
using GLLVModels, RCall, Test, Random, SHA, TOML, LinearAlgebra
using ..Main: parity_nb2_health, parity_nb2_original_Y, _core070_receipt_dir, _core070_sha256_file, parity_loadings_p5k2
source=read("test/parity/test_negbin_parity.jl",String)
helpers=source[findfirst("function _rand_poisson",source).start:findfirst("@testset \"NB2 GLLVModels",source).start-1]
dgp=source[findfirst("    Random.seed!(45)",source).start:findfirst("    jl_fit =",source).start-1]
include_string(@__MODULE__,helpers*dgp,"original_nb2_fixture")
native=fit_gllvm(Y;family=GLLVModels.NegativeBinomial(),K=K,g_tol=1e-7,iterations=800)
r=parity_nb2_health(Y,K,native; artifact_prefix="formula-nb2", receipt_tag="FORMULA",
    case_id="CORE070-FAMILY-05-LOG-FORMULA-INTERFACE")
wide=gllvm(@formula(y ~ 1),Y,(site=collect(1:n),);
    family=GLLVModels.NegativeBinomial(),K=K,g_tol=1e-7,iterations=800)
long=(y=reverse(vec(Y)),species=reverse(repeat(collect(1:p),n)),
      site=reverse(repeat(collect(1:n);inner=p)))
longfit=gllvm(@formula(y ~ 1),long;
    family=GLLVModels.NegativeBinomial(),K=K,g_tol=1e-7,iterations=800)
theta(f)=vcat(f.β,GLLVModels.pack_lambda(f.Λ),log.(f.r_group))
function error_name(f)
    try f(); "NO_ERROR" catch e;string(typeof(e)) end
end
badrows=error_name(()->gllvm(@formula(y ~ 1),Y,(site=collect(1:n-1),);
    family=GLLVModels.NegativeBinomial(),K=K,g_tol=1e-7,iterations=800))
missinglong=map(x->x[2:end],long)
duplicatelong=map(x->vcat(x,x[1]),long)
missingerror=error_name(()->gllvm(@formula(y ~ 1),missinglong;family=GLLVModels.NegativeBinomial(),K=K))
duplicateerror=error_name(()->gllvm(@formula(y ~ 1),duplicatelong;family=GLLVModels.NegativeBinomial(),K=K))
report=Dict("native_parameters"=>theta(native),"wide_parameters"=>theta(wide),
    "long_parameters"=>theta(longfit),"native_loglik"=>native.loglik,
    "wide_loglik"=>wide.loglik,"long_loglik"=>longfit.loglik,
    "native_converged"=>native.converged,"wide_converged"=>wide.converged,
    "long_converged"=>longfit.converged,"wide_type"=>string(typeof(wide)),
    "long_type"=>string(typeof(longfit)),"wide_hessian"=>string(wide.hessian),
    "long_hessian"=>string(longfit.hessian),"wrong_rows"=>badrows,
    "missing_long"=>missingerror,"duplicate_long"=>duplicateerror,
    "data_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(Float64.(Y))))),
    "native_health_sha256"=>_core070_sha256_file(joinpath(_core070_receipt_dir(),"formula-nb2-health.toml")))
file=joinpath(_core070_receipt_dir(),"nb2-formula.toml")
open(io->TOML.print(io,report),file,"w")
println("NB2_FORMULA_SHA256 ",_core070_sha256_file(file))
@testset "Original NB2 formula model and inputs" begin
    @test native.converged && r.converged
    @test r.health["native_gradient_max"]<=1e-4 && r.health["r_gradient_max"]<=1e-4
    @test abs(r.health["samepoint_delta"])<=1e-6
    @test native.loglik≈r.logLik rtol=1e-6
    for f in (wide,longfit)
        @test f isa NBGroupedFit
        @test f.converged
        @test f.hessian==:observed
        @test length(theta(f))==19
        @test theta(f)≈theta(native) rtol=0 atol=1e-10
        @test f.loglik≈native.loglik rtol=0 atol=1e-10
    end
    @test badrows=="DimensionMismatch"
    @test missingerror=="ArgumentError"
    @test duplicateerror=="ArgumentError"
end
println("NB2_FORMULA_QUALIFIED")

end
