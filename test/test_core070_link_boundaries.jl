using GLLVModels, Test, TOML, SHA
module Core070LinkBoundaries
using GLLVModels, Test, TOML, SHA
struct ResponseRead <: Exception end
struct NoRead <: AbstractMatrix{Int} end
Base.size(::NoRead)=(3,5)
Base.getindex(::NoRead,::Int,::Int)=throw(ResponseRead())
Y=NoRead(); N=fill(5,3,5)
results=Dict{String,Any}()
function check(id,f,expected)
 actual=try f();"RETURNED" catch e;e isa ResponseRead ? "RESPONSE_READ_SENTINEL" : string(nameof(typeof(e))) end
 results[id]=actual
 println(id,'\t',actual)
 @test actual==expected
end
@testset "Native nonreference link admission" begin
check("FAMILY-01-REJECT-LINK",()->fit_gllvm(Y;family=GLLVModels.Binomial(),K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-02-REJECT-LINK",()->fit_poisson_gllvm(Y;K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-03-REJECT-LINK",()->fit_lognormal_gllvm(Y;K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-04-REJECT-LINK",()->fit_gamma_gllvm(Y;K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-05-REJECT-LINK",()->fit_gllvm(Y;family=GLLVModels.NegativeBinomial(),K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-06-REJECT-LINK",()->fit_tweedie_gllvm_grouped(Y;K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-07-REJECT-LINK",()->fit_gllvm(Y;family=GLLVModels.Beta(),K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-08-REJECT-LINK",()->fit_gllvm(Y;family=BetaBinom(),N=N,K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-09-REJECT-LINK",()->fit_gllvm(Y;family=StudentTFamily(),K=1,link=LogLink()),"ArgumentError")
check("FAMILY-10-REJECT-LINK",()->fit_truncated_poisson_gllvm(Y;K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-11-REJECT-LINK",()->fit_truncated_nbinom2_gllvm_pertrait(Y;K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-14-REJECT-LINK",()->fit_ordinal_gllvm_pertrait(Y;K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-15-REJECT-LINK",()->fit_gllvm(Y;family=NB1(),K=1,link=IdentityLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-16-REJECT-LINK",()->fit_multinomial_gllvm(Y;link=IdentityLink()),"ArgumentError")
check("FAMILY-08-CLOGLOG-ENGINE-REJECT",()->fit_gllvm(Y;family=BetaBinom(),N=N,K=1,link=CLogLogLink()),"RESPONSE_READ_SENTINEL")
check("FAMILY-14-MUTATED-DESCRIPTOR-REJECT",()->fit_ordinal_gllvm_pertrait(Y;K=1,link=IdentityLink()),"ArgumentError")
check("FAMILY-16-MUTATED-DESCRIPTOR-REJECT",()->fit_multinomial_gllvm(Y;link=IdentityLink()),"ArgumentError")
check("CONTROL-LOGNORMAL",()->fit_lognormal_gllvm(Y;K=1,link=LogLink()),"RESPONSE_READ_SENTINEL")
check("CONTROL-TRUNCPOIS",()->fit_truncated_poisson_gllvm(Y;K=1,link=LogLink()),"RESPONSE_READ_SENTINEL")
check("CONTROL-TRUNCNB2",()->fit_truncated_nbinom2_gllvm_pertrait(Y;K=1,link=LogLink()),"RESPONSE_READ_SENTINEL")
check("CONTROL-MULTINOMIAL",()->fit_multinomial_gllvm(Y;link=LogitLink()),"RESPONSE_READ_SENTINEL")
end
if haskey(ENV,"CORE070_LINK_OUTPUT")
 file=ENV["CORE070_LINK_OUTPUT"]
 open(io->TOML.print(io,Dict("results"=>results,"scope"=>"ADMISSION_ONLY_NOT_FIT_PARITY",
  "package_root"=>realpath(Base.pkgdir(GLLVModels)),"julia_version"=>string(VERSION))),file,"w")
 println("LINK_BOUNDARY_SHA256 ",bytes2hex(sha256(read(file))))
end
end
