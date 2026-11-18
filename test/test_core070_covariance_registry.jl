# Wrapped in a module so the registry include/using below does not replace or
# clash with Core070CaseRegistry bindings in Main when other test files share the process.
module Core070CovarianceRegistryTest
using Test, TOML
include(joinpath(@__DIR__, "parity", "core070_case_registry.jl"))
using .Core070CaseRegistry
fixed = ["MODE-ORD-INDEP", "MODE-ORD-COMMON"]
modes = ["FIT-MODE-ORD-DEP"; ["FIT-MODE-$s-$m" for s in ("ANIMAL", "KERNEL") for m in ("INDEP", "COMMON", "DEP")]]
@testset "Required Gaussian covariance registry" begin
    @test Core070CaseRegistry.COVARIANCE_IDS == vcat(fixed,modes)
    @test requested_ids(join(vcat(fixed,modes),',')) == vcat(fixed,modes)
    for group in (fixed,modes)
        @test requested_ids(join(group,',')) == group
        @test_throws ArgumentError requested_ids(first(group))
        @test length(unique(FIXTURES[id] for id in group))==1
    end
    @test validate_manifest(TOML.parsefile(joinpath(@__DIR__,"..","docs","dev-log","core070","frozen-r070-contract.toml")))
end
end
