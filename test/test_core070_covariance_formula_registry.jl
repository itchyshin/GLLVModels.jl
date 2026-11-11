# Wrapped in a module so the registry include/using below does not replace or
# clash with Core070CaseRegistry bindings in Main when other test files share the process.
module Core070CovarianceFormulaRegistryTest
using Test,TOML
include(joinpath(@__DIR__,"parity","core070_case_registry.jl"))
using .Core070CaseRegistry
@testset "Covariance formula registration and native dependencies" begin
    @test Core070CaseRegistry.COVARIANCE_FORMULA_IDS == COVARIANCE_IDS .* "-FORMULA-INTERFACE"
    for group in (COVARIANCE_FIXED_IDS,COVARIANCE_MODE_IDS)
        formulas=group .* "-FORMULA-INTERFACE"
        @test_throws ArgumentError requested_ids(join(formulas,','))
        @test_throws ArgumentError requested_ids(join(vcat(group,[first(formulas)]),','))
        @test requested_ids(join(vcat(group,formulas),','))==vcat(group,formulas)
    end
    @test validate_manifest(TOML.parsefile(joinpath(@__DIR__,"..","docs/dev-log/core070/frozen-r070-contract.toml")))
end
end
