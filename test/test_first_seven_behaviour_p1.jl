# gllvm-parity-tag: P1
# Fast, fit-free checks for the dedicated public-door runners. The measured
# fit calls are run only by tools/first_seven_behaviour_{R,J}.R/.jl.
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const JRUN = read(joinpath(ROOT, "tools", "first_seven_behaviour_J.jl"), String)
const RRUN = read(joinpath(ROOT, "tools", "first_seven_behaviour_R.R"), String)
const DERIVE = read(joinpath(ROOT, "tools", "first_seven_behaviour_derive.py"), String)

@testset "first-seven public-door runner controls (no fits)" begin
    for id in ("ISDM-COUNT", "ISDM-EXTRA-SOURCE", "ISDM-MISSING-IN-TRAIT",
               "ISDM-MISSING-SOURCE", "ISDM-WRAPPER-LAW")
        @test occursin(id, JRUN)
        @test occursin(id, RRUN)
        @test occursin(id, DERIVE)
    end
    @test occursin("fit_isdm_gllvm(formula, data; family=family)", JRUN)
    @test occursin("gllvmTMB(f, data=dx, family=fam()", RRUN)
    @test occursin("length(family)", DERIVE)
    @test occursin("CORE070_FIRST7_DERIVATION_SELFTEST_OK", DERIVE)
end
