# test_core070_pin.jl -- plain unit test for test/parity/core070_pin.jl.
#
# No RCall, no R install: core070_pin.jl depends only on TOML + the standard
# library, so this test spawns small subprocesses (needed because the pin is
# resolved into `const`s at include-time -- a fresh process per scenario is
# the simplest way to exercise both GLLVM_PARITY_PIN=<unset> and ="P1" without
# redefining consts). Not included by runtests.jl or runparity.jl; run
# directly:
#   julia --project=. test/parity/test_core070_pin.jl

using Test

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
const PIN_FILE = joinpath(HERE, "core070_pin.jl")
const CONTRACT_FILE = joinpath(ROOT, "docs/dev-log/core070/frozen-r070-contract.toml")
const P0_COMMIT = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const P1_COMMIT = "9539352f66f2db2cc26b1c393e67212a359b60c9"

function run_pin_script(script::AbstractString; pin::Union{Nothing,AbstractString} = nothing)
    env = copy(ENV)
    delete!(env, "GLLVM_PARITY_PIN")
    pin === nothing || (env["GLLVM_PARITY_PIN"] = pin)
    out = IOBuffer()
    err = IOBuffer()
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e $script`, env)
    proc = run(pipeline(cmd; stdout = out, stderr = err); wait = false)
    wait(proc)
    return (success = proc.exitcode == 0, stdout = String(take!(out)), stderr = String(take!(err)))
end

@testset "core070_pin.jl" begin
    @testset "P0 is the default" begin
        r = run_pin_script("""include(raw"$PIN_FILE"); println(_CORE070_REFERENCE_COMMIT)""")
        @test r.success
        @test strip(r.stdout) == P0_COMMIT
    end

    @testset "GLLVM_PARITY_PIN=P1 selects the P1 commit" begin
        r = run_pin_script("""include(raw"$PIN_FILE"); println(_CORE070_REFERENCE_COMMIT)"""; pin = "P1")
        @test r.success
        @test strip(r.stdout) == P1_COMMIT
    end

    @testset "an unrecognized pin fails loud" begin
        r = run_pin_script("""include(raw"$PIN_FILE")"""; pin = "bogus")
        @test !r.success
        @test occursin("GLLVM_PARITY_PIN", r.stderr)
    end

    @testset "frozen-r070-contract.toml (P0) + P0 selection passes" begin
        r = run_pin_script("""
            include(raw"$PIN_FILE")
            manifest = TOML.parsefile(raw"$CONTRACT_FILE")
            _core070_check_frozen_contract_pin(manifest, raw"$CONTRACT_FILE")
            println("PIN_GUARD_OK")
            """)
        @test r.success
        @test occursin("PIN_GUARD_OK", r.stdout)
    end

    @testset "frozen-r070-contract.toml (P0) + P1 selection fails loud, naming both SHAs" begin
        r = run_pin_script("""
            include(raw"$PIN_FILE")
            manifest = TOML.parsefile(raw"$CONTRACT_FILE")
            _core070_check_frozen_contract_pin(manifest, raw"$CONTRACT_FILE")
            """; pin = "P1")
        @test !r.success
        @test occursin("GLLVM_PARITY_PIN", r.stderr)
        @test occursin(P0_COMMIT, r.stderr)
        @test occursin(P1_COMMIT, r.stderr)
    end
end
