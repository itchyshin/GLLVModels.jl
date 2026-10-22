# test_core070_pin.jl -- plain unit test for test/parity/core070_pin.jl.
#
# No RCall, no R install: core070_pin.jl depends only on TOML + the standard
# library, so this test spawns small subprocesses (needed because the pin is
# resolved into `const`s at include-time -- a fresh process per scenario is
# the simplest way to exercise both GLLVM_PARITY_PIN=<unset> and ="P0"/"P1" without
# redefining consts). Not included by runtests.jl or runparity.jl; run
# directly:
#   julia --project=. test/parity/test_core070_pin.jl

using Test, TOML

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
const PIN_FILE = joinpath(HERE, "core070_pin.jl")
const CONTRACT_FILE = joinpath(ROOT, "docs/dev-log/core070/frozen-r070-contract.toml")
const P0_COMMIT = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const P1_COMMIT = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const P0_VERSION = "0.7.0"
const P1_VERSION = "0.7.1"
const PIN_TABLE = TOML.parsefile(joinpath(ROOT, "tools/core070_oracle_pins.toml"))

function build_receipt_text(pin_name::AbstractString; version::Union{Nothing,AbstractString} = nothing)
    pin = PIN_TABLE[pin_name]
    fields = ["reference_commit" => pin["reference_commit"],
              "source_tree_sha256" => pin["source_tree_sha256"],
              "archive_sha256" => pin["archive_sha256"]]
    version === nothing || push!(fields, "gllvmTMB_version" => String(version))
    rows = ["  \"$key\": \"$value\"" for (key, value) in fields]
    return "{\n" * join(rows, ",\n") * "\n}\n"
end

function check_build_receipt(pin::AbstractString, text::AbstractString)
    path = tempname()
    write(path, text)
    try
        return run_pin_script("""
            include(raw"$PIN_FILE")
            _core070_check_oracle_receipt(read(raw"$path", String), "synthetic-build.json", :build)
            println("BUILD_VERSION_OK")
            """; pin = pin)
    finally
        rm(path; force = true)
    end
end

function run_pin_script(script::AbstractString; pin::Union{Nothing,AbstractString} = nothing,
                        oracle_build::Union{Nothing,AbstractString} = nothing)
    env = copy(ENV)
    delete!(env, "GLLVM_PARITY_PIN")
    delete!(env, "GLLVM_PARITY_ORACLE_BUILD")
    pin === nothing || (env["GLLVM_PARITY_PIN"] = pin)
    oracle_build === nothing || (env["GLLVM_PARITY_ORACLE_BUILD"] = oracle_build)
    out = IOBuffer()
    err = IOBuffer()
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e $script`, env)
    proc = run(pipeline(cmd; stdout = out, stderr = err); wait = false)
    wait(proc)
    return (success = proc.exitcode == 0, stdout = String(take!(out)), stderr = String(take!(err)))
end

@testset "core070_pin.jl" begin
    @testset "P1 is the default" begin
        r = run_pin_script("""include(raw"$PIN_FILE"); println(_CORE070_REFERENCE_COMMIT)""")
        @test r.success
        @test strip(r.stdout) == P1_COMMIT
    end

    @testset "GLLVM_PARITY_PIN=P0 selects the P0 commit" begin
        r = run_pin_script("""include(raw"$PIN_FILE"); println(_CORE070_REFERENCE_COMMIT)"""; pin = "P0")
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

    @testset "Gaussian random-effect contract is selected by exact source and version" begin
        script = """
            include(raw"$PIN_FILE")
            println(join(_core070_gaussian_random_contract("$P0_COMMIT", "$P0_VERSION"), ","))
            println(join(_core070_gaussian_random_contract("$P1_COMMIT", "$P1_VERSION"), ","))
            """
        r = run_pin_script(script)
        @test r.success
        @test split(strip(r.stdout), "\n") == ["z_B,s_B", "z_B"]

        for script in (
            "include(raw\"$PIN_FILE\"); _core070_gaussian_random_contract(\"$P0_COMMIT\", \"$P1_VERSION\")",
            "include(raw\"$PIN_FILE\"); _core070_gaussian_random_contract(\"$P1_COMMIT\", \"$P0_VERSION\")",
            "include(raw\"$PIN_FILE\"); _core070_gaussian_random_contract(repeat(\"0\", 40), \"$P0_VERSION\")",
        )
            r = run_pin_script(script)
            @test !r.success
            @test occursin("Gaussian random-effect", r.stderr)
        end
    end

    @testset "installed package version and marker must prove the selected oracle version" begin
        for (pin, version, commit) in (("P0", P0_VERSION, P0_COMMIT), ("P1", P1_VERSION, P1_COMMIT))
            valid = run_pin_script("""
                include(raw"$PIN_FILE")
                _core070_check_installed_package_version("$version", "$version")
                println("INSTALLED_VERSION_OK $commit")
                """; pin = pin)
            @test valid.success
            @test occursin("INSTALLED_VERSION_OK $commit", valid.stdout)
        end
        for (pin, expected, wrong) in (("P0", P0_VERSION, P1_VERSION), ("P1", P1_VERSION, P0_VERSION))
            wrong_installed = run_pin_script("""
                include(raw"$PIN_FILE")
                _core070_check_installed_package_version("$expected", "$wrong")
                """; pin = pin)
            @test !wrong_installed.success
            @test occursin("gllvmTMB version", wrong_installed.stderr)
            missing_marker = run_pin_script("""
                include(raw"$PIN_FILE")
                _core070_check_installed_package_version(nothing, "$expected")
                """; pin = pin)
            @test !missing_marker.success
            @test occursin("rebuild the pinned oracle", missing_marker.stderr)
        end
    end

    @testset "build receipt version is required and must match its exact pin" begin
        for (pin_name, version, commit) in (("P0", P0_VERSION, P0_COMMIT), ("P1", P1_VERSION, P1_COMMIT))
            valid = check_build_receipt(pin_name, build_receipt_text(pin_name; version = version))
            @test valid.success
            @test occursin("BUILD_VERSION_OK", valid.stdout)

            wrong = pin_name == "P0" ? P1_VERSION : P0_VERSION
            wrong_version = check_build_receipt(pin_name, build_receipt_text(pin_name; version = wrong))
            @test !wrong_version.success
            @test occursin("gllvmTMB_version", wrong_version.stderr)

            legacy = check_build_receipt(pin_name, build_receipt_text(pin_name))
            @test !legacy.success
            @test occursin("rebuild the pinned oracle", legacy.stderr)
        end
    end

    @testset "frozen-r070-contract.toml (P0) + P0 selection passes" begin
        r = run_pin_script("""
            include(raw"$PIN_FILE")
            manifest = TOML.parsefile(raw"$CONTRACT_FILE")
            _core070_check_frozen_contract_pin(manifest, raw"$CONTRACT_FILE")
            println("PIN_GUARD_OK")
            """; pin = "P0")
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

    @testset "runner manifest path follows the pin; each selected manifest passes its own guard" begin
        script = """
            include(raw"$PIN_FILE")
            rel = _core070_frozen_contract_rel()
            path = joinpath(raw"$ROOT", rel)
            _core070_check_frozen_contract_pin(TOML.parsefile(path), path)
            println(rel)
            """
        r0 = run_pin_script(script; pin = "P0")
        @test r0.success
        @test strip(r0.stdout) == "docs/dev-log/core070/frozen-r070-contract.toml"
        r1 = run_pin_script(script; pin = "P1")
        @test r1.success
        @test strip(r1.stdout) == "docs/dev-log/core070/true-parity-latest/frozen-r070-contract-p1.toml"
    end

    @testset "legacy oracle build receipts fail closed until rebuilt with version metadata" begin
        script = """
            include(raw"$PIN_FILE")
            rel = _core070_oracle_receipts_rel()
            for kind in (:build, :source)
                path = joinpath(raw"$ROOT", getfield(rel, kind))
                _core070_check_oracle_receipt(read(path, String), getfield(rel, kind), kind)
            end
            println(rel.build)
            """
        r1 = run_pin_script(script; pin = "P1")
        @test !r1.success
        @test occursin("gllvmTMB_version", r1.stderr)
        @test occursin("rebuild", r1.stderr)
        source_only = run_pin_script("""
            include(raw"$PIN_FILE")
            rel = _core070_oracle_receipts_rel()
            path = joinpath(raw"$ROOT", rel.source)
            _core070_check_oracle_receipt(read(path, String), rel.source, :source)
            println(rel.build)
            """; pin = "P1")
        @test source_only.success
        @test strip(source_only.stdout) == "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json"
        # P0 selection keeps the historical .unlazy paths.
        r0 = run_pin_script("""include(raw"$PIN_FILE"); println(_core070_oracle_receipts_rel().build)"""; pin = "P0")
        @test r0.success
        @test strip(r0.stdout) == ".unlazy/core070-aghq/oracle-receipts/build.json"
    end

    @testset "a registered host build swaps only the build receipt; an unknown host fails loud" begin
        script = """
            include(raw"$PIN_FILE")
            rel = _core070_oracle_receipts_rel()
            path = joinpath(raw"$ROOT", rel.source)
            _core070_check_oracle_receipt(read(path, String), rel.source, :source)
            println(rel.build); println(rel.source)
            """
        t = run_pin_script(script; pin = "P1", oracle_build = "totoro")
        @test t.success
        @test split(strip(t.stdout), "\n") ==
              ["docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro.json",
               "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"]
        bad = run_pin_script(script; pin = "P1", oracle_build = "elsewhere")
        @test !bad.success
        @test occursin("GLLVM_PARITY_ORACLE_BUILD", bad.stderr)
        p0 = run_pin_script(script; pin = "P0", oracle_build = "totoro")
        @test !p0.success
    end

    @testset "a P1 oracle receipt used under P0 fails loud" begin
        build = joinpath(ROOT, "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json")
        r = run_pin_script("""
            include(raw"$PIN_FILE")
            _core070_check_oracle_receipt(read(raw"$build", String), "build.json", :build)
            """; pin = "P0")
        @test !r.success
        @test occursin("reference_commit", r.stderr)
        @test occursin(P1_COMMIT, r.stderr)
    end

    @testset "a tampered or ambiguous oracle receipt fails loud" begin
        build = joinpath(ROOT, "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json")
        tampered = run_pin_script("""
            include(raw"$PIN_FILE")
            text = replace(read(raw"$build", String), r"\"source_tree_sha256\": \"[0-9a-f]+\"" => "\"source_tree_sha256\": \"$(repeat("0", 64))\"")
            _core070_check_oracle_receipt(text, "build.json", :build)
            """; pin = "P1")
        @test !tampered.success
        @test occursin("source_tree_sha256", tampered.stderr)
        partial = tempname()
        write(partial, "{\n  \"reference_commit\": \"$P1_COMMIT\"\n}\n")
        missing_key = run_pin_script("""
            include(raw"$PIN_FILE")
            _core070_check_oracle_receipt(read(raw"$partial", String), "build.json", :build)
            """; pin = "P1")
        rm(partial)
        @test !missing_key.success
        @test occursin("expected exactly one", missing_key.stderr)
    end
end
