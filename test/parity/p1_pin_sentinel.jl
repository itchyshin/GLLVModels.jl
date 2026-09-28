# gllvm-parity-tag: P1
#
# Sentinel P1 twin test (placeholder until real P1-pinned twin tests land).
#
# Confirms the P1 gllvmTMB pin recorded in tools/parity_oracle.py is exactly
# the SHA the maintainer re-targeted to (D-294/D-295), so
# .github/workflows/parity-p1-twin.yml has a real tagged file to discover and
# run instead of passing on an empty file list.
#
# Pure Julia, no R, no package dependencies beyond the Test stdlib -- this
# job covers Julia-only P1 twin tests. R-backed P1 twins (fitting against a
# live gllvmTMB checkout) route through CI.yml's `test-parity` job once that
# job is re-pinned from P0 to P1 (a separate, later PR); this file does not
# attempt that.
using Test

const EXPECTED_P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"

@testset "P1 pin sentinel" begin
    oracle_path = joinpath(@__DIR__, "..", "..", "tools", "parity_oracle.py")
    contents = read(oracle_path, String)
    m = match(r"""P1_GLLVMTMB_ORACLE\s*=\s*["']([0-9a-f]{40})["']""", contents)
    @test m !== nothing
    if m !== nothing
        @test m.captures[1] == EXPECTED_P1_SHA
    end
end
