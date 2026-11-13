# frozen_rng.jl — loader for stored (not re-simulated) parity data.
#
# Included by parity_helpers.jl. Needs only TOML and SHA, so it can be included alone.
# Julia's seeded streams differ between 1.10.12 and 1.13.2, so cells whose R receipt is
# bound to one draw read the draw from test/parity/fixtures/frozen_rng_*.toml, written by
# fixtures/generate_frozen_rng_w15.jl. Both the stored hash and the hash pinned in the test
# must match the bytes read; a mismatch is an error, never a silent re-simulation.
using TOML, SHA

_frozen_rng_sha256(a) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(a)))))

function parity_load_frozen_rng(name::AbstractString; expected_sha::AbstractString,
                                expected_x_sha::Union{Nothing, AbstractString} = nothing)
    t = TOML.parsefile(joinpath(@__DIR__, "fixtures", name))
    p, n = Int(t["p"]), Int(t["n"])
    Y = reshape(Float64.(t["Y_column_major"]), p, n)
    got = _frozen_rng_sha256(Y)
    (got == t["data_sha256"] == expected_sha) ||
        error("frozen data $name: sha256 $got does not match stored $(t["data_sha256"]) / pinned $expected_sha")
    x = nothing
    if expected_x_sha !== nothing
        x = Float64.(t["x"])
        gx = _frozen_rng_sha256(x)
        (gx == t["x_sha256"] == expected_x_sha) ||
            error("frozen data $name: x sha256 $gx does not match stored $(t["x_sha256"]) / pinned $expected_x_sha")
    end
    return (; Y, x)
end
