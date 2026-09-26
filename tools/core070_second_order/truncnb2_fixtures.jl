using TOML
using SHA

# Zero-truncated NB2 interior fixture for the `truncated_nbinom2` second-order cell.
# Stored, not redrawn from a seed -- Julia 1.10 and 1.12+ draw different numbers from
# the same seed (docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-
# pins.md), same discipline as `test/parity/fixtures/nb2_original_data.toml`.
# Screened (tools/truncnb2_parity_data_draw.jl;
# docs/dev-log/core070/truncnb2-interior-screen-20260925.md) so every trait has finite,
# interior per-trait dispersion on BOTH engines -- unlike the frozen NATIVE-12 fixture
# (test/parity/test_truncated_nbinom2_parity.jl, seed=58, n=120), whose se=TRUE
# per-trait R fit pushes one trait to the Poisson-limit boundary (phi ~ 1e7-1e8).
function truncnb2_interior_fixture()
    path = joinpath(@__DIR__, "..", "..", "test", "parity", "fixtures",
                     "truncnb2_interior_seed61_n150.toml")
    d = TOML.parsefile(path)
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    h = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
    h == d["data_sha256"] == "7be429abef925c940bf29a6aebfc4d24af900efc7b49ce6f5aece7356838ae62" ||
        error("stored truncated-NB2 interior data changed")
    return (; Y, p = d["p"], K = d["K"], n = d["n"], seed = d["seed"])
end

# Same discipline as truncnb2_interior_fixture() above, for the paired boundary draw
# used by the per-trait second-order Wald CI test (pr-493 review "seed 62/n150"):
# trait 1's r sits at the Poisson limit on both engines, so the joint Wald Hessian is
# not positive definite and r[1] must be conditioned out. Drawn with the same
# tools/truncnb2_parity_data_draw.jl generator; stored, not redrawn, for the same
# reason -- Julia 1.10 and 1.13 draw different numbers from the same seed.
function truncnb2_boundary_fixture()
    path = joinpath(@__DIR__, "..", "..", "test", "parity", "fixtures",
                     "truncnb2_boundary_seed62_n150.toml")
    d = TOML.parsefile(path)
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    h = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
    h == d["data_sha256"] ||
        error("stored truncated-NB2 boundary data changed")
    return (; Y, p = d["p"], K = d["K"], n = d["n"], seed = d["seed"])
end
