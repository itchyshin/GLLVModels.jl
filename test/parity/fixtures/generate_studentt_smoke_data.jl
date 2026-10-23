# Draw a stored Student-t family-smoke dataset (NATIVE-10) exactly as
# test/parity/test_studentt_parity.jl draws it from its seed, and write it as TOML.
#
#   julia +1.10.12 generate_studentt_smoke_data.jl cell9 <out.toml>          # seed 71
#   julia +1.13.1  generate_studentt_smoke_data.jl near_gaussian <out.toml>  # seed 73
#
# The seeded stream differs between Julia versions, so the parity cells read these stored
# draws instead of re-drawing (as for NB2, docs/dev-log/decisions/
# 2026-09-24-parity-reference-julia-and-fixture-pins.md). Each version above is the one whose
# draw keeps both engines off a degenerate boundary: the 1.13 seed-71 draw puts trait 1 at
# σ → 0, ν → ∞ (R stops with false convergence), and the 1.10 seed-73 draw stops R at the
# ν boundary.
using Random, Distributions, SHA, TOML

which, out = ARGS[1], ARGS[2]
if which == "cell9"
    seed = 71
    Random.seed!(seed)
    p, K, n = 5, 1, 130
    β = [0.2, -0.1, 0.3, 0.0, -0.2]
    Λ = 0.5 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3][:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = η[t, s] + 0.7 * rand(TDist(4.0))
    end
elseif which == "near_gaussian"
    seed = 73
    Random.seed!(seed)
    p, K, n = 3, 1, 400
    β = [0.2, -0.1, 0.3]
    Λ = reshape([0.45, -0.35, 0.25], p, K)
    σ = [0.6, 0.8, 0.7]
    η = β .+ Λ * randn(K, n)
    Y = [η[t, s] + σ[t] * rand(TDist(1.0e6)) for t in 1:p, s in 1:n]
else
    error("which must be cell9 or near_gaussian")
end
d = Dict("which" => which, "seed" => seed, "p" => p, "n" => n,
         "drawn_with_julia" => string(VERSION),
         "Y_column_major" => vec(Y),
         "data_sha256" => bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))))
open(out, "w") do io
    println(io, "# NATIVE-10 Student-t family-smoke data ($which, seed $seed), drawn by")
    println(io, "# test/parity/fixtures/generate_studentt_smoke_data.jl on Julia $(VERSION).")
    TOML.print(io, d; sorted = true)
end
println(d["data_sha256"])
