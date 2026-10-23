# Draw an NB2 dataset with the NATIVE-06 design and samplers, and store it with its hash.
#
#   julia +1.10.12 tools/nb2_parity_data_draw.jl SEED R_TRUE OUT.toml [N]
#
# Design as in test/parity/test_negbin_parity.jl: p = 5, K = 2, per-trait intercepts
# log.([2.5, 3.0, 2.0, 2.8, 2.2]), loadings 0.30 .* parity_loadings_p5k2(), shared true
# dispersion R_TRUE, n = 80 unless N is given. The samplers are read from that file so they
# stay identical. Julia 1.10 and 1.12+ draw different numbers from the same seed, so the
# stored draw, not the seed, is the fixture (docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md).
using Random, SHA, TOML

const ROOT = dirname(@__DIR__)
source = read(joinpath(ROOT, "test", "parity", "test_negbin_parity.jl"), String)
helpers = source[findfirst("function _rand_poisson", source).start:findfirst("@testset \"NB2 GLLVModels", source).start-1]
include_string(Main, helpers, "test_negbin_parity.jl samplers")

parity_loadings_p5k2() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]

seed = parse(Int, ARGS[1]); r_true = parse(Float64, ARGS[2]); out = ARGS[3]
n = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 80
p, K = 5, 2
Random.seed!(seed)
β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
Λ = 0.30 .* parity_loadings_p5k2()
Z = randn(K, n)
η = β .+ Λ * Z
Y = Matrix{Int}(undef, p, n)
for t in 1:p, s in 1:n
    Y[t, s] = _rand_nb2(exp(clamp(η[t, s], -8.0, 8.0)), r_true)
end
h = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
open(out, "w") do io
    println(io, "# NB2 data drawn by tools/nb2_parity_data_draw.jl (NATIVE-06 design and samplers).")
    println(io, "# Command: julia +$(VERSION) tools/nb2_parity_data_draw.jl $seed $r_true <out> $n")
    TOML.print(io, Dict("p" => p, "K" => K, "n" => n, "seed" => seed, "r_true" => r_true,
        "drawn_with_julia" => string(VERSION), "data_sha256" => h, "Y_column_major" => vec(Y)); sorted = true)
end
println("wrote ", out, " sha256=", h)
