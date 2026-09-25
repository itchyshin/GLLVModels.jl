# Draw one dataset of the Beta grouped-precision sibling screen (#480) and store it with its hash.
#
#   julia +1.10.12 --project=<test env> tools/beta_grouped_screen_data_draw.jl D OUT.toml
#
# Design (the screen that found #480): p = 5, K = 2, n = 80,
# Λ = 0.30 .* [0.8 0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3], β = [-1.5, -0.75, 0, 0.75, 1.5],
# z_i ~ N(0, I_2), y_ti ~ Beta(μ_ti φ, (1 - μ_ti) φ) on the logit link, clamped to
# [1e-12, 1 - 1e-12]. φ = 2 for D in 1:5 and φ = 50 for D in 6:10. RNG MersenneTwister(20260924 + D).
# Beta draws depend on the Julia and Distributions versions, so the stored draw, not the
# seed, is the fixture; the file records both versions.
using Random, SHA, TOML
import Distributions
import Pkg

d = parse(Int, ARGS[1]); out = ARGS[2]
p, K, n = 5, 2, 80
Λ = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
β = [-1.5, -0.75, 0.0, 0.75, 1.5]
φ = d <= 5 ? 2.0 : 50.0
rng = MersenneTwister(20260924 + d)
Y = Matrix{Float64}(undef, p, n)
for i in 1:n
    z = randn(rng, K)
    η = β .+ Λ * z
    for t in 1:p
        μ = 1 / (1 + exp(-η[t]))
        Y[t, i] = clamp(rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ)), 1e-12, 1 - 1e-12)
    end
end
h = bytes2hex(sha256(reinterpret(UInt8, vec(Y))))
dist_version = string(Pkg.dependencies()[Base.UUID("31c24e10-a181-5473-b8eb-7969acd0382f")].version)
open(out, "w") do io
    println(io, "# Beta grouped-precision screen dataset (#480), drawn by tools/beta_grouped_screen_data_draw.jl.")
    println(io, "# Command: julia +$(VERSION) --project=<test env> tools/beta_grouped_screen_data_draw.jl $d <out>")
    TOML.print(io, Dict("p" => p, "K" => K, "n" => n, "dataset" => d, "seed" => 20260924 + d,
        "phi_true" => φ, "drawn_with_julia" => string(VERSION),
        "drawn_with_distributions" => dist_version, "data_sha256" => h,
        "Y_column_major" => vec(Y)); sorted = true)
end
# The stored text must give back the same bits.
back = TOML.parsefile(out)
bytes2hex(sha256(reinterpret(UInt8, Float64.(back["Y_column_major"])))) == h ||
    error("TOML round trip changed the data")
println("wrote ", out, " sha256=", h)
