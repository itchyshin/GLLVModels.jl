# Draw a zero-truncated NB2 dataset with the NATIVE-12 design (per-trait dispersion,
# same intercepts/loadings/DGP shape as test/parity/test_truncated_nbinom2_parity.jl
# and the pre-fix tools/core070_second_order/cells.jl cell_truncated_nbinom2), and
# store it with its hash so every Julia version fits the same numbers (Julia 1.10 and
# 1.12+ draw different numbers from the same seed:
# docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md).
#
#   julia +1.10.0 tools/truncnb2_parity_data_draw.jl SEED R_TRUE OUT.toml N
#
# Screened (docs/dev-log/core070/truncnb2-interior-screen-20260925.md) against R's
# per-trait log_phi_truncnb2 (gllvmTMB's only mode for this family) so every trait has
# finite, interior dispersion on BOTH engines (no Poisson-limit boundary trait, unlike
# the frozen NATIVE-12 fixture at n=120, whose se=TRUE per-trait R fit pushes one trait
# to phi ~ 1e7-1e8).
using Random, SHA, TOML, Distributions

parity_loadings_p5k2() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]

seed = parse(Int, ARGS[1]); r_true = parse(Float64, ARGS[2]); out = ARGS[3]
n = parse(Int, ARGS[4])
p, K = 5, 1
Random.seed!(seed)
β = log.([4.0, 5.0, 3.5, 4.5, 4.0])
Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
Z = randn(K, n)
η = β .+ Λ * Z
Y = Matrix{Int}(undef, p, n)
for t in 1:p, s in 1:n
    μ = exp(clamp(η[t, s], -3.0, 3.5))
    while true                      # zero-truncated draw by rejection
        v = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
        if v >= 1
            Y[t, s] = v
            break
        end
    end
end
h = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
open(out, "w") do io
    println(io, "# Zero-truncated NB2 data drawn by tools/truncnb2_parity_data_draw.jl")
    println(io, "# (NATIVE-12 design and samplers: p=5,K=1, beta=log([4,5,3.5,4.5,4]),")
    println(io, "# Lambda=0.2*parity_loadings_p5k2()[:,1:1]). Screened so every trait has")
    println(io, "# finite, interior per-trait dispersion on both engines (docs/dev-log/core070/")
    println(io, "# truncnb2-interior-screen-20260925.md).")
    println(io, "# Command: julia +$(VERSION) tools/truncnb2_parity_data_draw.jl $seed $r_true <out> $n")
    TOML.print(io, Dict("p" => p, "K" => K, "n" => n, "seed" => seed, "r_true" => r_true,
        "drawn_with_julia" => string(VERSION), "data_sha256" => h, "Y_column_major" => vec(Y)); sorted = true)
end
println("wrote ", out, " sha256=", h)
