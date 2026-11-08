# Generator for the frozen Tweedie / Beta / Gamma parity data (Refs #739, PR 1037).
#
# These three Y matrices were simulated with Julia's seeded RNG. The stream is NOT stable across
# Julia versions (seed-82 Tweedie differs in 1 of 750 cells between 1.10.12 and 1.13.2), so the
# parity test reads the frozen CSVs below instead of replaying the simulators.
# Generated once on Julia 1.10.12, the version that produced the stored R receipts.
#   julia --project=<env with GLLVModels, Distributions, SHA> test/fixtures/gen_frozen_parity_739.jl <outdir>
# Writes (site,trait,y) CSVs with round-trip Float64 text and prints their sha256 for the TOML.
using GLLVModels, Random, SHA
import Distributions

outdir = isempty(ARGS) ? (@__DIR__) : ARGS[1]
loadings() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]   # parity_loadings_p5k2()

function rand_beta(a::Float64, b::Float64)   # Johnk, copied from test/parity/test_beta_parity.jl
    a = max(a, 1e-12); b = max(b, 1e-12)
    while true
        u = rand(); v = rand()
        x = u^(1 / a); y = v^(1 / b); s = x + y
        s <= 1.0 && s > 0.0 && return x / s
    end
end

function tweedie()
    Random.seed!(82)
    p, K, n = 5, 1, 150
    β = [0.5, -0.2, 0.3, -0.4, 0.1]
    Λ = 0.5 .* loadings()[:, 1:K]
    φ = [0.8, 1.0, 1.2, 0.9, 1.1]
    Z = randn(K, n)
    μ = exp.(β .+ Λ * Z)
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ[t], 1.5, Random.default_rng())
    end
    return Y
end

function beta()
    Random.seed!(45)
    p, K, n = 5, 1, 60
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    Λ = 0.15 .* loadings()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    return [begin
                μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
                rand_beta(μ * 12.0, (1 - μ) * 12.0)
            end for t in 1:p, s in 1:n]
end

function gamma()
    Random.seed!(54)
    p, K, n = 5, 1, 120
    β = log.([2.0, 2.5, 1.8, 2.2, 2.1])
    Λ = 0.2 .* loadings()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    return [begin
                μ = exp(clamp(η[t, s], -4.0, 4.0))
                rand(Distributions.Gamma(2.5, μ / 2.5)) + 1e-6
            end for t in 1:p, s in 1:n]
end

function write_csv(path, Y)
    open(path, "w") do io
        println(io, "site,trait,y")
        for s in axes(Y, 2), t in axes(Y, 1)
            println(io, s, ",", t, ",", repr(Float64(Y[t, s])))
        end
    end
    return bytes2hex(sha256(read(path)))
end

println("julia_version = ", VERSION)
for (name, f) in (("tweedie", tweedie), ("beta", beta), ("gamma", gamma))
    Y = f()
    file = "frozen_parity_739_$(name).csv"
    println(name, " size=", size(Y), " file=", file, " sha256=", write_csv(joinpath(outdir, file), Y))
end
