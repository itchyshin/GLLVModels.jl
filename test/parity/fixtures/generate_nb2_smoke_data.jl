# Draw the NATIVE-06 NB2 family-smoke dataset and store it with its hash.
#
#   julia +1.10.12 test/parity/fixtures/generate_nb2_smoke_data.jl SEED OUT.toml
#
# Same shape as the original NATIVE-06 data (p = 5, K = 2, n = 80, loadings
# 0.30 .* parity_loadings_p5k2(), samplers read from test/parity/test_negbin_parity.jl),
# but with real per-trait overdispersion (NB2 size 1 to 3) and means 5 to 8. The original
# data (nb2_original_data.toml) put traits 1 and 3 at the Poisson boundary on both engines.
# Not every draw of this design is interior: seed 39 was chosen by screening seeds 1-150
# (Julia 1.10.12, frozen b4d5fee64 source built locally) for a draw where Julia converges
# with no boundary trait, both engines agree, and pushing any one trait to the boundary
# and re-optimising lowers the Laplace log-likelihood (by 0.35 at best).
# Julia 1.10 and 1.12+ draw different numbers from the same seed, so the stored draw,
# not the seed, is the fixture.
using Random, SHA, TOML

const NB2_SMOKE_MEANS = [6.0, 8.0, 5.0, 7.0, 6.0]
const NB2_SMOKE_R_TRUE = [1.0, 1.5, 2.0, 2.5, 3.0]
const NB2_SMOKE_N = 80

let root = dirname(dirname(dirname(@__DIR__)))
    source = read(joinpath(root, "test", "parity", "test_negbin_parity.jl"), String)
    helpers = source[findfirst("function _rand_poisson", source).start:findfirst("@testset \"NB2 GLLVModels", source).start-1]
    include_string(@__MODULE__, helpers, "test_negbin_parity.jl samplers")
end

function draw_nb2_smoke(seed::Integer; n::Integer = NB2_SMOKE_N)
    p, K = 5, 2
    loadings = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]   # parity_loadings_p5k2()
    Random.seed!(seed)
    β = log.(NB2_SMOKE_MEANS)
    Λ = 0.30 .* loadings
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = _rand_nb2(exp(clamp(η[t, s], -8.0, 8.0)), NB2_SMOKE_R_TRUE[t])
    end
    return Y
end

nb2_smoke_sha256(Y) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))

if abspath(PROGRAM_FILE) == @__FILE__
    seed = parse(Int, ARGS[1]); out = ARGS[2]
    Y = draw_nb2_smoke(seed)
    h = nb2_smoke_sha256(Y)
    open(out, "w") do io
        println(io, "# NATIVE-06 NB2 family-smoke data (replaces nb2_original_data.toml for this cell only).")
        println(io, "# Drawn by test/parity/fixtures/generate_nb2_smoke_data.jl; means $(NB2_SMOKE_MEANS),")
        println(io, "# per-trait NB2 size r_true $(NB2_SMOKE_R_TRUE), loadings 0.30 .* parity_loadings_p5k2().")
        println(io, "# Command: julia +$(VERSION) test/parity/fixtures/generate_nb2_smoke_data.jl $seed <out>")
        TOML.print(io, Dict("p" => 5, "K" => 2, "n" => size(Y, 2), "seed" => seed,
            "means" => NB2_SMOKE_MEANS, "r_true" => NB2_SMOKE_R_TRUE,
            "drawn_with_julia" => string(VERSION), "data_sha256" => h,
            "Y_column_major" => vec(Y)); sorted = true)
    end
    println("wrote ", out, " sha256=", h)
end
