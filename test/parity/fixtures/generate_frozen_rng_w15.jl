# Draw the simulated Y of three opt-in parity cells once and store it with its hash.
#
#   julia +1.10.12 --project=. test/parity/fixtures/generate_frozen_rng_w15.jl [OUT_DIR]
#
# Cells (DGP lines copied unchanged from the parity files; only the draw is frozen):
#   frozen_rng_gaussian_s42.toml        test/parity/test_gaussian_parity.jl      seed 42,  p5 K2 n80
#   frozen_rng_xcov_gaussian_s420.toml  test/parity/test_x_covariate_parity.jl   seed 420, p5 K2 n30 (x and y)
#   frozen_rng_tweedie_s82.toml         test/parity/test_tweedie_parity.jl       seed 82,  p5 K1 n150
# Julia 1.10.12 and 1.13.2 draw different numbers from the same seed, so the stored draw,
# not the seed, is the fixture. Values are written with TOML's shortest round-trip Float64
# text; data_sha256 is the sha256 of the raw Float64 bytes. The Gaussian cells store the arrays
# AFTER the test's centring/demeaning: `mean` itself differs at the last bit between 1.10.12 and
# 1.13.2 (2.2e-16 on the seed-420 cell, whose raw randn draws are identical), so centring at run
# time would feed the two versions inputs that are not bit-identical. Refuses to run on any Julia other
# than 1.10.12 (the version behind the stored R receipts) unless FREEZE_ANY_VERSION=1.
using Random, SHA, TOML, Statistics, GLLVModels

if VERSION != v"1.10.12" && get(ENV, "FREEZE_ANY_VERSION", "0") != "1"
    error("generate_frozen_rng_w15.jl must run on Julia 1.10.12 (running $(VERSION)); " *
          "set FREEZE_ANY_VERSION=1 only to inspect drift")
end

frozen_rng_sha256(a) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(a)))))
parity_loadings_p5k2() = [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]

function draw_gaussian_s42()
    Random.seed!(42)
    p, K, n = 5, 2, 80
    Λ_true = parity_loadings_p5k2()
    σ_true = 0.7
    η = randn(K, n)
    y = Λ_true * η + σ_true * randn(p, n)
    y .-= mean(y; dims = 2)               # per-trait centring, as in the test
    return (; p, K, n, seed = 42, y)
end

function draw_xcov_gaussian_s420()
    Random.seed!(420)
    p, K, n = 5, 2, 30
    Λ = parity_loadings_p5k2()
    γ = 0.55
    x = randn(n)
    x .-= mean(x)
    Z = randn(K, n)
    y = γ .* x' .+ Λ * Z .+ 0.7 .* randn(p, n)
    y .-= mean(y; dims = 2)
    return (; p, K, n, seed = 420, y, x)
end

function draw_tweedie_s82()
    Random.seed!(82)
    p, K, n = 5, 1, 150
    β_true = [0.5, -0.2, 0.3, -0.4, 0.1]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    φ_true = [0.8, 1.0, 1.2, 0.9, 1.1]
    p_true = 1.5
    Z = randn(K, n)
    μ = exp.(β_true .+ Λ_true * Z)
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ_true[t], p_true, Random.default_rng())
    end
    return (; p, K, n, seed = 82, y = Y)
end

function write_frozen(out, d, desc)
    h = frozen_rng_sha256(d.y)
    open(out, "w") do io
        println(io, "# ", desc)
        println(io, "# Julia's seeded stream differs between 1.10.12 and 1.13.2, so this draw is stored, not re-simulated.")
        println(io, "# Drawn by test/parity/fixtures/generate_frozen_rng_w15.jl; Y_column_major is the p x n matrix, column-major.")
        println(io, "# Command: julia +$(VERSION) test/parity/fixtures/generate_frozen_rng_w15.jl <out_dir>")
        rec = Dict{String, Any}("p" => d.p, "K" => d.K, "n" => d.n, "seed" => d.seed,
            "drawn_with_julia" => string(VERSION), "data_sha256" => h,
            "Y_column_major" => vec(d.y))
        if hasproperty(d, :x)
            rec["x_sha256"] = frozen_rng_sha256(d.x)
            rec["x"] = d.x
        end
        TOML.print(io, rec; sorted = true)
    end
    xinfo = hasproperty(d, :x) ? " x_sha256=" * frozen_rng_sha256(d.x) : ""
    println("wrote ", out, " y_sha256=", h, xinfo)
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir = isempty(ARGS) ? @__DIR__() : ARGS[1]
    mkpath(dir)
    write_frozen(joinpath(dir, "frozen_rng_gaussian_s42.toml"), draw_gaussian_s42(),
        "NATIVE-01-GAUSSIAN Y (seed 42, p = 5, K = 2, n = 80), after per-trait centring (the array the fits receive).")
    write_frozen(joinpath(dir, "frozen_rng_xcov_gaussian_s420.toml"), draw_xcov_gaussian_s420(),
        "Gaussian + shared X cell (seed 420, p = 5, K = 2, n = 30): demeaned x and per-trait-centred y (the arrays the fits receive).")
    write_frozen(joinpath(dir, "frozen_rng_tweedie_s82.toml"), draw_tweedie_s82(),
        "NATIVE-07-TWEEDIE Y (seed 82, p = 5, K = 1, n = 150, power 1.5).")
end
