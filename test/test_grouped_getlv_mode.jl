using GLLVModels, Test, TOML, LinearAlgebra, StableRNGs, Statistics, Distributions

# getLV on a grouped fit must return the mode of the per-site log-posterior the fit's
# Laplace objective was evaluated at (#503 follow-up). Before the fix `_grouped_getLV`
# used the generic `_grouped_laplace_mode`, whose undamped small-step shortcut lets
# Fisher scoring oscillate where the observed curvature exceeds twice the Fisher
# curvature (NB2 with y >> μ; Gamma with y >> μ), so it returned a non-stationary z
# while the likelihood kernels (fixed in #479/#507/#521) used the true mode.

const _GLV_SITE = joinpath(@__DIR__, "fixtures", "nb2_grouped_mode_search_503.toml")
const _GLV_Y    = joinpath(@__DIR__, "fixtures", "nb2_grouped_seed1_Y.toml")

# Closed-form NB2/log and Gamma/log log-posterior gradients (share no code with the kernels).
_nb2_grad(y, Λ, β, r, z) = (μ = exp.(β .+ Λ * z); Λ' * ((y .- μ) .* r ./ (r .+ μ)) .- z)
_gam_grad(y, Λ, β, α, z) = (μ = exp.(β .+ Λ * z); Λ' * (α .* (y ./ μ .- 1)) .- z)

# Independent reference mode: exact observed-curvature Newton (concave for NB2/log).
function _glv_nb2_ref(y, Λ, β, r)
    z = zeros(size(Λ, 2))
    for _ in 1:200
        μ = exp.(β .+ Λ * z)
        W = μ .* r .* (r .+ y) ./ (r .+ μ) .^ 2
        Δ = (Λ' * (W .* Λ) + I) \ _nb2_grad(y, Λ, β, r, z)
        z .+= Δ
        maximum(abs, Δ) < 1e-13 && break
    end
    return z
end

_nbfit(β, Λ, r) = NBGroupedFit(β, Λ, fill(r, length(β)), collect(1:length(β)),
                               GLLVModels.LogLink(), NaN, true, 0, :observed)
_gamfit(β, Λ, α) = GammaGroupedFit(β, Λ, fill(α, length(β)), collect(1:length(β)),
                                   GLLVModels.LogLink(), NaN, true, 0, :observed)
_glv_warm(Z, K) = (β = vec(mean(Z; dims = 2)); F = svd(Z .- β);
                   (F.U[:, 1:K] .* (F.S[1:K]' ./ sqrt(size(Z, 2))), β))

@testset "getLV (NB2 grouped) returns the mode at the 2-cycle fixture site" begin
    d = TOML.parsefile(_GLV_SITE)
    y = Int64.(d["y"]); p, K = d["p"], d["K"]
    Λ = reshape(Float64.(d["Lambda_column_major"]), p, K); β = Float64.(d["beta"]); r = Float64(d["r"])
    zref = _glv_nb2_ref(y, Λ, β, r)
    @test norm(_nb2_grad(y, Λ, β, r, zref)) < 1e-10
    # Premise: the generic kernel is off the mode here (measured 4.7e-4, |grad| 0.066).
    zold = GLLVModels._grouped_laplace_mode([NegativeBinomial(r, 0.5) for _ in 1:p], y,
                                            ones(Int, p), Λ, β, GLLVModels.LogLink())
    @test maximum(abs, zold .- zref) > 1e-4
    z = vec(getLV(_nbfit(β, Λ, r), reshape(y, p, 1); rotate = false))
    @test maximum(abs, z .- zref) < 1e-8
    @test norm(_nb2_grad(y, Λ, β, r, z)) < 1e-7
end

@testset "getLV (NB2 grouped) is stationary at every site of the seed-1 panel" begin
    dY = TOML.parsefile(_GLV_Y)
    Y = permutedims(reshape(Int64.(dY["Y_row_major"]), dY["n"], dY["p"]))
    Λ, β = _glv_warm(log.(Y .+ 0.5), 1)
    r = 0.5                     # measured before the fix: 70/300 sites off the mode
    Z = getLV(_nbfit(β, Λ, r), Y; rotate = false)
    @test maximum(s -> norm(_nb2_grad(Y[:, s], Λ, β, r, Z[s, :])), axes(Y, 2)) < 1e-6
end

@testset "getLV (Gamma grouped) is stationary at every site" begin
    rng = StableRNG(503); p, n, α = 20, 200, 0.5
    Λt = 0.8 .* randn(rng, p, 2); βt = 0.5 .+ 0.5 .* randn(rng, p)
    μ = exp.(βt .+ Λt * randn(rng, 2, n))
    Y = [rand(rng, Gamma(α, μ[i] / α)) for i in CartesianIndices(μ)]
    Λ, β = _glv_warm(log.(Y), 2)
    Z = getLV(_gamfit(β, Λ, α), Y; rotate = false)   # before the fix: most sites off the mode
    @test maximum(s -> norm(_gam_grad(Y[:, s], Λ, β, α, Z[s, :])), axes(Y, 2)) < 1e-6
end

@testset "getLV (NB1 grouped) uses the NB1 likelihood's mode search" begin
    dY = TOML.parsefile(_GLV_Y)
    Y = permutedims(reshape(Int64.(dY["Y_row_major"]), dY["n"], dY["p"]))
    p = size(Y, 1)
    Λ, β = _glv_warm(log.(Y .+ 0.5), 2)
    fams = [NB1(2.0) for _ in 1:p]
    link = GLLVModels.LogLink()
    # Dispatch: NB1 markers must reach the NB1 chain, not the generic kernel.
    m = which(GLLVModels._grouped_site_mode,
              Tuple{typeof(fams), Vector{Int64}, Vector{Int}, typeof(Λ), typeof(β), typeof(link)})
    @test occursin("NB1", string(m.sig))
    fit = NB1GroupedFit(β, Λ, fill(2.0, p), collect(1:p), link, NaN, true, 0, :observed)
    Z = getLV(fit, Y; rotate = false)
    for s in axes(Y, 2)
        zs, ok = GLLVModels._nb1_grouped_site_mode(fams, Y[:, s], ones(Int, p), Λ, β, link)
        ok && @test Z[s, :] == zs
    end
end

@testset "getLV (NB2 grouped, covariates) is stationary with the covariate offset" begin
    dY = TOML.parsefile(_GLV_Y)
    Y = permutedims(reshape(Int64.(dY["Y_row_major"]), dY["n"], dY["p"]))
    p, n = size(Y)
    rng = StableRNG(521)
    X = randn(rng, p, n, 1)
    γ = [0.3]
    Λ, β = _glv_warm(log.(Y .+ 0.5), 1)
    r = 0.5
    fit = NBGroupedCovFit(β, γ, [false], Λ, fill(r, p), collect(1:p), GLLVModels.LogLink(),
                          NaN, true, 0)
    Z = getLV(fit, Y, X; rotate = false)
    O = GLLVModels._build_offset(X, γ)
    @test maximum(s -> norm(_nb2_grad(Y[:, s], Λ, β .+ O[:, s], r, Z[s, :])), axes(Y, 2)) < 1e-6
end
