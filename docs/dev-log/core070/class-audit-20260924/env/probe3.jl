using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
include(joinpath(@__DIR__, "probe_common.jl"))

clampη(η) = clamp(η, -30.0, 30.0)
for sc in (1.0, 2.5)
  for (nm, fam, gen) in (
      ("_twopart_mode ZINB(r=2)", G.ZINB(2.0), (rng, η) -> [rand(rng) < 0.3 ? 0 : rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)]),
      ("_twopart_mode ZIB(N=10)", G.ZIB(10), (rng, η) -> [rand(rng) < 0.3 ? 0 : rand(rng, Binomial(10, 1/(1+exp(-clampη(η[t]))))) for t in eachindex(η)]),
      ("_twopart_mode DeltaGamma(α=2)", G.DeltaGamma(2.0), (rng, η) -> [rand(rng) < 0.3 ? 0.0 : rand(rng, Gamma(2.0, min(exp(clampη(η[t])), 1e6)/2.0)) for t in eachindex(η)]),
      ("_twopart_mode BetaHurdle(φ=5)", G.BetaHurdle(5.0), (rng, η) -> [rand(rng) < 0.3 ? 0.0 : clamp(rand(rng, Beta(2,2)), 1e-4, 1-1e-4) for t in eachindex(η)]),
      ("_twopart_mode DeltaLogNormal(σ=1)", G.DeltaLogNormal(1.0), (rng, η) -> [rand(rng) < 0.3 ? 0.0 : exp(clampη(η[t]) + randn(rng)) for t in eachindex(η)]))
    run(nm, sc; gen = gen,
      mode = (y, Λ, β) -> G._twopart_mode(fam, y, zeros(size(Λ)), Λ, fill(-1.0, length(y)), β),
      resid = (y, Λ, β, z) -> begin
          ηc = G._clamp_eta.(β .+ Λ*z); ηz = G._clamp_eta.(fill(-1.0, length(y)))
          sc_ = [G._tp_pieces(fam, y[t], ηz[t], ηc[t])[2] for t in eachindex(y)]
          maximum(abs, Λ'*sc_ .- z) end,
      val = (y, Λ, β) -> G.twopart_loglik_site(fam, y, zeros(size(Λ)), Λ, fill(-1.0, length(y)), β))
  end
  # ordinal per-trait (C=4 per trait)
  p = 10; C = fill(4, p); τ = repeat([-1.0 0.0 1.0], p)
  run("_ordinal_laplace_mode_pertrait logit", sc;
      gen = (rng, η) -> [ (u = rand(rng); l = log(u/(1-u)) + η[t]; l < -1 ? 1 : l < 0 ? 2 : l < 1 ? 3 : 4) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._ordinal_laplace_mode_pertrait(y, Λ, β, τ, C, G.LogitLink()),
      resid = (y, Λ, β, z) -> (η = G._clamp_eta.(β .+ Λ*z); s = [G._ord_score_weight(y[t], η[t], view(τ, t, 1:3), G.LogitLink())[1] for t in eachindex(y)]; maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G.ordinal_loglik_site_pertrait(y, Λ, β, τ, C, G.LogitLink()))
  # quadratic Poisson with D = -0.3
  run("_quadratic_mode Poisson D=-0.3", sc;
      gen = (rng, η) -> [rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._quadratic_mode(Poisson(), y, ones(Int,length(y)), Λ, fill(-0.3, size(Λ)), β, G.LogLink()),
      resid = (y, Λ, β, z) -> (D = fill(-0.3, size(Λ)); η = G._clamp_eta.(β .+ Λ*z .+ D*(z.^2)); μ = G._clamp_mu.(Ref(Poisson()), exp.(η)); J = Λ .+ 2 .* D .* z'; maximum(abs, J'*(y .- μ) .- z)),
      val = (y, Λ, β) -> G.quadratic_loglik_site(Poisson(), y, ones(Int,length(y)), Λ, fill(-0.3, size(Λ)), β, G.LogLink()))
  # Tweedie grouped (φ=1, p=1.5)
  tf = [G.TweedieED(1.0, 1.5) for _ in 1:10]
  run("_tweedie_grouped_loglik_site loop", sc;
      gen = (rng, η) -> [ (μ = min(exp(clampη(η[t])), 1e4); λ = μ^(0.5)/(1.0*0.5); N = rand(rng, Poisson(λ)); N == 0 ? 0.0 : sum(rand(rng, Gamma(3.0, 1.0*0.5*μ^0.5)) for _ in 1:N)) for t in eachindex(η)],
      mode = (y, Λ, β) -> begin
          z = zeros(size(Λ,2)); n1 = ones(Int, length(y))
          for _ in 1:100
              η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(tf, exp.(η)); me = μ
              s = G._glm_score.(tf, μ, n1, me, y); W = G._glm_weight.(tf, μ, n1, me)
              Δ = G._safe_solve(Symmetric(Λ'*(W .* Λ) + I), Λ'*s .- z)
              (Δ === nothing || !all(isfinite, Δ)) && break
              z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
          end; z end,
      resid = (y, Λ, β, z) -> (η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(tf, exp.(η)); s = G._glm_score.(tf, μ, ones(Int,length(y)), μ, y); maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G._tweedie_grouped_loglik_site(tf, y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
  # missing-predictor augmented mode (Poisson, x missing)
  run("_mode_xs Poisson (x missing, b_x=1)", sc;
      gen = (rng, η) -> [rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._mode_xs(Poisson(), y, ones(Int,length(y)), Λ, β, G.LogLink(), 1.0, 0.0, 1.0)[1],
      resid = (y, Λ, β, z) -> begin
          zz, x = G._mode_xs(Poisson(), y, ones(Int,length(y)), Λ, β, G.LogLink(), 1.0, 0.0, 1.0)
          η = G._clamp_eta.(β .+ x .+ Λ*zz); μ = exp.(η)
          max(maximum(abs, Λ'*(y .- μ) .- zz), abs(sum(y .- μ) - x)) end,
      val = (y, Λ, β) -> G.laplace_loglik_site_xs(Poisson(), y, ones(Int,length(y)), Λ, β, G.LogLink(); x_obs=nothing, b_x=1.0, μ_x=0.0, σ_x2=1.0))
end
