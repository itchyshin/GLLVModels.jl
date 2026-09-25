using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
include(joinpath(@__DIR__, "probe_common.jl"))
clampη(η) = clamp(η, -30.0, 30.0)
function undamped(fams, y, Λ, β)
    z = zeros(size(Λ,2)); n1 = ones(Int, length(y))
    for _ in 1:100
        η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(fams, exp.(η)); me = μ
        s = G._glm_score.(fams, μ, n1, me, y); W = G._glm_weight.(fams, μ, n1, me)
        Δ = G._safe_solve(Symmetric(Λ'*(W .* Λ) + I), Λ'*s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
    end; z
end
resid(fams, y, Λ, β, z) = (η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(fams, exp.(η)); s = G._glm_score.(fams, μ, ones(Int,length(y)), μ, y); maximum(abs, Λ'*s .- z))
pois(rng, η) = [rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)]
for sc in (1.0, 2.5)
  for φ in (0.05, 0.8)
    fams = [G.NB1(φ) for _ in 1:10]
    run("NB1 grouped kernel φ=$φ", sc; gen = pois, mode = (y,Λ,β)->undamped(fams,y,Λ,β), resid = (y,Λ,β,z)->resid(fams,y,Λ,β,z),
        val = (y,Λ,β)->G._nb1_grouped_loglik_site(fams, y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
    run("  control: core NB1 φ=$φ (backtracked)", sc; gen = pois, mode = (y,Λ,β)->G._laplace_mode(G.NB1(φ), y, ones(Int,length(y)), Λ, β, G.LogLink()),
        resid = (y,Λ,β,z)->resid(fams,y,Λ,β,z),
        val = (y,Λ,β)->G.laplace_loglik_site(G.NB1(φ), y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
  end
  famsT = [G.TweedieED(1.0, 1.5) for _ in 1:10]
  run("core _laplace_mode TweedieED (shared fitter)", sc; gen = pois,
      mode = (y,Λ,β)->G._laplace_mode(G.TweedieED(1.0,1.5), float.(y), ones(Int,length(y)), Λ, β, G.LogLink()),
      resid = (y,Λ,β,z)->resid(famsT, float.(y), Λ, β, z),
      val = (y,Λ,β)->G.laplace_loglik_site(G.TweedieED(1.0,1.5), float.(y), ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
  # reference: NB2 grouped (excluded fitters) kernel with r=100
  famsN = [NegativeBinomial(100.0, 0.5) for _ in 1:10]
  run("ref(excluded): NB2 grouped kernel r=100", sc; gen = pois, mode = (y,Λ,β)->undamped(famsN,y,Λ,β), resid = (y,Λ,β,z)->resid(famsN,y,Λ,β,z),
      val = (y,Λ,β)->G._nb_grouped_loglik_site(famsN, y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
end
