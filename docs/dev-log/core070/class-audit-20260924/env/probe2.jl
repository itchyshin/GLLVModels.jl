using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels

report(name, sc, nsite, bad, finbad, worst, vmin) = println(rpad(name, 34), " scale=", sc,
    "  nonconverged=", lpad(bad, 3), "/", nsite, "  finite-value-when-nonconverged=", lpad(finbad,3),
    "  worst |grad|=", round(worst, sigdigits=3), "  min site value=", round(vmin, sigdigits=3))

function run(name, sc; nsite=300, p=10, K=2, seed=11, gen, mode, resid, val)
    rng = MersenneTwister(seed)
    Λ = sc .* randn(rng, p, K); β = 0.5 .* randn(rng, p)
    bad = 0; finbad = 0; worst = 0.0; vmin = Inf
    for s in 1:nsite
        z0 = randn(rng, K); η = β .+ Λ*z0
        y = gen(rng, η)
        z = mode(y, Λ, β)
        r = resid(y, Λ, β, z)
        v = try val(y, Λ, β) catch; NaN end
        if !(r <= 1e-4)
            bad += 1; isfinite(v) && (finbad += 1); worst = max(worst, isfinite(r) ? r : Inf)
            isfinite(v) && (vmin = min(vmin, v))
        end
    end
    report(name, sc, nsite, bad, finbad, worst, vmin)
end

clampη(η) = clamp(η, -30.0, 30.0)
ones_p(p) = ones(Int, p)

for sc in (1.0, 2.5)
  # --- two-part: ZIP / hurdle Poisson (Λz = 0) ---
  for (nm, fam) in (("_twopart_mode ZIP", G.ZIPoisson()), ("_twopart_mode HurdlePoisson", G.HurdlePoisson()), ("_twopart_mode HurdleNB(r=2)", G.HurdleNB(2.0)))
    run(nm, sc;
      gen = (rng, η) -> [rand(rng) < 0.3 ? 0 : rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._twopart_mode(fam, y, zeros(size(Λ)), Λ, fill(-1.0, length(y)), β),
      resid = (y, Λ, β, z) -> begin
          ηc = G._clamp_eta.(β .+ Λ*z); ηz = G._clamp_eta.(fill(-1.0, length(y)))
          sc_ = [G._tp_pieces(fam, y[t], ηz[t], ηc[t])[2] for t in eachindex(y)]
          maximum(abs, Λ'*sc_ .- z) end,
      val = (y, Λ, β) -> G.twopart_loglik_site(fam, y, zeros(size(Λ)), Λ, fill(-1.0, length(y)), β))
  end
  # --- beta-binomial (no eta clamp) ---
  run("_beta_binomial_mode (φ=5,N=10)", sc;
      gen = (rng, η) -> [rand(rng, Binomial(10, 1/(1+exp(-clampη(η[t]))))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._beta_binomial_mode(y, fill(10, length(y)), Λ, β, 5.0),
      resid = (y, Λ, β, z) -> (η = β .+ Λ*z; s = [G._bb_score_weight(y[t], η[t], 10, 5.0)[1] for t in eachindex(y)]; maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G._beta_binomial_loglik_site(y, fill(10, length(y)), Λ, β, 5.0))
  # --- COM-Poisson (no eta clamp) ---
  run("_compoisson_mode (ν=0.7)", sc;
      gen = (rng, η) -> [rand(rng, Poisson(min(exp(clampη(η[t])), 1e4))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._compoisson_mode(y, Λ, β, 0.7),
      resid = (y, Λ, β, z) -> (η = β .+ Λ*z; s = [G._cmp_score_weight(y[t], η[t], 0.7)[1] for t in eachindex(y)]; maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G._compoisson_loglik_site(y, Λ, β, 0.7))
  # --- ordered beta (no eta clamp) ---
  run("_ordered_beta_mode (c0=-2,c1=2,φ=5)", sc;
      gen = (rng, η) -> [ (u = rand(rng); u < 0.1 ? 0.0 : u > 0.9 ? 1.0 : clamp(rand(rng, Beta(2,2)), 1e-4, 1-1e-4)) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._ordered_beta_mode(y, Λ, β, -2.0, 2.0, 5.0),
      resid = (y, Λ, β, z) -> (η = β .+ Λ*z; s = [G._ob_score_weight(y[t], η[t], -2.0, 2.0, 5.0)[1] for t in eachindex(y)]; maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G._ordered_beta_loglik_site(y, Λ, β, -2.0, 2.0, 5.0))
  # --- NB1 grouped kernel vs NB1 via backtracked core ---
  fams = [G.NB1(0.8) for _ in 1:10]
  nb1gen = (rng, η) -> [ (m = min(exp(clampη(η[t])), 1e6); rand(rng, NegativeBinomial(m/0.8, 1/(1+0.8)))) for t in eachindex(η)]
  nb1res = (y, Λ, β, z) -> begin
      η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(fams, exp.(η)); me = μ
      s = G._glm_score.(fams, μ, ones(Int,length(y)), me, y); maximum(abs, Λ'*s .- z) end
  run("_nb1_grouped_loglik_site loop", sc; gen = nb1gen,
      mode = (y, Λ, β) -> begin   # replicate the site kernel's loop exactly by calling the fisher mode through grouped mode? no: kernel has no mode API
          # emulate: the NB1 grouped loop is identical to _laplace_mode_off with fams; use generic grouped mode w/o backtrack
          z = zeros(size(Λ,2))
          for _ in 1:100
              η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(fams, G.linkinv.(Ref(G.LogLink()), η)); me = G.mu_eta.(Ref(G.LogLink()), η)
              s = G._glm_score.(fams, μ, ones(Int,length(y)), me, y); W = G._glm_weight.(fams, μ, ones(Int,length(y)), me)
              Δ = G._safe_solve(Symmetric(Λ'*(W .* Λ) + I), Λ'*s .- z)
              (Δ === nothing || !all(isfinite, Δ)) && break
              z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
          end; z end,
      resid = nb1res,
      val = (y, Λ, β) -> G._nb1_grouped_loglik_site(fams, y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
  run("  (control) NB1 via core _laplace_mode", sc; gen = nb1gen,
      mode = (y, Λ, β) -> G._laplace_mode(G.NB1(0.8), y, ones(Int,length(y)), Λ, β, G.LogLink()),
      resid = nb1res,
      val = (y, Λ, β) -> G.laplace_loglik_site(G.NB1(0.8), y, ones(Int,length(y)), Λ, β, G.LogLink(); hessian=:fisher))
  # --- Student-t via core (not backtracked) ---
  tf = G.StudentTFamily(3.0, 1.0)
  run("core _laplace_mode StudentT(ν=3)", sc;
      gen = (rng, η) -> [η[t] + rand(rng, TDist(3.0)) * (rand(rng) < 0.1 ? 20 : 1) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._laplace_mode(tf, y, ones(Int,length(y)), Λ, β, G.IdentityLink()),
      resid = (y, Λ, β, z) -> (η = β .+ Λ*z; s = G._glm_score.(Ref(tf), η, ones(Int,length(y)), ones(length(y)), y); maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G.laplace_loglik_site(tf, y, ones(Int,length(y)), Λ, β, G.IdentityLink()))
  # --- GP1 via core (not backtracked) ---
  gp = G.GeneralizedPoisson1(0.3)
  run("core _laplace_mode GP1(α=0.3)", sc;
      gen = (rng, η) -> [rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._laplace_mode(gp, y, ones(Int,length(y)), Λ, β, G.LogLink()),
      resid = (y, Λ, β, z) -> (η = G._clamp_eta.(β .+ Λ*z); μ = G._clamp_mu.(Ref(gp), exp.(η)); s = G._glm_score.(Ref(gp), μ, ones(Int,length(y)), μ, y); maximum(abs, Λ'*s .- z)),
      val = (y, Λ, β) -> G.laplace_loglik_site(gp, y, ones(Int,length(y)), Λ, β, G.LogLink()))
  # --- mixed-family mode (all Poisson traits) ---
  famsP = Any[Poisson() for _ in 1:10]; linksP = [G.LogLink() for _ in 1:10]
  run("_mixed_laplace_mode (Poisson traits)", sc;
      gen = (rng, η) -> [rand(rng, Poisson(min(exp(clampη(η[t])), 1e6))) for t in eachindex(η)],
      mode = (y, Λ, β) -> G._mixed_laplace_mode(famsP, linksP, y, ones(Int,length(y)), Λ, β),
      resid = (y, Λ, β, z) -> (η = G._clamp_eta.(β .+ Λ*z); μ = exp.(η); maximum(abs, Λ'*(y .- μ) .- z)),
      val = (y, Λ, β) -> G._mixed_loglik_site(famsP, linksP, y, ones(Int,length(y)), Λ, β))
end
