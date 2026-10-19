using GLLVModels, Test, Random, LinearAlgebra, SparseArrays
using Distributions: Poisson, Binomial, NegativeBinomial

# Coverage for fit_phylo_glm beyond the Poisson smoke in test_phylo_glm.jl:
# argument validation, Binomial trials, dispersion plumbing, sigma2_phy ordering,
# the converged flag and its show output, start-value invariance, and agreement
# of fit.loglik with phylo_glm_marginal_loglik at the fitted parameters.

const _PGF_NEWICK = "((((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2):0.1,(G:0.3,H:0.3):0.4);"

# Simulate species x sites data from the phylo random-intercept model:
# a ~ N(0, sigma2 * Sigma_phy) shared across sites, eta = beta + a.
function _pgf_sim(phy, family, β, σ2, n; N = nothing, r = 5.0, seed = 1)
    rng = MersenneTwister(seed)
    p = phy.n_leaves
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Σa = σ2 .* inv(Qc)[leaf_pos, leaf_pos]
    a = cholesky(Symmetric(Σa)).L * randn(rng, p)
    Y = zeros(Int, p, n)
    for t in 1:p, s in 1:n
        η = β[t] + a[t]
        Y[t, s] = if family isa Poisson
            rand(rng, Poisson(exp(η)))
        elseif family isa Binomial
            rand(rng, Binomial(N[t, s], 1 / (1 + exp(-η))))
        else
            μ = exp(η); rand(rng, NegativeBinomial(r, r / (r + μ)))
        end
    end
    return Y
end

@testset "fit_phylo_glm: families, flags, consistency" begin
    phy = GLLVModels.augmented_phy(_PGF_NEWICK)
    p = phy.n_leaves; n = 40
    β = 0.3 .+ 0.2 .* collect(range(-1, 1; length = p))

    # (1) species-count mismatch
    Ybad = rand(0:3, p + 1, 5)
    err = try fit_phylo_glm(Ybad, phy); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("must equal phy.n_leaves", sprint(showerror, err))

    # (2) Binomial with trial matrix; omitting N falls back to ones
    Ntr = fill(10, p, n)
    βb = 0.2 .* collect(range(-1, 1; length = p))
    # a is shared across sites and beta is free per species, so beta-hat estimates
    # beta + a; keep the simulated phylo variance small so beta-hat tracks beta.
    Yb = _pgf_sim(phy, Binomial(1, 0.5), βb, 0.02, n; N = Ntr, seed = 11)
    fb = fit_phylo_glm(Yb, phy; family = Binomial(1, 0.5), N = Ntr, iterations = 100)
    @test all(isfinite, fb.β)
    @test isfinite(fb.loglik)
    @test maximum(abs.(fb.β .- βb)) < 0.5
    Yb1 = rand(MersenneTwister(12), 0:1, p, n)
    fb1 = fit_phylo_glm(Yb1, phy; family = Binomial(1, 0.5), iterations = 50)
    @test all(isfinite, fb1.β)
    @test isfinite(fb1.loglik)

    # (3) dispersion family vs no-dispersion family
    Yn = _pgf_sim(phy, NegativeBinomial(5.0, 0.5), β, 0.3, n; r = 5.0, seed = 21)
    fnb = fit_phylo_glm(Yn, phy; family = NegativeBinomial(5.0, 0.5), iterations = 100)
    fpo = fit_phylo_glm(Yn, phy; family = Poisson(), iterations = 100)
    @test isfinite(fnb.dispersion) && fnb.dispersion > 0
    @test isnan(fpo.dispersion)
    @test occursin("dispersion=", sprint(show, fnb))
    @test !occursin("dispersion=", sprint(show, fpo))

    # (4) simulated sigma2_phy ordering
    Yhi = _pgf_sim(phy, Poisson(), β, 0.5, n; seed = 31)
    Ylo = _pgf_sim(phy, Poisson(), β, 0.05, n; seed = 31)
    fhi = fit_phylo_glm(Yhi, phy; family = Poisson(), iterations = 100)
    flo = fit_phylo_glm(Ylo, phy; family = Poisson(), iterations = 100)
    # KNOWN LIMITATION: the random effect is one value per species shared by all
    # sites while beta is free per species, so sigma2_phy is not identified; the
    # marginal loglik decreases in sigma2_phy at the fitted beta and the fit sits on
    # the sigma2_phy -> 0 boundary for both data sets. Flip to @test when fixed.
    @test_broken fhi.σ²_phy > flo.σ²_phy

    # (5) converged flag and show output
    Ye = _pgf_sim(phy, Poisson(), β, 0.3, n; seed = 41)
    fe = fit_phylo_glm(Ye, phy; family = Poisson(), iterations = 300)
    @test fe.converged
    @test !occursin("NOT CONVERGED", sprint(show, fe))
    f1 = fit_phylo_glm(Ye, phy; family = Poisson(), iterations = 1)
    @test !f1.converged
    @test occursin("NOT CONVERGED", sprint(show, f1))

    # (6) the optimum does not depend on the sigma2_phy start
    fa = fit_phylo_glm(Ye, phy; family = Poisson(), σ²_phy_init = 0.1, iterations = 300)
    fc = fit_phylo_glm(Ye, phy; family = Poisson(), σ²_phy_init = 2.0, iterations = 300)
    @test abs(fa.loglik - fc.loglik) < 1e-3

    # (7) fit.loglik is the marginal at the fitted parameters
    Ntr1 = ones(Int, p, n)
    ℓ = phylo_glm_marginal_loglik(Poisson(), Ye, Ntr1, fe.β, fe.σ²_phy, phy; link = fe.link)
    @test isapprox(fe.loglik, ℓ; atol = 1e-6)
    ℓnb = phylo_glm_marginal_loglik(NegativeBinomial(fnb.dispersion, 0.5), Yn, Ntr1,
                                    fnb.β, fnb.σ²_phy, phy; link = fnb.link)
    @test isapprox(fnb.loglik, ℓnb; atol = 1e-6)
end
