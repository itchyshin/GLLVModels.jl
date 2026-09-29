using GLLVModels, Test, Random, TOML, SHA, QuadGK
const GM = GLLVModels

# Ordered-beta parametric bootstrap. On origin/main the `simulate` closure of
# `_family_ci(::OrderedBetaFit, Y)` (src/confint_family.jl) was a stub that
# errored, so `confint(fit, Y; method = :bootstrap)` returned all-NaN bounds with
# `n_converged = 0` (every replicate's simulate threw inside `_family_bootstrap`'s
# guard). It now draws from the law `ordered_beta_logp`
# (src/families/ordered_beta.jl) scores: z ~ N(0, I_K), η = β + Λz,
# P(y=0) = σ(c0 − η), P(y=1) = σ(η − c1), otherwise y ~ Beta(μφ, (1−μ)φ), μ = σ(η).
# This file checks (1) the draws against analytic moments of that law and
# (2) an end-to-end bootstrap on a literal fixture.

const OBBOOT_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "ordered_beta_boot.toml")
_obboot_sha(Y) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
_obσ(x) = inv(1 + exp(-x))

# E[g(η)] for η = β + ℓ'z, z ~ N(0, I_K), i.e. η ~ N(β, ‖ℓ‖²): one-dimensional
# adaptive quadrature against the standard normal density.
_obE(g, β, ℓ) = quadgk(u -> g(β + sqrt(sum(abs2, ℓ)) * u) * exp(-u^2 / 2) / sqrt(2π),
                       -Inf, Inf; rtol = 1e-11)[1]

@testset "Ordered-beta bootstrap simulator" begin

    @testset "draws match the analytic moments of ordered_beta_logp's law" begin
        # K = 2 so the per-site latent draw is a vector; trait 3 loads on both axes.
        β = [0.4, -0.8, 1.5]
        Λ = [0.7 0.0; -0.5 0.6; 1.1 -0.9]
        c0, c1, φ = -1.2, 0.9, 7.0
        p, K = size(Λ); n = 200_000
        fit = GM.OrderedBetaFit(β, Λ, c0, c1, φ, -1.0, true, 1)
        ad = GM._family_ci(fit, zeros(p, n))
        Yb = ad.simulate(MersenneTwister(20260929))
        @test size(Yb) == (p, n)
        @test all(y -> 0 <= y <= 1, Yb)
        # Every draw is a point the likelihood scores finitely at the generating values.
        @test all(isfinite(GM.ordered_beta_logp(Yb[t, s], β[t], c0, c1, φ))
                  for t in 1:p, s in 1:200)

        # Tolerance: 4 Monte Carlo standard errors from the ANALYTIC variance at this
        # n (not an empirical SD). A two-sided 4-SE miss has probability 6.3e-5 per
        # check under a correct simulator, under 1e-3 over the 10 checks below, and
        # the seed is fixed, so a failure here means a wrong law, not bad luck.
        for t in 1:p
            ℓ = Λ[t, :]
            π0 = _obE(η -> _obσ(c0 - η), β[t], ℓ)
            π1 = _obE(η -> _obσ(η - c1), β[t], ℓ)
            πI = 1 - π0 - π1
            mass(η) = _obσ(η - c0) - _obσ(η - c1)
            # interior: E[y | 0<y<1] and E[y² | 0<y<1] with y | η ~ Beta(μφ, (1−μ)φ)
            m1 = _obE(η -> mass(η) * _obσ(η), β[t], ℓ) / πI
            m2 = _obE(η -> (μ = _obσ(η); mass(η) * (μ * (1 - μ) / (φ + 1) + μ^2)), β[t], ℓ) / πI
            row = view(Yb, t, :)
            interior = filter(y -> 0 < y < 1, row)
            nI = length(interior)
            @test abs(count(==(0.0), row) / n - π0) < 4 * sqrt(π0 * (1 - π0) / n)
            @test abs(count(==(1.0), row) / n - π1) < 4 * sqrt(π1 * (1 - π1) / n)
            @test abs(sum(interior) / nI - m1) < 4 * sqrt((m2 - m1^2) / nI)
        end

        # The latent draw is shared by all traits of a site (not redrawn per cell):
        # P(y1 = 1, y3 = 1) = E_z[σ(η1 − c1) σ(η3 − c1)], a two-dimensional integral.
        q13 = quadgk(z1 -> quadgk(z2 -> begin
                    z = [z1, z2]
                    _obσ(β[1] + Λ[1, :]' * z - c1) * _obσ(β[3] + Λ[3, :]' * z - c1) *
                        exp(-(z1^2 + z2^2) / 2) / (2π)
                end, -Inf, Inf; rtol = 1e-10)[1], -Inf, Inf; rtol = 1e-10)[1]
        q13_indep = _obE(η -> _obσ(η - c1), β[1], Λ[1, :]) * _obE(η -> _obσ(η - c1), β[3], Λ[3, :])
        @test abs(q13 - q13_indep) > 20 * sqrt(q13 * (1 - q13) / n)   # the check can tell them apart
        emp13 = count(s -> Yb[1, s] == 1.0 && Yb[3, s] == 1.0, 1:n) / n
        @test abs(emp13 - q13) < 4 * sqrt(q13 * (1 - q13) / n)
    end

    @testset "confint(...; method = :bootstrap) on a fixture fit" begin
        fixture = TOML.parsefile(OBBOOT_FIXTURE_PATH)
        p, n, K = fixture["p"], fixture["n"], fixture["K"]
        Yv = Float64.(fixture["Y_column_major"])
        @test length(Yv) == p * n
        @test _obboot_sha(Yv) == fixture["data_sha256"]
        Y = reshape(Yv, p, n)

        fit = GM.fit_ordered_beta_gllvm(Y; K = K)
        @test fit.converged
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates.
        n_boot = 12
        ci = confint(fit, Y; method = :bootstrap, n_boot = n_boot, seed = 1)
        @test ci.method === :bootstrap
        @test ci.n_converged >= 10
        lin = [i for i in eachindex(ci.term) if ci.term[i] != "phi"]
        @test length(lin) == p + p * K + 2          # beta, Lambda (K = 1), cut0, cut1
        @test all(isfinite, ci.lower[lin]) && all(isfinite, ci.upper[lin])
        @test all(ci.lower[lin] .<= ci.estimate[lin] .<= ci.upper[lin])
        iφ = findfirst(==("phi"), ci.term)
        @test isfinite(ci.lower[iφ]) && ci.lower[iφ] > 0
    end
end
