using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

# confint on a fit made with an `offset` (known additive term in η = β + offset + Λz).
#
# The bug this file pins: `confint` rebuilt the marginal log-likelihood from
# (fit, Y, X, mask, N) only, so a fit made with an offset got the Hessian, the
# profile and the bootstrap of a DIFFERENT, offset-free objective, with no warning.
# On a Poisson fit (p = 5, n = 80, offset 0.5 * randn) the offset-free Hessian at the
# fit's own optimum was not positive definite (min eigenvalue -243) while the fit's own
# offset-aware Hessian was (+26.9), so `pd_hessian` came back `false` on a regular fit.
#
# The reference is not produced by the code under test: it is a central-difference
# Hessian of the fit's OWN offset-aware objective, written out in this file from the
# public family marginals, with its own step rule (not `GLLVModels._fd_hessian`).

# Central-difference Hessian with an independent step rule.
function _ref_hessian(f, x::AbstractVector; h0 = 1e-3)
    m = length(x)
    h = [h0 * max(abs(x[i]), 1.0) for i in 1:m]
    f0 = f(x)
    H = zeros(m, m)
    for i in 1:m
        xp = copy(x); xp[i] += h[i]
        xm = copy(x); xm[i] -= h[i]
        H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i]^2
    end
    for i in 1:m, j in (i + 1):m
        a = copy(x); a[i] += h[i]; a[j] += h[j]
        b = copy(x); b[i] += h[i]; b[j] -= h[j]
        c = copy(x); c[i] -= h[i]; c[j] += h[j]
        d = copy(x); d[i] -= h[i]; d[j] -= h[j]
        H[i, j] = H[j, i] = (f(a) - f(b) - f(c) + f(d)) / (4 * h[i] * h[j])
    end
    return H
end

_ref_se(H) = sqrt.(diag(inv(Symmetric((H .+ H') ./ 2))))

# Wald SEs and pd flag from the independent reference Hessian of `nll` at `θ`.
function _ref_wald(nll, θ)
    H = _ref_hessian(nll, θ)
    return (se = _ref_se(H), pd = isposdef(Symmetric((H .+ H') ./ 2)), mineig = minimum(eigvals(Symmetric((H .+ H') ./ 2))))
end

# ---- data generators (offset enters the linear predictor on the log/logit scale) --------
function _gen_counts(rng, p, n, K, O; kind = :poisson, r = 3.0)
    β0 = 0.4 .* randn(rng, p) .+ 0.3
    Λ0 = 0.6 .* randn(rng, p, K)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β0 .+ O[:, s] .+ Λ0 * randn(rng, K)
        for t in 1:p
            μ = exp(η[t])
            Y[t, s] = kind === :poisson ? rand(rng, Poisson(μ)) : rand(rng, NegativeBinomial(r, r / (r + μ)))
        end
    end
    return Y
end

function _gen_binomial(rng, p, n, K, O, Nm)
    β0 = 0.3 .* randn(rng, p)
    Λ0 = 0.6 .* randn(rng, p, K)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β0 .+ O[:, s] .+ Λ0 * randn(rng, K)
        for t in 1:p
            Y[t, s] = rand(rng, Binomial(Nm[t, s], 1 / (1 + exp(-η[t]))))
        end
    end
    return Y
end

function _gen_gamma(rng, p, n, K, O; α = 4.0)
    β0 = 0.3 .* randn(rng, p) .+ 1.0
    Λ0 = 0.4 .* randn(rng, p, K)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n
        η = β0 .+ O[:, s] .+ Λ0 * randn(rng, K)
        for t in 1:p
            μ = exp(η[t])
            Y[t, s] = rand(rng, Gamma(α, μ / α))
        end
    end
    return Y
end

# Zero-inflated / positive part: separate predictors, the offset sits on the positive part.
function _gen_exponential(rng, p, n, K, O)
    β0 = 0.3 .* randn(rng, p) .+ 1.0
    Λ0 = 0.4 .* randn(rng, p, K)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n
        η = β0 .+ O[:, s] .+ Λ0 * randn(rng, K)
        for t in 1:p
            Y[t, s] = rand(rng, Exponential(exp(η[t])))
        end
    end
    return Y
end

function _gen_zip(rng, p, n, K, O)
    βz = fill(-1.0, p) .+ 0.2 .* randn(rng, p)
    βc = 0.4 .* randn(rng, p) .+ 0.8
    Λ0 = 0.5 .* randn(rng, p, K)
    Y = zeros(Int, p, n)
    for s in 1:n
        ηc = βc .+ O[:, s] .+ Λ0 * randn(rng, K)
        for t in 1:p
            rand(rng) < 1 / (1 + exp(-βz[t])) || (Y[t, s] = rand(rng, Poisson(exp(ηc[t]))))
        end
    end
    return Y
end

function _gen_delta_gamma(rng, p, n, K, O; shared = false, α = 4.0)
    β = 0.3 .* randn(rng, p) .+ 1.0
    Λ0 = 0.4 .* randn(rng, p, K)
    Y = zeros(Float64, p, n)
    for s in 1:n
        z = randn(rng, K)
        for t in 1:p
            ηc = β[t] + O[t, s] + (Λ0 * z)[t]
            π = shared ? 1 / (1 + exp(-ηc)) : 0.7
            rand(rng) < π && (Y[t, s] = rand(rng, Gamma(α, exp(ηc) / α)))
        end
    end
    return Y
end

# ---- the fits to test: name, fit, reference nll (offset-aware, written from the public
# marginals), reference θ (read off the fit's own fields) ---------------------------------
function _cases(rng, p, n, K, O, Nm)
    cases = Any[]
    rr = GM.rr_theta_len(p, K)
    # Each case gets its own `let`: the objective closures capture Y and the fit, so a
    # rebinding of `Y` in the enclosing scope would silently point them at the last case.
    # Poisson
    let Y = _gen_counts(rng, p, n, K, O; kind = :poisson)
        f = fit_poisson_gllvm(Y; K = K, offset = O)
        push!(cases, (name = "Poisson", fit = f, Y = Y, kw = (;),
            θ = vcat(f.β, GM.pack_lambda(f.Λ)),
            nll = θv -> -GM.poisson_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(p + 1):end], p, K), θv[1:p],
                            f.link; offset = O, hessian = f.hessian, maxiter = 100, tol = 1e-9)))
    end
    # NB2 (shared r)
    let Y = _gen_counts(rng, p, n, K, O; kind = :nb)
        f = fit_nb_gllvm(Y; K = K, offset = O)
        push!(cases, (name = "NB2", fit = f, Y = Y, kw = (;),
            θ = vcat(f.β, GM.pack_lambda(f.Λ), log(f.r)),
            nll = θv -> -GM.nb_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(p + 1):(p + rr)], p, K), θv[1:p],
                            exp(θv[p + rr + 1]); link = f.link, offset = O, hessian = f.hessian,
                            maxiter = 100, tol = 1e-9)))
    end
    # Binomial with trial counts N
    let Y = _gen_binomial(rng, p, n, K, O, Nm)
        f = fit_binomial_gllvm(Y; K = K, N = Nm, offset = O)
        push!(cases, (name = "Binomial (N)", fit = f, Y = Y, kw = (N = Nm,),
            θ = vcat(f.β, GM.pack_lambda(f.Λ)),
            nll = θv -> -GM.binomial_marginal_loglik_laplace(Y, Nm, GM.unpack_lambda(θv[(p + 1):end], p, K), θv[1:p],
                            f.link; offset = O, hessian = f.hessian, maxiter = 100, tol = 1e-9)))
    end
    # Gamma
    let Y = _gen_gamma(rng, p, n, K, O)
        f = fit_gamma_gllvm(Y; K = K, offset = O)
        push!(cases, (name = "Gamma", fit = f, Y = Y, kw = (;),
            θ = vcat(f.β, GM.pack_lambda(f.Λ), log(f.α)),
            nll = θv -> -GM.gamma_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(p + 1):(p + rr)], p, K), θv[1:p],
                            exp(θv[p + rr + 1]); link = f.link, offset = O, hessian = f.hessian,
                            maxiter = 100, tol = 1e-9)))
    end
    # Exponential
    let Y = _gen_exponential(rng, p, n, K, O)
        f = fit_exponential_gllvm(Y; K = K, offset = O)
        push!(cases, (name = "Exponential", fit = f, Y = Y, kw = (;),
            θ = vcat(f.β, GM.pack_lambda(f.Λ)),
            nll = θv -> -GM.exponential_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(p + 1):end], p, K), θv[1:p];
                            link = f.link, offset = O, hessian = f.hessian, maxiter = 100, tol = 1e-9)))
    end
    # Zero-inflated Poisson (two-part; the offset is on the count part)
    let Y = _gen_zip(rng, p, n, K, O)
        f = fit_zip_gllvm(Y; K = K, offset = O)
        push!(cases, (name = "ZIP (two-part)", fit = f, Y = Y, kw = (;),
            θ = vcat(f.βz, f.βc, GM.pack_lambda(f.Λc)),
            nll = θv -> -GM.zip_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(2p + 1):(2p + rr)], p, K),
                            θv[1:p], θv[(p + 1):(2p)]; offsetc = O, maxiter = 100, tol = 1e-9)))
    end
    # Delta-Gamma, separate predictors (offset on the positive part only)
    let Y = _gen_delta_gamma(rng, p, n, K, O; shared = false)
        f = fit_delta_gamma_gllvm(Y; K = K, predictor = :separate, disp_group = :shared, offset = O)
        push!(cases, (name = "Delta-Gamma (separate)", fit = f, Y = Y, kw = (;),
            θ = vcat(f.βz, f.βc, GM.pack_lambda(f.Λc), log(f.α)),
            nll = θv -> -GM.delta_gamma_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(2p + 1):(2p + rr)], p, K),
                            θv[1:p], θv[(p + 1):(2p)], exp(θv[2p + rr + 1]); offsetc = O,
                            hessian = :observed, maxiter = 100, tol = 1e-9)))
    end
    # Delta-Gamma, shared predictor (offset on BOTH parts, so the tie is kept)
    let Y = _gen_delta_gamma(rng, p, n, K, O; shared = true)
        f = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, disp_group = :shared, offset = O)
        push!(cases, (name = "Delta-Gamma (shared)", fit = f, Y = Y, kw = (;),
            θ = vcat(f.βc, GM.pack_lambda(f.Λc), log(f.α)),
            nll = θv -> -GM.delta_gamma_marginal_loglik_laplace(Y, GM.unpack_lambda(θv[(p + 1):(p + rr)], p, K),
                            θv[1:p], θv[1:p], exp(θv[p + rr + 1]); Λz = GM.unpack_lambda(θv[(p + 1):(p + rr)], p, K),
                            offsetz = O, offsetc = O, hessian = :observed, maxiter = 100, tol = 1e-9)))
    end
    return cases
end

# The dataset behind the "spurious pd_hessian = false" case (seed 18 of a 30-seed scan;
# p = 5, n = 80, K = 1, offset 0.5 * randn): offset-aware min eigenvalue +26.9, offset-free
# -243.1 at the same point.
function _spurious_poisson()
    p, n, K = 5, 80, 1
    rng = MersenneTwister(18)
    β0 = randn(rng, p) .* 0.4 .+ 0.3
    Λ0 = reshape(0.6 .* randn(rng, p), p, 1)
    O = 0.5 .* randn(rng, p, n)
    Z = randn(rng, K, n)
    η = β0 .+ O .+ Λ0 * Z
    Y = Float64.([rand(rng, Poisson(exp(η[i, j]))) for i in 1:p, j in 1:n])
    return Y, O
end

# Every fit type whose fitter takes an offset, one small fit each. For each: the rebuilt
# objective reproduces fit.loglik with the offset (`confint` runs, whose built-in check is
# exactly that), is refused without it, the Wald route returns one SE per term, refitting
# the observed data through the bootstrap closure reproduces the fit, and the simulator
# returns data of the right shape.
function _sim_generic(rng, p, n, K, O, draw)
    β0 = 0.3 .* randn(rng, p) .+ 0.8
    Λ0 = 0.5 .* randn(rng, p, K)
    cols = map(1:n) do s
        η = β0 .+ O[:, s] .+ Λ0 * randn(rng, K)
        [draw(rng, t, η[t]) for t in 1:p]
    end
    return reduce(hcat, cols)
end

function _adapter_table(p, n, K, O)
    rng = MersenneTwister(2026)
    g2 = [1, 1, 2, 2]
    nbr = [3.0, 3.0, 8.0, 8.0]
    pos(x) = max(x, 1e-12)
    zinf(rng, f) = rand(rng) < 0.25 ? 0 : f()
    hur(rng, f) = rand(rng) < 0.3 ? 0 : f()
    rows = Any[]
    add(name, fit, Y; kw = (;)) = push!(rows, (name = name, fit = fit, Y = Y, kw = kw))
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> rand(r, NegativeBinomial(exp(η) / 0.8, 1 / 1.8)))
    add("NB1", fit_nb1_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> rand(r, Poisson(exp(η))))
    add("GP1", fit_gp1_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> clamp(rand(r, Beta(6 / (1 + exp(-η)), 6 * (1 - 1 / (1 + exp(-η))))), 1e-6, 1 - 1e-6))
    add("Beta", fit_beta_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> GM._rand_ztpois(r, exp(η)))
    add("Truncated Poisson", fit_truncated_poisson_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> GM._rand_ztnb(r, 3.0, exp(η)))
    add("Truncated NB2", fit_truncated_nbinom2_gllvm(Y; K = K, offset = O), Y)
    add("Truncated NB2 (per trait)", fit_truncated_nbinom2_gllvm_pertrait(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> rand(r, NegativeBinomial(nbr[t], nbr[t] / (nbr[t] + exp(η)))))
    add("NB2 grouped", fit_nb_gllvm_grouped(Y; K = K, group = g2, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> rand(r, NegativeBinomial(exp(η) / 0.8, 1 / 1.8)))
    add("NB1 grouped", fit_nb1_gllvm_grouped(Y; K = K, group = g2, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> clamp(rand(r, Beta(6 / (1 + exp(-η)), 6 * (1 - 1 / (1 + exp(-η))))), 1e-6, 1 - 1e-6))
    add("Beta grouped", fit_beta_gllvm_grouped(Y; K = K, group = g2, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> rand(r, Gamma(4.0, exp(η) / 4.0)))
    add("Gamma grouped", fit_gamma_gllvm_grouped(Y; K = K, group = g2, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> GM._rand_tweedie(r, exp(η), 1.0, 1.5))
    add("Tweedie grouped", fit_tweedie_gllvm_grouped(Y; K = K, group = g2, power = 1.5, offset = O), Y)
    add("Tweedie grouped (per-trait power)",
        fit_tweedie_gllvm_grouped(Y; K = K, group = g2, power_group = :species, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> hur(r, () -> exp(η + 0.5 * randn(r))))
    add("Delta-lognormal (separate)", fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared, offset = O), Y)
    add("Delta-lognormal (shared)", fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, disp_group = :shared, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> hur(r, () -> GM._rand_ztpois(r, exp(η))))
    add("Hurdle-Poisson", fit_hurdle_poisson_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> hur(r, () -> GM._rand_ztnb(r, 3.0, exp(η))))
    add("Hurdle-NB", fit_hurdle_nb_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> zinf(r, () -> rand(r, NegativeBinomial(3.0, 3.0 / (3.0 + exp(η))))))
    add("ZINB", fit_zinb_gllvm(Y; K = K, offset = O), Y)
    Y = _sim_generic(rng, p, n, K, O, (r, t, η) -> zinf(r, () -> rand(r, Binomial(6, 1 / (1 + exp(-η))))))
    add("ZIB", fit_zib_gllvm(Y; K = K, N = 6, offset = O), Y)
    return rows
end

@testset "confint and the fit's offset" begin
@testset "confint honours the fit's offset" begin
    p, n, K = 5, 80, 1
    rng = MersenneTwister(18)
    O = 0.5 .* randn(rng, p, n)                      # non-constant p x n offset
    Nm = fill(6, p, n)
    cases = _cases(rng, p, n, K, O, Nm)

    @testset "$(c.name): Wald on the fit's own offset-aware objective" for c in cases
        @test c.fit.converged
        # the reference objective is the fit's own: it reproduces fit.loglik at the optimum
        @test -c.nll(c.θ) ≈ c.fit.loglik atol = 1e-6 * max(1, abs(c.fit.loglik))
        ref = _ref_wald(c.nll, c.θ)
        ci = confint(c.fit, c.Y; c.kw..., offset = O)
        @test ci.pd_hessian == ref.pd
        @test ref.pd                                  # the fit's own Hessian is a proper maximum
        @test length(ci.se) == length(ref.se)
        @test isapprox(ci.se, ref.se; rtol = 5e-3)
        # vcov and coef_table take the same keyword
        V = vcov(c.fit, c.Y; c.kw..., offset = O)
        @test isapprox(sqrt.(diag(V)), ref.se; rtol = 5e-3)
    end

    @testset "$(c.name): a fit made with an offset refuses an offset-free confint" for c in cases
        # was: silently returned the interval of the offset-free objective
        @test_throws ArgumentError confint(c.fit, c.Y; c.kw...)
        @test_throws ArgumentError confint(c.fit, c.Y; c.kw..., offset = O .+ 0.3)   # a wrong offset
        @test_throws ArgumentError vcov(c.fit, c.Y; c.kw...)
    end

    @testset "spurious pd_hessian = false (Poisson, p = 5, n = 80, offset 0.5 * randn)" begin
        Ys, Os = _spurious_poisson()
        f = fit_poisson_gllvm(Ys; K = 1, offset = Os)
        @test f.converged
        θ = vcat(f.β, GM.pack_lambda(f.Λ))
        nll = θv -> -GM.poisson_marginal_loglik_laplace(Ys, GM.unpack_lambda(θv[(p + 1):end], p, 1), θv[1:p],
                        f.link; offset = Os, hessian = f.hessian, maxiter = 100, tol = 1e-9)
        # What confint used to report: the Hessian of the offset-free objective at the
        # fit's optimum is indefinite, although the fit's own Hessian is positive definite.
        free = GM._family_ci(f, Ys)                    # no offset: the old objective
        Hfree = _ref_hessian(free.nll, θ)
        @test minimum(eigvals(Symmetric((Hfree .+ Hfree') ./ 2))) < -100
        ref = _ref_wald(nll, θ)
        @test ref.mineig > 20
        ci = confint(f, Ys; offset = Os)
        @test ci.pd_hessian
        @test isapprox(ci.se, ref.se; rtol = 5e-3)
        @test all(isfinite, ci.lower) && all(isfinite, ci.upper)
        @test all(ci.lower .< ci.estimate .< ci.upper)
        @test_throws ArgumentError confint(f, Ys)      # was: pd_hessian = false, intervals of another objective
    end

    @testset "constant offset: same SEs, intercepts shifted by minus the constant" begin
        Yc = _gen_counts(MersenneTwister(5), p, n, K, zeros(p, n); kind = :poisson)
        cst = 1.5
        Oc = fill(cst, p, n)
        f0 = fit_poisson_gllvm(Yc; K = K)
        fc = fit_poisson_gllvm(Yc; K = K, offset = Oc)
        @test f0.loglik ≈ fc.loglik atol = 1e-4
        ci0 = confint(f0, Yc)
        cic = confint(fc, Yc; offset = Oc)
        @test ci0.pd_hessian == cic.pd_hessian
        @test isapprox(cic.se, ci0.se; rtol = 5e-3)
        @test isapprox(cic.estimate[1:p], ci0.estimate[1:p] .- cst; atol = 2e-3)   # beta shifts by -c
        @test isapprox(cic.estimate[(p + 1):end], ci0.estimate[(p + 1):end]; atol = 2e-3)
        # a scalar means the same constant matrix (as in fit_gllvm)
        cis = confint(fc, Yc; offset = cst)
        @test cis.se == cic.se
        # and the bootstrap must simulate from, and refit with, the same offset: with
        # exposure e^1.5 the fitted mean count is e^1.5 times the offset-free one
        ad = GM._family_ci(fc, Yc; offset = Oc)
        sims = [mean(ad.simulate(MersenneTwister(b))) for b in 1:20]
        @test isapprox(mean(sims), mean(Yc); rtol = 0.1)
        bad = GM._family_ci(fc, Yc)                    # offset-free closure: the old behaviour
        @test mean(mean(bad.simulate(MersenneTwister(b))) for b in 1:20) < 0.4 * mean(Yc)
        # refitting the observed data through the closure reproduces the fit (the offset-free
        # refit would land 1.5 away on every intercept)
        @test isapprox(ad.refit(Yc).θ, ad.θ; atol = 1e-4)
        @test maximum(abs, bad.refit(Yc).θ[1:p] .- ad.θ[1:p]) > 1.0
    end

    @testset "profile and bootstrap use the offset too" begin
        c = cases[1]
        w = confint(c.fit, c.Y; offset = O)
        pr = confint(c.fit, c.Y; offset = O, method = :profile, parm = "beta[1]")
        @test pr.status[1] in (:profile, :partial)
        i = findfirst(==("beta[1]"), w.term)
        isfinite(pr.lower[1]) && @test pr.lower[1] < w.estimate[i]
        isfinite(pr.upper[1]) && @test w.estimate[i] < pr.upper[1]
        # profile interval is in the Wald ballpark (it would be far off on the wrong objective)
        if all(isfinite, (pr.lower[1], pr.upper[1]))
            @test isapprox(pr.upper[1] - pr.lower[1], w.upper[i] - w.lower[i]; rtol = 0.25)
        end
        bt = confint(c.fit, c.Y; offset = O, method = :bootstrap, n_boot = 12, seed = 3, parm = "beta")
        @test bt.n_converged >= 10
        @test all(isfinite, bt.lower) && all(isfinite, bt.upper)
    end

    O4 = 0.4 .* randn(MersenneTwister(9), 4, 50)
    # fits on small simulated data may warn (a dispersion at its Poisson limit); not the point here
    adapter_rows = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        _adapter_table(4, 50, 1, O4)
    end
    @testset "every offset-capable adapter: $(r.name)" for r in adapter_rows
        @test isfinite(r.fit.loglik)
        ad = GM._family_ci(r.fit, r.Y; r.kw..., offset = O4)
        @test -ad.nll(ad.θ) ≈ r.fit.loglik atol = 1e-6 * max(1, abs(r.fit.loglik))
        @test_throws ArgumentError confint(r.fit, r.Y; r.kw...)
        ci = confint(r.fit, r.Y; r.kw..., offset = O4)
        @test length(ci.se) == length(ad.θ)
        @test isapprox(ad.refit(r.Y).θ, ad.θ; atol = 1e-3)
        S = ad.simulate(MersenneTwister(1))
        @test size(S) == size(r.Y)
    end

    @testset "the same check refuses a wrong N or a dropped mask" begin
        cb = cases[3]                                  # Binomial with N
        @test_throws ArgumentError confint(cb.fit, cb.Y; N = cb.kw.N .+ 1, offset = O)
        # coef_table forwards the keyword
        ct = coef_table(cases[1].fit, cases[1].Y; offset = O)
        @test length(ct.term) == length(cases[1].θ)
        # a masked fit is refused when the mask is dropped, and accepted when it is given
        Yp = cases[1].Y
        mk = trues(p, n); mk[1, 1:20] .= false
        fm = fit_poisson_gllvm(Yp; K = K, offset = O, mask = mk)
        @test_throws ArgumentError confint(fm, Yp; offset = O)
        @test confint(fm, Yp; offset = O, mask = mk).pd_hessian
        # the Exponential adapter used to ignore `mask` altogether
        Ye = first(c for c in cases if c.name == "Exponential").Y
        fe = fit_exponential_gllvm(Ye; K = K, offset = O, mask = mk)
        @test_throws ArgumentError confint(fe, Ye; offset = O)
        @test confint(fe, Ye; offset = O, mask = mk).pd_hessian
    end

    @testset "AGHQ fits keep their offset; Laplace kept under aghq = 1 needs it passed" begin
        pa, na = 5, 60
        ra = MersenneTwister(4)
        Oa = 0.5 .* randn(ra, pa, na)
        β0 = 0.3 .* randn(ra, pa) .+ 0.3; Λ0 = 0.6 .* randn(ra, pa, 1)
        Ya = Matrix{Int}(undef, pa, na)
        for s in 1:na
            η = β0 .+ Oa[:, s] .+ Λ0 * randn(ra, 1)
            for t in 1:pa
                Ya[t, s] = rand(ra, Poisson(exp(η[t])))
            end
        end
        fa = fit_poisson_gllvm(Ya; K = 1, aghq = 3, offset = Oa)
        @test fa.integration.actual === :aghq
        ca = confint(fa, Ya)                           # the offset is on the fit
        @test ca.objective === :aghq
        @test ca.pd_hessian
        @test isequal(confint(fa, Ya; offset = Oa).se, ca.se)
        @test_throws ArgumentError confint(fa, Ya; offset = Oa .+ 0.1)
        @test_throws ArgumentError confint(fa, Ya; offset = zeros(pa, na))
        fl = fit_poisson_gllvm(Ya; K = 1, aghq = 1, offset = Oa)   # Laplace retained
        @test fl.integration.actual === :laplace
        @test_throws ArgumentError confint(fl, Ya)
        @test confint(fl, Ya; offset = Oa).pd_hessian
        # binomial AGHQ
        Nb = fill(6, pa, na)
        Yb = Matrix{Int}(undef, pa, na)
        for s in 1:na
            η = 0.3 .* β0 .+ Oa[:, s] .+ Λ0 * randn(ra, 1)
            for t in 1:pa
                Yb[t, s] = rand(ra, Binomial(6, 1 / (1 + exp(-η[t]))))
            end
        end
        fb = fit_binomial_gllvm(Yb; K = 1, N = Nb, aghq = 3, offset = Oa)
        @test fb.integration.actual === :aghq
        @test confint(fb, Yb; N = Nb).pd_hessian
        @test_throws ArgumentError confint(fb, Yb; N = Nb, offset = Oa .+ 0.1)
    end

    @testset "a fit type that cannot take an offset refuses one" begin
        Yo = rand(MersenneTwister(2), 1:3, p, 60)
        fo = fit_ordinal_gllvm(Yo; K = 1)
        @test_throws ArgumentError confint(fo, Yo; offset = zeros(p, 60))
        # the VA objective has no offset route
        Yp = cases[1].Y
        @test_throws ArgumentError confint(cases[1].fit, Yp; offset = O, objective = :va)
    end

    @testset "offset shapes: scalar, per-trait, per-unit, p x n; bad shapes refused" begin
        Yp = cases[1].Y; fp = cases[1].fit
        base = confint(fp, Yp; offset = O)
        @test confint(fp, Yp; offset = copy(O)).se == base.se
        @test_throws ArgumentError confint(fp, Yp; offset = O[:, 1:10])           # wrong size
        @test_throws ArgumentError confint(fp, Yp; offset = zeros(p, n, 2))        # 3-d
        @test_throws ArgumentError confint(fp, Yp; offset = fill(NaN, p, n))       # non-finite
        @test_throws ArgumentError confint(fp, Yp; offset = "log(2)")
        # stretched shapes agree with the p x n matrix they stand for (constant fit)
        Yc = _gen_counts(MersenneTwister(7), p, n, K, zeros(p, n); kind = :poisson)
        ut = collect(range(-0.3, 0.3; length = p))                  # one offset per trait
        fu = fit_poisson_gllvm(Yc; K = K, offset = repeat(ut, 1, n))
        a = confint(fu, Yc; offset = repeat(ut, 1, n))
        @test confint(fu, Yc; offset = ut).se == a.se                          # length-p vector
        @test confint(fu, Yc; offset = reshape(ut, p, 1)).se == a.se           # p x 1
        un = collect(range(-0.4, 0.4; length = n))                  # one offset per unit
        fun = fit_poisson_gllvm(Yc; K = K, offset = repeat(reshape(un, 1, n), p, 1))
        b = confint(fun, Yc; offset = repeat(reshape(un, 1, n), p, 1))
        @test confint(fun, Yc; offset = reshape(un, 1, n)).se == b.se          # 1 x n
    end
end

@testset "Gaussian route keeps the offset on the fit" begin
    p, n, K = 5, 80, 1
    rng = MersenneTwister(31)
    O = 0.5 .* randn(rng, p, n)
    β0 = 0.5 .* randn(rng, p); Λ0 = 0.7 .* randn(rng, p, K); σ0 = 0.6
    Y = [β0[t] + O[t, s] for t in 1:p, s in 1:n] .+ Λ0 * randn(rng, K, n) .+ σ0 .* randn(rng, p, n)
    fit = fit_gllvm(Y; family = Normal(), K = K, offset = O)
    @test fit.converged
    q = length(fit.pars.β)
    @test q == p
    # reference: closed-form Gaussian marginal with the offset, from the packed layout
    # [β (q); log σ; pack_lambda(Λ)] — the offset-aware objective, written out here.
    rr = GM.rr_theta_len(p, K)
    nll = function (θ)
        β = θ[1:p]; σ = exp(θ[p + 1]); Λ = GM.unpack_lambda(θ[(p + 2):(p + 1 + rr)], p, K)
        Σ = Λ * Λ' + σ^2 * I
        F = cholesky(Symmetric(Σ))
        acc = 0.0
        for s in 1:n
            e = Y[:, s] .- β .- O[:, s]
            acc += (p * log(2π) + logdet(F) + dot(e, F \ e)) / 2
        end
        return acc
    end
    θ = copy(fit.pars.θ_packed)
    @test -nll(θ) ≈ fit.logLik atol = 1e-6 * abs(fit.logLik)
    ref = _ref_wald(nll, θ)
    ci = confint(fit, Y)                                  # no offset keyword: it is on the fit
    @test ci.pd_hessian == ref.pd
    @test isapprox(ci.se, ref.se; rtol = 5e-3)
    @test confint(fit, Y; offset = O).se == ci.se                  # passing the same one is fine
    @test_throws ArgumentError confint(fit, Y; offset = O .+ 0.1)  # another one is refused
    # profile and bootstrap run on the stored offset as well
    pr = confint(fit, Y; method = :profile, parm = "beta[1]")
    @test isapprox(pr.lower[1], ci.lower[1]; atol = 0.03) && isapprox(pr.upper[1], ci.upper[1]; atol = 0.03)
    bt = confint(fit, Y; method = :bootstrap, n_boot = 12, seed = 1, parm = "beta")
    @test bt.n_converged >= 10
    # a constant offset leaves the loading and sigma SEs unchanged and shifts the intercepts
    Yc = [β0[t] for t in 1:p, s in 1:n] .+ Λ0 * randn(MersenneTwister(3), K, n) .+ σ0 .* randn(MersenneTwister(4), p, n)
    f0 = fit_gllvm(Yc; family = Normal(), K = K)
    fc = fit_gllvm(Yc; family = Normal(), K = K, offset = fill(1.2, p, n))
    c0 = confint(f0, Yc); cc = confint(fc, Yc)
    @test isapprox(cc.se, c0.se; rtol = 5e-3)
    @test isapprox(cc.estimate[1:p], c0.estimate[1:p] .- 1.2; atol = 2e-3)
end

@testset "confint_lv_effects: offset fits are not read as offset-free" begin
    p, n, K = 5, 80, 1
    rng = MersenneTwister(12)
    X = reshape(randn(rng, n), n, 1)
    β0 = 0.3 .* randn(rng, p) .+ 0.5; Λ0 = 0.5 .* randn(rng, p, K)
    cst = 1.0
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        z = 0.8 .* X[s, 1] .+ randn(rng, K)
        η = β0 .+ cst .+ Λ0 * z
        for t in 1:p
            Y[t, s] = rand(rng, Poisson(exp(η[t])))
        end
    end
    Oc = fill(cst, p, n)
    f0 = fit_poisson_gllvm(Y; K = K, X_lv = X, offset = nothing)
    # shifted data: the same dataset fitted with the offset removed from the mean
    fo = fit_poisson_gllvm(Y; K = K, X_lv = X, offset = Oc)
    @test fo.converged
    @test_throws ArgumentError confint_lv_effects(fo, Y, X)            # was: offset-free objective
    w = confint_lv_effects(fo, Y, X; offset = Oc)
    @test w.pd_hessian
    @test all(isfinite, w.se)
    # the offset-free fit of the same data absorbs the constant in beta; B_lv is identical
    @test isapprox(vec(GM.extract_lv_effects(fo)), vec(GM.extract_lv_effects(f0)); atol = 5e-2)
    w0 = confint_lv_effects(f0, Y, X)
    @test isapprox(w.se, w0.se; rtol = 0.05)
    @test_throws ArgumentError confint_lv_effects(fo, Y, X; offset = Oc, method = :bootstrap, n_boot = 4)
    # the bootstrap route used to return before the objective check: a fit made with an offset,
    # bootstrapped without it, silently simulated from and refitted an offset-free model
    @test_throws ArgumentError confint_lv_effects(fo, Y, X; method = :bootstrap, n_boot = 4)
    # an offset-free X_lv fit bootstraps as before
    b0 = confint_lv_effects(f0, Y, X; method = :bootstrap, n_boot = 4, seed = 1)
    @test length(b0.term) == p
    # the message names every way the rebuilt objective can fail to be the fit's own
    msg = try confint_lv_effects(fo, Y, X); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    for word in ("offset", "`N`", "`X_lv`")
        @test occursin(word, msg)
    end
end

# ---------------------------------------------------------------------------------------
# The check must refuse a different objective, not a different way of fitting the same one.
# An offset-free confint is the route main takes: the Wald table of the adapter built
# directly, with no check in front of it. These references bypass the check.
# ---------------------------------------------------------------------------------------
function _unguarded_wald(fit, Y; covariance = false, kw...)
    ad = GM._family_ci(fit, Y; objective = :laplace, kw...)
    return GM._family_wald(ad, collect(1:length(ad.θ)), 0.95; covariance = covariance)
end

function _same_as_main(fit, Y; kw...)
    ref = _unguarded_wald(fit, Y; kw...)
    ci = confint(fit, Y; kw...)
    ok = isequal(ci.se, ref.se) && isequal(ci.lower, ref.lower) && isequal(ci.upper, ref.upper) &&
         isequal(ci.estimate, ref.estimate) && ci.pd_hessian == ref.pd_hessian
    V = vcov(fit, Y; kw...)
    ok &= isequal(V, _unguarded_wald(fit, Y; covariance = true, kw...).covariance)
    ct = coef_table(fit, Y; kw...)
    ok &= ct.term == ci.term && isequal(ct.std_error, ci.se)
    return ok
end

@testset "offset-free variational fits are not refused (default objective)" begin
    p, n, K = 6, 80, 1
    rng = MersenneTwister(2026)
    Z = zeros(p, n)
    Nm = fill(6, p, n)
    Yb = _sim_generic(rng, p, n, K, Z, (r, t, η) -> clamp(rand(r, Beta(5 / (1 + exp(-η)), 5 * (1 - 1 / (1 + exp(-η))))), 1e-4, 1 - 1e-4))
    rows = Any[
        ("Poisson", let Y = _gen_counts(rng, p, n, K, Z; kind = :poisson); (fit_poisson_gllvm_va(Y; K = K), Y, (;)) end),
        ("NB2", let Y = _gen_counts(rng, p, n, K, Z; kind = :nb); (fit_nb_gllvm_va(Y; K = K), Y, (;)) end),
        ("Binomial (N)", let Y = _gen_binomial(rng, p, n, K, Z, Nm); (fit_binomial_gllvm_va(Y; N = Nm, K = K), Y, (N = Nm,)) end),
        ("Beta", (fit_beta_gllvm_va(Yb; K = K), Yb, (;))),
        ("Gamma", let Y = _gen_gamma(rng, p, n, K, Z); (fit_gamma_gllvm_va(Y; K = K), Y, (;)) end),
        ("Exponential", let Y = _gen_exponential(rng, p, n, K, Z); (fit_exponential_gllvm_va(Y; K = K), Y, (;)) end),
        ("Delta-Gamma", let Y = _gen_delta_gamma(rng, p, n, K, Z); (fit_delta_gamma_gllvm_va(Y; K = K), Y, (;)) end),
    ]
    @testset "$(r[1]): runs and equals the unchecked Wald table" for r in rows
        fit, Y, kw = r[2]
        # the fit stores the ELBO, which is not the Laplace marginal the default objective rebuilds
        ad = GM._family_ci(fit, Y; objective = :laplace, kw...)
        @test abs(-ad.nll(ad.θ) - fit.loglik) > 1e-4
        @test _same_as_main(fit, Y; kw...)
        # a variational fit takes no offset, so one passed with it is refused
        @test_throws ArgumentError confint(fit, Y; kw..., offset = fill(0.5, p, n))
    end
end

@testset "offset-free hessian = :fisher fits are not refused" begin
    p, n, K = 6, 80, 1
    rng = MersenneTwister(2026)
    Z = zeros(p, n)
    O = 0.5 .* randn(rng, p, n)
    tnb(O) = _sim_generic(rng, p, n, K, O, (r, t, η) -> GM._rand_ztnb(r, 3.0, exp(η)))
    # name, fit(Y; offset, hessian), data generator
    builders = Any[
        ("Delta-Gamma (separate)", (Y; kw...) -> fit_delta_gamma_gllvm(Y; K = K, kw...),
            O -> _gen_delta_gamma(rng, p, n, K, O; shared = false)),
        ("Delta-Gamma (shared)", (Y; kw...) -> fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, kw...),
            O -> _gen_delta_gamma(rng, p, n, K, O; shared = true)),
        ("Truncated NB2", (Y; kw...) -> fit_truncated_nbinom2_gllvm(Y; K = K, kw...), tnb),
        ("Truncated NB2 (per trait)", (Y; kw...) -> fit_truncated_nbinom2_gllvm_pertrait(Y; K = K, kw...), tnb),
    ]
    @testset "$(b[1])" for b in builders
        _, fitter, gen = b
        Y = gen(Z)
        f = fitter(Y; hessian = :fisher)
        # the fit's own objective is the :fisher one; the adapter rebuilds the :observed one, so
        # the log-likelihoods differ, yet this is the offset-free route main took
        ad = GM._family_ci(f, Y; objective = :laplace)
        @test abs(-ad.nll(ad.θ) - f.loglik) > 1e-4
        @test _same_as_main(f, Y)
        # with an offset: accepted when it is passed, refused when dropped or wrong
        Yo = gen(O)
        fo = fitter(Yo; offset = O, hessian = :fisher)
        @test confint(fo, Yo; offset = O).pd_hessian isa Bool
        @test_throws ArgumentError confint(fo, Yo)
        @test_throws ArgumentError confint(fo, Yo; offset = O .+ 0.3)
        # and the :observed fit of the same data still checks out as before
        f2 = fitter(Yo; offset = O)
        @test confint(f2, Yo; offset = O).pd_hessian isa Bool
        @test_throws ArgumentError confint(f2, Yo)
    end
end

@testset "the refusal names every cause, and a non-default link is not blamed on the offset" begin
    p, n, K = 5, 80, 1
    rng = MersenneTwister(18)
    O = 0.5 .* randn(rng, p, n)
    Y = _gen_counts(rng, p, n, K, O; kind = :poisson)
    f = fit_poisson_gllvm(Y; K = K, offset = O)
    msg = try confint(f, Y); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    @test !isempty(msg)
    # offset, N, X, mask, the variational objective, the curvature, a non-default link
    for word in ("offset", "`N`", "`X`", "`mask`", "variational", "hessian = :fisher", "link")
        @test occursin(word, msg)
    end
    # a wrong offset gets the same text (it is told what to pass, not only that something is off)
    msg2 = try confint(f, Y; offset = O .+ 0.3); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    @test occursin("the offset you passed", msg2)

    # Beta (probit) and Gamma (identity): the fitters optimise the default-link objective while
    # the adapters rebuild the one for the fit's link, so the fit's own objective cannot be
    # reproduced. Offset-free calls are refused (they returned pd_hessian = false before), and
    # the text says that, not that an offset is missing. When the fitters are fixed these calls
    # succeed and the second branch applies.
    Zp = zeros(p, n)
    Yb = _sim_generic(rng, p, n, K, Zp, (r, t, η) -> clamp(rand(r, Beta(5 / (1 + exp(-η)), 5 * (1 - 1 / (1 + exp(-η))))), 1e-4, 1 - 1e-4))
    Yg = _gen_gamma(rng, p, n, K, Zp)
    @testset "$(nm)" for (nm, fit, Yx) in (
            ("Beta, probit link", fit_beta_gllvm(Yb; K = K, link = GM.ProbitLink()), Yb),
            ("Gamma, identity link", fit_gamma_gllvm(Yg; K = K, link = GM.IdentityLink()), Yg))
        r = try confint(fit, Yx); nothing catch e; e end
        if r === nothing
            @test true
        else
            @test r isa ArgumentError
            @test occursin("non-default link", r.msg)
            @test !occursin("If the fit was made with an `offset`", r.msg)
        end
    end
end

end  # outer testset
