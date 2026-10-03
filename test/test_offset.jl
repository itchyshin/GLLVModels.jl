using GLLVModels, Test, Random, Distributions, LinearAlgebra, SparseArrays, StatsModels

@testset "Offsets in the linear predictor" begin
    Random.seed!(4242)
    p, n, K = 4, 9, 2
    β = randn(p) .* 0.3
    Λ = randn(p, K) .* 0.4
    Y = rand(0:6, p, n)

    # ---- Anchor 1: offset = 0 ≡ no offset (machine precision) --------------
    ℓ0 = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β, LogLink())
    ℓz = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β, LogLink();
                                       offset = zeros(p, n))
    @test isapprox(ℓ0, ℓz; atol = 1e-10)

    # ---- Anchor 2: offset-absorption identity (machine precision) ----------
    # A constant per-species offset c_t shifts that species' intercept:
    #   η = β + offset + Λz  with offset[t,s] = c_t  ==  η = (β + c) + Λz.
    c = randn(p) .* 0.5
    O = repeat(c, 1, n)                       # p×n, constant within each species row
    ℓ_off  = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β,     LogLink(); offset = O)
    ℓ_shift = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β .+ c, LogLink())
    @test isapprox(ℓ_off, ℓ_shift; atol = 1e-9)

    # ---- Anchor 3: a general (non-constant) offset changes the marginal ----
    Ovar = randn(p, n) .* 0.3
    ℓ_var = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β, LogLink(); offset = Ovar)
    @test isfinite(ℓ_var)
    @test ℓ_var != ℓ0

    # ---- Fit-level absorption: fitting with a constant offset c recovers the
    # no-offset intercepts shifted by −c (same loglik), since the model is
    # reparameterised. -------------------------------------------------------
    Random.seed!(11)
    Yf = rand(0:5, p, n)
    f0 = fit_poisson_gllvm(Yf; K = K, iterations = 60)
    fO = fit_poisson_gllvm(Yf; K = K, offset = O, iterations = 60)
    @test isapprox(f0.loglik, fO.loglik; atol = 1e-4)         # same maximised likelihood
    @test isapprox(f0.β, fO.β .+ c; atol = 1e-2)              # intercepts shifted by −c

    # ---- Same absorption across the NB / Binomial / Beta / Gamma fitters ----
    # A constant per-species offset shifts β by −c, leaving the dispersion and the
    # maximised loglik unchanged (the warm start subtracts the offset, so the
    # optimisation traces the identical path up to the β-shift).
    @testset "offset absorption across GLM fitters" begin
        Random.seed!(7)
        pp, nn = 4, 12
        cc = randn(pp) .* 0.4
        O2 = repeat(cc, 1, nn)

        Yn = rand(0:6, pp, nn)
        a0 = fit_nb_gllvm(Yn; K = 1, iterations = 60)
        aO = fit_nb_gllvm(Yn; K = 1, offset = O2, iterations = 60)
        @test isapprox(a0.loglik, aO.loglik; atol = 1e-4)
        @test isapprox(a0.β, aO.β .+ cc; atol = 2e-2)
        @test isapprox(a0.r, aO.r; rtol = 1e-3)

        Ntr = fill(6, pp, nn); Yb = rand(0:6, pp, nn)
        b0 = fit_binomial_gllvm(Yb; K = 1, N = Ntr, iterations = 60)
        bO = fit_binomial_gllvm(Yb; K = 1, N = Ntr, offset = O2, iterations = 60)
        @test isapprox(b0.loglik, bO.loglik; atol = 1e-4)
        @test isapprox(b0.β, bO.β .+ cc; atol = 2e-2)

        Ybeta = clamp.(rand(pp, nn), 0.02, 0.98)
        c0 = fit_beta_gllvm(Ybeta; K = 1, iterations = 60)
        cO = fit_beta_gllvm(Ybeta; K = 1, offset = O2, iterations = 60)
        @test isapprox(c0.loglik, cO.loglik; atol = 1e-4)
        @test isapprox(c0.β, cO.β .+ cc; atol = 2e-2)
        @test isapprox(c0.φ, cO.φ; rtol = 1e-3)

        Yg = 0.5 .+ 2 .* rand(pp, nn)
        g0 = fit_gamma_gllvm(Yg; K = 1, iterations = 60)
        gO = fit_gamma_gllvm(Yg; K = 1, offset = O2, iterations = 60)
        @test isapprox(g0.loglik, gO.loglik; atol = 1e-4)
        @test isapprox(g0.β, gO.β .+ cc; atol = 2e-2)
        @test isapprox(g0.α, gO.α; rtol = 1e-3)
    end

    # ---- Two-part substrate: offset on the positive part (offsetc) ---------
    # η^c = β^c + offsetc + Λ^c z. offsetc = 0 ≡ no offset; a constant per-species
    # offsetc ≡ shifting β^c (the absorption identity), both machine precision.
    @testset "two-part offsetc absorption (Delta-Gamma marginal)" begin
        Random.seed!(515)
        pp, K, nn = 4, 1, 30
        βz = 0.3 .* randn(pp); βc = 0.2 .* randn(pp); α = 3.0
        Λc = 0.3 .* randn(pp, K)
        Y = zeros(pp, nn)
        for s in 1:nn
            ηc = βc .+ Λc * randn(K)
            for t in 1:pp
                rand() < inv(1 + exp(-βz[t])) && (Y[t, s] = rand(Gamma(α, exp(ηc[t]) / α)))
            end
        end

        ℓ0 = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α)
        ℓz = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α; offsetc = zeros(pp, nn))
        @test isapprox(ℓ0, ℓz; atol = 1e-9)

        cc = 0.5 .* randn(pp); O = repeat(cc, 1, nn)
        ℓ_off = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α; offsetc = O)
        ℓ_sh  = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc .+ cc, α)
        @test isapprox(ℓ_off, ℓ_sh; atol = 1e-8)

        # A non-constant offsetc changes the marginal.
        ℓ_v = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α; offsetc = 0.3 .* randn(pp, nn))
        @test isfinite(ℓ_v) && ℓ_v != ℓ0
    end

    # ---- fit_delta_gamma_gllvm offset kwarg (reference two-part fitter) -----
    # The maximised likelihood is invariant to a constant positive-part offset (it
    # reparameterises β^c), so the offset fit matches the no-offset fit's loglik and
    # its β^c is shifted by −c. (loglik invariance is robust to warm-start details.)
    @testset "fit_delta_gamma_gllvm offset" begin
        Random.seed!(616)
        p, K, n = 3, 1, 120
        βz = 0.4 .* randn(p) .+ 0.3; βc = 0.2 .* randn(p); α = 3.0
        Λc = 0.3 .* randn(p, K)
        Y = zeros(p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                rand() < inv(1 + exp(-βz[t])) && (Y[t, s] = rand(Gamma(α, exp(ηc[t]) / α)))
            end
        end
        cc = 0.5 .* randn(p); O = repeat(cc, 1, n)

        # disp_group=:shared pinned: the reparam-invariance tolerances below were
        # measured against the (previously default) 1-parameter shared alpha; under
        # :species (p=3 free shapes on this small dataset) the two 100-iteration
        # fits land at distinguishable local optima and these atols are too tight —
        # a convergence-budget artifact of :species, not a break in the offset
        # absorption identity itself. :shared keeps this test's original behaviour.
        f0 = fit_delta_gamma_gllvm(Y; K = K, disp_group = :shared, iterations = 100)
        fO = fit_delta_gamma_gllvm(Y; K = K, disp_group = :shared, offset = O, iterations = 100)
        @test isfinite(fO.loglik)
        @test isapprox(f0.loglik, fO.loglik; atol = 2e-2)      # reparam-invariant
        @test isapprox(f0.βc, fO.βc .+ cc; atol = 1.5e-1)      # β^c shifted by −c
    end

    # ---- offset on the remaining two-part fitters --------------------------
    # Marginal-level absorption (machine precision) + each fitter accepts offset.
    @testset "remaining two-part fitter offsets" begin
        Random.seed!(717)
        pp, K, nn = 3, 1, 40
        βz = 0.3 .* randn(pp); βc = 0.2 .* randn(pp)
        Λc = 0.3 .* randn(pp, K)
        Y = Float64.([rand() < inv(1 + exp(-βz[t])) ? rand(1:6) : 0 for t in 1:pp, s in 1:nn])
        cc = 0.4 .* randn(pp); O = repeat(cc, 1, nn)

        # Constant offsetc ≡ shifting β^c (hurdle-Poisson marginal), machine precision.
        ℓ_off = GLLVModels.hurdle_poisson_marginal_loglik_laplace(Y, Λc, βz, βc; offsetc = O)
        ℓ_sh  = GLLVModels.hurdle_poisson_marginal_loglik_laplace(Y, Λc, βz, βc .+ cc)
        @test isapprox(ℓ_off, ℓ_sh; atol = 1e-8)

        # Each remaining two-part fitter accepts `offset` and runs.
        @test isfinite(fit_hurdle_poisson_gllvm(Y; K = K, offset = O, iterations = 40).loglik)
        @test isfinite(fit_hurdle_nb_gllvm(Y; K = K, offset = O, iterations = 40).loglik)
        @test isfinite(fit_delta_lognormal_gllvm(Y; K = K, offset = O, iterations = 40).loglik)
        @test isfinite(fit_zip_gllvm(Y; K = K, offset = O, iterations = 40).loglik)
        @test isfinite(fit_zinb_gllvm(Y; K = K, offset = O, iterations = 40).loglik)
    end
end

# ---- fit_gllvm: a scalar offset is broadcast; unusable shapes are refused ------
# R's gllvmTMB accepts `offset(log(2))` (a constant) and broadcasts it to every row.
# Before the fix, `offset = log(2)` was accepted by the count routes and returned
# loglik = -Inf, converged = false (every objective call threw inside the fitter's
# `try ... catch`, which turns any exception into a sentinel); Gaussian threw a
# DimensionMismatch. The dispatcher now turns a real scalar into the p×n matrix
# `fill(c, p, n)` and refuses any shape it cannot broadcast unambiguously.
@testset "fit_gllvm scalar offset: broadcast, or a clear ArgumentError" begin
    Random.seed!(20261002)
    p, n = 4, 40                       # p ≠ n, so a transposed shape is detectable
    c = log(2)
    O = fill(c, p, n)
    Yc = rand(NegativeBinomial(1.5, 0.3), p, n)    # counts, clearly overdispersed (variance ≈ 3.3 × mean)
    Yn = randn(p, n) .+ 1.0                        # Gaussian responses

    # The scalar fit must be the p×n-matrix fit, to optimiser precision.
    function same_fit(f_scalar, f_matrix; ll = :loglik)
        a = getproperty(f_scalar, ll); b = getproperty(f_matrix, ll)
        @test isfinite(a)
        @test f_scalar.converged
        @test f_scalar.converged == f_matrix.converged
        @test isapprox(a, b; atol = 1e-8, rtol = 0)
    end

    @testset "Poisson" begin
        fs = fit_gllvm(Yc; family = Poisson(), K = 1, offset = c)
        fm = fit_gllvm(Yc; family = Poisson(), K = 1, offset = O)
        same_fit(fs, fm)
        @test isapprox(fs.β, fm.β; atol = 1e-8, rtol = 0)
        # a constant offset is absorbed by the intercepts: same logLik as no offset, β shifted by -c
        f0 = fit_gllvm(Yc; family = Poisson(), K = 1)
        @test isapprox(fs.loglik, f0.loglik; atol = 1e-4, rtol = 0)
        @test isapprox(fs.β, f0.β .- c; atol = 1e-3, rtol = 0)
        # an Integer scalar is a real scalar too
        fi = fit_gllvm(Yc; family = Poisson(), K = 1, offset = 1)
        same_fit(fi, fit_gllvm(Yc; family = Poisson(), K = 1, offset = fill(1.0, p, n)))
        # a zero scalar is the no-offset fit
        fz = fit_gllvm(Yc; family = Poisson(), K = 1, offset = 0.0)
        @test isapprox(fz.loglik, f0.loglik; atol = 1e-8, rtol = 0)
    end

    @testset "NB2 (shared r, and per-species r)" begin
        fs = fit_gllvm(Yc; family = NegativeBinomial(), K = 1, offset = c)
        fm = fit_gllvm(Yc; family = NegativeBinomial(), K = 1, offset = O)
        same_fit(fs, fm)
        fgs = fit_gllvm(Yc; family = NegativeBinomial(), K = 1, disp_group = :species, offset = c)
        fgm = fit_gllvm(Yc; family = NegativeBinomial(), K = 1, disp_group = :species, offset = O)
        same_fit(fgs, fgm)
    end

    @testset "Normal" begin
        fs = fit_gllvm(Yn; family = Normal(), K = 1, offset = c)
        fm = fit_gllvm(Yn; family = Normal(), K = 1, offset = O)
        same_fit(fs, fm; ll = :logLik)
        # a zero scalar is the no-offset fit (this used to be a DimensionMismatch)
        f0 = fit_gllvm(Yn; family = Normal(), K = 1)
        fz = fit_gllvm(Yn; family = Normal(), K = 1, offset = 0.0)
        @test isapprox(fz.logLik, f0.logLik; atol = 1e-8, rtol = 0)
    end

    @testset "K omitted (select_lv route) broadcasts too" begin
        sel = fit_gllvm(Yc; family = Poisson(), offset = c, Kmax = 2)
        @test isfinite(sel.loglik)
    end

    # Julia's own broadcast rule: a length-p vector and a p×1 matrix are one offset per trait,
    # a 1×n matrix is one offset per unit. The vector and the 1×n matrix already worked on the
    # count routes (the family code adds the offset by broadcast); the p×1 matrix returned a
    # -Inf or wrong fit. All three now expand to the p×n matrix before any fitting.
    @testset "length-p vector, p×1 and 1×n matrices stretch to p×n" begin
        v = [0.1, 0.5, -0.3, 0.8]                       # one offset per trait
        u = collect(range(0.0, 1.0; length = n))         # one offset per unit
        Ov = repeat(v, 1, n); Ou = repeat(u', p, 1)
        for (label, short, full) in (("length-p vector", v, Ov), ("p×1 matrix", reshape(v, p, 1), Ov),
                                     ("1×n matrix", reshape(u, 1, n), Ou), ("1×n adjoint", u', Ou),
                                     ("1×1 matrix", fill(c, 1, 1), O))
            @testset "$label" begin
                same_fit(fit_gllvm(Yc; family = Poisson(), K = 1, offset = short),
                         fit_gllvm(Yc; family = Poisson(), K = 1, offset = full))
                same_fit(fit_gllvm(Yn; family = Normal(), K = 1, offset = short),
                         fit_gllvm(Yn; family = Normal(), K = 1, offset = full); ll = :logLik)
            end
        end
        # the offsets are really applied, not just accepted: per-trait constants are absorbed by
        # the intercepts (same logLik, β shifted by -v); a per-unit offset is not absorbed
        f0 = fit_gllvm(Yc; family = Poisson(), K = 1)
        fv = fit_gllvm(Yc; family = Poisson(), K = 1, offset = v)
        @test isapprox(fv.loglik, f0.loglik; atol = 1e-3, rtol = 0)
        @test isapprox(fv.β, f0.β .- v; atol = 1e-2, rtol = 0)
        fu = fit_gllvm(Yc; family = Poisson(), K = 1, offset = reshape(u, 1, n))
        @test abs(fu.loglik - f0.loglik) > 1
        # the expansion itself is exact
        @test GLLVModels._normalize_offset(reshape(u, 1, n), p, n) == Ou
        @test GLLVModels._normalize_offset(reshape(v, p, 1), p, n) == Ov
        @test GLLVModels._normalize_offset(v, p, n) == Ov
        @test GLLVModels._normalize_offset(fill(c, 1, 1), p, n) == O
    end

    @testset "a p×n matrix and `nothing` pass through untouched" begin
        @test GLLVModels._normalize_offset(O, p, n) === O
        @test GLLVModels._normalize_offset(nothing, p, n) === nothing
        Ovar = randn(p, n)
        @test GLLVModels._normalize_offset(Ovar, p, n) === Ovar
    end

    @testset "shapes that cannot be broadcast unambiguously are refused" begin
        bad = [
            "length-n vector"       => fill(c, n),
            "wrong-length vector"   => fill(c, p + 1),
            "n×p matrix"            => fill(c, n, p),
            "2×n matrix"            => fill(c, 2, n),
            "p×2 matrix"            => fill(c, p, 2),
            "p×(n+1) matrix"        => fill(c, p, n + 1),
            "(p+1)×n matrix"        => fill(c, p + 1, n),
            "3-d array"             => fill(c, p, n, 1),
            "NaN scalar"            => NaN,
            "Inf scalar"            => Inf,
            "missing"               => missing,
            "string"                => "log(2)",
        ]
        for (label, off) in bad
            @testset "$label" begin
                @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = 1, offset = off)
                @test_throws ArgumentError fit_gllvm(Yn; family = Normal(), K = 1, offset = off)
            end
        end
        # the message names what is accepted and the shape that was expected
        err = try
            fit_gllvm(Yc; family = Poisson(), K = 1, offset = fill(c, n)); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("scalar", err.msg)
        @test occursin("$(p)×$(n) matrix", err.msg)
        @test occursin("length-$(p) vector", err.msg)
        # the refusal comes before any fitting, so no fit object can carry a -Inf
        @test_throws ArgumentError fit_gllvm(Yc; family = NegativeBinomial(), K = 1,
                                             disp_group = :species, offset = fill(c, n))
    end
end

# ---- Review fixes for the scalar-offset work (#693) ------------------------------
# The guiding rule is usability: no offset may give a silent, wrong or -Inf fit. It either
# means what the user meant, or it raises an ArgumentError that says what to do.
_off_ll(f) = hasproperty(f, :loglik) ? f.loglik : f.logLik
# The intercept that the offset enters through: the count / positive-part intercept on the
# two-part and zero-inflated fits (the occurrence intercept `βz` has no offset), `β` elsewhere.
_off_beta(f) = hasproperty(f, :β) ? f.β : hasproperty(f, :βc) ? f.βc : f.pars.β

# One small data set per family. p ≠ n, so a transposed shape is detectable.
function _off_data(; p = 5, n = 36, seed = 693)
    Random.seed!(seed)
    z = randn(n); lam = [0.8, -0.6, 0.5, 0.3, -0.4]
    η = 0.6 .+ 0.3 .* randn(p) .+ lam * z'
    sig(x) = 1 / (1 + exp(-x))
    d = Dict{Symbol,Any}()
    d[:pois]  = Float64.([rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n])
    d[:nb]    = Float64.([rand(NegativeBinomial(1.5, 1.5 / (1.5 + exp(η[t, s])))) for t in 1:p, s in 1:n])   # overdispersed, so r is identified
    d[:norm]  = η .+ 0.5 .* randn(p, n)
    d[:N]     = fill(8, p, n)
    d[:bin]   = Float64.([rand(Binomial(8, sig(η[t, s]))) for t in 1:p, s in 1:n])
    d[:tpois] = Float64.([(y = rand(Poisson(exp(η[t, s] + 0.5))); y == 0 ? 1.0 : Float64(y)) for t in 1:p, s in 1:n])
    d[:logn]  = exp.(η .+ 0.4 .* randn(p, n))
    d[:gam]   = Float64.([rand(Gamma(3.0, exp(η[t, s]) / 3.0)) for t in 1:p, s in 1:n])
    d[:expo]  = Float64.([rand(Exponential(exp(η[t, s]))) for t in 1:p, s in 1:n])
    d[:beta]  = Float64.([clamp(rand(Beta(4 * sig(η[t, s]), 4 * (1 - sig(η[t, s])))), 0.01, 0.99) for t in 1:p, s in 1:n])
    ord = [clamp(round(Int, η[t, s] + 2 + 0.7 * randn()), 1, 4) for t in 1:p, s in 1:n]
    ord[:, 1] .= 1; ord[:, 2] .= 4; ord[:, 3] .= 2; ord[:, 4] .= 3   # every trait shows every category
    d[:ord]   = ord
    d[:two]   = Float64.([rand() < 0.7 ? d[:gam][t, s] : 0.0 for t in 1:p, s in 1:n])
    d[:twoc]  = Float64.([rand() < 0.7 ? Float64(max(1, d[:pois][t, s])) : 0.0 for t in 1:p, s in 1:n])
    d[:twonb] = Float64.([rand() < 0.7 ? Float64(max(1, d[:nb][t, s])) : 0.0 for t in 1:p, s in 1:n])
    d[:twob]  = Float64.([rand() < 0.7 ? d[:beta][t, s] : 0.0 for t in 1:p, s in 1:n])
    d[:zip]   = Float64.([rand() < 0.3 ? 0.0 : d[:pois][t, s] for t in 1:p, s in 1:n])
    d[:zinb]  = Float64.([rand() < 0.3 ? 0.0 : d[:nb][t, s] for t in 1:p, s in 1:n])
    d[:zib]   = Float64.([rand() < 0.3 ? 0.0 : d[:bin][t, s] for t in 1:p, s in 1:n])
    d[:ob]    = Float64.([(u = rand(); u < 0.15 ? 0.0 : u > 0.85 ? 1.0 : d[:beta][t, s]) for t in 1:p, s in 1:n])
    d[:tw]    = Float64.([rand() < 0.3 ? 0.0 : d[:gam][t, s] for t in 1:p, s in 1:n])
    d[:mn]    = reshape(rand(1:3, n), 1, n)
    return d
end

@testset "offset: every route that takes one absorbs a constant; every route that does not refuses" begin
    G = GLLVModels
    d = _off_data()
    p, n = size(d[:pois]); c = log(2)

    # Routes that take an offset: a constant offset c is absorbed by the intercepts, so the
    # maximised logLik equals the no-offset logLik and the fit converges (on the AGHQ routes:
    # reaches the same convergence verdict as the fit without an offset). A fit that returns
    # -Inf, does not converge, or lands on another logLik has not applied the offset to the
    # mean it fits.
    accepted = [
        ("Normal",               d[:norm], Normal(),                       (;)),
        ("Normal aghq",          d[:norm], Normal(),                       (; aghq = 3)),
        ("Poisson",              d[:pois], Poisson(),                      (;)),
        ("Poisson aghq",         d[:pois], Poisson(),                      (; aghq = 3)),
        ("Binomial",             d[:bin],  Binomial(),                     (; N = d[:N])),
        ("Binomial aghq",        d[:bin],  Binomial(),                     (; N = d[:N], aghq = 3)),
        ("TruncatedPoisson",     d[:tpois], G.TruncatedPoisson(),          (;)),
        ("CensoredPoisson",      d[:pois], G.CensoredPoisson(),            (;)),
        ("Lognormal",            d[:logn], G.Lognormal(),                  (;)),
        ("TruncatedNegBin2",     d[:tpois], G.TruncatedNegBin2(),          (;)),
        ("NegBin2 shared",       d[:nb],   NegativeBinomial(),             (;)),
        ("NegBin2 per-species",  d[:nb],   NegativeBinomial(),             (; disp_group = :species)),
        ("NegBin2 aghq",         d[:nb],   NegativeBinomial(),             (; disp_group = :species, aghq = 3)),
        ("Beta",                 d[:beta], Beta(),                         (;)),
        ("NB1",                  d[:nb],   G.NB1(),                        (;)),
        ("Gamma",                d[:gam],  Gamma(),                        (;)),
        ("Gamma per-species",    d[:gam],  Gamma(),                        (; disp_group = :species)),
        ("Exponential",          d[:expo], G.Exponential(),                (;)),
        ("DeltaLogNormal",       d[:two],  G.DeltaLogNormal(),             (;)),
        ("DeltaGamma",           d[:two],  G.DeltaGamma(),                 (;)),
        ("DeltaGamma aghq",      d[:two],  G.DeltaGamma(),                 (; aghq = 3)),
        ("HurdlePoisson",        d[:twoc], G.HurdlePoisson(),              (;)),
        ("HurdleNB",             d[:twonb], G.HurdleNB(),                  (;)),
        ("GeneralizedPoisson1",  d[:pois], G.GeneralizedPoisson1(0.1),     (;)),
        ("ZIPoisson",            d[:zip],  G.ZIPoisson(),                  (;)),
        ("ZINegBin",             d[:zinb], G.ZINegBin(),                   (;)),
        ("ZIB",                  d[:zib],  G.ZIB(8),                       (;)),
        ("Tweedie per-species",  d[:tw],   G.TweedieED(1.0, 1.5),          (; disp_group = :species)),
        # (Tweedie with aghq is left out: about 70 s for the pair of fits.)
    ]
    # The identity is exact for the objective; the fits are optimiser runs (finite-difference
    # gradients, g_tol = 1e-5), so two fits that differ only by rounding can stop a little
    # apart. Measured spread of logLik(offset) - logLik(none) over this data and others: 1e-14
    # to 1e-10 on most routes, up to about 3e-6 on NB1 and DeltaGamma. Most AGHQ routes
    # report converged = false on data this small, with or without an offset, and NB2 with
    # AGHQ can end 2e-3 apart. A mis-applied offset moves
    # the logLik by 0.1 or more (Lognormal before the fix: 80 to 110), so these tolerances
    # still tell the two apart.
    tol_of(label) = label in ("NB1", "DeltaGamma", "DeltaGamma aghq", "Binomial aghq", "Poisson aghq") ? 1e-5 :
                    label == "NegBin2 aghq" ? 1e-2 : 1e-6
    @testset "absorption: $label" for (label, Y, fam, kw) in accepted
        f0 = fit_gllvm(Y; family = fam, K = 1, kw...)
        fc = fit_gllvm(Y; family = fam, K = 1, kw..., offset = c)
        aghq = haskey(kw, :aghq) && label != "Normal aghq"
        # The AGHQ engines report their own, stricter verdict. Per-species NB1 can stop short of
        # its convergence test on this data on some platforms, with or without an offset (the
        # Linux CI runners, 2026-10-03; the NB1 precision fix is itchyshin/GLLVModels.jl#694).
        # Both are engine properties, not the offset's, so those routes check that the offset
        # fit reaches the same verdict as the fit without one, plus the logLik and intercepts.
        (aghq || label == "NB1") || @test f0.converged
        @test fc.converged == f0.converged
        @test isfinite(_off_ll(fc))
        @test isapprox(_off_ll(fc), _off_ll(f0); atol = tol_of(label), rtol = 0)
        # The logLik alone cannot tell a dropped, sign-flipped or scaled offset from the right
        # one (the intercepts absorb a constant either way). The intercepts can: with the
        # offset applied once and with the right sign they move by exactly -c. A dropped
        # offset leaves them where they were, a flipped one moves them by +c, a half-applied
        # one by -c/2; all are 0.35 or more from the target, far above this tolerance.
        @test isapprox(_off_beta(fc), _off_beta(f0) .- c; atol = label == "NegBin2 aghq" ? 2e-2 : 1e-3, rtol = 0)
        if haskey(kw, :aghq)                            # the scalar is the expanded matrix, exactly
            ff = fit_gllvm(Y; family = fam, K = 1, kw..., offset = fill(c, p, n))
            @test _off_ll(fc) == _off_ll(ff)
        end
    end

    # Routes whose fitter has no offset keyword: a clear ArgumentError naming the family, for
    # any offset (a valid scalar, a p×n matrix, or a shape that would otherwise fail the shape
    # check), not a MethodError and not a misleading shape error.
    refused = [
        ("StudentT",             d[:norm], G.StudentTFamily(5.0),          (;)),
        ("StudentT per-species", d[:norm], G.StudentTFamily(5.0),          (; disp_group = :species)),
        ("Ordinal",              d[:ord],  G.Ordinal(),                    (;)),
        ("Ordinal aghq",         d[:ord],  G.Ordinal(),                    (; aghq = 3)),
        ("BetaBinom",            d[:bin],  G.BetaBinom(),                  (; N = d[:N])),
        ("BetaHurdle",           d[:twob], G.BetaHurdle(),                 (;)),
        ("COMPoisson",           d[:pois], G.COMPoisson(),                 (;)),
        ("OrderedBeta",          d[:ob],   G.OrderedBeta(),                (;)),
        # the gllvmTMB-twin zero-inflated families (zi_poisson() etc.) have no offset keyword
        ("ZiPoisson",            d[:zip],  G.zi_poisson(),                 (;)),
        ("ZiNbinom2",            d[:zinb], G.zi_nbinom2(),                 (;)),
        ("ZiBinomial",           d[:zib],  G.zi_binomial(),                (; trials = 8)),
    ]
    @testset "refused: $label" for (label, Y, fam, kw) in refused
        for off in (c, fill(c, p, n), fill(c, n))
            err = try
                fit_gllvm(Y; family = fam, K = 1, kw..., offset = off); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            err isa ArgumentError && @test occursin("offset is not supported for family $(nameof(typeof(fam)))", err.msg)
        end
        # `offset = nothing` is "no offset", on a route that takes none too
        @test isfinite(_off_ll(fit_gllvm(Y; family = fam, K = 1, kw..., offset = nothing)))
    end
    @testset "refused: Multinomial" begin
        err = try fit_gllvm(d[:mn]; family = G.Multinomial(), offset = c); nothing catch e e end
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin("offset is not supported for family Multinomial", err.msg)
    end
    @testset "refused: the K-omitted route says so before sweeping K" begin
        err = try fit_gllvm(d[:ord]; family = G.Ordinal(), offset = c); nothing catch e e end
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin("offset is not supported for family Ordinal", err.msg)
    end

    # Structural routes that take no offset: pervar, row_eff, explicit grouping, explicit phylo.
    @testset "refused: pervar, row_eff, grouping, phylo" begin
        Yn = d[:norm]
        e1 = try fit_gllvm(Yn; family = Normal(), K = 1, pervar = true, offset = c); nothing catch e e end
        @test e1 isa ArgumentError && occursin("offset is not supported for pervar", e1.msg)
        for re in (:fixed, :random)
            e2 = try fit_gllvm(d[:pois]; family = Poisson(), K = 1, row_eff = re, offset = c); nothing catch e e end
            @test e2 isa ArgumentError && occursin("offset is not supported for row_eff = :$re", e2.msg)
        end
        cluster = repeat(1:6, inner = 6)
        e3 = try
            fit_gllvm(Yn; family = Normal(), grouping = [GroupingTerm(:cluster; mode = :indep)],
                      cluster = cluster, offset = c); nothing
        catch e
            e
        end
        @test e3 isa ArgumentError && occursin("offset is not supported for explicit grouping", e3.msg)
        Q = sparse([4.0 -1.0 -1.0 0.0; -1.0 3.0 0.0 -1.0; -1.0 0.0 3.0 0.0; 0.0 -1.0 0.0 3.0])
        i, j, x = findnz(Q)
        phy = PrecisionPhy(i, j, x, 4, 2, ["a1", "a2", "s1", "s2"],
                           logdet(cholesky(Symmetric(Matrix(Q)))), 1.7, [3, 4])
        e4 = try
            fit_gllvm(randn(3, 4); family = Normal(), phylo = phy, species_id = [1, 2, 1, 2], offset = c); nothing
        catch e
            e
        end
        @test e4 isa ArgumentError && occursin("offset is not supported for explicit precision", e4.msg)
    end
end

@testset "offset: Lognormal centres log(Y) - offset, so the offset is part of the mean" begin
    Random.seed!(2026100201)
    p, n = 6, 40; c = log(2)
    Yl = exp.(0.5 .+ 0.3 .* randn(p, n))
    f0 = fit_gllvm(Yl; family = GLLVModels.Lognormal(), K = 1)
    # scalar c: absorbed by the intercepts (logLik unchanged, β shifted by exactly -c), as for Poisson
    fs = fit_gllvm(Yl; family = GLLVModels.Lognormal(), K = 1, offset = c)
    @test fs.converged
    @test isapprox(fs.loglik, f0.loglik; atol = 1e-8, rtol = 0)
    @test isapprox(fs.β, f0.β .- c; atol = 1e-10, rtol = 0)
    @test isapprox(fs.σ, f0.σ; rtol = 1e-8)
    # a non-constant p×n offset is the same model as the Normal fit of log(Y) with that offset
    O = 0.4 .* randn(p, n)
    fl = fit_gllvm(Yl; family = GLLVModels.Lognormal(), K = 1, offset = O)
    fn = fit_gllvm(log.(Yl); family = Normal(), K = 1, offset = O)
    @test fl.converged
    @test isapprox(fl.loglik, fn.logLik - sum(log.(Yl)); atol = 1e-4, rtol = 0)
    @test isapprox(fl.σ, fn.pars.σ_eps; rtol = 1e-3)
    @test isapprox(fl.β, fn.pars.β; atol = 1e-3, rtol = 0)
    @test abs(fl.loglik - f0.loglik) > 1            # a non-constant offset is not absorbed
    # the shorter shapes stretch to the same p×n offset
    u = collect(range(0.0, 1.0; length = n))
    fu = fit_gllvm(Yl; family = GLLVModels.Lognormal(), K = 1, offset = reshape(u, 1, n))
    fu2 = fit_gllvm(Yl; family = GLLVModels.Lognormal(), K = 1, offset = repeat(u', p, 1))
    @test isapprox(fu.loglik, fu2.loglik; atol = 1e-10, rtol = 0)
    # the named fitter is correct too, and checks the shape it receives
    @test isapprox(fit_lognormal_gllvm(Yl; K = 1, offset = O).loglik, fl.loglik; atol = 1e-10, rtol = 0)
    @test_throws ArgumentError fit_lognormal_gllvm(Yl; K = 1, offset = fill(c, n))
    @test_throws ArgumentError fit_lognormal_gllvm(Yl; K = 1, offset = fill(NaN, p, n))
end

@testset "offset: a bare vector is ambiguous when p == n" begin
    Random.seed!(2026100202)
    p = n = 6
    Y = Float64.(rand(Poisson(3.0), p, n)); Yn = randn(p, n) .+ 1
    o = collect(range(0.5, 3.0; length = n))        # a per-unit effort vector, the natural use
    for (Yx, fam) in ((Y, Poisson()), (Yn, Normal()))
        err = try fit_gllvm(Yx; family = fam, K = 1, offset = o); nothing catch e e end
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin("reshape(o, 1, $n)", err.msg)
        err isa ArgumentError && @test occursin("reshape(o, $p, 1)", err.msg)
    end
    # the two explicit forms say which was meant: per-trait is absorbed, per-unit is not
    f0 = fit_gllvm(Y; family = Poisson(), K = 1)
    fu = fit_gllvm(Y; family = Poisson(), K = 1, offset = reshape(o, 1, n))
    ft = fit_gllvm(Y; family = Poisson(), K = 1, offset = reshape(o, p, 1))
    @test abs(fu.loglik - f0.loglik) > 0.1
    @test isapprox(ft.loglik, f0.loglik; atol = 1e-3, rtol = 0)
    # a square matrix is read as traits × units (the documented p×n layout), untouched
    M = randn(p, n)
    @test GLLVModels._normalize_offset(M, p, n) === M
    # with p ≠ n a length-p vector is still one offset per trait, and a length-n vector is refused
    @test GLLVModels._normalize_offset(zeros(4), 4, 7) == zeros(4, 7)
    @test_throws ArgumentError GLLVModels._normalize_offset(zeros(7), 4, 7)
    # the refusal text names the size properly
    err = try GLLVModels._normalize_offset(zeros(7), 4, 7); nothing catch e e end
    @test occursin("a length-7 vector", err.msg)
    @test !occursin("7 Array", err.msg)
    err = try GLLVModels._normalize_offset(zeros(7, 4), 4, 7); nothing catch e e end
    @test occursin("a 7×4 matrix", err.msg)
end

@testset "offset: non-finite values are refused at observed cells, allowed where unobserved" begin
    Random.seed!(2026100203)
    d = _off_data(p = 5, n = 30, seed = 2026100203)
    p, n = size(d[:pois]); G = GLLVModels
    O = fill(0.2, p, n)
    cases = [("Poisson", Poisson(), d[:pois]), ("Gamma", Gamma(), d[:gam]),
             ("ZIPoisson", G.ZIPoisson(), d[:zip]), ("ZINegBin", G.ZINegBin(), d[:zip]),
             ("DeltaLogNormal", G.DeltaLogNormal(), d[:two]),
             ("HurdlePoisson", G.HurdlePoisson(), d[:twoc]), ("Normal", Normal(), d[:norm])]
    @testset "$label" for (label, fam, Y) in cases
        for bad in (NaN, Inf, -Inf)
            Ob = copy(O); Ob[2, 3] = bad
            @test_throws ArgumentError fit_gllvm(Y; family = fam, K = 1, offset = Ob)
        end
        v = fill(0.2, p); v[2] = NaN                      # a length-p vector with a NaN
        @test_throws ArgumentError fit_gllvm(Y; family = fam, K = 1, offset = v)
        u = fill(0.2, 1, n); u[1, 4] = NaN                # a 1×n matrix with a NaN
        @test_throws ArgumentError fit_gllvm(Y; family = fam, K = 1, offset = u)
    end
    err = try fit_gllvm(d[:zip]; family = G.ZIPoisson(), K = 1, offset = let o = copy(O); o[2, 3] = NaN; o end); nothing catch e e end
    @test occursin("trait 2, unit 3", err.msg)
    # `missing` inside an offset is refused too, not just NaN
    Om = Matrix{Union{Missing,Float64}}(O); Om[1, 1] = missing
    @test_throws ArgumentError fit_gllvm(d[:zip]; family = G.ZIPoisson(), K = 1, offset = Om)

    # An unobserved cell is never read, so its offset may be NaN: with a mask, or with `missing` in Y.
    mask = trues(p, n); mask[2, 3] = false
    On = copy(O); On[2, 3] = NaN
    fa = fit_gllvm(d[:pois]; family = Poisson(), K = 1, mask = mask, offset = On)
    fb = fit_gllvm(d[:pois]; family = Poisson(), K = 1, mask = mask, offset = O)
    @test fa.converged && isfinite(fa.loglik)
    @test isapprox(fa.loglik, fb.loglik; atol = 1e-8, rtol = 0)
    Ym = Matrix{Union{Missing,Float64}}(d[:pois]); Ym[2, 3] = missing
    fm = fit_gllvm(Ym; family = Poisson(), K = 1, offset = On)
    @test fm.converged && isapprox(fm.loglik, fa.loglik; atol = 1e-8, rtol = 0)
    fg = fit_gllvm(d[:norm]; family = Normal(), K = 1, mask = mask, offset = On)
    @test fg.converged && isfinite(fg.logLik)
    # ... but a NaN at any other, observed cell is still refused
    On2 = copy(On); On2[1, 1] = NaN
    @test_throws ArgumentError fit_gllvm(d[:pois]; family = Poisson(), K = 1, mask = mask, offset = On2)
    # a whole trait masked out may carry a NaN offset (length-p vector, stretched along units)
    mask2 = trues(p, n); mask2[2, :] .= false
    vN = fill(0.2, p); vN[2] = NaN
    @test GLLVModels._normalize_offset(vN, p, n; mask = mask2) isa Matrix
    @test_throws ArgumentError GLLVModels._normalize_offset(vN, p, n; mask = mask)
end

@testset "offset: the @formula front end applies the same rules" begin
    G = GLLVModels
    d = _off_data(p = 5, n = 30, seed = 2026100204)
    p, n = size(d[:pois]); c = log(2)
    site = (x = randn(n),)
    @testset "intercept-only ZIPoisson / ZINegBin: a scalar broadcasts (was -Inf, unconverged)" for fam in (G.ZIPoisson(), G.ZINegBin())
        f0 = gllvm(@formula(y ~ 1), d[:zip], site; family = fam, K = 1)
        fs = gllvm(@formula(y ~ 1), d[:zip], site; family = fam, K = 1, offset = c)
        fm = gllvm(@formula(y ~ 1), d[:zip], site; family = fam, K = 1, offset = fill(c, p, n))
        @test fs.converged && isfinite(fs.loglik)
        @test isapprox(fs.loglik, fm.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(fs.loglik, f0.loglik; atol = 1e-6, rtol = 0)
        @test_throws ArgumentError gllvm(@formula(y ~ 1), d[:zip], site; family = fam, K = 1, offset = fill(c, n))
        @test_throws ArgumentError gllvm(@formula(y ~ 1), d[:zip], site; family = fam, K = 1,
                                         offset = let o = fill(c, p, n); o[1, 1] = NaN; o end)
    end
    @testset "intercept-only Normal: scalar and 1×n stretch (was a DimensionMismatch)" begin
        f0 = gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(), K = 1)
        fs = gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(), K = 1, offset = c)
        @test isapprox(fs.logLik, f0.logLik; atol = 1e-6, rtol = 0)
        u = collect(range(0.0, 1.0; length = n))
        f1 = gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(), K = 1, offset = reshape(u, 1, n))
        f2 = gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(), K = 1, offset = repeat(u', p, 1))
        @test isapprox(f1.logLik, f2.logLik; atol = 1e-8, rtol = 0)
        # y ~ 0 (zero mean) goes through the plain Gaussian fitter and takes the same shapes
        @test isfinite(gllvm(@formula(y ~ 0), d[:norm], site; family = Normal(), K = 1, offset = c).logLik)
    end
    @testset "Normal with a covariate takes the offset; covariate routes without one refuse" begin
        f0 = gllvm(@formula(y ~ 1 + x), d[:norm], site; family = Normal(), K = 1)
        fs = gllvm(@formula(y ~ 1 + x), d[:norm], site; family = Normal(), K = 1, offset = c)
        @test isapprox(fs.logLik, f0.logLik; atol = 1e-6, rtol = 0)
        for (fam, Y) in ((Poisson(), d[:pois]), (NegativeBinomial(), d[:nb]), (G.ZIPoisson(), d[:zip]))
            err = try gllvm(@formula(y ~ 1 + x), Y, site; family = fam, K = 1, offset = c); nothing catch e e end
            @test err isa ArgumentError
            err isa ArgumentError && @test occursin("offset is not supported", err.msg)
        end
    end
    @testset "other formula routes: refuse an offset they cannot use, clearly" begin
        # pervar and the explicit-grouping route go through fit_gllvm
        @test_throws ArgumentError gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(), K = 1, pervar = true, offset = c)
        err = try gllvm(@formula(y ~ 1), d[:norm], site; family = Normal(),
                        sources = [], offset = c); nothing catch e e end
        @test err isa ArgumentError
        # a family fit_gllvm refuses is refused at the formula too
        err = try gllvm(@formula(y ~ 1), d[:norm], site; family = G.StudentTFamily(5.0), K = 1, offset = c); nothing catch e e end
        @test err isa ArgumentError && occursin("offset is not supported for family StudentTFamily", err.msg)
    end
end

# ---- Round-2 review fixes (#693): direct named-fitter calls ---------------------------
# `fit_gllvm` normalises an offset before it reaches a fitter, but the named fitters are
# public too. Each now runs the same normaliser first thing, so a direct call takes the same
# shapes and raises the same ArgumentError. Before, a length-p vector handed straight to a
# two-part fitter was read out of bounds by the warm start (a throw, a -Inf, or a wrong
# converged fit, depending on what was in memory).
@testset "offset: named two-part fitters take the shapes fit_gllvm takes" begin
    G = GLLVModels
    d = _off_data()
    p, n = size(d[:pois]); c = log(2)
    v = [0.7, -0.4, 1.1, 0.2, -0.9]                     # one offset per trait
    u = collect(range(0.0, 1.0; length = n))             # one offset per unit
    named = [
        ("fit_zip_gllvm",              (; kw...) -> G.fit_zip_gllvm(d[:zip]; K = 1, kw...)),
        ("fit_zinb_gllvm",             (; kw...) -> G.fit_zinb_gllvm(d[:zinb]; K = 1, kw...)),
        ("fit_zib_gllvm",              (; kw...) -> G.fit_zib_gllvm(d[:zib]; K = 1, N = 8, kw...)),
        ("fit_hurdle_poisson_gllvm",   (; kw...) -> G.fit_hurdle_poisson_gllvm(d[:twoc]; K = 1, kw...)),
        ("fit_hurdle_nb_gllvm",        (; kw...) -> G.fit_hurdle_nb_gllvm(d[:twonb]; K = 1, kw...)),
        ("fit_delta_lognormal_gllvm",  (; kw...) -> G.fit_delta_lognormal_gllvm(d[:two]; K = 1, kw...)),
        ("fit_delta_gamma_gllvm",      (; kw...) -> G.fit_delta_gamma_gllvm(d[:two]; K = 1, kw...)),
    ]
    @testset "$label" for (label, call) in named
        f0 = call()
        # a length-p vector is one offset per trait: identical to the explicit p×n matrix,
        # and (per-trait constants being absorbed by the intercepts) the same maximum with
        # the count-part intercepts shifted by -v
        fv = call(; offset = v)
        fm = call(; offset = repeat(v, 1, n))
        @test fv.converged && isfinite(fv.loglik)
        @test fv.loglik == fm.loglik
        @test fv.βc == fm.βc
        @test isapprox(fv.loglik, f0.loglik; atol = 1e-3, rtol = 0)
        @test isapprox(fv.βc, f0.βc .- v; atol = 1e-3, rtol = 0)
        # the other shapes fit_gllvm accepts
        @test call(; offset = reshape(v, p, 1)).loglik == fm.loglik                   # p×1
        fu = call(; offset = reshape(u, 1, n))                                         # 1×n
        @test fu.loglik == call(; offset = repeat(u', p, 1)).loglik
        @test abs(fu.loglik - f0.loglik) > 0.1                                         # applied, not absorbed
        # a scalar is the explicit matrix (it used to read out of bounds or return -Inf)
        fs = call(; offset = c)
        @test fs.loglik == call(; offset = fill(c, p, n)).loglik
        @test isapprox(fs.βc, f0.βc .- c; atol = 1e-3, rtol = 0)
        # the same call with no offset is untouched by any of this
        @test call(; offset = nothing).loglik == f0.loglik
    end
    # the helper itself refuses a shape it cannot index, instead of reading past the array
    @testset "the warm-start helper checks the shape it is given" begin
        Y = d[:zip]
        βc0 = zeros(p); Zc = zeros(p, n)
        @test_throws ArgumentError G._tp_offset_warmstart!(βc0, Zc, Y, v)
        @test_throws ArgumentError G._tp_offset_warmstart!(βc0, Zc, Y, c)
        @test_throws ArgumentError G._tp_offset_warmstart!(βc0, Zc, Y, zeros(p, n + 1))
        @test_throws ArgumentError G._zi_warmstart(Y, 1; offset = v)
        @test_throws ArgumentError G._zib_warmstart(Y, 8, 1; offset = v)
        @test G._tp_offset_warmstart!(βc0, Zc, Y, nothing) === βc0
    end
end

@testset "offset: every named fitter refuses a shape it cannot broadcast, with an ArgumentError" begin
    G = GLLVModels
    d = _off_data()
    p, n = size(d[:pois])
    named = [
        ("fit_gaussian_gllvm",                (; kw...) -> G.fit_gaussian_gllvm(d[:norm]; K = 1, kw...)),
        ("fit_gaussian_gllvm",                (; kw...) -> G.fit_gaussian_gllvm(d[:norm]; K = 1, aghq = 3, kw...)),
        ("fit_poisson_gllvm",                 (; kw...) -> G.fit_poisson_gllvm(d[:pois]; K = 1, kw...)),
        ("fit_poisson_gllvm",                 (; kw...) -> G.fit_poisson_gllvm(d[:pois]; K = 1, aghq = 3, kw...)),
        ("fit_binomial_gllvm",                (; kw...) -> G.fit_binomial_gllvm(d[:bin]; K = 1, N = d[:N], kw...)),
        ("fit_binomial_gllvm",                (; kw...) -> G.fit_binomial_gllvm(d[:bin]; K = 1, N = d[:N], aghq = 3, kw...)),
        ("fit_nb_gllvm",                      (; kw...) -> G.fit_nb_gllvm(d[:nb]; K = 1, kw...)),
        ("fit_nb1_gllvm",                     (; kw...) -> G.fit_nb1_gllvm(d[:nb]; K = 1, kw...)),
        ("fit_beta_gllvm",                    (; kw...) -> G.fit_beta_gllvm(d[:beta]; K = 1, kw...)),
        ("fit_gamma_gllvm",                   (; kw...) -> G.fit_gamma_gllvm(d[:gam]; K = 1, kw...)),
        ("fit_exponential_gllvm",             (; kw...) -> G.fit_exponential_gllvm(d[:expo]; K = 1, kw...)),
        ("fit_gp1_gllvm",                     (; kw...) -> G.fit_gp1_gllvm(d[:pois]; K = 1, kw...)),
        ("fit_censored_poisson_gllvm",        (; kw...) -> G.fit_censored_poisson_gllvm(d[:pois]; K = 1, kw...)),
        ("fit_truncated_poisson_gllvm",       (; kw...) -> G.fit_truncated_poisson_gllvm(d[:tpois]; K = 1, kw...)),
        ("fit_truncated_nbinom2_gllvm",       (; kw...) -> G.fit_truncated_nbinom2_gllvm(d[:tpois]; K = 1, kw...)),
        ("fit_truncated_nbinom2_gllvm_pertrait", (; kw...) -> G.fit_truncated_nbinom2_gllvm_pertrait(d[:tpois]; K = 1, kw...)),
        ("fit_nb_gllvm_grouped",              (; kw...) -> G.fit_nb_gllvm_grouped(d[:nb]; K = 1, group = collect(1:p), kw...)),
        ("fit_nb_gllvm_grouped_aghq",         (; kw...) -> G.fit_nb_gllvm_grouped_aghq(d[:nb]; K = 1, group = collect(1:p), aghq = 3, kw...)),
        ("fit_beta_gllvm_grouped",            (; kw...) -> G.fit_beta_gllvm_grouped(d[:beta]; K = 1, kw...)),
        ("fit_gamma_gllvm_grouped",           (; kw...) -> G.fit_gamma_gllvm_grouped(d[:gam]; K = 1, kw...)),
        ("fit_nb1_gllvm_grouped",             (; kw...) -> G.fit_nb1_gllvm_grouped(d[:nb]; K = 1, kw...)),
        ("fit_tweedie_gllvm_grouped",         (; kw...) -> G.fit_tweedie_gllvm_grouped(d[:tw]; K = 1, power = 1.5, kw...)),
        ("fit_tweedie_gllvm_grouped_aghq",    (; kw...) -> G.fit_tweedie_gllvm_grouped_aghq(d[:tw]; K = 1, power = 1.5, aghq = 3, kw...)),
        ("fit_lognormal_gllvm",               (; kw...) -> G.fit_lognormal_gllvm(d[:logn]; K = 1, kw...)),
        ("fit_delta_gamma_gllvm_aghq",        (; kw...) -> G.fit_delta_gamma_gllvm_aghq(d[:two]; K = 1, aghq = 3, kw...)),
        ("fit_zip_gllvm",                     (; kw...) -> G.fit_zip_gllvm(d[:zip]; K = 1, kw...)),
        ("fit_zinb_gllvm",                    (; kw...) -> G.fit_zinb_gllvm(d[:zinb]; K = 1, kw...)),
        ("fit_zib_gllvm",                     (; kw...) -> G.fit_zib_gllvm(d[:zib]; K = 1, N = 8, kw...)),
        ("fit_hurdle_poisson_gllvm",          (; kw...) -> G.fit_hurdle_poisson_gllvm(d[:twoc]; K = 1, kw...)),
        ("fit_hurdle_nb_gllvm",               (; kw...) -> G.fit_hurdle_nb_gllvm(d[:twonb]; K = 1, kw...)),
        ("fit_delta_lognormal_gllvm",         (; kw...) -> G.fit_delta_lognormal_gllvm(d[:two]; K = 1, kw...)),
        ("fit_delta_gamma_gllvm",             (; kw...) -> G.fit_delta_gamma_gllvm(d[:two]; K = 1, kw...)),
    ]
    bad = [
        "length-n vector"     => zeros(n),
        "wrong-length vector" => zeros(p - 1),
        "p×(n-1) matrix"      => zeros(p, n - 1),
        "(p+1)×n matrix"      => zeros(p + 1, n),
        "NaN matrix"          => fill(NaN, p, n),
        "string"              => "log(2)",
    ]
    @testset "$label" for (label, call) in named
        for (blabel, off) in bad
            err = try call(; offset = off); nothing catch e e end
            @test err isa ArgumentError
            err isa ArgumentError && @test occursin(label, err.msg)    # names the fitter that was called
        end
    end
end

@testset "offset: `missing` at an unobserved cell is read as NaN, on every route that takes a mask" begin
    G = GLLVModels
    d = _off_data(p = 5, n = 30, seed = 2026100205)
    p, n = size(d[:pois])
    mask = trues(p, n); mask[2, 3] = false
    O = fill(0.2, p, n)
    On = copy(O); On[2, 3] = NaN
    Om = Matrix{Union{Missing,Float64}}(O); Om[2, 3] = missing
    # every route below honours `mask`; the offset at the masked cell is never read, so a
    # finite value, NaN and `missing` all give the very same fit (Poisson and Gamma used to
    # throw a MethodError on `missing`)
    cases = [
        ("Normal",           d[:norm],  Normal(),                      (;)),
        ("Normal aghq",      d[:norm],  Normal(),                      (; aghq = 3)),
        ("Poisson",          d[:pois],  Poisson(),                     (;)),
        ("Poisson aghq",     d[:pois],  Poisson(),                     (; aghq = 3)),
        ("Binomial",         d[:bin],   Binomial(),                    (; N = d[:N])),
        ("TruncatedPoisson", d[:tpois], G.TruncatedPoisson(),          (;)),
        ("CensoredPoisson",  d[:pois],  G.CensoredPoisson(),           (;)),
        ("TruncatedNegBin2", d[:tpois], G.TruncatedNegBin2(),          (;)),
        ("NegBin2",          d[:nb],    NegativeBinomial(),            (;)),
        ("NegBin2 species",  d[:nb],    NegativeBinomial(),            (; disp_group = :species)),
        ("NegBin2 aghq",     d[:nb],    NegativeBinomial(),            (; disp_group = :species, aghq = 3)),
        ("Beta",             d[:beta],  Beta(),                        (;)),
        ("NB1",              d[:nb],    G.NB1(),                       (;)),
        ("Gamma",            d[:gam],   Gamma(),                       (;)),
        ("Exponential",      d[:expo],  G.Exponential(),               (;)),
        ("GeneralizedPoisson1", d[:pois], G.GeneralizedPoisson1(0.1),  (;)),
    ]
    @testset "$label" for (label, Y, fam, kw) in cases
        ff = fit_gllvm(Y; family = fam, K = 1, mask = mask, kw..., offset = O)
        fn = fit_gllvm(Y; family = fam, K = 1, mask = mask, kw..., offset = On)
        fm = fit_gllvm(Y; family = fam, K = 1, mask = mask, kw..., offset = Om)
        @test isfinite(_off_ll(fm))
        @test _off_ll(fm) == _off_ll(fn)
        @test isapprox(_off_ll(fm), _off_ll(ff); atol = 1e-8, rtol = 0)
    end
    # a named fitter called directly takes it too, and `missing` in Y marks the cell as well
    @test G.fit_poisson_gllvm(d[:pois]; K = 1, mask = mask, offset = Om).loglik ==
          G.fit_poisson_gllvm(d[:pois]; K = 1, mask = mask, offset = On).loglik
    @test isfinite(G.fit_gamma_gllvm(d[:gam]; K = 1, mask = mask, offset = Om).loglik)
    Ym = Matrix{Union{Missing,Float64}}(d[:pois]); Ym[2, 3] = missing
    @test fit_gllvm(Ym; family = Poisson(), K = 1, offset = Om).converged
    # `missing` at an observed cell is still refused, on a named fitter too
    Om2 = copy(Om); Om2[1, 1] = missing
    @test_throws ArgumentError fit_gllvm(d[:pois]; family = Poisson(), K = 1, mask = mask, offset = Om2)
    @test_throws ArgumentError G.fit_poisson_gllvm(d[:pois]; K = 1, mask = mask, offset = Om2)
    @test_throws ArgumentError G.fit_gamma_gllvm(d[:gam]; K = 1, mask = mask, offset = Om2)
    # routes with no mask cannot have an unobserved cell, so `missing` / NaN anywhere is refused
    for (fam, Y) in ((G.ZIPoisson(), d[:zip]), (G.DeltaLogNormal(), d[:two]), (G.Lognormal(), d[:logn]))
        @test_throws ArgumentError fit_gllvm(Y; family = fam, K = 1, offset = Om)
        @test_throws ArgumentError fit_gllvm(Y; family = fam, K = 1, offset = On)
    end
    # Lognormal has no mask: a NaN offset names the cell and says so (even when a mask marks
    # that cell unobserved), and a mask is refused outright (before this, a mask was accepted
    # and gave wrong intercepts and logLik)
    for call in (() -> fit_gllvm(d[:logn]; family = G.Lognormal(), K = 1, offset = On),
                 () -> fit_gllvm(d[:logn]; family = G.Lognormal(), K = 1, mask = mask, offset = On),
                 () -> fit_lognormal_gllvm(d[:logn]; K = 1, offset = On))
        err = try call(); nothing catch e e end
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin("trait 2, unit 3", err.msg)
        err isa ArgumentError && @test occursin("does not support mask", err.msg)
        err isa ArgumentError && @test !occursin("mark it unobserved", err.msg)
    end
    err = try fit_lognormal_gllvm(d[:logn]; K = 1, offset = On); nothing catch e e end
    @test occursin("fit_lognormal_gllvm", err.msg)
    err = try fit_gllvm(d[:logn]; family = G.Lognormal(), K = 1, mask = mask); nothing catch e e end
    @test err isa ArgumentError && occursin("mask is not supported", err.msg)
    err = try fit_lognormal_gllvm(d[:logn]; K = 1, mask = mask, offset = O); nothing catch e e end
    @test err isa ArgumentError && occursin("mask is not supported", err.msg)
    # the two-part and zero-inflated routes have no mask either: same wording
    for (fam, Y) in ((G.ZIPoisson(), d[:zip]), (G.DeltaGamma(), d[:two]), (G.HurdlePoisson(), d[:twoc]))
        err = try fit_gllvm(Y; family = fam, K = 1, offset = On); nothing catch e e end
        @test err isa ArgumentError && occursin("does not support mask", err.msg)
    end
    # a route that does take a mask keeps the hint to mark the cell unobserved
    err = try fit_gllvm(d[:pois]; family = Poisson(), K = 1, offset = On); nothing catch e e end
    @test err isa ArgumentError && occursin("mark it unobserved", err.msg)
    @test isfinite(fit_gllvm(d[:logn]; family = G.Lognormal(), K = 1, offset = O).loglik)
end
