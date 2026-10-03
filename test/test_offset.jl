using GLLVModels, Test, Random, Distributions

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
