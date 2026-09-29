using GLLVModels, Test, LinearAlgebra, Random, Distributions

@testset "Missing data (NA) handling" begin
    Random.seed!(2024)
    p, n, K = 5, 8, 2
    β = randn(p) .* 0.3
    Λ = randn(p, K) .* 0.4
    Y = rand(0:6, p, n)

    mask = trues(p, n)
    missing_cells = [(1, 2), (3, 5), (4, 8), (2, 1), (5, 3)]
    for (t, s) in missing_cells
        mask[t, s] = false
    end

    # ---- Anchor 1: all-observed mask == no mask (backward compat) ----------
    ℓ_full = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β, LogLink())
    ℓ_truemask = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), Λ, β,
                                               LogLink(); mask = trues(p, n))
    @test isapprox(ℓ_full, ℓ_truemask; atol = 1e-10)

    # ---- Anchor 2: marginal is invariant to the values in masked cells -----
    # The defining property of NA handling: garbage in the unobserved cells must
    # not change the log-likelihood.
    Yg = copy(Y)
    for (t, s) in missing_cells
        Yg[t, s] = 9999
    end
    ℓ_a = GLLVModels.marginal_loglik_laplace(Poisson(), Y,  ones(Int, p, n), Λ, β, LogLink(); mask = mask)
    ℓ_b = GLLVModels.marginal_loglik_laplace(Poisson(), Yg, ones(Int, p, n), Λ, β, LogLink(); mask = mask)
    @test isapprox(ℓ_a, ℓ_b; atol = 1e-10)
    @test ℓ_a != ℓ_full             # masking changes the marginal (sanity: mask is active)

    # ---- Anchor 3: a fully-masked site contributes exactly 0 ---------------
    maskcol = trues(p, 2)
    maskcol[:, 2] .= false          # second site entirely unobserved
    Y2 = Y[:, 1:2]
    ℓ_2 = GLLVModels.marginal_loglik_laplace(Poisson(), Y2, ones(Int, p, 2), Λ, β,
                                        LogLink(); mask = maskcol)
    ℓ_1 = GLLVModels.laplace_loglik_site(Poisson(), view(Y2, :, 1), ones(Int, p), Λ, β, LogLink())
    @test isapprox(ℓ_2, ℓ_1; atol = 1e-10)   # masked site adds 0 ⇒ equals the lone observed site

    # ---- observed_mask derives the mask from `missing` ---------------------
    Ym = Matrix{Union{Missing, Int}}(Y)
    for (t, s) in missing_cells
        Ym[t, s] = missing
    end
    @test GLLVModels.observed_mask(Ym) == mask

    # ---- Anchor 4: the FIT is invariant to masked-cell values --------------
    fitA = fit_poisson_gllvm(Y;  K = K, mask = mask, iterations = 40)
    fitB = fit_poisson_gllvm(Yg; K = K, mask = mask, iterations = 40)
    @test isapprox(fitA.loglik, fitB.loglik; atol = 1e-8)
    @test isapprox(fitA.β, fitB.β; atol = 1e-7)
    @test isapprox(vec(fitA.Λ), vec(fitB.Λ); atol = 1e-7)

    # ---- Fit straight from a matrix containing `missing` -------------------
    fitM = fit_poisson_gllvm(Ym; K = K, iterations = 40)
    @test isfinite(fitM.loglik)
    @test isapprox(fitM.loglik, fitA.loglik; atol = 1e-8)

    # ---- NB fitter: same NA invariance (the masked cells are ignored) ------
    nbA = fit_nb_gllvm(Y;  K = K, mask = mask, iterations = 40)
    nbB = fit_nb_gllvm(Yg; K = K, mask = mask, iterations = 40)
    @test isapprox(nbA.loglik, nbB.loglik; atol = 1e-8)
    @test isapprox(nbA.r, nbB.r; atol = 1e-6)

    # ---- Beta / Gamma / Binomial fitters honour NA ------------------------
    # Fitting with an explicit mask (original values in the masked cells) must
    # equal fitting the same data with those cells set to `missing` — both ignore
    # the masked cells (objective + mask-respecting warm start), so identical fit.
    @testset "continuous & binomial fitters honour the mask" begin
        Random.seed!(99)
        pp, nn = 4, 10
        msk2 = trues(pp, nn)
        for (t, s) in [(1, 3), (2, 7), (4, 1), (3, 9)]
            msk2[t, s] = false
        end
        miss = findall(.!msk2)

        # Beta (proportions in (0,1))
        Yb  = clamp.(rand(pp, nn), 0.02, 0.98)
        Ybm = Matrix{Union{Missing, Float64}}(Yb); for I in miss; Ybm[I] = missing; end
        fb1 = fit_beta_gllvm(Yb;  K = 1, mask = msk2, iterations = 30)
        fb2 = fit_beta_gllvm(Ybm; K = 1, iterations = 30)
        @test isapprox(fb1.loglik, fb2.loglik; atol = 1e-7)

        # Gamma (positive)
        Yg2 = 0.5 .+ 2 .* rand(pp, nn)
        Ygm = Matrix{Union{Missing, Float64}}(Yg2); for I in miss; Ygm[I] = missing; end
        fg1 = fit_gamma_gllvm(Yg2; K = 1, mask = msk2, iterations = 30)
        fg2 = fit_gamma_gllvm(Ygm; K = 1, iterations = 30)
        @test isapprox(fg1.loglik, fg2.loglik; atol = 1e-7)

        # Binomial (counts out of N trials)
        Ntr = fill(5, pp, nn)
        Yco = rand(0:5, pp, nn)
        Ycm = Matrix{Union{Missing, Int}}(Yco); for I in miss; Ycm[I] = missing; end
        fc1 = fit_binomial_gllvm(Yco; K = 1, N = Ntr, mask = msk2, iterations = 30)
        fc2 = fit_binomial_gllvm(Ycm; K = 1, N = Ntr, iterations = 30)
        @test isapprox(fc1.loglik, fc2.loglik; atol = 1e-7)

        # Exponential (positive, no dispersion)
        Ye  = 0.2 .+ randexp(pp, nn)
        Yem = Matrix{Union{Missing, Float64}}(Ye); for I in miss; Yem[I] = missing; end
        fe1 = fit_exponential_gllvm(Ye;  K = 1, mask = msk2, iterations = 30)
        fe2 = fit_exponential_gllvm(Yem; K = 1, iterations = 30)
        @test isapprox(fe1.loglik, fe2.loglik; atol = 1e-7)
    end

    # ---- Exponential: complete-data equivalence + NA invariance ------------
    @testset "Exponential NA-FIML" begin
        Random.seed!(321)
        pe, ne = 5, 9
        Λe = randn(pe, 1) .* 0.4
        βe = randn(pe) .* 0.3
        Ye = 0.1 .+ randexp(pe, ne)

        # (1) complete-data equivalence: no mask == an all-true mask (marginal).
        ℓ_full = GLLVModels.exponential_marginal_loglik_laplace(Ye, Λe, βe)
        ℓ_mask = GLLVModels.exponential_marginal_loglik_laplace(Ye, Λe, βe; mask = trues(pe, ne))
        @test isapprox(ℓ_full, ℓ_mask; atol = 1e-10)

        # complete-data equivalence at the fit level: default call == all-true mask.
        feA = fit_exponential_gllvm(Ye; K = 1, iterations = 40)
        feB = fit_exponential_gllvm(Ye; K = 1, mask = trues(pe, ne), iterations = 40)
        @test isapprox(feA.loglik, feB.loglik; atol = 1e-10)
        @test isapprox(feA.β, feB.β; atol = 1e-10)
        @test isapprox(vec(feA.Λ), vec(feB.Λ); atol = 1e-10)

        # (2) a finite NA fit: a few masked cells, marginal invariant to garbage.
        mske = trues(pe, ne)
        for (t, s) in [(1, 4), (3, 2), (5, 7), (2, 9)]
            mske[t, s] = false
        end
        Yeg = copy(Ye); for I in findall(.!mske); Yeg[I] = 9999.0; end
        ℓ_a = GLLVModels.exponential_marginal_loglik_laplace(Ye,  Λe, βe; mask = mske)
        ℓ_b = GLLVModels.exponential_marginal_loglik_laplace(Yeg, Λe, βe; mask = mske)
        @test isapprox(ℓ_a, ℓ_b; atol = 1e-10)
        @test ℓ_a != ℓ_full

        feM = fit_exponential_gllvm(Ye; K = 1, mask = mske, iterations = 40)
        @test isfinite(feM.loglik)
        feG = fit_exponential_gllvm(Yeg; K = 1, mask = mske, iterations = 40)
        @test isapprox(feM.loglik, feG.loglik; atol = 1e-8)
        @test isapprox(feM.β, feG.β; atol = 1e-7)
        @test isapprox(vec(feM.Λ), vec(feG.Λ); atol = 1e-7)
    end

    # ---- Ordinal: NA-FIML mask honoured (completes the response-family grid) ----
    @testset "Ordinal NA-FIML" begin
        Random.seed!(20260613)
        po, no, Ko = 6, 40, 2
        Λo = randn(po, Ko) .* 0.4
        for a in 1:Ko, b in 1:Ko
            a < b && (Λo[a, b] = 0.0)
        end
        τo = [-1.2, 0.0, 1.2]
        Mo = Λo * randn(Ko, no)
        Fo(x) = 1 / (1 + exp(-x))
        drw(η, τ, u) = (for c in 1:length(τ); u <= Fo(τ[c] - η) && return c; end; length(τ) + 1)
        Yo = [drw(Mo[t, s], τo, rand()) for t in 1:po, s in 1:no]
        Λe = randn(po, Ko) .* 0.3
        τe = [-1.0, 0.0, 1.0]

        misso = [(1, 2), (3, 5), (4, 8), (2, 1), (5, 3)]
        msko = trues(po, no); for (t, s) in misso; msko[t, s] = false; end
        Yog = copy(Yo); for (t, s) in misso; Yog[t, s] = 9999; end

        # complete-data equivalence: no mask == an all-true mask (marginal)
        @test isapprox(GLLVModels.ordinal_marginal_loglik_laplace(Yo, Λe, τe),
                       GLLVModels.ordinal_marginal_loglik_laplace(Yo, Λe, τe; mask = trues(po, no));
                       atol = 1e-10)
        # marginal invariant to garbage in masked cells; mask is active
        ℓo = GLLVModels.ordinal_marginal_loglik_laplace(Yo, Λe, τe; mask = msko)
        @test isapprox(ℓo, GLLVModels.ordinal_marginal_loglik_laplace(Yog, Λe, τe; mask = msko); atol = 1e-10)
        @test ℓo != GLLVModels.ordinal_marginal_loglik_laplace(Yo, Λe, τe)
        # a fully-masked site contributes exactly 0
        mc = trues(po, 2); mc[:, 2] .= false
        @test isapprox(GLLVModels.ordinal_marginal_loglik_laplace(Yo[:, 1:2], Λe, τe; mask = mc),
                       GLLVModels.ordinal_loglik_site(view(Yo, :, 1), Λe, τe, GLLVModels.LogitLink());
                       atol = 1e-10)
        # the FIT is invariant to masked-cell values (sentinel-invariance)
        foA = fit_ordinal_gllvm(Yo;  K = Ko, mask = msko, iterations = 80)
        foB = fit_ordinal_gllvm(Yog; K = Ko, mask = msko, iterations = 80)
        @test foA.converged
        @test isapprox(foA.loglik, foB.loglik; atol = 1e-7)
        @test isapprox(vec(foA.Λ), vec(foB.Λ); atol = 1e-6)
        @test isapprox(foA.τ, foB.τ; atol = 1e-6)
    end
end
