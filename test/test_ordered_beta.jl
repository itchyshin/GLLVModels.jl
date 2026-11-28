using GLLVModels, Test, Random, Distributions, Statistics

# Hand reference for the three-branch ordered-beta log-density log p(y|η).
function _ref_ob_logp(y, η, c0, c1, φ)
    σ(x) = 1 / (1 + exp(-x))
    if y == 0
        return log(σ(c0 - η))
    elseif y == 1
        return log(σ(η - c1))
    else
        μ = σ(η)
        return log(σ(η - c0) - σ(η - c1)) + logpdf(Beta(μ * φ, (1 - μ) * φ), y)
    end
end

@testset "Ordered-beta family" begin
    @testset "Λ = 0 reduces to independent ordered-beta logp (exact)" begin
        Random.seed!(2023)
        p, K, n = 5, 2, 30
        β  = [-0.5, 0.2, 1.0, -1.2, 0.0]
        c0, c1, φ = -0.8, 0.9, 9.0
        # Build Y in [0,1] with some exact 0s, exact 1s, and interior Beta draws.
        Y = Matrix{Float64}(undef, p, n)
        for t in 1:p, i in 1:n
            r = (t + i) % 7
            if r == 0
                Y[t, i] = 0.0
            elseif r == 1
                Y[t, i] = 1.0
            else
                μ = 1 / (1 + exp(-β[t]))
                Y[t, i] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
            end
        end

        ll = GLLVModels.ordered_beta_marginal_loglik_laplace(Y, zeros(p, K), β, c0, c1, φ)
        ll_indep = sum(_ref_ob_logp(Y[t, i], β[t], c0, c1, φ) for t in 1:p, i in 1:n)
        @test ll ≈ ll_indep atol = 1e-8
    end

    @testset "K = 1 single site ≈ numerical quadrature" begin
        Random.seed!(7)
        p = 6
        β  = 0.3 .* randn(p)
        c0, c1, φ = -1.0, 1.0, 12.0
        Λ  = reshape(0.4 .* randn(p), p, 1)
        ztrue = 0.6
        η  = β .+ Λ[:, 1] .* ztrue
        # Generate a mix of point masses and interior draws.
        y = Vector{Float64}(undef, p)
        for t in 1:p
            if t == 1
                y[t] = 0.0
            elseif t == 2
                y[t] = 1.0
            else
                μ = 1 / (1 + exp(-η[t]))
                y[t] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
            end
        end
        Y = reshape(y, p, 1)

        ll_lap = GLLVModels.ordered_beta_marginal_loglik_laplace(Y, Λ, β, c0, c1, φ)

        zs = range(-8, 8; length = 4001); dz = step(zs)
        marg = 0.0
        for z in zs
            ηz = β .+ Λ[:, 1] .* z
            logp = sum(_ref_ob_logp(y[t], ηz[t], c0, c1, φ) for t in 1:p)
            marg += exp(logp) * pdf(Normal(), z) * dz
        end
        ll_quad = log(marg)
        @test ll_lap ≈ ll_quad atol = 1e-1
    end

    @testset "fit machinery" begin
        Random.seed!(11)
        p, K, n = 4, 1, 60
        βt = [-0.3, 0.5, 0.0, 0.8]
        Λt = reshape([0.6, -0.4, 0.5, 0.3], p, 1)
        c0, c1, φ = -1.0, 1.0, 10.0
        Y = Matrix{Float64}(undef, p, n)
        for i in 1:n
            z = randn()
            for t in 1:p
                η = βt[t] + Λt[t, 1] * z
                u = 1 / (1 + exp(-(η - c0)))
                v = 1 / (1 + exp(-(η - c1)))
                p0 = 1 - u
                p1 = v
                r = rand()
                if r < p0
                    Y[t, i] = 0.0
                elseif r > 1 - p1
                    Y[t, i] = 1.0
                else
                    μ = 1 / (1 + exp(-η))
                    Y[t, i] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
                end
            end
        end

        fit = fit_ordered_beta_gllvm(Y; K = K, iterations = 100)
        @test fit isa OrderedBetaFit
        @test isfinite(fit.loglik)
        @test fit.c0 < fit.c1
        @test fit.φ > 0
        @test size(fit.Λ) == (p, K)
        @test length(fit.β) == p
    end

    @testset "post-fit: getLV/predict" begin
        Random.seed!(12)
        p, K, n = 4, 1, 50
        βt = [-0.3, 0.5, 0.0, 0.8]
        Λt = reshape([0.6, -0.4, 0.5, 0.3], p, 1)
        c0, c1, φ = -1.0, 1.0, 10.0
        Y = Matrix{Float64}(undef, p, n)
        for i in 1:n
            z = randn()
            for t in 1:p
                η = βt[t] + Λt[t, 1] * z
                u = 1 / (1 + exp(-(η - c0)))
                v = 1 / (1 + exp(-(η - c1)))
                p0 = 1 - u
                p1 = v
                r = rand()
                if r < p0
                    Y[t, i] = 0.0
                elseif r > 1 - p1
                    Y[t, i] = 1.0
                else
                    μ = 1 / (1 + exp(-η))
                    Y[t, i] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
                end
            end
        end
        fit = fit_ordered_beta_gllvm(Y; K = K, iterations = 100)

        S = getLV(fit, Y)
        @test size(S) == (n, K)
        @test all(isfinite, S)

        ηhat = predict(fit, Y; type = :link)
        @test size(ηhat) == (p, n)
        μhat = predict(fit, Y; type = :mean)
        @test size(μhat) == (p, n)
        @test all(isfinite, μhat)
        @test all(0 .<= μhat .<= 1)

        ord = ordination(fit, Y)
        @test size(ord.sites) == (n, K)
    end

    @testset "Wald CIs + coef_table" begin
        Random.seed!(909)
        p, K, n = 4, 1, 150
        βt = 0.4 .* randn(p)
        φ  = 8.0
        Λt = 0.3 .* randn(p, K)
        Y = Matrix{Float64}(undef, p, n)
        for t in 1:p, s in 1:n
            r = (t + 3s) % 6
            if r == 0
                Y[t, s] = 0.0
            elseif r == 1
                Y[t, s] = 1.0
            else
                μ = inv(1 + exp(-βt[t]))
                Y[t, s] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
            end
        end
        fit = fit_ordered_beta_gllvm(Y; K = K)

        ci = confint(fit, Y; method = :wald)
        nterm = p + (p * K - div(K * (K - 1), 2)) + 3      # β + Λ + cut0 + cut1 + φ
        @test length(ci.term) == nterm
        @test ci.method == :wald
        @test "cut0" in ci.term && "cut1" in ci.term
        iφ = findfirst(==("phi"), ci.term)
        @test iφ !== nothing && ci.estimate[iφ] > 0       # φ on the positive scale
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end

        ct = coef_table(fit, Y)
        @test ct isa GllvmCoefTable
        @test length(ct.term) == nterm
    end

    @testset "fit_gllvm reaches no-X ordered-beta via the OrderedBeta() marker" begin
        Random.seed!(1608)
        p, K, n = 4, 1, 60
        βt = [-0.3, 0.5, 0.0, 0.8]
        Λt = reshape([0.6, -0.4, 0.5, 0.3], p, 1)
        c0, c1, φ = -1.0, 1.0, 10.0
        Y = Matrix{Float64}(undef, p, n)
        for i in 1:n
            z = randn()
            for t in 1:p
                η = βt[t] + Λt[t, 1] * z
                u = 1 / (1 + exp(-(η - c0)))
                v = 1 / (1 + exp(-(η - c1)))
                p0 = 1 - u
                p1 = v
                r = rand()
                if r < p0
                    Y[t, i] = 0.0
                elseif r > 1 - p1
                    Y[t, i] = 1.0
                else
                    μ = 1 / (1 + exp(-η))
                    Y[t, i] = clamp(rand(Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
                end
            end
        end

        @test OrderedBeta().c0 == -1.0
        @test OrderedBeta().c1 == 1.0
        @test OrderedBeta().φ == 10.0

        fit = fit_gllvm(Y; family = OrderedBeta(), K = K, iterations = 40)
        @test fit isa OrderedBetaFit
        # FIXED 2026-08-26. Root cause was `ordered_beta_logp`'s interior-mass branch
        # computing log(σ(η−c0) − σ(η−c1)) unguarded, which underflows to log(0.0) at
        # large |η|; rewritten via the stable logσ(a) + log1mexp(logσ(b) − logσ(a))
        # identity (ordered_beta.jl:81-96). Was: iterations=0.
        @test fit.converged
        @test isfinite(fit.loglik)
        @test fit.iterations > 0
        direct = fit_ordered_beta_gllvm(Y; K = K, iterations = 40)
        @test fit.loglik ≈ direct.loglik atol = 1e-8
        @test fit.c0 ≈ direct.c0 atol = 1e-8
        @test fit.c1 ≈ direct.c1 atol = 1e-8
        @test fit.φ ≈ direct.φ atol = 1e-8

        # Marker c0, c1, φ are tag payloads — never read, never starting values.
        tagged = fit_gllvm(Y; family = OrderedBeta(0.0, 2.0, 3.0), K = K, iterations = 40)
        @test tagged.loglik ≈ fit.loglik atol = 1e-8
        @test tagged.c0 ≈ fit.c0 atol = 1e-8
        @test tagged.c1 ≈ fit.c1 atol = 1e-8
        @test tagged.φ ≈ fit.φ atol = 1e-8

        ff = gllvm(@formula(y ~ 1), Y, (; temp = randn(n));
                   family = OrderedBeta(), K = K, iterations = 40)
        @test ff isa OrderedBetaFit
        @test ff.loglik ≈ direct.loglik atol = 1e-8
    end

    @testset "constant interior guard (#1583)" begin
        Y_const = fill(0.5, 3, 12)
        err = @test_throws ArgumentError fit_ordered_beta_gllvm(Y_const; K = 1, iterations = 80)
        msg = sprint(showerror, err.value)
        @test occursin("interior", lowercase(msg))
        @test occursin("identical", msg)
        @test_throws ArgumentError fit_gllvm(Y_const; family = OrderedBeta(), K = 1, iterations = 80)

        Y_varied = [0.3 0.5; 0.4 0.6]
        @test !GLLVModels._ordered_beta_constant_interior(Y_varied, nothing)
        @test GLLVModels._ordered_beta_convergence_veto(Y_varied, nothing, true)

        @test GLLVModels._ordered_beta_constant_interior(Y_const, nothing)
        @test !GLLVModels._ordered_beta_convergence_veto(Y_const, nothing, true)
        @test !GLLVModels._ordered_beta_convergence_veto(Y_const, nothing, false)

        Y_mass = [0.0 1.0; 0.0 1.0]
        @test !GLLVModels._ordered_beta_constant_interior(Y_mass, nothing)
    end
end
