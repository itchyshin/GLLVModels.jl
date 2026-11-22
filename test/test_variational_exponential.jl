using GLLVModels, Test, Random, Distributions, Statistics

@testset "Variational (VA) marginal — Exponential" begin
    @testset "Λ=0 reduces to independent Exponential loglik (exact)" begin
        Random.seed!(320)
        p, K, n = 6, 2, 40
        β = 0.3 .* randn(p) .+ 0.5
        Y = [rand(Exponential(exp(β[t]))) for t in 1:p, s in 1:n]
        va = GLLVModels.exponential_marginal_loglik_va(Y, zeros(p, K), β)
        ref = 0.0
        for t in 1:p, s in 1:n
            ref += logpdf(Exponential(exp(β[t])), Y[t, s])
        end
        @test va ≈ ref atol = 1e-8
    end

    @testset "ELBO is a lower bound on the exact marginal, and tight (K=1)" begin
        Random.seed!(321)
        p = 6
        β = 0.3 .* randn(p) .+ 0.5
        Λ = reshape(0.4 .* randn(p), p, 1)
        ztrue = randn()
        y = [rand(Exponential(exp(β[t] + Λ[t, 1] * ztrue))) for t in 1:p]
        Y = reshape(y, p, 1)
        va = GLLVModels.exponential_marginal_loglik_va(Y, Λ, β)

        zs = range(-10, 10; length = 8001); dz = step(zs)
        marg = 0.0
        for z in zs
            lp = 0.0
            for t in 1:p
                μ = exp(β[t] + Λ[t, 1] * z)
                lp += logpdf(Exponential(μ), y[t])
            end
            marg += exp(lp) * pdf(Normal(), z) * dz
        end
        quad = log(marg)

        @test va ≤ quad + 1e-4                  # ELBO ≤ log-marginal (Jensen / KL ≥ 0)
        @test isapprox(va, quad; atol = 0.3)    # and tight
    end

    @testset "analytic inner gradient matches central finite difference" begin
        Random.seed!(322)
        p, K = 5, 2
        β = 0.3 .* randn(p) .+ 0.5
        Λ = 0.4 .* randn(p, K)
        Λ2 = Λ .^ 2
        y = [rand(Exponential(exp(β[t]))) for t in 1:p]
        negelbo(ψ) = -GLLVModels._va_site_exponential_elbo(ψ, y, Λ, Λ2, β)
        h = 1e-6
        for _ in 1:3
            ψ = randn(2K)
            G = zeros(2K)
            GLLVModels._va_site_exponential_grad!(G, ψ, y, Λ, Λ2, β)
            for i in 1:(2K)
                ψp = copy(ψ); ψp[i] += h
                ψm = copy(ψ); ψm[i] -= h
                fd = (negelbo(ψp) - negelbo(ψm)) / (2h)
                @test isapprox(G[i], fd; atol = 1e-5)
            end
        end
    end

    @testset "fit_exponential_gllvm_va runs and returns a finite ELBO" begin
        Random.seed!(323)
        p, K, n = 6, 2, 80
        βtrue = 0.3 .* randn(p) .+ 0.5
        Λtrue = 0.4 .* randn(p, K)
        Y = Matrix{Float64}(undef, p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                Y[t, s] = rand(Exponential(exp(βtrue[t] + sum(Λtrue[t, :] .* z))))
            end
        end
        fit = GLLVModels.fit_exponential_gllvm_va(Y; K = K)
        @test isfinite(fit.loglik)
        @test size(fit.Λ) == (p, K)
        @test length(fit.β) == p
    end

    @testset "VA-based standard errors (objective=:va)" begin
        Random.seed!(324)
        p, K, n = 6, 1, 150
        β = 0.8 .+ 0.3 .* randn(p)
        Λ = 0.7 .* randn(p, K)
        Y = Matrix{Float64}(undef, p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                Y[t, s] = rand(Exponential(exp(β[t] + sum(Λ[t, :] .* z))))
            end
        end
        fit = GLLVModels.fit_exponential_gllvm_va(Y; K = K)

        ci_va = confint(fit, Y; method = :wald, objective = :va)
        ci_la = confint(fit, Y; method = :wald, objective = :laplace)
        nterm = p + (p * K - div(K * (K - 1), 2))
        @test length(ci_va.term) == nterm
        @test ci_va.method == :wald
        fin = isfinite.(ci_va.se)
        @test any(fin)
        for i in eachindex(ci_va.term)
            if isfinite(ci_va.lower[i]) && isfinite(ci_va.upper[i])
                @test ci_va.lower[i] ≤ ci_va.estimate[i] ≤ ci_va.upper[i]
            end
        end
        @test ci_va.term == ci_la.term
        @test ci_va.estimate ≈ ci_la.estimate
        @test ci_va.estimate[1] ≈ fit.β[1] atol = 1e-8

        ct = coef_table(fit, Y; objective = :va)
        ctfin = isfinite.(ct.std_error)
        @test any(ctfin)
        @test all(ct.lower[ctfin] .< ct.estimate[ctfin] .< ct.upper[ctfin])

        @test_throws ArgumentError confint(fit, Y; method = :profile, objective = :va)
    end
end
