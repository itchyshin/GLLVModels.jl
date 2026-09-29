using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

@testset "Variational (VA) marginal — Delta-Gamma" begin
    @testset "Λc=0 reduces to independent two-part Delta-Gamma loglik (exact)" begin
        Random.seed!(320)
        p, K, n = 6, 2, 40
        α = 3.0
        βz = 0.4 .* randn(p) .+ 0.2          # occurrence logits
        βc = 0.3 .* randn(p) .+ 0.5          # positive-part log-mean intercepts
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            π = inv(1 + exp(-βz[t]))
            if rand() < π
                Y[t, s] = rand(Gamma(α, exp(βc[t]) / α))
            else
                Y[t, s] = 0.0
            end
        end
        va = GLLVModels.delta_gamma_marginal_loglik_va(Y, zeros(p, K), βz, βc, α)
        ref = 0.0
        for t in 1:p, s in 1:n
            π = inv(1 + exp(-βz[t]))
            ref += Y[t, s] > 0 ?
                log(π) + logpdf(Gamma(α, exp(βc[t]) / α), Y[t, s]) :
                log(1 - π)
        end
        @test va ≈ ref atol = 1e-8
    end

    @testset "fit machinery (no recovery thresholds)" begin
        Random.seed!(321)
        p, K, n = 6, 2, 150
        α = 3.0
        βz = 0.5 .* randn(p) .+ 0.3
        βc = 0.3 .* randn(p) .+ 0.6
        Λc = 0.4 .* randn(p, K)
        Y = zeros(p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                π = inv(1 + exp(-βz[t]))
                if rand() < π
                    μ = exp(βc[t] + dot(Λc[t, :], z))
                    Y[t, s] = rand(Gamma(α, μ / α))
                else
                    Y[t, s] = 0.0
                end
            end
        end
        fit = GLLVModels.fit_delta_gamma_gllvm_va(Y; K = K)
        @test fit isa GLLVModels.DeltaGammaFit
        @test isfinite(fit.loglik)
        @test 0 < fit.α < 1e3
        @test size(fit.Λc) == (p, K)
    end

    @testset "analytic inner gradient matches central finite difference" begin
        Random.seed!(322)
        p, K = 5, 2
        α = 3.0
        βz = 0.5 .* randn(p) .+ 0.3
        βc = 0.3 .* randn(p) .+ 0.6
        Λc = 0.4 .* randn(p, K)
        Λc2 = Λc .^ 2
        # one site's y: a mix of zeros (absences) and positive Gamma draws
        y = zeros(p)
        for t in 1:p
            π = inv(1 + exp(-βz[t]))
            y[t] = rand() < π ? rand(Gamma(α, exp(βc[t]) / α)) : 0.0
        end
        y[1] = 0.0; y[2] = rand(Gamma(α, exp(βc[2]) / α))  # ensure both branches present
        negelbo(ψ) = -GLLVModels._va_site_dgamma_elbo(ψ, y, Λc, Λc2, βz, βc, α)
        h = 1e-6
        for _ in 1:3
            ψ = randn(2K)
            G = zeros(2K)
            GLLVModels._va_site_dgamma_grad!(G, ψ, y, Λc, Λc2, βz, βc, α)
            for i in 1:(2K)
                ψp = copy(ψ); ψp[i] += h
                ψm = copy(ψ); ψm[i] -= h
                fd = (negelbo(ψp) - negelbo(ψm)) / (2h)
                @test isapprox(G[i], fd; atol = 1e-5)
            end
        end
    end
end
