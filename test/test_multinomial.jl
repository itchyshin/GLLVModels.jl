using GLLVModels, Test, Random, LinearAlgebra, ForwardDiff, Distributions

# `using Distributions` after `using GLLVModels` makes bare `Multinomial`
# undefined (both export it). Identity keeps the public marker as
# `GLLVModels.Multinomial`; qualify here so the suite stays usable after
# earlier `runtests.jl` files have already imported Distributions.

# Central FD (same stencil as test_lognormal.jl / test_studentt.jl).
function _mn_central_fd_gradient(f, theta; h = 1e-6)
    g = similar(theta)
    @inbounds for i in eachindex(theta)
        step = h * max(1.0, abs(theta[i]))
        tp = copy(theta); tp[i] += step
        tm = copy(theta); tm[i] -= step
        g[i] = (f(tp) - f(tm)) / (2 * step)
    end
    return g
end

function _mn_logsumexp(η)
    m = maximum(η)
    return m + log(sum(exp(ηj - m) for ηj in η))
end

@testset "multinomial family (FE softmax, twin fid 16)" begin

    @testset "marker is Multinomial, not Categorical" begin
        m = GLLVModels.Multinomial()
        @test m isa GLLVModels.Multinomial
        @test nameof(typeof(m)) === :Multinomial
        @test nameof(typeof(m)) !== :Categorical
        @test !(m isa GLLVModels.Ordinal)
    end

    @testset "GLLVModels.Multinomial vs Distributions.Multinomial" begin
        m = GLLVModels.Multinomial()
        d = Distributions.Multinomial(3, [0.2, 0.3, 0.5])
        @test m isa GLLVModels.Multinomial
        @test d isa Distributions.Multinomial
        @test typeof(m) !== typeof(d)
        @test length(d) == 3
    end

    @testset "K < 3 fail-loud (use binomial-logit)" begin
        Y = reshape(Int[1, 2, 1, 2, 1, 2], 1, 6)
        err = try
            fit_multinomial_gllvm(Y)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = lowercase(sprint(showerror, err))
        @test occursin("k", msg) || occursin("categor", msg)
        @test occursin("binomial", msg)
    end

    @testset "y outside 1:K fail-loud" begin
        Y = reshape(Int[1, 2, 3, 0], 1, 4)
        @test_throws ArgumentError fit_multinomial_gllvm(Y; n_categories = 3)
        Y2 = reshape(Int[1, 2, 4], 1, 3)
        @test_throws ArgumentError fit_multinomial_gllvm(Y2; n_categories = 3)
    end

    @testset "one trait only; no TMB pseudo-rows" begin
        Y2 = Int[1 2 3; 2 1 3]
        @test_throws ArgumentError fit_multinomial_gllvm(Y2)
    end

    @testset "packing length is (K-1)(1+p); η₁ ≡ 0" begin
        @test GLLVModels.multinomial_pack_len(3, 0) == 2
        @test GLLVModels.multinomial_pack_len(4, 2) == 9
        @test_throws ArgumentError GLLVModels.multinomial_pack_len(2, 0)
        β = [0.4, -0.2]
        γ = zeros(2, 0)
        η = GLLVModels.multinomial_eta(β, γ, Float64[])
        @test length(η) == 3
        @test η[1] == 0
        @test η[2] ≈ 0.4
        @test η[3] ≈ -0.2
        βx = [0.1, 0.3]
        γx = [0.5  -0.2; 0.0  0.4]   # (K-1) × p, contrast-major rows
        ηx = GLLVModels.multinomial_eta(βx, γx, [1.0, 2.0])
        @test ηx[1] == 0
        @test ηx[2] ≈ 0.1 + 0.5 * 1.0 + (-0.2) * 2.0
        @test ηx[3] ≈ 0.3 + 0.0 * 1.0 + 0.4 * 2.0
    end

    @testset "one softmax per observation matches independent logsumexp" begin
        Random.seed!(601)
        y = Int[1, 2, 3, 2, 1, 3, 2, 1]
        Y = reshape(y, 1, length(y))
        β = [0.6, -0.3]
        θ = copy(β)
        ll = GLLVModels.multinomial_loglik(Y, θ; n_categories = 3)
        η = [0.0, β[1], β[2]]
        lse = _mn_logsumexp(η)
        ll_ref = sum(η[yi] - lse for yi in y)
        @test ll ≈ ll_ref atol = 1e-12
    end

    @testset "rejects LV K / num_lv" begin
        Y = reshape(Int[1, 2, 3, 1, 2, 3, 1, 2, 3], 1, 9)
        @test_throws ArgumentError fit_gllvm(Y; family = GLLVModels.Multinomial(), K = 1)
        @test_throws ArgumentError fit_gllvm(Y; family = GLLVModels.Multinomial(), num_lv = 2)
        @test_throws ArgumentError fit_multinomial_gllvm(Y; K = 1)
    end

    @testset "packed NLL FD vs ForwardDiff ≤ 1e-6" begin
        Random.seed!(602)
        n, ncat, p_cov = 40, 4, 2
        y = rand(1:ncat, n)
        Y = reshape(y, 1, n)
        X = randn(n, p_cov)
        θ = 0.3 .* randn(GLLVModels.multinomial_pack_len(ncat, p_cov))
        nll = θv -> -GLLVModels.multinomial_loglik(Y, θv; X = X, n_categories = ncat)
        gad = ForwardDiff.gradient(nll, θ)
        gfd = _mn_central_fd_gradient(nll, θ)
        @test maximum(abs, gad .- gfd) ≤ 1e-6
    end

    @testset "smoke fit recovers no-X intercepts" begin
        Random.seed!(603)
        n, ncat = 800, 3
        β_true = [0.8, -0.4]
        η = [0.0, β_true[1], β_true[2]]
        π = exp.(η) ./ sum(exp.(η))
        y = [findfirst(rand() .≤ cumsum(π)) for _ in 1:n]
        Y = reshape(Int.(y), 1, n)
        fit = fit_multinomial_gllvm(Y)
        @test fit isa MultinomialFit
        @test fit.n_categories == 3
        @test length(fit.theta_packed) == 2
        @test isfinite(fit.loglik)
        @test maximum(abs, fit.β .- β_true) < 0.25
        fit2 = fit_gllvm(Y; family = GLLVModels.Multinomial())
        @test fit2 isa MultinomialFit
        @test fit2.loglik ≈ fit.loglik atol = 1e-8
    end

    @testset "smoke fit +X packing (K-1)(1+p)" begin
        Random.seed!(604)
        n, ncat, p_cov = 600, 3, 1
        β_true = [0.5, -0.3]
        γ_true = reshape([0.8, -0.6], 2, 1)
        X = randn(n, p_cov)
        y = Vector{Int}(undef, n)
        for i in 1:n
            η = GLLVModels.multinomial_eta(β_true, γ_true, vec(X[i, :]))
            π = exp.(η) ./ sum(exp.(η))
            y[i] = findfirst(rand() .≤ cumsum(π))
        end
        Y = reshape(y, 1, n)
        fit = fit_multinomial_gllvm(Y; X = X)
        @test length(fit.theta_packed) == (ncat - 1) * (1 + p_cov)
        @test size(fit.γ) == (ncat - 1, p_cov)
        @test isfinite(fit.loglik)
        @test maximum(abs, fit.β .- β_true) < 0.35
        @test maximum(abs, fit.γ .- γ_true) < 0.35
    end

    @testset "post-fit predict / fitted / residuals / simulate (#1552)" begin
        Random.seed!(99)
        Kcat, n = 4, 40
        Y = zeros(Int, Kcat, n)
        for j in 1:n
            c = rand(1:Kcat)
            Y[c, j] = 1
        end
        fit = fit_multinomial_gllvm(Y; iterations = 200)
        @test fit.converged

        pr = predict(fit, Y; type = :prob)
        @test size(pr) == (1, n, Kcat)
        @test all(i -> sum(pr[1, i, :]) ≈ 1.0, 1:n)
        cls = predict(fit, Y; type = :class)
        @test size(cls) == (1, n)
        @test all(1 .<= cls[1, :] .<= Kcat)
        @test fitted(fit, Y) == cls
        η = predict(fit, Y; type = :link)
        @test size(η) == (Kcat, n)
        @test all(η[1, :] .== 0)

        rng = MersenneTwister(1552)
        R = residuals(fit, Y; rng = rng)
        @test size(R) == (1, n)
        @test all(isfinite, R)

        Ysim = simulate(fit, 12; rng = MersenneTwister(1553))
        @test size(Ysim) == (1, 12)
        @test all(1 .<= Ysim[1, :] .<= Kcat)

        fit2 = fit_multinomial_gllvm(reshape(Int[1, 2, 3, 2, 1, 3], 1, 6); iterations = 80)
        @test predict(fit2, [1, 2, 3, 2, 1, 3]) isa Matrix{Int}
    end
end
