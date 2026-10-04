using GLLVModels, Test, LinearAlgebra, Random, Statistics, ForwardDiff

# Missing-predictor FIML (the mi() axis), Gaussian Phase-2a slice: a site-level
# continuous predictor x (one value per site, may be `missing`) modelled as
# x ~ N(μ_x, σ_x²) and integrated out in CLOSED FORM via the joint Gaussian of
# (y_s, x_s). Faithful to gllvmTMB's mi() unit-level semantic: a single slope b_x
# broadcast across all traits. No Laplace, no formula parser.
@testset "fit_gaussian_mi_fiml (missing-predictor FIML, Gaussian)" begin
    # y_s = a + b_x x_s 1_p + Λ η_s + σ_eps ε_s ;  x_s ~ N(μ_x, σ_x²)
    function simulate(; p = 4, n = 300, K = 1, a, b_x, μ_x, σ_x, σ_eps, Λ, seed = 1)
        Random.seed!(seed)
        x = μ_x .+ σ_x .* randn(n)
        η = randn(K, n)
        y = a .+ b_x .* x' .+ Λ * η .+ σ_eps .* randn(p, n)
        return collect(y), collect(x)
    end

    @testset "complete-data equivalence with fit_gaussian_gllvm" begin
        p, n, K = 4, 300, 1
        a = [0.5, -0.3, 0.2, 0.1]
        b_x = 0.8
        Λ = reshape([0.7, 0.5, -0.4, 0.3], p, K)
        y, x = simulate(; p, n, K, a, b_x, μ_x = 1.0, σ_x = 0.7, σ_eps = 0.3, Λ, seed = 7)

        # ordinary fit: per-trait intercepts (p cols) + broadcast x covariate (1 col)
        Xfull = zeros(p, n, p + 1)
        for t in 1:p
            Xfull[t, :, t] .= 1.0
        end
        Xfull[:, :, p + 1] .= reshape(x, 1, n)
        fit_g = fit_gaussian_gllvm(y; K = K, X = Xfull)
        b_x_g = fit_g.pars.β[end]
        a_g = fit_g.pars.β[1:p]

        res = fit_gaussian_mi_fiml(y, x; K = K)
        @test res.converged
        # slope + intercepts are rotation-invariant mean params: must match the
        # ordinary fit when x is fully observed (the x-model factors out).
        @test res.b_x ≈ b_x_g atol = 1e-2
        @test res.a ≈ a_g atol = 1e-2
    end

    @testset "recovers b_x on complete data (smoke)" begin
        p, n, K = 4, 400, 1
        b_x = 1.0
        Λ = reshape([0.6, 0.5, -0.4, 0.3], p, K)
        y, x = simulate(; p, n, K, a = zeros(p), b_x, μ_x = 0.5, σ_x = 0.8, σ_eps = 0.3, Λ, seed = 3)
        res = fit_gaussian_mi_fiml(y, x; K = K)
        @test abs(res.b_x - b_x) < 0.15
    end

    @testset "fits with missing x cells and returns EBLUPs" begin
        p, n, K = 4, 300, 1
        b_x = 0.9
        Λ = reshape([0.6, 0.5, -0.4, 0.3], p, K)
        y, x = simulate(; p, n, K, a = zeros(p), b_x, μ_x = 0.5, σ_x = 0.8, σ_eps = 0.3, Λ, seed = 5)
        xm = Vector{Union{Missing,Float64}}(x)
        miss = [3, 17, 50, 120, 200, 250]
        xtrue = x[miss]
        xm[miss] .= missing
        res = fit_gaussian_mi_fiml(y, xm; K = K)
        @test res.converged
        @test length(res.eblup_x) == n
        @test all(!ismissing, res.eblup_x)
        # EBLUPs at missing sites track the held-out truth
        @test cor(res.eblup_x[miss], xtrue) > 0.5
    end

    @testset "packed NLL is AD-clean (ForwardDiff vs central FD ≤ 1e-6)" begin
        p, n, K = 4, 120, 1
        Λ = reshape([0.6, 0.5, -0.4, 0.3], p, K)
        y, x = simulate(; p, n, K, a = [0.2, -0.1, 0.0, 0.3], b_x = 0.8,
                        μ_x = 0.5, σ_x = 0.7, σ_eps = 0.3, Λ, seed = 2)
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[5, 20, 60, 90]] .= missing
        isobs = [!ismissing(xi) for xi in xm]
        xobs = [isobs[s] ? Float64(xm[s]) : 0.0 for s in 1:n]
        f(θ) = GLLVModels._mi_fiml_nll(θ, y, xobs, isobs, p, n, K)
        θ = vcat([0.2, -0.1, 0.0, 0.3], 0.8, 0.5, log(0.7), log(0.3), vec(Λ))
        g_ad = ForwardDiff.gradient(f, θ)
        h = 1e-6
        g_fd = similar(θ)
        for i in eachindex(θ)
            θp = copy(θ)
            θm = copy(θ)
            θp[i] += h
            θm[i] -= h
            g_fd[i] = (f(θp) - f(θm)) / (2h)
        end
        @test maximum(abs, g_ad .- g_fd) < 1e-6
    end

    # Heavy MC gate: FIML recovers b_x under MAR and beats complete-case deletion.
    # Opt-in (slow): GLLVM_SLOW_TESTS=1.
    if get(ENV, "GLLVM_SLOW_TESTS", "") == "1"
        @testset "FIML recovers b_x under MAR and beats complete-case" begin
            p, n, K = 4, 500, 1
            b_x_true = 1.0
            Λ = reshape([0.6, 0.5, -0.4, 0.3], p, K)
            bf = Float64[]
            bc = Float64[]
            for r in 1:50
                y, x = simulate(; p, n, K, a = zeros(p), b_x = b_x_true,
                                μ_x = 0.5, σ_x = 0.8, σ_eps = 0.7, Λ, seed = 100 + r)
                Random.seed!(900 + r)
                y1 = y[1, :]                                   # MAR: missingness on a trait, not x
                pmiss = 1 ./ (1 .+ exp.(-(-0.4 .+ 3.0 .* (y1 .- mean(y1)) ./ std(y1))))
                miss = rand(n) .< pmiss
                xm = Vector{Union{Missing,Float64}}(x)
                xm[miss] .= missing
                push!(bf, fit_gaussian_mi_fiml(y, xm; K = K).b_x)
                obs = .!miss
                push!(bc, fit_gaussian_mi_fiml(y[:, obs], x[obs]; K = K).b_x)  # complete-case
            end
            @test abs(mean(bf) - b_x_true) < 0.04                          # FIML ~unbiased
            @test mean(abs.(bc .- b_x_true)) > 1.8 * mean(abs.(bf .- b_x_true))  # cc more biased
        end
    end
end

# Response mask (gllvmTMB miss_control(response = "include", predictor = "model")): missing or NaN
# cells of y contribute nothing; the rest of each site enters the observed-data likelihood. The
# reference below is written independently of the fitter's conditional factorisation: the joint
# Gaussian of (y_s, x_s), subset to the observed components, evaluated with Distributions.MvNormal.
@testset "fit_gaussian_mi_fiml: response mask (observed-data likelihood)" begin
    using Distributions: MvNormal, logpdf

    # params packed as the fitter's layout: [a; b_x; μ_x; γ; log σ_x; log σ_eps; vec(Λ)]
    function dense_ll(θ, y, x, Z, p, n, K)
        q = Z === nothing ? 0 : size(Z, 2)
        a = θ[1:p]; b = θ[p + 1]; μ = θ[p + 2]
        γ = θ[(p + 3):(p + 2 + q)]
        σx = exp(θ[p + 3 + q]); σe = exp(θ[p + 4 + q])
        Λ = reshape(θ[(p + 5 + q):end], p, K)
        ll = zero(eltype(θ))
        for s in 1:n
            m = q == 0 ? μ : μ + dot(Z[s, :], γ)
            mean_j = vcat(a .+ b * m, m)
            S = zeros(eltype(θ), p + 1, p + 1)
            S[1:p, 1:p] = Λ * Λ' + σe^2 * I + b^2 * σx^2 * ones(p, p)
            S[1:p, p + 1] .= b * σx^2
            S[p + 1, 1:p] .= b * σx^2
            S[p + 1, p + 1] = σx^2
            o = vcat([!(ismissing(y[t, s]) || isnan(y[t, s])) for t in 1:p], !ismissing(x[s]))
            any(o) || continue
            v = vcat([ismissing(y[t, s]) ? 0.0 : y[t, s] for t in 1:p], ismissing(x[s]) ? 0.0 : x[s])
            ll += logpdf(MvNormal(mean_j[o], Symmetric(S[o, o])), v[o])
        end
        return ll
    end
    # E[x_s | observed y_s] from the same joint
    function dense_eblup(r, y, x, Z, s, p)
        m = Z === nothing ? r.μ_x : r.μ_x + dot(Z[s, :], r.γ)
        o = [!(ismissing(y[t, s]) || isnan(y[t, s])) for t in 1:p]
        any(o) || return m
        Syy = r.Λ * r.Λ' + r.σ_eps^2 * I + r.b_x^2 * r.σ_x^2 * ones(p, p)
        ry = Float64[y[t, s] for t in 1:p if o[t]] .- (r.a[o] .+ r.b_x * m)
        return m + r.b_x * r.σ_x^2 * sum(Symmetric(Syy[o, o]) \ ry)
    end
    packed(r) = vcat(r.a, r.b_x, r.μ_x, r.γ, log(r.σ_x), log(r.σ_eps), vec(r.Λ))

    p, n, K = 4, 150, 1
    rng = Random.MersenneTwister(11)
    z = randn(rng, n)
    x = 0.3 .+ 0.7 .* z .+ 0.6 .* randn(rng, n)
    Λt = [0.9, 0.6, -0.5, 0.4]
    y = [0.5, -0.2, 0.1, 0.3] .+ 0.8 .* x' .+ Λt * randn(rng, n)' .+ 0.5 .* randn(rng, p, n)
    xm = Vector{Union{Missing,Float64}}(x)
    xmiss = [3, 8, 21, 40, 41, 77, 100, 133]
    xm[xmiss] .= missing
    ym = Matrix{Union{Missing,Float64}}(y)
    for s in 1:n, t in 1:p
        rand(rng) < 0.12 && (ym[t, s] = missing)
    end
    ym[1, 3] = missing          # site 3 misses x and one response
    ym[:, 60] .= missing        # a site with no response, x observed: only the x density
    ym[:, 41] .= missing        # a site with no response and no x: contributes nothing
    Zm = reshape(z, n, 1)

    r = fit_gaussian_mi_fiml(ym, xm; K = K, Z = Zm)
    @test r.converged
    @test r.n_missing == length(xmiss)
    @test r.n_missing_y == count(ismissing, ym)
    θ = packed(r)
    @test isapprox(r.logLik, dense_ll(θ, ym, xm, Zm, p, n, K); rtol = 1e-10)
    # a stationary point of the independent reference
    @test maximum(abs, ForwardDiff.gradient(t -> dense_ll(t, ym, xm, Zm, p, n, K), θ)) < 1e-3
    # conditional modes at the missing-x sites use only the observed responses
    @test all(isapprox(r.eblup_x[s], dense_eblup(r, ym, xm, Zm, s, p); atol = 1e-10) for s in xmiss)
    @test r.eblup_x[41] ≈ r.μ_x + r.γ[1] * z[41]

    # NaN marks a missing response exactly as `missing` does
    yn = Float64[ismissing(v) ? NaN : v for v in ym]
    rn = fit_gaussian_mi_fiml(yn, xm; K = K, Z = Zm)
    @test rn.logLik == r.logLik
    @test rn.eblup_x == r.eblup_x

    # complete responses: the same reference
    rc = fit_gaussian_mi_fiml(y, xm; K = K, Z = Zm)
    @test rc.n_missing_y == 0
    @test isapprox(rc.logLik, dense_ll(packed(rc), y, xm, Zm, p, n, K); rtol = 1e-10)

    # a trait with no observed response is refused
    yb = copy(ym); yb[2, :] .= missing
    @test_throws ArgumentError fit_gaussian_mi_fiml(yb, xm; K = K, Z = Zm)
end
