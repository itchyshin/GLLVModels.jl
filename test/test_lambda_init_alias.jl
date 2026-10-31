using GLLVModels, Test, Distributions, Random, LinearAlgebra

# #1008: `fit_gllvm(Y; family = Binomial(), λ_init = L)` used to MethodError
# because the Laplace path takes `Λ_init`. `λ_init` is now an alias. Omitting
# it must keep the existing warm start (no default change).

function _binomial_default_start(Y, K)
    p, n = size(Y)
    link = GLLVModels.LogitLink()
    Zemp = [GLLVModels.linkfun(link, clamp((Y[t, i] + 0.5) / 2, 1e-4, 1 - 1e-4))
            for t in 1:p, i in 1:n]
    β = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β)
    Λ = F.U[:, 1:K] .* (F.S[1:K]' ./ sqrt(n))
    return β, Λ
end

function _lambda_init_binary(; p = 5, n = 40, K = 2, seed = 20261030)
    rng = MersenneTwister(seed)
    Λ = 0.6 .* randn(rng, p, K)
    β = 0.2 .* randn(rng, p)
    η = β .+ Λ * randn(rng, K, n)
    P = 1 ./ (1 .+ exp.(-η))
    return Int.(rand(rng, p, n) .< P), K
end

function _lambda_init_counts(; p = 4, n = 30, K = 1, seed = 20261030)
    rng = MersenneTwister(seed)
    Λ = 0.4 .* randn(rng, p, K)
    η = log(2.0) .+ Λ * randn(rng, K, n)
    return [rand(rng, Poisson(exp(x))) for x in η], K
end

@testset "fit_gllvm Binomial accepts λ_init (#1008)" begin
    Y, K = _lambda_init_binary()
    base = fit_gllvm(Y; family = Binomial(), K = K)
    again = fit_gllvm(Y; family = Binomial(), K = K)
    @test again.loglik == base.loglik
    @test again.β == base.β
    @test again.Λ == base.Λ

    β0, Λ0 = _binomial_default_start(Y, K)
    same_L = fit_gllvm(Y; family = Binomial(), K = K, Λ_init = Λ0)
    same_l = fit_gllvm(Y; family = Binomial(), K = K, λ_init = Λ0)
    @test same_L.loglik == base.loglik
    @test same_l.loglik == base.loglik
    @test same_l.β == same_L.β
    @test same_l.Λ == same_L.Λ

    held = fit_gllvm(Y; family = Binomial(), K = K, λ_init = base.Λ,
                     β_init = base.β, iterations = 0)
    @test held.β == base.β
    @test held.Λ == base.Λ

    named = fit_binomial_gllvm(Y; K = K, λ_init = base.Λ, β_init = base.β,
                               iterations = 0)
    @test named.β == base.β
    @test named.Λ == base.Λ

    @test_throws ArgumentError fit_gllvm(Y; family = Binomial(), K = K,
                                         λ_init = Λ0, Λ_init = zero(Λ0))
end

@testset "fit_gllvm sibling Laplace paths accept λ_init (#1008)" begin
    Y, K = _lambda_init_counts()
    base = fit_gllvm(Y; family = Poisson(), K = K)
    again = fit_gllvm(Y; family = Poisson(), K = K)
    @test again.loglik == base.loglik
    @test again.β == base.β
    @test again.Λ == base.Λ

    via_l = fit_gllvm(Y; family = Poisson(), K = K, λ_init = base.Λ,
                      β_init = base.β, iterations = 0)
    via_L = fit_gllvm(Y; family = Poisson(), K = K, Λ_init = base.Λ,
                      β_init = base.β, iterations = 0)
    @test via_l.β == via_L.β == base.β
    @test via_l.Λ == via_L.Λ == base.Λ

    named = fit_poisson_gllvm(Y; K = K, λ_init = base.Λ, β_init = base.β,
                              iterations = 0)
    @test named.β == base.β
    @test named.Λ == base.Λ
end

@testset "two-part and ZI routes refuse λ_init by name (#1008)" begin
    Y = [0 1 0 2; 1 0 3 1]
    L = ones(2, 1)
    for family in (ZIPoisson(), ZINegBin(), HurdlePoisson(), DeltaLogNormal())
        err = try
            fit_gllvm(Y; family = family, K = 1, λ_init = L)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("λ_init", sprint(showerror, err))
        @test occursin("loading start", sprint(showerror, err))
        @test !occursin("Λ_init", sprint(showerror, err))
    end
end

@testset "λ_init and Λ_init compare with isequal (#1008)" begin
    L = fill(NaN, 2, 1)
    @test GLLVModels._resolve_loading_init(L, L; caller = "test") === L
    @test_throws ArgumentError GLLVModels._resolve_loading_init(zeros(2, 1), L;
                                                                caller = "test")
end
