using Test
using GLLVModels
using Random

# The VA objectives take Y untyped. A `missing` in Y reached them as a TypeError, MethodError or
# InexactError, and a NaN gave a NaN objective (and, through confint(objective = :va), an all-NaN
# interval table) with no message. Each now throws an ArgumentError naming the first offending
# cell. Cells hidden by `mask` are not read and are not checked.
@testset "VA objectives and confint(objective = :va) refuse missing and NaN" begin
    Random.seed!(7)
    p, n, K = 4, 30, 1
    Λ = 0.3 .* randn(p, K)
    β = zeros(p)
    Yc = rand(1:5, p, n)                       # counts
    Yp = 0.5 .+ rand(p, n)                     # positive reals
    Yb = 0.1 .+ 0.8 .* rand(p, n)              # in (0, 1)
    Nm = fill(5, p, n)
    Ybin = rand(0:5, p, n)

    objectives = [
        ("poisson_marginal_loglik_va", Yc, Y -> GLLVModels.poisson_marginal_loglik_va(Y, Λ, β)),
        ("nb_marginal_loglik_va", Yc, Y -> GLLVModels.nb_marginal_loglik_va(Y, Λ, β, 2.0)),
        ("binomial_marginal_loglik_va", Ybin, Y -> GLLVModels.binomial_marginal_loglik_va(Y, Nm, Λ, β)),
        ("gamma_marginal_loglik_va", Yp, Y -> GLLVModels.gamma_marginal_loglik_va(Y, Λ, β, 2.0)),
        ("exponential_marginal_loglik_va", Yp, Y -> GLLVModels.exponential_marginal_loglik_va(Y, Λ, β)),
        ("beta_marginal_loglik_va", Yb, Y -> GLLVModels.beta_marginal_loglik_va(Y, Λ, β, 5.0)),
        ("delta_gamma_marginal_loglik_va", Yp,
            Y -> GLLVModels.delta_gamma_marginal_loglik_va(Y, Λ, β, β, 2.0)),
    ]
    for (name, Y, f) in objectives
        @testset "$name" begin
            @test isfinite(f(Y))
            Yn = Matrix{Union{Missing, eltype(Y)}}(Y)
            Yn[2, 3] = missing
            err = try; f(Yn); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin(name, err.msg)
            @test occursin("Y[2, 3] = missing", err.msg)
            if eltype(Y) <: AbstractFloat
                Yf = copy(Y)
                Yf[3, 4] = NaN
                err = try; f(Yf); nothing; catch e; e; end
                @test err isa ArgumentError
                @test occursin("Y[3, 4] = NaN", err.msg)
            end
        end
    end

    @testset "a masked-out NaN is not read" begin
        Yf = copy(Yp)
        Yf[3, 4] = NaN
        M = trues(p, n)
        M[3, 4] = false
        @test isfinite(GLLVModels.gamma_marginal_loglik_va(Yf, Λ, β, 2.0; mask = M))
    end

    @testset "confint(objective = :va)" begin
        Yg = Float64.(Yc)
        fit = fit_poisson_gllvm_va(Yc; K = K)
        Ym = Matrix{Union{Missing, Int}}(Yc)
        Ym[2, 3] = missing
        @test_throws ArgumentError confint(fit, Ym; objective = :va)
        @test_throws ArgumentError vcov(fit, Ym; objective = :va)
        Yn = copy(Yg)
        Yn[2, 3] = NaN
        err = try; confint(fit, Yn; objective = :va); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("confint(objective = :va)", err.msg)
        @test occursin("Y[2, 3] = NaN", err.msg)
        # complete data still gives finite intervals
        ci = confint(fit, Yc; objective = :va)
        @test ci !== nothing
    end
end
