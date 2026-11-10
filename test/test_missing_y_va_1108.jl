# #1108: the Gaussian-variational (VA) fitters are typed on Integer / Real matrices, so a
# `Matrix{Union{Missing,T}}` raised a no-method MethodError. They now throw an ArgumentError
# naming the fitter and the first missing cell. A `Union{Missing,T}` eltype with no `missing`
# actually present fits like the plain matrix.
using Test, GLLVModels, Random

@testset "#1108 VA fitters and missing in Y" begin
    rng = MersenneTwister(1)
    p, n = 5, 30
    Yp = rand(rng, 0:4, p, n)
    Yn = rand(rng, 0:1, p, n)
    Yg = exp.(randn(rng, p, n))
    Yb = 0.05 .+ 0.9 .* rand(rng, p, n)

    U(Y) = Matrix{Union{Missing,eltype(Y)}}(Y)          # Union eltype, nothing missing
    function M(Y)                                       # one missing cell at (1, 1)
        Z = U(Y); Z[1, 1] = missing; return Z
    end
    thrown(f) = try; f(); nothing; catch e; e; end
    typed(e, name) = e isa ArgumentError && occursin(name, e.msg) &&
        occursin("(row, column) = (1, 1)", e.msg)

    cases = (
        ("fit_poisson_gllvm_va", fit_poisson_gllvm_va, Yp),
        ("fit_binomial_gllvm_va", fit_binomial_gllvm_va, Yn),
        ("fit_nb_gllvm_va", fit_nb_gllvm_va, Yp),
        ("fit_gamma_gllvm_va", fit_gamma_gllvm_va, Yg),
        ("fit_beta_gllvm_va", fit_beta_gllvm_va, Yb),
        ("fit_exponential_gllvm_va", fit_exponential_gllvm_va, Yg),
        ("fit_delta_gamma_gllvm_va", fit_delta_gamma_gllvm_va, Yg),
    )

    @testset "missing cells give a typed ArgumentError" begin
        for (name, f, Y) in cases
            @test typed(thrown(() -> f(M(Y); K = 1)), name)
        end
    end

    @testset "Union eltype without missing fits like the plain matrix" begin
        for (name, f, Y) in cases
            base = f(Y; K = 1)
            alt = f(U(Y); K = 1)
            @test alt.loglik == base.loglik
        end
    end

    @testset "plain matrices unchanged" begin
        for (name, f, Y) in cases
            @test isfinite(f(Y; K = 1).loglik)
        end
    end

    @testset "Laplace Poisson control still derives a mask" begin
        @test isfinite(fit_poisson_gllvm(M(Yp); K = 1, iterations = 20).loglik)
    end
end
