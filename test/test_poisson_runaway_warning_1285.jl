using GLLVModels, Test, Random, Distributions, LinearAlgebra

# Synthetic counts with a separation-prone trait (seed 8 → max |Λ| ≈ 15 on trait 5).
function _poisson_separation_counts(seed::Integer = 8)
    rng = MersenneTwister(seed)
    p, n = 10, 30
    z = randn(rng, n)
    Y = zeros(Int, p, n)
    for t in 1:p, i in 1:n
        if t == 5
            Y[t, i] = z[i] > 1.2 ? rand(rng, Poisson(5)) + 1 : 0
        elseif t == 3
            Y[t, i] = z[i] > 0.5 ? rand(rng, Poisson(3)) + 2 : 0
        else
            Y[t, i] = rand(rng, Poisson(1))
        end
    end
    return Y
end

@testset "Poisson runaway-loading warning (#1285)" begin
  @testset "fit_poisson_gllvm warns and still returns a fit" begin
    Y = _poisson_separation_counts(8)
    pat = r"Poisson fit has runaway loadings on trait\(s\) 5"
    fit = @test_logs (:warn, pat) fit_poisson_gllvm(Y; K = 1, _warn_zero_columns = false)
    @test maximum(abs, fit.Λ) > 8
    @test fit.converged isa Bool
    @test isfinite(fit.loglik)
  end

  @testset "fit_gllvm Poisson route warns without changing the fit" begin
    Y = _poisson_separation_counts(8)
    pat = r"Poisson fit has runaway loadings"
    fit = @test_logs (:warn, pat) fit_gllvm(Y; family = Poisson(), K = 1, _warn_zero_columns = false)
    @test isfinite(fit.loglik)
    @test fit.converged isa Bool
  end
end