using GLLVModels, Test, Random

# Hand-built Laplace fits: the bug is in `residuals`, not in the optimiser.
# A masked cell must be NaN, and the values in the other cells must not
# depend on the number stored in a masked cell (#1151).

function _m1151_poisson()
    β = [0.2, -0.1, 0.4]
    Λ = reshape([0.3, -0.2, 0.1], 3, 1)
    return GLLVModels.PoissonFit(β, Λ, GLLVModels.LogLink(), -10.0, true, 1)
end

function _m1151_binomial()
    β = [0.1, -0.3, 0.2]
    Λ = reshape([0.4, 0.1, -0.2], 3, 1)
    return GLLVModels.BinomialFit(β, Λ, GLLVModels.LogitLink(), -8.0, true, 1)
end

function _m1151_check(fit, Y, Y2, mask; N = nothing)
    for type in (:pearson, :dunnsmyth)
        # A fresh stream per call: one shared generator would make the second
        # Dunn–Smyth draw continue where the first left off.
        r1 = N === nothing ? residuals(fit, Y; mask = mask, type = type, rng = MersenneTwister(1)) :
                             residuals(fit, Y; N = N, mask = mask, type = type, rng = MersenneTwister(1))
        r2 = N === nothing ? residuals(fit, Y2; mask = mask, type = type, rng = MersenneTwister(1)) :
                             residuals(fit, Y2; N = N, mask = mask, type = type, rng = MersenneTwister(1))
        @test all(isnan, r1[.!mask])
        @test all(isnan, r2[.!mask])
        @test r1[mask] == r2[mask]
        @test all(isfinite, r1[mask])
    end
end

@testset "masked Laplace residuals are NaN and ignore the placeholder (#1151)" begin
    Y = [1 2 0 3; 0 1 4 1; 2 0 1 2]
    Y2 = copy(Y)
    Y2[1, 1] = 50
    Y2[2, 3] = 7
    mask = trues(size(Y))
    mask[1, 1] = false
    mask[2, 3] = false
    @testset "Poisson" begin
        _m1151_check(_m1151_poisson(), Y, Y2, mask)
    end
    @testset "Binomial" begin
        N = fill(8, size(Y))
        _m1151_check(_m1151_binomial(), Y, Y2, mask; N = N)
    end
end
