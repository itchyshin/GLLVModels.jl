using GLLVModels, Test, Random, Distributions, StatsModels

# Issue #742: on the default `gllvm(formula, Y, data)` route a user-supplied `X`
# used to replace the formula design silently (the later splatted keyword wins).
@testset "gllvm(formula, Y, data) refuses X on the default route (#742)" begin
    rng = MersenneTwister(742)
    p, n, K = 3, 24, 1
    temp = randn(rng, n)
    data = (temp = temp,)
    Yn = randn(rng, p, n)
    Yp = rand(rng, Poisson(2.0), p, n)
    Xother = randn(rng, p, n, 2)
    for (Y, fam) in ((Yn, Normal()), (Yp, Poisson()))
        # covariate route
        @test_throws ArgumentError gllvm(@formula(y ~ 1 + temp), Y, data;
                                         family = fam, K = K, X = Xother)
        # intercept-only route
        @test_throws ArgumentError gllvm(@formula(y ~ 1), Y, data;
                                         family = fam, K = K, X = Xother)
    end
end
