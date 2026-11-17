using GLLVModels, Test, Distributions

# fit_quadratic_gllvm built the family marker from exp(log-dispersion) OUTSIDE the
# objective's try block (as fit_rrr_gllvm did, see test_rrr_dispersion_underflow.jl). A
# line-search step that drove the log-dispersion very negative made exp() underflow to
# 0.0 and Distributions threw `DomainError: NegativeBinomial: the condition r > zero(r)
# is not satisfied` out of the fit; the sibling fitters return the 1e12 failure value for
# such a point. The data are a sparse count community (p = 8 species, n = 50 sites, counts
# 0 to 3; Poisson draws from unimodal responses along one gradient) on which the
# main-branch fit threw.
@testset "quadratic: dispersion underflow in the line search does not escape" begin
    Y = [
    0 0 0 0 0 0 0 0 0 2 0 0 1 0 0 0 0 0 0 0 0 0 1 0 2 1 0 1 2 0 0 0 1 0 0 0 0 0 0 1 2 0 0 1 2 0 2 0 0 0;
    0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 2 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 1;
    0 0 0 1 0 1 0 0 0 0 0 0 0 2 0 0 0 2 0 0 0 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 2 0 1 1 0;
    1 2 1 1 0 0 2 0 0 1 1 0 0 1 0 1 1 0 0 0 0 1 1 1 0 2 0 1 0 2 0 1 0 1 0 0 0 1 2 0 0 0 0 2 0 1 0 0 1 0;
    0 0 0 3 0 1 0 0 3 0 0 0 0 0 0 1 0 0 0 2 0 0 0 0 0 0 0 1 0 2 0 0 1 1 0 2 0 0 0 2 2 0 0 0 2 0 0 2 0 1;
    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 2 0 2 2 0 0 0 1 0 0 1 0 2 3 1 0 0 0 0 0 0 0 1 1 0;
    1 0 3 0 0 1 0 0 0 0 1 0 0 0 0 0 0 2 0 0 1 1 0 0 1 0 0 2 0 0 0 0 0 1 0 1 0 1 1 0 2 1 0 0 0 0 3 0 0 1;
    0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 2 0 1 0 0 0 0 0 2 1 1 0 0 0 1 0 0 0 1 2 0 0 0 0 1 0 0 0 0 2 0 0 0;
    ]
    @test size(Y) == (8, 50)
    fit = fit_quadratic_gllvm(Y; family = NegativeBinomial(), K = 1)
    @test isfinite(fit.loglik)
    @test isfinite(fit.dispersion) && fit.dispersion > 0
end
