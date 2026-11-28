using GLLVModels, Random, Test

# #1408: imputed training Y (no Missing eltype) should count the same nobs/bic as Ym
# when the fit retained the Gaussian integration mask.
@testset "issue 1408: nobs/bic with imputed Y after masked Gaussian fit" begin
    Random.seed!(991)
    Y = randn(3, 8)
    Ym = Matrix{Union{Missing, Float64}}(Y)
    Ym[2, 3] = missing
    Ym[1, 5] = missing
    Yimp = coalesce.(Ym, 0.0)

    fit = fit_gaussian_gllvm(Ym; K = 1, iterations = 150)
    @test GLLVModels._has_gaussian_record(fit)

    n_m = nobs(fit, Ym)
    n_imp = nobs(fit, Yimp)
    @test n_m == 22
    @test n_imp == n_m

    bic_m = bic(fit, Ym)
    bic_imp = bic(fit, Yimp)
    @test bic_imp ≈ bic_m atol = 1e-10

    mask = BitMatrix(.!ismissing.(Ym))
    @test nobs(fit, Yimp; mask = mask) == n_m
    @test bic(fit, Yimp; mask = mask) ≈ bic_m atol = 1e-10
end