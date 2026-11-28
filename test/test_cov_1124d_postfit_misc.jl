# Lane 1124d: fourthcorner / mixed / summary_table / model_selection utilities.
using Test, GLLVModels, Random, Distributions, LinearAlgebra
const G = GLLVModels

@testset "1124d fourthcorner predict type" begin
  Random.seed!(1124)
  p, n, q, r, K = 4, 8, 2, 2, 1
  Y = rand(0:3, p, n)
  Xenv = randn(n, q)
  TR = randn(p, r)
  fit = fit_fourthcorner_gllvm(Y; family = Poisson(), Xenv = Xenv, TR = TR, K = K, iterations = 5)
  @test_throws ArgumentError predict(fit, Y, Xenv, TR; type = :linear)
  @test predict(fit, Y, Xenv, TR; type = :mean) == predict(fit, Y, Xenv, TR; type = :response)
end

@testset "1124d mixed.jl helpers" begin
  @test G._mixed_has_dispersion(Poisson()) == false
  @test_throws ArgumentError G._mixed_has_dispersion(Poisson)
  @test_throws ArgumentError G._check_mixed_offset_admission(
    [0.0 0.0; 0.5 0.0], [Poisson(), Normal()])
  offset = fill(0.0, 2, 5)
  G._check_mixed_offset_admission(offset, [Poisson(), Normal()]) === nothing

  Random.seed!(42)
  p, n = 3, 12
  Y = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
  fams = fill(Poisson(), p)
  fit = fit_mixed_gllvm(Y; families = fams, K = 1, iterations = 8)
  @test_throws ArgumentError predict(fit, Y; type = :fitted)
end

@testset "1124d summary_table show" begin
  ci = (
    estimate = [0.1, 0.2],
    se = [0.05, Inf],
    lower = [0.0, NaN],
    upper = [0.2, NaN],
    term = ["b1", "b2"],
  )
  ct = G._coef_table_from_ci(ci)
  @test ct isa GllvmCoefTable
  buf = sprint(show, "text/plain", ct)
  @test occursin("GllvmCoefTable", buf)
  @test occursin("estimate", buf)
  @test occursin("<1e-4", buf) || occursin("p", buf)
end

@testset "1124d model_selection show helpers" begin
  @test G._lv_format_flag(true) == "TRUE"
  @test G._lv_format_flag(false) == "FALSE"
  @test G._lv_format_flag(missing) == "NA"
  sel = LVSelection(
    [1, 2], [5, 7], [-10.0, -9.5], [20.0, 19.0], [25.0, 24.0], [22.0, 21.0],
    [21.5, 20.5], [true, false], [true, missing], 1, nothing,
    [(K = 1, status = :ok, loglik = -10.0, message = ""),
     (K = 2, status = :unconverged, loglik = -9.5, message = "stalled")],
    :bic,
  )
  buf = sprint(show, "text/plain", sel)
  @test occursin("latent-dimension selection", buf)
  @test occursin("pdHess", buf)
  @test sprint(show, sel) == "LVSelection(K=[1, 2], best_k=1)"
end

@testset "1124d postfit prediction mode guard" begin
  @test_throws ArgumentError G._check_prediction_modes(:invalid)
  @test G._check_prediction_modes(:refit)
end