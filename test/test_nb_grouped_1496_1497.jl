using Test
using GLLVModels
using Random

@testset "issues #1496 and #1497 grouped NB2 fitter guards and post-fit extractors" begin
  @testset "#1496 fit_nb_gllvm_grouped without group refuses cleanly" begin
      Random.seed!(261126)
      Y = rand(0:8, 4, 20)
      err = try
          fit_nb_gllvm_grouped(Y; K = 1, iterations = 30, g_tol = 1e-4)
          nothing
      catch e
          e
      end
      @test err isa ArgumentError
      @test occursin("`group`", sprint(showerror, err))
      @test !occursin("UndefKeywordError", sprint(showerror, err))
  end

  @testset "#1497 NBGroupedFit communality / extract_communality / correlation" begin
      Random.seed!(42)
      p, n = 4, 18
      Y = rand(0:6, p, n)
      fit = fit_nb_gllvm_grouped(Y; K = 1, group = collect(1:p), iterations = 50, g_tol = 1e-4)
      @test fit isa NBGroupedFit
      @test isfinite(fit.loglik)
      @test residuals(fit, Y) isa AbstractMatrix
      comm = communality(fit, Y)
      @test length(comm) == p
      @test all(x -> isfinite(x) && 0 <= x <= 1, comm)
      @test extract_communality(fit, Y) ≈ comm
      R = correlation(fit, Y)
      @test size(R) == (p, p)
      @test all(isfinite, R)
      @test all(t -> R[t, t] ≈ 1.0, 1:p)
      @test extract_correlations(fit, Y) ≈ R
  end
end