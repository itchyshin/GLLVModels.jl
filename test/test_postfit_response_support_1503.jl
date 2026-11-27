using GLLVModels, Test, Random, Distributions
using GLLVModels: predict, fitted, residuals

@testset "post-fit response support matches fitters (#1503)" begin
    Random.seed!(27001)

  @testset "Binomial" begin
    Y = rand(0:1, 4, 20)
    fit = fit_binomial_gllvm(Y; K = 1, iterations = 120)
    Ybad = Y .+ 1
    @test_throws ArgumentError fit_binomial_gllvm(Ybad; K = 1)
    for f in (predict, fitted, residuals)
      err = try
        f(fit, Ybad)
        nothing
      catch e
        e
      end
      @test err isa ArgumentError
      @test occursin("fit_binomial_gllvm", sprint(showerror, err))
    end
  end

  @testset "Poisson" begin
    Yp = rand.(Poisson(2), 4, 18)
    fitp = fit_poisson_gllvm(Yp; K = 1, iterations = 100)
    Ypbad = Yp .- 3
    @test_throws ArgumentError fit_poisson_gllvm(Ypbad; K = 1)
    for f in (predict, fitted, residuals)
      err = try
        f(fitp, Ypbad)
        nothing
      catch e
        e
      end
      @test err isa ArgumentError
      @test occursin("fit_poisson_gllvm", sprint(showerror, err))
    end
  end

  @testset "Gamma" begin
    Yg = rand(Gamma(2.0, 1.0), 4, 16)
    fitg = fit_gamma_gllvm(Yg; K = 1, iterations = 100)
    Ygbad = Yg .- 3
    @test_throws ArgumentError fit_gamma_gllvm(Ygbad; K = 1)
    for f in (predict, fitted, residuals)
      err = try
        f(fitg, Ygbad)
        nothing
      catch e
        e
      end
      @test err isa ArgumentError
      @test occursin("fit_gamma_gllvm", sprint(showerror, err))
    end
  end
end