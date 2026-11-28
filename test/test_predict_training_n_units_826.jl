using GLLVModels, Test, Random, Distributions

# Issue #826: fits without a stored offset keep the training unit count so
# predict(...; modes = :training) can refuse the wrong column count.

@testset "predict modes = :training training n_units (#826)" begin
  p, n = 4, 30
  rng = MersenneTwister(826)
  Y = [rand(rng, Poisson(3)) for _ in 1:p, _ in 1:n]
  f = fit_gllvm(Y; family = Poisson(), K = 1)
  @test f.offset === nothing
  @test f.n_units == n
  @test_throws ArgumentError predict(f, Y[:, 1:(n ÷ 2)]; type = :link, modes = :training)
  err = try
    predict(f, Y[:, 1:(n ÷ 2)]; type = :link, modes = :training)
  catch e
    e
  end
  @test err isa ArgumentError && occursin("training data", err.msg) && occursin("$p×$n", err.msg)

  fg = fit_nb_gllvm_grouped(Y; K = 1, group = 1:p)
  @test fg.offset === nothing
  @test fg.n_units == n
  @test_throws ArgumentError predict(fg, Y[:, 1:5]; type = :link, modes = :training)
end