using GLLVModels, Test, Random

@testset "Ledermann K bound for fit_gaussian_pervar_gllvm (#723)" begin
    Random.seed!(2)
    Y = randn(4, 30)
    err = @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y; K = 2, method = :lbfgs, iterations = 2)
    msg = sprint(showerror, err.value)
    @test occursin("Ledermann", msg)
    @test occursin("(p - K)^2", msg)

    Y5 = randn(5, 40)
    err2 = @test_throws ArgumentError fit_gaussian_pervar_gllvm(Y5; K = 3, method = :lbfgs, iterations = 2)
    @test occursin("Ledermann", sprint(showerror, err2.value))

    # K = 2 is admissible for p = 5; guard must not fire (fit may still stop early).
    fit = fit_gaussian_pervar_gllvm(Y5; K = 2, method = :lbfgs, iterations = 5)
    @test fit isa GaussianPerVarFit
end
