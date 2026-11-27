# #1486: all-zero matrix on Hurdle-Poisson emits the same species-column warning as Poisson (#730).
using Test, GLLVModels, Logging, Random

@testset "#1486 Hurdle-Poisson all-zero warning" begin
    Random.seed!(11261426)
    Y = zeros(Int, 4, 25)
    fit = @test_logs (:warn, r"fit_hurdle_poisson_gllvm.*species column") fit_gllvm(
        Y; family = HurdlePoisson(), K = 1)
    @test fit isa HurdlePoissonFit
end