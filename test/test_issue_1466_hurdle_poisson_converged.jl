using Test, GLLVModels, Distributions, Random, Logging

@testset "issue 1466: hurdle-Poisson all-zero species warns and converged false" begin
    Random.seed!(55)
    Y = rand(Poisson(2), 4, 10)
    Y[3, :] .= 0

    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = Logging.with_logger(logger) do
        fit_hurdle_poisson_gllvm(Y; K = 1, iterations = 150)
    end
    @test !fit.converged
    @test isfinite(fit.loglik)
    warn_msgs = [string(l.message) for l in logger.logs if l.level == Logging.Warn]
    @test any(m -> occursin("fit_hurdle_poisson_gllvm", m) && occursin("species column(s) [3]", m), warn_msgs)
end