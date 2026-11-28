using Test, GLLVModels, Logging

@testset "issue 1467: binomial runaway loadings => converged false" begin
    Y = zeros(Int, 3, 20)
    N = fill(5, 3, 20)
    Y[1, :] .= (1:20) .% 2 .* 5

    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = Logging.with_logger(logger) do
        fit_binomial_gllvm(Y; N = N, K = 1, iterations = 300)
    end
    @test !fit.converged
    @test maximum(abs, fit.Λ) > 8
    warn_msgs = [string(l.message) for l in logger.logs if l.level == Logging.Warn]
    @test any(m -> occursin("runaway loadings", m), warn_msgs)
end