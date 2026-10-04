using GLLVModels, Test, Random, LinearAlgebra, Distributions

function coevolution_marginal_input_error(; N, β)
    T, n = 3, 12
    K = Matrix(Symmetric(I(n)))
    Y = Float64.(rand(0:5, T, n))
    Λ = 0.3 .* ones(T, 1)
    try
        coevolution_glm_marginal_loglik(Binomial(), Y, N, β, Λ, 1.0, K)
        return nothing
    catch e
        return e
    end
end

@testset "coevolution_glm: N and β must match Y (#765)" begin
    errN = coevolution_marginal_input_error(N = fill(8.0, 3, 4), β = zeros(3))
    @test errN isa ArgumentError
    @test occursin("N", sprint(showerror, errN))

    errβ = coevolution_marginal_input_error(N = fill(8.0, 3, 12), β = zeros(2))
    @test errβ isa ArgumentError
    @test occursin("β", sprint(showerror, errβ))
end
