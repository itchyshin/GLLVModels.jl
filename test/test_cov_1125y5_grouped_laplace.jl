# Lane 1125y5: grouped_laplace.jl joint kernels and helpers.
using Test, GLLVModels, SparseArrays, LinearAlgebra, Distributions
const G = GLLVModels

@testset "1125y5 grouped_laplace helpers" begin
    @test G._grouped_hf_ho_identical(Poisson(), LogLink())
    @test G._grouped_hf_ho_identical(fill(Poisson(), 2), LogLink())
    @test G._joint_grouped_link_ok([Beta(4.0, 1.0), Beta(8.0, 1.0)], LogitLink())
    st = G._joint_grouped_data_status(Poisson(), [1.0, 2.0], [1.0, 1.0])
    @test st === :ok
    stb = G._joint_grouped_data_status(Beta(2.0, 1.0), [0.3, 1.5], [1.0, 1.0])
    @test stb === :invalid_data

    fail = G._joint_grouped_failure(:maxiter, 2; iterations = 50)
    @test fail.status == :maxiter
    @test length(fail.mode) == 2

    X = ones(2, 1)
    W = spzeros(2, 0)
    res = G.joint_grouped_laplace_loglik(Poisson(), [2.0, 1.0], ones(2), X, [log(2.0)], W;
        link = LogLink())
    @test res.status === :ok
    @test isfinite(res.loglik)

    S = sparse([1.0 0.0; 0.0 2.0])
    @test G._grouped_sparse_is_diagonal(S)
    d = G._grouped_diag_factor(S)
    @test collect(d.d) == [1.0, 2.0]

    A = sparse([1.0 0.0; 1.0 1.0])
    L = [1.0 2.0]
    Wdes = G.grouped_trait_design(A, L)
    @test Wdes == sparse(kron(A, L))

    beta_f = [Beta(4.0, 1.0), Beta(8.0, 1.0)]
    rb = G.joint_grouped_laplace_loglik(beta_f, [0.3, 0.6], ones(2), X, [0.0], W;
        link = LogitLink())
    @test rb.status == :ok
    @test rb.status === :ok
    @test isfinite(rb.loglik)
end