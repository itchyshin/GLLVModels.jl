# test/test_latte_kernel_identity.jl — Latte-kernel S4 identity gate.
#
# Keyword-ON must match keyword-OFF at rtol 1e-8 on Poisson, NB2 joint, and
# phylo-correlated W. No wall-time claim. NM-alone demotion FORBIDDEN.
#
# Run:
#   JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     test/test_latte_kernel_identity.jl

using Test
using LinearAlgebra
using SparseArrays
using Random
using Distributions
using GLLVModels

const RTOL = 1e-8

function _check_pair!(label, off, on)
    @test off.status === :ok && off.converged
    @test on.status === :ok && on.converged
    @test isapprox(on.loglik, off.loglik; rtol = RTOL, atol = 0)
    @test isapprox(on.mode, off.mode; rtol = RTOL, atol = 0)
    @test isapprox(on.logdet_precision, off.logdet_precision; rtol = RTOL, atol = 0)
    @test on.factor isa SparseArrays.CHOLMOD.Factor
    @info "latte-kernel identity" label=label Δll=abs(on.loglik - off.loglik) /
        max(1.0, abs(off.loglik))
end

@testset "Latte-kernel S4 identity (rtol 1e-8)" begin
    @testset "Poisson direct: diag + identical Hf≡Ho" begin
        unit = [1, 1, 2, 2, 3, 3, 4, 4]
        n_obs = length(unit)
        Z = GLLVModels._grouped_incidence(unit, n_obs)
        W = GLLVModels._grouped_laplace_design([Z], [reshape([0.55], 1, 1)])
        @test GLLVModels._grouped_sparse_is_diagonal(sparse(W' * W + I))
        @test GLLVModels._grouped_hf_ho_identical(Poisson(), LogLink())
        y = Float64.([1, 0, 2, 3, 1, 4, 2, 5])
        X = ones(n_obs, 1)
        beta = [0.4]
        off = GLLVModels.joint_grouped_laplace_loglik(Poisson(), y, ones(n_obs), X, beta, W;
            link = LogLink())
        on = GLLVModels.joint_grouped_laplace_loglik(Poisson(), y, ones(n_obs), X, beta, W;
            link = LogLink(), diag_precision = true, reuse_identical_hf_ho = true)
        _check_pair!("poisson_direct", off, on)
    end

    @testset "NB2 direct + fit: diag ON, Hf≢Ho reuse cold" begin
        fam = NegativeBinomial(4.0, 0.5)
        @test !GLLVModels._grouped_hf_ho_identical(fam, LogLink())
        unit = [1, 1, 2, 2, 3, 3, 4, 4]
        n_obs = length(unit)
        Z = GLLVModels._grouped_incidence(unit, n_obs)
        W = GLLVModels._grouped_laplace_design([Z], [reshape([0.7], 1, 1)])
        y = Float64.([1, 0, 2, 3, 1, 4, 2, 5])
        X = ones(n_obs, 1)
        beta = [0.2]
        off = GLLVModels.joint_grouped_laplace_loglik(fam, y, ones(n_obs), X, beta, W;
            link = LogLink())
        on = GLLVModels.joint_grouped_laplace_loglik(fam, y, ones(n_obs), X, beta, W;
            link = LogLink(), diag_precision = true, reuse_identical_hf_ho = true)
        _check_pair!("nb2_direct", off, on)

        term = GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)
        Y = reshape(y, 1, :)
        start = [0.2, log(0.7), log(4.0)]
        fit_off = GLLVModels.fit_grouped_nongaussian(Y; family = fam, terms = [term],
            unit = unit, dispersion = :shared, start = start, iterations = 80,
            hessian = :fd, diag_precision_kernel = false)
        fit_on = GLLVModels.fit_grouped_nongaussian(Y; family = fam, terms = [term],
            unit = unit, dispersion = :shared, start = start, iterations = 80,
            hessian = :fd, diag_precision_kernel = true)
        @test isfinite(fit_off.loglik) && isfinite(fit_on.loglik)
        @test isapprox(fit_on.loglik, fit_off.loglik; rtol = RTOL, atol = 0)
        @test isapprox(fit_on.parameters, fit_off.parameters; rtol = RTOL, atol = 0)
        @info "NB2 joint fit identity" Δll=abs(fit_on.loglik - fit_off.loglik) /
            max(1.0, abs(fit_off.loglik))
    end

    @testset "Phylo-correlated W: non-diagonal Hf (diag falls back)" begin
        Random.seed!(20260923)
        tips = 4
        Aphy = Symmetric([1.0 0.55 0.30 0.12;
                          0.55 1.0 0.40 0.18;
                          0.30 0.40 1.0 0.50;
                          0.12 0.18 0.50 1.0])
        L = Matrix(cholesky(Aphy).L)
        n_obs = 16
        tip_id = repeat(1:tips; inner = n_obs ÷ tips)
        Z = zeros(Float64, n_obs, tips)
        for i in 1:n_obs
            Z[i, tip_id[i]] = 1.0
        end
        W = sparse(Z * L)
        y = Float64.(rand(Poisson(2.5), n_obs))
        X = ones(n_obs, 1)
        beta = [log(2.5)]
        off = GLLVModels.joint_grouped_laplace_loglik(Poisson(), y, ones(n_obs), X, beta, W;
            link = LogLink())
        on = GLLVModels.joint_grouped_laplace_loglik(Poisson(), y, ones(n_obs), X, beta, W;
            link = LogLink(), diag_precision = true, reuse_identical_hf_ho = true)
        _check_pair!("phylo_correlated_W", off, on)
        P = Matrix(off.precision)
        @test any(abs(P[i, j]) > 1e-10 for i in 1:tips, j in 1:tips if i != j)
    end

    @testset "phylo GLM dense-anchor regression" begin
        Random.seed!(606)
        phy = GLLVModels.augmented_phy(
            "(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        p = phy.n_leaves
        n = 8
        β = 0.3 .* randn(p)
        Y = rand(0:6, p, n)
        Ntr = ones(Int, p, n)
        σ2 = 0.5
        ℓ_sparse = GLLVModels.phylo_glm_marginal_loglik(Poisson(), Y, Ntr, β, σ2, phy;
            link = LogLink())
        keep = filter(i -> i != phy.root_index, 1:phy.n_total)
        Qc = Matrix(phy.Q_topology[keep, keep])
        leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
        Σa = σ2 .* (inv(Qc)[leaf_pos, leaf_pos])
        Pa = inv(Σa)
        rowsum = vec(sum(Y; dims = 2))
        a = zeros(p)
        Hd = Matrix{Float64}(I, p, p)
        for _ in 1:200
            μ = exp.(β .+ a)
            s_tot = rowsum .- n .* μ
            W_tot = n .* μ
            Hd = Pa + Diagonal(W_tot)
            a .+= Hd \ (s_tot .- Pa * a)
        end
        μ = exp.(β .+ a)
        ℓd = sum(GLLVModels._glm_logpdf(Poisson(), μ[t], 1, Y[t, s]) for t in 1:p, s in 1:n)
        ℓ_dense = ℓd - 0.5 * dot(a, Pa * a) + 0.5 * logdet(Pa) - 0.5 * logdet(Hd)
        @test isapprox(ℓ_sparse, ℓ_dense; atol = 1e-6)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("GATE S4 latte-kernel identity complete")
end
