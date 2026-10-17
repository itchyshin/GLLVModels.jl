# #822: fit_phylo_latent_gllvm accepts a stop at the objective's rounding floor as converged, under a
# documented criterion (src/phylo_latent.jl, `_phylo_latent_floor_check`): PD finite-difference Hessian,
# Newton decrement g' H^-1 g / 2 <= 1000 * eps() * max(1, |nll|), and max |g| <= 100 * max(g_tol, 1e-5).
#
# Forcing design (platform-independent): `g_tol = 1e-9` is below what the central-difference gradient
# can resolve (its rounding noise is about eps()^(2/3) * |nll| / h-scale, orders of magnitude above 1e-9),
# so on the base commit the fit stops at the floor with `converged = false` / `:gradient_not_converged`
# on every platform, at a correct optimum. The criterion must report it converged with
# `:rounding_floor`, and the fit must be stationary by an independent computation. A fit that is NOT
# at a stationary point (iteration-limited, or parameters displaced from the optimum) must not be accepted.
using Test, LinearAlgebra, Random, StableRNGs, GLLVModels

const _PLRF_NEWICK = "(((s1:2,s2:2):1,(s3:1,s4:1):2):1,((s5:1.5,s6:1.5):1,(s7:1,s8:1):1.5):1.5);"

function _plrf_data(; seed = 822, n_traits = 3, reps = 4)
    rng = StableRNG(seed)
    tips = ["s$(i)" for i in 1:8]
    phy = GLLVModels.augmented_phy(_PLRF_NEWICK; correlation = true)
    C = GLLVModels.sigma_phy_dense(phy)
    o = sortperm(phy.leaf_names)
    C = C[o, o]
    L = cholesky(Symmetric(C + 1e-12I)).L
    loading = [0.8 0.0; 0.4 0.5; -0.3 0.6][1:n_traits, :]
    g = reduce(hcat, [L * randn(rng, 8) for _ in 1:2])
    obs = repeat(1:8; inner = reps)
    Y = collect(range(-0.5, 0.5; length = n_traits)) .+ loading * g[obs, :]' .+
        0.3 .* randn(rng, n_traits, length(obs))
    return Y, tips[obs]
end

function _plrf_stationarity(fit)
    f = theta -> GLLVModels._precision_multivariate_nll(fit.response, fit.phy, theta;
        rank = fit.rank, mode = fit.mode, residual_mode = fit.residual_mode,
        species_id = fit.species_id, mean_design = fit.mean_design)
    g = similar(fit.parameters)
    GLLVModels._pmv_fd_gradient!(g, f, fit.parameters)
    H = Symmetric(GLLVModels._fd_hessian(f, fit.parameters))
    return (pd = isposdef(H), decrement = dot(g, H \ g) / 2, gmax = maximum(abs, g),
        value = f(fit.parameters))
end

@testset "phylo_latent rounding-floor convergence (#822)" begin
    Y, sp = _plrf_data()
    default = fit_phylo_latent_gllvm(Y, sp; d = 2, tree = _PLRF_NEWICK)
    @test default.converged && default.stopping_reason === :converged   # default g_tol: unchanged path

    @testset "g_tol below FD resolution: accepted only as :rounding_floor" begin
        fit = fit_phylo_latent_gllvm(Y, sp; d = 2, tree = _PLRF_NEWICK, g_tol = 1e-9)
        @test fit.converged                                  # forcing assertion: false on the base commit
        @test fit.stopping_reason === :rounding_floor
        st = _plrf_stationarity(fit)                         # independent of the fitter's own gate
        @test st.pd && fit.hessian_positive_definite
        @test st.decrement <= 1000 * eps() * max(1.0, abs(st.value))
        @test st.gmax <= 100 * 1e-5
        @test abs(fit.loglik - default.loglik) <= 1e-6       # same optimum as the default-tolerance fit
    end

    @testset "not accepted away from a stationary point" begin
        few = fit_phylo_latent_gllvm(Y, sp; d = 2, tree = _PLRF_NEWICK, g_tol = 1e-9, iterations = 2)
        @test !few.converged && few.stopping_reason === :iteration_limit
        # Displaced parameters: the criterion itself must refuse (gradient / decrement far from the floor).
        moved = GLLVModels.fit_precision_multivariate(Y, default.phy; rank = 2, mode = :barelowrank,
            residual_mode = :shared, species_id = default.species_id, start = default.parameters .+ 0.3,
            iterations = 1, g_tol = 1e-9)
        @test !moved.converged
        accepted, decrement, gmax = GLLVModels._phylo_latent_floor_check(moved, 1e-9)
        @test !accepted
        @test decrement > 1000 * eps() * max(1.0, abs(moved.loglik)) || gmax > 100 * 1e-5
    end
end
