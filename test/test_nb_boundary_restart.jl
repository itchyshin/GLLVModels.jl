using GLLVModels, Test, SHA, TOML

# fit_nb_gllvm_grouped (the default NB2 route, per-trait dispersion) could stall with a
# group's r pushed to the Poisson boundary, well below a better point (#477). The fitter now
# restarts the boundary groups at r = 1 (together, and each on its own) and keeps the best
# result only if it is better. The NB2 likelihood on these small datasets has several
# maxima, so the tests assert "no worse than the best point we know", not a global optimum.
# Data: tools/nb2_parity_data_draw.jl SEED 1.0 (NATIVE-06 design, drawn on Julia 1.10.12).

const _NB2_RESTART_DIR = joinpath(@__DIR__, "fixtures")

function _nb2_restart_fixture(file)
    d = TOML.parsefile(joinpath(_NB2_RESTART_DIR, file))
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]
    return Y
end

@testset "NB2 per-trait dispersion: boundary restart (#477)" begin
    @testset "seed 46: reaches gllvmTMB's point instead of stalling" begin
        # Before: logLik -836.924 with groups 2 and 5 at the boundary. gllvmTMB (frozen
        # b4d5fee64) reaches -835.369277 with every r finite; that point is a local
        # maximum (a point with trait 3 at the Poisson limit is 0.317 higher).
        Y = _nb2_restart_fixture("nb2_restart_seed46.toml")
        fit = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
        @test fit isa NBGroupedFit
        @test fit.loglik >= -835.369277 - 1e-5
    end

    @testset "seed 52: a genuine boundary group is kept while a stalled one is reset" begin
        # Before: -835.7705 with groups 2 and 5 at the boundary. Resetting both gives
        # -833.6085; resetting group 5 alone gives -833.1814 with trait 2 still at the
        # boundary, which is the better point.
        Y = _nb2_restart_fixture("nb2_restart_seed52.toml")
        fit = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
        @test fit.loglik >= -833.181377 - 1e-5
        @test fit.dispersion_boundary[2]
        @test !fit.converged   # a boundary group is still flagged
    end

    @testset "covariate route: fit_nb_gllvm_grouped_cov gets the same restart" begin
        # Sibling screen d7 (docs/dev-log/core070/nb2-boundary-screen-20260924/siblings/): the
        # fit used to stop at -853.078517 with groups 3 and 4 at the boundary; the restart
        # reaches -851.780905 with groups 2 and 5 there instead.
        d = TOML.parsefile(joinpath(_NB2_RESTART_DIR, "nb2cov_restart_d7.toml"))
        Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
        x = Float64.(d["x"])
        @test bytes2hex(sha256(vcat(reinterpret(UInt8, vec(Float64.(Y))), reinterpret(UInt8, x)))) ==
              d["data_sha256"]
        X = Array{Float64}(undef, d["p"], d["n"], 1)
        for t in 1:d["p"], s in 1:d["n"]
            X[t, s, 1] = x[s]
        end
        fit = fit_nb_gllvm_grouped_cov(Y; X = X, K = d["K"], group = collect(1:d["p"]))
        @test fit.loglik >= -851.780905 - 1e-5
    end

    @testset "restart helper: keeps a better restart, discards one that is not better" begin
        Optim = GLLVModels.Optim
        ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
        opts = Optim.Options(g_tol = 1e-8, iterations = 200)
        # θ[2] is a log-dispersion; exp(θ[2]) > 1e6 counts as the boundary.
        # A genuine boundary optimum: the objective keeps falling as θ[2] grows.
        genuine(θ) = (θ[1] - 1)^2 + exp(-θ[2])
        res = Optim.optimize(genuine, [0.0, 20.0], ls, opts; autodiff = :finite)
        @test GLLVModels._nb_boundary_restart(genuine, res, ls, opts, 2) === res
        # A stall: flat far out, a clearly better basin at θ[2] = 0.
        stalled(θ) = (θ[1] - 1)^2 - exp(-θ[2]^2)
        res = Optim.optimize(stalled, [0.0, 20.0], ls, opts; autodiff = :finite)
        kept = GLLVModels._nb_boundary_restart(stalled, res, ls, opts, 2)
        @test kept !== res
        @test Optim.minimum(kept) < Optim.minimum(res) - 0.5
    end
end
