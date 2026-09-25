using GLLVModels, Test, SHA, TOML

# fit_beta_gllvm_grouped (the default route for fit_gllvm(...; family = Beta())) could report
# converged = true at a point that is not stationary (#480). Optim stopped on a zero-length
# line-search step, which it counts as "objective did not change" (f_converged), while the
# gradient was still large (Optim g residual 9.06 against g_tol 1e-5). On the screen dataset
# d05 (true φ = 2) the fit stopped at logLik 265.976379 with φ[5] = 36.2; a fresh start with
# every log φ = 0 reaches 272.609413, where the gradient is about 6e-6.
# Now `converged` requires Optim's gradient criterion, and a run that stops without it is
# restarted (from the warm start with every log φ = 0, and from the returned point); the best
# run is kept only if it lowers the negative log-likelihood by more than 1e-6.
# Data: tools/beta_grouped_screen_data_draw.jl (drawn on Julia 1.10.12).

const _BETA480_DIR = joinpath(@__DIR__, "fixtures")

function _beta480_fixture(file)
    d = TOML.parsefile(joinpath(_BETA480_DIR, file))
    Y = reshape(Float64.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Y)))) == d["data_sha256"]
    return Y, d["K"]
end

# Largest central-difference gradient of the Beta grouped negative log-likelihood at the
# fitted point, computed from the public marginal and not from Optim's own bookkeeping.
function _beta480_max_grad(Y, β, Λ, φ, group)
    p, K = size(Λ)
    rr = GLLVModels.rr_theta_len(p, K)
    θ = vcat(β, GLLVModels.pack_lambda(Λ), log.(φ))
    function f(θ)
        φg = exp.(θ[(p + rr + 1):end])
        -GLLVModels.beta_grouped_marginal_loglik_laplace(Y,
            GLLVModels.unpack_lambda(θ[(p + 1):(p + rr)], p, K), θ[1:p], φg[group])
    end
    h = 1e-5
    return maximum(eachindex(θ)) do j
        e = zeros(length(θ)); e[j] = h
        abs(f(θ .+ e) - f(θ .- e)) / (2h)
    end
end

@testset "Beta grouped precision: converged means stationary (#480)" begin
    @testset "d05: public route no longer stops at a non-stationary point" begin
        # Before: logLik 265.976379, converged = true, φ = [2.52, 2.94, 1.66, 2.08, 36.17],
        # central-difference max |gradient| 5.985.
        Y, K = _beta480_fixture("beta_grouped_screen_d05.toml")
        fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = K)
        @test fit isa BetaGroupedFit
        @test fit.loglik >= 272.609413 - 1e-5
        @test fit.converged
        @test _beta480_max_grad(Y, fit.β, fit.Λ, fit.φ, fit.group) < 1e-3
    end

    @testset "d05: fit_beta_gllvm_grouped_cov shares the path and the fix" begin
        # With an all-zero covariate the offset Xγ is exactly zero, so this objective is the
        # no-X objective plus a coordinate with zero gradient, and the stall reproduces.
        Y, K = _beta480_fixture("beta_grouped_screen_d05.toml")
        p, n = size(Y)
        fit = fit_beta_gllvm_grouped_cov(Y; X = zeros(p, n, 1), K = K)
        @test fit isa BetaGroupedCovFit
        @test fit.loglik >= 272.609413 - 1e-5
        @test fit.converged
        @test _beta480_max_grad(Y, fit.β, fit.Λ, fit.φ, fit.group) < 1e-3
    end

    @testset "d01: a fit whose first run meets the gradient criterion is unchanged" begin
        # Before this change: logLik 272.920887270, converged = true, 50 iterations, stopped
        # on the gradient norm (Optim g residual 7.9e-6).
        Y, K = _beta480_fixture("beta_grouped_screen_d01.toml")
        fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = K)
        @test fit.loglik ≈ 272.920887270 atol = 1e-6
        @test fit.converged
    end

    @testset "d01: a gradient criterion that is not met is reported as not converged" begin
        # With g_tol = 1e-12 Optim stops on "objective did not change" at the same point, with a
        # g residual near the finite-difference noise floor (about 4e-7), far above the
        # scale-aware threshold max(g_tol, g_tol * |nll|) of about 2.7e-10, and calls that
        # converged. No restart does better, so the first run is kept, and it must be reported
        # as not converged. Before: converged = true.
        Y, K = _beta480_fixture("beta_grouped_screen_d01.toml")
        fit = fit_beta_gllvm_grouped(Y; K = K, g_tol = 1e-12)
        @test fit.loglik ≈ 272.920887270 atol = 1e-6
        @test !fit.converged
    end

    @testset "restart helper: a gradient-converged run is returned as it is" begin
        Optim = GLLVModels.Optim
        ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
        opts = Optim.Options(g_tol = 1e-8, iterations = 200)
        # θ[2] plays log φ. Two wells in θ[2]: a local minimum near +2 and a lower one near
        # -2. From θ[2] = 3 the run converges (gradient criterion met) in the upper well; a
        # restart with θ[2] = 0 would reach the lower well, so returning `res` here shows the
        # helper did not restart a run that met the gradient criterion.
        twowell(θ) = (θ[1] - 1)^2 + (θ[2]^2 - 4)^2 / 4 + 0.5 * θ[2]
        θ_warm = [0.0, 3.0]
        res = Optim.optimize(twowell, θ_warm, ls, opts; autodiff = :finite)
        @test Optim.g_converged(res)
        @test Optim.minimum(Optim.optimize(twowell, [0.0, 0.0], ls, opts; autodiff = :finite)) <
              Optim.minimum(res) - 1
        @test GLLVModels._beta_grouped_gradient_restart(twowell, res, θ_warm, ls, opts, 2) === res
    end
end
