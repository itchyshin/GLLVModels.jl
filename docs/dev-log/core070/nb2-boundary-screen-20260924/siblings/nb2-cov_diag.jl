# Diagnostic follow-up to nb2-cov.jl for the three stall datasets (d = 6, 7, 8).
#  1. Replicate the fitter's OPTIMIZER run from θ0 (not just the objective) and confirm it
#     lands on the public fit's loglik.
#  2. 1-D slice of the objective along each boundary group's log r at θ̂ (others fixed):
#     is the boundary a local max in that coordinate (separate basin) or an interior bump?
#  3. Apply the package's own _nb_boundary_restart (the #477 fix used by
#     fit_nb_gllvm_grouped) to the cov fitter's replicated objective: does it recover the gain?
include(joinpath(@__DIR__, "nb2-cov.jl"))   # defines helpers; its main() runs on ARGS -> pass ds

function diag(ds)
    println("\n\n######## DIAGNOSTICS ########")
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = G_TOL, iterations = ITERS)
    for d in ds
        Y, X, _ = simulate(d)
        pr = build_problem(Y, X)
        fit = GM.fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        res = Optim.optimize(pr.negll, pr.θ0, ls, opts; autodiff = :finite)
        @printf("\n[d=%d] replicated optimizer from θ0: loglik=%.9f (public %.9f, |diff|=%.1e)  Optim.converged=%s iters=%d\n",
                d, -Optim.minimum(res), fit.loglik, abs(-Optim.minimum(res) - fit.loglik),
                Optim.converged(res), Optim.iterations(res))
        θ̂ = theta_of_fit(fit)
        ll̂ = -pr.negll(θ̂)
        for g in findall(fit.dispersion_boundary)
            vals = Float64[]
            grid = [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, θ̂[pr.first_log_r - 1 + g]]
            for v in grid
                θs = copy(θ̂); θs[pr.first_log_r - 1 + g] = v
                push!(vals, -pr.negll(θs) - ll̂)
            end
            println("   slice group $g (log r -> Δloglik vs θ̂): ",
                    join([@sprintf("%.3g:%+.4f", grid[i], vals[i]) for i in eachindex(grid)], "  "))
        end
        rr = GM._nb_boundary_restart(pr.negll, res, ls, opts, pr.first_log_r)
        @printf("   _nb_boundary_restart applied: loglik=%.6f  Δ vs public=%+.6f  r=[%s]\n",
                -Optim.minimum(rr), -Optim.minimum(rr) - fit.loglik,
                fmt(exp.(Optim.minimizer(rr)[pr.first_log_r:end])))
    end
end

diag([6, 7, 8])
