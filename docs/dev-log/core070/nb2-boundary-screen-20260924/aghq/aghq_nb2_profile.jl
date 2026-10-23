# Stage 3: profile log-likelihood over the boundary trait's log r, Laplace vs AGHQ (k = 9,
# re-evaluated at k = 15), other 18 parameters re-optimised at each fixed value (warm starts).
include("/tmp/claude-503/nb2-aghq/aghq_nb2_eval_lib.jl")

pts = deserialize(joinpath(OUT, "points.jls"))
mx = deserialize(joinpath(OUT, "aghq_maxima.jls"))
K = 2; p = 5; rr = G.rr_theta_len(p, K); ir = p + rr

function make_negll_aghq(Y, K, k)
    p = size(Y, 1)
    function f(θ)
        β, Λ, r = unpack(θ, p, K)
        v = try -aghq_loglik(Y, Λ, β, r, k) catch; return 1e12 end
        return isfinite(v) ? v : 1e12
    end
    return f
end
function reopt_fixed(f, θ0, fixed; time_limit = 25.0, g_tol = 1e-5)
    free = setdiff(1:length(θ0), fixed)
    g(φ) = (θ = copy(θ0); θ[free] = φ; f(θ))
    res = Optim.optimize(g, θ0[free], LS,
        Optim.Options(g_tol = g_tol, iterations = 1000, time_limit = time_limit); autodiff = :finite)
    θ = copy(θ0); θ[free] = Optim.minimizer(res)
    return θ, res
end

t0 = time()
cases = (("seed46", load_fixture("nb2_restart_seed46.toml"), 3,
          [1.517, 1.0, 0.5, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0],
          pts["theta46_interior"], mx["seed46"]["interior"]),
         ("seed52", load_fixture("nb2_restart_seed52.toml"), 2,
          [0.078, -0.5, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
          pts["theta52_interior"], mx["seed52"]["interior"]))
prof = Dict{String,Any}()
for (lab, Y, b, grid, θlap0, θagh0) in cases
    fL = make_negll(Y, K); fA = make_negll_aghq(Y, K, 9); fA15 = make_negll_aghq(Y, K, 15)
    rows = Tuple{Float64,Float64,Float64,Float64}[]
    # warm starts: walk outward from the first grid value; below-first values restart from it
    θL_first = nothing; θA_first = nothing; θL_prev = θlap0; θA_prev = θagh0
    for (idx, v) in enumerate(grid)
        if idx > 1 && v < grid[1]
            sL, sA = θL_first, θA_first
        elseif idx > 1 && grid[idx - 1] < grid[1]
            sL, sA = θL_first, θA_first
        else
            sL, sA = θL_prev, θA_prev
        end
        sL = copy(sL); sL[ir + b] = v
        sA = copy(sA); sA[ir + b] = v
        θL, rL = reopt_fixed(fL, sL, [ir + b])
        θA, rA = reopt_fixed(fA, sA, [ir + b])
        a15 = -fA15(θA)
        push!(rows, (v, -Optim.minimum(rL), -Optim.minimum(rA), a15))
        @printf("%s logr%d=%6.3f  Laplace-profile %.6f (conv=%s)  AGHQ9-profile %.6f (conv=%s, k15 %.6f)  [%.0fs]\n",
                lab, b, v, -Optim.minimum(rL), Optim.converged(rL), -Optim.minimum(rA),
                Optim.converged(rA), a15, time() - t0)
        if idx == 1
            θL_first, θA_first = θL, θA
        end
        θL_prev, θA_prev = θL, θA
    end
    prof[lab] = sort(rows)
end
serialize(joinpath(OUT, "profiles.jls"), prof)
println("\n==== sorted profiles (logr, Laplace, AGHQ9, AGHQ15) ====")
for lab in ("seed46", "seed52")
    for r in prof[lab]
        @printf("%s %7.3f  %.6f  %.6f  %.6f\n", lab, r...)
    end
end
@printf("total %.1fs\n", time() - t0)
