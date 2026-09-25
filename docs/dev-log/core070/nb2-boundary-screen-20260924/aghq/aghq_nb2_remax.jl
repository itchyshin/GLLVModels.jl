# Stage 2: re-maximise the AGHQ (k = 9) log-likelihood inside each basin, then
# evaluate the AGHQ maxima at k = 9, 15, 21 (+ trapezoid). Reuses stage-1 helpers.
include("/tmp/claude-503/nb2-aghq/aghq_nb2_eval_lib.jl")

pts = deserialize(joinpath(OUT, "points.jls"))
K = 2
Y46 = load_fixture("nb2_restart_seed46.toml")
Y52 = load_fixture("nb2_restart_seed52.toml")
p = 5; rr = G.rr_theta_len(p, K); ir = p + rr  # log r_t lives at θ[ir + t]

function make_negll_aghq(Y, K, k)
    p = size(Y, 1)
    function f(θ)
        β, Λ, r = unpack(θ, p, K)
        v = try
            -aghq_loglik(Y, Λ, β, r, k)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    return f
end

# Optimise with some coordinates held fixed.
function reopt_fixed(f, θ0, fixed::Vector{Int}; time_limit = 70.0, g_tol = 1e-5)
    free = setdiff(1:length(θ0), fixed)
    g(φ) = (θ = copy(θ0); θ[free] = φ; f(θ))
    res = Optim.optimize(g, θ0[free], LS,
                         Optim.Options(g_tol = g_tol, iterations = 1000, time_limit = time_limit);
                         autodiff = :finite)
    θ = copy(θ0); θ[free] = Optim.minimizer(res)
    return θ, res
end

t0 = time()
f46 = make_negll_aghq(Y46, K, 9)
tt = time(); for _ in 1:5; f46(pts["theta46_interior"]); end
@printf("one k=9 AGHQ negll eval: %.1f ms\n", (time() - tt) / 5 * 1000)

out = Dict{String,Any}()
for (lab, Y, θint, θbd, b) in (("seed46", Y46, pts["theta46_interior"], pts["theta46_boundary"], 3),
                               ("seed52", Y52, pts["theta52_interior"], pts["theta52_boundary"], 2))
    f = make_negll_aghq(Y, K, 9)
    # interior basin: all parameters free, start at the Laplace interior maximiser
    θi, ri = reopt_fixed(f, θint, Int[])
    @printf("%s AGHQ9 interior max: %.6f (start %.6f) logr=%s conv=%s it=%d |g|=%.1e (%.0fs)\n",
            lab, -Optim.minimum(ri), -f(θint), string(round.(θi[(ir + 1):end]; digits = 3)),
            Optim.converged(ri), Optim.iterations(ri), Optim.g_residual(ri), time() - t0)
    # boundary basin: trait b's log r held at 30 (Poisson limit), rest free
    θb0 = copy(θbd); θb0[ir + b] = 30.0
    θb, rb = reopt_fixed(f, θb0, [ir + b])
    @printf("%s AGHQ9 boundary max (logr%d=30 fixed): %.6f (start %.6f) logr=%s conv=%s it=%d |g|=%.1e (%.0fs)\n",
            lab, b, -Optim.minimum(rb), -f(θb0), string(round.(θb[(ir + 1):end]; digits = 3)),
            Optim.converged(rb), Optim.iterations(rb), Optim.g_residual(rb), time() - t0)
    out[lab] = Dict("interior" => θi, "boundary" => θb, "b" => b)
end
serialize(joinpath(OUT, "aghq_maxima.jls"), out)

println("\n==== AGHQ maxima evaluated at higher k ====")
for (lab, Y) in (("seed46", Y46), ("seed52", Y52))
    ev = Dict{String,Any}()
    for which in ("interior", "boundary")
        θ = out[lab][which]
        β, Λ, r = unpack(θ, p, K)
        lap = G.nb_grouped_marginal_loglik_laplace(Y, Λ, β, r; hessian = :observed,
                                                   maxiter = 100, tol = 1e-9)
        a = Dict(k => aghq_loglik(Y, Λ, β, r, k) for k in (1, 3, 5, 9, 15, 21))
        tr, e = trapezoid_loglik(Y, Λ, β, r; L = 9.0, h = 0.1)
        @printf("%s %-8s Laplace %.6f | AGHQ k1 %.6f k3 %.6f k5 %.6f k9 %.6f k15 %.6f k21 %.6f | trap %.6f\n",
                lab, which, lap, a[1], a[3], a[5], a[9], a[15], a[21], tr)
        ev[which] = (lap, a, tr)
    end
    @printf("%s boundary - interior at AGHQ maxima: k9 %+.6f  k15 %+.6f  k21 %+.6f  trap %+.6f\n",
            lab, ev["boundary"][2][9] - ev["interior"][2][9],
            ev["boundary"][2][15] - ev["interior"][2][15],
            ev["boundary"][2][21] - ev["interior"][2][21], ev["boundary"][3] - ev["interior"][3])
end
@printf("total %.1fs\n", time() - t0)
