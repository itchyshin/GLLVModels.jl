# Stage 4: fine AGHQ (k = 9) and Laplace profile over log r3 for seed 46 on [1.3, 2.5],
# tighter g_tol, warm starts walking outward from the stage-2 AGHQ interior stationary point.
include("/tmp/claude-503/nb2-aghq/aghq_nb2_eval_lib.jl")
mx = deserialize(joinpath(OUT, "aghq_maxima.jls"))
pts = deserialize(joinpath(OUT, "points.jls"))
K = 2; p = 5; rr = G.rr_theta_len(p, K); ir = p + rr; b = 3
Y = load_fixture("nb2_restart_seed46.toml")
function make_negll_aghq(Y, K, k)
    function f(θ)
        β, Λ, r = unpack(θ, size(Y, 1), K)
        v = try -aghq_loglik(Y, Λ, β, r, k) catch; return 1e12 end
        return isfinite(v) ? v : 1e12
    end
end
function reopt_fixed(f, θ0, fixed; time_limit = 30.0, g_tol = 1e-7)
    free = setdiff(1:length(θ0), fixed)
    g(φ) = (θ = copy(θ0); θ[free] = φ; f(θ))
    res = Optim.optimize(g, θ0[free], LS,
        Optim.Options(g_tol = g_tol, iterations = 2000, time_limit = time_limit); autodiff = :finite)
    θ = copy(θ0); θ[free] = Optim.minimizer(res)
    return θ, res
end
t0 = time()
fA = make_negll_aghq(Y, K, 9); fL = make_negll(Y, K)
θA0 = mx["seed46"]["interior"]; θL0 = pts["theta46_interior"]
# gradient of the full AGHQ objective at the stage-2 interior stationary point (central FD)
gfd = [(e = zeros(length(θA0)); e[j] = 1e-4; (fA(θA0 .+ e) - fA(θA0 .- e)) / 2e-4) for j in eachindex(θA0)]
@printf("stage-2 AGHQ interior point: -f=%.8f  max|grad|=%.2e  d/dlogr3=%.2e\n", -fA(θA0),
        maximum(abs, gfd), gfd[ir + b])
up = [1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]; down = [1.4, 1.3]
rows = Tuple{Float64,Float64,Float64}[]
for path in (up, down)
    θA, θL = copy(θA0), copy(θL0)
    for v in path
        sA = copy(θA); sA[ir + b] = v; sL = copy(θL); sL[ir + b] = v
        θA, rA = reopt_fixed(fA, sA, [ir + b]); θL, rL = reopt_fixed(fL, sL, [ir + b])
        push!(rows, (v, -Optim.minimum(rL), -Optim.minimum(rA)))
        @printf("logr3=%.2f Laplace %.7f (conv=%s) AGHQ9 %.7f (conv=%s |g|=%.1e) [%.0fs]\n", v,
                -Optim.minimum(rL), Optim.converged(rL), -Optim.minimum(rA), Optim.converged(rA),
                Optim.g_residual(rA), time() - t0)
    end
end
push!(rows, (θA0[ir + b], -fL(θA0), -fA(θA0)))  # AGHQ value at stage-2 point (Laplace not profiled there)
println("\nsorted (logr3, Laplace-profile, AGHQ9-profile); the 1.517 row is the stage-2 point, Laplace column not profiled:")
for r in sort(rows); @printf("%.3f  %.7f  %.7f\n", r...); end
serialize(joinpath(OUT, "fine_profile46.jls"), sort(rows))
@printf("total %.1fs\n", time() - t0)
