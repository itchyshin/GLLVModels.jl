# AGHQ evaluator for the NB2 per-trait-dispersion GLLVM (K = 2, log link), built on
# the SAME per-site pieces as GLLVModels.nb_grouped_marginal_loglik_laplace:
#   - mode: exact copy of the package's Fisher-scored Newton loop in
#     _nb_grouped_loglik_site (z0 = 0, maxiter 100, tol 1e-9, max|Δ| stopping rule);
#   - adaptation curvature: H = Λ' diag(W_obs) Λ + I with the package's observed NB2
#     weight (_nb_grouped_laplace_weight(:observed, ...));
#   - log-density: package _glm_logpdf(NegativeBinomial(r_t), μ, 1, y) with the
#     package clamps (_clamp_eta, _clamp_mu), plus the N(0, I_K) prior incl. constants;
#   - quadrature: package aghq_grid(d, k), aghq_adaptation(mode, H),
#     aghq_frozen_logintegral(logjoint, adaptation, grid).
# Independent cross-check: brute-force trapezoid on a wide fine grid (no GH).
using GLLVModels, LinearAlgebra, TOML, SHA, Printf, Serialization
const G = GLLVModels
const Optim = G.Optim
const OUT = "/tmp/claude-503/nb2-aghq"
const REPO = "/Users/z3437171/local-scratch/gllvm-nb2-finite-20260924"

function load_fixture(file)
    d = TOML.parsefile(joinpath(REPO, "test", "fixtures", file))
    Y = reshape(Int.(d["Y_column_major"]), d["p"], d["n"])
    @assert bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y))))) == d["data_sha256"]
    return Y
end

unpack(θ, p, K) = (θ[1:p], G.unpack_lambda(θ[(p + 1):(p + G.rr_theta_len(p, K))], p, K),
                   exp.(θ[(p + G.rr_theta_len(p, K) + 1):end]))

# The package's own negll closure (fit_nb_gllvm_grouped, group = 1:p, no mask/offset).
function make_negll(Y, K)
    p = size(Y, 1)
    function negll(θ)
        β, Λ, r = unpack(θ, p, K)
        v = try
            -G.nb_grouped_marginal_loglik_laplace(Y, Λ, β, r; hessian = :observed,
                                                  maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    return negll
end

const LINK = G.LogLink()

function site_mode(fams, y, Λ, β; maxiter = 100, tol = 1e-9)
    p, K = size(Λ)
    n = ones(Int, p)
    z = zeros(K)
    for _ in 1:maxiter
        η  = G._clamp_eta.(β .+ Λ * z)
        μ  = G._clamp_mu.(fams, G.linkinv.(Ref(LINK), η))
        me = G.mu_eta.(Ref(LINK), η)
        s  = G._glm_score.(fams, μ, n, me, y)
        W  = G._nb_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(LINK))
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = G._safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z  = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

function site_logjoint(fams, y, Λ, β, z)
    d = length(z)
    η = G._clamp_eta.(β .+ Λ * z)
    μ = G._clamp_mu.(fams, G.linkinv.(Ref(LINK), η))
    ℓ = 0.0
    @inbounds for t in eachindex(y)
        ℓ += G._glm_logpdf(fams[t], μ[t], 1, y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * d * log(2π)
end

function site_adaptation(fams, y, Λ, β)
    z = site_mode(fams, y, Λ, β)
    η  = G._clamp_eta.(β .+ Λ * z)
    μ  = G._clamp_mu.(fams, G.linkinv.(Ref(LINK), η))
    me = G.mu_eta.(Ref(LINK), η)
    Wo = G._nb_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(LINK))
    H  = Λ' * (Wo .* Λ) + I
    return G.aghq_adaptation(z, Matrix(H))
end

# Total AGHQ log-likelihood (per-site adaptation at the Laplace mode, observed Hessian).
function aghq_loglik(Y, Λ, β, r, k; per_site = false)
    p, n = size(Y)
    fams = [G.NegativeBinomial(float(r[t]), 0.5) for t in 1:p]
    grid = G.aghq_grid(size(Λ, 2), k)
    vals = zeros(n)
    for s in 1:n
        y = view(Y, :, s)
        ad = site_adaptation(fams, y, Λ, β)
        vals[s] = G.aghq_frozen_logintegral(z -> site_logjoint(fams, y, Λ, β, z), ad, grid)
    end
    return per_site ? vals : sum(vals)
end

# Brute-force trapezoid in adapted coordinates u (z = m + R^{-1} u), u ∈ [-L, L]^2.
function trapezoid_loglik(Y, Λ, β, r; L = 9.0, h = 0.1)
    p, n = size(Y)
    fams = [G.NegativeBinomial(float(r[t]), 0.5) for t in 1:p]
    us = collect(-L:h:L)
    acc = 0.0
    edge_max = -Inf   # max log-integrand share on the boundary of the box (truncation check)
    for s in 1:n
        y = view(Y, :, s)
        ad = site_adaptation(fams, y, Λ, β)
        lv = Matrix{Float64}(undef, length(us), length(us))
        for (j, u2) in enumerate(us), (i, u1) in enumerate(us)
            lv[i, j] = site_logjoint(fams, y, Λ, β, ad.mode + ad.inverse_root * [u1, u2])
        end
        m = maximum(lv)
        # trapezoid weights: interior 1, edges 1/2, corners 1/4 (all × h^2)
        w1 = ones(length(us)); w1[1] = 0.5; w1[end] = 0.5
        tot = 0.0
        for j in eachindex(us), i in eachindex(us)
            tot += w1[i] * w1[j] * exp(lv[i, j] - m)
        end
        acc += ad.logjac + m + log(tot * h^2)
        e = max(maximum(lv[1, :]), maximum(lv[end, :]), maximum(lv[:, 1]), maximum(lv[:, end]))
        edge_max = max(edge_max, e - m)
    end
    return acc, edge_max
end

const LS = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
reopt(f, θ0; g_tol = 1e-7, iterations = 800) =
    Optim.optimize(f, θ0, LS, Optim.Options(g_tol = g_tol, iterations = iterations);
                   autodiff = :finite)

θ_of(fit) = vcat(fit.β, G.pack_lambda(fit.Λ), log.(fit.r_group))

const KS = [1, 3, 5, 9, 15, 21]

function evaluate_point(label, Y, θ, K)
    p = size(Y, 1)
    β, Λ, r = unpack(θ, p, K)
    lap = G.nb_grouped_marginal_loglik_laplace(Y, Λ, β, r; hessian = :observed,
                                               maxiter = 100, tol = 1e-9)
    vals = Dict{Int,Float64}()
    for k in KS
        vals[k] = aghq_loglik(Y, Λ, β, r, k)
    end
    trap1, e1 = trapezoid_loglik(Y, Λ, β, r; L = 9.0, h = 0.1)
    trap2, e2 = trapezoid_loglik(Y, Λ, β, r; L = 10.0, h = 0.07)
    @printf("%-22s logr=%s\n", label, string(round.(log.(r); digits = 3)))
    @printf("   Laplace (package)       %.9f\n", lap)
    for k in KS
        @printf("   AGHQ k=%-2d               %.9f   (k=1 minus Laplace: %s)\n", k, vals[k],
                k == 1 ? @sprintf("%.3e", vals[k] - lap) : "-")
    end
    @printf("   trapezoid L=9  h=0.1    %.9f   (edge log-ratio %.1f)\n", trap1, e1)
    @printf("   trapezoid L=10 h=0.07   %.9f   (edge log-ratio %.1f)\n", trap2, e2)
    return (label = label, θ = θ, laplace = lap, aghq = vals, trap = (trap1, trap2),
            edge = (e1, e2))
end

results = Dict{String,Any}()
K = 2
t0 = time()

# ---------------- seed 46 ----------------
Y46 = load_fixture("nb2_restart_seed46.toml")
p = size(Y46, 1); rr = G.rr_theta_len(p, K)
negll46 = make_negll(Y46, K)
fit46 = fit_gllvm(Y46; family = G.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
@assert fit46.group == collect(1:p)
θ46_int = θ_of(fit46)
@printf("seed46 package fit loglik %.6f  negll check %.6f  r=%s  (%.1fs)\n", fit46.loglik,
        -negll46(θ46_int), string(round.(fit46.r_group; sigdigits = 4)), time() - t0)
θs = copy(θ46_int); θs[p + rr + 3] = 30.0
res46b = reopt(negll46, θs)
θ46_bd = Optim.minimizer(res46b)
@printf("seed46 boundary reopt (start logr3=30) loglik %.6f  logr=%s  conv=%s iters=%d (%.1fs)\n",
        -Optim.minimum(res46b), string(round.(θ46_bd[(p + rr + 1):end]; digits = 3)),
        Optim.converged(res46b), Optim.iterations(res46b), time() - t0)

# ---------------- seed 52 ----------------
Y52 = load_fixture("nb2_restart_seed52.toml")
negll52 = make_negll(Y52, K)
fit52 = fit_gllvm(Y52; family = G.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
θ52_bd = θ_of(fit52)
@printf("seed52 package fit loglik %.6f  r=%s  (%.1fs)\n", fit52.loglik,
        string(round.(fit52.r_group; sigdigits = 4)), time() - t0)
θs = copy(θ52_bd); θs[p + rr + 2] = 0.0
res52i = reopt(negll52, θs)
θ52_int = Optim.minimizer(res52i)
@printf("seed52 interior reopt (reset logr2=0) loglik %.6f  logr=%s conv=%s iters=%d (%.1fs)\n",
        -Optim.minimum(res52i), string(round.(θ52_int[(p + rr + 1):end]; digits = 3)),
        Optim.converged(res52i), Optim.iterations(res52i), time() - t0)

serialize(joinpath(OUT, "points.jls"), Dict(
    "theta46_interior" => θ46_int, "theta46_boundary" => θ46_bd,
    "theta52_boundary" => θ52_bd, "theta52_interior" => θ52_int))

println("\n==== AGHQ evaluation ====")
r46i = evaluate_point("seed46 interior", Y46, θ46_int, K)
r46b = evaluate_point("seed46 boundary", Y46, θ46_bd, K)
r52i = evaluate_point("seed52 interior", Y52, θ52_int, K)
r52b = evaluate_point("seed52 boundary", Y52, θ52_bd, K)

println("\n==== boundary minus interior (positive = boundary higher) ====")
for (lab, a, b) in (("seed46", r46i, r46b), ("seed52", r52i, r52b))
    @printf("%s Laplace %+.6f\n", lab, b.laplace - a.laplace)
    for k in KS
        @printf("%s AGHQ k=%-2d %+.6f\n", lab, k, b.aghq[k] - a.aghq[k])
    end
    @printf("%s trapezoid %+.6f\n", lab, b.trap[2] - a.trap[2])
end
serialize(joinpath(OUT, "eval_results.jls"), Dict("r46i" => r46i, "r46b" => r46b,
    "r52i" => r52i, "r52b" => r52b))
@printf("total %.1fs\n", time() - t0)
