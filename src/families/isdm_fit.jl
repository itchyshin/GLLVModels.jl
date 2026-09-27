# iSDM fitter (docs/design/isdm-port-spec.md section 3.2): packed
# theta = [b_fix; pack_lambda(Λ)] (with latent(..., unique = TRUE), R's default,
# theta = [b_fix; pack_lambda(Λ); theta_diag_B]), Optim L-BFGS on the negative Laplace
# marginal with the one-step implicit gradient of isdm_grad.jl, warm-started
# from a per-row link-scale pseudodata regression.

"""
    IsdmFit

Result of [`fit_isdm_gllvm`](@ref), an integrated species-distribution model.

Fields:
- `b_fix::Vector{Float64}`, `b_names::Vector{String}`: fixed coefficients and
  their R-style names (the twin of gllvmTMB's `b_fix` and `X_fix_names`).
- `Λ::Matrix{Float64}`: `p x K` shared loadings (`K = 0` for a GLM fit). Only
  `Λ * Λ'` is identified.
- `zhat::Matrix{Float64}`: `K x n_units` conditional modes of the latent scores.
- `unique::Bool`: whether the fit carries R's default `latent(..., unique = TRUE)`
  per-trait unit-level unique variance.
- `theta_diag_B::Vector{Float64}`: with `unique`, the length-`p` log standard
  deviations of the unit-level unique effects (gllvmTMB's `theta_diag_B`; the
  trait covariance is `Λ * Λ' + Diagonal(exp.(2 .* theta_diag_B))`); empty
  otherwise.
- `s_B::Matrix{Float64}`: `p x n_units` conditional modes of the unique effects
  (gllvmTMB's `s_B`); zeros without `unique`.
- `eta::Vector{Float64}`: per-row linear predictor at the modes, including the
  offset (the twin of gllvmTMB's `fit\$report\$eta`).
- `loglik::Float64`, `converged::Bool`, `iterations::Int`: Laplace marginal at
  the optimum, the verdict (optimiser converged AND every cell's mode search
  converged), and optimiser iterations.
- `cell_converged::Vector{Bool}`: per-unit mode-search verdicts at the optimum.
- `table::IsdmTable`, `sources::IsdmSources`, `formula::Expr`: the inputs.
- `hessian_used::Symbol`: `:observed` (the log-determinant curvature).

Everything the fit reports is relative intensity: presence-only data cannot
identify absolute abundance, occupancy or detectability.
"""
struct IsdmFit
    b_fix::Vector{Float64}
    b_names::Vector{String}
    Λ::Matrix{Float64}
    zhat::Matrix{Float64}
    unique::Bool
    theta_diag_B::Vector{Float64}
    s_B::Matrix{Float64}
    eta::Vector{Float64}
    loglik::Float64
    converged::Bool
    cell_converged::Vector{Bool}
    iterations::Int
    table::IsdmTable
    sources::IsdmSources
    formula::Expr
    hessian_used::Symbol
end

Base.show(io::IO, f::IsdmFit) = print(io, "IsdmFit(loglik = ", round(f.loglik; digits = 4),
    ", K = ", size(f.Λ, 2), f.unique ? ", unique" : "", ", ", length(f.b_fix), " fixed coefficients, ",
    f.converged ? "converged" : "NOT CONVERGED", ")")

loglikelihood(f::IsdmFit) = f.loglik
coef(f::IsdmFit) = f.b_fix
nobs(f::IsdmFit) = length(f.eta)

const _ISDM_NOTICE_SHOWN = Ref(false)

function _isdm_experimental_notice()
    _ISDM_NOTICE_SHOWN[] && return nothing
    _ISDM_NOTICE_SHOWN[] = true
    @info "The integrated multi-source route is experimental. It combines presence-only " *
          "count streams with structured detection/non-detection data under one shared " *
          "ecological linear predictor. Everything it reports is relative intensity: " *
          "presence-only data cannot identify absolute abundance, occupancy, or " *
          "detectability, and this fit does not estimate them. Give every presence-only arm " *
          "its own reporting-rate term (an interaction with a source indicator). Check " *
          "convergence on every fit, and expect this interface to change."
    return nothing
end

# Per-row link-scale pseudodata: log(y + 0.5) on count rows, the cloglog of
# (y + 0.5) / 2 on detection rows, both less the offset.
function _isdm_pseudo_eta(table::IsdmTable)
    out = similar(table.y)
    @inbounds for i in eachindex(table.y)
        y = table.y[i]
        out[i] = (table.fid[i] == 2 ? log(y + 0.5) : linkfun(CLogLogLink(), (y + 0.5) / 2)) -
                 table.offset[i]
    end
    return out
end

function _isdm_start(table::IsdmTable, K::Int)
    ỹ = _isdm_pseudo_eta(table)
    b0 = table.X \ ỹ
    p = length(table.trait_levels); n = length(table.unit_levels)
    K == 0 && return b0, zeros(p, 0)
    r = ỹ .- table.X * b0
    R = zeros(p, n); cnt = zeros(p, n)
    @inbounds for i in eachindex(r)
        R[table.trait_id[i], table.unit_id[i]] += r[i]
        cnt[table.trait_id[i], table.unit_id[i]] += 1
    end
    R ./= max.(cnt, 1)
    F = svd(R)
    L = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    if K > 1                                   # rotate to the lower-triangular convention
        Q = Matrix(qr(permutedims(L[1:K, :])).Q)
        L = L * Q
        for k in 2:K, i in 1:(k - 1)
            L[i, k] = 0.0
        end
    end
    for k in 1:K                               # keep the start off the Λ = 0 saddle
        abs(L[k, k]) < 0.1 && (L[k, k] = 0.1)
    end
    return b0, L
end

"""
    fit_isdm_gllvm(formula::Expr, data; family::IsdmSources, trait = :trait,
                   unit = :cell_id, weights = nothing, n_trials = nothing, kwargs...) -> IsdmFit
    fit_isdm_gllvm(table::IsdmTable; K = table.K, b_init = nothing, Λ_init = nothing,
                   theta_diag_B_init = nothing, g_tol = 1e-6, iterations = 1000,
                   newton_maxiter = 100, newton_tol = 1e-9, gradient = :analytic) -> IsdmFit

Fit an integrated species-distribution model, the Julia twin of gllvmTMB's
public door `gllvmTMB(..., family = isdm_sources(...))` at pin P1
(non-spatial, Laplace). Several named data sources observe one ecological
linear predictor per (unit, trait), each under its own law: Poisson-log count
rows and Bernoulli-cloglog detection rows, with a known per-row support as an
offset. The loadings `Λ` and fixed coefficients are shared across sources.
With `latent(..., unique = TRUE)`, R's default, each (unit, trait) also carries
a unique effect `s_B(t, s) ~ N(0, exp(theta_diag_B[t])^2)`, shared by that
trait's rows in that unit across sources, so the between-unit trait covariance
is `Λ * Λ' + Diagonal(exp.(2 .* theta_diag_B))`; the unique effects are
integrated by Laplace jointly with the latent scores.

The formula is a quoted expression (StatsModels' `@formula` cannot parse the
keyword arguments of `latent()`); interactions are written with `&`:

```julia
fam = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))
fit = fit_isdm_gllvm(
    :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) +
      latent(0 + trait | cell_id, d = 1)),
    dat; family = fam, trait = :trait, unit = :cell_id)
predict(fit; type = :response)
```

A formula without `latent()` fits `K = 0`: a GLM through the same kernel.
`latent(..., unique = FALSE)` fits the loadings-only model. The unique variances
are identified only when there are enough traits: a one-factor model needs at
least three (in general `p >= 2K + 1`); with two traits `theta_diag_B` runs
toward the boundary (a unique SD near zero) in R and in Julia alike, and only
the log-likelihood is comparable.

The marginal is a per-cell Laplace approximation with the observed curvature of
the summed density; the gradient is the one-step implicit gradient with a
central finite-difference fallback wherever a cell's mode search fails
(`gradient = :finite` forces the fallback everywhere). `converged` requires the
optimiser to converge and every cell's mode search to converge at the optimum.

Everything the fit reports is relative intensity: presence-only data cannot
identify absolute abundance, occupancy or detectability.
"""
function fit_isdm_gllvm(formula::Expr, data; family::IsdmSources, trait::Symbol = :trait,
        unit::Symbol = :cell_id, weights = nothing, n_trials = nothing, kwargs...)
    table = isdm_table(formula, data; family = family, trait = trait, unit = unit,
                       weights = weights, n_trials = n_trials)
    return fit_isdm_gllvm(table; kwargs...)
end

function fit_isdm_gllvm(table::IsdmTable; K::Integer = table.K, b_init = nothing,
        Λ_init = nothing, theta_diag_B_init = nothing, g_tol::Real = 1e-6,
        iterations::Integer = 1000,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9,
        gradient::Symbol = :analytic)
    gradient in (:analytic, :finite) || throw(ArgumentError("gradient must be :analytic or :finite"))
    table.admitted && _isdm_experimental_notice()
    pX = size(table.X, 2); p = length(table.trait_levels)
    b0, L0 = _isdm_start(table, Int(K))
    b_init === nothing || (b0 = collect(float.(b_init)))
    Λ_init === nothing || (L0 = collect(float.(Λ_init)))
    uniq = table.unique && K > 0
    # R starts theta_diag_B at log(1) on non-Gaussian fits (R/fit-multi.R:5765-5869).
    θd0 = theta_diag_B_init === nothing ? zeros(p) : collect(float.(theta_diag_B_init))
    θ0 = uniq ? _isdm_pack(b0, L0, θd0) : _isdm_pack(b0, L0)
    unpackθ(θ) = uniq ? _isdm_unpack_unique(θ, pX, p, K) :
                      (_isdm_unpack(θ, pX, p, K)..., nothing, nothing)

    function negll(θ)
        b, Λ, θd, _ = unpackθ(θ)
        v = isdm_marginal_loglik_laplace(table, Λ, b; theta_diag_B = θd,
                                         maxiter = newton_maxiter, tol = newton_tol)
        return isfinite(v) ? -v : _NLL_SENTINEL
    end
    function agrad(θ)
        gradient === :finite && return nothing
        g = isdm_laplace_grad(table, θ; K = K, unique = uniq, maxiter = newton_maxiter,
                              tol = newton_tol)
        return g === nothing ? nothing : -g
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = _optimize_with_analytic(negll, agrad, θ0, ls,
                                  Optim.Options(g_tol = g_tol, iterations = iterations))
    θ̂ = Optim.minimizer(res)
    b̂, Λ̂, θ̂d, Λ̂a = unpackθ(θ̂)
    Λk = uniq ? Λ̂a : Λ̂                       # the kernel's loading matrix
    loglik, conv, iters = _fit_verdict(res)

    # Modes, per-row eta and per-cell verdicts at the optimum.
    eta0 = table.X * b̂ .+ table.offset
    nU = length(table.rows_by_unit)
    Ẑ = zeros(K, nU); S = zeros(p, nU); cell_ok = trues(nU)
    eta = copy(eta0)
    if K > 0
        for (u, rows) in enumerate(table.rows_by_unit)
            z, ok = _isdm_cell_mode_retry(view(table.y, rows), view(table.fid, rows),
                                          view(table.trait_id, rows), view(eta0, rows), Λk;
                                          maxiter = newton_maxiter, tol = newton_tol)
            Ẑ[:, u] = z[1:K]; cell_ok[u] = ok
            uniq && (S[:, u] = exp.(θ̂d) .* z[(K + 1):end])
            for o in rows
                eta[o] = _isdm_eta(eta0[o], Λk, table.trait_id[o], z)
            end
        end
    end
    conv = conv && all(cell_ok) && isfinite(loglik)
    return IsdmFit(b̂, copy(table.X_names), Matrix{Float64}(Λ̂), Ẑ, uniq,
                   uniq ? collect(Float64, θ̂d) : Float64[], S, eta, loglik, conv,
                   collect(cell_ok), iters, table, table.sources, table.formula, :observed)
end
