# cells.jl -- cell registry for the core070 second-order batch (2026-09-03).
# include()d by run_cell.jl. Each cell: build Y (+X, +N) per its DGP source,
# fit Julia, fit R (se=TRUE), extract the three contract quantities (SE,
# fixed-effect vcov block, Wald CI endpoints), return a receipt Dict.
#
# Shared computation for every cell (kept inline per-cell rather than over-
# abstracted, since each family's fitter/CI accessor signature differs and a
# live run is exactly where a wrong abstraction would hide a bug).

using Random
using LinearAlgebra
using ForwardDiff
using Distributions: TDist, NegativeBinomial

# ---------------------------------------------------------------------------
# Cell dispatcher
# ---------------------------------------------------------------------------
function run_one_cell(cell_id::AbstractString)
    if cell_id == "gaussian"
        return cell_gaussian()
    elseif cell_id == "poisson"
        return cell_poisson()
    elseif cell_id == "binomial_logit"
        return cell_binomial(:logit, 43)
    elseif cell_id == "binomial_probit"
        return cell_binomial(:probit, 143)
    elseif cell_id == "binomial_cloglog"
        return cell_binomial(:cloglog, 144)
    elseif cell_id == "beta_logit"
        return cell_beta()
    elseif cell_id == "nb2_log"
        return cell_nb2()
    elseif cell_id == "gamma_log"
        return cell_gamma_grouped()
    elseif cell_id == "nb1_log"
        return cell_nb1_grouped()
    elseif cell_id == "betabinomial_logit"
        return cell_betabinomial_grouped()
    elseif cell_id == "betabinomial_shared"
        return cell_betabinomial_shared()
    elseif cell_id == "gaussian_x"
        return cell_gaussian_x()
    elseif cell_id == "binomial_x"
        return cell_binomial_x()
    elseif cell_id == "poisson_x"
        return cell_poisson_x()
    elseif cell_id == "nb2_x"
        return cell_nb2_x()
    elseif cell_id == "beta_x"
        return cell_beta_x()
    elseif cell_id == "gamma_x"
        return cell_gamma_x()
    elseif cell_id == "nb1_x"
        return cell_nb1_x()
    elseif cell_id == "betabinomial_x"
        return cell_betabinomial_x()
    elseif cell_id == "poisson_speciesx"
        return cell_poisson_speciesx()
    elseif cell_id == "binomial_speciesx"
        return cell_binomial_speciesx()
    elseif cell_id == "delta_lognormal"
        return cell_delta_lognormal()
    elseif cell_id == "delta_gamma"
        return cell_delta_gamma()
    elseif cell_id == "tweedie_fixed"
        return cell_tweedie_fixed()
    elseif cell_id == "tweedie_shared"
        return cell_tweedie_shared()
    elseif cell_id == "tweedie_species"
        return cell_tweedie_species()
    elseif cell_id == "lognormal"
        return cell_lognormal()
    elseif cell_id == "ordinal_pertrait_probit"
        return cell_ordinal_pertrait_probit()
    elseif cell_id == "truncated_poisson"
        return cell_truncated_poisson()
    elseif cell_id == "truncated_nbinom2"
        return cell_truncated_nbinom2()
    elseif cell_id == "multinomial_fe"
        return cell_multinomial_fe()
    elseif cell_id == "studentt_fixed_nu"
        return cell_studentt_fixed_nu()
    else
        error("unknown cell id: $cell_id")
    end
end

# ===========================================================================
# no-X cells
# ===========================================================================

function cell_gaussian()
    seed = 42
    Random.seed!(seed)
    p, K, n = 5, 2, 80
    Λ_true = parity_loadings_p5k2()
    σ_true = 0.7
    η = randn(K, n)
    y = Λ_true * η + σ_true * randn(p, n)
    y .-= sum(y; dims = 2) ./ n

    t0 = time()
    fit = fit_gaussian_gllvm(y; K = K)
    wall_fit = time() - t0

    terms, kinds = GLLVModels._confint_all_term_names(fit)
    θ̂ = fit.pars.θ_packed
    nll = GLLVModels._confint_reconstruct_nll(fit, y, nothing, nothing)
    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch
        nothing
    end
    Σ = nothing
    pdh = false
    if H !== nothing && all(isfinite, H)
        try
            Σ = inv(Symmetric((H .+ H') ./ 2))
            pdh = all(x -> isfinite(x) && x > 0, diag(Σ))
        catch
            Σ = nothing
        end
    end
    se = Σ === nothing ? fill(NaN, length(θ̂)) : sqrt.(max.(diag(Σ), 0.0))

    r = r_fit_se(y, K; family = :gaussian)

    d = Dict{String,Any}()
    d["cell_id"] = "gaussian"
    d["dgp_source"] = "test/parity/test_gaussian_parity.jl (seed=42,p=5,K=2,n=80); reused verbatim from second-order-prerun-2026-09-02.md"
    d["family"] = "Gaussian (no dispersion-selector: exact marginal, ForwardDiff Hessian)"
    d["p"] = p; d["K"] = K; d["n"] = n; d["seed"] = seed
    d["hessian_selector"] = "exact_gaussian_forwarddiff"
    d["hessian_selector_disputed"] = false
    d["matched_coordinates"] = false
    d["jl_converged"] = fit.converged
    d["jl_logLik"] = fit.logLik
    d["r_converged"] = r.converged
    d["r_logLik"] = r.logLik
    d["loglik_delta_jl_minus_r"] = fit.logLik - r.logLik
    d["r_has_sd_report"] = r.has_sd
    d["wall_fit_julia_sec"] = wall_fit
    d["wall_fit_r_sec"] = r.wall_fit
    d["pd_hessian_native"] = pdh
    d["boundary_terms"] = String[]
    d["pd_hessian_r"] = r.pd_hessian
    d["r_condition_number"] = r.r_condition_number
    d["derived_quantity"] = nothing
    d["r_objective"] = r.objective
    d["note"] = "Gaussian has no beta (Y pre-centred); compared quantity is sigma_eps + the (sigma_eps,Lambda) vcov block, per the pre-run convention. NOT compared against a b_fix block (none exists on the Julia side)."

    if !r.has_sd
        d["skip_reason"] = "R se=TRUE produced no sd_report"
        d["se_max_relative_delta"] = nothing
        d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing
        d["native_condition_number"] = nothing
        return d
    end

    sig_idx_jl = findfirst(==("sigma_eps"), terms)
    sig_idx_r = findfirst(==("log_sigma_eps"), r.names)
    if sig_idx_jl === nothing || sig_idx_r === nothing || Σ === nothing
        d["skip_reason"] = "sigma_eps term not found on one side, or Hessian inversion failed"
        d["se_max_relative_delta"] = nothing
        d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing
        d["native_condition_number"] = nothing
        return d
    end
    se_log_jl = se[sig_idx_jl]
    se_log_r = sqrt(r.cov_fixed[sig_idx_r, sig_idx_r])
    d["se_max_abs_delta"] = abs(se_log_jl - se_log_r)
    d["se_max_relative_delta"] = abs(se_log_jl - se_log_r) / max(se_log_r, 1e-12)

    z = 1.959963984540054
    est_log_jl = θ̂[sig_idx_jl]
    est_log_r = r.par_fixed[sig_idx_r]
    lo_jl = exp(est_log_jl - z * se_log_jl); hi_jl = exp(est_log_jl + z * se_log_jl)
    lo_r = exp(est_log_r - z * se_log_r); hi_r = exp(est_log_r + z * se_log_r)
    d["ci_endpoint_max_delta"] = max(abs(lo_jl - lo_r), abs(hi_jl - hi_r))
    d["vcov_frobenius_relative_delta"] = nothing  # single scalar; Frobenius not meaningful for a 1x1 block here
    d["native_condition_number"] = try; cond(Symmetric((H .+ H') ./ 2)); catch; NaN; end
    return d
end

function cell_poisson()
    seed = 44
    Random.seed!(seed)
    p, K, n = 5, 2, 60
    β = log.([3.0, 5.0, 2.0, 4.0, 3.5])
    Λ = 0.45 .* parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [_rand_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_poisson_gllvm(Y; K = K, hessian = :observed)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald)
    ad = GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :poisson)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("poisson", "test/parity/test_poisson_parity.jl (seed=44,p=5,K=2,n=60)",
        "Poisson-log", "observed (explicit override; package default :fisher for LogLink)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_binomial(link::Symbol, seed::Int)
    Random.seed!(seed)
    p, K, n = 5, 2, 60
    β = [-0.5, 0.0, 0.5, -0.2, 0.3]
    Λ = parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = if link == :logit
        [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]
    elseif link == :probit
        [rand() < GLLVModels._ord_F(η[t, s], ProbitLink()) ? 1 : 0 for t in 1:p, s in 1:n]
    elseif link == :cloglog
        [rand() < (1 - exp(-exp(clamp(η[t, s], -8.0, 8.0)))) ? 1 : 0 for t in 1:p, s in 1:n]
    else
        error("unsupported link $link")
    end

    jl_link = link == :logit ? LogitLink() : link == :probit ? ProbitLink() : CLogLogLink()
    hessian_override = link == :logit ? :observed : GLLVModels._default_hessian(GLLVModels.Binomial(), jl_link)
    t0 = time()
    fit = fit_binomial_gllvm(Y; K = K, link = jl_link, hessian = hessian_override)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald)
    ad = GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :binomial, binomial_link = link)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    disputed = false  # §2 ACCEPTED (A) 2026-09-15 — cloglog default :observed ratified
    sel_str = link == :logit ? "observed (explicit override; package default :fisher for LogitLink)" :
              "$(hessian_override) (family default for $(link))"

    return _assemble("binomial_$(link)", "test/parity/test_binomial_parity.jl DGP pattern (K=2,n=60,p=5); link=$(link) is a fresh draw at seed=$(seed) (same beta/Lambda, different link inverse)",
        "Binomial-$(link)", sel_str, disputed,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_beta()
    seed = 45
    Random.seed!(seed)
    p, K, n = 5, 1, 60
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    φ_true = 12.0
    Λ = 0.15 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [begin
        μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
        _rand_beta_jonk(μ * φ_true, (1 - μ) * φ_true)
    end for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gllvm(Y; family = GLLVModels.Beta(), K = K, g_tol = 1e-7, iterations = 800)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y; objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Y, K; family = :beta)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("beta_logit", "test/parity/test_beta_parity.jl (seed=45,p=5,K=1,n=60)",
        "Beta-logit", "$(fit.hessian) (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_nb2()
    seed = 45
    Random.seed!(seed)
    p, K, n = 5, 2, 80
    β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
    r_true = 4.0
    Λ = 0.30 .* parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -8.0, 8.0))
        Y[t, s] = _rand_nb2_ms(μ, r_true)
    end

    t0 = time()
    fit = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = K, g_tol = 1e-7, iterations = 800)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald)
    ad = GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :negbinomial)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("nb2_log", "test/parity/test_negbin_parity.jl DGP pattern (seed=45,p=5,K=2,n=80); reused verbatim from second-order-prerun-2026-09-02.md",
        "NB2-log", "$(fit.hessian) (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_gamma_grouped()
    seed = 54
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([2.0, 2.5, 1.8, 2.2, 2.1])
    α_true = 2.5
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [begin
        μ = exp(clamp(η[t, s], -4.0, 4.0))
        rand(Gamma(α_true, μ / α_true)) + 1e-6
    end for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gamma_gllvm_grouped(Y; K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y; objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Y, K; family = :gamma)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("gamma_log", "test/parity/test_nox_dispersion_parity.jl (seed=54,p=5,K=1,n=120, per-trait alpha)",
        "Gamma-log (per-trait shape)", "observed (grouped default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_nb1_grouped()
    seed = 55
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([1.8, 2.2, 1.6, 2.0, 1.9])
    φ_true = 0.85
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -3.5, 3.5))
        Y[t, s] = rand(NegativeBinomial(μ / φ_true, 1 / (1 + φ_true)))
    end

    t0 = time()
    fit = fit_nb1_gllvm_grouped(Y; K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald)
    ad = GLLVModels._family_ci(fit, Float64.(Y); objective = :laplace)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :nb1)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("nb1_log", "test/parity/test_nox_dispersion_parity.jl (seed=55,p=5,K=1,n=120, per-trait phi)",
        "NB1-log (per-trait phi)", "observed (grouped default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_betabinomial_grouped()
    seed = 56
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    φ_true = 8.0
    N = fill(8, p, n)
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
        psucc = clamp(rand(Beta(μ * φ_true, (1 - μ) * φ_true)), 1e-6, 1 - 1e-6)
        Y[t, s] = rand(Binomial(N[t, s], psucc))
    end

    t0 = time()
    fit = fit_beta_binomial_gllvm_grouped(Y; K = K, N = N, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, N = N)
    ad = GLLVModels._family_ci(fit, Float64.(Y); N = N)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :betabinomial, N = N)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("betabinomial_logit", "test/parity/test_nox_dispersion_parity.jl (seed=56,p=5,K=1,n=120, per-trait phi, N=8)",
        "BetaBinomial-logit (per-trait phi)", "observed (grouped default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_betabinomial_shared()
    seed = 57
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    φ_true = 8.0
    N = fill(8, p, n)
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
        psucc = clamp(rand(Beta(μ * φ_true, (1 - μ) * φ_true)), 1e-6, 1 - 1e-6)
        Y[t, s] = rand(Binomial(N[t, s], psucc))
    end

    t0 = time()
    fit = fit_beta_binomial_gllvm(Y; K = K, N = N)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, N = N)
    ad = GLLVModels._family_ci(fit, Float64.(Y); N = N)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Float64.(Y), K; family = :betabinomial, N = N)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("betabinomial_shared",
        "test/parity/test_nox_dispersion_parity.jl (seed=57,p=5,K=1,n=120, SHARED phi, N=8)",
        "BetaBinomial-logit (shared phi; Julia scalar vs R per-trait — beta[] block only)",
        "fisher (no observed-Hessian variant; G0 lock)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "R gllvmTMB::betabinomial() has no public shared-phi knob (per-trait log_phi_betabinom always); " *
        "compared quantity is the beta[] fixed-effect block only, mirroring cell_beta(). " *
        "phi itself is not paired — different estimand (Julia: 1 free scalar; R: p free per-trait)."
    d["parameterisation_gap"] = false
    return d
end

# ===========================================================================
# shared site-X cells
# ===========================================================================

function cell_gaussian_x()
    seed = 420
    Random.seed!(seed)
    p, K, n = 5, 2, 30
    Λ = parity_loadings_p5k2()
    γ_true = 0.55
    x = randn(n); x .-= sum(x) / n
    X = parity_site_design(x, p)
    Z = randn(K, n)
    y = γ_true .* x' .+ Λ * Z .+ 0.7 .* randn(p, n)
    y .-= sum(y; dims = 2) ./ n

    t0 = time()
    fit = fit_gaussian_gllvm(y; K = K, X = X)
    wall_fit = time() - t0

    terms, kinds = GLLVModels._confint_all_term_names(fit)
    θ̂ = fit.pars.θ_packed
    nll = GLLVModels._confint_reconstruct_nll(fit, y, X, nothing)
    H = try; ForwardDiff.hessian(nll, θ̂); catch; nothing; end
    Σ = nothing; pdh = false
    if H !== nothing && all(isfinite, H)
        try; Σ = inv(Symmetric((H .+ H') ./ 2)); pdh = all(v -> isfinite(v) && v > 0, diag(Σ)); catch; Σ = nothing; end
    end

    r = r_fit_se_x(y, x, K; family = :gaussian)

    d = Dict{String,Any}()
    d["cell_id"] = "gaussian_x"
    d["dgp_source"] = "test/parity/test_x_covariate_parity.jl (seed=420,p=5,K=2,n=30,q=1 shared X)"
    d["family"] = "Gaussian + shared X"
    d["p"] = p; d["K"] = K; d["n"] = n; d["seed"] = seed
    d["hessian_selector"] = "exact_gaussian_forwarddiff"
    d["hessian_selector_disputed"] = false
    d["matched_coordinates"] = false
    d["jl_converged"] = fit.converged
    d["jl_logLik"] = fit.logLik
    d["r_converged"] = r.converged
    d["r_logLik"] = r.logLik
    d["loglik_delta_jl_minus_r"] = fit.logLik - r.logLik
    d["r_has_sd_report"] = r.has_sd
    d["wall_fit_julia_sec"] = wall_fit
    d["wall_fit_r_sec"] = r.wall_fit
    d["pd_hessian_native"] = pdh
    d["boundary_terms"] = String[]
    d["pd_hessian_r"] = r.pd_hessian
    d["r_condition_number"] = r.r_condition_number
    d["derived_quantity"] = nothing
    d["r_objective"] = r.objective
    d["note"] = "Gaussian has no per-trait intercept on the Julia side (Y pre-centred); the only rotation-safe fixed effect is the shared X slope gamma, if the Gaussian+X packing exposes one as a named term (see gamma_idx below)."

    if !r.has_sd
        d["skip_reason"] = "R se=TRUE produced no sd_report"
        d["se_max_relative_delta"] = nothing; d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing; d["native_condition_number"] = nothing
        return d
    end
    # fit_gaussian_gllvm(...; X=) has NO per-trait intercept (Y is pre-centred);
    # its sole "beta[1]" term IS the shared-X slope gamma (verified empirically:
    # theta_packed[1] tracks the true gamma=0.55 used to generate y).
    gamma_idx_jl = findfirst(t -> startswith(t, "beta["), terms)
    r_beta_idx = findall(==("b_fix"), r.names)
    if gamma_idx_jl === nothing || Σ === nothing
        d["skip_reason"] = "no shared-X gamma term found in Julia's Gaussian+X term names (packing does not expose one distinctly), or Hessian inversion failed -- skipping the fixed-effect SE/vcov/CI comparison for this cell (logLik agreement is still reported above)"
        d["se_max_relative_delta"] = nothing; d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing; d["native_condition_number"] = nothing
        return d
    end
    se_jl = sqrt(max(Σ[gamma_idx_jl, gamma_idx_jl], 0.0))
    # R's b_fix here is [intercepts... , x] (p+1 entries); the shared slope is the last one.
    slope_idx_r = length(r_beta_idx) >= p + 1 ? r_beta_idx[p + 1] : nothing
    if slope_idx_r === nothing
        d["skip_reason"] = "R b_fix block shorter than p+1; cannot locate the shared X slope"
        d["se_max_relative_delta"] = nothing; d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing; d["native_condition_number"] = nothing
        return d
    end
    se_r = sqrt(r.cov_fixed[slope_idx_r, slope_idx_r])
    d["se_max_abs_delta"] = abs(se_jl - se_r)
    d["se_max_relative_delta"] = abs(se_jl - se_r) / max(se_r, 1e-12)
    z = 1.959963984540054
    est_jl = θ̂[gamma_idx_jl]; est_r = r.par_fixed[slope_idx_r]
    d["ci_endpoint_max_delta"] = max(abs((est_jl - z * se_jl) - (est_r - z * se_r)),
                                      abs((est_jl + z * se_jl) - (est_r + z * se_r)))
    d["vcov_frobenius_relative_delta"] = nothing
    d["native_condition_number"] = try; cond(Symmetric((H .+ H') ./ 2)); catch; NaN; end
    return d
end

function cell_binomial_x()
    seed = 431
    Random.seed!(seed)
    p, K, n = 5, 2, 80
    β = [-0.3, 0.0, 0.3, -0.15, 0.2]
    Λ = 0.35 .* parity_loadings_p5k2()
    γ_true = 0.5
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gllvm_cov(Y; family = GLLVModels.Binomial(), X = X, K = K)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Float64.(Y); X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Float64.(Y), x, K; family = :binomial)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("binomial_x", "test/parity/test_x_covariate_parity.jl (seed=431,p=5,K=2,n=80,q=1 shared X)",
        "Binomial-logit + shared X", "not exposed by fit_gllvm_cov (no hessian kwarg; fixed internal curvature convention)", true,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_poisson_x()
    seed = 422
    Random.seed!(seed)
    p, K, n = 5, 2, 30
    β = log.([2.5, 4.0, 2.0, 3.5, 3.0])
    Λ = 0.4 .* parity_loadings_p5k2()
    γ_true = 0.6
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = [_rand_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gllvm_cov(Y; family = GLLVModels.Poisson(), X = X, K = K)
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Float64.(Y); X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Float64.(Y), x, K; family = :poisson)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("poisson_x", "test/parity/test_x_covariate_parity.jl (seed=422,p=5,K=2,n=30,q=1 shared X)",
        "Poisson-log + shared X", "not exposed by fit_gllvm_cov (no hessian kwarg; fixed internal curvature convention)", true,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_nb2_x()
    seed = 45
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
    r_true = 1.5
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    γ_true = 0.4
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = [_rand_nb2_ms(exp(clamp(η[t, s], -8.0, 8.0)), r_true) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Float64.(Y); X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Float64.(Y), x, K; family = :negbinomial)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("nb2_x", "test/parity/test_x_covariate_parity.jl (seed=45,p=5,K=1,n=120,q=1 shared X, per-trait r)",
        "NB2-log + shared X (per-trait r)", "observed (grouped-cov default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_beta_x()
    seed = 45
    Random.seed!(seed)
    p, K, n = 5, 1, 80
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    φ_true = 8.0
    Λ = 0.1 .* parity_loadings_p5k2()[:, 1:K]
    γ_true = 0.35
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = [begin
        μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
        _rand_beta_jonk(μ * φ_true, (1 - μ) * φ_true)
    end for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_beta_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Y; X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Y, x, K; family = :beta)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("beta_x", "test/parity/test_x_covariate_parity.jl (seed=45,p=5,K=1,n=80,q=1 shared X, per-trait phi)",
        "Beta-logit + shared X (per-trait phi)", "observed (grouped-cov default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_gamma_x()
    seed = 46
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([2.0, 2.5, 1.8, 2.2, 2.1])
    α_true = 2.5
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    γ_true = 0.4
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = [begin
        μ = exp(clamp(η[t, s], -4.0, 4.0))
        _rand_gamma_ms(α_true, μ / α_true) + 1e-6
    end for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gamma_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Y; X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Y, x, K; family = :gamma)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("gamma_x", "test/parity/test_x_covariate_parity.jl (seed=46,p=5,K=1,n=120,q=1 shared X, per-trait alpha)",
        "Gamma-log + shared X (per-trait alpha)", "observed (grouped-cov default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_nb1_x()
    seed = 48
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([1.8, 2.2, 1.6, 2.0, 1.9])
    φ_true = 0.85
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    γ_true = 0.4
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -3.5, 3.5))
        Y[t, s] = rand(NegativeBinomial(μ / φ_true, 1 / (1 + φ_true)))
    end

    t0 = time()
    fit = fit_nb1_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, X = X)
    ad = GLLVModels._family_ci(fit, Float64.(Y); X = X)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Float64.(Y), x, K; family = :nb1)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("nb1_x", "test/parity/test_x_covariate_parity.jl (seed=48,p=5,K=1,n=120,q=1 shared X, per-trait phi)",
        "NB1-log + shared X (per-trait phi)", "observed (grouped-cov default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

function cell_betabinomial_x()
    seed = 49
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = [0.30, -0.20, 0.25, -0.15, 0.05]
    φ_true = 8.0
    N = fill(8, p, n)
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    γ_true = 0.35
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ γ_true .* x' .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
        psucc = clamp(_rand_beta_jonk(μ * φ_true, (1 - μ) * φ_true), 1e-6, 1 - 1e-6)
        Y[t, s] = count(_ -> rand() < psucc, 1:N[t, s])
    end

    t0 = time()
    fit = fit_beta_binomial_gllvm_grouped_cov(Y; X = X, K = K, N = N, group = collect(1:p))
    wall_fit = time() - t0
    ci = confint(fit, Float64.(Y); method = :wald, X = X, N = N)
    ad = GLLVModels._family_ci(fit, Float64.(Y); X = X, N = N)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_x(Float64.(Y), x, K; family = :betabinomial, N = N)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "gamma["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble("betabinomial_x", "test/parity/test_x_covariate_parity.jl (seed=49,p=5,K=1,n=120,q=1 shared X, per-trait phi, N=8)",
        "BetaBinomial-logit + shared X (per-trait phi)", "observed (grouped-cov default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
end

# ===========================================================================
# species-specific X ((0+trait):x) cells -- confint_speciescov, no full Σ
# ===========================================================================

function cell_poisson_speciesx()
    seed = 48
    Random.seed!(seed)
    p, K, n = 5, 1, 80
    β = log.([2.5, 4.0, 2.0, 3.5, 3.0])
    B = reshape([0.45, 0.25, -0.20, 0.35, 0.15], p, 1)
    Λ = 0.30 .* parity_loadings_p5k2()[:, 1:K]
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ B[:, 1] .* x' .+ Λ * Z
    Y = [_rand_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gllvm_speciescov(Y; family = GLLVModels.Poisson(), X = X, K = K)
    wall_fit = time() - t0
    ci = confint_speciescov(fit, Float64.(Y), X)

    r = r_fit_se_species_x(Float64.(Y), x, K; family = :poisson)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "B["), ci.term)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble_speciescov("poisson_speciesx", "test/parity/test_species_x_parity.jl (seed=48,p=5,K=1,n=80,q=1 per-trait B)",
        "Poisson-log species-XB", "not exposed by fit_gllvm_speciescov (no hessian kwarg)", true,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, beta_idx_jl, r_beta_idx)
end

function cell_binomial_speciesx()
    seed = 49
    Random.seed!(seed)
    p, K, n = 5, 1, 80
    β = [-0.3, 0.0, 0.3, -0.15, 0.2]
    B = reshape([0.45, 0.25, -0.20, 0.35, 0.15], p, 1)
    Λ = 0.30 .* parity_loadings_p5k2()[:, 1:K]
    x = randn(n)
    X = parity_site_design(x, p)
    Z = randn(K, n)
    η = β .+ B[:, 1] .* x' .+ Λ * Z
    Y = [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_gllvm_speciescov(Y; family = GLLVModels.Binomial(), X = X, K = K)
    wall_fit = time() - t0
    ci = confint_speciescov(fit, Float64.(Y), X)

    r = r_fit_se_species_x(Float64.(Y), x, K; family = :binomial)
    beta_idx_jl = findall(t -> startswith(t, "beta[") || startswith(t, "B["), ci.term)
    r_beta_idx = findall(==("b_fix"), r.names)

    return _assemble_speciescov("binomial_speciesx", "test/parity/test_species_x_parity.jl (seed=49,p=5,K=1,n=80,q=1 per-trait B, Bernoulli)",
        "Binomial-logit species-XB", "not exposed by fit_gllvm_speciescov (no hessian kwarg)", true,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, beta_idx_jl, r_beta_idx)
end

function cell_delta_lognormal()
    seed = 61
    Random.seed!(seed)
    p, K, n = 5, 1, 130
    β_true = [0.2, -0.1, 0.3, 0.0, -0.2]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    σ_true = 0.5
    Z = randn(K, n)
    η = β_true .+ Λ_true * Z
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        π = 1 / (1 + exp(-η[t, s]))
        rand() < π && (Y[t, s] = exp(η[t, s] + σ_true * randn()))
    end

    t0 = time()
    # DRAFT Option A scaffold: Julia twin cell uses per-trait σ (matches R).
    # Decision doc stays PENDING until paste `accept delta dispersion A`.
    fit = fit_delta_lognormal_gllvm(Y; K = K, predictor = :shared, disp_group = :species,
                                    hessian = :observed, iterations = 500)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y; hessian = :observed)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_delta(Y, K; family = :delta_lognormal)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("delta_lognormal",
        "test/parity/test_delta_lognormal_parity.jl (seed=61,p=5,K=1,n=130,predictor=:shared,disp_group=:species)",
        "Delta-lognormal (shared η, per-trait σ; twin fid 12)", "observed (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only. Julia disp_group=:species matches R per-trait sigma_lognormal_delta (Option A scaffold; decision PENDING until paste)."
    d["parameterisation_gap"] = false
    return d
end

function cell_delta_gamma()
    seed = 62
    Random.seed!(seed)
    p, K, n = 5, 1, 130
    β_true = [0.2, -0.1, 0.3, 0.0, -0.2]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    α_true = 4.0
    Z = randn(K, n)
    η = β_true .+ Λ_true * Z
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        π = 1 / (1 + exp(-η[t, s]))
        if rand() < π
            μ = exp(η[t, s])
            Y[t, s] = rand(Gamma(α_true, μ / α_true))
        end
    end

    t0 = time()
    # DRAFT Option A scaffold: Julia twin cell uses per-trait α (matches R).
    # Decision doc stays PENDING until paste `accept delta dispersion A`.
    fit = fit_delta_gamma_gllvm(Y; K = K, predictor = :shared, disp_group = :species,
                                hessian = :observed, iterations = 500)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y; hessian = :observed)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_delta(Y, K; family = :delta_gamma)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("delta_gamma",
        "test/parity/test_delta_gamma_parity.jl (seed=62,p=5,K=1,n=130,predictor=:shared,disp_group=:species)",
        "Delta-Gamma (shared η, per-trait α; twin fid 13)", "observed (family default; specialised Wc)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only. Julia disp_group=:species matches R per-trait phi_gamma_delta (Option A scaffold; decision PENDING until paste)."
    d["parameterisation_gap"] = false
    return d
end

function _tp_rand_trunc_poisson(λ::Float64)
    while true
        k = _rand_poisson(clamp(λ, 0.0, 1e6))
        k >= 1 && return k
    end
end

function _r_trait_intercept_idx(r, n_beta::Int)
    idx = findall(==("b_fix"), r.names)
    if length(idx) == n_beta
        return idx
    end
    if length(r.par_fixed) == n_beta
        return collect(1:n_beta)
    end
    return Int[]
end

function cell_lognormal()
    seed = 52
    Random.seed!(seed)
    p, K, n = 5, 2, 60
    β = log.([3.0, 5.0, 2.0, 4.0, 3.5])
    Λ = 0.45 .* parity_loadings_p5k2()
    σ = 0.5
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = exp.(η .+ σ .* randn(p, n))

    t0 = time()
    fit = fit_lognormal_gllvm(Y; K = K)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Y, K; family = :lognormal)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("lognormal",
        "test/parity/test_lognormal_parity.jl (seed=52,p=5,K=2,n=60)",
        "Lognormal (shared σ; β[] block only)", "exact marginal (closed-form)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only; shared log_sigma_eps vs Julia single σ is not paired."
    d["parameterisation_gap"] = false
    return d
end

function cell_ordinal_pertrait_probit()
    seed = 46
    Random.seed!(seed)
    p, K, n, C = 5, 1, 60, 3
    β = [0.30, -0.20, 0.15, -0.10, 0.05]
    Λ = reshape([0.8, 0.5, 0.3, -0.2, 0.1], p, K)
    τ = [0.0, 0.75]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n, t in 1:p
        u = rand()
        Y[t, s] = u < GLLVModels._ord_F(τ[1] - η[t, s], ProbitLink()) ? 1 :
                  u < GLLVModels._ord_F(τ[2] - η[t, s], ProbitLink()) ? 2 : 3
    end

    t0 = time()
    fit = fit_ordinal_gllvm_pertrait(Y; K = K, link = ProbitLink(), iterations = 1_000)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_ordinal_probit(Y, K)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("ordinal_pertrait_probit",
        "test/parity/test_ordinal_probit_parity.jl (seed=46,p=5,K=1,n=60,C=3,probit)",
        "Ordinal-probit per-trait τ (β[] block only)", "observed (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only; free cutpoints tau[t,c] are not paired in this toy cell."
    d["parameterisation_gap"] = false
    return d
end

function cell_truncated_poisson()
    seed = 53
    Random.seed!(seed)
    p, K, n = 5, 2, 60
    β = log.([3.0, 5.0, 2.0, 4.0, 3.5])
    Λ = 0.45 .* parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [_tp_rand_trunc_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_truncated_poisson_gllvm(Y; K = K)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Y, K; family = :truncated_poisson)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("truncated_poisson",
        "test/parity/test_truncated_poisson_parity.jl (seed=53,p=5,K=2,n=60)",
        "Truncated-Poisson-log (no dispersion)", "observed (Laplace default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["parameterisation_gap"] = false
    return d
end

function cell_truncated_nbinom2()
    seed = 58
    Random.seed!(seed)
    p, K, n = 5, 1, 120
    β = log.([4.0, 5.0, 3.5, 4.5, 4.0])
    r_true = 4.0
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -3.0, 3.5))
        while true
            v = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
            if v >= 1
                Y[t, s] = v
                break
            end
        end
    end

    t0 = time()
    fit = fit_truncated_nbinom2_gllvm(Y; K = K, hessian = :observed)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y; hessian = :observed)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se(Y, K; family = :truncated_nbinom2)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("truncated_nbinom2",
        "test/parity/test_truncated_nbinom2_parity.jl (seed=58,p=5,K=1,n=120; Julia shared r)",
        "Truncated-NB2-log (β[] block only)", "observed (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Julia TruncatedNegBin2Fit uses one shared r; R uses per-trait log_phi_truncnb2 — β[] block paired only."
    d["parameterisation_gap"] = false
    return d
end

function cell_multinomial_fe()
    seed = 57
    Random.seed!(seed)
    ncat, n = 4, 400
    β_true = [0.0, 0.6, -0.4, 0.25]
    w = exp.(β_true)
    prob = w ./ sum(w)
    cum = cumsum(prob)
    y = [findfirst(>=(rand()), cum) for _ in 1:n]

    t0 = time()
    fit = fit_multinomial_gllvm(y; n_categories = ncat)
    wall_fit = time() - t0
    Y = reshape(collect(y), 1, n)
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_multinomial(y, ncat)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("multinomial_fe",
        "test/parity/test_multinomial_parity.jl (seed=57,ncat=4,n=400, FE-only)",
        "Multinomial FE softmax (contrast beta[k], k≥2)", "exact FE (no Laplace)", false,
        1, 0, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "No latent term on either side; K=0 on Julia metadata. Compared block is softmax contrast intercepts."
    d["parameterisation_gap"] = false
    return d
end

function cell_studentt_fixed_nu()
    seed = 71
    Random.seed!(seed)
    p, K, n = 5, 1, 130
    β_true = [0.2, -0.1, 0.3, 0.0, -0.2]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    σ_true = 0.7
    ν_true = 4.0
    Z = randn(K, n)
    η = β_true .+ Λ_true * Z
    Y = [η[t, s] + σ_true * rand(TDist(ν_true)) for t in 1:p, s in 1:n]

    t0 = time()
    fit = fit_studentt_gllvm(Y; K = K, nu = ν_true, disp_group = :species, iterations = 400)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_student(Y, K; df_fixed = ν_true)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = _r_trait_intercept_idx(r, length(beta_idx_jl))

    d = _assemble("studentt_fixed_nu",
        "test/parity/test_studentt_parity.jl (seed=71,p=5,K=1,n=130,disp_group=:species,ν=4 fixed)",
        "Student-t identity (fixed ν; per-trait σ; β[] block)", "observed (family default)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Julia disp_group=:species matches R per-trait log_sigma_student; ν pinned on both sides."
    d["parameterisation_gap"] = false
    return d
end

function cell_tweedie_fixed()
    # Fixed common power — honest twin of Julia `power=p_true` (parity seed 82).
    seed = 82
    Random.seed!(seed)
    p, K, n = 5, 1, 150
    β_true = [0.5, -0.2, 0.3, -0.4, 0.1]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    φ_true = [0.8, 1.0, 1.2, 0.9, 1.1]
    p_true = 1.5
    Z = randn(K, n)
    μ = exp.(β_true .+ Λ_true * Z)
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ_true[t], p_true, Random.default_rng())
    end

    t0 = time()
    fit = fit_tweedie_gllvm_grouped(Y; K = K, power = p_true, hessian = :observed, iterations = 400)
    wall_fit = time() - t0
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_tweedie(Y, K; p_fixed = p_true)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("tweedie_fixed",
        "test/parity/test_tweedie_parity.jl (seed=82,p=5,K=1,n=150,power=1.5 fixed)",
        "Tweedie-log (fixed power; per-trait φ via group=1:p)", "observed (family default; §2 A)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix; power fixed on both sides. Julia CI profiles φ_g with power plug-in (TweedieFit contract)."
    d["parameterisation_gap"] = false
    return d
end

function _tweedie_parity_fixture_y(seed::Integer = 82)
    Random.seed!(seed)
    p, K, n = 5, 1, 150
    β_true = [0.5, -0.2, 0.3, -0.4, 0.1]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    φ_true = [0.8, 1.0, 1.2, 0.9, 1.1]
    p_true = 1.5
    Z = randn(K, n)
    μ = exp.(β_true .+ Λ_true * Z)
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = GLLVModels._tweedie_sample(μ[t, s], φ_true[t], p_true, Random.default_rng())
    end
    return Y, p, K, n, seed
end

function cell_tweedie_shared()
    # Estimated shared power — option A: compare b_fix / beta only (parity seed 82).
    Y, p, K, n, seed = _tweedie_parity_fixture_y(82)
    t0 = time()
    fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :shared, hessian = :observed, iterations = 400)
    wall_fit = time() - t0
    fit isa GLLVModels.TweedieGroupedFit && !fit.power_fixed ||
        error("tweedie_shared: expected TweedieGroupedFit with free shared power")
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_tweedie_shared(Y, K)
    r.reference_constraint_adapter && r.n_power_free == 1 ||
        error("tweedie_shared: R path must use the shared-power adapter with n_power_free=1")
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("tweedie_shared",
        "test/parity/test_tweedie_parity.jl (seed=82,p=5,K=1,n=150,power_group=:shared)",
        "Tweedie-log (estimated shared power; per-trait φ)", "observed (family default; §2 A)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only (option A). Julia Wald plug-ins shared power; R sdreport includes logit_p_tweedie in the full fixed block — not compared."
    d["parameterisation_gap"] = false
    d["reference_constraint_adapter"] = r.reference_constraint_adapter
    d["n_power_free"] = r.n_power_free
    return d
end

function cell_tweedie_species()
    Y, p, K, n, seed = _tweedie_parity_fixture_y(82)
    t0 = time()
    fit = fit_tweedie_gllvm_grouped(Y; K = K, power_group = :species, hessian = :observed, iterations = 400)
    wall_fit = time() - t0
    fit isa GLLVModels.TweediePerTraitPowerFit ||
        error("tweedie_species: expected TweediePerTraitPowerFit with free per-trait power")
    ci = confint(fit, Y; method = :wald)
    ad = GLLVModels._family_ci(fit, Y)
    H = GLLVModels._fd_hessian(ad.nll, ad.θ)
    Σ = _safe_inv(H)

    r = r_fit_se_tweedie_species(Y, K)
    beta_idx_jl = findall(t -> startswith(t, "beta["), ad.names)
    r_beta_idx = findall(==("b_fix"), r.names)

    d = _assemble("tweedie_species",
        "test/parity/test_tweedie_parity.jl (seed=82,p=5,K=1,n=150,power_group=:species)",
        "Tweedie-log (per-trait estimated power; per-trait φ)", "observed (family default; §2 A)", false,
        p, K, n, seed, fit.converged, fit.loglik, wall_fit, r, ci, Σ, ad.names, beta_idx_jl, r_beta_idx)
    d["note"] = "Compared quantity is trait intercept block b_fix only (option A). Julia Wald plug-ins per-trait power; R par.fixed includes logit_p_tweedie — not compared."
    d["parameterisation_gap"] = false
    d["reference_constraint_adapter"] = r.reference_constraint_adapter
    d["n_power_free"] = r.n_power_free
    return d
end

# ===========================================================================
# shared assembly helpers
# ===========================================================================

function _safe_inv(H::AbstractMatrix)
    all(isfinite, H) || return nothing
    try
        return inv(Symmetric((H .+ H') ./ 2))
    catch
        return nothing
    end
end

function _assemble(cell_id, dgp_source, family_display, hessian_selector, disputed::Bool,
        p::Int, K::Int, n::Int, seed::Int,
        jl_conv::Bool, jl_logL::Float64, wall_fit_jl::Float64,
        r::NamedTuple, ci_pub, Σ::Union{Nothing,AbstractMatrix}, names_native::Vector{String},
        beta_idx_jl::Vector{Int}, r_beta_idx::Vector{Int})

    d = Dict{String,Any}()
    d["cell_id"] = cell_id
    d["dgp_source"] = dgp_source
    d["family"] = family_display
    d["p"] = p; d["K"] = K; d["n"] = n; d["seed"] = seed
    d["hessian_selector"] = hessian_selector
    d["hessian_selector_disputed"] = disputed
    d["matched_coordinates"] = false
    d["jl_converged"] = jl_conv
    d["jl_logLik"] = jl_logL
    d["r_converged"] = r.converged
    d["r_logLik"] = r.logLik
    d["loglik_delta_jl_minus_r"] = jl_logL - r.logLik
    d["r_has_sd_report"] = r.has_sd
    d["wall_fit_julia_sec"] = wall_fit_jl
    d["wall_fit_r_sec"] = r.wall_fit
    d["pd_hessian_native"] = ci_pub.pd_hessian
    d["boundary_terms"] = hasproperty(ci_pub, :boundary_terms) ? collect(ci_pub.boundary_terms) : String[]
    d["pd_hessian_r"] = r.pd_hessian
    d["r_condition_number"] = r.r_condition_number
    d["derived_quantity"] = nothing
    d["r_objective"] = r.objective

    if !r.has_sd
        d["skip_reason"] = "R se=TRUE produced no sd_report"
        d["se_max_relative_delta"] = nothing; d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing; d["native_condition_number"] = nothing
        return d
    end
    n_beta = length(beta_idx_jl)
    if length(r_beta_idx) != n_beta
        d["skip_reason"] = "fixed-effect block length mismatch: Julia=$n_beta vs R b_fix=$(length(r_beta_idx))"
        d["se_max_relative_delta"] = nothing; d["vcov_frobenius_relative_delta"] = nothing
        d["ci_endpoint_max_delta"] = nothing; d["native_condition_number"] = nothing
        return d
    end
    pf_r = r.par_fixed[r_beta_idx]
    cv_r = r.cov_fixed[r_beta_idx, r_beta_idx]
    se_r = sqrt.(max.(diag(cv_r), 0.0))
    z = 1.959963984540054

    # SE + Wald-endpoint comparison uses the PUBLIC confint()'s own per-term
    # SE/lower/upper -- these reflect T14 F1's boundary-aware partial
    # degradation (docs/dev-log/core070/t14-nb2-wald-nan-diagnosis.md): a
    # cell with one boundary dispersion parameter (e.g. NB2 r[t] at a
    # Poisson-limit boundary) can still carry FINITE SE for the
    # well-identified beta/gamma/B terms even though `pd_hessian` is false
    # for the full joint Hessian. Using the private re-inverted Sigma here
    # (all-or-nothing) would have thrown away exactly the cells this
    # comparison most needs to see. The full vcov Frobenius block still
    # needs Sigma (off-diagonals are not exposed by confint()), so it is
    # skipped (not this whole cell) when Sigma is unavailable.
    se_jl = collect(ci_pub.se)[beta_idx_jl]
    se_delta = abs.(se_jl .- se_r)
    se_rel = se_delta ./ max.(se_r, 1e-12)
    finite_se = isfinite.(se_delta) .& isfinite.(se_rel)
    if any(finite_se)
        d["se_max_abs_delta"] = maximum(se_delta[finite_se])
        d["se_max_relative_delta"] = maximum(se_rel[finite_se])
    else
        d["se_max_abs_delta"] = nothing
        d["se_max_relative_delta"] = nothing
    end
    d["se_n_finite_of_total"] = "$(count(finite_se))/$(n_beta)"

    lower_jl = collect(ci_pub.lower)[beta_idx_jl]
    upper_jl = collect(ci_pub.upper)[beta_idx_jl]
    lower_r = pf_r .- z .* se_r
    upper_r = pf_r .+ z .* se_r
    finite_deltas = filter(isfinite, vcat(abs.(lower_jl .- lower_r), abs.(upper_jl .- upper_r)))
    d["ci_endpoint_max_delta"] = isempty(finite_deltas) ? nothing : maximum(finite_deltas)

    if Σ === nothing
        d["vcov_frobenius_relative_delta"] = nothing
        d["native_condition_number"] = nothing
        d["vcov_skip_reason"] = "native fixed-effect covariance unavailable (FD Hessian inversion failed on the full joint Hessian); SE/CI above still computed via confint()'s T14 boundary-degraded route"
        return d
    end

    Σ_beta = Σ[beta_idx_jl, beta_idx_jl]
    fro_num = sqrt(sum((Σ_beta .- cv_r) .^ 2))
    fro_den = sqrt(sum(cv_r .^ 2))
    d["vcov_frobenius_relative_delta"] = fro_den > 0 ? fro_num / fro_den : NaN

    d["native_condition_number"] = try
        cond(Symmetric((Σ_beta .+ Σ_beta') ./ 2))
    catch
        NaN
    end
    return d
end

function _assemble_speciescov(cell_id, dgp_source, family_display, hessian_selector, disputed::Bool,
        p::Int, K::Int, n::Int, seed::Int,
        jl_conv::Bool, jl_logL::Float64, wall_fit_jl::Float64,
        r::NamedTuple, ci, beta_idx_jl::Vector{Int}, r_beta_idx::Vector{Int})

    d = Dict{String,Any}()
    d["cell_id"] = cell_id
    d["dgp_source"] = dgp_source
    d["family"] = family_display
    d["p"] = p; d["K"] = K; d["n"] = n; d["seed"] = seed
    d["hessian_selector"] = hessian_selector
    d["hessian_selector_disputed"] = disputed
    d["matched_coordinates"] = false
    d["jl_converged"] = jl_conv
    d["jl_logLik"] = jl_logL
    d["r_converged"] = r.converged
    d["r_logLik"] = r.logLik
    d["loglik_delta_jl_minus_r"] = jl_logL - r.logLik
    d["r_has_sd_report"] = r.has_sd
    d["wall_fit_julia_sec"] = wall_fit_jl
    d["wall_fit_r_sec"] = r.wall_fit
    d["pd_hessian_native"] = nothing
    d["boundary_terms"] = String[]
    d["pd_hessian_r"] = r.pd_hessian
    d["r_condition_number"] = r.r_condition_number
    d["derived_quantity"] = nothing
    d["r_objective"] = r.objective
    d["vcov_frobenius_relative_delta"] = nothing
    d["note"] = "species-XB cells use confint_speciescov, which has no private _family_ci accessor; only per-term SE and Wald endpoints are compared (no full vcov Frobenius)."

    if !r.has_sd
        d["skip_reason"] = "R se=TRUE produced no sd_report"
        d["se_max_relative_delta"] = nothing; d["ci_endpoint_max_delta"] = nothing
        d["native_condition_number"] = nothing
        return d
    end
    n_beta = length(beta_idx_jl)
    if length(r_beta_idx) != n_beta
        d["skip_reason"] = "fixed-effect block length mismatch: Julia=$n_beta vs R b_fix=$(length(r_beta_idx))"
        d["se_max_relative_delta"] = nothing; d["ci_endpoint_max_delta"] = nothing
        d["native_condition_number"] = nothing
        return d
    end
    se_jl = collect(ci.se)[beta_idx_jl]
    pf_r = r.par_fixed[r_beta_idx]
    cv_r = r.cov_fixed[r_beta_idx, r_beta_idx]
    se_r = sqrt.(max.(diag(cv_r), 0.0))
    se_delta = abs.(se_jl .- se_r)
    se_rel = se_delta ./ max.(se_r, 1e-12)
    d["se_max_abs_delta"] = maximum(se_delta)
    d["se_max_relative_delta"] = maximum(se_rel)

    z = 1.959963984540054
    lower_jl = collect(ci.lower)[beta_idx_jl]
    upper_jl = collect(ci.upper)[beta_idx_jl]
    lower_r = pf_r .- z .* se_r
    upper_r = pf_r .+ z .* se_r
    finite_deltas = filter(isfinite, vcat(abs.(lower_jl .- lower_r), abs.(upper_jl .- upper_r)))
    d["ci_endpoint_max_delta"] = isempty(finite_deltas) ? NaN : maximum(finite_deltas)
    d["native_condition_number"] = NaN
    return d
end
