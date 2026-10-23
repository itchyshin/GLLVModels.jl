#!/usr/bin/env julia
# D-220 proof: one paired R↔Julia Gaussian latent() cell (first-order).
# Fixture shared with test/parity/test_gaussian_parity.jl and core070_second_order gaussian cell.
#
# Usage:
#   julia --project=. tools/d220_paired_gaussian_cell.jl [output.json]

using GLLVModels, RCall, Random, LinearAlgebra, Statistics, Dates

const ORACLE_REF = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
const OUT_DEFAULT = joinpath(@__DIR__, "..", "docs", "dev-log", "core070",
    "d220-paired-gaussian-cell-receipt-2026-09-05.json")

include(joinpath(@__DIR__, "core070_second_order", "common.jl"))

function run_d220_cell()
    seed = 42
    Random.seed!(seed)
    p, K, n = 5, 2, 80
    Λ_true = parity_loadings_p5k2()
    σ_true = 0.7
    η = randn(K, n)
    y = Λ_true * η + σ_true * randn(p, n)
    y .-= mean(y; dims = 2)

    t0 = time()
    jl_fit = fit_gaussian_gllvm(y; K = K)
    wall_jl = time() - t0

    jl_logL = jl_fit.logLik
    jl_σ = jl_fit.pars.σ_eps
    jl_Σ = jl_fit.pars.Λ * jl_fit.pars.Λ' + jl_σ^2 * I(p)

    r = r_fit_se(y, K; family = :gaussian)
    # first-order path uses se=FALSE equivalent — pull logLik/objective from r_fit_se's fit
    # For sigma/Sigma_y we need extractors; use inline R block matching test_gaussian_parity.jl
    _require_gllvmtmb!()
    @rput y K p n
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(y)
    )
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long, unit = "site", trait = "trait",
        family = gaussian(),
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    r_logL  <- as.numeric(stats::logLik(fit_r))
    r_sigma <- as.numeric(fit_r$report$sigma_eps)
    Sig_sh  <- extract_Sigma(fit_r, level = "unit", part = "shared")$Sigma
    Sigma_y_r <- Sig_sh + diag(r_sigma^2, p)
    r_obj   <- as.numeric(fit_r$opt$objective)
    r_conv  <- identical(as.integer(fit_r$opt$convergence), 0L)
    pkg_ver <- as.character(packageVersion("gllvmTMB"))
    """
    r_logL = rcopy(Float64, R"r_logL")
    r_sigma = rcopy(Float64, R"r_sigma")
    r_Σ = rcopy(Matrix{Float64}, R"Sigma_y_r")
    r_obj = rcopy(Float64, R"r_obj")
    r_conv = rcopy(Bool, R"r_conv")
    r_pkg = rcopy(String, R"pkg_ver")

    fro_Σ = sqrt(sum((jl_Σ .- r_Σ) .^ 2))

    Dict{String,Any}(
        "receipt_id" => "d220-paired-gaussian-latent-bare",
        "receipt_date" => string(today()),
        "lane" => "cursor/m2-foundation-day1-20260905",
        "d220_proof" => true,
        "claim_boundary" => "parity evidence only — NOT true-parity claim",
        "oracle_ref" => ORACLE_REF,
        "oracle_note" => "Frozen gllvmTMB 0.7.0 export surface; live R package version recorded separately.",
        "r_gllvmTMB_version" => r_pkg,
        "model" => "Gaussian identity + ordinary latent(0+trait|site,d=K,unique=FALSE)",
        "dgp_source" => "test/parity/test_gaussian_parity.jl (seed=42,p=5,K=2,n=80, trait-centred Y)",
        "p" => p, "K" => K, "n" => n, "seed" => seed,
        "jl_converged" => jl_fit.converged,
        "r_converged" => r_conv,
        "jl_logLik" => jl_logL,
        "r_logLik" => r_logL,
        "loglik_delta_jl_minus_r" => jl_logL - r_logL,
        "r_objective_neg_loglik" => -r_obj,
        "jl_sigma_eps" => jl_σ,
        "r_sigma_eps" => r_sigma,
        "sigma_eps_abs_delta" => abs(jl_σ - r_sigma),
        "sigma_y_frobenius_delta" => fro_Σ,
        "first_order_pass" => (
            jl_fit.converged && r_conv &&
            abs(jl_logL - r_logL) ≤ 1e-6 &&
            abs(jl_σ - r_sigma) / max(r_sigma, 1e-12) ≤ 1e-4 &&
            fro_Σ / sqrt(sum(r_Σ .^ 2)) ≤ 1e-4
        ),
        "wall_fit_julia_sec" => wall_jl,
        "wall_fit_r_sec" => r.wall_fit,
        "git_head" => try
            chomp(read(`git rev-parse HEAD`, String))
        catch
            "unknown"
        end,
    )
end

out_path = length(ARGS) ≥ 1 ? ARGS[1] : OUT_DEFAULT
mkpath(dirname(out_path))
result = run_d220_cell()
write_json(out_path, result)
status = get(result, "first_order_pass", false) ? "PASS" : "FAIL"
println("D220_GAUSSIAN_CELL $status")
for (k, v) in result
    println("  $k = $v")
end
exit(status == "PASS" ? 0 : 1)
