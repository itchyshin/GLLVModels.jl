# gllvm-parity-tag: P1
#
# extract_latent_scores() twin test — reads recorded R (gllvmTMB 0.7.1, pin
# P1 9539352f66f2db2cc26b1c393e67212a359b60c9) values from
# test/fixtures/extract_latent_scores_p1/ (sha256-guarded below); runs no R
# and no RCall. R was fit with:
#
#   value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE)
#
# on a long-format (site, trait, value) table with exactly one row per
# (site, trait) cell — the degenerate single-tier case of gllvmTMB's
# `gllvmTMB_multi` class (no `unit_obs` replication), which is exactly the
# ordinary p-trait/species x n-site GLLVM this package fits directly as a
# p x n matrix. `unique = FALSE` removes R's default per-trait Psi_B
# companion (Sigma = Lambda Lambda' + Psi) so the fitted model is the same
# homoscedastic-residual factor model this package's `fit_gaussian_gllvm`
# assumes (see docs/dev-log/p1-export-recon.md for the recon and the
# "R differences" note in `extract_latent_scores`'s docstring).
using Test
using GLLVModels
using SHA

const _ELS_FIXDIR = joinpath(@__DIR__, "fixtures", "extract_latent_scores_p1")

const _ELS_SHA256 = Dict(
    "Y_gauss.csv"         => "2362c7e7472c978cd1e746904e2c20d4d5fb6fdb236ffad698c4bbfb8a11341d",
    "Y_pois.csv"          => "0f5b241dc0fc0eaa76bb78e8592845146d6d53c110f082c496689c0ea9960436",
    "z_gauss_unit.csv"    => "378ee2d6508f4edf2ba835d2e94cb5258bdbdf7d43f649ff4824efd8bf251d88",
    "z_pois_unit.csv"     => "add182d306e2c653a5315ae26156ab8bd070b51587b8fb032115c0278a662ab0",
    "Lambda_gauss_hat.csv"=> "98c4305d0f52b3fe26841f826e58d25276c6cf67ae58df02e3446ef9b5d78bc0",
    "Lambda_pois_hat.csv" => "acf73fb8c52a5ea1dcc556c8dfb5e746721fb4f414a660419f513fd02a588949",
    "beta_gauss_hat.txt"  => "00faefbadc41dd12cca661a3fbd7331a90bbefc690b224878fd66a10c461179a",
    "beta_pois_hat.txt"   => "1b941089b6ae488c75b26fc308288aecd39090e259eadb92fc6964d23dea412c",
    "loglik_gauss.txt"    => "16744c87ecdd8a28d9a93cc8fa7d9916bc2e90f374e1b3544ecce720edc8aaee",
    "loglik_pois.txt"     => "bdaf6225a745b9cfd8c7c55ee0d14a12071781a1898b34f4e52c6a7daa5a6ffa",
    "sigma_eps_hat.txt"   => "8c28c39d3be580254744eba48426deb1a83d323bff8e78092fc4c4c10d0b9597",
)

function _els_verify_and_path(name::AbstractString)
    path = joinpath(_ELS_FIXDIR, name)
    isfile(path) || error("missing extract_latent_scores fixture: $path")
    got = bytes2hex(sha256(read(path)))
    got == _ELS_SHA256[name] || error(
        "fixture $name sha256 mismatch (expected $(_ELS_SHA256[name]), got $got); " *
        "the R-recorded fixture has drifted from what this test expects")
    return path
end

# Minimal dependency-free CSV reader (no DelimitedFiles dep in the main
# Project.toml — this file is run under `--project=.`): a header row of
# quoted column names, then comma-separated Float64 rows.
function _els_read_csv(path::AbstractString)
    lines = readlines(path)
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end]]
    return permutedims(reduce(hcat, rows))  # nrows x ncols
end

_els_read_vector(path::AbstractString) = parse.(Float64, readlines(path))
_els_read_scalar(path::AbstractString) = parse(Float64, only(readlines(path)))

@testset "extract_latent_scores (P1 twin)" begin
    Y_gauss_sites = _els_read_csv(_els_verify_and_path("Y_gauss.csv"))   # n x p
    Y_pois_sites  = _els_read_csv(_els_verify_and_path("Y_pois.csv"))    # n x p
    z_r_gauss     = _els_read_csv(_els_verify_and_path("z_gauss_unit.csv"))    # n x K
    z_r_pois      = _els_read_csv(_els_verify_and_path("z_pois_unit.csv"))     # n x K
    Lambda_r_gauss = _els_read_csv(_els_verify_and_path("Lambda_gauss_hat.csv")) # p x K
    Lambda_r_pois  = _els_read_csv(_els_verify_and_path("Lambda_pois_hat.csv"))  # p x K
    beta_r_gauss  = _els_read_vector(_els_verify_and_path("beta_gauss_hat.txt"))
    beta_r_pois   = _els_read_vector(_els_verify_and_path("beta_pois_hat.txt"))
    loglik_r_gauss = _els_read_scalar(_els_verify_and_path("loglik_gauss.txt"))
    loglik_r_pois  = _els_read_scalar(_els_verify_and_path("loglik_pois.txt"))
    sigma_eps_r    = _els_read_scalar(_els_verify_and_path("sigma_eps_hat.txt"))

    Y_gauss = permutedims(Y_gauss_sites)      # p x n
    Y_pois  = Int.(permutedims(Y_pois_sites)) # p x n
    p, n = size(Y_gauss)
    K = size(z_r_gauss, 2)
    @test size(Y_pois) == (p, n)
    @test size(z_r_pois) == (n, K)

    # Per-trait dummy design recovering R's `0 + trait` fixed intercept
    # (fit_gaussian_gllvm's default X=nothing is a zero-mean model; R's
    # formula always fits one intercept per trait).
    X = zeros(p, n, p)
    for i in 1:p, s in 1:n
        X[i, s, i] = 1.0
    end

    @testset "Gaussian: own optimum" begin
        fit = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
        @test fit.converged
        @test isapprox(fit.logLik, loglik_r_gauss; atol = 1e-2)
        @test isapprox(fit.pars.β, beta_r_gauss; atol = 1e-4)
        @test isapprox(fit.pars.σ_eps, sigma_eps_r; atol = 1e-4)

        z = extract_latent_scores(fit, Y_gauss; level = :unit, X = X)
        @test size(z) == (n, K)
        # extract_latent_scores(level=:unit) is exactly getLV(component=:innovation, rotate=false)
        @test z == getLV(fit, Y_gauss; component = :innovation, rotate = false, X = X)

        # Rotation/sign-invariant comparison: the fitted linear-predictor
        # contribution Lambda * z' is identified even though (Lambda, z)
        # individually are identified only up to an orthogonal rotation.
        Lambda_j = getLoadings(fit; rotate = false)
        LZt_j = Lambda_j * z'
        LZt_r = Lambda_r_gauss * z_r_gauss'
        @test isapprox(LZt_j, LZt_r; atol = 1e-3)

        @test extract_latent_scores(fit, Y_gauss; level = :unit_obs) === nothing
    end

    @testset "Gaussian: at R's fitted parameters" begin
        # Isolates the *definition* of extract_latent_scores from optimiser
        # differences: plug R's fitted Lambda/beta/sigma_eps directly into
        # this package's posterior-mean formula and compare to R's own
        # extract_latent_scores() output at those same parameters.
        fit0 = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
        fit_atR = GLLVModels.GllvmFit(fit0.model,
            merge(fit0.pars, (Λ = Lambda_r_gauss, β = beta_r_gauss, σ_eps = sigma_eps_r)),
            fit0.logLik, fit0.n_iter, fit0.converged, fit0.optim_result, fit0.cputime,
            fit0.integration)
        z_atR = extract_latent_scores(fit_atR, Y_gauss; level = :unit, X = X)
        @test isapprox(z_atR, z_r_gauss; atol = 1e-8)
    end

    @testset "Poisson: own optimum" begin
        fit = fit_poisson_gllvm(Y_pois; K = K)
        @test fit.converged
        @test isapprox(fit.loglik, loglik_r_pois; atol = 1e-2)
        @test isapprox(fit.β, beta_r_pois; atol = 1e-3)

        z = extract_latent_scores(fit, Y_pois; level = :unit)
        @test size(z) == (n, K)
        @test z == getLV(fit, Y_pois; component = :innovation, rotate = false)

        Lambda_j = getLoadings(fit; rotate = false)
        LZt_j = Lambda_j * z'
        LZt_r = Lambda_r_pois * z_r_pois'
        @test isapprox(LZt_j, LZt_r; atol = 1e-2)

        @test extract_latent_scores(fit, Y_pois; level = :unit_obs) === nothing
    end

    @testset "Poisson: at R's fitted parameters" begin
        fit0 = fit_poisson_gllvm(Y_pois; K = K)
        fit_atR = GLLVModels.PoissonFit(beta_r_pois, Lambda_r_pois, fit0.link, fit0.loglik,
            fit0.converged, fit0.iterations, fit0.alpha_lv, fit0.theta_packed, fit0.hessian,
            fit0.integration)
        z_atR = extract_latent_scores(fit_atR, Y_pois; level = :unit)
        @test isapprox(z_atR, z_r_pois; atol = 1e-6)
    end

    @testset "level validation and default fallback" begin
        fit = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
        @test_throws ArgumentError extract_latent_scores(fit, Y_gauss; level = :bogus, X = X)
        # Mirrors R's extract_latent_scores.default abort for an unsupported type.
        @test_throws ArgumentError extract_latent_scores([1, 2, 3])
        @test_throws ArgumentError extract_latent_scores("not a fit")
    end
end
