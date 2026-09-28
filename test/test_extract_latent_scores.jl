# gllvm-parity-tag: P1
#
# extract_latent_scores() twin test — reads recorded R (gllvmTMB 0.7.1, pin
# P1 9539352f66f2db2cc26b1c393e67212a359b60c9) values from
# test/fixtures/extract_latent_scores_p1/ (sha256-guarded below; the R script
# and exact install call that produced them live alongside the fixtures as
# generate_fixture.R). Runs no R and no RCall. Gaussian and Poisson were fit
# with:
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
# assumes. NB2 (per-trait dispersion, `family = nbinom2()`) uses the same
# formula on its own, larger dataset (n_sites = 60; per-trait dispersion +
# rank-2 loadings needs more data than the 15-site Gaussian/Poisson fixture
# to stay numerically well-posed). See the "R differences" note in
# `extract_latent_scores`'s docstring (src/extract_latent_scores.jl) for the
# full semantic comparison.
using Test
using GLLVModels
using SHA
using Distributions: Poisson

const _ELS_FIXDIR = joinpath(@__DIR__, "fixtures", "extract_latent_scores_p1")

const _ELS_SHA256 = Dict(
    "Y_gauss.csv"          => "2362c7e7472c978cd1e746904e2c20d4d5fb6fdb236ffad698c4bbfb8a11341d",
    "Y_pois.csv"           => "0f5b241dc0fc0eaa76bb78e8592845146d6d53c110f082c496689c0ea9960436",
    "Y_nb2.csv"            => "194de3673547a751a68c8f2761d33d9ed735cf4e68a84e409080ccb3f108221a",
    "z_gauss_unit.csv"     => "378ee2d6508f4edf2ba835d2e94cb5258bdbdf7d43f649ff4824efd8bf251d88",
    "z_pois_unit.csv"      => "add182d306e2c653a5315ae26156ab8bd070b51587b8fb032115c0278a662ab0",
    "z_nb2_unit.csv"       => "ee62d25bf0badf3d431bd88a28c631906ad654b18d964f9e84889d0b06020d4c",
    "Lambda_gauss_hat.csv" => "98c4305d0f52b3fe26841f826e58d25276c6cf67ae58df02e3446ef9b5d78bc0",
    "Lambda_pois_hat.csv"  => "acf73fb8c52a5ea1dcc556c8dfb5e746721fb4f414a660419f513fd02a588949",
    "Lambda_nb2_hat.csv"   => "4b344e362d13cfaa6ab81a22db4870c1a4fe5162312a55d4cd5c7676f49cd813",
    "beta_gauss_hat.txt"   => "00faefbadc41dd12cca661a3fbd7331a90bbefc690b224878fd66a10c461179a",
    "beta_pois_hat.txt"    => "1b941089b6ae488c75b26fc308288aecd39090e259eadb92fc6964d23dea412c",
    "beta_nb2_hat.txt"     => "58df42488c0c240815adf0b4d459a771003d019650bed2f9a645e66524c627e2",
    "phi_nb2_hat.txt"      => "cfcef573cbf18da51018be8e7bee45bf9b145cea5523f02f4d305e9bfbada496",
    "loglik_gauss.txt"     => "16744c87ecdd8a28d9a93cc8fa7d9916bc2e90f374e1b3544ecce720edc8aaee",
    "loglik_pois.txt"      => "bdaf6225a745b9cfd8c7c55ee0d14a12071781a1898b34f4e52c6a7daa5a6ffa",
    "loglik_nb2.txt"       => "a17ac5719d4d83c1bbaed3d9736d789bddcb84a6bdc2ca767bd114912b8e2df0",
    "sigma_eps_hat.txt"    => "8c28c39d3be580254744eba48426deb1a83d323bff8e78092fc4c4c10d0b9597",
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
        @test isapprox(z_atR, z_r_gauss; atol = 1e-8)  # measured 5.6e-15 (machine precision)
    end

    @testset "matches extract_ordination on a no-X, no-X_lv fit" begin
        # extract_ordination(fit, Y; rotate=false).sites is
        # getLV(fit, Y; rotate=false) at getLV's *default* component (:total,
        # not :innovation — see the dispatch note in
        # src/extract_latent_scores.jl), and neither extract_ordination nor
        # the underlying ordination() forwards a fixed-effect X at all, so
        # this identity only holds for a fit with no X (zero fixed-effect
        # mean) and no X_lv, where :total == :innovation exactly (R's own
        # documented extract_ordination(component = "innovation") identity).
        fit0 = fit_gaussian_gllvm(Y_gauss; K = K)  # X = nothing: zero-mean model
        z0 = extract_latent_scores(fit0, Y_gauss; level = :unit)
        ord0 = extract_ordination(fit0, Y_gauss; rotate = false)
        @test ord0.sites == z0
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
        @test isapprox(z_atR, z_r_pois; atol = 1e-6)  # measured 7.2e-11 (Laplace-Newton tolerance)
    end

    @testset "NB2 (NBGroupedFit, per-trait dispersion): at R's fitted parameters" begin
        # Own (larger) n_sites: see generate_fixture.R. R's fit here is
        # boundary-hugging on 3 of 6 traits (per-trait NB2 dispersion + rank-2
        # loadings is a lot of parameters for this sample size) — irrelevant
        # to this test, which only checks that this package's posterior-mode
        # formula agrees with R's AT R's own fitted (however imperfect)
        # parameters, not that either side recovered the truth well.
        Y_nb2_sites = _els_read_csv(_els_verify_and_path("Y_nb2.csv"))
        z_r_nb2      = _els_read_csv(_els_verify_and_path("z_nb2_unit.csv"))
        Lambda_r_nb2 = _els_read_csv(_els_verify_and_path("Lambda_nb2_hat.csv"))
        beta_r_nb2   = _els_read_vector(_els_verify_and_path("beta_nb2_hat.txt"))
        phi_r_nb2    = _els_read_vector(_els_verify_and_path("phi_nb2_hat.txt"))

        Y_nb2 = Int.(permutedims(Y_nb2_sites))
        p_nb2, n_nb2 = size(Y_nb2)
        @test p_nb2 == p
        @test size(z_r_nb2) == (n_nb2, K)

        fit_atR = GLLVModels.NBGroupedFit(beta_r_nb2, Lambda_r_nb2, phi_r_nb2,
            collect(1:p_nb2), LogLink(), NaN, true, 0)
        z_atR = extract_latent_scores(fit_atR, Y_nb2; level = :unit)
        @test isapprox(z_atR, z_r_nb2; atol = 1e-6)  # measured 5.7e-11
        @test extract_latent_scores(fit_atR, Y_nb2; level = :unit_obs) === nothing
    end

    @testset "component-less fit types return an n×K matrix (no MethodError)" begin
        # NB1Fit, TweedieFit, NBGroupedFit and RowRandomFit have no `X_lv`
        # support, so their `getLV` has no `component` keyword at all —
        # extract_latent_scores must NOT forward `component = :innovation` to
        # them (that would raise a MethodError; see the dispatch note in
        # src/extract_latent_scores.jl).
        fit_nb1 = fit_nb1_gllvm(Y_pois; K = K)
        z_nb1 = extract_latent_scores(fit_nb1, Y_pois; level = :unit)
        @test size(z_nb1) == (n, K)
        @test extract_latent_scores(fit_nb1, Y_pois; level = :unit_obs) === nothing

        fit_tw = fit_tweedie_gllvm(Float64.(Y_pois); K = K)
        z_tw = extract_latent_scores(fit_tw, Float64.(Y_pois); level = :unit)
        @test size(z_tw) == (n, K)

        fit_nbg = fit_nb_gllvm_grouped(Y_pois; K = K, group = collect(1:p))
        z_nbg = extract_latent_scores(fit_nbg, Y_pois; level = :unit)
        @test size(z_nbg) == (n, K)

        fit_rr = fit_row_random_gllvm(Y_pois; K = K)
        z_rr = extract_latent_scores(fit_rr, Y_pois; level = :unit)
        @test size(z_rr) == (n, K)
    end

    @testset "fit types needing an extra positional getLV argument refuse cleanly" begin
        # GllvmCovFit's getLV needs X positionally (getLV(fit, Y, X; ...)):
        # extract_latent_scores's (fit, y; kwargs...) signature cannot route
        # that, and refuses with a named ArgumentError rather than passing X
        # as a keyword (which getLV does not accept and would raise a plain
        # MethodError).
        fit_cov = fit_gllvm_cov(Y_pois; family = Poisson(), X = X, K = K)
        @test_throws ArgumentError extract_latent_scores(fit_cov, Y_pois)
    end

    @testset "RRRFit refuses: no innovation score" begin
        # RRRFit's getLV(fit, X; rotate) is a plain 2-argument call (no
        # missing positional argument), but its z_s = B' x_s is a
        # deterministic, fully predictor-driven projection with no residual
        # latent variable at all -- "innovation" does not apply, and calling
        # getLV(fit, Y) with this wrapper's response matrix in place of
        # RRRFit's covariate design X would raise a raw DimensionMismatch.
        X_rr = reshape(collect(1.0:n) ./ n, n, 1)  # n×1 site-covariate design
        fit_rrr = fit_rrr_gllvm(Y_pois; family = Poisson(), X = X_rr, K = K)
        @test_throws ArgumentError extract_latent_scores(fit_rrr, Y_pois)
    end

    @testset "every AnyGllvmFit member with a getLV method is in exactly one dispatch union" begin
        # Guards the three-Union accounting in src/extract_latent_scores.jl:
        # a newly added fit type with a getLV method that nobody sorts into
        # _ComponentAwareGllvmFit / _PlainGllvmFit / _PositionalArgGllvmFit
        # turns this test red instead of silently defaulting (there is no
        # catch-all _extract_latent_scores_unit method any more). Types with
        # no getLV method at all (e.g. MultinomialFit, StudentTFit, the
        # phylo/spatial-only fits) are correctly excluded from this
        # accounting -- calling extract_latent_scores on them still fails
        # loudly with a MethodError, exactly as calling getLV on them
        # directly already does.
        function has_getLV_method(::Type{T}) where {T}
            for m in methods(getLV)
                params = m.sig.parameters
                length(params) >= 2 || continue
                P1 = params[2]
                T <: P1 && return true
            end
            return false
        end

        buckets = (GLLVModels._ComponentAwareGllvmFit, GLLVModels._PlainGllvmFit,
                   GLLVModels._PositionalArgGllvmFit)
        for T in Base.uniontypes(GLLVModels.AnyGllvmFit)
            has_getLV_method(T) || continue
            n_buckets = count(U -> T <: U, buckets)
            @test n_buckets == 1
        end
    end

    @testset "level validation and default fallback" begin
        fit = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
        @test_throws ArgumentError extract_latent_scores(fit, Y_gauss; level = :bogus, X = X)
        # R accepts deprecated level = "B"/"W" aliases with a warning; this
        # method does not (see "Differences from R" in the docstring).
        @test_throws ArgumentError extract_latent_scores(fit, Y_gauss; level = :B, X = X)
        # Mirrors R's extract_latent_scores.default abort for an unsupported type.
        @test_throws ArgumentError extract_latent_scores([1, 2, 3])
        @test_throws ArgumentError extract_latent_scores("not a fit")
    end
end
