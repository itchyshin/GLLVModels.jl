# gllvm-parity-tag: P1
#
# Temporal fits against gllvmTMB P1 (9539352f6) receipts, spec section 5:
#   (i)   Julia's NLL at R's opt$par equals R's objective (1e-8);
#   (ii)  R's objective at Julia's optimum (cross_objective.toml, computed by
#         the R generator) equals Julia's NLL there (1e-8);
#   (iii) between optima: |Δ logLik| <= 1e-6 against R's tight-tolerance
#         optimum, and at interior optima phi/kappa atol 1e-5, Sigma_T atol
#         1e-5 (Sigma_T, not the sign-ambiguous factor), psi and sigma_eps
#         rtol 1e-5, beta atol 1e-6;
# plus extract_temporal, nobs, AIC and BIC against R's report at R's
# coordinates. Fits cover indep/dep/latent/latent-unique x AR1/OU,
# unreplicated and replicated.
using Test, GLLVModels, LinearAlgebra
const GMF = GLLVModels
include(joinpath(@__DIR__, "fixtures", "temporal_p1", "fixture_helpers.jl"))

@testset "temporal fits vs gllvmTMB P1 receipts" begin
    F = temporal_p1_load("fits.toml")
    X = temporal_p1_load("cross_objective.toml")
    cross = Dict(c["id"] => c for c in X["cells"])

    @testset "receipt integrity" begin
        @test F["gllvmTMB_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test temporal_p1_sha("fits.toml") == TEMPORAL_P1_SHA256["fits.toml"]
        @test temporal_p1_sha("cross_objective.toml") == TEMPORAL_P1_SHA256["cross_objective.toml"]
        for (name, ds) in F["datasets"]
            @test temporal_p1_value_sha(temporal_p1_table(ds)) == ds["value_sha256"]
        end
        @test length(F["fits"]) == 21
        @test Set(keys(cross)) == Set(c["id"] for c in F["fits"])
    end

    worst = Dict{String,Float64}()
    note(k, v) = (worst[k] = max(get(worst, k, 0.0), v))
    @testset "$(c["id"])" for c in F["fits"]
        rpar = temporal_p1_vec(c["par"]); rtight = temporal_p1_vec(c["par_tight"])
        at_r = temporal_p1_fit(F, c; start=rpar, iterations=0)
        at_rt = temporal_p1_fit(F, c; start=rtight, iterations=0)
        jf = temporal_p1_fit(F, c; g_tol=TEMPORAL_P1_GTOL, iterations=TEMPORAL_P1_ITERATIONS)

        # Coordinates: R's names(opt$par), order and multiplicity.
        @test jf.parameter_names == String.(c["par_names"])

        # (i) Julia's objective at R's coordinates.
        @test abs(-at_r.loglik - c["objective"]) <= 1e-8
        @test abs(-at_rt.loglik - c["objective_tight"]) <= 1e-8
        note("i: |NLL_J(R par) - R objective|", max(abs(-at_r.loglik - c["objective"]),
            abs(-at_rt.loglik - c["objective_tight"])))

        # (ii) R's objective at Julia's coordinates (recorded Julia optimum).
        xc = cross[c["id"]]
        jpar = temporal_p1_vec(xc["julia_par"])
        at_j = temporal_p1_fit(F, c; start=jpar, iterations=0)
        @test abs(-at_j.loglik - xc["r_fn_at_julia_par"]) <= 1e-8
        note("ii: |R objective(J par) - NLL_J(J par)|", abs(-at_j.loglik - xc["r_fn_at_julia_par"]))

        # (iii) between optima.
        dll = jf.loglik - (-c["objective_tight"])
        @test dll >= -1e-8                      # Julia is never worse than R's best
        if dll > 1e-6
            # Julia found a higher likelihood than R's optimiser: R's own
            # objective must confirm it at Julia's point (recorded finding).
            @test xc["r_fn_at_julia_par"] < c["objective_tight"] - 1e-6
            @info "Julia optimum above R's reported optimum (R objective confirms)" c["id"] dll
        else
            @test abs(dll) <= 1e-6
            note("iii: |Δ logLik|", abs(dll))
        end
        interior = jf.hessian_min_eigenvalue > 1e-3 && dll <= 1e-6
        if interior
            @test jf.converged
            # nlminb stops before stationarity even at rel.tol = 1e-14 (its
            # recorded gradient at par_tight reaches 1.8e-3 on these cells),
            # so the parameters are compared at R's point after one Newton
            # step built from R's own recorded gradient. The raw differences
            # are reported alongside.
            nll(t) = GMF.temporal_marginal_nll(t, at_rt.y, at_rt.X, at_rt.spec)
            gR = temporal_p1_vec(c["gradient_at_par_tight"])
            @test maximum(abs, GMF.ForwardDiff.gradient(nll, rtight) .- gR) <= 1e-8
            H = GMF.ForwardDiff.hessian(nll, rtight)
            at_rn = temporal_p1_fit(F, c; start=rtight .- (Symmetric(H) \ gR), iterations=0)
            @test abs(jf.time_value - at_rn.time_value) <= 1e-5
            @test maximum(abs, jf.Sigma_T - at_rn.Sigma_T) <= 1e-5
            @test abs(jf.sigma_eps / at_rn.sigma_eps - 1) <= 1e-5
            jf.psi === nothing || @test maximum(abs, jf.psi ./ at_rn.psi .- 1) <= 1e-5
            @test maximum(abs, jf.beta - at_rn.beta) <= 1e-6
            note("iii: |Δ time| (Newton-polished R)", abs(jf.time_value - at_rn.time_value))
            note("iii: |Δ Sigma_T| (Newton-polished R)", maximum(abs, jf.Sigma_T - at_rn.Sigma_T))
            note("iii: |Δ beta| (Newton-polished R)", maximum(abs, jf.beta - at_rn.beta))
            note("iii: rel |Δ sigma, psi| (Newton-polished R)", max(abs(jf.sigma_eps / at_rn.sigma_eps - 1),
                jf.psi === nothing ? 0.0 : maximum(abs, jf.psi ./ at_rn.psi .- 1)))
            note("raw: |Δ time| (R par_tight)", abs(jf.time_value - at_rt.time_value))
            note("raw: |Δ Sigma_T| (R par_tight)", maximum(abs, jf.Sigma_T - at_rt.Sigma_T))
            note("raw: |Δ beta| (R par_tight)", maximum(abs, jf.beta - at_rt.beta))
            note("raw: rel |Δ sigma, psi| (R par_tight)", max(abs(jf.sigma_eps / at_rt.sigma_eps - 1),
                jf.psi === nothing ? 0.0 : maximum(abs, jf.psi ./ at_rt.psi .- 1)))
        end

        # extract_temporal and the fit methods at R's default coordinates.
        e = extract_temporal(at_r)
        @test e.parameters.mode == c["mode"]
        @test e.parameters.structure == c["structure"]
        @test e.parameters.workflow == c["workflow"]
        @test e.parameters.n_series == c["n_series"]
        @test e.parameters.n_pairs == c["n_pairs"]
        @test e.time.parameter == c["time_parameter"]
        @test abs(e.time.value - c["time_value"]) <= 1e-12
        @test e.pair_index.pair_id == String.(c["pair_id"])
        @test e.pair_index.series == String.(c["pair_series"])
        @test e.pair_index.time == temporal_p1_vec(c["pair_time"])
        if haskey(c, "loadings")
            @test e.loadings !== nothing
            @test vec(e.loadings) ≈ temporal_p1_vec(c["loadings"]) atol = 1e-12
        else
            @test e.loadings === nothing
        end
        @test e.variance.component == String.(c["variance_component"])
        @test e.variance.value ≈ temporal_p1_vec(c["variance"]) atol = 1e-12
        @test vec(at_r.Sigma_T) ≈ temporal_p1_vec(c["Sigma_T"]) atol = 1e-12
        @test abs(at_r.sigma_eps - c["sigma_eps"]) <= 1e-12
        @test nobs(at_r) == c["nobs"]
        @test dof(at_r) == length(c["par"])
        @test abs(aic(at_r) - c["AIC"]) <= 1e-8
        @test abs(bic(at_r) - c["BIC"]) <= 1e-8
        @test abs(loglikelihood(at_r) - c["logLik"]) <= 1e-8
    end
    println("temporal fit receipt maxima:")
    for (k, v) in sort(collect(worst)); println("  ", rpad(k, 46), v); end
end
