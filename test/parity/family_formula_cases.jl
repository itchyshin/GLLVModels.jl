# Formula equivalence inherits R health only at identical native coordinates.
# Registry dependencies require the native case in this same required run.
module Core070FamilyFormulas
using GLLVModels, Test, Random, SHA, TOML
import Distributions
using Main: _core070_receipt_dir, _core070_sha256_file, parity_loadings_p5k2, core070_record_values!

function run(family::Symbol)
    settings = Dict(
        :poisson => ("02-LOG", "NATIVE-03-POISSON", 5, 60, 2, 14,
                     "44e647ac7ff29672efa12ca03b7b628dd37a02ac0f8d2421a19d3969743e26da"),
        :beta => ("07-LOGIT", "NATIVE-08-BETA", 5, 60, 1, 15,
                  "3ddd9a90ae2e933af8c9f45f72fc7edda96dc0fda2a4389d3708a003d07ce099"),
        :truncnb2 => ("11-LOG", "NATIVE-12-TRUNCATED-NB2", 5, 120, 1, 15,
                      "ecbcf9f501c7e618131f2c3f1f0d213bb0e92364a72c0519095c52ef30930948"))
    tag, native_id, p, n, K, nfree, data_sha = settings[family]
    id = "CORE070-FAMILY-" * tag * "-FORMULA-INTERFACE"
    dir = _core070_receipt_dir()
    health_name = family == :truncnb2 ? "truncnb2-policy.toml" : string(family) * "-health.toml"
    health_path = joinpath(dir, health_name)
    health = TOML.parsefile(health_path) # Missing native receipt is a hard failure.
    if family == :truncnb2
        source = read(joinpath(@__DIR__, "test_truncated_nbinom2_parity.jl"), String)
        a = findfirst("    Random.seed!(_TNB2_SEED)", source).start
        b = findnext("    @testset \"support", source, a).start
        mod = Module(:OriginalTruncatedFormula)
        Core.eval(mod, :(using Random, Distributions))
        Core.eval(mod, :(using Main: parity_loadings_p5k2))
        Y = Base.include_string(mod, "const _TNB2_SEED=58\n" * source[a:b-1] * "\nY\n")
        marker = TruncatedNegBin2()
        controls = (; disp_group=:species, hessian=:observed)
        native = fit_truncated_nbinom2_gllvm_pertrait(Y; K=K, hessian=:observed)
        theta = f -> vcat(f.β, GLLVModels.pack_lambda(f.Λ), log.(f.r))
    else
        data = TOML.parsefile(joinpath(dir, string(family) * "-fixture.toml"))
        (data["p"], data["n"], data["K"]) == (p, n, K) || error("original shape changed")
        Y = reshape(data["Y_column_major"], p, n)
        marker = family == :poisson ? GLLVModels.Poisson() : GLLVModels.Beta()
        controls = family == :poisson ? (;) : (; g_tol=1e-7, iterations=800)
        native = fit_gllvm(Y; family=marker, K=K, controls...)
        theta = f -> vcat(f.β, GLLVModels.pack_lambda(f.Λ), family == :beta ? log.(f.φ) : Float64[])
    end
    curvature = f -> family == :truncnb2 ? controls.hessian : f.hessian
    curvature_provenance = family == :truncnb2 ? "explicit_keyword_not_stored_in_fit" : "fitted_object"
    wide = gllvm(@formula(y ~ 1), Y, (site=collect(1:n),); family=marker, K=K, controls...)
    long = (y=reverse(vec(Y)), species=reverse(repeat(collect(1:p), n)),
            site=reverse(repeat(collect(1:n); inner=p)))
    longfit = gllvm(@formula(y ~ 1), long; family=marker, K=K, controls...)
    error_name(f) = try f(); "NO_ERROR" catch err; string(typeof(err)) end
    errors = Dict(
        "wrong_rows" => error_name(() -> gllvm(@formula(y ~ 1), Y, (site=1:n-1,);
                                                family=marker, K=K, controls...)),
        "missing_long" => error_name(() -> gllvm(@formula(y ~ 1), map(x -> x[2:end], long);
                                                  family=marker, K=K, controls...)),
        "duplicate_long" => error_name(() -> gllvm(@formula(y ~ 1), map(x -> vcat(x, x[1]), long);
                                                    family=marker, K=K, controls...)))
    actual_data_sha = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(Y)))))
    # R's gradient is recorded, not a gate, for every family (decision 2026-09-30): nlminb's
    # stopping gradient varies by machine at boundary dispersion. R's convergence code
    # (r_converged / r_code == 0) and the native gradient stay gates.
    healthy = health["native_converged"] && get(health, "r_converged", get(health, "r_code", -1) == 0) &&
              health["native_gradient_max"] <= 1e-4
    println("  gllvmTMB r_gradient_max = ", health["r_gradient_max"], " (recorded, not a gate)")
    report = Dict("id"=>id, "native_id"=>native_id, "data_sha256"=>actual_data_sha,
                  "native_health_file"=>health_name, "native_health_sha256"=>_core070_sha256_file(health_path),
                  "health_proof"=>"same-run native health at identical fitted coordinates",
                  "curvature_provenance"=>curvature_provenance, "nfree"=>nfree, "curvature"=>string(curvature(native)),
                  "reference_control_policy"=>health["policy"], "r_loglik"=>health["r_loglik"],
                  "r_gradient_max"=>health["r_gradient_max"],
                  "input_errors"=>errors)
    for (label, fit) in (("native", native), ("wide", wide), ("long", longfit))
        report[label] = Dict("parameters"=>theta(fit), "loglik"=>fit.loglik,
                             "converged"=>fit.converged, "type"=>string(typeof(fit)),
                             "hessian"=>string(curvature(fit)))
    end
    output = joinpath(dir, string(family) * "-formula.toml")
    open(io -> TOML.print(io, report), output, "w")
    println("FAMILY_FORMULA_SHA256 ", basename(output), " ", _core070_sha256_file(output))
    @testset "$id original model through formulas" begin
        @test healthy
        @test actual_data_sha == data_sha == health["data_sha256"]
        @test native.converged
        @test length(theta(native)) == nfree
        @test theta(native) ≈ health["native_parameters"] atol=1e-10 rtol=0
        for (route, fit) in (("wide", wide), ("long", longfit))
            core070_record_values!("logLik, $route formula"; julia=fit.loglik, r=health["r_loglik"], rtol=1e-6,
                test="@test fit.loglik ≈ health[\"r_loglik\"] atol=0 rtol=1e-6")
            @test typeof(fit) == typeof(native)
            @test fit.converged
            @test curvature(fit) == curvature(native)
            @test length(theta(fit)) == nfree
            @test theta(fit) ≈ theta(native) atol=1e-10 rtol=0
            @test fit.loglik ≈ native.loglik atol=1e-10 rtol=0
            @test fit.loglik ≈ health["r_loglik"] atol=0 rtol=1e-6
        end
        @test errors["wrong_rows"] == "DimensionMismatch"
        @test errors["missing_long"] == "ArgumentError"
        @test errors["duplicate_long"] == "ArgumentError"
    end
end
end
