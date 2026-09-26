# Pinned RCall runner for the seven fitted Core070 Gaussian covariance models.
#
# This is intentionally an evidence harness, not a production adapter.  It runs
# each public R fit from its own seeded default start, then separately runs the
# native dense source fit from its deterministic default start.  R coordinates
# are translated only to re-evaluate the native objective diagnostically; they
# are never used as a native optimizer start.
using GLLVModels, RCall, LinearAlgebra, Statistics, TOML, SHA, Test

include(joinpath(@__DIR__, "..", "test", "parity", "parity_helpers.jl"))

length(ARGS) in (1,2) || error("expected fresh output directory and optional tight-control")
control_policy = length(ARGS) == 2 ? ARGS[2] : "default"
control_policy in ("default","tight-control") || error("invalid control policy")
control_policy == "tight-control" && !isdir("baseline") && error("retained baseline required")
output = abspath(ARGS[1])
ispath(output) && error("output directory must be fresh: $output")
mkpath(output)
get(ENV, "CORE070_PARITY_REQUIRED", "") == "1" || error("required pinned oracle missing")
_parity_require_gllvmtmb!()
source_pin = _core070_source_pin!()

const _CORE070_COVARIANCE_FITS_FIXTURE =
    joinpath("test", "parity", "fixtures", "core070_covariance_fits.R")
const _CORE070_COVARIANCE_FITS_IDS = [
    "FIT-MODE-ORD-DEP",
    "FIT-MODE-ANIMAL-INDEP", "FIT-MODE-ANIMAL-COMMON", "FIT-MODE-ANIMAL-DEP",
    "FIT-MODE-KERNEL-INDEP", "FIT-MODE-KERNEL-COMMON", "FIT-MODE-KERNEL-DEP",
]

_rows(A::AbstractMatrix) = [collect(Float64, row) for row in eachrow(A)]
_rows(A::AbstractVector) = collect(Float64, A)
_maxabs(A) = isempty(A) ? 0.0 : maximum(abs, A)
_bools_ok(checks) = all(values(checks))

function _packed_lower3(L::AbstractMatrix)
    size(L) == (3, 3) || throw(DimensionMismatch("Core070 runner expects a 3 × 3 loading matrix"))
    return Float64[L[1, 1], L[2, 2], L[3, 3], L[2, 1], L[3, 1], L[3, 2]]
end

function _native_r_coordinate(beta, U, sigma, source_name, mode)
    all(isfinite, beta) && all(isfinite, U) && isfinite(sigma) && sigma > 0 ||
        throw(ArgumentError("R coordinates are not finite"))
    if mode == "DEP"
        L = cholesky(Symmetric(Matrix{Float64}(U))).L
        covariance_coordinates = _packed_lower3(L)
    elseif source_name == "ANIMAL" && mode == "COMMON"
        # R loglambda_phy is a log variance; native common independent uses log SD.
        covariance_coordinates = [log(sqrt(U[1, 1]))]
    elseif mode == "COMMON"
        covariance_coordinates = [log(sqrt(mean(diag(U))))]
    else
        covariance_coordinates = log.(sqrt.(diag(U)))
    end
    return vcat(Float64.(beta), covariance_coordinates, log(sigma))
end

function _source_for_case(C, groups, id, mode)
    native_mode = mode == "DEP" ? :dep : :indep
    return SourceCovariance(C; groups = Int.(groups), name = Symbol(replace(id, '-' => '_')),
        mode = native_mode, common = mode == "COMMON")
end

function _write_report(path, rows)
    report = Dict(
        "matrix_encoding" => "rows",
        "control_policy" => control_policy,
        "source" => source_pin,
        "fixture" => _CORE070_COVARIANCE_FITS_FIXTURE,
        "fixture_sha256" => bytes2hex(sha256(read(_CORE070_COVARIANCE_FITS_FIXTURE))),
        "runner_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "case_ids" => _CORE070_COVARIANCE_FITS_IDS,
        "cases" => rows,
        "failures" => [row["id"] for row in rows if !row["all_checks"]],
        "all_checks" => length(rows) == length(_CORE070_COVARIANCE_FITS_IDS) &&
            all(row -> row["all_checks"], rows),
    )
    open(path, "w") do io
        TOML.print(io, report)
    end
end

R"source('test/parity/fixtures/core070_covariance_fits.R')"
if control_policy == "tight-control"
    R"control <- gllvmTMBcontrol(n_init=1L,se=FALSE,aghq=FALSE,optArgs=list(control=list(rel.tol=1e-12,sing.tol=1e-12,eval.max=2000L,iter.max=1500L)))"
end
rcopy(Int, R"length(cases)") == length(_CORE070_COVARIANCE_FITS_IDS) ||
    error("fixture did not declare exactly seven fit cases")

rows = Dict{String, Any}[]
for index in eachindex(_CORE070_COVARIANCE_FITS_IDS)
    @rput index output
    R"""
    case <- cases[[index]]
    df <- case$data
    C <- case$C
    .core070_warnings <- character()
    .core070_error <- ''
    .core070_fit <- withCallingHandlers(
      tryCatch({
        set.seed(case$fit_seed)
        eval(case$call)
      }, error=function(e) {
        .core070_error <<- conditionMessage(e)
        NULL
      }),
      warning=function(w) {
        .core070_warnings <<- c(.core070_warnings, conditionMessage(w))
        invokeRestart('muffleWarning')
      })
    .core070_evidence <- NULL
    if (is.null(.core070_fit)) {
      saveRDS(list(id=case$id, original_id=case$original_id, warnings=.core070_warnings,
                   error=.core070_error, fit=NULL), file.path(output, paste0(case$id, '.rds')))
      .core070_ok <- FALSE
    } else {
      .core070_ok <- tryCatch({
      .core070_obj <- .core070_fit$tmb_obj
      .core070_outer <- .core070_fit$opt$par
      # Evaluate at the retained outer vector before using env$last.par.
      .core070_objective <- as.numeric(.core070_obj$fn(.core070_outer))
      .core070_gradient <- as.numeric(.core070_obj$gr(.core070_outer))
      .core070_parameters <- .core070_obj$env$parList(x=.core070_outer,
        par=.core070_obj$env$last.par)
      .core070_hessian <- tryCatch(optimHess(.core070_outer, .core070_obj$fn,
        .core070_obj$gr), error=function(e) NULL)
      .core070_hessian_min <- if (is.null(.core070_hessian) ||
        any(!is.finite(.core070_hessian))) NA_real_ else
        min(eigen((.core070_hessian+t(.core070_hessian))/2, symmetric=TRUE,
                  only.values=TRUE)$values)
      .core070_source <- case$source
      .core070_mode <- case$mode
      .core070_ordinary <- .core070_source == 'ORD'
      .core070_propto <- .core070_source == 'ANIMAL' && .core070_mode == 'COMMON'
      .core070_data <- .core070_fit$tmb_data
      .core070_baseline_same <- TRUE
      if (dir.exists('baseline')) {
        prior <- readRDS(file.path('baseline',paste0(case$id,'.rds')))
        .core070_baseline_same <- identical(.core070_data,prior$evidence$data) &&
          identical(.core070_obj$env$map,prior$evidence$map) &&
          identical(names(.core070_outer),names(prior$evidence$outer))
      }
      .core070_C_effective <- if (.core070_ordinary) diag(.core070_data$n_sites) else if
        (.core070_propto) solve(as.matrix(.core070_data$Cphy_inv)) else
        solve(as.matrix(.core070_data$Ainv_phy_rr))
      .core070_groups <- if (.core070_ordinary) .core070_data$site_id + 1L else if
        (.core070_propto) .core070_data$species_id + 1L else .core070_data$species_aug_id + 1L
      .core070_field <- if (.core070_ordinary && .core070_mode != 'DEP') 'theta_diag_B' else if
        (.core070_ordinary) 'theta_rr_B' else if (.core070_propto) 'loglambda_phy' else 'theta_rr_phy'
      .core070_map_string <- function(x) if (is.null(x)) 'unmapped' else
        paste(ifelse(is.na(x), 'NA', as.character(as.integer(x))), collapse=',')
      .core070_U <- if (.core070_field == 'theta_diag_B')
        diag(exp(2*as.numeric(.core070_parameters[[.core070_field]]))) else if
        (.core070_field == 'loglambda_phy') diag(exp(as.numeric(.core070_parameters[[.core070_field]])), 3L) else {
          z <- as.numeric(.core070_parameters[[.core070_field]])
          L <- diag(z[1:3]); L[2,1] <- z[4]; L[3,1] <- z[5]; L[3,2] <- z[6]
          tcrossprod(L)
        }
      .core070_evidence <- list(
        data=.core070_data, map=.core070_obj$env$map, parameters=.core070_parameters,
        outer=.core070_outer, outer_names=names(.core070_outer), gradient=.core070_gradient,
        objective=.core070_objective, loglik=as.numeric(logLik(.core070_fit)),
        optimizer=.core070_fit$opt, code=as.integer(.core070_fit$opt$convergence),
        warnings=.core070_warnings, hessian_min=.core070_hessian_min,
        source_effective=.core070_C_effective, source_groups=.core070_groups,
        covariance=.core070_U, covariance_field=.core070_field,
        covariance_map=.core070_map_string(.core070_obj$env$map[[.core070_field]]),
        sigma_map=.core070_map_string(.core070_obj$env$map$log_sigma_eps))
      TRUE
      }, error=function(e) {
        .core070_error <<- paste('post-fit extraction:', conditionMessage(e))
        FALSE
      })
      saveRDS(list(id=case$id, original_id=case$original_id, full_fit=.core070_fit,
                   evidence=.core070_evidence), file.path(output, paste0(case$id, '.rds')))
    }
    """
    id = rcopy(String, R"case$id")
    original_id = rcopy(String, R"case$original_id")
    source_name = rcopy(String, R"case$source")
    mode = rcopy(String, R"case$mode")
    fit_seed = rcopy(Int, R"case$fit_seed")
    warnings = rcopy(Vector{String}, R".core070_warnings")
    r_error = rcopy(String, R".core070_error")
    expected_id = _CORE070_COVARIANCE_FITS_IDS[index]

    if id != expected_id || !rcopy(Bool, R".core070_ok")
        checks = Dict("expected_id" => id == expected_id, "r_fit_available" => false)
        row = Dict{String, Any}(
            "id" => id, "original_id" => original_id, "source" => source_name,
            "mode" => mode, "fit_seed" => fit_seed, "r_warnings" => warnings,
            "r_error" => r_error, "checks" => checks, "all_checks" => false,
        )
        push!(rows, row); _write_report(joinpath(output, "result.toml"), rows)
        continue
    end

    Y = rcopy(Matrix{Float64}, R"case$Y")
    input_C = rcopy(Matrix{Float64}, R"case$C")
    X_fix = rcopy(Matrix{Float64}, R".core070_data$X_fix")
    y = rcopy(Vector{Float64}, R"as.numeric(.core070_data$y)")
    trait = rcopy(Vector{Int}, R"as.integer(.core070_data$trait_id)+1L")
    site = rcopy(Vector{Int}, R"as.integer(.core070_data$site_id)+1L")
    groups = rcopy(Vector{Int}, R"as.integer(.core070_groups)")
    effective_C = rcopy(Matrix{Float64}, R".core070_C_effective")
    rbeta = rcopy(Vector{Float64}, R"as.numeric(.core070_parameters$b_fix)")
    rU = rcopy(Matrix{Float64}, R".core070_U")
    rsigma = rcopy(Float64, R"as.numeric(exp(.core070_parameters$log_sigma_eps))")
    router = rcopy(Vector{Float64}, R"as.numeric(.core070_outer)")
    rnames = rcopy(Vector{String}, R"names(.core070_outer)")
    rgradient = rcopy(Vector{Float64}, R".core070_gradient")
    robjective = rcopy(Float64, R".core070_objective")
    rloglik = rcopy(Float64, R".core070_evidence$loglik")
    rcode = rcopy(Int, R".core070_evidence$code")
    rhessian_min = rcopy(Float64, R".core070_evidence$hessian_min")
    covariance_field = rcopy(String, R".core070_field")
    covariance_map = rcopy(String, R".core070_evidence$covariance_map")
    sigma_map = rcopy(String, R".core070_evidence$sigma_map")
    r_covariance_free = count(==(covariance_field), rnames)
    r_sigma_free = count(==("log_sigma_eps"), rnames)

    Y_from_data = fill(NaN, 3, 36)
    for i in eachindex(y)
        1 <= trait[i] <= 3 && 1 <= site[i] <= 36 || continue
        Y_from_data[trait[i], site[i]] = y[i]
    end
    expected_X = Matrix{Float64}(I, 3, 3)[trait, :]
    expected_C = source_name == "ORD" ? Matrix{Float64}(I, 36, 36) :
        input_C + 1e-8 * Matrix{Float64}(I, 12, 12)
    expected_groups = source_name == "ORD" ? collect(1:36) : repeat(collect(1:12), inner=3)
    expected_field = source_name == "ORD" ? "theta_rr_B" :
        (source_name == "ANIMAL" && mode == "COMMON" ? "loglambda_phy" : "theta_rr_phy")
    expected_covariance_map = mode == "DEP" || (source_name == "ANIMAL" && mode == "COMMON") ?
        "unmapped" : mode == "COMMON" ? "1,1,1,NA,NA,NA" : "1,2,3,NA,NA,NA"
    expected_covariance_free = mode == "DEP" ? 6 : mode == "COMMON" ? 1 : 3
    groups_by_site = zeros(Int, 36)
    group_consistent = true
    for i in eachindex(site)
        current = groups_by_site[site[i]]
        if current == 0
            groups_by_site[site[i]] = groups[i]
        elseif current != groups[i]
            group_consistent = false
        end
    end
    all(>(0), groups_by_site) || (group_consistent = false)
    # Use the independently declared source matrix.  The inverted R effective
    # matrix remains evidence and is checked above, but is not copied into the
    # native model where solve() roundoff can destroy exact symmetry.
    source = _source_for_case(expected_C, groups_by_site, id, mode)
    source_projection = copy(source.projection)

    native = nothing
    native_error = ""
    native_point = Float64[]
    native_r_nll = NaN
    try
        # Deliberately no `start=`: this tests the public default native start.
        native = fit_gaussian_sources(Y; sources=[source], g_tol=1e-7, iterations=2000)
        native_point = _native_r_coordinate(rbeta, rU, rsigma, source_name, mode)
        native_r_nll = GLLVModels._gaussian_sources_nll(Y, [source], native_point)
    catch e
        native_error = sprint(showerror, e)
    end

    r_gradient_max = _maxabs(rgradient)
    base_checks = Dict(
        "expected_id" => id == expected_id,
        "expected_original_id" => original_id == replace(expected_id, "FIT-" => ""),
        "fixture_shape" => size(Y) == (3, 36) && length(y) == 108,
        "centered_Y_rank_three" => rank(Y .- mean(Y, dims=2)) == 3,
        "data_reconstructs_fixture_Y" => all(isfinite, Y_from_data) && isapprox(Y_from_data, Y; atol=0, rtol=0),
        "trait_order" => trait == repeat(collect(1:3), inner=36),
        "site_order" => site == repeat(collect(1:36), outer=3),
        "fixedmeans_X" => size(X_fix) == (108, 3) && maximum(abs, X_fix - expected_X) == 0,
        "source_groups" => group_consistent && groups == expected_groups[site] &&
            groups_by_site == expected_groups,
        "source_effective_matrix" => size(effective_C) == size(expected_C) &&
            maximum(abs, effective_C - expected_C) <= 1e-12,
        "r_code" => rcode == 0,
        "r_gradient" => all(isfinite, rgradient) && r_gradient_max <= 1e-4,
        "r_objective_report" => isfinite(rloglik) && isfinite(robjective) && abs(rloglik + robjective) <= 1e-8,
        "r_free_parameters" => length(router) == (mode == "DEP" ? 10 : mode == "COMMON" ? 5 : 7),
        "r_covariance_parameterization" => covariance_field == expected_field &&
            covariance_map == expected_covariance_map && r_covariance_free == expected_covariance_free,
        "r_residual_parameterization" => sigma_map == "unmapped" && r_sigma_free == 1,
        "baseline_data_map_unchanged" => rcopy(Bool,R".core070_baseline_same"),
        "native_fit_available" => native !== nothing,
    )
    checks = copy(base_checks)
    if native !== nothing
        merge!(checks, Dict(
            "native_health" => native.converged && isfinite(native.gradient_norm) && native.gradient_norm <= 1e-7,
            "likelihood" => isfinite(native.loglik) && abs(native.loglik - rloglik) <= 1e-6,
            "beta" => isapprox(native.beta, rbeta; atol=1e-5, rtol=1e-5),
            "native_free_parameters" => GLLVModels.dof(native) == length(router) &&
                GLLVModels.dof(native) == (mode == "DEP" ? 10 : mode == "COMMON" ? 5 : 7),
            "native_objective_at_r_coordinates" => isfinite(native_r_nll) &&
                abs(native_r_nll - robjective) <= 1e-6,
        ))
        if source_name == "ORD"
            # This fixture observes one source level per row.  It can validate
            # U + sigma²I, but cannot separately identify U and sigma².
            merge!(checks, Dict(
                "ordinary_total_covariance" => isapprox(only(native.trait_covariances) +
                    native.sigma_eps^2 * Matrix{Float64}(I, 3, 3),
                    rU + rsigma^2 * Matrix{Float64}(I, 3, 3); atol=1e-5, rtol=1e-5),
            ))
        else
            merge!(checks, Dict(
                "structured_source_covariance" => isapprox(only(native.trait_covariances), rU; atol=1e-5, rtol=1e-5),
                "structured_residual_variance" => isapprox(native.sigma_eps^2, rsigma^2; atol=1e-5, rtol=1e-5),
            ))
        end
    end

    native_record = native === nothing ? Dict{String, Any}(
        "available" => false, "error" => native_error,
    ) : Dict{String, Any}(
        "available" => true, "parameters" => native.parameters, "beta" => native.beta,
        "source_covariance" => _rows(only(native.trait_covariances)),
        "residual_sd" => native.sigma_eps, "residual_variance" => native.sigma_eps^2,
        "loglik" => native.loglik, "gradient_max" => native.gradient_norm,
        "converged" => native.converged, "stopping_reason" => String(native.stopping_reason),
        "iterations" => native.iterations, "hessian_min" => native.hessian_min_eigenvalue,
        "hessian_positive_definite" => native.hessian_positive_definite, "dof" => GLLVModels.dof(native),
    )
    r_record = Dict{String, Any}(
        "warnings" => warnings, "error" => r_error, "outer" => router,
        "outer_names" => rnames, "gradient" => rgradient, "gradient_max" => r_gradient_max,
        "beta" => rbeta, "covariance" => _rows(rU), "residual_sd" => rsigma,
        "residual_variance" => rsigma^2, "loglik" => rloglik, "objective" => robjective,
        "code" => rcode, "hessian_min" => rhessian_min, "covariance_field" => covariance_field,
        "covariance_map" => covariance_map, "covariance_free" => r_covariance_free,
        "sigma_map" => sigma_map, "sigma_free" => r_sigma_free,
    )
    row = Dict{String, Any}(
        "id" => id, "original_id" => original_id, "source" => source_name, "mode" => mode,
        "fit_seed" => fit_seed, "Y" => _rows(Y), "input_C" => _rows(input_C),
        "source_effective" => _rows(effective_C), "source_groups" => groups,
        "source_groups_by_site" => groups_by_site, "source_projection" => _rows(source_projection),
        "trait_order" => trait, "site_order" => site,
        "X_fix" => _rows(X_fix), "centered_Y_rank" => rank(Y .- mean(Y, dims=2)),
        "r" => r_record, "native" => native_record,
        "diagnostic" => Dict("native_nll_at_r_coordinates" => native_r_nll,
            "native_nll_minus_r_objective" => native_r_nll - robjective),
        "checks" => checks, "all_checks" => _bools_ok(checks),
    )
    push!(rows, row)
    _write_report(joinpath(output, "result.toml"), rows)
    println(id, " r_grad=", r_gradient_max, " checks=", checks)
end

@testset "Core070 fitted Gaussian covariance modes" begin
    @test length(rows) == length(_CORE070_COVARIANCE_FITS_IDS)
    @test [row["id"] for row in rows] == _CORE070_COVARIANCE_FITS_IDS
    for row in rows
        label = row["id"]
        @testset "$label" begin
            for key in sort!(collect(keys(row["checks"])))
                @test row["checks"][key]
            end
        end
    end
end
println("CORE070_COVARIANCE_MODE_FITS_PASS")
