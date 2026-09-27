# Julia side of the phylo_latent twin receipts at gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9).
#
#   julia --project=. tools/phylo_latent/compare_phylo_latent_p1.jl fit CASE DATA_JSON OUT_JSON
#       Fit `fit_phylo_latent_gllvm` on the literal fixture and write the Julia
#       receipt (theta in both coordinate orders, logLik, wall time).
#   julia --project=. tools/phylo_latent/compare_phylo_latent_p1.jl compare CASE DATA_JSON JULIA_JSON R_JSON
#       Evaluate the Julia objective at R's optimum and print every paired
#       quantity against the signed tolerances.
#
# The replay test `test/test_phylo_latent_paired_p1.jl` includes this file for
# the shared helpers; nothing here runs at include time.
using JSON3, SHA, LinearAlgebra, GLLVModels

const PL_P1_SOURCE_PIN = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const PL_P1_CASES = ("struct_phy_tree_rr", "struct_phy_dense_rr", "cov_phylo_latent_rsz")
const PL_P1_TOL = (loglik_rtol = 1e-6, cross_abs = 1e-8, estimate_rtol = 1e-4,
                   logdet_abs = 1e-8, cross_gradient = 1e-4)

_pl_matrix(rows) = permutedims(reduce(hcat, [Float64.(collect(r)) for r in rows]))

"""Little-endian Float64 column-major SHA-256 of a response matrix (R's `yhash`)."""
function pl_y_hash(Y::AbstractMatrix{Float64})
    io = IOBuffer()
    for x in vec(Y)
        write(io, htol(x))
    end
    return bytes2hex(sha256(take!(io)))
end

"""Read a fixture JSON: response, labels, tree, dense correlation matrix."""
function pl_read_fixture(path)
    fx = JSON3.read(read(path, String))
    Y = _pl_matrix(fx.Y_traits_by_observations)
    pl_y_hash(Y) == fx.data_sha256 || error("fixture response hash mismatch")
    return (Y = Y, species = String.(collect(fx.observation_species)),
            tips = String.(collect(fx.tip_labels)), newick = String(fx.newick),
            vcv = _pl_matrix(fx.vcv_corr), rank = Int(fx.rank),
            data_sha256 = String(fx.data_sha256), fixture = String(fx.fixture))
end

pl_route(case) = case == "struct_phy_dense_rr" ? :vcv : :tree

"""The twin's precision and observation map for one case (same code path as the fit)."""
function pl_precision(fx, case)
    if pl_route(case) === :tree
        phy, labels = GLLVModels._phylo_latent_tree_precision(fx.newick)
    else
        levels = sort(unique(fx.species))
        phy = GLLVModels._phylo_latent_dense_precision(fx.vcv, fx.tips, levels)
        labels = levels
    end
    position = Dict(l => i for (i, l) in enumerate(labels))
    return phy, [position[s] for s in fx.species]
end

"""R coordinate order `[b_fix; log_sigma_eps; theta_rr_phy]` to Julia `[beta; rr; log_sd]`."""
pl_r_to_julia(theta, n_traits) = vcat(theta[1:n_traits], theta[(n_traits + 2):end], theta[n_traits + 1])
pl_julia_to_r(theta, n_traits) = vcat(theta[1:n_traits], theta[end], theta[(n_traits + 1):(end - 1)])

"""Julia marginal negative log-likelihood at a Julia-order `theta`."""
function pl_julia_objective(fx, case, theta_julia)
    phy, species_id = pl_precision(fx, case)
    phy = GLLVModels._validate_precision_fit_input(phy)
    return GLLVModels._precision_multivariate_nll(fx.Y, phy, theta_julia; rank = fx.rank,
        mode = :barelowrank, residual_mode = :shared, species_id = species_id)
end

function pl_fit(fx, case; kwargs...)
    source = pl_route(case) === :tree ? (tree = fx.newick,) : (vcv = fx.vcv, tip_labels = fx.tips)
    return fit_phylo_latent_gllvm(fx.Y, fx.species; d = fx.rank, source..., kwargs...)
end

_pl_rows(M) = [collect(M[i, :]) for i in 1:size(M, 1)]

function pl_fit_receipt(case, data_path, out_path)
    isfile(out_path) && error("use a fresh output path; receipts are immutable")
    case in PL_P1_CASES || error("unknown case $case")
    fx = pl_read_fixture(data_path)
    started = time_ns()
    fit = pl_fit(fx, case)
    elapsed = (time_ns() - started) / 1e9
    n_traits = size(fx.Y, 1)
    phy = fit.phy
    receipt = Dict{String,Any}(
        "schema_version" => "phylo-latent-p1-julia-receipt-1", "case" => case,
        "status" => "recorded", "source_pin" => PL_P1_SOURCE_PIN,
        "julia_version" => string(VERSION),
        "git_head" => strip(read(`git -C $(@__DIR__) rev-parse HEAD`, String)),
        "data_file_sha256" => bytes2hex(sha256(read(data_path))),
        "data_sha256" => fx.data_sha256, "route" => String(pl_route(case)),
        "entry" => "fit_phylo_latent_gllvm", "residual_mode" => "shared",
        "theta_julia_order" => fit.parameters,
        "theta_r_order" => pl_julia_to_r(fit.parameters, n_traits),
        "parameter_labels_julia" => fit.parameter_labels,
        "loglik" => fit.loglik, "objective" => -fit.loglik,
        "converged" => fit.converged, "gradient_norm" => fit.gradient_norm,
        "iterations" => fit.iterations, "stopping_reason" => String(fit.stopping_reason),
        "hessian_positive_definite" => fit.hessian_positive_definite,
        "hessian_condition_number" => fit.hessian_condition_number,
        "Sigma_phy" => _pl_rows(fit.loading * fit.loading'),
        "beta" => fit.beta, "sigma_eps2" => fit.residual_variance[1],
        "log_det_Q" => phy.log_det, "n_aug" => phy.n_aug,
        "species_aug_id" => phy.species_aug_id,
        "fit_elapsed_seconds" => elapsed, "qualified" => false)
    open(out_path, "w") do io
        JSON3.pretty(io, receipt)
    end
    println("Julia $case loglik=$(fit.loglik) converged=$(fit.converged) elapsed=$(round(elapsed; digits = 2))s")
    return receipt
end

"""Every paired quantity for one case, from the fixture and both receipts."""
function pl_compare(case, data_path, julia_path, r_path)
    fx = pl_read_fixture(data_path)
    jr = JSON3.read(read(julia_path, String))
    rr = JSON3.read(read(r_path, String))
    n_traits = size(fx.Y, 1)
    r_theta = Float64.(collect(rr.theta_hat))
    j_theta = Float64.(collect(jr.theta_julia_order))
    julia_at_r = pl_julia_objective(fx, case, pl_r_to_julia(r_theta, n_traits))
    julia_at_j = pl_julia_objective(fx, case, j_theta)
    Sigma_r = _pl_matrix(rr.Sigma_phy)
    Sigma_j = _pl_matrix(jr.Sigma_phy)
    beta_r = r_theta[1:n_traits]
    s2_r = exp(2 * r_theta[n_traits + 1])
    phy, _ = pl_precision(fx, case)
    return (
        loglik_rel = abs(jr.loglik - rr.loglik) / abs(rr.loglik),
        julia_own_reeval_abs = abs(julia_at_j - jr.objective),
        cross_julia_at_r_abs = abs(julia_at_r - rr.objective),
        cross_r_at_julia_abs = abs(rr.cross.r_objective_at_julia_theta - julia_at_j),
        r_gradient_at_julia = rr.cross.r_gradient_max_abs_at_julia_theta,
        Sigma_rel = maximum(abs.(Sigma_j .- Sigma_r)) / maximum(abs.(Sigma_r)),
        beta_rel = maximum(abs.(Float64.(collect(jr.beta)) .- beta_r) ./ max.(abs.(beta_r), 1e-12)),
        sigma_eps2_rel = abs(jr.sigma_eps2 - s2_r) / s2_r,
        logdet_abs = abs(-phy.log_det - rr.log_det_A_phy_rr),
        n_aug_match = phy.n_aug == rr.n_aug_phy,
        julia_at_r = julia_at_r, r_objective = rr.objective,
        julia_objective = julia_at_j, r_at_julia = rr.cross.r_objective_at_julia_theta)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    mode = isempty(ARGS) ? "" : ARGS[1]
    if mode == "fit" && length(ARGS) == 4
        pl_fit_receipt(ARGS[2], ARGS[3], ARGS[4])
    elseif mode == "compare" && length(ARGS) == 5
        c = pl_compare(ARGS[2:5]...)
        for (k, v) in pairs(c)
            println(rpad(string(k), 26), v)
        end
    else
        error("usage: compare_phylo_latent_p1.jl fit CASE DATA OUT | compare CASE DATA JULIA R")
    end
end
