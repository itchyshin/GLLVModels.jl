using Test
using LinearAlgebra
using SparseArrays
using SHA
using GLLVModels

include(joinpath(@__DIR__, "..", "tools", "destination_b",
    "a4_s4_fixed_coordinate_evaluator.jl"))

const _TEST_A4S4_FIXED_ROOT = joinpath(@__DIR__, "..", "docs", "dev-log", "core070")
const _TEST_A4S4_R_NAMES = ["b_fix", "b_fix", "b_fix", "log_sigma_eps",
    "theta_rr_phy", "theta_rr_phy", "theta_rr_phy"]
const _TEST_A4S4_JULIA_NAMES = ["beta[1]", "beta[2]", "beta[3]",
    "lambda[1]", "lambda[2]", "lambda[3]", "log_sd_residual_shared"]
const _TEST_A4S4_THETA_R = [0.4, -0.2, 0.3, -1.2039728043259361, 0.9, 0.5, -0.6]

function _a4s4_summary(row_id; theta = _TEST_A4S4_THETA_R, r_nll = 0.0)
    kind, aug, scale, log_det, ridge, ridge_once, ridge_operation =
        row_id == "tree_height4_nonunit_ultrametric" ?
        ("tree", [13, 6, 11, 8, 12, 7, 10, 9], 4.0, 15.706819081565975,
            0.0, false, nothing) :
        row_id == "pedigree_12_nodes_8_observed_4_unobserved" ?
        ("pedigree", [11, 4, 8, 6, 9, 5, 10, 7], 1.0, 5.966390909555864,
            0.0, false, nothing) :
        ("dense", collect(0:7), 1.0, 7.861154398716261,
            1e-8, true, "A_ridged = A_original + 1e-8 * I; Q_canonical = solve(A_ridged)")
    response_sha = kind == "tree" ? "a096e8a4f4408923ea0e906defef936f133b92b65202a9baac9fff0d197373c2" :
        kind == "pedigree" ? "025d9ca79375962ef4110c3ce62f84b30cdeea2939957e03f7235e2352e5142b" :
        "e33fcb6d2b391e70b1a8a19d8285c6d0205bdf80d97f35c65d82395d9eef3243"
    observation_map = reduce(vcat, ([node, node] for node in aug))
    row = Dict(
        "row_id" => row_id,
        "kind" => kind,
        "theta_r" => Dict("names" => _TEST_A4S4_R_NAMES, "values" => theta),
        "r_nll" => r_nll,
        "raw_r_repeated_marginal_nll" => r_nll,
        "raw_artifact" => Dict("path" => "private/raw-r-summary.json", "sha256" => repeat("a", 64),
            "attestation_status" => "verified_installed_marker"),
        "provenance" => Dict("source" => "source-attested frozen R raw receipt",
            "source_pin" => "b4d5fee64def88bc768dda1f1f77c29b295edd86"),
        "source_response_sha256" => response_sha,
        "precision_scale" => scale,
        "dense_original_vcv_only" => kind == "dense",
        "construction" => Dict("route" => "gllvmTMB(..., engine = 'tmb')"),
        "canonical_input" => Dict("provenance" => Dict("source_pin" => "b4d5fee64def88bc768dda1f1f77c29b295edd86"),
        "map" => Dict("species_id_one_based" => repeat(collect(1:8); inner = 2),
            "species_aug_id_zero_based" => aug, "observation_to_augmented_zero_based" => observation_map,
            "n_augmented" => length(aug) == 8 && kind == "tree" ? 14 : kind == "pedigree" ? 12 : 8,
            "n_species_observed" => 8, "n_observations" => 16),
        "precision" => Dict("log_det_Q" => log_det, "scale" => scale, "ridge" => ridge,
            "ridge_applied_once" => ridge_once, "ridge_operation" => ridge_operation)),
        "qualified" => false, "public_formula_admission" => "closed")
    return Dict(
        "schema_version" => "destination-b-a4-s4-fixed-coordinate-summary-1",
        "status" => "raw summary exported unqualified",
        "source" => Dict("raw_r_schema_version" => "destination-b-a4-s4-frozen-r-raw-4",
            "input_rds_path" => "private/raw.rds", "input_rds_sha256" => repeat("b", 64),
            "summary_adapter_status" => "runner_recorded_unverified", "claim_boundary" => "closed"),
        "r_provenance" => Dict("source_pin" => "b4d5fee64def88bc768dda1f1f77c29b295edd86",
            "dll_sha256" => repeat("c", 64), "oracle_build_receipt_sha256" => repeat("d", 64),
            "source_provenance" => "verified_installed_marker"),
        "coordinate_labels" => ["b1", "b2", "b3", "log_sigma", "lambda1", "lambda2", "lambda3"],
        "rows" => [row], "qualified" => false, "public_formula_admission" => "closed",
        "claim_boundary" => "private raw summary only")
end

function _a4s4_full_normalizer_nll(fixture, theta_r)
    theta_julia = [theta_r[1:3]; theta_r[5:7]; theta_r[4]]
    beta, lambda, log_sd = theta_julia[1:3], theta_julia[4:6], theta_julia[7]
    residual = fixture.Y .- reshape(beta, :, 1)
    A = inv(Matrix(fixture.phy.Q))
    nodes = fixture.phy.species_aug_id[fixture.species_id]
    covariance = kron(A[nodes, nodes], lambda * lambda') + exp(2log_sd) * I
    factor = cholesky(Symmetric(covariance))
    return 0.5 * (length(residual) * log(2pi) + logdet(factor) +
        dot(vec(residual), factor \ vec(residual)))
end

function _a4s4_full_summary()
    first = _a4s4_summary("tree_height4_nonunit_ultrametric")
    return merge(first, Dict("rows" => [first["rows"][1],
        _a4s4_summary("pedigree_12_nodes_8_observed_4_unobserved")["rows"][1],
        _a4s4_summary("dense_vcv_ridged_once")["rows"][1]]))
end

function _a4s4_copy_core070(directory)
    copied = joinpath(directory, "core070")
    cp(_TEST_A4S4_FIXED_ROOT, copied; force = true)
    return copied
end

@testset "A4/S4 fixed-coordinate evaluator" begin
    tree = a4_s4_fixed_coordinate_fixture(:tree; core070 = _TEST_A4S4_FIXED_ROOT)
    pedigree = a4_s4_fixed_coordinate_fixture(:pedigree; core070 = _TEST_A4S4_FIXED_ROOT)
    dense = a4_s4_fixed_coordinate_fixture(:dense; core070 = _TEST_A4S4_FIXED_ROOT)

    @test tree.species_id == repeat(collect(1:8); inner = 2)
    @test tree.phy.species_aug_id == [14, 7, 12, 9, 13, 8, 11, 10]
    @test pedigree.phy.species_aug_id == [12, 5, 9, 7, 10, 6, 11, 8]
    @test dense.phy.species_aug_id == collect(1:8)
    @test tree.phy.scale == 4.0
    @test pedigree.phy.scale == dense.phy.scale == 1.0

    oracle = _a4s4_full_normalizer_nll(tree, _TEST_A4S4_THETA_R)
    result = evaluate_a4_s4_fixed_coordinate(_a4s4_summary(
        "tree_height4_nonunit_ultrametric"; r_nll = oracle);
        core070 = _TEST_A4S4_FIXED_ROOT)
    @test result["qualified"] === false
    @test result["public_formula_admission"] == "closed"
    @test result["schema_version"] == "destination-b-a4-s4-fixed-coordinate-kernel-cross-evaluation-2"
    @test result["status"] == "fixed_coordinate_kernel_cross_evaluation_unqualified"
    @test result["evaluation_kind"] == "private_kernel_cross_evaluation"
    @test result["coordinate_mapping"]["r_names"] == _TEST_A4S4_R_NAMES
    @test result["coordinate_mapping"]["julia_names"] == _TEST_A4S4_JULIA_NAMES
    @test result["coordinate_mapping"]["julia_values"] ==
        [_TEST_A4S4_THETA_R[1:3]; _TEST_A4S4_THETA_R[5:7]; _TEST_A4S4_THETA_R[4]]
    @test result["julia_nll"] ≈ oracle atol = 1e-10 rtol = 1e-10
    @test result["delta"] ≈ 0.0 atol = 1e-10
    @test result["validation"]["species_id_one_based"] == repeat(collect(1:8); inner = 2)
    immutable = result["validation"]["immutable_input_bytes"]
    @test immutable["fixtures_sha256"] == "089f87d0dcf3c1014fc99646953d8696a0d5aa747287561dd815b5b8bf097549"
    @test immutable["reference_sha256"] == "08c2c0f8dca3bb601e716da30d64dae2cd0ce3834766fa557dabb0444ddf4bd7"
    @test immutable["precision_source_sha256"] == "ab01ee47206565eb1af7da391af313953b8a97c6bb11d9b6160655aaa0429170"
    @test immutable["decoded_response_sha256"] == "a096e8a4f4408923ea0e906defef936f133b92b65202a9baac9fff0d197373c2"

    full = evaluate_a4_s4_fixed_coordinate(_a4s4_full_summary(); core070 = _TEST_A4S4_FIXED_ROOT)
    @test full["qualified"] === false
    @test full["public_formula_admission"] == "closed"
    @test length(full["rows"]) == 3
    @test [row["row_id"] for row in full["rows"]] == [
        "tree_height4_nonunit_ultrametric",
        "pedigree_12_nodes_8_observed_4_unobserved", "dense_vcv_ridged_once"]

    # Q already encodes height-four scaling. Reapplying it changes the model.
    doubled = PrecisionPhy(findnz(tree.phy.Q)[1], findnz(tree.phy.Q)[2],
        4 .* findnz(tree.phy.Q)[3], tree.phy.n_aug, tree.phy.n_leaves,
        tree.phy.node_labels, tree.phy.log_det + tree.phy.n_aug * log(4), 4.0,
        tree.phy.species_aug_id)
    doubled_nll = GLLVModels._precision_multivariate_nll(tree.Y, doubled,
        [_TEST_A4S4_THETA_R[1:3]; _TEST_A4S4_THETA_R[5:7]; _TEST_A4S4_THETA_R[4]];
        rank = 1, mode = :barelowrank, residual_mode = :shared,
        species_id = tree.species_id)
    @test abs(doubled_nll - oracle) > 1e-5

    # The dense fixture supplies Q=(A+1e-8I)^-1. A second ridge/inversion is wrong.
    A_once = inv(Matrix(dense.phy.Q))
    Q_twice = inv(A_once + 1e-8I)
    ii, jj, vv = findnz(sparse(Q_twice))
    twice = PrecisionPhy(ii, jj, vv, dense.phy.n_aug, dense.phy.n_leaves,
        dense.phy.node_labels, logdet(cholesky(Symmetric(Q_twice))), 1.0,
        dense.phy.species_aug_id)
    dense_theta = [_TEST_A4S4_THETA_R[1:3]; _TEST_A4S4_THETA_R[5:7]; _TEST_A4S4_THETA_R[4]]
    once_nll = GLLVModels._precision_multivariate_nll(dense.Y, dense.phy, dense_theta;
        rank = 1, mode = :barelowrank, residual_mode = :shared,
        species_id = dense.species_id)
    twice_nll = GLLVModels._precision_multivariate_nll(dense.Y, twice, dense_theta;
        rank = 1, mode = :barelowrank, residual_mode = :shared,
        species_id = dense.species_id)
    @test abs(twice_nll - once_nll) > 1e-10

    # Immutable bytes are bound before JSON parsing.  Each copy is modified
    # independently so an earlier failed mutation cannot mask a later gate.
    mktempdir() do directory
        copied = _a4s4_copy_core070(directory)
        @test a4_s4_fixed_coordinate_fixture(:tree; core070 = copied).Y == tree.Y
        fixture_path = joinpath(copied, "destination-b-adapter", "fixtures-01.json")
        write(fixture_path, replace(read(fixture_path, String), "\"x\": [10" => "\"x\": [10.01"; count = 1))
        @test_throws ArgumentError a4_s4_fixed_coordinate_fixture(:tree; core070 = copied)
    end
    mktempdir() do directory
        copied = _a4s4_copy_core070(directory)
        reference_path = joinpath(copied, "destination-b-tree", "r-bfgs-attempt-01.json")
        write(reference_path, replace(read(reference_path, String), "-0.57814795578161382" => "-0.57814795578161381"; count = 1))
        @test_throws ArgumentError a4_s4_fixed_coordinate_fixture(:tree; core070 = copied)
    end
    mktempdir() do directory
        copied = _a4s4_copy_core070(directory)
        source_path = joinpath(copied, "destination-b-tree", "precision-reference.json")
        write(source_path, read(source_path, String) * " ")
        @test_throws ArgumentError a4_s4_fixed_coordinate_fixture(:tree; core070 = copied)
    end
    @test _a4s4_response_sha256(tree.Y) ==
        "a096e8a4f4408923ea0e906defef936f133b92b65202a9baac9fff0d197373c2"

    @test_throws ArgumentError evaluate_a4_s4_fixed_coordinate(Dict(
        "schema_version" => "destination-b-a4-s4-fixed-coordinate-summary-1"))
    @test_throws ArgumentError evaluate_a4_s4_fixed_coordinate(_a4s4_summary(
        "tree_height4_nonunit_ultrametric"; r_nll = GLLVModels._NLL_SENTINEL))

    mktempdir() do directory
        input = joinpath(directory, "bad.json")
        write(input, "{\"schema_version\":\"wrong\"}")
        command = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "tools", "destination_b", "a4_s4_fixed_coordinate_evaluator.jl")) $input`
        process = run(pipeline(ignorestatus(command), stdout = devnull, stderr = devnull))
        @test !success(process)

        valid = joinpath(directory, "summary.json")
        output = joinpath(directory, "cross-evaluation.json")
        write(valid, _a4s4_json_write(_a4s4_summary("tree_height4_nonunit_ultrametric"; r_nll = oracle)))
        write_process = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "tools", "destination_b", "a4_s4_fixed_coordinate_evaluator.jl")) $valid $output`), stdout = devnull, stderr = devnull))
        @test success(write_process)
        written = _a4s4_json_read(read(output, String))
        @test written["status"] == "fixed_coordinate_kernel_cross_evaluation_unqualified"
        runner = written["runner_provenance"]
        @test runner["attestation_status"] == "runner_recorded_unverified"
        @test runner["input_summary_path"] == valid
        @test runner["input_summary_sha256"] == bytes2hex(sha256(read(valid)))
        @test runner["julia_version"] == string(VERSION)
        @test length(runner["evaluator_source_sha256"]) == 64
        @test Set(keys(runner["gllvm_kernel_source_sha256"])) == Set([
            "precision_multivariate_fit.jl", "precision_multivariate.jl"])
        @test !success(run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "tools", "destination_b", "a4_s4_fixed_coordinate_evaluator.jl")) $valid $output`), stdout = devnull, stderr = devnull)))
        link = joinpath(directory, "output-link.json")
        symlink(joinpath(directory, "elsewhere.json"), link)
        @test !success(run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "tools", "destination_b", "a4_s4_fixed_coordinate_evaluator.jl")) $valid $link`), stdout = devnull, stderr = devnull)))
    end
end

println("A4_S4_FIXED_COORDINATE_EVALUATOR_OK")
