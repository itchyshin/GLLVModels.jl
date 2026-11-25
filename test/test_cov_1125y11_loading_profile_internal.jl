# Lane cursor-cloud-1125y11 — loading_profile_confirmatory_internal.jl (unexported)
using GLLVModels, Test

@testset "cov 1125y11 D3 loading profile internals" begin
    @testset "paste guard" begin
        @test GLLVModels._d3_loading_profile_stage1_paste_authorized() ==
            (get(ENV, "GLLVM_STAGE1_PASTE", "") == "G0 Stage 1")
        @test_throws ArgumentError GLLVModels._d3_loading_profile_stage1_require_paste!()
    end

    @testset "_lambda_constraint_is_pinned" begin
        M = fill(NaN, 4, 2)
        M[1, 2] = 0.0
        M[2, 1] = 0.5
        pins = GLLVModels._lambda_constraint_is_pinned(M, 4, 2)
        @test pins[1, 2]     # user pin
        @test pins[2, 1]
        @test pins[1, 1] == false
        @test pins[2, 2] == false
        @test !pins[3, 2]    # row 3 is outside min(p, K) structural band
        pins0 = GLLVModels._lambda_constraint_is_pinned(nothing, 3, 2)
        @test pins0[1, 2]
        @test !pins0[2, 1]
        @test_throws ArgumentError GLLVModels._lambda_constraint_is_pinned(M, 3, 2)
    end

    @testset "_normalize_lambda_constraint_pin_matrix" begin
        M = [0.1 NaN; NaN 0.2; 0.3 NaN]
        out = GLLVModels._normalize_lambda_constraint_pin_matrix(M)
        @test out[1, 1] == 0.1
        @test isnan(out[1, 2])
        @test isnan(out[2, 1])  # above diagonal cleared
        @test out[2, 2] == 0.2
    end

    @testset "_enumerate_free_lambda_entries" begin
        M = fill(NaN, 3, 2)
        M[1, 1] = 0.0
        free = GLLVModels._enumerate_free_lambda_entries(M, 3, 2)
        @test (2, 1) in free
        @test (2, 2) in free
        @test (3, 1) in free
        @test (1, 1) ∉ free
        @test (1, 2) ∉ free
        subset = GLLVModels._enumerate_free_lambda_entries(M, 3, 2;
            entries = [2 1; 3 2])
        @test subset == [(2, 1), (3, 2)]
        @test_throws ArgumentError GLLVModels._enumerate_free_lambda_entries(M, 3, 2;
            entries = [1 2 3])
        @test_throws ArgumentError GLLVModels._enumerate_free_lambda_entries(M, 3, 2;
            entries = [4 1])
    end
end