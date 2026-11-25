# Lane cursor-cloud-1125y6: fixed_coefficients.jl mask/slice/expand helpers.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y6 fixed coefficient helpers" begin
    @testset "_fixed_zero_mask" begin
        @test G._fixed_zero_mask(nothing, 4, "β_fixed") == falses(4)
        @test G._fixed_zero_mask(Bool[true, false, true, false], 4, "β_fixed") ==
            Bool[1, 0, 1, 0]
        @test_throws ArgumentError G._fixed_zero_mask(Bool[true, false], 4, "β_fixed")
        @test G._fixed_zero_mask([1, 3], 4, "γ_fixed") == Bool[1, 0, 1, 0]
        @test_throws ArgumentError G._fixed_zero_mask([0, 5], 4, "γ_fixed")
        @test G._fixed_zero_mask([1.0, 0.0, 1.0, 0.0], 4, "β_fixed") == Bool[1, 0, 1, 0]
        @test G._fixed_zero_mask([1, 2], 4, "β_fixed") == Bool[1, 1, 0, 0]
        @test_throws ArgumentError G._fixed_zero_mask([1.5, 2.0], 4, "β_fixed")
        m = G._fixed_zero_mask(Dict(2 => 0, "4" => 0.0), 4, "β_fixed")
        @test m == Bool[0, 1, 0, 1]
        @test_throws ArgumentError G._fixed_zero_mask(Dict(2 => 1), 4, "β_fixed")
        @test_throws ArgumentError G._fixed_zero_mask("bad", 3, "β_fixed")
        @test_throws ArgumentError G._fixed_zero_mask(nothing, -1, "β_fixed")
    end

    @testset "_free_coeff_indices and slice" begin
        mask = Bool[1, 0, 1, 0]  # true = fixed coefficient
        @test G._free_coeff_indices(mask) == [2, 4]
        p, n, q = 2, 3, 4
        X = reshape(collect(1.0:(p * n * q)), p, n, q)
        Xs, free = G._slice_fixed_X(X, mask)
        @test free == [2, 4]
        @test size(Xs) == (p, n, 2)
        @test Xs[1, 1, 1] == X[1, 1, 2]
        @test_throws ArgumentError G._slice_fixed_X(X, falses(3))
    end

    @testset "_expand_fixed_zero" begin
        mask = Bool[1, 0, 1, 0]  # indices 1 and 3 fixed at zero
        out = G._expand_fixed_zero([2.5, -1.0], mask)
        @test out == [0.0, 2.5, 0.0, -1.0]
        @test_throws ArgumentError G._expand_fixed_zero([1.0], mask)
    end

    @testset "_fixed_status" begin
        st = G._fixed_status(Bool[1, 0, 1])
        @test st == ["fixed", "estimated", "fixed"]
    end

    @testset "_pars_fixed_mask and init" begin
        pars = (β_fixed = Bool[1, 0, 0],)
        @test G._pars_fixed_mask(pars, :β_fixed, 3) == Bool[1, 0, 0]
        @test G._pars_fixed_mask((;), :β_fixed, 3) == falses(3)
        @test_throws ArgumentError G._pars_fixed_mask(pars, :β_fixed, 2)
        mask = Bool[1, 0, 0]
        @test G._fixed_init_free([0.1, 0.2, 0.3], mask, "β_init") == [0.2, 0.3]
        @test G._fixed_init_free(nothing, mask, "β_init") === nothing
        @test_throws ArgumentError G._fixed_init_free([0.1], mask, "β_init")
    end
end