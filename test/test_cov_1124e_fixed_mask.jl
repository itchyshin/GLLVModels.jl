# Coverage: zero-constraint masks in src/fixed_coefficients.jl.
# Argument checks and the free/fixed index helpers. No fit.
using Test
using GLLVModels

grab(f) = try
    f()
    nothing
catch e
    e
end

@testset "fixed-coefficient masks (1124e)" begin
    label = "beta_fixed"

    @testset "q and nothing" begin
        e = grab(() -> GLLVModels._fixed_zero_mask(nothing, -1, label))
        @test e isa ArgumentError
        @test occursin("beta_fixed: q must be non-negative", e.msg)
        @test GLLVModels._fixed_zero_mask(nothing, 0, label) == falses(0)
        @test GLLVModels._fixed_zero_mask(nothing, 4, label) == falses(4)
    end

    @testset "Bool mask" begin
        mask = GLLVModels._fixed_zero_mask([true, false, true], 3, label)
        @test mask == [true, false, true]
        e = grab(() -> GLLVModels._fixed_zero_mask([true, false], 3, label))
        @test e isa ArgumentError
        @test occursin("must have length q = 3; got 2", e.msg)
        bits = GLLVModels._fixed_zero_mask(BitVector([false, true]), 2, label)
        @test bits == [false, true]
    end

    @testset "integer indices" begin
        mask = GLLVModels._fixed_zero_mask([3, 1, 3], 4, label)
        @test mask == [true, false, true, false]
        e = grab(() -> GLLVModels._fixed_zero_mask([0], 3, label))
        @test e isa ArgumentError
        @test occursin("index 0 is outside 1:3", e.msg)
        e = grab(() -> GLLVModels._fixed_zero_mask([4], 3, label))
        @test e isa ArgumentError
        @test occursin("index 4 is outside 1:3", e.msg)
        e = grab(() -> GLLVModels._fixed_zero_mask(Int8[2], 0, label))
        @test e isa ArgumentError
        @test occursin("index 2 is outside 1:0", e.msg)
    end

    @testset "real 0/1 mask versus integer-valued indices" begin
        mask = GLLVModels._fixed_zero_mask([0.0, 1.0, 0.0], 3, label)
        @test mask == [false, true, false]
        # Length matches q, but 2.0 is not a 0/1 flag, so it is read as an index.
        e = grab(() -> GLLVModels._fixed_zero_mask([0.0, 2.0], 2, label))
        @test e isa ArgumentError
        @test occursin("index 0 is outside 1:2", e.msg)
        idx = GLLVModels._fixed_zero_mask([2.0, 2.0], 4, label)
        @test idx == [false, true, false, false]
        e = grab(() -> GLLVModels._fixed_zero_mask([1.5, 0.0], 3, label))
        @test e isa ArgumentError
        @test occursin("numeric vectors must be a 0/1 mask of length q or integer indices", e.msg)
    end

    @testset "Dict index => 0" begin
        mask = GLLVModels._fixed_zero_mask(Dict(1 => 0, "3" => 0.0), 3, label)
        @test mask == [true, false, true]
        e = grab(() -> GLLVModels._fixed_zero_mask(Dict(2 => 1), 3, label))
        @test e isa ArgumentError
        @test occursin("only supports zero constraints; index 2 has value 1", e.msg)
        e = grab(() -> GLLVModels._fixed_zero_mask(Dict(0 => 0), 3, label))
        @test e isa ArgumentError
        @test occursin("index 0 is outside 1:3", e.msg)
    end

    @testset "rejected type" begin
        e = grab(() -> GLLVModels._fixed_zero_mask("no", 2, label))
        @test e isa ArgumentError
        @test occursin("must be nothing, a Bool vector", e.msg)
        @test occursin("String", e.msg)
    end

    @testset "slice, expand, status, pars, init" begin
        X = reshape(collect(1.0:12.0), 2, 2, 3)
        e = grab(() -> GLLVModels._slice_fixed_X(X, [true, false]))
        @test e isa ArgumentError
        @test occursin("fixed mask length 2 must equal size(X, 3) = 3", e.msg)
        sliced, free = GLLVModels._slice_fixed_X(X, [true, false, true])
        @test free == [2]
        @test size(sliced) == (2, 2, 1)
        @test sliced[:, :, 1] == X[:, :, 2]

        e = grab(() -> GLLVModels._expand_fixed_zero([1.0, 2.0], [true, false, true]))
        @test e isa ArgumentError
        @test occursin("free coefficient vector has length 2; expected 1", e.msg)
        expanded = GLLVModels._expand_fixed_zero([4.5], [true, false, true])
        @test expanded == [0.0, 4.5, 0.0]
        @test GLLVModels._fixed_status([true, false, true]) == ["fixed", "estimated", "fixed"]
        @test GLLVModels._free_coeff_indices([true, false, true]) == [2]

        pars = (beta_fixed = [true, false],)
        @test GLLVModels._pars_fixed_mask(pars, :beta_fixed, 2) == [true, false]
        @test GLLVModels._pars_fixed_mask((;), :beta_fixed, 3) == falses(3)
        e = grab(() -> GLLVModels._pars_fixed_mask(pars, :beta_fixed, 3))
        @test e isa ArgumentError
        @test occursin("beta_fixed length (2) must equal coefficient length 3", e.msg)

        @test GLLVModels._fixed_init_free(nothing, [true, false], "beta_init") === nothing
        e = grab(() -> GLLVModels._fixed_init_free([1.0], [false, false], "beta_init"))
        @test e isa ArgumentError
        @test occursin("beta_init length (1) must equal q = 2", e.msg)
        @test GLLVModels._fixed_init_free([9.0, 3.0, 1.0], [true, false, true], "beta_init") == [3.0]
    end
end
