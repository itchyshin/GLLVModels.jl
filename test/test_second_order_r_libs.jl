# The second-order tools must load gllvmTMB from GLLVM_PARITY_R_LIBS when it is
# set, and refuse when that library has no gllvmTMB. Before this guard,
# `_require_gllvmtmb!` ignored the variable, so a run that set only
# GLLVM_PARITY_R_LIBS silently used R's default gllvmTMB (found in the
# 2026-09-24 Delta A D1 remeasure).

using Test

include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "r_lib.jl"))

@testset "second-order R library choice (GLLVM_PARITY_R_LIBS)" begin
    @test second_order_r_lib(Dict{String,String}()) === nothing
    @test second_order_r_lib(Dict("GLLVM_PARITY_R_LIBS" => "")) === nothing
    @test second_order_r_lib(Dict("GLLVM_PARITY_R_LIBS" => "  ")) === nothing
    mktempdir() do dir
        # Set but unusable: refuse rather than fall back to R's default library.
        @test_throws ArgumentError second_order_r_lib(Dict("GLLVM_PARITY_R_LIBS" => dir))
        @test_throws ArgumentError second_order_r_lib(
            Dict("GLLVM_PARITY_R_LIBS" => joinpath(dir, "absent")))
        mkdir(joinpath(dir, "gllvmTMB"))
        @test second_order_r_lib(Dict("GLLVM_PARITY_R_LIBS" => dir)) == realpath(dir)
    end
end

@testset "second-order tools load gllvmTMB from GLLVM_PARITY_R_LIBS (live R)" begin
    lib = get(ENV, "GLLVM_PARITY_R_LIBS", "")
    if get(ENV, "GLLVM_PARITY_TESTS", "0") != "1" || isempty(lib)
        @test_skip "set GLLVM_PARITY_TESTS=1 and GLLVM_PARITY_R_LIBS to check the R load path"
    else
        using RCall
        include(joinpath(@__DIR__, "..", "tools", "core070_second_order", "common.jl"))
        mktempdir() do dir
            withenv("GLLVM_PARITY_R_LIBS" => dir) do
                @test_throws ArgumentError _require_gllvmtmb!()
            end
        end
        _require_gllvmtmb!()
        loaded = RCall.rcopy(RCall.reval(
            "normalizePath(getNamespaceInfo('gllvmTMB', 'path'), mustWork = TRUE)"))
        @test loaded == realpath(joinpath(lib, "gllvmTMB"))
        # Already loaded from `lib`; asking for another library must stop, not
        # quietly keep the loaded build.
        mktempdir() do dir
            mkdir(joinpath(dir, "gllvmTMB"))
            withenv("GLLVM_PARITY_R_LIBS" => dir) do
                @test_throws RCall.REvalError _require_gllvmtmb!()
            end
        end
    end
end

# test/parity/parity_helpers.jl had its own, older, quieter version of this
# same bug: _parity_prepend_twin_lib!() read GLLVM_PARITY_R_LIBS itself (not
# through second_order_r_lib()) and `return nothing`d -- silent fall-back to
# R's default library -- whenever the named library had no gllvmTMB. Every
# tool that calls _parity_require_gllvmtmb!() inherited that silence,
# including the second-order tools this file already covers plus the seven
# core070 tools that used to call `library(gllvmTMB)` directly (2026-09-25
# fix, docs/dev-log/core070/r-lib-provenance-audit-2026-09-25.md).
@testset "parity_helpers.jl routes GLLVM_PARITY_R_LIBS through second_order_r_lib (live R)" begin
    lib = get(ENV, "GLLVM_PARITY_R_LIBS", "")
    if get(ENV, "GLLVM_PARITY_TESTS", "0") != "1" || isempty(lib)
        @test_skip "set GLLVM_PARITY_TESTS=1 and GLLVM_PARITY_R_LIBS to check the parity_helpers load path"
    else
        using RCall
        include(joinpath(@__DIR__, "parity", "parity_helpers.jl"))
        mktempdir() do dir
            withenv("GLLVM_PARITY_R_LIBS" => dir) do
                # _parity_prepend_twin_lib!() is the function parity_helpers.jl
                # itself uses; it must refuse before it ever touches R.
                @test_throws ArgumentError _parity_prepend_twin_lib!()
                # _parity_require_gllvmtmb!() is the entry point every migrated
                # tool now calls first (e.g. tools/core070_aghq_gaussian_pair_run.jl,
                # tools/core070_source_fixed_residual_pair.jl,
                # tools/core070_covariance_mode_fits.jl) -- it must refuse too,
                # not quietly fit against whatever gllvmTMB R's default library holds.
                @test_throws ArgumentError _parity_require_gllvmtmb!()
            end
        end
        # A correctly named library still loads, and loads from that library.
        _parity_require_gllvmtmb!()
        loaded = RCall.rcopy(RCall.reval(
            "normalizePath(getNamespaceInfo('gllvmTMB', 'path'), mustWork = TRUE)"))
        @test loaded == realpath(joinpath(lib, "gllvmTMB"))
    end
end
