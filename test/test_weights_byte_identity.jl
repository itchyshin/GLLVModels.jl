# Byte-identity guard for observation weights (slice W4-1a). Adding `weights` to the shared Laplace
# mode code must leave every unweighted fit exactly as it was: `weights = nothing` takes the old
# arithmetic path, never a multiplication by 1.0.
#
# (1) Cross-process: origin/main's outputs (test/fixtures/weights_byte_identity/main_capture.toml,
#     captured by gen_main.jl in a detached worktree of origin/main, which has no weights keyword)
#     are compared bit for bit with this tree's outputs, with weights omitted and with
#     weights = nothing. Float64 bit patterns depend on Julia version, CPU and BLAS, so this part runs
#     only when this host's fingerprint equals the capture's; elsewhere it is skipped with a note.
# (2) In-process, on every host: weights omitted and weights = nothing give isequal results.
using Test
using TOML
include(joinpath(@__DIR__, "fixtures", "weights_byte_identity", "cases.jl"))

@testset "weights = nothing is byte-identical to main" begin
    cap = TOML.parsefile(joinpath(@__DIR__, "fixtures", "weights_byte_identity", "main_capture.toml"))
    # Compare at the capture's BLAS thread count, then restore the suite's setting.
    blas_threads = GLLVModels.LinearAlgebra.BLAS.get_num_threads()
    GLLVModels.LinearAlgebra.BLAS.set_num_threads(1)
    same_host = cap["fingerprint"] == _wbi_fingerprint()
    same_host || @info "weights byte-identity: host fingerprint differs from the capture; cross-process check skipped" capture = cap["fingerprint"] here = _wbi_fingerprint()
    cases = _wbi_cases()
    @test Set(first.(cases)) == Set(k for k in keys(cap) if cap[k] isa Dict)
    for (name, thunk) in cases
        @testset "$name" begin
            r_omit = _wbi_record(thunk((;)))
            r_none = _wbi_record(thunk((; weights = nothing)))
            ref = cap[name]
            for (fld, bits) in ref
                @test haskey(r_omit, fld) && haskey(r_none, fld)
                @test r_omit[fld] == r_none[fld]                  # (2), every host
                if same_host
                    @test r_omit[fld] == bits                     # (1)
                    @test r_none[fld] == bits
                else
                    @test_skip r_omit[fld] == bits
                end
            end
        end
    end
    GLLVModels.LinearAlgebra.BLAS.set_num_threads(blas_threads)
end
