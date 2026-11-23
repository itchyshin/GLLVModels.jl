# R scaffold `gllvm_julia()`: NB1 with row.eff = "fixed" or "random" is refused
# before any fit (refs #1271). No Julia fitting; the test reads the R source and,
# when Rscript is available, runs the sibling behavioural script on the helper
# defined above `library(JuliaConnectoR)`.
using Test

const _SCAFFOLD = joinpath(@__DIR__, "..", "..", "r", "gllvmtmb_julia.R")
const _RSCRIPT_TEST = joinpath(@__DIR__, "test_gllvm_julia_nb1_roweff_guard_1271.R")

function _find_rscript()
    w = Sys.which("Rscript")
    w !== nothing && return w
    for cand in (
        "/home/box/micromamba/envs/rlane/bin/Rscript",
        joinpath(homedir(), "micromamba", "envs", "rlane", "bin", "Rscript"),
    )
        isfile(cand) && return cand
    end
    return nothing
end

@testset "gllvm_julia NB1 + row.eff pre-fit guard (refs #1271)" begin
    @test isfile(_SCAFFOLD)
    @test isfile(_RSCRIPT_TEST)
    src = read(_SCAFFOLD, String)
    def_at = findfirst(".check_nb1_row_eff <- function", src)
    call_at = findfirst(".check_nb1_row_eff(family, row.eff)", src)
    lib_at = findfirst("library(JuliaConnectoR)", src)
    dispatch_at = findfirst(".dispatch_fit", src)
    @test def_at !== nothing
    @test call_at !== nothing
    @test lib_at !== nothing
    @test dispatch_at !== nothing
    @test def_at < lib_at
    @test lib_at < call_at
    @test call_at < dispatch_at

    rscript = _find_rscript()
    if rscript === nothing
        @info "Rscript not found; skipping behavioural check of .check_nb1_row_eff"
        @test_skip "Rscript not available"
    else
        out = Pipe()
        err = Pipe()
        proc = run(pipeline(`$rscript --vanilla $_RSCRIPT_TEST $_SCAFFOLD`; stdout=out, stderr=err); wait=false)
        close(out.in)
        close(err.in)
        stdout_s = read(out, String)
        stderr_s = read(err, String)
        wait(proc)
        if !success(proc)
            @info "Rscript failed" stdout = stdout_s stderr = stderr_s
        end
        @test success(proc)
        @test occursin("OK", stdout_s)
    end
end
