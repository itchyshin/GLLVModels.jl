using GLLVModels, Test, TOML, SHA, Logging
const GM = GLLVModels

# Grouped NB dispersion: the upper end warns, only the lower end blocks `converged`.
# Maintainer decision 2026-09-29: a fitted NB2 r above 1e6 (the Poisson limit) must not
# make `converged` false (gllvmTMB 0.7.1 itself puts one trait's dispersion above 1e6 on
# 4 of 5 ordinary per-trait NB fits), it only emits a warning; a fitted r below 1e-6
# still makes `converged` false. `dispersion_boundary` keeps flagging BOTH ends (the
# Wald/bootstrap interval code reads it). Literal data, sha256-checked:
# test/fixtures/nb_grouped_upper_warn.toml and [nb_upper] in
# test/fixtures/gamma_beta_upper_boundary.toml.

_nbuw_sha(A, T) = bytes2hex(sha256(reinterpret(UInt8, vec(T.(A)))))

@testset "grouped NB2: upper end warns, lower end blocks converged" begin
    up = TOML.parsefile(joinpath(@__DIR__, "fixtures", "gamma_beta_upper_boundary.toml"))
    lo = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb_grouped_upper_warn.toml"))["lower_case"]
    cu = up["nb_upper"]
    Yu = reshape(Float64.(cu["Y_column_major"]), up["p"], up["n"])
    Yl = reshape(Int.(lo["Y_column_major"]), lo["p"], lo["n"])

    @testset "fixture integrity" begin
        @test _nbuw_sha(Yu, Float64) == cu["data_sha256"]
        @test _nbuw_sha(Yl, Int64) == lo["data_sha256"]
    end

    @testset "boundary helpers" begin
        d = [1e-7, 1.0, 2e6]
        @test GM._dispersion_group_boundary(d) == [true, false, true]        # field: both ends
        @test GM._dispersion_group_lower_boundary(d) == [true, false, false]  # verdict: lower only
    end

    @testset "r > 1e6: warning, dispersion_boundary true, converged is the optimizer verdict" begin
        Yi = round.(Int, Yu)
        fit = nothing
        @test_logs (:warn, r"above 1e6.*Poisson limit.*other estimates are unaffected") match_mode = :any begin
            fit = GM.fit_nb_gllvm_grouped(Yi; K = up["K"], group = Int.(cu["group"]))
        end
        @test all(>(1e6), fit.r_group)
        @test fit.dispersion_boundary == [true]
        @test fit.converged == true            # the recorded Optim verdict; was false before 2026-09-29
        # The warning fires on every call under a test logger (maxlog only limits console output).
        @test_logs (:warn, r"above 1e6") match_mode = :any GM.fit_nb_gllvm_grouped(Yi; K = up["K"], group = Int.(cu["group"]))
    end

    @testset "r < 1e-6: warning says converged is false, converged is false" begin
        fit = nothing
        @test_logs (:warn, r"lower boundary.*converged is false") match_mode = :any begin
            fit = GM.fit_nb_gllvm_grouped(Yl; K = lo["K"], group = Int.(lo["group"]))
        end
        @test all(<(1e-6), fit.r_group)
        @test fit.dispersion_boundary == [true]
        @test fit.converged == false
    end
end
