using GLLVModels, Test, TOML, SHA
const GM = GLLVModels

# Gamma grouped fits: an upper shape-boundary flag that only warns (#864).
# On per-trait USairpollution (K = 1) the `manu` shape stops at about 3.7e5 where
# gllvmTMB reaches 2.8e8, with loglik within 1e-4 of TMB: the likelihood is flat in
# that direction. Maintainer decision 2026-10-25: tell the user, change nothing else.
# `dispersion_upper` (per group, shape above 1e5) is informational; it is separate from
# `dispersion_boundary` (lower end only for Gamma since 2026-09-29) and `converged` is
# left as the optimizer verdict. Literal data: test/fixtures/usairpollution_gamma.toml.

_gsuw_sha(A) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(A)))))

@testset "Gamma grouped: upper shape flag is warn-only (#864)" begin
    fx = TOML.parsefile(joinpath(@__DIR__, "fixtures", "usairpollution_gamma.toml"))
    Y = reshape(Float64.(fx["Y_column_major"]), fx["p"], fx["n"])
    g = collect(1:fx["p"])
    manu = findfirst(==("manu"), fx["traits"])

    @testset "fixture integrity" begin
        @test _gsuw_sha(Y) == fx["data_sha256"]
    end

    @testset "boundary helpers" begin
        d = [1e-8, 1.0, 1e5, 1.0001e5, 1e8]
        @test GM._gamma_shape_upper(d) == [false, false, false, true, true]
        @test GM._dispersion_group_lower_boundary(d) == [true, false, false, false, false]
    end

    @testset "per-trait USairpollution: manu is flagged, nothing else changes" begin
        fit = GM.fit_gamma_gllvm_grouped(Y; K = fx["K"], group = g)
        @test fit.α[manu] > 1e5
        @test fit.dispersion_upper[manu]
        @test fit.dispersion_upper == (fit.α .> 1e5)     # the rule, whatever else stops high
        @test !any(fit.dispersion_boundary)              # the lower-end flag is untouched
        @test fit.converged                              # warn-only: the optimizer verdict stands
        @test fit.loglik ≈ -1297.33495 atol = 1e-3       # TMB: -1297.334874
        @test occursin("shape above 1e5", sprint(show, fit))
        @test_logs (:warn, r"shape α above 1e5.*converged is not affected") match_mode = :any GM.fit_gamma_gllvm_grouped(Y; K = fx["K"], group = g)
    end

    @testset "no warning and no flag on the healthy Gamma fixtures" begin
        bf = TOML.parsefile(joinpath(@__DIR__, "fixtures", "gamma_beta_upper_boundary.toml"))
        hk = sort([k for k in keys(bf) if startswith(k, "healthy_") && bf[k]["family"] == "gamma"])
        @test !isempty(hk)
        for key in hk
            c = bf[key]
            Yh = reshape(Float64.(c["Y_column_major"]), bf["p"], bf["n"])
            gh = Int.(c["group"])
            fit = if haskey(c, "X_column_major")
                Xh = reshape(Float64.(c["X_column_major"]), bf["p"], bf["n"], bf["q"])
                @test_logs min_level = Base.CoreLogging.Warn GM.fit_gamma_gllvm_grouped_cov(Yh; X = Xh, K = bf["K"], group = gh)
            else
                @test_logs min_level = Base.CoreLogging.Warn GM.fit_gamma_gllvm_grouped(Yh; K = bf["K"], group = gh)
            end
            @test all(<(1e5), fit.α)
            @test !any(fit.dispersion_upper)
            @test !occursin("shape above", sprint(show, fit))
        end
    end

    @testset "covariate fitter: the same flag and warning" begin
        bf = TOML.parsefile(joinpath(@__DIR__, "fixtures", "gamma_beta_upper_boundary.toml"))
        c = bf["gamma_huge_cov"]
        p, n, q = bf["p"], bf["n"], bf["q"]
        Yc = reshape(Float64.(c["Y_column_major"]), p, n)
        Xc = reshape(Float64.(c["X_column_major"]), p, n, q)
        gc = Int.(c["group"])
        fit = GM.fit_gamma_gllvm_grouped_cov(Yc; X = Xc, K = bf["K"], group = gc)
        @test all(>(1e6), fit.α)
        @test fit.dispersion_upper == fill(true, length(fit.α))
        @test !any(fit.dispersion_boundary)              # a large Gamma alpha is still not a boundary
        @test fit.converged                              # unchanged since 2026-09-29
        @test occursin("shape above 1e5", sprint(show, fit))
        @test_logs (:warn, r"shape α above 1e5.*converged is not affected") match_mode = :any GM.fit_gamma_gllvm_grouped_cov(Yc; X = Xc, K = bf["K"], group = gc)
    end

    @testset "positional constructors derive the flag" begin
        β = zeros(2); Λ = zeros(2, 1); gg = [1, 1]; γ = [0.0]; γf = [false]
        for (v, flag) in ((1e8, true), (2e5, true), (5.0, false), (1e-8, false))
            @test GM.GammaGroupedFit(β, Λ, [v], gg, GM.LogLink(), -1.0, true, 1).dispersion_upper == [flag]
            @test GM.GammaGroupedFit(β, Λ, [v], gg, GM.LogLink(), -1.0, true, 1, :observed, [false], nothing).dispersion_upper == [flag]
            @test GM.GammaGroupedCovFit(β, γ, γf, Λ, [v], gg, GM.LogLink(), -1.0, true, 1).dispersion_upper == [flag]
            @test GM.GammaGroupedCovFit(β, γ, γf, Λ, [v], gg, GM.LogLink(), -1.0, true, 1, :observed, [false]).dispersion_upper == [flag]
        end
    end
end
