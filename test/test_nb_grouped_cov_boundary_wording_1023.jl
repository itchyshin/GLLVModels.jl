using GLLVModels, Test, TOML, SHA, Logging
const GM = GLLVModels

# Issue 1023: on the issue's own data (cell 5, replicate 1) `fit_nb_gllvm_grouped_cov`
# sends two species' NB2 size r above 1e6 while `converged` is true, and Λ Λ' is inflated
# (the fit's mean diag(Λ Λ') is 0.69 against a truth of 0.286). The one-time warning used
# to say "the fit's other estimates are unaffected", which that fit contradicts. This
# test pins the corrected wording in the situation it describes, and pins that the fit
# itself is unchanged: wording only, no start, threshold, default or convergence change.
# Literal data, sha256-checked: test/fixtures/nb_grouped_cov_boundary_1023.toml.

@testset "grouped NB2 cov: Poisson-boundary warning wording (#1023)" begin
    fx = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb_grouped_cov_boundary_1023.toml"))
    p, n, K = fx["p"], fx["n"], fx["K"]
    Y = reshape(Int.(fx["Y_column_major"]), p, n)
    x = reshape(Float64.(fx["X_column_major"]), n, 2)
    X = zeros(p, n, 2 * p)   # per-species slopes for x1 and x2, as in the issue
    for t in 1:p, k in 1:2
        X[t, :, (k - 1) * p + t] .= x[:, k]
    end

    @testset "fixture integrity" begin
        @test bytes2hex(sha256(reinterpret(UInt8, Int64.(vec(Y))))) == fx["data_sha256"]
        @test bytes2hex(sha256(reinterpret(UInt8, vec(x)))) == fx["X_sha256"]
    end

    logger = Test.TestLogger()
    fit = Logging.with_logger(logger) do
        GM.fit_nb_gllvm_grouped_cov(Y; X = X, K = K)
    end
    warns = [string(r.message) for r in logger.logs if r.level == Logging.Warn]
    boundary_warns = filter(m -> occursin("above 1e6", m), warns)

    # The situation the wording describes: runaway species with converged = true.
    @test fit.converged
    @test any(>(1e6), fit.r_group)
    @test count(fit.dispersion_boundary) == count(>(1e6), fit.r_group)
    @test length(boundary_warns) == 1
    msg = only(boundary_warns)

    @testset "wording" begin
        @test occursin("Poisson limit", msg)
        @test occursin("converged is not affected", msg)
        @test occursin(r"inflate Λ Λ'", msg)               # names the consequence for the loadings
        @test occursin("issue 1023", msg)
        @test occursin("fit.r_group", msg)                  # says what to inspect
        @test !occursin("other estimates are unaffected", msg)   # the sentence the numbers contradict
    end

    @testset "fit unchanged (wording only)" begin
        # Fingerprint recorded on main 48187dd8 (Linux x86_64, Julia 1.13.3): sha256 of the
        # Float64 bytes of β, γ, vec(Λ), r_group. Bit-identical is only claimed on that
        # platform; elsewhere the rule above is what is checked.
        buf = IOBuffer()
        for v in (fit.β, fit.γ, vec(fit.Λ), fit.r_group)
            write(buf, Float64.(v))
        end
        if Sys.islinux() && Sys.ARCH === :x86_64
            @test bytes2hex(sha256(take!(buf))) == "d4e93e48ceea0d0ece03ec463dacf1ede12db322cd00b3a9ec2d59ee1bd6a4ad"
            @test reinterpret(UInt64, fit.loglik) == 0xc08ddf7bcaeaf341
            @test fit.iterations == 93
        end
    end
end
