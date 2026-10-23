using GLLVModels, Test, TOML
const GM = GLLVModels

# GP-1 inner mode search (#611). GP-1 did not opt into the damped backtracking of
# `_laplace_mode`, so undamped Fisher steps from z = 0 could stop far from the
# conditional mode. On origin/main 0ce4a35aa (after #606), macOS aarch64, Julia
# 1.10.12, at the point below the search returned z = 0.705 (d log joint / dz =
# -42.8) while the single mode is at -1.624, leaving that site's Laplace value
# 36.6 too low. The error switched on and off as β moved by 1e-6, so the outer
# L-BFGS stopped at the edge of the jump and reported converged; fits on the same
# data from different starts then reached -1180.22 and -1160.61 (seed 104) or
# -1259.00 and -1256.66 (seed 101). With backtracking, every start tried reaches
# -1143.77 (seed 104) and -1249.34 (seed 101).
#
# The mode checks compare against a brute-force 1-D maximisation of the log joint,
# and the fit checks compare fits with each other, so nothing here depends on
# platform-recorded values.

@testset "GP-1 mode search backtracks (#611)" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "gp1_verdict.toml"))
    p, n, K = fixture["p"], fixture["n"], fixture["K"]

    @testset "mode at the recorded overshoot site" begin
        y = [0, 38, 79, 1]
        N1 = ones(Int, 4)
        Λ = reshape([0.6354695270765459, -0.9565159991418574,
                     -0.9104610603921144, 0.7934508176336181], 4, 1)
        α = 0.17747917955744713
        fam = GM.GeneralizedPoisson1(α)
        β0 = [0.7260226279548277, 1.8634868731757008, 1.8481013924288594, 0.8263875411142447]
        logjoint(z, β) = sum(GM._glm_logpdf(fam, exp(β[t] + Λ[t, 1] * z), 1, y[t])
                             for t in 1:4) - z^2 / 2
        # Golden-section search on [-4, 4]: the log joint is unimodal there
        # (log-concave in z for these counts), so this is the reference mode.
        function ref_mode(β)
            a, b = -4.0, 4.0
            ϕ = (sqrt(5) - 1) / 2
            for _ in 1:200
                c = b - ϕ * (b - a); d = a + ϕ * (b - a)
                logjoint(c, β) > logjoint(d, β) ? (b = d) : (a = c)
            end
            return (a + b) / 2
        end
        for δ in (0.0, -1e-6, -1e-5, 1e-5)
            β = copy(β0); β[2] += δ
            ẑ = GM._laplace_mode(fam, y, N1, Λ, β, GM.LogLink())[1]
            @test ẑ ≈ ref_mode(β) atol = 1e-6
        end
        # The site's Laplace value is continuous in β across the old jump.
        site(β) = GM.laplace_loglik_site(fam, y, N1, Λ, β, GM.LogLink())
        βm = copy(β0); βm[2] -= 1e-5
        @test abs(site(βm) - site(β0)) < 1e-3
    end

    @testset "fits from different starts reach one optimum" begin
        # Literal starts that reached the low optima on origin/main (see above).
        starts = Dict(
            "healthy_seed_104" => ([1.067, 0.294, 0.907, 1.058],
                                   [-1.415, -1.946, -0.046, 0.241]),
            "healthy_seed_101" => ([0.706, 2.23, 2.018, 1.055],
                                   [0.749, -1.15, -1.11, 0.253]))
        for (key, (βi, λi)) in sort!(collect(starts))
            Y = reshape(Int64.(fixture[key]["Y_column_major"]), p, n)
            f_default = GM.fit_gp1_gllvm(Y; K = K)
            f_start = GM.fit_gp1_gllvm(Y; K = K, β_init = βi, Λ_init = reshape(λi, p, K))
            @test f_default.converged
            @test f_start.converged
            @test abs(f_default.loglik - f_start.loglik) < 1e-2
        end
    end
end
