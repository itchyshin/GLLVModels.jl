# gllvm-parity-tag: P1
#
# Fit-level twins for the core070 rows data/DATA-MISS-MODEL (miss_control(predictor = "model")) and
# data/DATA-MISS-BOTH (miss_control(response = "include", predictor = "model")), and for the
# namespace row S3method/imputed,gllvmTMB, against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). No R at test time: R's values are read from
# test/fixtures/data_twins_2_p1.toml (generated once by test/fixtures/gen_data_twins_2_p1.R against
# a lane-local gllvmTMB install at the pin; the file records R version, commit and the data sha256).
# Each R fit converged with a positive-definite Hessian (asserted there).
#
# R: value ~ 0 + trait + mi(x) + latent(0 + trait | unit, d = 1, unique = FALSE), Gaussian,
# impute = list(x = x ~ z); x is NA at 18 of 120 units. Julia: fit_gaussian_mi_fiml(Y, x; K = 1,
# Z = z), the same model (shared slope on x, covariate model x ~ N(mu_x + gamma z, sigma_x^2), one
# latent factor, common residual SD), with the missing x integrated out in closed form where R uses
# the Laplace approximation (exact here, the model being Gaussian). For miss_both, 36 response cells
# are NA: R keeps them under response = "include", Julia takes them as `missing` cells of Y.
# imputed(): R's imputed(fit)$estimate (the conditional mode of each missing x) against Julia's
# imputed(fit, x).estimate at the same units. R's standard errors are recorded, not compared (Julia
# reports none).
#
# Tolerances are those of the sibling data twins (test_data_twins_p1.jl), not loosened to pass:
# logLik 1e-6 (absolute), intercepts and the mean-model coefficients 1e-4, the SDs 1e-3,
# Lambda Lambda' 1e-3, conditional modes 1e-4. The sign of a K = 1 loading axis is not identified,
# so loadings are compared through Lambda Lambda'.
using Test
using GLLVModels
using TOML
using SHA

const _DT2_DIR = joinpath(@__DIR__, "fixtures")
const _DT2_TOML = joinpath(_DT2_DIR, "data_twins_2_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns; NA for a missing cell) -> p x n matrix.
function _dt2_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
    hdr = split(readline(path), ",")
    ci = findfirst(==("\"" * col * "\""), hdr)
    ci === nothing && error("column $col not found in $path")
    M = Matrix{Union{Missing,Float64}}(missing, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            v = a[ci] == "NA" ? missing : parse(Float64, a[ci])
            M[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = v
        end
    end
    return M
end

_dt2_mat(v, p) = reshape(Float64.(v), p, p)

@testset "miss_control(predictor = \"model\") / (\"include\", \"model\") and imputed(): gllvmTMB P1 (9539352f6)" begin
    if !isfile(_DT2_TOML)
        @warn "data twins 2 P1 fixture absent; twin gate NOT RUN" _DT2_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_DT2_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        csv = joinpath(_DT2_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(csv))) == fx["data_sha256"]
        X = _dt2_load(csv, "x", p, n)
        x = X[1, :]                                   # unit-level: the same value on every trait row
        @test all(isequal(X[t, :], x) for t in 1:p)
        z = reshape(Float64.(_dt2_load(csv, "z", p, n)[1, :]), n, 1)
        miss_units = Int.(fx["missing_x_units"])
        @test findall(ismissing, x) == miss_units

        dm = fx["miss_model"]
        @test dm["converged"] && dm["pd_hessian"]
        Ym = Float64.(_dt2_load(csv, "value", p, n))
        fm = fit_gaussian_mi_fiml(Ym, x; K = 1, Z = z)
        @test fm.converged
        @test isapprox(fm.logLik, Float64(dm["loglik"]); atol = 1e-6, rtol = 0)
        @test isapprox(fm.a, Float64.(dm["intercepts"]); atol = 1e-4, rtol = 0)
        @test isapprox(fm.b_x, Float64(dm["b_x"]); atol = 1e-4, rtol = 0)
        @test isapprox(vcat(fm.μ_x, fm.γ), [Float64(dm["mu_x"]), Float64(dm["gamma_z"])]; atol = 1e-4, rtol = 0)
        @test isapprox([fm.σ_x, fm.σ_eps], [Float64(dm["sigma_x"]), Float64(dm["sigma_eps"])]; atol = 1e-3, rtol = 0)
        @test isapprox(fm.Λ * fm.Λ', _dt2_mat(dm["LLt"], p); atol = 1e-3, rtol = 0)
        im = imputed(fm, x)
        @test im.level[.!im.observed] == Int.(dm["imputed_level_id"])
        @test isapprox(im.estimate[.!im.observed], Float64.(dm["imputed_estimate"]); atol = 1e-4, rtol = 0)

        db = fx["miss_both"]
        @test db["converged"] && db["pd_hessian"]
        Yb = _dt2_load(csv, "value_na", p, n)
        @test count(ismissing, Yb) == p * n - Int(db["nobs"])
        fb = fit_gaussian_mi_fiml(Yb, x; K = 1, Z = z)
        @test fb.converged
        @test fb.n_missing_y == p * n - Int(db["nobs"])
        @test isapprox(fb.logLik, Float64(db["loglik"]); atol = 1e-6, rtol = 0)
        @test isapprox(fb.a, Float64.(db["intercepts"]); atol = 1e-4, rtol = 0)
        @test isapprox(fb.b_x, Float64(db["b_x"]); atol = 1e-4, rtol = 0)
        @test isapprox(vcat(fb.μ_x, fb.γ), [Float64(db["mu_x"]), Float64(db["gamma_z"])]; atol = 1e-4, rtol = 0)
        @test isapprox([fb.σ_x, fb.σ_eps], [Float64(db["sigma_x"]), Float64(db["sigma_eps"])]; atol = 1e-3, rtol = 0)
        @test isapprox(fb.Λ * fb.Λ', _dt2_mat(db["LLt"], p); atol = 1e-3, rtol = 0)
        ib = imputed(fb, x)
        @test ib.level[.!ib.observed] == Int.(db["imputed_level_id"])
        @test isapprox(ib.estimate[.!ib.observed], Float64.(db["imputed_estimate"]); atol = 1e-4, rtol = 0)

        # The comparison discriminates: unit miss_units[1] lost one response as well as its x, and
        # its conditional mode under the response mask differs from the all-responses one by far
        # more than the tolerance (R and Julia agree on both).
        @test ismissing(Yb[1, miss_units[1]])
        @test abs(Float64(db["imputed_estimate"][1]) - Float64(dm["imputed_estimate"][1])) > 100 * 1e-4
    end
end
