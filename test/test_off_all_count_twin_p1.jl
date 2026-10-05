# gllvm-parity-tag: P1
#
# Fit-level twin for the core070 row data/DATA-OFF-ALL-COUNT against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). The R batch case prepares one exposure offset for three
# count families (family ids 5, 10, 11: nbinom2, truncated Poisson, truncated nbinom2) and replays to
# a vector with no fit number. Here each of the three families is fitted with an exposure offset, in R
# and in Julia, and the maximised logLik and the parameters are compared. No R at test time: R's
# values are read from test/fixtures/off_all_count_twin_p1.toml (generated once by
# test/fixtures/gen_off_all_count_twin_p1.R against a lane-local gllvmTMB install at the pin; the file
# records R version, commit and the data sha256). Each R fit converged with a positive-definite
# Hessian, and no per-trait dispersion is at the Poisson limit (asserted there).
#
# R: value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE), p = 6,
# n = 150, with family = nbinom2(), truncated_poisson() or truncated_nbinom2(), one data set each.
# Julia: fit_gllvm(Y; family, K = 1, offset = log.(E)), with disp_group = :species for the two NB2
# families (per-trait dispersion, as in R). Julia has no fit that mixes families with an offset, so
# the three families are three single-family fits, not R's one mixed-family model.
#
# Tolerances are those of the sibling data twins (test_data_twins_p1.jl), not loosened to pass:
# logLik 1e-6 (absolute), intercepts 1e-4, Lambda Lambda' 1e-3, dispersion 1e-3. The sign of a K = 1
# loading axis is not identified, so loadings are compared through Lambda Lambda'.
using Test
using GLLVModels
using Distributions: NegativeBinomial
using TOML
using SHA

const _OAC_DIR = joinpath(@__DIR__, "fixtures")
const _OAC_TOML = joinpath(_OAC_DIR, "off_all_count_twin_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns) -> p x n matrix.
function _oac_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
    hdr = split(readline(path), ",")
    ci = findfirst(==("\"" * col * "\""), hdr)
    ci === nothing && error("column $col not found in $path")
    M = Matrix{Float64}(undef, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            M[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] = parse(Float64, a[ci])
        end
    end
    return M
end

_oac_mat(v, p) = permutedims(reshape(Float64.(v), p, p))

@testset "exposure offset on nbinom2, truncated Poisson, truncated nbinom2: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_OAC_TOML)
        @warn "off-all-count P1 fixture absent; twin gate NOT RUN" _OAC_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_OAC_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        csv = joinpath(_OAC_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(csv))) == fx["data_sha256"]

        # (section, family id, Julia family, extra keywords, dispersion field, marginal at a point)
        cases = (
            ("nb2", 5, NegativeBinomial(1.0, 0.5), (; disp_group = :species), :r_group,
             (Y, f, off) -> nb_grouped_marginal_loglik_laplace(Y, f.Λ, f.β, f.r_group; offset = off)),
            ("tpois", 10, TruncatedPoisson(), (;), nothing,
             (Y, f, off) -> truncated_poisson_marginal_loglik_laplace(Y, f.Λ, f.β; offset = off)),
            ("tnb2", 11, TruncatedNegBin2(), (; disp_group = :species), :r,
             (Y, f, off) -> truncated_nbinom2_pertrait_marginal_loglik_laplace(Y, f.Λ, f.β, f.r; offset = off)))
        @test [c[2] for c in cases] == [5, 10, 11]          # the family ids of the R batch case
        for (sec, fid, fam, kw, dfld, marg) in cases
            @testset "$sec (family id $fid)" begin
                d = fx[sec]
                @test d["family_id"] == fid
                @test d["converged"] && d["pd_hessian"]
                Y = Int.(_oac_load(csv, d["response_column"], p, n))
                E = _oac_load(csv, d["exposure_column"], p, n)
                fid == 5 || @test all(>=(1), Y)                # zero-truncated support
                f = fit_gllvm(Y; family = fam, K = 1, offset = log.(E), kw...)
                @test f.converged
                @test isapprox(f.loglik, Float64(d["loglik"]); atol = 1e-6, rtol = 0)
                @test isapprox(f.β, Float64.(d["beta"]); atol = 1e-4, rtol = 0)
                @test isapprox(f.Λ * f.Λ', _oac_mat(d["lambda_lambdat"], p); atol = 1e-3, rtol = 0)
                if dfld !== nothing
                    @test length(getproperty(f, dfld)) == p    # per trait, as in R
                    @test isapprox(getproperty(f, dfld), Float64.(d["phi"]); atol = 1e-3, rtol = 0)
                end
                # The offset is applied, not ignored: at the fitted values the marginal with the
                # offset reproduces the fit's logLik, and without it is lower by far more than any
                # tolerance.
                @test isapprox(marg(Y, f, log.(E)), f.loglik; atol = 1e-6, rtol = 0)
                @test f.loglik - marg(Y, f, nothing) > 1
                # R values are not degenerate: pairwise distinct intercepts
                @test length(unique(round.(Float64.(d["beta"]); digits = 3))) == p
            end
        end
    end
end
