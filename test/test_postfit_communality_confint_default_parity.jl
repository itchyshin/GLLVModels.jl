# gllvm-parity-tag: P1
#
# Post-fit parity checks against gllvmTMB at P1 (9539352f66f2db2cc26b1c393e67212a359b60c9),
# tests only. Refs #131 (communality / correlation total-variance denominator) and #1020
# (confint default method differs: R "profile" vs Julia :wald).
#
# No R at test time. R's recorded values are read from the existing fixture
# test/fixtures/variance_decomp_p1.toml (see gen_variance_decomp_p1.R). No new R output is
# invented here: the tier-local communality R's extract_communality() reports is
# shared / (shared + unique) at the SAME tier (issue #131), derived from R's recorded
# extract_proportions() `variance` column, which holds the shared and unique components.
#
#   1. one-level has_diag fit: Julia communality(fit) / extract_communality(fit) vs R's
#      shared_unit proportion (tier-local = full, since R fixes sigma_eps ~ 0 there).
#   2. two-level fit: Julia communality_B / communality_W vs R tier-local
#      shared_unit/(shared_unit+unique_unit) and shared_unit_obs/(shared_unit_obs+unique_unit_obs).
#   3. #1020: Julia's confint default is :wald; R's default is "profile". The agreement
#      "default == profile" is @test_broken (a documented default difference; no default is changed).
#
# Result at this commit: communality (both fits) agrees with R's tier-local value, so #131's
# disagreement is not reproduced for extract_communality / communality_B/W / the has_diag
# GllvmFit; the raw communality(::GllvmFit) full-Sigma_y_site denominator with sigma_eps > 0 is
# not exercised here because no R fixture records that case.
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA

const _CD_DIR = joinpath(@__DIR__, "fixtures")

function _cd_load_csv(path::AbstractString, trait_names::Vector{String}, n_cols::Integer;
                      obs_col::Union{Nothing,Int} = nothing)
    p = length(trait_names)
    Y = zeros(Float64, p, n_cols)
    ind = zeros(Int, n_cols)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            col = obs_col === nothing ? unit : parse(Int, strip(parts[obs_col], '"'))
            t = findfirst(==(strip(parts[obs_col === nothing ? 2 : 3], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, col] = parse(Float64, parts[obs_col === nothing ? 3 : 4])
            ind[col] = unit
        end
    end
    return Y, ind
end

@testset "post-fit parity: communality and confint default (refs #131 #1020)" begin
    vdf = joinpath(_CD_DIR, "variance_decomp_p1.toml")
    if !isfile(vdf)
        @warn "variance_decomp_p1 fixture absent; gate NOT RUN" vdf
        @test_skip false
    else
        fx = TOML.parsefile(vdf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

        @testset "one-level has_diag: communality vs R shared_unit proportion" begin
            u = fx["unique"]
            dp = joinpath(_CD_DIR, u["data_file"])
            @test bytes2hex(sha256(read(dp))) == u["data_sha256"]
            Y, _ = _cd_load_csv(dp, String.(u["trait_names"]), Int(u["n_unit"]))
            fit = fit_gllvm(Y; family = Normal(), K = 2, has_diag = true)
            @test fit.converged
            r_sh = Float64.(u["shared_unit_proportion"])
            @test isapprox(extract_communality(fit; level = :unit), r_sh; atol = 2e-5, rtol = 0)
            @test isapprox(GLLVModels.communality(fit), r_sh; atol = 2e-5, rtol = 0)
        end

        @testset "confint default method (#1020)" begin
            # Identified rank-2 fit (no has_diag), the model of confint_inspect_p1 / postfit_twins_p1.
            u = fx["unique"]
            dp = joinpath(_CD_DIR, u["data_file"])
            Y, _ = _cd_load_csv(dp, String.(u["trait_names"]), Int(u["n_unit"]))
            fit = fit_gllvm(Y; family = Normal(), K = 2)
            @test fit.converged
            ci_def = confint(fit, Y; parm = "sigma_eps")
            ci_w = confint(fit, Y; parm = "sigma_eps", method = :wald)
            ci_p = confint(fit, Y; parm = "sigma_eps", method = :profile)
            @test all(isfinite, ci_w.lower) && all(isfinite, ci_p.lower)
            # Julia's own default is Wald, as documented.
            @test isequal(ci_def.lower, ci_w.lower) && isequal(ci_def.upper, ci_w.upper)
            # R's confint() defaults to method = "profile". Julia's default interval is not the
            # profile interval, so default-vs-default parity does not hold (#1020). No default changed.
            @test_broken isapprox(ci_def.lower, ci_p.lower; atol = 1e-8, rtol = 0) &&
                         isapprox(ci_def.upper, ci_p.upper; atol = 1e-8, rtol = 0)
        end

        @testset "two-level: communality_B/W vs R tier-local shared/(shared+unique)" begin
            t = fx["two"]
            dp = joinpath(_CD_DIR, t["data_file"])
            @test bytes2hex(sha256(read(dp))) == t["data_sha256"]
            Y, ind = _cd_load_csv(dp, String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
            fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)
            @test fit.converged
            p = Int(t["p"])
            comps = String.(t["components"])
            rv = Float64.(t["variance"])
            blk(name) = rv[(findfirst(==(name), comps) - 1) * p .+ (1:p)]
            r_cB = blk("shared_unit") ./ (blk("shared_unit") .+ blk("unique_unit"))
            r_cW = blk("shared_unit_obs") ./ (blk("shared_unit_obs") .+ blk("unique_unit_obs"))
            @test length(unique(round.(r_cB; digits = 3))) > 2   # R values not degenerate
            @test isapprox(communality_B(fit), r_cB; atol = 2e-4, rtol = 0)
            @test isapprox(communality_W(fit), r_cW; atol = 2e-4, rtol = 0)
            @test isapprox(extract_communality(fit; level = :unit), r_cB; atol = 2e-4, rtol = 0)
            @test isapprox(extract_communality(fit; level = :unit_obs), r_cW; atol = 2e-4, rtol = 0)
        end
    end
end
