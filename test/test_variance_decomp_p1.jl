# gllvm-parity-tag: P1
#
# Numeric twins of gllvmTMB's extract_proportions() and extract_residual_split() at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9): postfit row POSTFIT-SURFACE-extract_proportions and
# namespace rows export/extract_proportions and export/extract_residual_split. No R at test time:
# R's recorded values are read from test/fixtures/variance_decomp_p1.toml (generated once by
# test/fixtures/gen_variance_decomp_p1.R against a lane-local gllvmTMB install at the pin; it records
# R version, commit and the data sha256s). Both R fits converged with a positive-definite Hessian
# (asserted).
#
# unique : one tier, value ~ 0 + trait + latent(0 + trait | unit, d = 2) + unique(0 + trait | unit),
#          p = 6, 200 units, one observation per cell. R's shared_unit proportion is
#          diag(Lambda Lambda') / (diag(Lambda Lambda') + Psi) with Psi absorbing the whole residual.
#          <-> fit_gllvm(Y; family = Normal(), K = 2, has_diag = true) and extract_proportions(fit),
#          whose :unit denominator is the identified total sigma_y_site(fit) on a has_diag fit with
#          K_W == 0 (#701). Only the shared_unit proportion is compared: the GllvmFit method returns
#          that vector alone.
# two    : two-level, p = 5, 120 units x 4 observations, latent(d = 1) + unique at unit and at unit_obs
#          <-> fit_twolevel_gaussian(Y, individual; K_B = 1, K_W = 1). R's extract_proportions() (all
#          four components: variance and proportion, in R's row order) and extract_residual_split()
#          (sigma2_d, sigma2_e, sigma2_total) against the TwoLevelFit methods.
#
# Disclosed: the R fits are made on the in-memory data (as for the namespace twins). Written
# with write.csv they are byte-identical to the tracked CSVs (the generator checks the sha256),
# so the doubles Julia reads from the CSV agree with R's to about 5e-15. R's two-level fit on the
# CSV-read data instead stops with nlminb code 1 ("false convergence (8)", gradient ~2e-3) at the
# same log-likelihood; the recorded fit is the in-memory one, which converged with code 0.
#
# Limits, disclosed: Gaussian only, so R's sigma2_d is 0 for every trait (identity link) and the
# non-Gaussian link-residual values of extract_residual_split()/extract_proportions() are not
# reachable (the package has no non-Gaussian two-level fit). The comparison is of the Julia accessor
# against R's accessor, each on its own fit of the same data (same optimum, log-likelihoods compared).
using Test
using GLLVModels
using Distributions: Normal
using TOML
using SHA

const _VD_DIR = joinpath(@__DIR__, "fixtures")

# R's write.csv long data -> p x n_cols response matrix and the unit index per column.
# `obs_col` = column holding the observation id (two-level data), `nothing` for one row per unit.
function _vd_load_csv(path::AbstractString, trait_names::Vector{String}, n_cols::Integer;
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

@testset "variance decomposition twins: gllvmTMB P1 (9539352f6)" begin
    vdf = joinpath(_VD_DIR, "variance_decomp_p1.toml")
    if !isfile(vdf)
        @warn "variance decomposition P1 fixture absent; twin gate NOT RUN" vdf
        @test_skip false
    else
        fx = TOML.parsefile(vdf)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

        @testset "unique: extract_proportions on a has_diag GllvmFit" begin
            u = fx["unique"]
            dp = joinpath(_VD_DIR, u["data_file"])
            @test bytes2hex(sha256(read(dp))) == u["data_sha256"]
            @test u["converged"] && u["pd_hessian"]
            Y, _ = _vd_load_csv(dp, String.(u["trait_names"]), Int(u["n_unit"]))
            fit = fit_gllvm(Y; family = Normal(), K = 2, has_diag = true)
            @test fit.converged
            @test isapprox(fit.logLik, Float64(u["loglik"]); atol = 1e-6, rtol = 0)   # same optimum as R
            r_sh = Float64.(u["shared_unit_proportion"])
            # not degenerate: not one constant, and strictly inside (0, 1)
            @test all(0 .< r_sh .< 1) && length(unique(round.(r_sh; digits = 3))) > 3
            @test isapprox(extract_proportions(fit), r_sh; atol = 2e-5, rtol = 0)   # shared_unit proportion (observed 2.5e-6)
        end

        @testset "two: extract_proportions and extract_residual_split on a TwoLevelFit" begin
            t = fx["two"]
            dp = joinpath(_VD_DIR, t["data_file"])
            @test bytes2hex(sha256(read(dp))) == t["data_sha256"]
            @test t["converged"] && t["pd_hessian"]
            Y, ind = _vd_load_csv(dp, String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
            fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)
            @test fit.converged
            @test isapprox(fit.loglik, Float64(t["loglik"]); atol = 1e-6, rtol = 0)   # same optimum as R
            p = Int(t["p"])
            comps = Symbol.(String.(t["components"]))
            pr = extract_proportions(fit)
            # same rows, same order as R's long frame: component-major, trait index within
            @test pr.component == repeat(comps; inner = p)
            @test pr.trait == repeat(collect(1:p), length(comps))
            r_prop = Float64.(t["proportion"])
            r_var = Float64.(t["variance"])
            @test length(unique(round.(r_prop; digits = 3))) > length(r_prop) ÷ 2   # R values are not degenerate
            @test isapprox(pr.proportion, r_prop; atol = 2e-4, rtol = 0)   # proportion, all four components (observed 2.4e-5)
            @test isapprox(pr.variance, r_var; atol = 3e-4, rtol = 0)      # variance, all four components (observed 4.8e-5)
            # the proportions of each trait sum to one
            @test all(isapprox.(sum(reshape(pr.proportion, p, length(comps)); dims = 2), 1.0; atol = 1e-12))
            wide = extract_proportions(fit; format = :wide)
            @test collect(keys(wide)) == [:trait; comps; :total_variance]
            @test isapprox(wide.total_variance, vec(sum(reshape(r_var, p, length(comps)); dims = 2)); atol = 3e-4, rtol = 0)

            rs = extract_residual_split(fit)
            @test keys(rs) == (:trait, :sigma2_d, :sigma2_e, :sigma2_total)
            r_e = Float64.(t["sigma2_e"])
            @test length(unique(round.(r_e; digits = 3))) == p                      # five distinct R values
            @test all(==(0.0), Float64.(t["sigma2_d"])) && all(==(0.0), rs.sigma2_d)   # Gaussian: no link residual
            @test isapprox(rs.sigma2_e, r_e; atol = 2e-5, rtol = 0)                 # sigma2_e (observed 2.7e-6)
            @test isapprox(rs.sigma2_total, Float64.(t["sigma2_total"]); atol = 2e-5, rtol = 0)   # sigma2_total (observed 2.7e-6)
            # sigma2_e is the unique_unit_obs component of extract_proportions
            @test rs.sigma2_e == pr.variance[pr.component .== :unique_unit_obs]
        end
    end
end
