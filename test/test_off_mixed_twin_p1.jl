# gllvm-parity-tag: P1
#
# Fit-level twin for the core070 row data/DATA-OFF-MIXED against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). The R batch case
# gll_prepare_offset(quote(c(1,0,4)), c(2L,0L,5L), ...) admits, in one mixed-family fit with family
# ids 2, 0, 5 (poisson, gaussian, nbinom2), a zero offset on the non-count row next to nonzero
# offsets on the count rows, and replays to c(1, 0, 4) with no fit number
# (docs/dev-log/core070/data-required-case-plan.json). Here that is one real mixed-family fit with an
# exposure offset that is zero on the gaussian trait, in R and in Julia, and the maximised logLik
# and the parameters are compared. The refusal half is checked too: R refuses the same fit with a
# nonzero offset on the gaussian trait (message recorded in the fixture) and Julia raises an
# ArgumentError naming the trait and family. No R at test time: R's values are read from
# test/fixtures/off_mixed_twin_p1.toml (generated once by test/fixtures/gen_off_mixed_twin_p1.R
# against a lane-local gllvmTMB install at the pin; the file records R version, commit and the data
# sha256). The R fit converged with a positive-definite Hessian (asserted there).
#
# R: value ~ 0 + trait + offset(log(e)) + latent(0 + trait | unit, d = 1, unique = FALSE), p = 6,
# n = 150, family = list(poisson, gaussian, nbinom2, poisson, nbinom2, poisson) by trait.
# Julia: fit_mixed_gllvm(Y; families, K = 1, offset = log.(E)). One gaussian trait only: gllvmTMB
# shares one sigma across the gaussian traits of a mixed fit, Julia gives each Normal trait its own,
# and with one gaussian trait the two coincide. nbinom2 dispersion is per trait on both sides.
#
# Tolerances are those of the sibling data twins (test_data_twins_p1.jl), not loosened to pass:
# logLik 1e-6 (absolute), intercepts 1e-4, Lambda Lambda' 1e-3, dispersion 1e-3. The sign of a K = 1
# loading axis is not identified, so loadings are compared through Lambda Lambda'.
using Test
using GLLVModels
using Distributions: Poisson, Normal, NegativeBinomial
using TOML
using SHA

const _OMX_DIR = joinpath(@__DIR__, "fixtures")
const _OMX_TOML = joinpath(_OMX_DIR, "off_mixed_twin_p1.toml")

# R's write.csv long data (unit, trait "t#", then named columns) -> p x n matrix.
function _omx_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
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

_omx_mat(v, p) = permutedims(reshape(Float64.(v), p, p))

const _OMX_FAMILY = Dict("poisson" => (2, () -> Poisson()), "gaussian" => (0, () -> Normal()),
                         "nbinom2" => (5, () -> NegativeBinomial(1.0, 0.5)))

@testset "mixed poisson/gaussian/nbinom2 fit, offset zero on the gaussian trait: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_OMX_TOML)
        @warn "off-mixed P1 fixture absent; twin gate NOT RUN" _OMX_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_OMX_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test fx["converged"] && fx["pd_hessian"]
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        csv = joinpath(_OMX_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(csv))) == fx["data_sha256"]
        famnames = String.(fx["families"])
        @test [_OMX_FAMILY[k][1] for k in famnames] == Int.(fx["family_ids"])
        @test Set(fx["family_ids"]) == Set([2, 0, 5])        # the family ids of the R batch case
        fams = [_OMX_FAMILY[k][2]() for k in famnames]
        Y = _omx_load(csv, fx["response_column"], p, n)
        E = _omx_load(csv, fx["exposure_column"], p, n)
        O = log.(E)
        gauss = findall(==("gaussian"), famnames)
        cnt = findall(!=("gaussian"), famnames)
        @test length(gauss) == 1
        # The admission the R case asserts: zero offset on the non-count trait, nonzero on the counts.
        @test all(iszero, O[gauss, :])
        @test all(!iszero, O[cnt, :])

        f = fit_mixed_gllvm(Y; families = fams, K = 1, offset = O)
        @test f.converged
        @test f.offset == O                                   # the training offset is kept
        @test isapprox(f.loglik, Float64(fx["loglik"]); atol = 1e-6, rtol = 0)
        @test isapprox(f.β, Float64.(fx["beta"]); atol = 1e-4, rtol = 0)
        @test isapprox(f.Λ * f.Λ', _omx_mat(fx["lambda_lambdat"], p); atol = 1e-3, rtol = 0)
        nb = findall(==("nbinom2"), famnames)
        @test isapprox(f.dispersion[nb], Float64.(fx["phi"]); atol = 1e-3, rtol = 0)
        @test isapprox(f.dispersion[only(gauss)], Float64(fx["sigma"]); atol = 1e-3, rtol = 0)
        # R values are not degenerate: pairwise distinct intercepts
        @test length(unique(round.(Float64.(fx["beta"]); digits = 3))) == p

        # The offset is applied, not ignored: at the fitted values the marginal with the offset
        # reproduces the fit's logLik and without it is far lower, and the offset-free refit (R's
        # recorded no-offset fit, matched here) scores lower than the offset fit by more than 1.
        fams_hat = [k == "poisson" ? Poisson() : k == "gaussian" ? Normal(0.0, f.dispersion[t]) :
                    NegativeBinomial(f.dispersion[t], 0.5) for (t, k) in enumerate(famnames)]
        Nm = ones(Int, p, n)
        marg(off) = mixed_marginal_loglik_laplace(fams_hat, f.links, Y, Nm, f.Λ, f.β; offset = off)
        @test isapprox(marg(O), f.loglik; atol = 1e-6, rtol = 0)
        @test f.loglik - marg(nothing) > 1
        f0 = fit_mixed_gllvm(Y; families = fams, K = 1)
        @test f0.converged
        @test f0.offset === nothing
        @test MixedFamilyFit(f.β, f.Λ, f.families, f.links, f.dispersion, f.disp_index, f.n_disp,
                             f.link, f.loglik, f.converged, f.iterations).offset === nothing   # old arity
        @test isapprox(f0.loglik, Float64(fx["loglik_no_offset"]); atol = 1e-6, rtol = 0)
        @test f.loglik - f0.loglik > 1

        # The training offset is used by the post-fit calls on the training data (#787's rule, extended in #807).
        @test predict(f, Y; type = :link) == predict(f, Y; type = :link, offset = O)
        @test getLV(f, Y) == getLV(f, Y; offset = O)
        @test maximum(abs, predict(f, Y; type = :link) .- predict(f, Y; type = :link, offset = 0.0)) > 0.1
        @test_throws ArgumentError predict(f, Y[:, 1:10])     # new units need their own offset
        @test size(predict(f, Y[:, 1:10]; offset = O[:, 1:10])) == (p, 10)

        # The refusal half of the row-wise admission (gll_prepare_offset, R/offset.R at P1): R
        # refuses this fit with a nonzero offset on the gaussian trait (recorded by the generator),
        # and so does Julia, naming the trait and family. Zero on that trait is accepted (above).
        @test startswith(fx["nonzero_gaussian_offset_refusal"], "offsets are supported for count families")
        @test occursin("trait t2 uses gaussian", fx["nonzero_gaussian_offset_refusal"])
        Obad = copy(O); Obad[only(gauss), :] .= 0.5
        @test_throws ArgumentError fit_mixed_gllvm(Y; families = fams, K = 1, offset = Obad)
        msg = try
            fit_mixed_gllvm(Y; families = fams, K = 1, offset = Obad); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("offsets are supported for count families", msg)
        @test occursin("trait $(only(gauss)) uses Normal", msg)
        Obad1 = copy(O); Obad1[only(gauss), 7] = -0.25             # one nonzero cell is enough
        @test_throws ArgumentError fit_mixed_gllvm(Y; families = fams, K = 1, offset = Obad1)
        @test_throws ArgumentError fit_mixed_gllvm(Y; families = fams, K = 1, offset = 0.5)
    end
end
