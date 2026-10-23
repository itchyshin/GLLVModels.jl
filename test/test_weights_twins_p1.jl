# gllvm-parity-tag: P1
#
# Fit-level twins for the 13 core070 rows data/DATA-W-* against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9). Each row's R batch case replays the internal shape
# helper normalise_weights() (R/weights-shape.R) to an identical() expectation, with no fit number.
# Here each weight shape is used in a real Poisson fit, in R and in Julia, and the maximised
# weighted objective and the parameters are compared. No R at test time: R's values are read from
# test/fixtures/weights_twins_p1.toml (generated once by test/fixtures/gen_weights_twins_p1.R
# against a lane-local gllvmTMB install at the pin; the file records R version, commit and the data
# sha256). Every R fit converged with a positive-definite Hessian (asserted there).
#
# R: value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE), family = poisson(),
# p = 6, n = 120, on three routes: the long API (weights = a length-nrow(data) vector), the wide
# matrix route (gllvmTMB_wide()'s own normalise_weights(..., "wide_matrix") and pivot, refitted with
# unique = FALSE because gllvmTMB_wide() hardcodes latent()'s default unique = TRUE), and the
# traits() route (weights = one per unit). Julia: fit_gllvm(Y; family = Poisson(), K = 1, weights),
# with Y p x n (traits x units): a scalar, a length-n vector (one per unit), or a p x n matrix.
# gllvmTMB's logLik() aborts for non-unit weights, so the compared number is the maximised weighted
# objective (R -fit$opt$objective, Julia fit.loglik); for the unweighted fit it is logLik().
#
# Tolerances are those of the sibling data twins (test_data_twins_p1.jl), not loosened to pass:
# objective 1e-6 (absolute), intercepts 1e-4, Lambda Lambda' 1e-3. The sign of a K = 1 loading axis
# is not identified, so loadings are compared through Lambda Lambda'.
using Test
using GLLVModels
using Distributions: Poisson
using TOML
using SHA

const _WT_DIR = joinpath(@__DIR__, "fixtures")
const _WT_TOML = joinpath(_WT_DIR, "weights_twins_p1.toml")

# R's write.csv long data (unit, "t#" trait, then named columns; "NA" for missing) -> p x n matrix.
function _wt_load(path::AbstractString, col::AbstractString, p::Integer, n::Integer)
    hdr = split(readline(path), ",")
    ci = findfirst(==("\"" * col * "\""), hdr)
    ci === nothing && error("column $col not found in $path")
    M = Matrix{Union{Missing, Float64}}(missing, p, n)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            M[parse(Int, strip(a[2], ['"', 't'])), parse(Int, strip(a[1], '"'))] =
                a[ci] == "NA" ? missing : parse(Float64, a[ci])
        end
    end
    return M
end

_wt_mat(v, p) = permutedims(reshape(Float64.(v), p, p))

# Julia call for each fixture section: (Y, weights, extra keywords), from the loaded columns.
function _wt_julia_inputs(sec, col, p, n)
    Y   = Int.(col("y"))                                   # complete counts
    Yna = map(v -> ismissing(v) ? missing : Int(v), col("y_na"))   # 14 cells missing
    obs = .!ismissing.(Yna)
    Yfill = Int.(coalesce.(Yna, 0))                         # placeholder 0 at the masked cells
    Wcell = Float64.(col("w_cell"))
    Wcell_na = ifelse.(obs, Wcell, NaN)                     # NaN where Y is NA (R: NA there)
    wunit = Float64.(col("w_unit")[1, :])                    # one weight per unit
    return Dict(
        "null"                => (Y, nothing, (;)),
        "long"                => (Y, Float64.(col("w_int")), (;)),
        "fractional"          => (Y, Float64.(col("w_frac")), (;)),
        "zero"                => (Y, Float64.(col("w_zero")), (;)),
        "matrix_scalar"       => (Y, 2.0, (;)),
        "matrix_unit"         => (Y, wunit, (;)),
        "matrix_cells"        => (Y, permutedims(permutedims(Wcell)), (;)),   # R's n x p, transposed
        "matrix_mask_drop"    => (Yna, Wcell_na, (;)),
        "matrix_mask_include" => (Yfill, Wcell_na, (; mask = obs)),
        "matrix_mask_scalar"  => (Yna, 2.0, (;)),
        "df_unit"             => (Y, wunit, (;)),
        "df_mask_drop"        => (Yna, wunit, (;)),
        "df_mask_include"     => (Yfill, wunit, (; mask = obs)))[sec]
end

const _WT_SECTIONS = ("null", "long", "fractional", "zero", "matrix_scalar", "matrix_unit",
    "matrix_cells", "matrix_mask_drop", "matrix_mask_include", "matrix_mask_scalar",
    "df_unit", "df_mask_drop", "df_mask_include")

@testset "observation weights on a Poisson fit: gllvmTMB P1 (9539352f6)" begin
    if !isfile(_WT_TOML)
        @warn "weights P1 fixture absent; twin gate NOT RUN" _WT_TOML
        @test_skip false
    else
        fx = TOML.parsefile(_WT_TOML)
        @test fx["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        p, n = Int(fx["p"]), Int(fx["n_unit"])
        csv = joinpath(_WT_DIR, fx["data_file"])
        @test bytes2hex(sha256(read(csv))) == fx["data_sha256"]
        col = c -> _wt_load(csv, c, p, n)
        @test Set(_WT_SECTIONS) == Set(k for k in keys(fx) if fx[k] isa Dict)
        @test length(_WT_SECTIONS) == 13
        # the NA cells of y_na are the ones the fixture lists
        Yna = col("y_na")
        @test sort([(Int(u), Int(t)) for (u, t) in zip(fx["na_units"], fx["na_traits"])]) ==
              sort([(s, t) for t in 1:p, s in 1:n if ismissing(Yna[t, s])])
        null_obj = Float64(fx["null"]["objective"])
        @test null_obj == Float64(fx["null"]["loglik"])
        fits = Dict{String, Any}()
        for sec in _WT_SECTIONS
            @testset "$sec ($(fx[sec]["source_id"]))" begin
                d = fx[sec]
                @test d["converged"] && d["pd_hessian"]
                Y, W, kw = _wt_julia_inputs(sec, col, p, n)
                f = fit_gllvm(Y; family = Poisson(), K = 1, weights = W, kw...)
                fits[sec] = f
                @test f.converged
                @test isapprox(f.loglik, Float64(d["objective"]); atol = 1e-6, rtol = 0)
                @test isapprox(f.β, Float64.(d["beta"]); atol = 1e-4, rtol = 0)
                @test isapprox(f.Λ * f.Λ', _wt_mat(d["lambda_lambdat"], p); atol = 1e-3, rtol = 0)
                # The weights are applied, not ignored: every weighted R objective is far from the
                # unweighted one (the null section is the unweighted fit itself).
                sec == "null" || @test abs(Float64(d["objective"]) - null_obj) > 1
                @test (W === nothing) == (f.weights === nothing)
            end
        end
        # weights = nothing is the unweighted fit, and `nothing` equals omitting the keyword
        @test fit_gllvm(Int.(col("y")); family = Poisson(), K = 1).loglik == fits["null"].loglik
        # weight 0 drops a cell: the zero-weight fit equals the fit with those cells masked
        Wz = Float64.(col("w_zero"))
        fm = fit_gllvm(Int.(col("y")); family = Poisson(), K = 1, mask = Wz .!= 0)
        @test isapprox(fm.loglik, fits["zero"].loglik; atol = 1e-6)
        # drop and include reach the same optimum on both sides
        @test fx["matrix_mask_drop"]["objective"] ≈ fx["matrix_mask_include"]["objective"] atol = 1e-6
        @test fx["df_mask_drop"]["objective"] ≈ fx["df_mask_include"]["objective"] atol = 1e-6
        # one weight per unit is the same objective through the wide-matrix and traits() routes
        @test fx["matrix_unit"]["objective"] ≈ fx["df_unit"]["objective"] atol = 1e-6
    end
end
