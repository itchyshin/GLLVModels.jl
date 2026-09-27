# gllvm-parity-tag: P1
#
# Twin test for gllvmTMB's ordinal_logit() (family_id 20, gllvmTMB >= 0.7.1,
# pinned at commit 9539352f66f2db2cc26b1c393e67212a359b60c9 -- "P1" in the
# true-parity export ledger). No RCall: this file only reads R's recorded
# fitted values from test/fixtures/ordinal_logit_p1.toml (generated once by
# test/fixtures/gen_ordinal_logit_p1.R against a real gllvmTMB install at that
# commit; see that file's header for how to regenerate it) and replays the
# same dataset in Julia.
#
# Semantics: gllvmTMB's ordinal_logit() is a cumulative-logit threshold model
# with a per-trait intercept and per-trait cutpoints (tau_1 = 0 fixed, K_t - 2
# free log-spaced cutpoints). GLLVModels.jl's Ordinal() family already fits
# exactly this model by default (fit_ordinal_gllvm_pertrait, link =
# LogitLink()) -- this is the twin-default per the 2026-08-03
# ordinal-x-cutpoint-identity decision. ordinal_logit() is a thin, literal
# name-twin of Ordinal() so R's public export has a matching Julia name; it
# changes no numerics.
using Test
using GLLVModels
using TOML
using SHA

const _ORDLOGIT_FIXTURE_DIR = joinpath(@__DIR__, "fixtures")
const _ORDLOGIT_TOML = joinpath(_ORDLOGIT_FIXTURE_DIR, "ordinal_logit_p1.toml")

# Parse the fixture's (unit, trait, value) CSV into a p x n_unit Int matrix,
# trait rows in `trait_names` order. Written by hand (no CSV.jl test dep) --
# the file format is exactly what R's write.csv() emits for this data.frame.
function _load_ordinal_logit_csv(path::AbstractString, trait_names::Vector{String}, n_unit::Integer)
    Y = zeros(Int, length(trait_names), n_unit)
    open(path) do io
        readline(io)                          # header: "unit","trait","value"
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            trait = strip(parts[2], '"')
            value = parse(Int, parts[3])
            t = findfirst(==(trait), trait_names)
            t === nothing && error("unrecognised trait \"$trait\" in $path")
            Y[t, unit] = value
        end
    end
    return Y
end

@testset "ordinal_logit() API (name-twin of gllvmTMB's ordinal_logit())" begin
    @test ordinal_logit() isa Ordinal
    @test ordinal_logit(; link = LogitLink()) isa Ordinal
    # Mirrors gllvmTMB's `ordinal_logit(link = "probit")` refusal (naming
    # ordinal_probit()); see R/families.R's ordinal_logit().
    @test_throws ArgumentError ordinal_logit(; link = ProbitLink())
    # Dispatches identically to Ordinal() (same default link, same fitter).
    Y0 = [1 2 3 1 2 3 1 2 3 1 2 3
          1 2 1 2 1 2 1 2 1 2 1 2]
    f1 = fit_gllvm(Y0; family = Ordinal(), K = 1)
    f2 = fit_gllvm(Y0; family = ordinal_logit(), K = 1)
    @test f1.loglik == f2.loglik
    @test isequal(f1.τ, f2.τ)          # isequal: NaN padding compares equal to itself
end

@testset "ordinal_logit() twin: gllvmTMB P1 (9539352f6) fixed dataset" begin
    if !isfile(_ORDLOGIT_TOML)
        @warn "ordinal_logit P1 fixture absent; twin gate NOT RUN" _ORDLOGIT_TOML
        @test_skip false
    else
        fixture = TOML.parsefile(_ORDLOGIT_TOML)
        trait_names = String.(fixture["trait_names"])
        n_unit = Int(fixture["n_unit"])
        data_path = joinpath(_ORDLOGIT_FIXTURE_DIR, fixture["data_file"])

        # Guard: the checked-in CSV must be byte-identical to what R fitted.
        @test bytes2hex(sha256(read(data_path))) == fixture["data_sha256"]

        Y = _load_ordinal_logit_csv(data_path, trait_names, n_unit)
        @test size(Y) == (length(trait_names), n_unit)

        fit = fit_gllvm(Y; family = ordinal_logit(), K = 1)
        @test fit isa OrdinalPerTraitFit
        @test fit.converged
        @test fit.C == fill(4, length(trait_names))

        r = fixture["r_reference"]

        # (1) Direct optimum-vs-optimum: Julia's own converged logLik against
        # R's recorded logLik, at the tolerance the true-parity ledger asks
        # for a light logLik cell (1e-6 absolute).
        @test isapprox(fit.loglik, r["loglik"]; atol = 1e-6)

        # (2) Cross-objective: Julia's OWN Laplace marginal, evaluated AT R's
        # fitted (beta, Lambda, tau) rather than Julia's own optimum. This is
        # the likelihood-FUNCTION identity claim (both engines compute the
        # same Laplace approximation of the same model), independent of
        # either side's optimizer path.
        β_r = Float64.(r["b_fix"])
        λ_r = Float64.(r["lambda_b"])
        Λ_r = reshape(λ_r, length(trait_names), 1)
        cp2 = Float64.(r["cutpoint_2"])
        cp3 = Float64.(r["cutpoint_3"])
        τ_r = zeros(length(trait_names), 3)
        for t in eachindex(trait_names)
            τ_r[t, 2] = cp2[t]
            τ_r[t, 3] = cp3[t]
        end
        C = fill(4, length(trait_names))
        ll_at_r = GLLVModels.ordinal_marginal_loglik_laplace_pertrait(Y, Λ_r, β_r, τ_r, C;
            link = LogitLink())
        @test isapprox(ll_at_r, r["loglik"]; atol = 1e-6)

        # (3) Cutpoints by name (trait, cutpoint_2 / cutpoint_3), not by raw
        # packed position.
        for (t, name) in enumerate(trait_names)
            @test isapprox(fit.τ[t, 2], r["cutpoint_2"][t]; atol = 1e-3)
            @test isapprox(fit.τ[t, 3], r["cutpoint_3"][t]; atol = 1e-3)
        end

        # (4) Loadings via Lambda * Lambda' (K = 1 loadings are only
        # identified up to sign; the crossproduct is the sign-free twin
        # target R's alignment note calls for).
        LLt_julia = fit.Λ * fit.Λ'
        LLt_r = reduce(hcat, [Float64.(row) for row in r["lambda_lambda_t"]])'
        @test isapprox(LLt_julia, Matrix(LLt_r); atol = 1e-3)

        # (5) Fixed logistic residual variance, exact on both sides.
        @test isapprox(pi^2 / 3, r["sigma_d2"]; atol = 1e-12)
    end
end
