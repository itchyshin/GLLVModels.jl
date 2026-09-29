# Writes Julia's fitted parameters for the three zi_* P1 fixtures in the plain-text
# format test/fixtures/gen_zi_p1.R (phase 2) reads, so R can evaluate its own TMB
# objective at Julia's optimum (the R-side half of the cross-objective check). NOT
# run by CI or by any test; provenance only. Usage, from the repo root:
#   julia --project=. test/fixtures/gen_zi_p1_julia_params.jl <out.txt>
using GLLVModels

const _DIR = @__DIR__

function _read_zi_csv(path, p)
    rows = [split(l, ",") for l in readlines(path)[2:end] if !isempty(l)]
    n = maximum(parse(Int, r[1]) for r in rows)
    Y = zeros(Int, p, n)
    N = zeros(Int, p, n)
    for r in rows
        s = parse(Int, r[1]); t = parse(Int, r[2])
        Y[t, s] = parse(Int, r[3])
        length(r) >= 4 && (N[t, s] = parse(Int, r[4]))
    end
    return Y, N
end

fmt(x) = join((repr(Float64(v)) for v in x), " ")

out = ARGS[1]
open(out, "w") do io
    for (name, fam) in (("zi_poisson", zi_poisson()), ("zi_nbinom2", zi_nbinom2()),
                        ("zi_binomial", zi_binomial()))
        Y, N = _read_zi_csv(joinpath(_DIR, "$(name)_p1_data.csv"), 3)
        fit = fam isa ZiBinomial ? fit_zi_gllvm(Y; family = fam, K = 1, trials = N) :
                                   fit_zi_gllvm(Y; family = fam, K = 1)
        println(io, name, " beta ", fmt(fit.β))
        println(io, name, " theta_rr_B ", fmt(GLLVModels.pack_lambda(fit.Λ)))
        println(io, name, " logit_zi ", fmt(fit.logit_zi))
        isempty(fit.phi) || println(io, name, " log_phi ", fmt(log.(fit.phi)))
        println(io, name, " loglik ", fmt([fit.loglik]))
    end
end
