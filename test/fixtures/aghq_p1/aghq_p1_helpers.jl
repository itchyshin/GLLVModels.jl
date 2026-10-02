# Shared by test/test_aghq_p1_twin.jl and tools/true_parity_julia_receipts.jl, so the receipt
# tool repeats exactly the computation the test asserts on. No test-only dependency.
# Needs `GLLVModels`, `TOML`, `SHA` and `Distributions.Normal` in scope at the include site.

const AGHQ_P1_DIR = joinpath(@__DIR__)
const AGHQ_P1_TOML = joinpath(AGHQ_P1_DIR, "aghq_p1.toml")

# (unit, trait, value[, trials]) CSV from R's write.csv -> p x n responses and trials.
function aghq_p1_load(ds)
    tn = String.(ds["trait_names"]); p, n = Int(ds["p"]), Int(ds["n_unit"])
    Y = zeros(Float64, p, n); N = fill(1.0, p, n)
    open(joinpath(AGHQ_P1_DIR, ds["file"])) do io
        hdr = readline(io); has_trials = occursin("trials", hdr)
        for line in eachline(io)
            isempty(line) && continue
            q = split(line, ",")
            u = parse(Int, strip(q[1], '"')); t = findfirst(==(strip(q[2], '"')), tn)
            t === nothing && error("unrecognised trait in $(ds["file"])")
            Y[t, u] = parse(Float64, q[3]); has_trials && (N[t, u] = parse(Float64, q[4]))
        end
    end
    return Y, N
end

aghq_p1_request(s::AbstractString) = s == "default" || s == "false" ? false : s == "auto" ? :auto : parse(Int, s)

"""Fit the case's dataset in Julia with the same request R was given. Returns the quantities
the twin compares. Loadings are sign-aligned to R's (a fixed +-1 on the single latent axis)."""
function aghq_p1_fit(fx, caseid)
    c = fx["case"][caseid]; ds = fx["dataset"][c["dataset"]]; r = c["r"]
    Y, N = aghq_p1_load(ds); aghq = aghq_p1_request(c["aghq_request"]); fam = ds["family"]
    fit = fam == "poisson" ? GLLVModels.fit_poisson_gllvm(round.(Int, Y); K = 1, aghq = aghq) :
          fam == "gaussian" ? GLLVModels.fit_gllvm(Y; family = Normal(), K = 1, aghq = aghq) :
          fam == "ordinal" ? GLLVModels.fit_gllvm(round.(Int, Y); family = GLLVModels.Ordinal(), K = 1, link = GLLVModels.ProbitLink(), aghq = aghq) :
          fam == "nb2" ? GLLVModels.fit_gllvm(round.(Int, Y); family = NegativeBinomial(), K = 1, disp_group = :species, aghq = aghq) :
          GLLVModels.fit_binomial_gllvm(round.(Int, Y); K = 1, N = round.(Int, N), aghq = aghq)
    gauss = fit isa GLLVModels.GllvmFit
    ll = gauss ? fit.logLik : fit.loglik
    phi = fam == "nb2" ? Float64.(fit.r_group) : nothing
    log_incr = fam == "ordinal" ? GLLVModels._ord_psi_from_tau(fit.τ, fit.C) : nothing
    beta = Float64.(gauss ? fit.pars.β : fit.β)
    lam = vec(Float64.(gauss ? fit.pars.Λ : fit.Λ))
    s = sum(lam .* Float64.(r["lambda"])) < 0 ? -1.0 : 1.0
    info = fit.integration
    used = info !== nothing && info.actual === :aghq
    return (; phi, log_incr, loglik = Float64(ll), beta, lambda = s .* lam, converged = fit.converged,
        sigma_eps = gauss ? Float64(fit.pars.σ_eps) : NaN,
        used, nodes = used ? info.k : 0,
        reason = info === nothing ? :none : info.reason, actual = info === nothing ? :laplace : info.actual)
end

# One fit per distinct (dataset, request): AGHQ-POLICY-TRAITS20 is the same observation as
# AGHQ-POLICY-AUTO-ENFORCE-CUTOFF (the P0 runner calls it a boundary duplicate).
aghq_p1_fit_id(rowid) = rowid == "AGHQ-POLICY-TRAITS20" ? "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF" : rowid
