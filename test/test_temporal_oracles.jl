# gllvm-parity-tag: P1
#
# Temporal likelihood oracles (gllvmTMB P1 9539352f6; spec section 4.1 rows
# sixth-source-oracles.R:214, :342, :359, :375 and ar1-oracles.R:3, :16).
#
# Three independent checks of `temporal_marginal_nll`:
#   1. R's native TMB objective and gradient, recorded at fixed coordinates in
#      test/fixtures/temporal_p1/oracle.toml by generate_temporal_p1.R;
#   2. an independently written dense builder below (a second implementation:
#      its own loading unpack, float powers and Distributions' MvNormal),
#      agreeing to 1e-10;
#   3. ForwardDiff's theta_time derivative against a central difference of (2).
using Test, GLLVModels, LinearAlgebra, ForwardDiff
import Distributions
const GMT = GLLVModels
include(joinpath(@__DIR__, "fixtures", "temporal_p1", "fixture_helpers.jl"))

# ---- independent dense oracle (translation of R's oracles.R:18-106) --------
function oracle_unpack(theta, p, rank)
    out = zeros(eltype(theta), p, rank)
    cursor = 1
    for c in 1:rank
        out[c, c] = theta[cursor]; cursor += 1
    end
    for c in 1:rank, r in (c+1):p
        out[r, c] = theta[cursor]; cursor += 1
    end
    @assert cursor == length(theta) + 1
    return out
end

function oracle_covariance(par, names, tbl, time_col, mode, structure, unique; iid_psi=false)
    blk(n) = par[names .== n]
    traits = sort(unique_(tbl.trait))
    p = length(traits)
    tr = [findfirst(==(t), traits) for t in tbl.trait]
    time = getproperty(tbl, time_col)
    theta = blk("theta_temporal_time")[1]
    n = length(time)
    corr = zeros(eltype(par), n, n)
    for i in 1:n, j in 1:n
        tbl.series[i] == tbl.series[j] || continue
        gap = abs(time[i] - time[j])
        corr[i, j] = structure == "ar1" ? ((1 - 1e-6) * tanh(theta))^gap : exp(-exp(theta) * gap)
    end
    S = if mode == "indep"
        Matrix(Diagonal(exp.(2 .* blk("theta_temporal_diag"))))
    elseif mode == "dep"
        L = oracle_unpack(blk("theta_temporal_rr"), p, p); L * L'
    else
        L = oracle_unpack(blk("theta_temporal_rr"), p, 1)
        unique ? L * L' + Diagonal(exp.(2 .* blk("theta_temporal_diag"))) : L * L'
    end
    V = corr .* S[tr, tr]
    if iid_psi && mode == "latent" && unique
        L = oracle_unpack(blk("theta_temporal_rr"), p, 1)
        V = corr .* (L * L')[tr, tr]
        psi = exp.(2 .* blk("theta_temporal_diag"))
        for i in 1:n  # Psi only at the same state and trait (retired prototype)
            V[i, i] += psi[tr[i]]
        end
    end
    V += exp(2 * blk("log_sigma_eps")[1]) * I
    return V
end
unique_(x) = unique(x)

function oracle_nll(par, names, tbl, time_col, mode, structure, unique; kw...)
    V = oracle_covariance(par, names, tbl, time_col, mode, structure, unique; kw...)
    traits = sort(unique_(tbl.trait))
    beta = par[names .== "b_fix"]
    mu = [beta[findfirst(==(t), traits)] for t in tbl.trait]
    return -Distributions.logpdf(Distributions.MvNormal(mu, Symmetric(Matrix(V))), tbl.value)
end

function oracle_term(mode, structure, time_col, unique)
    t = Symbol(time_col); s = Symbol(structure)
    mode == "indep" ? temporal_indep(:(0 + trait | series), t; structure=s) :
    mode == "dep" ? temporal_dep(:(0 + trait | series), t; structure=s) :
    temporal_latent(:(0 + trait | series), t; structure=s, unique=unique)
end

function structured_pieces(tbl, term)
    f = fit_temporal_gllvm(tbl; formula=@formula(value ~ 0 + trait), temporal=term,
        iterations=0)
    return f.y, f.X, f.spec, f.parameter_names
end

@testset "temporal oracles (gllvmTMB P1)" begin
    O = temporal_p1_load("oracle.toml")
    tbl = temporal_p1_table(O["datasets"]["oracle"])

    @testset "receipt integrity" begin
        @test O["gllvmTMB_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test temporal_p1_sha("oracle.toml") == TEMPORAL_P1_SHA256["oracle.toml"]
        @test temporal_p1_value_sha(tbl) == O["datasets"]["oracle"]["value_sha256"]
        @test length(O["cells"]) == 4 * 5 + 4 * 3   # AR1: 3 values + 2 extremes; OU: 1 + 2
    end

    max_native = Ref(0.0); max_grad = Ref(0.0); max_dense = Ref(0.0); max_fd = Ref(0.0)
    @testset "native objective, dense oracle, gradient ($(c["mode"]) $(c["unique"] ? "unique " : "")$(c["structure"]) $(c["label"]))" for c in O["cells"]
        term = oracle_term(c["mode"], c["structure"], c["time"], c["unique"])
        y, X, spec, jnames = structured_pieces(tbl, term)
        rnames = String.(c["par_names"])
        par = temporal_p1_vec(c["par"])
        # R's names(opt$par), order and multiplicity, equal the Julia layout.
        @test jnames == rnames
        nll(t) = GMT.temporal_marginal_nll(t, y, X, spec)
        native = nll(par)
        @test isfinite(native)
        # (1) Julia's exact marginal at R's coordinates vs R's TMB objective.
        @test abs(native - c["fn"]) <= 1e-8
        max_native[] = max(max_native[], abs(native - c["fn"]))
        g = ForwardDiff.gradient(nll, par)
        @test all(isfinite, g)
        @test maximum(abs, g .- temporal_p1_vec(c["gr"])) <= 1e-6
        max_grad[] = max(max_grad[], maximum(abs, g .- temporal_p1_vec(c["gr"])))
        # (2) independent dense oracle.
        dense(t) = oracle_nll(t, rnames, tbl, Symbol(c["time"]), c["mode"], c["structure"], c["unique"])
        @test abs(native - dense(par)) <= 1e-10
        max_dense[] = max(max_dense[], abs(native - dense(par)))
        # The package's dense covariance equals the oracle's.
        @test GMT._temporal_covariance(par, spec, size(X, 2)) ≈
              oracle_covariance(par, rnames, tbl, Symbol(c["time"]), c["mode"], c["structure"], c["unique"]) atol = 1e-12
        # (3) theta_time derivative vs central difference of the dense oracle.
        if startswith(c["label"], "value=")
            k = findfirst(==("theta_temporal_time"), rnames)
            h = 1e-5
            up = copy(par); up[k] += h; dn = copy(par); dn[k] -= h
            central = (dense(up) - dense(dn)) / (2h)
            @test abs(g[k] - central) <= 1e-6
            max_fd[] = max(max_fd[], abs(g[k] - central))
        end
    end
    @info "temporal oracle maxima" max_native[] max_grad[] max_dense[] max_fd[]

    @testset "negative persistence changes sign at an odd lag (oracles.R:342)" begin
        t2 = merge(tbl, (occasion=[Dict(1.0 => 1.0, 3.0 => 3.0, 7.0 => 8.0)[x] for x in tbl.occasion],))
        term = temporal_indep(:(0 + trait | series), :occasion)
        y, X, spec, names = structured_pieces(t2, term)
        c = only(filter(c -> c["mode"] == "indep" && c["structure"] == "ar1" && c["label"] == "value=-0.7", O["cells"]))
        par = temporal_p1_vec(c["par"])
        V = GMT._temporal_covariance(par, spec, size(X, 2))
        first_row = findfirst(o -> spec.state_id[o] == 1 && spec.trait_id[o] == 1, eachindex(y))
        odd_lag = findfirst(o -> spec.state_id[o] == 3 && spec.trait_id[o] == 1, eachindex(y))
        @test spec.pair_table.time[3] - spec.pair_table.time[1] == 7
        @test V[first_row, odd_lag] < 0
    end

    @testset "latent unique Psi is correlated across occasions (oracles.R:375)" begin
        c = only(filter(c -> c["mode"] == "latent" && c["unique"] && c["structure"] == "ar1" &&
            c["label"] == "value=0.65", O["cells"]))
        term = oracle_term("latent", "ar1", "occasion", true)
        y, X, spec, names = structured_pieces(tbl, term)
        par = temporal_p1_vec(c["par"])
        V = GMT._temporal_covariance(par, spec, size(X, 2))
        Viid = oracle_covariance(par, names, tbl, :occasion, "latent", "ar1", true; iid_psi=true)
        target = findfirst(o -> spec.state_id[o] == 1 && spec.trait_id[o] == 1, eachindex(y))
        later = findfirst(o -> spec.state_id[o] == 2 && spec.trait_id[o] == 1, eachindex(y))
        @test abs(V[target, later] - Viid[target, later]) > 1e-4
        @test abs(GMT.temporal_marginal_nll(par, y, X, spec) -
            oracle_nll(par, names, tbl, :occasion, "latent", "ar1", true; iid_psi=true)) > 1e-4
    end

    @testset "AR1 sign, zero, gap and boundary identities (ar1-oracles.R:16)" begin
        d = (series=repeat(["s1", "s2", "s3"], 9), occasion=repeat(repeat([1.0, 3.0, 7.0], inner=3), 3),
             trait=repeat(["t1", "t2", "t3"], inner=9), value=collect(1:27) ./ 10)
        term = temporal_latent(:(0 + trait | series), :occasion; unique=true)
        y, X, spec, names = structured_pieces(d, term)
        L = GMT.TemporalLayout(size(X, 2), spec)
        fixed = zeros(L.total)
        fixed[L.beta] .= X \ y
        fixed[L.time] = atanh(0.65 / (1 - 1e-6))
        fixed[L.rr] .= 0.4
        fixed[L.diag] .= log(0.6)
        fixed[L.log_sigma] = log(0.5)
        signed = copy(fixed); signed[L.rr] .= -0.4
        nll(t) = GMT.temporal_marginal_nll(t, y, X, spec)
        @test abs(nll(signed) - nll(fixed)) <= 1e-10
        @test GMT._temporal_phi(0.0) == 0
        @test 0.65^abs(1 - 7) == 0.65^6
        for theta in (-20.0, 20.0)
            b = copy(fixed); b[L.time] = theta
            @test isfinite(nll(b))
            @test all(isfinite, ForwardDiff.gradient(nll, b))
        end
    end

    @testset "retired iid-Psi prototype differs from the temporal row (ar1-oracles.R:3)" begin
        phi = 0.65; gap = 2; lambda = [0.4, -0.2, 0.3]; psi = [0.25, 0.36, 0.49]
        K = phi^gap
        retired = K * lambda * lambda'
        current = K * (lambda * lambda' + Diagonal(psi))
        @test retired[1, 1] ≈ K * lambda[1]^2 atol = 1e-12
        @test current[1, 1] ≈ K * (lambda[1]^2 + psi[1]) atol = 1e-12
        @test abs(current[1, 1] - retired[1, 1]) > 1e-6
    end
end
