# gllvm-parity-tag: P1
#
# Temporal source composed with ordinary unit / unit_obs terms (temporal port
# slice 2) against gllvmTMB P1 (9539352f6) receipts in
# test/fixtures/temporal_p1/composed.toml and composed_cross.toml
# (generate_temporal_p1_slice2.R). Spec section 4.1 row
# sixth-source-oracles.R:318 and section 5 for every composed cell:
#   - R's names(opt$par), order and multiplicity, and the active tier set;
#   - fixed point: Julia's NLL and gradient at R's coordinates equal R's TMB
#     fn / gr (1e-8), both at R's optimum and at deterministic coordinates away
#     from it;
#   - an independent dense oracle (written below, its own unpack and
#     Distributions' MvNormal) agrees with the structured NLL to 1e-10, and
#     ForwardDiff's theta_time derivative with its central difference (1e-6);
#   - cross-objective: R's objective at Julia's optimum equals Julia's NLL
#     there (1e-8);
#   - between optima against R's tight run, with R's early stop handled as in
#     slice 1: raw gaps at loose bounds plus a one-Newton-step comparison;
#   - the sigma_eps suppression rule (R/fit-multi.R:6959-6967): on a replicated
#     fit with a per-row unit_obs diagonal, log_sigma_eps leaves the vector and
#     is fixed at log(max(1e-3 sd(y), 1e-6)); df drops by one;
#   - conditional predictor (R's report$eta), logLik, AIC, BIC, nobs, and the
#     ordinary unit ordination (R's extract_ordination(level = "unit")).
using Test, GLLVModels, LinearAlgebra, Statistics, ForwardDiff
import Distributions
const GMC = GLLVModels
include(joinpath(@__DIR__, "fixtures", "temporal_p1", "fixture_helpers.jl"))

# ---- independent dense oracle ---------------------------------------------
function composed_unpack(theta, p, rank)
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
rank_from_len(n, p) = n == 0 ? 0 : only(r for r in 1:p if p * r - r * (r - 1) ÷ 2 == n)

# Trait covariance of one tier from its rr and diag blocks, as the R oracle
# reconstructs it (oracles.R:36-106): Lambda Lambda' + diag(exp(2 theta)).
function composed_tier(par, names, rr, dg, p)
    L = composed_unpack(par[names .== rr], p, rank_from_len(count(==(rr), names), p))
    S = L * L'
    d = par[names .== dg]
    isempty(d) || (S = S + Diagonal(exp.(2 .* d)))
    return S
end

function composed_oracle_cov(par, names, tbl, time_col, structure; unit_obs)
    traits = sort(unique(tbl.trait)); p = length(traits)
    tr = [findfirst(==(t), traits) for t in tbl.trait]
    time = getproperty(tbl, time_col)
    n = length(time)
    theta = par[names .== "theta_temporal_time"][1]
    ST = composed_tier(par, names, "theta_temporal_rr", "theta_temporal_diag", p)
    SB = composed_tier(par, names, "theta_rr_B", "theta_diag_B", p)
    SW = composed_tier(par, names, "theta_rr_W", "theta_diag_W", p)
    re = par[names .== "log_sigma_re_int"]
    uo = unit_obs === nothing ? nothing : getproperty(tbl, unit_obs)
    V = zeros(eltype(par), n, n)
    for i in 1:n, j in 1:n
        tbl.series[i] == tbl.series[j] || continue
        gap = abs(time[i] - time[j])
        c = structure == "ar1" ? ((1 - 1e-6) * tanh(theta))^gap : exp(-exp(theta) * gap)
        V[i, j] = c * ST[tr[i], tr[j]] + SB[tr[i], tr[j]]      # unit = series here
        isempty(re) || (V[i, j] += exp(2 * re[1]))                # (1 | series)
        if uo !== nothing && uo[i] == uo[j]
            V[i, j] += SW[tr[i], tr[j]]
        end
    end
    s = par[names .== "log_sigma_eps"]
    sigma = isempty(s) ? max(1e-3 * std(tbl.value), 1e-6) : exp(s[1])
    return V + sigma^2 * I
end

function composed_oracle_nll(par, names, tbl, time_col, structure; unit_obs)
    V = composed_oracle_cov(par, names, tbl, time_col, structure; unit_obs=unit_obs)
    traits = sort(unique(tbl.trait))
    beta = par[names .== "b_fix"]
    mu = [beta[findfirst(==(t), traits)] for t in tbl.trait]
    return -Distributions.logpdf(Distributions.MvNormal(mu, Symmetric(Matrix(V))), tbl.value)
end

# Sign-invariant summaries of a coordinate vector: theta_time, beta, the
# residual log SD, and every tier covariance (the loading factors are
# identified only up to column signs, spec section 1.6).
function composed_summary(par, names, p)
    out = Float64[]
    append!(out, par[names .== "theta_temporal_time"])
    append!(out, par[names .== "b_fix"])
    append!(out, par[names .== "log_sigma_eps"])
    for (rr, dg) in (("theta_temporal_rr", "theta_temporal_diag"), ("theta_rr_B", "theta_diag_B"),
            ("theta_rr_W", "theta_diag_W"))
        append!(out, vec(composed_tier(par, names, rr, dg, p)))
    end
    append!(out, exp.(2 .* par[names .== "log_sigma_re_int"]))
    return out
end

composed_time_col(c) = occursin(":elapsed", c["julia_temporal"]) ? :elapsed : :occasion
composed_structure(c) = occursin("structure = :ou", c["julia_temporal"]) ? "ou" : "ar1"
composed_uo(c) = (u = get(c, "unit_obs", ""); isempty(u) ? nothing : Symbol(u))

@testset "temporal + unit / unit_obs vs gllvmTMB P1 receipts" begin
    C = temporal_p1_load("composed.toml")
    X = temporal_p1_load("composed_cross.toml")
    cross = Dict(c["id"] => c for c in X["cells"])

    @testset "receipt integrity" begin
        @test C["gllvmTMB_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test temporal_p1_sha("composed.toml") == TEMPORAL_P1_SHA256["composed.toml"]
        @test temporal_p1_sha("composed_cross.toml") == TEMPORAL_P1_SHA256["composed_cross.toml"]
        for (name, ds) in C["datasets"]
            @test temporal_p1_value_sha(temporal_p1_composed_table(ds)) == ds["value_sha256"]
        end
        @test length(C["fits"]) == 25
        @test Set(keys(cross)) == Set(c["id"] for c in C["fits"])
    end

    @testset "dense oracle with unit and unit_obs (oracles.R:318)" begin
        o = C["oracle318"]
        tbl = temporal_p1_composed_table(C["datasets"][o["dataset"]])
        f = fit_temporal_gllvm(tbl; formula=@formula(value ~ 0 + trait),
            temporal=temporal_p1_eval(o["julia_temporal"]), structure=temporal_p1_terms(o),
            unit=:series, unit_obs=:unit_obs, iterations=0)
        names = String.(o["par_names"]); par = temporal_p1_vec(o["par"])
        @test f.parameter_names == names
        nll(t) = GMC.temporal_marginal_nll(t, f.y, f.X, f.spec)
        dense(t) = composed_oracle_nll(t, names, tbl, :occasion, "ar1"; unit_obs=:unit_obs)
        @test abs(nll(par) - o["fn"]) <= 1e-8
        @test maximum(abs, ForwardDiff.gradient(nll, par) .- temporal_p1_vec(o["gr"])) <= 1e-8
        @test abs(nll(par) - dense(par)) <= 1e-10
        k = findfirst(==("theta_temporal_time"), names); h = 1e-5
        e = zeros(length(par)); e[k] = h
        central = (dense(par .+ e) - dense(par .- e)) / (2h)
        @test abs(ForwardDiff.gradient(nll, par)[k] - central) <= 1e-6
    end

    worst = Dict{String,Float64}()
    note(k, v) = (worst[k] = max(get(worst, k, 0.0), v))
    @testset "$(c["id"])" for c in C["fits"]
        tbl = temporal_p1_composed_table(C["datasets"][c["dataset"]])
        names = String.(c["par_names"])
        rpar = temporal_p1_vec(c["par"]); rtight = temporal_p1_vec(c["par_tight"])
        at_r = temporal_p1_composed_fit(C, c; start=rpar, iterations=0)
        at_rt = temporal_p1_composed_fit(C, c; start=rtight, iterations=0)
        jf = temporal_p1_composed_fit(C, c; g_tol=TEMPORAL_P1_GTOL, iterations=TEMPORAL_P1_ITERATIONS)

        # Coordinates and tiers.
        @test jf.parameter_names == names
        @test GMC._temporal_other_tiers(jf) == String.(c["tiers"])
        @test String(jf.spec.workflow) == c["workflow"]

        # sigma_eps rule: suppressed exactly when R maps log_sigma_eps off.
        suppressed = !("log_sigma_eps" in names)
        @test (jf.spec.composition.sigma_fixed !== nothing) == suppressed
        @test abs(log(at_r.sigma_eps) - c["log_sigma_eps_full"]) <= 1e-12
        suppressed && @test abs(jf.sigma_eps - max(1e-3 * std(tbl.value), 1e-6)) <= 1e-15

        # Fixed point at R's optimum and at deterministic coordinates.
        nll(t) = GMC.temporal_marginal_nll(t, at_r.y, at_r.X, at_r.spec)
        @test abs(nll(rpar) - c["objective"]) <= 1e-8
        @test abs(nll(rtight) - c["objective_tight"]) <= 1e-8
        fx = temporal_p1_vec(c["fixed_par"])
        @test abs(nll(fx) - c["fixed_fn"]) <= 1e-8
        g_fx = ForwardDiff.gradient(nll, fx)
        @test maximum(abs, g_fx .- temporal_p1_vec(c["fixed_gr"])) <= 1e-8
        g_r = ForwardDiff.gradient(nll, rpar)
        if suppressed
            # With sigma_eps fixed at ~1e-3 sd(y), TMB's inner Laplace solve has
            # a residual precision near 1e6 and its gradient at R's optimum
            # carries ~2e-8 of rounding error (measured: Julia's gradient agrees
            # with a 256-bit central difference to 1e-13 while R's differs from
            # it by 1.9e-8 on sim_rw__Tindep_Wrowlatent). There Julia is checked
            # against the 256-bit reference at 1e-10 and against R at 5e-8.
            big_nll(t) = GMC.temporal_marginal_nll(t, at_r.y, at_r.X, at_r.spec)
            g_big = setprecision(256) do
                map(eachindex(rpar)) do k
                    h = big"1e-20"; e = zeros(BigFloat, length(rpar)); e[k] = h
                    x = BigFloat.(rpar)
                    Float64((big_nll(x .+ e) - big_nll(x .- e)) / (2h))
                end
            end
            @test maximum(abs, g_r .- g_big) <= 1e-10
            @test maximum(abs, g_r .- temporal_p1_vec(c["gradient_at_par"])) <= 5e-8
        else
            @test maximum(abs, g_r .- temporal_p1_vec(c["gradient_at_par"])) <= 1e-8
        end
        note("fixed point: |NLL_J - R fn|", max(abs(nll(rpar) - c["objective"]), abs(nll(fx) - c["fixed_fn"])))
        note("fixed point: |grad_J - R gr|", maximum(abs, g_fx .- temporal_p1_vec(c["fixed_gr"])))

        # Independent dense oracle and the theta_time derivative.
        dense(t) = composed_oracle_nll(t, names, tbl, composed_time_col(c), composed_structure(c);
            unit_obs=composed_uo(c))
        @test abs(nll(fx) - dense(fx)) <= 1e-10
        @test abs(nll(rpar) - dense(rpar)) <= 1e-10
        note("dense oracle: |NLL_J - dense|", max(abs(nll(fx) - dense(fx)), abs(nll(rpar) - dense(rpar))))
        k = findfirst(==("theta_temporal_time"), names); h = 1e-5
        e = zeros(length(fx)); e[k] = h
        @test abs(g_fx[k] - (dense(fx .+ e) - dense(fx .- e)) / (2h)) <= 1e-6

        # Cross-objective (R's objective at Julia's recorded optimum).
        xc = cross[c["id"]]
        jpar = temporal_p1_vec(xc["julia_par"])
        @test abs(nll(jpar) - xc["r_fn_at_julia_par"]) <= 1e-8
        note("cross: |R fn(J par) - NLL_J(J par)|", abs(nll(jpar) - xc["r_fn_at_julia_par"]))

        # Methods at R's coordinates.
        @test abs(loglikelihood(at_r) - c["logLik"]) <= 1e-8
        @test dof(at_r) == length(rpar) == c["logLik_df"]
        @test abs(aic(at_r) - c["AIC"]) <= 1e-8 && abs(bic(at_r) - c["BIC"]) <= 1e-8
        @test nobs(at_r) == c["nobs"]
        @test abs(at_r.time_value - c["time_value"]) <= 1e-12
        d_eta = maximum(abs, predict(at_r).est .- temporal_p1_vec(c["eta"]))
        @test d_eta <= 1e-7
        note("conditional eta: |eta_J - R report eta|", d_eta)
        if haskey(c, "ordination_scores")
            o = extract_ordination(at_r; level=:unit)
            @test o.row_id == String.(c["ordination_row_id"])
            @test vec(o.loadings) ≈ temporal_p1_vec(c["ordination_loadings"]) atol = 1e-12
            d = maximum(abs, vec(o.scores) .- temporal_p1_vec(c["ordination_scores"]))
            @test d <= 1e-7
            @test !haskey(o, :pair_id) && !haskey(o, :row_index)
            note("unit ordination: |scores_J - R scores|", d)
        end

        # Between optima.
        dll = jf.loglik - (-c["objective_tight"])
        @test dll >= -1e-8                      # Julia is never worse than R's best
        if dll > 1e-6
            # A higher optimum than R's stopping point; R's own objective must
            # confirm that Julia's point is better.
            @test xc["r_fn_at_julia_par"] < c["objective_tight"] - 1e-6
            @info "Julia at a higher optimum than R's stopping point (R objective confirms)" c["id"] dll
        else
            @test abs(dll) <= 1e-6
            note("between optima: |Δ logLik|", abs(dll))
        end
        interior = jf.hessian_min_eigenvalue > 1e-3 && dll <= 1e-6
        if interior
            @test jf.converged
            # R's nlminb stops before stationarity even at rel.tol = 1e-14, so
            # the raw gaps to its stopped point are asserted loosely (1e-4) and
            # the tight comparison is made at R's point after one Newton step
            # built from R's own recorded gradient.
            p = length(jf.spec.traits)
            sJ = composed_summary(jf.parameters, names, p)
            @test abs(jf.time_value - at_rt.time_value) <= 1e-4
            @test maximum(abs, sJ .- composed_summary(rtight, names, p)) <= 1e-4
            gR = temporal_p1_vec(c["gradient_at_par_tight"])
            @test maximum(abs, ForwardDiff.gradient(nll, rtight) .- gR) <= (suppressed ? 5e-8 : 1e-8)
            H = ForwardDiff.hessian(nll, rtight)
            newton = rtight .- (Symmetric(H) \ gR)
            sN = composed_summary(newton, names, p)
            @test maximum(abs, sJ .- sN) <= 1e-5
            @test maximum(abs, jf.beta .- newton[1:length(jf.beta)]) <= 1e-6
            note("raw: |Δ summary| (R par_tight)", maximum(abs, sJ .- composed_summary(rtight, names, p)))
            note("between optima: |Δ summary| (Newton-polished R)", maximum(abs, sJ .- sN))
        end
    end
    println("temporal composed receipt maxima:")
    for (k, v) in sort(collect(worst)); println("  ", rpad(k, 50), v); end
end
