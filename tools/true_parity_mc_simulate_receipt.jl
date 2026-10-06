#!/usr/bin/env julia
# Julia twin receipts for the Monte-Carlo moment twins (maintainer ruling 2026-10-05, vault D-319, item N4):
#   receipts/julia-twins/postfit-twins/simulate_unit_trait.json  postfit/POSTFIT-SURFACE-simulate_unit_trait
#   receipts/julia-twins/postfit-twins/simulate_default.json     postfit-policy/POST-SIMULATE-DEFAULT
#
# A standalone sibling of tools/true_parity_julia_receipts.jl. R values (replicate means and standard deviations
# of each moment) are copied from test/fixtures/mc_simulate_p1.toml; nothing on the R side is recomputed here.
# The Julia side runs the helpers of test/mc_simulate_helpers_p1.jl with the same inputs and seeds as
# test/test_mc_simulate_p1.jl. Each moment is one comparison case whose tolerance is the Monte-Carlo rule stated
# (before any run) in those files; the rule text is copied into the receipt. If any moment is outside its
# tolerance, or a discrimination control passes, nothing is written.
#
# Usage (from the repository root):
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. tools/true_parity_mc_simulate_receipt.jl

using TOML, SHA
include(joinpath(@__DIR__, "..", "test", "mc_simulate_helpers_p1.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins/postfit-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const FIXTURE = "test/fixtures/mc_simulate_p1.toml"
const GEN = "test/fixtures/gen_mc_simulate_p1.R"
const TEST = "test/test_mc_simulate_p1.jl"
const HELPERS = "test/mc_simulate_helpers_p1.jl"
const RULE = "Monte-Carlo tolerance rule (maintainer ruling 2026-10-05, D-319, item N4; stated before any run in " *
             "$GEN, $HELPERS and $TEST): each engine draws B independent replicates; for moment k, tolerance = " *
             "z * sqrt(s_R^2/B_k + s_J^2/B_k), with s the replicate standard deviations over B_k replicate values and " *
             "z = quantile(Normal(), 1 - 0.01/(2K)) (two-sided Bonferroni over the K moments, familywise alpha 0.01); " *
             "pass when |m_R - m_J| <= tolerance for every moment, and a discrimination control must fail the same rule."

sha_file(rel) = bytes2hex(sha256(read(joinpath(ROOT, rel))))
function findline(rel, frag)
    hits = [i for (i, l) in enumerate(readlines(joinpath(ROOT, rel))) if occursin(frag, l)]
    length(hits) == 1 || error("$rel: fragment $(repr(frag)) on $(length(hits)) lines")
    return hits[1], strip(readlines(joinpath(ROOT, rel))[hits[1]])
end

js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
js(x::Bool) = x ? "true" : "false"
js(x::Integer) = string(x)
js(x::AbstractFloat) = (isfinite(x) || error("non-finite number"); repr(Float64(x)))
js(x::AbstractVector) = "[" * join(js.(x), ", ") * "]"
js(x::Vector{<:Pair}) = "{\n" * join(("  " * js(k) * ": " * replace(js(v), "\n" => "\n  ") for (k, v) in x), ",\n") * "\n}"

upper_labels(T) = ["$i-$j" for j in 1:T for i in 1:j]

function cases(ids, quantities, r_mean, r_n, j_mean, diff, tol, rsrc, jsrc, tline)
    all(diff .<= tol) || error("a moment is outside its Monte-Carlo tolerance; do not bind the row")
    return [Pair{String,Any}["case_id" => ids[k], "quantity" => quantities[k], "r_source" => rsrc, "julia_source" => jsrc,
                              "r_value" => r_mean[k], "julia_value" => j_mean[k], "abs_diff" => abs(r_mean[k] - j_mean[k]),
                              "tolerance" => tol[k], "tolerance_source" => "$TEST:$(tline[1])",
                              "tolerance_source_line" => String(tline[2]), "tolerance_rule" => RULE,
                              "replicates" => r_n[k]] for k in eachindex(ids)]
end

function receipt(sid, cs, control, note)
    head = strip(read(setenv(`git rev-parse HEAD`; dir = ROOT), String))
    dirty = !isempty(strip(read(setenv(`git status --porcelain --untracked-files=no`; dir = ROOT), String)))
    return Pair{String,Any}[
        "schema" => "true-parity-julia-twin-receipt/v1", "source_ids" => [sid], "verdict" => "PASS",
        "evidence_kind" => "julia_recomputed_vs_recorded_r", "pin" => "P1", "reference_commit" => P1_SHA,
        "ruling" => "maintainer ruling 2026-10-05 (D-319), item N4 (Monte-Carlo tolerance rule)",
        "generator" => "tools/true_parity_mc_simulate_receipt.jl", "julia_version" => string(VERSION),
        "gllvmodels_commit" => head, "gllvmodels_worktree_dirty" => dirty,
        "source_fixtures" => [Pair{String,Any}["path" => FIXTURE, "sha256" => sha_file(FIXTURE)],
                              Pair{String,Any}["path" => GEN, "sha256" => sha_file(GEN)]],
        "source_tests" => [Pair{String,Any}["path" => TEST, "sha256" => sha_file(TEST)],
                           Pair{String,Any}["path" => HELPERS, "sha256" => sha_file(HELPERS)]],
        "monte_carlo_rule" => RULE, "discrimination_control" => control, "note" => note,
        "what_this_is_not" => "R replicate moments are copied from the tracked fixture; Julia replicate moments are " *
                              "recomputed here with independent random streams. This is a distributional comparison, " *
                              "not a draw-for-draw replay: no random number is shared between the engines.",
        "comparison" => Pair{String,Any}["pin" => "P1", "cases" => cs]]
end

fxa = TOML.parsefile(joinpath(ROOT, FIXTURE))
B, α = fxa["B"], fxa["alpha"]

# ---- simulate_unit_trait
fx = fxa["unit_trait"]
T = fx["n_traits"]
z = fx["z"]; abs(z - mc_z(α, fx["K"])) < 1e-12 || error("fixture z differs from the rule")
j = mc_unit_trait_julia(fx, B)
d, t = mc_rule(fx["r_mean"], fx["r_sd"], fx["r_n"], j.mean, j.sd, j.n, z)
alt = mc_unit_trait_julia(fx, B; psi_scale = 2.0, seed0 = 20_000)
da, ta = mc_rule(fx["r_mean"], fx["r_sd"], fx["r_n"], alt.mean, alt.sd, alt.n, z)
any(da .> ta) || error("unit_trait discrimination control passed the rule; do not bind")
labs = upper_labels(T)
ids = vcat(["P1-JULIA-MC-UNIT-TRAIT-MEAN-T$i" for i in 1:T], ["P1-JULIA-MC-UNIT-TRAIT-WITHIN-COV-$l" for l in labs],
           ["P1-JULIA-MC-UNIT-TRAIT-UNITMEAN-COV-$l" for l in labs])
qs = vcat(["replicate mean of the trait-$i mean" for i in 1:T],
          ["replicate mean of the pooled within-unit covariance [$l] (divisor U(O-1))" for l in labs],
          ["replicate mean of the covariance of unit means [$l] (divisor U-1)" for l in labs])
cs = cases(ids, qs, fx["r_mean"], fill(fx["r_n"], length(ids)), j.mean, d, t,
           "$FIXTURE [unit_trait] r_mean / r_sd (gllvmTMB::simulate_unit_trait, seeds 1:$B)",
           "GLLVModels.simulate_unit_trait via mc_unit_trait_julia ($HELPERS), Random.Xoshiro(10_000 + b)",
           findline(TEST, "MC-RULE-ASSERT unit_trait"))
ctrl = Pair{String,Any}["alternative" => "Julia simulate_unit_trait with psi_B doubled (Xoshiro(20_000 + b))",
                        "moments_failing_rule" => count(da .> ta), "max_diff_over_tolerance" => maximum(da ./ ta),
                        "must_fail" => true, "failed" => true,
                        "asserted_at" => "$TEST:$(findline(TEST, "psi_B doubled must fail")[1])"]
write(joinpath(ROOT, OUT, "simulate_unit_trait.json"),
      js(receipt("postfit/POSTFIT-SURFACE-simulate_unit_trait", cs, ctrl,
                 "Same two-level Gaussian DGP on both sides (T = 4 traits, U = 20 units, O = 3 observations; Lambda_B " *
                 "4x2, Lambda_W 4x1, psi_B = psi_W = 0.3, sigma2_eps = 0.4, alpha given explicitly because R draws a " *
                 "random alpha by default and Julia uses zeros). Largest |m_R - m_J| / tolerance = " *
                 "$(round(maximum(d ./ t); digits = 3)) over the 24 moments.")) * "\n")

# ---- simulate() default
fx = fxa["simulate_default"]
p = fx["p"]
z = fx["z"]; abs(z - mc_z(α, fx["K"])) < 1e-12 || error("fixture z differs from the rule")
fit, j = mc_simulate_default_julia(fx, B)
abs(GLLVModels.loglikelihood(fit) - fx["loglik"]) <= 1e-4 || error("Julia and R fits differ in logLik")
d1, t1 = mc_rule(fx["r_mean_m1"], fx["r_sd_m1"], fx["r_n_m1"], j.m1.mean, j.m1.sd, j.m1.n, z)
d2, t2 = mc_rule(fx["r_mean_m2"], fx["r_sd_m2"], fx["r_n_m2"], j.m2.mean, j.m2.sd, j.m2.n, z)
c1, u1 = mc_rule(fx["cond_mean_m1"], fx["cond_sd_m1"], fx["r_n_m1"], j.m1.mean, j.m1.sd, j.m1.n, z)
c2, u2 = mc_rule(fx["cond_mean_m2"], fx["cond_sd_m2"], fx["r_n_m2"], j.m2.mean, j.m2.sd, j.m2.n, z)
cd, cu = vcat(c1, c2), vcat(u1, u2)
any(cd .> cu) || error("simulate_default discrimination control passed the rule; do not bind")
labs = upper_labels(p)
ids = vcat(["P1-JULIA-MC-SIMULATE-DEFAULT-MEAN-T$i" for i in 1:p], ["P1-JULIA-MC-SIMULATE-DEFAULT-COV-$l" for l in labs],
           ["P1-JULIA-MC-SIMULATE-DEFAULT-CELLVAR-T$i" for i in 1:p])
qs = vcat(["replicate mean of the trait-$i mean over units" for i in 1:p],
          ["replicate mean of the across-unit covariance [$l] (divisor n-1)" for l in labs],
          ["pair-replicate mean of the trait-$i within-cell variance mean_s (Y1 - Y2)^2 / 2" for i in 1:p])
rn = vcat(fill(fx["r_n_m1"], length(d1)), fill(fx["r_n_m2"], length(d2)))
cs = cases(ids, qs, vcat(fx["r_mean_m1"], fx["r_mean_m2"]), rn, vcat(j.m1.mean, j.m2.mean), vcat(d1, d2), vcat(t1, t2),
           "$FIXTURE [simulate_default] (simulate(fit, nsim = $B, seed = $(fx["r_seed_default"])), all other arguments default)",
           "GLLVModels.simulate(fit::PoissonFit, n; rng) via mc_simulate_default_julia ($HELPERS), one Xoshiro(30_000) stream",
           findline(TEST, "MC-RULE-ASSERT simulate_default"))
ctrl = Pair{String,Any}["alternative" => "R simulate(fit, nsim = $B, seed = $(fx["r_seed_conditional"]), condition_on_RE = TRUE) " *
                                         "(draws given the fitted latent modes) against the Julia default",
                        "moments_failing_rule" => count(cd .> cu), "max_diff_over_tolerance" => maximum(cd ./ cu),
                        "must_fail" => true, "failed" => true,
                        "asserted_at" => "$TEST:$(findline(TEST, "condition_on_RE = TRUE must fail")[1])"]
write(joinpath(ROOT, OUT, "simulate_default.json"),
      js(receipt("postfit-policy/POST-SIMULATE-DEFAULT", cs, ctrl,
                 "The row's question is the default of simulate(): R's simulate.gllvmTMB_multi defaults to " *
                 "condition_on_RE = FALSE (latent scores redrawn), and Julia's simulate(fit, n) also draws a fresh " *
                 "z ~ N(0, I) per unit. Poisson GLLVM (p = 5, n = 60, d = 1, unique = FALSE) fitted by each engine " *
                 "to the same counts (logLik R $(fx["loglik"]), Julia $(GLLVModels.loglikelihood(fit))). Largest " *
                 "|m_R - m_J| / tolerance = $(round(maximum(vcat(d1, d2) ./ vcat(t1, t2)); digits = 3)) over 25 " *
                 "moments; the conditional R draws fail $(count(cd .> cu)) of 25, so the comparison separates the " *
                 "two defaults.")) * "\n")
println("wrote $OUT/simulate_unit_trait.json and $OUT/simulate_default.json")
