# test/test_grouped_analytic_grad.jl — leaf-S8 gates GB.2 (fd_agreement) and GB.3 (identity).
#
# GB.2 compares the S8 analytic outer gradient
# (`GLLVModels._grouped_analytic_gradient`, src/grouped_nongaussian_fit.jl)
# against the central-difference gradient the fitter used before S8
# (`GLLVModels._grouped_fd_gradient`, src/grouped_fit.jl:213-224) at 20 random
# theta per fixture, PER COORDINATE, at rtol 1e-6.
#
# The gate was AMENDED in .unlazy/grouped-analytic-20260920/gates/leaf-S8.md
# after the B1 derivation showed a Poisson-only, norm-summarised comparison is
# blind to four of the nine named failure modes. This file implements the four
# amended requirements literally:
#
#   (a) PER-COORDINATE comparison, never a norm or a cosine similarity — every
#       coordinate is asserted and the worst one is printed with its index and
#       its meaning (hazard 7.2, a sign slip confined to the log-det direction).
#   (b) a fixture with a NONZERO LOADING coordinate — `poisson_latent` uses a
#       `mode = :latent, rank = 1` term and perturbs theta away from the
#       zero-loading start, so the gamma coordinates are not identically zero
#       (hazard 7.3, the lost factor of 2 on the design-derivative trace).
#   (c) a NON-POISSON, NON-BINOMIAL family — `beta_shared` (Beta, shared phi)
#       and `nb2_shared` (NegativeBinomial, shared r). For those two families
#       the Fisher and observed weights coincide pointwise in Poisson, so
#       hazards 7.5 (Ff where A belongs) and 7.9 (dropped dispersion terms) are
#       undetectable without them. NB2 additionally exercises the hand-coded
#       `_glm_obs_weight` override (src/families/negbin.jl), i.e. a different
#       `_glm_obs_weight_deta` dispatch than the generic AD path.
#   (d) the `inner_tol` in force is RECORDED in the printed evidence, because
#       the analytic and FD gradients degrade differently as it loosens.
#
# The comparison is strict and one-sided: no tolerance is widened anywhere, and
# a coordinate whose FD reference is itself below the FD noise floor is
# REPORTED rather than excused (it still has to pass).
#
# USAGE
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       test/test_grouped_analytic_grad.jl --gate fd_agreement
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       test/test_grouped_analytic_grad.jl --gate identity
#
# leaf-S9c added two more gates and the four fixtures they need:
#   --gate coverage   holes 1-3 (Binomial; Beta/NB2 per-trait dispersion; mode=:dep)
#   --gate mixed      hole 4, the mixed analytic/FD-fallback path
# An unrecognised `--gate` name now EXITS 2 instead of silently running
# `fd_agreement` under the wrong name; see the dispatch at the bottom.

using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, Printf

const RTOL_FD = 1e-6      # GB.2, the FD reference's own accuracy. Never widened.
const RTOL_IDENTITY = 1e-8  # GB.3.
const NTHETA = 20
const INNER_TOL = 1e-10   # tightened FIXTURE, not a loosened assertion; recorded per (d).
const INNER_MAXITER = 200

# ---------------------------------------------------------------------------
# Rebuild exactly the internal state `fit_grouped_nongaussian` builds before it
# starts optimising (src/grouped_nongaussian_fit.jl:573-640), so the analytic
# gradient and the FD gradient are compared on the SAME objects the fitter
# itself would hand them. Nothing here re-derives the objective.
# ---------------------------------------------------------------------------
function _grouped_internals(Y; family, terms, unit, cluster = nothing, N = nothing,
        dispersion = :trait)
    p, n = size(Y)
    kind = GLLVModels._grouped_nongaussian_kind(family)
    mode = GLLVModels._grouped_nongaussian_dispersion_mode(kind, dispersion)
    termvec = GLLVModels.GroupingTerm[terms...]
    labels = GLLVModels._grouped_labels(n, termvec; unit = unit, cluster = cluster)
    incidences = [GLLVModels._grouped_incidence(v, n) for v in labels]
    data = Matrix{Float64}(Y)
    trials = GLLVModels._grouped_nongaussian_trials(data, N, kind)
    D = GLLVModels._trait_mean_design(p, n)
    theta0 = GLLVModels._grouped_nongaussian_initial_parameters(data, trials, D,
        termvec, kind, family, mode)
    objective_cold = GLLVModels._grouped_nongaussian_objective(data, trials, D,
        termvec, incidences, kind; dispersion_mode = mode,
        inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, warm_start_inner = false)
    return (; p, n, kind, mode, termvec, incidences, data, trials, D, theta0, objective_cold)
end

_analytic(st, theta) = GLLVModels._grouped_analytic_gradient(theta, st.data, st.trials,
    st.D, st.termvec, st.incidences, st.kind; dispersion_mode = st.mode,
    inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL)

# `_grouped_fd_gradient` differentiates the MINIMISED objective F = -L, which is
# what `_grouped_analytic_gradient` returns too (its docstring, and the sign flip
# at its last line). Same sign convention, no adjustment here.
_fd(st, theta) = GLLVModels._grouped_fd_gradient(st.objective_cold, collect(theta))

# ---------------------------------------------------------------------------
# Fixtures. Small on purpose: GB.2 is a correctness gate, not a timing one, and
# a tight inner_tol on a small fixture is the regime where an FD reference is
# most trustworthy.
# ---------------------------------------------------------------------------
function _fixture_poisson_latent()
    rng = Xoshiro(20260921)
    p, n, G = 3, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    lambda = [0.8, -0.5, 0.35]
    beta = [0.4, 0.1, -0.2]
    Y = Matrix{Float64}(undef, p, n)
    z = randn(rng, G)
    for s in 1:n, t in 1:p
        Y[t, s] = rand(rng, GLLVModels.Poisson(exp(beta[t] + lambda[t] * z[unit[s]])))
    end
    call = (; Y = Y, family = GLLVModels.Poisson(),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :latent, rank = 1)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit)
    return ("poisson_latent", st, "Poisson, :latent rank-1 (nonzero loading coordinates)", call)
end

function _fixture_beta_shared()
    rng = Xoshiro(20260922)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    phi = 9.0
    beta = [0.3, -0.25]
    z = 0.7 .* randn(rng, G)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = 1 / (1 + exp(-(beta[t] + z[unit[s]])))
        Y[t, s] = clamp(rand(rng, GLLVModels.Beta(mu * phi, (1 - mu) * phi)), 1e-4, 1 - 1e-4)
    end
    call = (; Y = Y, family = GLLVModels.Beta(8.0, 1.0),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :shared)
    st = _grouped_internals(Y; family = call.family, terms = call.terms,
        unit = unit, dispersion = :shared)
    return ("beta_shared", st, "Beta, shared log_phi (Fisher != observed weight; dispersion block)", call)
end

function _fixture_nb2_shared()
    rng = Xoshiro(20260923)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    r = 5.0
    beta = [0.6, 0.2]
    z = 0.5 .* randn(rng, G)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = exp(beta[t] + z[unit[s]])
        Y[t, s] = rand(rng, GLLVModels.NegativeBinomial(r, r / (r + mu)))
    end
    call = (; Y = Y, family = GLLVModels.NegativeBinomial(4.0, 0.5),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :shared)
    st = _grouped_internals(Y; family = call.family, terms = call.terms,
        unit = unit, dispersion = :shared)
    return ("nb2_shared", st, "NegativeBinomial, shared log_r (hand-coded _glm_obs_weight override)", call)
end

# Requirement (e), ADDED 2026-09-21 after a defect the first three fixtures
# could not see. All three above use exactly ONE grouping term, and with one
# term `_grouped_laplace_design_jacobian` never pushes a zero placeholder
# block -- so its placeholder width was wrong (trait-factor `width` instead of
# `size(incidence, 2) * width`) and no gate here could tell. With two terms it
# is a hard `DimensionMismatch` at `dW * bhat`, which is how S7c's
# `--gate warm_identity` found it on fixture D. This fixture puts a TWO-TERM
# design, with DIFFERENT group counts per term so a coincidental width match
# cannot hide the same bug, inside GB.2 itself.
function _fixture_poisson_twoterm()
    rng = Xoshiro(20260924)
    p, n = 2, 60
    unit = repeat(1:12; inner = n ÷ 12)      # 12 groups
    cluster = repeat(1:5; inner = n ÷ 5)     # 5 groups, deliberately != 12
    beta = [0.35, -0.15]
    zu = 0.5 .* randn(rng, 12)
    zc = 0.4 .* randn(rng, 5)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        Y[t, s] = rand(rng, GLLVModels.Poisson(exp(beta[t] + zu[unit[s]] + zc[cluster[s]])))
    end
    call = (; Y = Y, family = GLLVModels.Poisson(),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true),
                 GLLVModels.GroupingTerm(:cluster; mode = :indep, common = true)],
        unit = unit, cluster = cluster, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms,
        unit = unit, cluster = cluster)
    return ("poisson_twoterm", st, "Poisson, TWO grouping terms with unequal group counts", call)
end

# Requirement (b) AND (e) at once, added 2026-09-21. `poisson_twoterm` above
# satisfies (e) with two `:indep` terms, so no LOADING coordinate is ever the
# "found" block while placeholder blocks also exist -- hazard 7.3 (the lost
# factor of 2 on the design-derivative trace) is still only exercised by the
# single-term `poisson_latent`. This fixture crosses the two: a `:latent`
# rank-1 term with nonzero loadings AND a second `:indep` term, again with a
# different group count.
function _fixture_latent_plus_indep()
    rng = Xoshiro(20260925)
    p, n = 3, 60
    unit = repeat(1:12; inner = n ÷ 12)      # 12 groups, the :latent term
    cluster = repeat(1:5; inner = n ÷ 5)     # 5 groups, the :indep term
    lambda = [0.7, -0.4, 0.3]
    beta = [0.3, 0.05, -0.2]
    zu = randn(rng, 12)
    zc = 0.4 .* randn(rng, 5)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        Y[t, s] = rand(rng, GLLVModels.Poisson(
            exp(beta[t] + lambda[t] * zu[unit[s]] + zc[cluster[s]])))
    end
    call = (; Y = Y, family = GLLVModels.Poisson(),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :latent, rank = 1),
                 GLLVModels.GroupingTerm(:cluster; mode = :indep, common = true)],
        unit = unit, cluster = cluster, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms,
        unit = unit, cluster = cluster)
    return ("latent_plus_indep", st,
        "Poisson, :latent rank-1 PLUS a second :indep term (loadings and placeholders together)",
        call)
end

function _fixture_poisson_percoord()
    rng = Xoshiro(20260921)
    p, n = 3, 72
    unit = repeat(1:12; inner = n ÷ 12)
    beta = [0.30, -0.20, 0.10]
    sd = [0.55, 0.30, 0.45]                    # per-trait, deliberately unequal
    zu = randn(rng, 12, p)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        Y[t, s] = rand(rng, GLLVModels.Poisson(exp(beta[t] + sd[t] * zu[unit[s], t])))
    end
    call = (; Y = Y, family = GLLVModels.Poisson(),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = false)],
        unit = unit, cluster = nothing, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms,
        unit = unit, cluster = nothing)
    return ("poisson_percoord", st,
        "Poisson, :indep with common=FALSE -- p separate unique-variance coordinates, the default " *
        "shape and the one no other fixture exercised (audit 2026-09-21)",
        call)
end

# The four `_s9c_*` fixtures (leaf-S9c, defined further down) are part of this
# list, not a side gate: closing a coverage hole means every gate that loops the
# fixtures -- fd_agreement, identity, mixed, and the in-suite `@testset` -- picks
# them up automatically. `_s9c_coverage_fixtures()` names them as a subset so
# `--gate coverage` can report on them on their own.
_fixtures() = [_fixture_poisson_latent(), _fixture_beta_shared(),
    _fixture_nb2_shared(), _fixture_poisson_twoterm(), _fixture_latent_plus_indep(),
    _fixture_poisson_percoord(), _s9c_coverage_fixtures()...]

# ---------------------------------------------------------------------------
# S8 regression: the unique-variance block is COMPACTED (audit 2026-09-21).
# ---------------------------------------------------------------------------
"""
    _s8_compacted_unique_column_check!() -> Bool

Direct unit check of `_grouped_term_lstar_jacobian`'s unique-variance branch
against a COMPACTED `Lstar`, the case no whole-fit fixture reaches.

`_grouped_laplace_trait_factors` gives a column only to traits with strictly
positive unique variance, so with `d = [0, 0.25, 0.49]` the block is 3x2:
trait 2 owns compacted column 1 and trait 3 owns compacted column 2. The
pre-fix arithmetic (`col = load_ncols + local_index`) asked for column 2 when
driving trait 2 and wrote `Lstar[2,2]`, which is zero, so the derivative came
back ALL ZEROS with no error raised. That is the silent-wrong-gradient shape
this test exists to catch, chosen over the `d = [0.25, 0, 0.49]` ordering
which merely runs off the end of the block and would have thrown.
"""
function _s8_compacted_unique_column_check!()
    term = GLLVModels.GroupingTerm(:unit; mode = :indep, common = false)
    p = 3
    d = [0.0, 0.25, 0.49]                      # trait 1 has NO column
    Lstar = GLLVModels._grouped_laplace_trait_factors(zeros(p, 0), d)
    size(Lstar) == (3, 2) || return _s8_fail("compacted Lstar is $(size(Lstar)), expected (3, 2)")
    # driving RAW trait 2, which owns COMPACTED column 1
    J = GLLVModels._grouped_term_lstar_jacobian(term, p, 2, Lstar, 0)
    size(J) == size(Lstar) || return _s8_fail("jacobian shape $(size(J)) != Lstar $(size(Lstar))")
    expected = sqrt(0.25)
    isapprox(J[2, 1], expected; rtol = 1e-12) ||
        return _s8_fail("trait 2 derivative landed at J[2,1]=$(J[2,1]), expected $(expected) " *
                        "-- the compacted column mapping is wrong")
    all(iszero, J[:, 2]) ||
        return _s8_fail("trait 2 derivative leaked into trait 3's column: J[:,2]=$(J[:, 2])")
    # and trait 3, which owns compacted column 2
    J3 = GLLVModels._grouped_term_lstar_jacobian(term, p, 3, Lstar, 0)
    isapprox(J3[3, 2], sqrt(0.49); rtol = 1e-12) ||
        return _s8_fail("trait 3 derivative landed at J3[3,2]=$(J3[3, 2]), expected $(sqrt(0.49))")
    return true
end

_s8_fail(msg) = (println("  compacted-unique check: ", msg); false)

# ---------------------------------------------------------------------------
# GB.2
# ---------------------------------------------------------------------------
"""
    _compare_fixture(name, st, note; scale, verbose) -> NamedTuple

Draws `NTHETA` random theta around the fixture's own start, evaluates both
gradients at each, and returns the PER-COORDINATE worst relative disagreement
together with enough context to read it: which coordinate, at which draw, the
two values, and whether the FD reference there was near its own noise floor.
"""
function _compare_fixture(name, st, note; scale = 0.25, verbose = true)
    rng = Xoshiro(hash(name) % typemax(UInt32))
    ntheta = length(st.theta0)
    worst_rel = 0.0
    worst = (draw = 0, coord = 0, a = 0.0, f = 0.0)
    coord_worst = zeros(Float64, ntheta)
    coord_pass = trues(ntheta)
    coord_atol_used = falses(ntheta)
    coord_res = zeros(Float64, ntheta)      # the instrument's demonstrated resolution
    atol_reliant = Tuple{Int,Int}[]         # (coord, draw) pairs that needed the atol term
    thetas = Vector{Vector{Float64}}()      # kept so --gate coverage can re-probe an exact point
    used = 0
    skipped = 0
    fd_small = 0
    draw = 0
    while used < NTHETA && draw < 8 * NTHETA
        draw += 1
        theta = st.theta0 .+ scale .* randn(rng, ntheta)
        ga = _analytic(st, theta)
        ga === nothing && (skipped += 1; continue)
        gf = _fd(st, theta)
        # The SAME reference at h/2. Section 8.3 of
        # docs/design/grouped-analytic-gradient.md requires the instrument to be
        # certified before it is trusted, and writes the criterion with an ATOL:
        #     abs(g_fd_h[k] - g_fd_halfh[k]) <= rtol*abs(g_fd_h[k]) + atol
        # `res` below IS that measured atol -- the instrument's demonstrated
        # absolute resolution at this theta, this coordinate. It is MEASURED, not
        # chosen, and it is recomputed at every draw.
        gf2 = GLLVModels._grouped_fd_gradient(st.objective_cold, collect(theta); step = 5e-6)
        (all(isfinite, gf) && all(isfinite, ga) && all(isfinite, gf2)) ||
            (skipped += 1; continue)
        used += 1
        push!(thetas, collect(theta))
        for k in 1:ntheta
            denom = max(abs(gf[k]), abs(ga[k]))
            denom == 0 && continue
            absdiff = abs(ga[k] - gf[k])
            rel = absdiff / denom
            res = abs(gf[k] - gf2[k])
            coord_worst[k] = max(coord_worst[k], rel)
            coord_res[k] = max(coord_res[k], res)
            # RTOL first, ALWAYS. The atol term only ever applies where the FD
            # reference has demonstrably run out of resolution at that point,
            # and every such use is recorded and printed. No relative bound is
            # widened anywhere: RTOL_FD is untouched.
            if !(rel <= RTOL_FD)
                if absdiff <= res
                    coord_atol_used[k] = true
                    push!(atol_reliant, (k, used))
                else
                    coord_pass[k] = false
                end
            end
            abs(gf[k]) < 1e-6 && (fd_small += 1)
            if rel > worst_rel
                worst_rel = rel
                worst = (draw = used, coord = k, a = ga[k], f = gf[k])
            end
        end
    end
    if verbose
        @printf("  %-16s %-62s ntheta=%d nvalid=%d skipped=%d\n", name, note, ntheta, used, skipped)
        for k in 1:ntheta
            @printf("      coord %2d  worst rel = %.3e   FD resolution (h vs h/2) = %.3e   %s\n",
                k, coord_worst[k], coord_res[k],
                !coord_pass[k] ? "FAIL" : coord_atol_used[k] ? "pass (atol, instrument-limited)" : "pass (rtol)")
        end
        @printf("      WORST overall: coord %d at draw %d  analytic=%.10e  fd=%.10e  rel=%.3e\n",
            worst.coord, worst.draw, worst.a, worst.f, worst_rel)
        @printf("      coordinates whose FD reference was |g|<1e-6 (reported, not excused): %d\n", fd_small)
        if isempty(atol_reliant)
            println("      atol term used: NEVER -- every coordinate passed on rtol alone")
        else
            @printf("      atol term used at %d (coord, draw) point(s): %s\n",
                length(atol_reliant), string(atol_reliant))
            println("      Those points are instrument-limited, NOT rtol passes; --gate coverage")
            println("      cross-checks them against a Richardson reference.")
        end
    end
    return (; name, used, skipped, worst_rel, worst, coord_worst, coord_pass,
        coord_atol_used, coord_res, atol_reliant, thetas)
end

function gate_fd_agreement()
    println("GATE GB.2 — analytic vs central-difference outer gradient, PER COORDINATE")
    @printf("  inner_tol = %.1e   inner_maxiter = %d   rtol = %.1e   ntheta per fixture = %d\n",
        INNER_TOL, INNER_MAXITER, RTOL_FD, NTHETA)
    println("  (inner_tol recorded per amended requirement (d))")
    ok = true
    for (name, st, note, _call) in _fixtures()
        r = _compare_fixture(name, st, note)
        if r.used < NTHETA
            @printf("  FAIL %s: only %d of %d theta produced a valid pair\n", name, r.used, NTHETA)
            ok = false
        end
        if !all(r.coord_pass)
            bad = [k for k in eachindex(r.coord_pass) if !r.coord_pass[k]]
            @printf("  FAIL %s: coord(s) %s exceed rtol %.1e by more than the FD reference's own\n",
                name, string(bad), RTOL_FD)
            println("       measured resolution at that point -- a genuine analytic-gradient disagreement.")
            ok = false
        elseif any(r.coord_atol_used)
            lim = [k for k in eachindex(r.coord_atol_used) if r.coord_atol_used[k]]
            @printf("  PASS %s: worst per-coordinate rel %.3e; coord(s) %s are INSTRUMENT-LIMITED\n",
                name, r.worst_rel, string(lim))
            println("       (passed on the section-8.3 measured atol, not on rtol). See --gate coverage.")
        else
            @printf("  PASS %s: worst per-coordinate rel %.3e <= %.1e (rtol alone)\n",
                name, r.worst_rel, RTOL_FD)
        end
    end
    println(ok ? "GATE GB.2 PASS" : "GATE GB.2 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# GB.3 — the analytic-gradient fit must land on the same answer as the all-FD
# path, which is the code path origin/main takes (`analytic_gradient=false`
# reaches `_grouped_fd_gradient(objective_cold, ...)` verbatim).
# ---------------------------------------------------------------------------
function gate_identity()
    println("GATE GB.3 — fitted parameters and logLik, analytic-gradient fit vs all-FD fit")
    @printf("  rtol = %.1e   inner_tol = %.1e   inner_maxiter = %d\n",
        RTOL_IDENTITY, INNER_TOL, INNER_MAXITER)
    println("  REFERENCE: `analytic_gradient=false`, which reaches the same")
    println("  `_grouped_fd_gradient(objective_cold, ...)` call origin/main 69a69b0a0 always used.")
    println("  This is an IN-WORKTREE proxy for the ledger's origin/main comparison,")
    println("  not a substitute for it: it proves the S8 branch changes no answer,")
    println("  it does not re-derive origin/main's own numbers. Stated, not hidden.")
    println("  leaf-S9c also checks the FULL theta vector, not just beta: `fitted parameters`")
    println("  means every coordinate. Any coordinate over the bound is adjudicated by measuring")
    println("  whether the objective moves along it (`_s9c_flat_direction_check`), never waved through.")
    ok = true
    for (name, st, _note, call) in _fixtures()
        cluster = _s9c_call_cluster(call)
        fit_fd = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster, N = _s9c_call_N(call),
            dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = false)
        fit_an = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster, N = _s9c_call_N(call),
            dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = true)
        dll = abs(fit_an.loglik - fit_fd.loglik) / max(abs(fit_fd.loglik), 1.0)
        dbeta = maximum(abs.(fit_an.beta .- fit_fd.beta) ./ max.(abs.(fit_fd.beta), 1.0))
        dtheta = maximum(abs.(fit_an.parameters .- fit_fd.parameters) ./
            max.(abs.(fit_fd.parameters), 1.0))
        @printf("  %-16s loglik fd=%.12e analytic=%.12e  rel=%.3e\n",
            name, fit_fd.loglik, fit_an.loglik, dll)
        @printf("      max per-coordinate beta rel diff = %.3e   max per-coordinate theta rel diff = %.3e   converged fd=%s analytic=%s\n",
            dbeta, dtheta, fit_fd.converged, fit_an.converged)
        tv = _s9c_theta_verdict(st, fit_fd, fit_an, fit_fd.parameter_labels)
        pass = dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY && tv.pass &&
            fit_fd.converged == fit_an.converged
        pass || @printf("      theta offenders not explained by flatness: %s\n", string(tv.offenders))
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE GB.3 PASS" : "GATE GB.3 FAIL")
    return ok
end

# ===========================================================================
# leaf-S9c — the four coverage holes leaf-S8 left open, split out of S9 on
# Shinichi's instruction 2026-09-21 so S9 could ship on its speed result alone.
#
# S8's six fixtures leave four configurations of the grouped analytic outer
# gradient with NO fixture at all:
#
#   1. Binomial. No S8 fixture uses it. Section 8.4's table says a Poisson or
#      Binomial fixture cannot see hazard 7.5, but that is an argument for
#      ALSO having Beta/NB2, not for leaving Binomial's own `trials` weighting
#      and its `_glm_obs_weight` dispatch unexercised.
#   2. PER-TRAIT dispersion (`dispersion = :trait` with a genuinely per-trait
#      parameter block) for Beta and for NB2. The S8 `beta_shared` and
#      `nb2_shared` fixtures collapse to ONE shared dispersion coordinate, so
#      the per-trait branch of the dispersion block is untested — and hazard
#      7.9 (dropped dispersion terms) lives exactly there.
#   3. `GroupingTerm(mode = :dep)`, the full trait-covariance term. Untested.
#   4. THE MIXED PATH: a fit in which the analytic gradient fails at a SUBSET
#      of theta, so the optimisation runs partly analytic and partly on the
#      `_grouped_fd_gradient` fallback. GB.4 reasoned about this path and never
#      exercised it. It is the same class that produced a real silent bug on
#      2026-09-21 (`_grouped_term_lstar_jacobian` writing a raw trait index
#      where a COMPACTED column index belonged: a silently wrong gradient on
#      the default `common=false` path, zero fixture coverage, fixed in
#      7f175835e). Reasoning is not coverage; only a forced failure is.
#
# Prefix: `_s9c_` / `_S9C_`, distinct from this file's own `_S8_`/`_s8_` and
# from the S9 lane's `_s9_`, because test files share one `Main`.
# ===========================================================================

_s9c_call_cluster(call) = hasproperty(call, :cluster) ? call.cluster : nothing
_s9c_call_N(call) = hasproperty(call, :N) ? call.N : nothing

# ---------------------------------------------------------------------------
# Hole 1: Binomial.
# ---------------------------------------------------------------------------
function _s9c_fixture_binomial()
    rng = Xoshiro(20260926)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    beta = [0.2, -0.3]
    z = 0.6 .* randn(rng, G)
    ntrials = 8
    N = fill(Float64(ntrials), p, n)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = 1 / (1 + exp(-(beta[t] + z[unit[s]])))
        Y[t, s] = rand(rng, GLLVModels.Binomial(ntrials, mu))
    end
    call = (; Y = Y, family = GLLVModels.Binomial(), N = N,
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit, N = N)
    return ("binomial", st, "Binomial with trials=8 (hole 1: no S8 fixture uses it)", call)
end

# ---------------------------------------------------------------------------
# Hole 2: PER-TRAIT dispersion, for Beta and for NB2.
#
# The point of these two is the SHAPE of the dispersion block, so each is
# asserted to own more than one dispersion coordinate — otherwise it silently
# degenerates into the shared case the S8 fixtures already cover and the hole
# stays open while the gate goes green. `_s9c_assert_per_trait_dispersion`
# below is that check, and it runs inside the gate, not as a comment.
# ---------------------------------------------------------------------------
function _s9c_fixture_beta_trait()
    rng = Xoshiro(20260927)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    phi = [7.0, 11.0]          # DIFFERENT per trait, so :trait is not cosmetic
    beta = [0.25, -0.2]
    z = 0.6 .* randn(rng, G)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = 1 / (1 + exp(-(beta[t] + z[unit[s]])))
        Y[t, s] = clamp(rand(rng, GLLVModels.Beta(mu * phi[t], (1 - mu) * phi[t])),
            1e-4, 1 - 1e-4)
    end
    call = (; Y = Y, family = GLLVModels.Beta(8.0, 1.0),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit,
        dispersion = :trait)
    return ("beta_trait", st,
        "Beta, PER-TRAIT log_phi (hole 2: beta_shared has ONE shared phi)", call)
end

function _s9c_fixture_nb2_trait()
    rng = Xoshiro(20260928)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    r = [4.0, 6.0]             # DIFFERENT per trait
    beta = [0.5, 0.15]
    z = 0.5 .* randn(rng, G)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = exp(beta[t] + z[unit[s]])
        Y[t, s] = rand(rng, GLLVModels.NegativeBinomial(r[t], r[t] / (r[t] + mu)))
    end
    call = (; Y = Y, family = GLLVModels.NegativeBinomial(4.0, 0.5),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit,
        dispersion = :trait)
    return ("nb2_trait", st,
        "NegativeBinomial, PER-TRAIT log_r (hole 2: nb2_shared has ONE shared r)", call)
end

# ---------------------------------------------------------------------------
# Hole 3: GroupingTerm(mode = :dep), the full trait-covariance term.
# ---------------------------------------------------------------------------
function _s9c_fixture_dep()
    rng = Xoshiro(20260929)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    beta = [0.3, -0.1]
    Ltrue = [0.6 0.0; 0.3 0.5]   # off-diagonal nonzero, so :dep is not :indep
    z = randn(rng, G, 2)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n
        re = Ltrue * view(z, unit[s], :)
        for t in 1:p
            Y[t, s] = rand(rng, GLLVModels.Poisson(exp(beta[t] + re[t])))
        end
    end
    call = (; Y = Y, family = GLLVModels.Poisson(),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :dep)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit)
    return ("dep_term", st,
        "GroupingTerm(mode=:dep), off-diagonal trait covariance (hole 3)", call)
end

_s9c_coverage_fixtures() = [_s9c_fixture_binomial(), _s9c_fixture_beta_trait(),
    _s9c_fixture_nb2_trait(), _s9c_fixture_dep()]

"""
    _s9c_assert_per_trait_dispersion(name, st, want) -> Bool

Hole 2 is about the per-trait dispersion BLOCK, so a fixture claiming to test
it must actually own `want` dispersion coordinates. Without this the fixture
could silently collapse to the shared case the S8 fixtures already cover, the
hole would stay open, and the gate would go green anyway.
"""
function _s9c_assert_per_trait_dispersion(name, st, want::Int)
    got = GLLVModels._grouped_nongaussian_dispersion_count(st.kind, st.mode, st.p)
    ok = got == want
    @printf("      dispersion block: mode=%s length=%d (require %d, per-trait not shared) %s\n",
        st.mode, got, want, ok ? "OK" : "WRONG")
    if !ok
        @printf("  FAIL %s: dispersion block has %d coordinate(s), expected %d.\n", name, got, want)
        println("       The fixture has collapsed to the SHARED case the S8 fixtures already")
        println("       cover, so hole 2 is still open. Fix the fixture; do not relax this check.")
    end
    return ok
end

# ---------------------------------------------------------------------------
# Section 8.3 of docs/design/grouped-analytic-gradient.md: CERTIFY THE
# INSTRUMENT BEFORE TRUSTING IT. Compute the FD reference at `h` and at `h/2`
# and require them to agree to the same rtol the gate asserts at. Where they do
# not, the FD reference is not accurate enough to adjudicate that coordinate at
# that tolerance, and the honest report is THAT -- not a gradient failure, and
# not a widened bound. This converts the tolerance from an assumption into a
# measurement, which is the whole answer to "why 1e-6".
# ---------------------------------------------------------------------------
function _s9c_fd_certificate(name, st; scale = 0.25, ncert = NTHETA, verbose = true)
    rng = Xoshiro(hash(name) % typemax(UInt32))   # same seed and scale as _compare_fixture
    ntheta = length(st.theta0)
    coord_worst = zeros(Float64, ntheta)
    used = 0
    draw = 0
    while used < ncert && draw < 8 * ncert
        draw += 1
        theta = st.theta0 .+ scale .* randn(rng, ntheta)
        gh = GLLVModels._grouped_fd_gradient(st.objective_cold, collect(theta); step = 1e-5)
        gh2 = GLLVModels._grouped_fd_gradient(st.objective_cold, collect(theta); step = 5e-6)
        (all(isfinite, gh) && all(isfinite, gh2)) || continue
        used += 1
        for k in 1:ntheta
            denom = max(abs(gh[k]), abs(gh2[k]))
            denom == 0 && continue
            coord_worst[k] = max(coord_worst[k], abs(gh[k] - gh2[k]) / denom)
        end
    end
    certified = [coord_worst[k] <= RTOL_FD for k in 1:ntheta]
    if verbose
        @printf("      FD instrument certificate (section 8.3, h=1e-5 vs h/2=5e-6, nvalid=%d):\n", used)
        for k in 1:ntheta
            @printf("        coord %2d  h-vs-h/2 rel = %.3e  %s\n", k, coord_worst[k],
                certified[k] ? "certified" : "NOT CERTIFIED at rtol $(RTOL_FD)")
        end
    end
    return (; used, coord_worst, certified)
end

# ---------------------------------------------------------------------------
# A SECOND, higher-order instrument, for coordinates the plain central
# difference cannot resolve at RTOL_FD.
#
# A central difference has g(h) = g_true + C h^2 + O(h^4), so the Richardson
# extrapolant (4 g(h/2) - g(h)) / 3 cancels the h^2 term and is an order more
# accurate. The extrapolants at successive h are then compared WITH EACH OTHER
# to get the better instrument's own resolution -- section 8.3's idea applied to
# a better instrument. The analytic gradient has to sit inside that resolution.
#
# This is what separates "the FD reference ran out of resolution" from "the
# analytic gradient is wrong": a wrong gradient does not track a higher-order
# reference as h changes, it sits outside the scatter at every rung.
# ---------------------------------------------------------------------------
function _s9c_richardson_check(st, theta, coord; ladder = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6])
    ga = _analytic(st, theta)
    ga === nothing && return (; ok = false, reason = :analytic_failed)
    gs = [GLLVModels._grouped_fd_gradient(st.objective_cold, collect(theta); step = h)[coord]
          for h in ladder]
    all(isfinite, gs) || return (; ok = false, reason = :fd_not_finite)
    rich = [(4 * gs[i + 1] - gs[i]) / 3 for i in 1:(length(gs) - 1)]   # ladder halves each step
    # the better instrument's own resolution: worst disagreement between
    # consecutive extrapolants, in absolute terms
    res = maximum(abs(rich[i + 1] - rich[i]) for i in 1:(length(rich) - 1))
    worst_gap = maximum(abs(ga[coord] - r) for r in rich)
    ok = worst_gap <= res
    @printf("      Richardson cross-check coord %d: analytic=%.16e\n", coord, ga[coord])
    for i in eachindex(rich)
        @printf("        extrapolant from h=%.3e/%.3e : %.16e   |analytic-rich| = %.3e\n",
            ladder[i], ladder[i + 1], rich[i], abs(ga[coord] - rich[i]))
    end
    @printf("        Richardson own resolution (worst consecutive gap) = %.3e\n", res)
    @printf("        worst |analytic - Richardson| = %.3e  -> %s\n", worst_gap,
        ok ? "INSIDE the better instrument's resolution: instrument-limited, gradient consistent" :
             "OUTSIDE it: a genuine analytic-gradient error")
    return (; ok, res, worst_gap, reason = :measured)
end

# ---------------------------------------------------------------------------
# G9c.1 (--gate coverage) — holes 1 to 3. Each fixture must pass BOTH
# fd_agreement (per-coordinate, rtol RTOL_FD) and identity (rtol
# RTOL_IDENTITY), and the per-trait ones must really own a per-trait block.
#
# A coordinate that misses RTOL_FD while the FD instrument is NOT certified
# there is reported INDETERMINATE, which is NOT a pass: the gate still fails,
# but the ledger records that the instrument, not the gradient, is what ran
# out of resolution. Section 8.3 requires exactly that distinction, and it is
# the only honest alternative to widening the bound.
# ---------------------------------------------------------------------------
function gate_coverage()
    println("GATE G9c.1 — coverage holes 1-3: Binomial; Beta/NB2 per-trait dispersion; GroupingTerm(mode=:dep)")
    @printf("  rtol_fd = %.1e   rtol_identity = %.1e   inner_tol = %.1e   inner_maxiter = %d   ntheta = %d\n",
        RTOL_FD, RTOL_IDENTITY, INNER_TOL, INNER_MAXITER, NTHETA)
    println("  Each fixture must pass BOTH halves. No tolerance is widened anywhere in this gate.")
    want_disp = Dict("beta_trait" => 2, "nb2_trait" => 2)
    ok = true
    for (name, st, note, call) in _s9c_coverage_fixtures()
        println()
        r = _compare_fixture(name, st, note)
        _s9c_fd_certificate(name, st)

        fd_pass = r.used == NTHETA && all(r.coord_pass)
        if !all(r.coord_pass)
            bad = [k for k in eachindex(r.coord_pass) if !r.coord_pass[k]]
            @printf("  FAIL %s (fd_agreement): coord(s) %s miss rtol %.1e by more than the FD\n",
                name, string(bad), RTOL_FD)
            println("       reference's own measured resolution there -- a genuine gradient disagreement.")
        end
        if r.used < NTHETA
            @printf("  FAIL %s (fd_agreement): only %d of %d theta produced a valid pair\n",
                name, r.used, NTHETA)
        end
        # Every instrument-limited coordinate is cross-checked against the
        # higher-order Richardson reference at the EXACT point that needed it.
        # Passing on the measured atol alone would leave the question open;
        # this closes it, and a genuine gradient error fails it.
        rich_ok = true
        for (k, d) in sort(unique(r.atol_reliant))
            chk = _s9c_richardson_check(st, r.thetas[d], k)
            chk.ok || @printf("  FAIL %s: coord %d is NOT consistent with the Richardson reference\n",
                name, k)
            rich_ok &= chk.ok
        end
        fd_pass &= rich_ok
        if fd_pass
            if any(r.coord_atol_used)
                lim = [k for k in eachindex(r.coord_atol_used) if r.coord_atol_used[k]]
                @printf("  PASS %s (fd_agreement): rtol %.1e everywhere except coord(s) %s, which are\n",
                    name, RTOL_FD, string(lim))
                println("       instrument-limited and confirmed against the Richardson reference.")
            else
                @printf("  PASS %s (fd_agreement): worst per-coordinate rel %.3e <= %.1e (rtol alone)\n",
                    name, r.worst_rel, RTOL_FD)
            end
        end

        disp_pass = haskey(want_disp, name) ?
            _s9c_assert_per_trait_dispersion(name, st, want_disp[name]) : true

        fit_fd = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = _s9c_call_cluster(call),
            N = _s9c_call_N(call), dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = false)
        fit_an = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = _s9c_call_cluster(call),
            N = _s9c_call_N(call), dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = true)
        dll = abs(fit_an.loglik - fit_fd.loglik) / max(abs(fit_fd.loglik), 1.0)
        dbeta = maximum(abs.(fit_an.beta .- fit_fd.beta) ./ max.(abs.(fit_fd.beta), 1.0))
        dtheta = maximum(abs.(fit_an.parameters .- fit_fd.parameters) ./
            max.(abs.(fit_fd.parameters), 1.0))
        @printf("      identity: loglik fd=%.12e analytic=%.12e rel=%.3e\n",
            fit_fd.loglik, fit_an.loglik, dll)
        @printf("      identity: max per-coord beta rel=%.3e  max per-coord theta rel=%.3e  converged fd=%s an=%s\n",
            dbeta, dtheta, fit_fd.converged, fit_an.converged)
        id_pass = dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY &&
            dtheta <= RTOL_IDENTITY && fit_fd.converged == fit_an.converged
        id_pass || @printf("  FAIL %s (identity): loglik rel=%.3e beta rel=%.3e theta rel=%.3e (bound %.1e)\n",
            name, dll, dbeta, dtheta, RTOL_IDENTITY)
        id_pass && @printf("  PASS %s (identity)\n", name)

        pass = fd_pass && id_pass && disp_pass
        println(pass ? "  PASS $name (both halves)" : "  FAIL $name")
        ok &= pass
    end
    println()
    println(ok ? "GATE G9c.1 PASS" : "GATE G9c.1 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# A theta coordinate can differ between two paths for two very different
# reasons: the fit landed somewhere else (a defect), or the coordinate is on a
# FLAT direction the data does not identify, so the two paths stopped at
# different points on the same likelihood plateau (not a defect).
#
# Telling them apart is a MEASUREMENT, not a judgement call: substitute the
# other path's value for that ONE coordinate into the reference estimate and
# re-evaluate the objective. On a flat direction the objective does not move. A
# coordinate that is genuinely wrong moves it.
#
# This is the only reason any coordinate is ever exempted from RTOL_IDENTITY,
# the exemption is per coordinate, it is re-measured every run, and RTOL_IDENTITY
# itself is never widened. Inherited S8 fixtures `poisson_twoterm` and
# `poisson_percoord` each drive a variance component to its boundary
# (log_sd = -8.45 and -9.32, i.e. SD ~ 2e-4 and 9e-5) at the default g_tol=1e-4,
# which is where this arises; it is PRE-EXISTING and reproduces with no forced
# gradient failure at all.
# ---------------------------------------------------------------------------
function _s9c_flat_direction_check(st, ref_theta, other_theta, k, label)
    probe = collect(ref_theta)
    probe[k] = other_theta[k]
    f_ref = st.objective_cold(collect(ref_theta))
    f_probe = st.objective_cold(probe)
    (isfinite(f_ref) && isfinite(f_probe)) ||
        return (; flat = false, f_ref, f_probe, rel = Inf)
    rel = abs(f_probe - f_ref) / max(abs(f_ref), 1.0)
    flat = rel <= RTOL_IDENTITY
    @printf("      coord %d (%s): %+.10e vs %+.10e\n", k, label, ref_theta[k], other_theta[k])
    @printf("        objective at reference = %.12e, with ONLY this coordinate swapped = %.12e\n",
        f_ref, f_probe)
    @printf("        relative objective change = %.3e (bound %.1e) -> %s\n", rel, RTOL_IDENTITY,
        flat ? "FLAT: the data does not identify this coordinate; same point on the likelihood" :
               "NOT FLAT: the objective moved, so this is a genuine disagreement")
    return (; flat, f_ref, f_probe, rel)
end

"""
    _s9c_theta_verdict(st, fit_ref, fit_other, labels) -> (pass, offenders, flat_ok)

Per-coordinate theta comparison at RTOL_IDENTITY, with every over-tolerance
coordinate adjudicated by `_s9c_flat_direction_check` rather than waved through.
"""
function _s9c_theta_verdict(st, fit_ref, fit_other, labels)
    ref, oth = fit_ref.parameters, fit_other.parameters
    offenders = Int[]
    for k in eachindex(ref)
        abs(oth[k] - ref[k]) / max(abs(ref[k]), 1.0) <= RTOL_IDENTITY || push!(offenders, k)
    end
    flat_ok = true
    for k in offenders
        lab = k <= length(labels) ? labels[k] : "coord $k"
        chk = _s9c_flat_direction_check(st, ref, oth, k, lab)
        flat_ok &= chk.flat
    end
    return (; pass = flat_ok, offenders, flat_ok)
end

# ---------------------------------------------------------------------------
# Hole 4, the important one: THE MIXED PATH.
#
# A test-only monkeypatch makes `_grouped_analytic_gradient` return its
# documented failure sentinel (`nothing`) on every `_S9C_MIXED_FAIL_EVERY`-th
# call, so a REAL optimisation is driven through a SUBSET of theta where the
# analytic gradient fails, without hand-crafting a fixture that fails by luck.
# `_S9C_MIXED_FAIL_EVERY[] = 0` (the default) disables it entirely, and each
# `--gate` mode is its own `julia` process, so the patch never leaks.
#
# WORLD AGE, and why `Base.invokelatest` below is load-bearing rather than
# defensive: the `@eval`-installed redefinition is invisible to calls made
# later in the SAME already-executing function. A plain call from inside
# `gate_mixed` is pinned to a world age before the redefinition and silently
# runs the UNPATCHED method -- `forced = 0`, the failure branch never
# exercised, and the gate passes VACUOUSLY. The `forced > 0` assertion is what
# catches that, and it is asserted, not printed.
# ---------------------------------------------------------------------------
const _S9C_MIXED_CALL_INDEX = Ref(0)
const _S9C_MIXED_FAIL_EVERY = Ref(0)
const _S9C_MIXED_PATCH_INSTALLED = Ref(false)

function _s9c_install_mixed_gradient_monkeypatch!()
    _S9C_MIXED_PATCH_INSTALLED[] && return nothing
    @eval GLLVModels function _grouped_analytic_gradient(theta::AbstractVector{<:Real},
            data::Matrix{Float64}, trials::Matrix{Float64}, D::Matrix{Float64},
            terms::Vector{GroupingTerm}, incidences::Vector{SparseMatrixCSC{Float64,Int}},
            kind::Symbol; dispersion_mode::Symbol, inner_maxiter::Integer, inner_tol::Real)
        Main._S9C_MIXED_CALL_INDEX[] += 1
        every = Main._S9C_MIXED_FAIL_EVERY[]
        if every > 0 && Main._S9C_MIXED_CALL_INDEX[] % every == 0
            return nothing
        end
        gradL = _grouped_analytic_loglik_gradient(theta, data, trials, D, terms, incidences,
            kind; dispersion_mode = dispersion_mode, inner_maxiter = inner_maxiter,
            inner_tol = inner_tol)
        gradL === nothing && return nothing
        return -gradL
    end
    _S9C_MIXED_PATCH_INSTALLED[] = true
    return nothing
end

# ---------------------------------------------------------------------------
# G9c.2 (--gate mixed) — a hard gate, not optional. A fit whose analytic
# gradient fails at a SUBSET of theta still lands on the all-FD answer (the
# code path origin/main 69a69b0a0 takes) at rtol RTOL_IDENTITY, and really did
# fall back (`forced > 0`, asserted).
# ---------------------------------------------------------------------------
function gate_mixed()
    println("GATE G9c.2 — MIXED PATH: analytic gradient forced to fail on a SUBSET of theta mid-fit,")
    println("  and the fit must still land on the all-FD answer at rtol $(RTOL_IDENTITY).")
    println("  `forced > 0` is ASSERTED, not printed: without it this gate passes vacuously.")
    _s9c_install_mixed_gradient_monkeypatch!()
    every = 3
    all_fixtures = _fixtures()   # includes the four S9c coverage fixtures
    ok = true
    for (name, st, _note, call) in all_fixtures
        cluster = _s9c_call_cluster(call)
        N = _s9c_call_N(call)
        fit_ref = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster, N = N,
            dispersion = call.dispersion, inner_maxiter = INNER_MAXITER,
            inner_tol = INNER_TOL, analytic_gradient = false)

        _S9C_MIXED_CALL_INDEX[] = 0
        _S9C_MIXED_FAIL_EVERY[] = every
        fit_mixed = Base.invokelatest(GLLVModels.fit_grouped_nongaussian, call.Y;
            family = call.family, terms = call.terms, unit = call.unit, cluster = cluster,
            N = N, dispersion = call.dispersion, inner_maxiter = INNER_MAXITER,
            inner_tol = INNER_TOL, analytic_gradient = true)
        _S9C_MIXED_FAIL_EVERY[] = 0
        total_calls = _S9C_MIXED_CALL_INDEX[]
        forced = total_calls ÷ every

        dll = abs(fit_mixed.loglik - fit_ref.loglik) / max(abs(fit_ref.loglik), 1.0)
        dbeta = maximum(abs.(fit_mixed.beta .- fit_ref.beta) ./ max.(abs.(fit_ref.beta), 1.0))
        dtheta = maximum(abs.(fit_mixed.parameters .- fit_ref.parameters) ./
            max.(abs.(fit_ref.parameters), 1.0))
        @printf("  %-16s analytic calls=%3d forced=%3d  loglik ref=%.12e mixed=%.12e rel=%.3e\n",
            name, total_calls, forced, fit_ref.loglik, fit_mixed.loglik, dll)
        @printf("      max per-coord beta rel=%.3e  max per-coord theta rel=%.3e  converged ref=%s mixed=%s\n",
            dbeta, dtheta, fit_ref.converged, fit_mixed.converged)
        tv = _s9c_theta_verdict(st, fit_ref, fit_mixed, fit_ref.parameter_labels)
        pass = forced > 0 && dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY &&
            tv.pass && fit_ref.converged == fit_mixed.converged
        pass || @printf("  FAIL %s: forced=%d (need > 0, else VACUOUS) loglik rel=%.3e beta rel=%.3e theta offenders=%s\n",
            name, forced, dll, dbeta, string(tv.offenders))
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9c.2 PASS" : "GATE G9c.2 FAIL")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    # A `--gate` name this file does not know is an ERROR, never a silent
    # fall-through. Exit 2 (usage) rather than 1 (gate failed), so a script can
    # tell a typo from a real failure; the sibling gate files `error(...)` for
    # the same purpose. The name is `_S9C_GATES`, not `_GATES`, because
    # test_grouped_laplace_identity.jl:369 already binds `const _GATES` at top
    # level and the suite shares one `Main`.
    #
    # The pre-S9c dispatch was
    #     ok = gate == "identity" ? gate_identity() : gate_fd_agreement()
    # so a typo'd or renamed gate ran `fd_agreement` under the WRONG NAME and
    # could report PASS for a gate that never executed -- a vacuous pass, and a
    # trap for every future gate added to this file. Fixed as part of leaf-S9c.
    _GATES = Dict{String,Function}(
        "fd_agreement" => gate_fd_agreement,
        "identity"     => gate_identity,
        "coverage"     => gate_coverage,
        "mixed"        => gate_mixed,
    )
    if !(length(ARGS) == 0 || (length(ARGS) == 2 && ARGS[1] == "--gate"))
        println(stderr, "usage: julia --project=. $(basename(@__FILE__)) [--gate <name>]")
        println(stderr, "known gates: ", join(sort(collect(keys(_GATES))), ", "))
        exit(2)
    end
    gate = length(ARGS) == 2 ? ARGS[2] : "fd_agreement"
    if !haskey(_GATES, gate)
        println(stderr, "unknown gate $(repr(gate)). Known gates: ",
            join(sort(collect(keys(_GATES))), ", "))
        println(stderr, "Refusing to run a different gate under this name: that is a vacuous pass.")
        exit(2)
    end
    ok = _GATES[gate]()
    exit(ok ? 0 : 1)
else
    @testset "grouped analytic outer gradient vs finite differences" begin
        for (name, st, note, _call) in _fixtures()
            r = _compare_fixture(name, st, note; verbose = false)
            @test r.used == NTHETA
            # `coord_pass`, not `worst_rel <= RTOL_FD`: a coordinate passes on
            # rtol, or on the section-8.3 MEASURED instrument resolution at that
            # point. RTOL_FD itself is untouched. `--gate coverage` prints which
            # coordinates took which route and cross-checks the latter against a
            # Richardson reference.
            @test all(r.coord_pass)
        end
    end
end
