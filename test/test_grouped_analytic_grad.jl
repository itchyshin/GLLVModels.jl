# test/test_grouped_analytic_grad.jl — leaf-S8 gates GB.2 (fd_agreement) and
# GB.3 (identity); leaf-S9 gates G9.1-G9.7 (`--gate identity/hessian/
# nm_fallback/counts/coverage/mixed/final_gradient`; G9.8/G9.9 are the
# orchestrator's, not this file's).
#
# S9 (`.unlazy/grouped-analytic-20260920/gates/leaf-S9.md`) demotes the
# unconditional value-only Nelder-Mead phase to a fallback when the analytic
# gradient is in use, replaces the final O(nθ²) `_grouped_fd_hessian` with a
# Hessian obtained by finite-differencing the ANALYTIC gradient (D-274), and
# makes the FINAL REPORTED gradient the analytic one on the analytic path.
# New gates, following this file's own `_S8_`-prefix convention with `_S9_`:
#   --gate hessian        G9.2  grad-FD Hessian vs `_grouped_fd_hessian`,
#                                compared as STANDARD ERRORS, rtol 1e-4.
#   --gate nm_fallback     G9.3  the FD path (nelder_mead=true default,
#                                hessian=:fd default) runs verbatim; a
#                                mid-fit forced analytic-gradient failure
#                                still completes rather than throwing.
#   --gate counts          G9.4  objective calls / inner Newton iterations
#                                bounded BELOW the S8-measured baseline on
#                                both bench fixtures, reported not pinned.
#   --gate coverage        G9.5  the four coverage holes S8 left (Binomial;
#                                Beta/NB2 dispersion=:trait; mode=:dep).
#   --gate mixed            G9.6 THE MIXED PATH: analytic gradient forced to
#                                fail at a SUBSET of theta during a real
#                                optimisation, still lands on the FD answer.
#   --gate final_gradient   G9.7 the reported gradient on the analytic path
#                                IS the analytic gradient.
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

using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, Printf, DelimitedFiles, Optim

const RTOL_FD = 1e-6      # GB.2, the FD reference's own accuracy. Never widened.
const RTOL_IDENTITY = 1e-8  # GB.3, and S9's G9.1/G9.6.
const RTOL_HESSIAN = 1e-4   # S9 G9.2 -- the FD Hessian oracle's own accuracy, per D-274. Never widened.
const RTOL_GRADIENT_NORM = 1e-6  # S9 G9.7.
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

_fixtures() = [_fixture_poisson_latent(), _fixture_beta_shared(),
    _fixture_nb2_shared(), _fixture_poisson_twoterm(), _fixture_latent_plus_indep(),
    _fixture_poisson_percoord()]

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
        (all(isfinite, gf) && all(isfinite, ga)) || (skipped += 1; continue)
        used += 1
        for k in 1:ntheta
            denom = max(abs(gf[k]), abs(ga[k]))
            denom == 0 && continue
            rel = abs(ga[k] - gf[k]) / denom
            coord_worst[k] = max(coord_worst[k], rel)
            # FD noise floor for a central difference with step 1e-5*max(1,|θ|)
            # on an objective of size |F|: roundoff ~ eps*|F|/h. Reported, not
            # used to excuse anything.
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
            @printf("      coord %2d  worst rel = %.3e\n", k, coord_worst[k])
        end
        @printf("      WORST overall: coord %d at draw %d  analytic=%.10e  fd=%.10e  rel=%.3e\n",
            worst.coord, worst.draw, worst.a, worst.f, worst_rel)
        @printf("      coordinates whose FD reference was |g|<1e-6 (reported, not excused): %d\n", fd_small)
    end
    return (; name, used, skipped, worst_rel, worst, coord_worst)
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
        if !(r.worst_rel <= RTOL_FD)
            @printf("  FAIL %s: worst per-coordinate rel %.3e exceeds rtol %.1e\n",
                name, r.worst_rel, RTOL_FD)
            ok = false
        else
            @printf("  PASS %s: worst per-coordinate rel %.3e <= %.1e\n", name, r.worst_rel, RTOL_FD)
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
    ok = true
    for (name, _st, _note, call) in _fixtures()
        cluster = hasproperty(call, :cluster) ? call.cluster : nothing
        fit_fd = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster,
            dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = false)
        fit_an = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster,
            dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = true)
        dll = abs(fit_an.loglik - fit_fd.loglik) / max(abs(fit_fd.loglik), 1.0)
        dbeta = maximum(abs.(fit_an.beta .- fit_fd.beta) ./ max.(abs.(fit_fd.beta), 1.0))
        @printf("  %-16s loglik fd=%.12e analytic=%.12e  rel=%.3e\n",
            name, fit_fd.loglik, fit_an.loglik, dll)
        @printf("      max per-coordinate beta rel diff = %.3e   converged fd=%s analytic=%s\n",
            dbeta, fit_fd.converged, fit_an.converged)
        pass = dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY &&
            fit_fd.converged == fit_an.converged
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE GB.3 PASS" : "GATE GB.3 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# S9 shared helper: pull `N` and `cluster` out of a fixture's `call`
# NamedTuple when present, matching `gate_identity`'s existing `cluster`
# pattern (not every fixture has either field).
# ---------------------------------------------------------------------------
_s9_call_cluster(call) = hasproperty(call, :cluster) ? call.cluster : nothing
_s9_call_N(call) = hasproperty(call, :N) ? call.N : nothing

# ---------------------------------------------------------------------------
# G9.2 (--gate hessian) — the grad-FD Hessian (D-274) vs `_grouped_fd_hessian`,
# compared as STANDARD ERRORS (the gate's own instruction: the two are
# different estimators of the same matrix and SEs are what users see, not raw
# Hessian entries). Both differenced at the SAME converged estimate, on all
# six fixtures (including Beta and NB2, as G9.2 requires).
# ---------------------------------------------------------------------------
function _hessian_standard_errors(H::Matrix{Float64})
    Hs = Symmetric((H .+ H') ./ 2)
    Hinv = try
        inv(Hs)
    catch
        return nothing
    end
    d = diag(Hinv)
    all(isfinite, d) && all(>(0.0), d) || return nothing
    return sqrt.(d)
end

function gate_hessian()
    println("GATE G9.2 — grad-FD Hessian (D-274) vs _grouped_fd_hessian, STANDARD ERRORS, PER COORDINATE")
    @printf("  rtol = %.1e   inner_tol = %.1e   inner_maxiter = %d\n", RTOL_HESSIAN, INNER_TOL, INNER_MAXITER)
    ok = true
    for (name, st, note, call) in _fixtures()
        fit = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
            unit = call.unit, cluster = _s9_call_cluster(call), N = _s9_call_N(call),
            dispersion = call.dispersion, inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = true)
        estimate = copy(fit.parameters)
        H_gradfd = GLLVModels._grouped_fd_hessian_from_gradient(
            v -> GLLVModels._grouped_analytic_gradient(v, st.data, st.trials, st.D, st.termvec,
                st.incidences, st.kind; dispersion_mode = st.mode,
                inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL),
            estimate)
        H_fd = GLLVModels._grouped_fd_hessian(st.objective_cold, estimate)
        se_gradfd = _hessian_standard_errors(H_gradfd)
        se_fd = _hessian_standard_errors(H_fd)
        if se_gradfd === nothing || se_fd === nothing
            @printf("  FAIL %s: Hessian not invertible/positive-definite at the converged estimate (grad_fd ok=%s, fd ok=%s)\n",
                name, se_gradfd !== nothing, se_fd !== nothing)
            ok = false
            continue
        end
        worst = 0.0; worst_k = 0
        for k in eachindex(se_fd)
            rel = abs(se_gradfd[k] - se_fd[k]) / abs(se_fd[k])
            @printf("      coord %2d  se_grad_fd=%.10e  se_fd=%.10e  rel=%.3e\n", k, se_gradfd[k], se_fd[k], rel)
            rel > worst && (worst = rel; worst_k = k)
        end
        pass = worst <= RTOL_HESSIAN
        @printf("  %-16s %-62s worst per-coord SE rel = %.3e (coord %d)\n", name, note, worst, worst_k)
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9.2 PASS" : "GATE G9.2 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# Shared monkeypatch infrastructure for G9.3(b) and G9.6 — forces
# `GLLVModels._grouped_analytic_gradient` to return `nothing` (its documented
# failure sentinel) on every `_S9_MIXED_FAIL_EVERY`-th call, so a fit can be
# driven through a SUBSET of theta where the analytic gradient fails without
# hand-crafting a fixture that fails "naturally". `_S9_MIXED_FAIL_EVERY[] =
# 0` (the default) disables it entirely. Installed only by the gates that use
# it, and each `--gate` mode is its own `julia` process (this file's own
# `--gate` convention), so the patch never leaks into another gate's run.
# ---------------------------------------------------------------------------
const _S9_MIXED_CALL_INDEX = Ref(0)
const _S9_MIXED_FAIL_EVERY = Ref(0)
const _S9_MIXED_PATCH_INSTALLED = Ref(false)

function _s9_install_mixed_gradient_monkeypatch!()
    _S9_MIXED_PATCH_INSTALLED[] && return nothing
    @eval GLLVModels function _grouped_analytic_gradient(theta::AbstractVector{<:Real},
            data::Matrix{Float64}, trials::Matrix{Float64}, D::Matrix{Float64},
            terms::Vector{GroupingTerm}, incidences::Vector{SparseMatrixCSC{Float64,Int}},
            kind::Symbol; dispersion_mode::Symbol, inner_maxiter::Integer, inner_tol::Real)
        Main._S9_MIXED_CALL_INDEX[] += 1
        every = Main._S9_MIXED_FAIL_EVERY[]
        if every > 0 && Main._S9_MIXED_CALL_INDEX[] % every == 0
            return nothing
        end
        gradL = _grouped_analytic_loglik_gradient(theta, data, trials, D, terms, incidences, kind;
            dispersion_mode = dispersion_mode, inner_maxiter = inner_maxiter, inner_tol = inner_tol)
        gradL === nothing && return nothing
        return -gradL
    end
    _S9_MIXED_PATCH_INSTALLED[] = true
    return nothing
end

# ---------------------------------------------------------------------------
# G9.3 (--gate nm_fallback) — the fallback contract.
# ---------------------------------------------------------------------------
function gate_nm_fallback()
    println("GATE G9.3 — fallback contract: analytic_gradient=false runs the pre-S9 path verbatim;")
    println("  a fit whose analytic gradient fails mid-optimisation still completes, warning not throwing.")
    (name, st, _note, call) = _fixture_poisson_latent()
    cluster = _s9_call_cluster(call)

    # (a) analytic_gradient=false's DEFAULTS (nelder_mead=true, hessian=:fd)
    # reproduce the pre-S9 path exactly, and Nelder-Mead actually EXECUTES --
    # checked as more inner-Laplace-fit calls than the demoted analytic-
    # default path costs, not merely assumed from the kwarg default.
    GLLVModels._grouped_chol_stats_reset!()
    fit_fd_default = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
        unit = call.unit, cluster = cluster, dispersion = call.dispersion,
        inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = false)
    calls_fd_default = GLLVModels._grouped_chol_stats().calls

    GLLVModels._grouped_chol_stats_reset!()
    fit_fd_explicit = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
        unit = call.unit, cluster = cluster, dispersion = call.dispersion,
        inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = false,
        nelder_mead = true, hessian = :fd)
    calls_fd_explicit = GLLVModels._grouped_chol_stats().calls

    GLLVModels._grouped_chol_stats_reset!()
    fit_an_default = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
        unit = call.unit, cluster = cluster, dispersion = call.dispersion,
        inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = true)
    calls_an_default = GLLVModels._grouped_chol_stats().calls

    @printf("  inner Laplace-fit calls: FD default (nelder_mead=true, hessian=:fd) = %d, FD explicit-kwargs = %d, analytic default (nelder_mead=false, hessian=:grad_fd) = %d\n",
        calls_fd_default, calls_fd_explicit, calls_an_default)

    default_matches_explicit = calls_fd_default == calls_fd_explicit &&
        isapprox(fit_fd_default.loglik, fit_fd_explicit.loglik; rtol = 1e-12) &&
        isapprox(fit_fd_default.beta, fit_fd_explicit.beta; rtol = 1e-12)
    default_matches_explicit ||
        println("  FAIL: analytic_gradient=false's DEFAULT kwargs do not reproduce nelder_mead=true, hessian=:fd given explicitly")

    nm_ran = calls_fd_default > calls_an_default
    nm_ran || @printf("  FAIL: expected the FD-default path (Nelder-Mead executing) to cost MORE inner Laplace fits than the demoted analytic-default path; got %d vs %d\n",
        calls_fd_default, calls_an_default)

    # `_grouped_fd_hessian` supplies H on the FD path: the fit's own
    # hessian_min_eigenvalue must match a FRESH direct call to
    # `_grouped_fd_hessian` at the SAME estimate.
    H_direct = GLLVModels._grouped_fd_hessian(st.objective_cold, fit_fd_default.parameters)
    me_direct = all(isfinite, H_direct) ? eigmin(Symmetric(H_direct)) : NaN
    hessian_matches = isfinite(me_direct) && isapprox(fit_fd_default.hessian_min_eigenvalue, me_direct; rtol = 1e-8)
    hessian_matches || @printf("  FAIL: fit's hessian_min_eigenvalue=%.6e does not match a direct _grouped_fd_hessian call=%.6e\n",
        fit_fd_default.hessian_min_eigenvalue, me_direct)

    # (b) a mid-fit forced analytic-gradient failure still completes, warning
    # rather than throwing (the `grad_fn` try/catch + nothing/non-finite
    # fallback is S8 machinery, unchanged by S9 -- this exercises it under a
    # SUBSET-failure regime rather than assuming it still holds).
    _s9_install_mixed_gradient_monkeypatch!()
    _S9_MIXED_CALL_INDEX[] = 0
    _S9_MIXED_FAIL_EVERY[] = 2
    completed = true
    fit_mixed = nothing
    try
        # `Base.invokelatest`, not a plain call: the `@eval GLLVModels` above ran
        # from WITHIN this same function's dynamic extent, so ordinary dispatch
        # here is still pinned to the world age from BEFORE the redefinition and
        # would silently call the UNPATCHED method (measured: without
        # `invokelatest`, `_S9_MIXED_CALL_INDEX` stays exactly 0 through this
        # call). `invokelatest` forces the lookup to the current world.
        fit_mixed = Base.invokelatest(GLLVModels.fit_grouped_nongaussian, call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster, dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = true)
    catch err
        completed = false
        @printf("  FAIL: fit threw instead of completing: %s\n", sprint(showerror, err))
    end
    _S9_MIXED_FAIL_EVERY[] = 0
    forced = _S9_MIXED_CALL_INDEX[] ÷ 2
    @printf("  mid-fit, every 2nd analytic-gradient call forced to `nothing`: total calls=%d forced>=%d completed=%s converged=%s\n",
        _S9_MIXED_CALL_INDEX[], forced, completed, completed ? fit_mixed.converged : "n/a")
    (completed && forced > 0) ||
        @printf("  FAIL: expected completion with the forced-failure branch actually exercised (completed=%s forced=%d)\n", completed, forced)

    ok = default_matches_explicit && nm_ran && hessian_matches && completed && forced > 0
    println(ok ? "GATE G9.3 PASS" : "GATE G9.3 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# G9.4 (--gate counts) — objective calls / inner Newton iterations, S9
# default (nelder_mead=false) vs the S8-measured baseline, on BOTH bench
# fixtures. Fixture CONSTRUCTION only is duplicated from
# bench/profile_grouped_glmm.jl (same seed for the large one, the same
# committed CSV for the small one) -- the bench DRIVER is not reused, because
# it shadows the PRE-S9 unconditional NM-then-BFGS sequence verbatim and
# would misreport call counts for the now-demoted default path. This harness
# mirrors the CURRENT `fit_grouped_nongaussian` control flow instead,
# wrapping the real objective/gradient factories with counters, never
# reimplementing their bodies.
# ---------------------------------------------------------------------------
const _S9_LARGE_SEED = 20260920
const _S9_LARGE_P = 3
const _S9_LARGE_N = 5000
const _S9_LARGE_G = 500

function _s9_make_large_fixture()
    rng = Xoshiro(_S9_LARGE_SEED)
    beta_true = [log(4.0), log(6.0), log(2.5)]
    sd_true = [0.4, 0.3, 0.5]
    group = rand(rng, 1:_S9_LARGE_G, _S9_LARGE_N)
    b = [sd_true[t] .* randn(rng, _S9_LARGE_G) for t in 1:_S9_LARGE_P]
    Y = zeros(Float64, _S9_LARGE_P, _S9_LARGE_N)
    for i in 1:_S9_LARGE_N, t in 1:_S9_LARGE_P
        Y[t, i] = rand(rng, GLLVModels.Poisson(exp(beta_true[t] + b[t][group[i]])))
    end
    return Y, group
end

function _s9_load_small_fixture()
    path = joinpath(@__DIR__, "..", "bench", "fixtures", "glmm_200x5.csv")
    M = readdlm(path, ',', Int)
    y = M[:, 1]; group = M[:, 2]
    return reshape(Float64.(y), 1, :), group
end

"""
    _s9_counted_fit(Y, group; g_tol, iterations, inner_maxiter, inner_tol) -> NamedTuple

Runs the S9 `fit_grouped_nongaussian` CONTROL FLOW (`nelder_mead=false`: BFGS
starts directly from `theta0`, with the safety-net fallbacks wired exactly as
`src/grouped_nongaussian_fit.jl` now implements them) on a single `:indep`
Poisson grouping fixture, wrapping the REAL objective factory's returned
closures with a call counter -- never reimplementing the objective body.
`_grouped_chol_stats()` is reset first and read after, so
`inner_iters_sum` is DERIVED from it via the documented (bench/
profile_grouped_glmm.jl header) `2*iterations(+1)` relationship, assuming
(nearly) every inner Laplace fit converges `:ok` -- reported, not silently
assumed: `chol_stats` is also returned so a reader can check that assumption.
"""
function _s9_counted_fit(Y::Matrix{Float64}, group::Vector{Int};
        g_tol = 1e-4, iterations = 100, inner_maxiter = 100, inner_tol = 1e-8)
    p, n = size(Y)
    terms = GLLVModels.GroupingTerm[GLLVModels.GroupingTerm(:unit; mode = :indep)]
    kind = GLLVModels._grouped_nongaussian_kind(GLLVModels.Poisson())
    mode = GLLVModels._grouped_nongaussian_dispersion_mode(kind, :trait)
    labels = GLLVModels._grouped_labels(n, terms; unit = group, unit_obs = nothing,
        cluster = nothing, cluster2 = nothing)
    incidences = [GLLVModels._grouped_incidence(v, n) for v in labels]
    data = Matrix{Float64}(Y)
    trials = GLLVModels._grouped_nongaussian_trials(data, nothing, kind)
    D = GLLVModels._trait_mean_design(p, n)
    theta0 = GLLVModels._grouped_nongaussian_initial_parameters(data, trials, D, terms,
        kind, GLLVModels.Poisson(), mode)

    objective_cold = GLLVModels._grouped_nongaussian_objective(data, trials, D, terms, incidences, kind;
        dispersion_mode = mode, inner_maxiter = Int(inner_maxiter), inner_tol = Float64(inner_tol),
        warm_start_inner = false)
    objective_warm = GLLVModels._grouped_nongaussian_objective(data, trials, D, terms, incidences, kind;
        dispersion_mode = mode, inner_maxiter = Int(inner_maxiter), inner_tol = Float64(inner_tol),
        warm_start_inner = true)

    obj_calls = Ref(0)
    counted_cold = value -> (obj_calls[] += 1; objective_cold(value))
    counted_warm = value -> (obj_calls[] += 1; objective_warm(value))

    grad_fn = value -> begin
        g = try
            GLLVModels._grouped_analytic_gradient(value, data, trials, D, terms, incidences, kind;
                dispersion_mode = mode, inner_maxiter = Int(inner_maxiter), inner_tol = Float64(inner_tol))
        catch
            nothing
        end
        (g === nothing || !all(isfinite, g)) ? GLLVModels._grouped_fd_gradient(counted_cold, value) : g
    end

    GLLVModels._grouped_chol_stats_reset!()

    run_nm = () -> Optim.optimize(counted_warm, theta0, Optim.NelderMead(),
        Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
    result = nothing
    candidate = collect(theta0)
    candidate_gradient = grad_fn(candidate)
    if !all(isfinite, candidate_gradient)
        result = run_nm()
        candidate = collect(Optim.minimizer(result))
        candidate_gradient = grad_fn(candidate)
    end
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= grad_fn(value))
        refined = try
            Optim.optimize(counted_warm, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = counted_cold(refined_estimate)
            if isfinite(refined_value) && !GLLVModels._nll_failed(refined_value) &&
                    refined_value <= counted_cold(candidate)
                result = refined
            end
        end
    end
    result === nothing && (result = run_nm())
    estimate = collect(Optim.minimizer(result))
    value = counted_cold(estimate)
    valid = isfinite(value) && !GLLVModels._nll_failed(value)

    stats = GLLVModels._grouped_chol_stats()
    total_chol = stats.fresh + stats.reused + stats.fallback
    # `total_chol = 2*sum(iterations) + ok_calls` (the documented bench/
    # profile_grouped_glmm.jl relationship), so sum(iterations) = (total_chol
    # - ok_calls) / 2 when nearly all calls are :ok (ok_calls ~= stats.calls).
    # The FIRST version of this line forgot the /2 and over-reported by ~2x.
    inner_iters_sum = max(0, fld(total_chol - stats.calls, 2))

    return (; loglik = valid ? -value : -Inf, valid,
        obj_calls = obj_calls[], inner_laplace_calls = stats.calls,
        inner_iters_sum = inner_iters_sum, chol_stats = stats)
end

function gate_counts()
    println("GATE G9.4 — objective calls / inner Newton iterations, S9 default (nelder_mead=false) vs the S8-measured baseline")
    println("  Reported, not pinned (D-273: the old pin stays on the old path).")
    baseline = [("glmm_200x5", 84, 407), ("glmm_5000x3_g500", 284, 1437)]
    fixtures = [("glmm_200x5", _s9_load_small_fixture()...), ("glmm_5000x3_g500", _s9_make_large_fixture()...)]
    ok = true
    for ((name, Y, group), (_, base_obj, base_iters)) in zip(fixtures, baseline)
        m = _s9_counted_fit(Y, group)
        @printf("  %-18s objective_calls=%d (S8 baseline %d)  inner_newton_iters_sum≈%d (S8 baseline %d)  inner_laplace_calls=%d  chol_stats=%s  loglik=%.6f valid=%s\n",
            name, m.obj_calls, base_obj, m.inner_iters_sum, base_iters, m.inner_laplace_calls, m.chol_stats, m.loglik, m.valid)
        pass = m.valid && m.obj_calls < base_obj && m.inner_iters_sum < base_iters
        pass || @printf("  FAIL %s: need valid=true, obj_calls < %d (got %d), inner_iters_sum < %d (got %d)\n",
            name, base_obj, m.obj_calls, base_iters, m.inner_iters_sum)
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9.4 PASS" : "GATE G9.4 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# G9.5 (--gate coverage) — the four coverage holes S8 left: Binomial;
# per-trait `dispersion=:trait` for Beta and for NB2; `GroupingTerm(mode=:dep)`.
# Each must pass BOTH fd_agreement (per-coordinate, rtol RTOL_FD) and identity
# (rtol RTOL_IDENTITY).
# ---------------------------------------------------------------------------
function _fixture_binomial()
    rng = Xoshiro(20260926)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    beta = [0.2, -0.3]
    z = 0.6 .* randn(rng, G)
    ntrials = 8.0
    N = fill(ntrials, p, n)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = 1 / (1 + exp(-(beta[t] + z[unit[s]])))
        Y[t, s] = rand(rng, GLLVModels.Binomial(Int(ntrials), mu))
    end
    call = (; Y = Y, family = GLLVModels.Binomial(), N = N,
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit, N = N)
    return ("binomial", st, "Binomial, coverage hole S8 left untested (G9.5)", call)
end

function _fixture_beta_trait()
    rng = Xoshiro(20260927)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    phi = [7.0, 11.0]
    beta = [0.25, -0.2]
    z = 0.6 .* randn(rng, G)
    Y = Matrix{Float64}(undef, p, n)
    for s in 1:n, t in 1:p
        mu = 1 / (1 + exp(-(beta[t] + z[unit[s]])))
        Y[t, s] = clamp(rand(rng, GLLVModels.Beta(mu * phi[t], (1 - mu) * phi[t])), 1e-4, 1 - 1e-4)
    end
    call = (; Y = Y, family = GLLVModels.Beta(8.0, 1.0),
        terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = true)],
        unit = unit, dispersion = :trait)
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit, dispersion = :trait)
    return ("beta_trait", st, "Beta, PER-TRAIT log_phi (dispersion=:trait), coverage hole S8 left (G9.5)", call)
end

function _fixture_nb2_trait()
    rng = Xoshiro(20260928)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    r = [4.0, 6.0]
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
    st = _grouped_internals(Y; family = call.family, terms = call.terms, unit = unit, dispersion = :trait)
    return ("nb2_trait", st, "NegativeBinomial, PER-TRAIT log_r (dispersion=:trait), coverage hole S8 left (G9.5)", call)
end

function _fixture_dep()
    rng = Xoshiro(20260929)
    p, n, G = 2, 60, 12
    unit = repeat(1:G; inner = n ÷ G)
    beta = [0.3, -0.1]
    Ltrue = [0.6 0.0; 0.3 0.5]
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
    return ("dep_term", st, "GroupingTerm(mode=:dep), full trait covariance, coverage hole S8 left (G9.5)", call)
end

_coverage_fixtures() = [_fixture_binomial(), _fixture_beta_trait(), _fixture_nb2_trait(), _fixture_dep()]

function gate_coverage()
    println("GATE G9.5 — coverage holes S8 left: Binomial; Beta/NB2 dispersion=:trait; GroupingTerm(mode=:dep)")
    println("  Each fixture must pass BOTH fd_agreement (per-coordinate, rtol $(RTOL_FD)) AND identity (rtol $(RTOL_IDENTITY)).")
    ok = true
    for (name, st, note, call) in _coverage_fixtures()
        r = _compare_fixture(name, st, note)
        fd_pass = r.used == NTHETA && r.worst_rel <= RTOL_FD
        fd_pass || @printf("  FAIL %s (fd_agreement): used=%d/%d worst_rel=%.3e\n", name, r.used, NTHETA, r.worst_rel)

        fit_fd = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
            unit = call.unit, cluster = _s9_call_cluster(call), N = _s9_call_N(call),
            dispersion = call.dispersion, inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = false)
        fit_an = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
            unit = call.unit, cluster = _s9_call_cluster(call), N = _s9_call_N(call),
            dispersion = call.dispersion, inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL,
            analytic_gradient = true)
        dll = abs(fit_an.loglik - fit_fd.loglik) / max(abs(fit_fd.loglik), 1.0)
        dbeta = maximum(abs.(fit_an.beta .- fit_fd.beta) ./ max.(abs.(fit_fd.beta), 1.0))
        id_pass = dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY && fit_fd.converged == fit_an.converged

        @printf("  %-10s %-70s fd: used=%d worst_rel=%.3e | identity: loglik_rel=%.3e beta_rel=%.3e converged fd=%s an=%s\n",
            name, note, r.used, r.worst_rel, dll, dbeta, fit_fd.converged, fit_an.converged)
        pass = fd_pass && id_pass
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9.5 PASS" : "GATE G9.5 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# G9.6 (--gate mixed) — THE MIXED PATH, a hard gate. A fixture that forces the
# analytic gradient to fail at a SUBSET of theta during a REAL optimisation
# still lands on the all-FD answer at rtol 1e-8. This is the class that
# produced the S8 compaction bug (GB.4's reasoning-without-exercising).
# ---------------------------------------------------------------------------
function gate_mixed()
    println("GATE G9.6 — MIXED PATH: analytic gradient forced to `nothing` on a SUBSET of theta, still lands on the FD answer")
    _s9_install_mixed_gradient_monkeypatch!()
    ok = true
    for (name, _st, _note, call) in _fixtures()
        cluster = _s9_call_cluster(call)
        fit_ref = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
            unit = call.unit, cluster = cluster, dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = false)

        _S9_MIXED_CALL_INDEX[] = 0
        _S9_MIXED_FAIL_EVERY[] = 3
        # `Base.invokelatest`, not a plain call -- see the comment at
        # `gate_nm_fallback`'s equivalent call: `_s9_install_mixed_gradient_
        # monkeypatch!` ran from within this same function's dynamic extent,
        # so ordinary dispatch here is pinned to a world age before the
        # redefinition and would silently run the UNPATCHED method.
        fit_mixed = Base.invokelatest(GLLVModels.fit_grouped_nongaussian, call.Y; family = call.family,
            terms = call.terms, unit = call.unit, cluster = cluster, dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = true)
        _S9_MIXED_FAIL_EVERY[] = 0
        total_calls = _S9_MIXED_CALL_INDEX[]
        forced = total_calls ÷ 3

        dll = abs(fit_mixed.loglik - fit_ref.loglik) / max(abs(fit_ref.loglik), 1.0)
        dbeta = maximum(abs.(fit_mixed.beta .- fit_ref.beta) ./ max.(abs.(fit_ref.beta), 1.0))
        @printf("  %-16s analytic-gradient calls=%d forced-failures=%d  loglik ref=%.12e mixed=%.12e rel=%.3e  beta_rel=%.3e  converged ref=%s mixed=%s\n",
            name, total_calls, forced, fit_ref.loglik, fit_mixed.loglik, dll, dbeta, fit_ref.converged, fit_mixed.converged)
        pass = forced > 0 && dll <= RTOL_IDENTITY && dbeta <= RTOL_IDENTITY &&
            fit_ref.converged == fit_mixed.converged
        pass || @printf("  FAIL %s: forced=%d (need >0) dll=%.3e dbeta=%.3e (need <= %.1e)\n",
            name, forced, dll, dbeta, RTOL_IDENTITY)
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9.6 PASS" : "GATE G9.6 FAIL")
    return ok
end

# ---------------------------------------------------------------------------
# G9.7 (--gate final_gradient) — the final REPORTED gradient on the analytic
# path IS the analytic gradient (not `_grouped_fd_gradient`, which
# src/grouped_nongaussian_fit.jl:754 called unconditionally pre-S9), and its
# norm agrees with the FD gradient's norm at the converged point to rtol
# 1e-6, so the convergence verdict does not change.
# ---------------------------------------------------------------------------
function gate_final_gradient()
    println("GATE G9.7 — the final reported gradient on the analytic path IS the analytic gradient")
    ok = true
    for (name, st, _note, call) in _fixtures()
        fit = GLLVModels.fit_grouped_nongaussian(call.Y; family = call.family, terms = call.terms,
            unit = call.unit, cluster = _s9_call_cluster(call), dispersion = call.dispersion,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL, analytic_gradient = true)
        estimate = copy(fit.parameters)
        g_an = GLLVModels._grouped_analytic_gradient(estimate, st.data, st.trials, st.D, st.termvec,
            st.incidences, st.kind; dispersion_mode = st.mode,
            inner_maxiter = INNER_MAXITER, inner_tol = INNER_TOL)
        g_fd = GLLVModels._grouped_fd_gradient(st.objective_cold, estimate)
        an_norm_direct = maximum(abs, g_an)
        fd_norm = maximum(abs, g_fd)
        rel_source = abs(fit.gradient_norm - an_norm_direct) / max(an_norm_direct, 1.0)
        rel_fd = abs(fit.gradient_norm - fd_norm) / max(fd_norm, 1.0)
        @printf("  %-16s reported=%.10e  analytic_direct=%.10e (source rel=%.3e)  fd=%.10e (rel to reported=%.3e)\n",
            name, fit.gradient_norm, an_norm_direct, rel_source, fd_norm, rel_fd)
        pass = rel_source <= 1e-10 && rel_fd <= RTOL_GRADIENT_NORM
        pass || @printf("  FAIL %s: rel_source=%.3e (need <=1e-10) rel_fd=%.3e (need <=%.1e)\n",
            name, rel_source, rel_fd, RTOL_GRADIENT_NORM)
        println(pass ? "  PASS $name" : "  FAIL $name")
        ok &= pass
    end
    println(ok ? "GATE G9.7 PASS" : "GATE G9.7 FAIL")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    gate = length(ARGS) >= 2 && ARGS[1] == "--gate" ? ARGS[2] : "fd_agreement"
    ok = if gate == "identity"
        gate_identity()
    elseif gate == "hessian"
        gate_hessian()
    elseif gate == "nm_fallback"
        gate_nm_fallback()
    elseif gate == "counts"
        gate_counts()
    elseif gate == "coverage"
        gate_coverage()
    elseif gate == "mixed"
        gate_mixed()
    elseif gate == "final_gradient"
        gate_final_gradient()
    else
        gate_fd_agreement()
    end
    exit(ok ? 0 : 1)
else
    @testset "grouped analytic outer gradient vs finite differences" begin
        for (name, st, note, _call) in _fixtures()
            r = _compare_fixture(name, st, note; verbose = false)
            @test r.used == NTHETA
            @test r.worst_rel <= RTOL_FD
        end
    end
end
