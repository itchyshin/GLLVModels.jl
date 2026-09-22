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

if abspath(PROGRAM_FILE) == @__FILE__
    gate = length(ARGS) >= 2 && ARGS[1] == "--gate" ? ARGS[2] : "fd_agreement"
    # An UNRECOGNISED gate name must ERROR, never fall through. It used to run
    # `gate_fd_agreement()`, so any CHECK naming a gate this branch does not
    # implement printed "GATE GB.2 PASS" and exited 0. leaf-S9.md ships seven
    # CHECK lines on this branch and six of them name gates that live only on
    # the S9 lanes, so six acceptance gates would have passed vacuously against
    # a gate that tests something else entirely. A typo did the same.
    ok = if gate == "fd_agreement"
        gate_fd_agreement()
    elseif gate == "identity"
        gate_identity()
    else
        error("unknown --gate $(gate); gates implemented on this branch: fd_agreement, identity")
    end
    exit(ok ? 0 : 1)
else
    @testset "grouped analytic outer gradient vs finite differences" begin
        for (name, st, note, _call) in _fixtures()
            r = _compare_fixture(name, st, note; verbose = false)
            @test r.used == NTHETA
            @test r.worst_rel <= RTOL_FD
        end
        # The regression check for the compacted-unique-variance column bug
        # (7f175835e) was defined and documented and then called from nowhere,
        # so the one test written to catch a silently WRONG gradient had never
        # executed. It returns Bool and prints its own diagnostic.
        @test _s8_compacted_unique_column_check!()
    end
end
