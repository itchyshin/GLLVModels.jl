# test/test_sparse_phy_identities.jl — leaf-S7 identity gates G7.1-G7.4
# (G7.5 is a separate CLI run: the after-fix scaling bench + a full
# `Pkg.test()`, not exercised from inside this file).
#
# TDD contract: these gates pin the numerical IDENTITY between the sparse
# phylo-EM path (`_estep_sparse`, the sparse-routed monotonicity check, the
# `node_grad` gradient dedup, and the CHOLMOD symbolic-reuse workspace) and
# the pre-existing dense reference (`_estep_dense`, `gaussian_marginal_loglik`,
# a from-scratch two-call `takahashi_diag` reference) — origin/main 69a69b0a0
# routed EVERY per-iteration monotonicity check through the dense path and
# called `takahashi_diag` twice per `node_grad`; this arc changes ONLY which
# code path computes those numbers and how many times a factorisation runs,
# never the numbers themselves. A genuine numerical regression must fail one
# of these; a caching/refactor bug that happens not to move today's fixture
# would still be caught by the tight (1e-10 .. 1e-12) tolerances below.
#
# Runnable two ways:
#   - `include`d from test/runtests.jl as normal @testsets
#   - `julia --project=. test/test_sparse_phy_identities.jl --gate {loglik|gradient|estep|monotone}`
#     (prints "GATE G7.x PASS" / "GATE G7.x FAIL <reason>", exit 0/1)

using Test, GLLVModels, Random, LinearAlgebra, SparseArrays, Statistics

const _RESULTS = Bool[]
const _REASONS = String[]

function _check!(ok::Bool, msg::AbstractString)
    push!(_RESULTS, ok)
    ok || push!(_REASONS, msg)
    @test ok
    return ok
end

# ---------------------------------------------------------------------------
# Shared fixture — same shape as bench/profile_em_phylo_scaling.jl's
# `make_fixture` (K_B=1, n=200, σ_phy_scale=0.9, σ_eps=0.5), reused verbatim
# so the identity fixtures here match what the scaling bench profiles.
# ---------------------------------------------------------------------------
function make_em_fixture(p; K_B = 1, n = 200, σ_phy_scale = 0.9, σ_eps = 0.5, seed = 30)
    Random.seed!(seed)
    phy   = random_balanced_tree(p; branch_length = 0.1)
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
    Λ_B   = randn(p, K_B)
    for k in 1:K_B, i in 1:(k - 1)
        Λ_B[i, k] = 0.0
    end
    σ_phy = fill(σ_phy_scale, p)
    η_B   = randn(K_B, n)
    φ     = cholesky(Symmetric(Σ_phy)).L * randn(p)
    z     = σ_phy .* φ
    y     = Λ_B * η_B .+ reshape(z, p, 1) .+ σ_eps .* randn(p, n)
    return (; phy, Σ_phy, Λ_B, σ_phy, σ_eps, y, n)
end

# Warm start identical to `em_fit_phylo`'s own default (src/em_phylo.jl), so
# comparisons below are at a realistic (not degenerate) iterate.
function warmstart(fx)
    p = size(fx.y, 1)
    Λ0, σ0 = GLLVModels.ppca_init(Matrix{Float64}(fx.y), 1)
    Λ_B0   = Matrix{Float64}(Λ0)
    σ_eps0 = float(σ0)
    σ_phy0 = fill(0.1 * sqrt(mean(abs2, fx.y)), p)
    return Λ_B0, σ_eps0, σ_phy0
end

relerr(a, b) = maximum(abs.(a .- b)) / max(1.0, maximum(abs.(b)))

# ---------------------------------------------------------------------------
# G7.1 — the per-iteration monotonicity check: routed sparse value equals the
# dense reference within rtol 1e-12 at p=200 and p=1000.
# ---------------------------------------------------------------------------
function run_loglik_checks()
    empty!(_RESULTS); empty!(_REASONS)
    for p in (200, 1000)
        fx = make_em_fixture(p)
        Λ_B0, σ_eps0, σ_phy0 = warmstart(fx)
        ll_dense = GLLVModels.gaussian_marginal_loglik(fx.y, Λ_B0, σ_eps0;
            σ_phy = σ_phy0, Σ_phy = fx.Σ_phy)
        ll_sparse = gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B0, σ_eps0;
            σ_phy = σ_phy0, phy = fx.phy, σ²_phy = 1.0)
        rel = abs(ll_sparse - ll_dense) / abs(ll_dense)
        _check!(rel <= 1e-12,
            "p=$p: sparse-routed loglik check vs dense rel=$rel > 1e-12 " *
            "(dense=$ll_dense, sparse=$ll_sparse)")
    end
    return all(_RESULTS), copy(_REASONS)
end

# ---------------------------------------------------------------------------
# G7.2 — `node_grad`'s Takahashi-diagonal dedup (S7 item 3): values bitwise
# equal to a from-scratch reference that calls `takahashi_diag` twice
# independently (mirroring origin/main's un-deduped `node_dσ_phy` +
# `node_scalar_grads` each calling it on their own), and the call COUNT for
# the deduped `node_grad` is exactly 1.
# ---------------------------------------------------------------------------
function run_gradient_checks()
    empty!(_RESULTS); empty!(_REASONS)
    Random.seed!(700)
    p = 60; K_B = 2; n = p + 5
    phy = random_balanced_tree(p; branch_length = 0.1)
    Λ_B = 0.7 .* randn(p, K_B)
    σ_phy = abs.(randn(p)) .+ 0.3
    σ_eps = 0.6; σ²_phy = 0.8
    y = randn(p, n)
    st = GLLVModels.build_sparse_phy_state(y, Λ_B, σ_eps; σ_phy = σ_phy,
                                            phy = phy, σ²_phy = σ²_phy)

    # Reference: exactly what origin/main computed, two INDEPENDENT calls.
    cc      = GLLVModels._Cinv(st, st.m)
    Ainv_Yc = GLLVModels._AinvM(st, st.Y_c)
    Qeff_diag_a = GLLVModels.takahashi_diag(st.chol_Q_eff)
    Qeff_diag_b = GLLVModels.takahashi_diag(st.chol_Q_eff)
    dσ_phy_ref            = GLLVModels.node_dσ_phy(st, cc, Qeff_diag_a)
    dσ²phy_ref, dσ²eps_ref = GLLVModels.node_scalar_grads(st, cc, Ainv_Yc, Qeff_diag_b)
    dΛB_ref               = GLLVModels.node_dΛ_B(st, cc, Ainv_Yc)

    GLLVModels._node_grad_takahashi_calls_reset!()
    g = node_grad(st)
    calls = GLLVModels._node_grad_takahashi_calls()

    _check!(g.dσ_phy == dσ_phy_ref, "node_grad.dσ_phy not bitwise-equal to the two-call reference")
    _check!(g.dσ²_phy == dσ²phy_ref, "node_grad.dσ²_phy not bitwise-equal to the two-call reference")
    _check!(g.dσ²_eps == dσ²eps_ref, "node_grad.dσ²_eps not bitwise-equal to the two-call reference")
    _check!(g.dΛ_B == dΛB_ref, "node_grad.dΛ_B not bitwise-equal to the two-call reference")
    _check!(calls == 1,
        "node_grad called takahashi_diag(st.chol_Q_eff) $calls time(s); expected exactly 1 " *
        "(S7 item 3: computed once, passed through to both consumers)")

    return all(_RESULTS), copy(_REASONS)
end

# ---------------------------------------------------------------------------
# G7.3 — E-step moments (β, diag(Vφ), μ_φ, μ_z) within rtol 1e-10 of
# `_estep_dense` at p=200/1000, plus the EM trajectory (per-iteration loglik
# and final θ) within rtol 1e-10 of the dense-estep driver over 50 forced
# iterations on the p=200 fixture.
# ---------------------------------------------------------------------------
function run_estep_checks()
    empty!(_RESULTS); empty!(_REASONS)
    for p in (200, 1000)
        fx = make_em_fixture(p)
        Λ_B0, σ_eps0, σ_phy0 = warmstart(fx)
        ss_sparse = GLLVModels._estep_sparse(fx.y, Λ_B0, σ_eps0, σ_phy0, fx.phy; σ²_phy = 1.0)
        ss_dense  = GLLVModels._estep_dense(fx.y, Λ_B0, σ_eps0, σ_phy0, fx.Σ_phy)

        rel_beta = relerr(ss_sparse.β, ss_dense.β)
        _check!(rel_beta <= 1e-10, "p=$p: β rel diff $rel_beta > 1e-10")

        diagVφ_sparse = ss_sparse.Eφ2 .- ss_sparse.μ_φ .^ 2
        diagVφ_dense  = ss_dense.Eφ2 .- ss_dense.μ_φ .^ 2
        rel_dv = relerr(diagVφ_sparse, diagVφ_dense)
        _check!(rel_dv <= 1e-10, "p=$p: diag(Vφ) rel diff $rel_dv > 1e-10")

        rel_muphi = relerr(ss_sparse.μ_φ, ss_dense.μ_φ)
        _check!(rel_muphi <= 1e-10, "p=$p: μ_φ (conditional mean) rel diff $rel_muphi > 1e-10")

        rel_muz = relerr(ss_sparse.μ_z, ss_dense.μ_z)
        _check!(rel_muz <= 1e-10, "p=$p: μ_z (conditional mean, data scale) rel diff $rel_muz > 1e-10")
    end

    # EM trajectory parity: 50 FORCED iterations (tol=0.0 never declares
    # convergence early) on the p=200 fixture, sparse (default) vs dense
    # (force_dense_estep=true), same warm start.
    fx = make_em_fixture(200)
    Λ_B0, σ_eps0, σ_phy0 = warmstart(fx)
    common_kwargs = (; max_iter = 50, tol = 0.0, assert_monotone = true,
                     λ_init = Λ_B0, σ_eps_init = σ_eps0, σ_phy_init = σ_phy0)
    emf_sparse = em_fit_phylo(fx.y, 1, fx.Σ_phy; phy = fx.phy, common_kwargs...)
    emf_dense  = em_fit_phylo(fx.y, 1, fx.Σ_phy; phy = fx.phy, force_dense_estep = true,
                              common_kwargs...)

    _check!(emf_sparse.n_iter == 50 && emf_dense.n_iter == 50,
        "EM trajectory: expected both paths to run exactly 50 forced iterations " *
        "(sparse ran $(emf_sparse.n_iter), dense ran $(emf_dense.n_iter))")

    n_common = min(length(emf_sparse.loglik_trace), length(emf_dense.loglik_trace))
    rel_ll_traj = maximum(abs.(emf_sparse.loglik_trace[1:n_common] .-
                               emf_dense.loglik_trace[1:n_common]) ./
                          max.(1.0, abs.(emf_dense.loglik_trace[1:n_common])))
    _check!(rel_ll_traj <= 1e-10,
        "EM trajectory: per-iteration loglik rel diff $rel_ll_traj > 1e-10")

    rel_theta = max(relerr(emf_sparse.Λ_B, emf_dense.Λ_B),
                     abs(emf_sparse.σ_eps - emf_dense.σ_eps) / max(1.0, abs(emf_dense.σ_eps)),
                     relerr(emf_sparse.σ_phy, emf_dense.σ_phy))
    _check!(rel_theta <= 1e-10,
        "EM trajectory: final θ (Λ_B, σ_eps, σ_phy) rel diff $rel_theta > 1e-10 after 50 iterations")

    return all(_RESULTS), copy(_REASONS)
end

# ---------------------------------------------------------------------------
# G7.4 — the sparse monotonicity check flags the same iteration as the dense
# one on a fixture constructed to force a genuine decrease; SQUAREM (a
# separate, un-invoked-by-default function that exclusively calls
# `_estep_dense`/`gaussian_marginal_loglik` — grep-verified, untouched by
# S7 items 1-4) sanity-checked to still converge.
# ---------------------------------------------------------------------------
function run_monotone_checks()
    empty!(_RESULTS); empty!(_REASONS)
    p = 100
    fx = make_em_fixture(p)
    Λ_B0, σ_eps0, σ_phy0 = warmstart(fx)

    ll1_dense  = GLLVModels.gaussian_marginal_loglik(fx.y, Λ_B0, σ_eps0;
        σ_phy = σ_phy0, Σ_phy = fx.Σ_phy)
    ll1_sparse = gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B0, σ_eps0;
        σ_phy = σ_phy0, phy = fx.phy, σ²_phy = 1.0)

    # Deliberately worse second parameter set — forces a genuine, large
    # log-lik decrease so the monotonicity check actually fires.
    Λ_B_bad   = 0.01 .* Λ_B0
    σ_phy_bad = 5.0 .* σ_phy0
    ll2_dense  = GLLVModels.gaussian_marginal_loglik(fx.y, Λ_B_bad, σ_eps0;
        σ_phy = σ_phy_bad, Σ_phy = fx.Σ_phy)
    ll2_sparse = gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B_bad, σ_eps0;
        σ_phy = σ_phy_bad, phy = fx.phy, σ²_phy = 1.0)

    inc_dense  = ll2_dense  - ll1_dense
    inc_sparse = ll2_sparse - ll1_sparse
    flagged_dense  = inc_dense  < -1e-7
    flagged_sparse = inc_sparse < -1e-7

    _check!(flagged_dense,
        "fixture did not construct an actual decrease on the dense check (inc=$inc_dense)")
    _check!(flagged_sparse == flagged_dense,
        "sparse monotonicity check flags a DIFFERENT iteration than dense " *
        "(sparse flagged=$flagged_sparse, dense flagged=$flagged_dense)")
    rel_inc = abs(inc_sparse - inc_dense) / max(1.0, abs(inc_dense))
    _check!(rel_inc <= 1e-8,
        "sparse vs dense increment magnitude rel diff $rel_inc > 1e-8 " *
        "(inc_dense=$inc_dense, inc_sparse=$inc_sparse)")

    # SQUAREM: untouched by this arc (see docstring above); sanity-run only.
    sq = em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy; max_iter = 20, tol = 1e-8)
    _check!(isfinite(sq.logLik),
        "em_fit_phylo_squarem produced a non-finite logLik after this arc's changes " *
        "(it is untouched, so this would indicate an UNRELATED pre-existing issue)")

    return all(_RESULTS), copy(_REASONS)
end

const _GATES = Dict(
    "loglik"   => ("G7.1", run_loglik_checks),
    "gradient" => ("G7.2", run_gradient_checks),
    "estep"    => ("G7.3", run_estep_checks),
    "monotone" => ("G7.4", run_monotone_checks),
)

function _parse_gate_arg(argv, default = "loglik")
    parsed = default
    for (i, a) in enumerate(argv)
        if a == "--gate" && i < length(argv)
            parsed = argv[i + 1]
        end
    end
    return parsed
end

if abspath(PROGRAM_FILE) == @__FILE__
    gate_arg = _parse_gate_arg(ARGS)
    valid_gates = join(collect(keys(_GATES)), ", ")
    haskey(_GATES, gate_arg) ||
        error("unknown --gate '$gate_arg'; expected one of $valid_gates")
    gate_label, runner = _GATES[gate_arg]
    ok = false; reasons = String[]
    try
        @testset "sparse phylo identities — $gate_label ($gate_arg)" begin
            global ok, reasons = runner()
        end
    catch e
        @info "testset reported failures; continuing to the GATE line" exception = e
    end
    if ok
        println("GATE $(gate_label) PASS")
        exit(0)
    else
        println("GATE $(gate_label) FAIL ", join(reasons, "; "))
        exit(1)
    end
else
    # Included from test/runtests.jl: run all four as normal @testsets and let
    # genuine failures propagate.
    @testset "sparse phylo identities (S7) — loglik check (G7.1)" begin
        run_loglik_checks()
    end
    @testset "sparse phylo identities (S7) — node_grad dedup (G7.2)" begin
        run_gradient_checks()
    end
    @testset "sparse phylo identities (S7) — E-step (G7.3)" begin
        run_estep_checks()
    end
    @testset "sparse phylo identities (S7) — monotonicity check (G7.4)" begin
        run_monotone_checks()
    end
end
