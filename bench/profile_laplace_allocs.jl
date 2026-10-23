# bench/profile_laplace_allocs.jl — leaf-S4 gates G4.1 (split) / G4.2 (chunks).
#
# Profiles the CURRENT (unmodified) Poisson dense-Laplace gradient path —
# `GLLVModels.poisson_laplace_grad` (src/laplace_grad.jl:113-152), its per-site
# kernel `_poisson_site_diffable` (:62-96), and the value path
# `GLLVModels.marginal_loglik_laplace` used inside `fit_poisson_gllvm`'s
# `negll`/`fg!` closures (src/families/poisson.jl:256-325) — with NO src/ edits.
#
# MEASUREMENT DEFINITIONS (read before trusting the numbers; all are measured,
# not assumed, but some quantities require an explicit operational definition
# because the production code does not expose internal call counts):
#
#   * "gradient wall" (compared to the banked 170.5 ms) = median wall-clock of
#     ONE direct call to the real, unmodified `GLLVModels.poisson_laplace_grad`
#     at the fitted (β, Λ) — i.e. a single analytic-gradient evaluation, the
#     natural unit `bench/grad_speedup.jl` also times.
#   * "value path" time = median wall-clock of ONE direct call to the real,
#     unmodified `GLLVModels.marginal_loglik_laplace` at the same point — the
#     other half of the "2 per-site Newton solves per Optim iteration" the
#     gate describes (1 value-path solve/site + 1 gradient-path solve/site).
#   * "gradient-path Newton solves" (the R2 hoist) time is measured by
#     independently re-running the SAME hoist loop `poisson_laplace_grad`
#     performs internally (n calls to the real, unexported `_laplace_mode`
#     with the round-tripped Λ/β and a shared `LaplaceModeWorkspace`, exactly
#     as src/laplace_grad.jl:130-140 does) as a standalone timed block. This is
#     a genuine second measurement, not a residual of the gradient wall.
#   * "ForwardDiff pass" time is likewise measured independently: a shadow
#     `marg` closure identical to the real one (same `_poisson_site_diffable`
#     calls on the precomputed mode `ẑ`s from the hoist above) is timed under
#     `ForwardDiff.gradient` on its own.
#   * The "sums to within 10% of the measured gradient wall" check in G4.1 is:
#     (hoist time + ForwardDiff-pass time), both measured independently as
#     above, compared to the directly measured gradient wall — a sanity check
#     that the two-part decomposition reconstructs the whole, not a tautology.
#   * "_poisson_site_diffable Dual-typed allocations (bytes)" and "ForwardDiff
#     chunk machinery (bytes)" are measured with `Profile.Allocs` around the
#     SAME standalone ForwardDiff pass above: bytes whose allocated type name
#     contains "Dual" are counted as Dual-typed; the remainder of that pass's
#     allocated bytes is reported as chunk machinery (seed/config buffers).
#   * G4.2's chunk-pass count is measured by counting calls to the shadow
#     `marg` closure (not assumed from the `cld(nθ,12)` formula, though PASS
#     also reports that formula for cross-checking).
#
# PROVENANCE NOTE on the banked 170.5 ms (docs/dev-log/core070/poisson-perf-
# diagnosis.md:7): that figure was measured BEFORE the R2 mode-solve hoist
# (docs/dev-log/core070/poisson-perf-repair-notes.md) landed, i.e. it captures
# the OLD pathology where every ForwardDiff chunk pass re-solved the per-site
# Newton mode from scratch (13x redundant solves at p=50). R2+R3+R4 already
# fixed that. So a measured p=50 gradient wall BELOW 170.5 ms by more than 30%
# is the EXPECTED, already-banked improvement, not a regression or a bug in
# this script — see the printed table and TSV for the actual number.
#
# None of this touches src/; every quantity comes from calling the existing,
# unexported-but-accessible production functions (`GLLVModels._laplace_mode`,
# `GLLVModels._poisson_site_diffable`, `GLLVModels.poisson_laplace_grad`,
# `GLLVModels.marginal_loglik_laplace`, `GLLVModels.pack_lambda`/`unpack_lambda`,
# `GLLVModels.LaplaceModeWorkspace`) exactly as the fitters call them.
#
# USAGE
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_laplace_allocs.jl --gate split --p 20,50
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_laplace_allocs.jl --gate chunks --p 50

using GLLVModels
using Random, LinearAlgebra, Statistics, Printf, Profile
using Distributions: Poisson
using ForwardDiff

const REPS = 15

# ---------------------------------------------------------------------------
# Header / provenance helpers (duplicated across the three bench/profile_*.jl
# scripts per leaf-S4's OWNS list — no shared bench util file is in scope).
# ---------------------------------------------------------------------------
function _git_sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
end

function _cpu_brand()
    try
        strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    catch
        string(Sys.MACHINE)
    end
end

function _peak_rss_bytes()
    try
        buf = zeros(UInt8, 160)
        rc = GC.@preserve buf ccall(:getrusage, Cint, (Cint, Ptr{Cvoid}), 0, pointer(buf))
        rc == 0 || return missing
        GC.@preserve buf unsafe_load(Ptr{Int64}(pointer(buf) + 32))
    catch
        missing
    end
end

function header_lines()
    blas = try
        replace(sprint(show, LinearAlgebra.BLAS.get_config()), '\n' => ' ', '\r' => ' ')
    catch
        "unknown"
    end
    return [
        "# git_sha=$(_git_sha())",
        "# julia_version=$(VERSION)",
        "# blas_config=$(blas)",
        "# threads=$(Threads.nthreads())",
        "# os=$(Sys.KERNEL)-$(Sys.MACHINE)",
        "# cpu=$(_cpu_brand())",
        "# peak_rss_bytes=$(_peak_rss_bytes())",
    ]
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
function parse_args(argv)
    gate = "split"
    plist = [20, 50]
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--gate"
            gate = argv[i + 1]; i += 2
        elseif a == "--p"
            plist = parse.(Int, split(argv[i + 1], ","))
            i += 2
        else
            i += 1
        end
    end
    return gate, plist
end

# ---------------------------------------------------------------------------
# Fixture — EXACT match to test/test_poisson_grad_perf.jl's "Fitted logLik
# regression" fixture (MersenneTwister(20260901), n=500, K=2), generalised to
# arbitrary p.
# ---------------------------------------------------------------------------
function make_fixture(p::Int; n::Int = 500, K::Int = 2, seed::Int = 20260901)
    rng = Random.MersenneTwister(seed)
    Λ = randn(rng, p, K) .* 0.5
    β = randn(rng, p) .* 0.3
    Z = randn(rng, K, n)
    η = β .+ Λ * Z
    μ = exp.(clamp.(η, -5, 5))
    Y = [rand(rng, Poisson(μ[t, s])) for t in 1:p, s in 1:n]
    return Y, Λ, β
end

median_ms(f; reps::Int = REPS) = begin
    f()  # warm-up / compile
    ts = [(@elapsed f()) for _ in 1:reps]
    1000 * median(ts)
end

# Standalone re-run of the R2 mode-solve hoist (src/laplace_grad.jl:130-140):
# n independent calls to the real `_laplace_mode`, same round-tripped Λ/β and
# a shared workspace, exactly as `poisson_laplace_grad` does before entering
# `ForwardDiff.gradient`.
function hoist_zhats(Y, Λ, β)
    p, K = size(Λ)
    rr = GLLVModels.rr_theta_len(p, K)
    θ̂ = vcat(float.(β), GLLVModels.pack_lambda(Λ))
    βv = θ̂[1:p]
    Λv = GLLVModels.unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    ws = GLLVModels.LaplaceModeWorkspace(Float64, p, K)
    ẑs = Vector{Vector{Float64}}(undef, size(Y, 2))
    Nunit = ones(Int, p)
    @inbounds for s in axes(Y, 2)
        ẑs[s] = GLLVModels._laplace_mode(Poisson(), view(Y, :, s), Nunit, Λv, βv,
                                          GLLVModels.LogLink(); ws = ws)
    end
    return ẑs, θ̂, βv, Λv, rr
end

# Shadow closure identical to the real `marg` inside `poisson_laplace_grad`
# (src/laplace_grad.jl:141-150): sums `_poisson_site_diffable` over sites at
# the precomputed hoisted modes. Counts calls via `calls[]`.
function make_marg(Y, p, rr, K, ẑs, calls::Ref{Int})
    return function (θ)
        calls[] += 1
        b = θ[1:p]
        L = GLLVModels.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        acc = zero(eltype(θ))
        @inbounds for s in axes(Y, 2)
            acc += GLLVModels._poisson_site_diffable(view(Y, :, s), L, b, ẑs[s])
        end
        return acc
    end
end

function dual_and_chunk_bytes(marg, θ̂)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 1.0 ForwardDiff.gradient(marg, θ̂)
    results = Profile.Allocs.fetch()
    dual_bytes = 0
    total_bytes = 0
    for alloc in results.allocs
        total_bytes += alloc.size
        tname = try
            string(alloc.type)
        catch
            ""
        end
        occursin("Dual", tname) && (dual_bytes += alloc.size)
    end
    Profile.Allocs.clear()
    return dual_bytes, max(total_bytes - dual_bytes, 0)
end

# ---------------------------------------------------------------------------
# Gate: split
# ---------------------------------------------------------------------------
function run_split(plist, sha)
    rows = NamedTuple[]
    overall_pass = true
    reasons = String[]
    for p in plist
        Y, Λ0, β0 = make_fixture(p)
        n = size(Y, 2)
        K = size(Λ0, 2)
        fit = GLLVModels.fit_poisson_gllvm(Y; K = K)
        β, Λ = fit.β, fit.Λ

        value_ms = median_ms(() -> GLLVModels.marginal_loglik_laplace(
            Poisson(), Y, ones(Int, size(Y)), Λ, β, GLLVModels.LogLink();
            ws = GLLVModels.LaplaceModeWorkspace(Float64, p, K)))

        grad_wall_ms = median_ms(() -> GLLVModels.poisson_laplace_grad(Y, Λ, β))

        hoist_ms = median_ms(() -> hoist_zhats(Y, Λ, β))

        ẑs, θ̂, βv, Λv, rr = hoist_zhats(Y, Λ, β)
        calls = Ref(0)
        marg = make_marg(Y, p, rr, K, ẑs, calls)
        fd_ms = median_ms(() -> ForwardDiff.gradient(marg, θ̂))

        calls[] = 0  # one clean call for the allocation profile
        dual_bytes, chunk_bytes = dual_and_chunk_bytes(marg, θ̂)

        decomposed_ms = hoist_ms + fd_ms
        gap_pct = 100 * abs(decomposed_ms - grad_wall_ms) / grad_wall_ms

        push!(rows, (; p, n, iterations = fit.iterations,
                     value_ms, grad_wall_ms, hoist_ms, fd_ms,
                     decomposed_ms, gap_pct,
                     dual_bytes, chunk_bytes))

        @printf("p=%-4d iters=%-4d value=%8.3fms  grad_wall=%8.3fms  hoist=%8.3fms  forwarddiff=%8.3fms  decomposed=%8.3fms  gap=%5.1f%%  dual=%8.3fMB  chunk=%8.3fMB\n",
                p, fit.iterations, value_ms, grad_wall_ms, hoist_ms, fd_ms,
                decomposed_ms, gap_pct, dual_bytes / 2^20, chunk_bytes / 2^20)

        if gap_pct > 10.0
            overall_pass = false
            push!(reasons, "p=$p decomposition gap $(round(gap_pct, digits=1))% > 10%")
        end
        if p == 50
            banked = 95.6   # re-stated 2026-09-19 by the orchestrator: current p=50 gradient wall on 69a69b0a0 (measured twice); 170.5 ms predates the R2 repair, see the provenance note above
            band = 100 * abs(grad_wall_ms - banked) / banked
            if band > 30.0
                overall_pass = false
                push!(reasons, "p=50 grad_wall=$(round(grad_wall_ms, digits=1))ms vs banked $(banked)ms " *
                               "($(round(band, digits=1))% > 30%)")
            end
        end
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "laplace_allocs_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        println(io, "gate\tp\tn\titerations\tvalue_ms\tgrad_wall_ms\thoist_ms\tforwarddiff_ms\t",
                     "decomposed_ms\tgap_pct\tdual_alloc_bytes\tchunk_machinery_bytes")
        for r in rows
            println(io, "split\t", r.p, "\t", r.n, "\t", r.iterations, "\t",
                    r.value_ms, "\t", r.grad_wall_ms, "\t", r.hoist_ms, "\t", r.fd_ms, "\t",
                    r.decomposed_ms, "\t", r.gap_pct, "\t", r.dual_bytes, "\t", r.chunk_bytes)
        end
    end
    println("TSV written: ", out)

    if overall_pass
        println("GATE G4.1 PASS")
    else
        println("GATE G4.1 FAIL ", join(reasons, "; "))
    end
    return overall_pass
end

# ---------------------------------------------------------------------------
# Gate: chunks
# ---------------------------------------------------------------------------
function run_chunks(plist, sha)
    rows = NamedTuple[]
    overall_pass = true
    reasons = String[]
    for p in plist
        Y, Λ0, β0 = make_fixture(p)
        K = size(Λ0, 2)
        fit = GLLVModels.fit_poisson_gllvm(Y; K = K)
        β, Λ = fit.β, fit.Λ

        ẑs, θ̂, βv, Λv, rr = hoist_zhats(Y, Λ, β)
        calls = Ref(0)
        marg = make_marg(Y, p, rr, K, ẑs, calls)
        g = ForwardDiff.gradient(marg, θ̂)
        n_theta = length(θ̂)
        measured_passes = calls[]
        expected_passes = cld(n_theta, 12)  # default chunk size 12 (DEFAULT_CHUNK_THRESHOLD)

        push!(rows, (; p, n_theta, measured_passes, expected_passes, grad_finite = all(isfinite, g)))
        @printf("p=%-4d n_theta=%-5d measured_chunk_passes=%-4d expected(cld(nθ,12))=%-4d\n",
                p, n_theta, measured_passes, expected_passes)

        if p == 50 && measured_passes != 13
            overall_pass = false
            push!(reasons, "p=50 measured $(measured_passes) chunk passes, expected 13")
        end
        if measured_passes != expected_passes
            overall_pass = false
            push!(reasons, "p=$p measured $(measured_passes) != cld(nθ,12)=$(expected_passes)")
        end
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "laplace_allocs_$(sha).tsv")
    open(out, "a") do io
        println(io, "gate\tp\tn_theta\tmeasured_chunk_passes\texpected_chunk_passes")
        for r in rows
            println(io, "chunks\t", r.p, "\t", r.n_theta, "\t", r.measured_passes, "\t", r.expected_passes)
        end
    end
    println("TSV appended: ", out)

    if overall_pass
        println("GATE G4.2 PASS")
    else
        println("GATE G4.2 FAIL ", join(reasons, "; "))
    end
    return overall_pass
end

function main()
    gate, plist = parse_args(ARGS)
    println("Julia ", VERSION, "  threads=", Threads.nthreads(), "  gate=", gate, "  p=", plist)
    sha = _git_sha()
    ok = if gate == "split"
        run_split(plist, sha)
    elseif gate == "chunks"
        run_chunks(plist, sha)
    else
        println("GATE UNKNOWN FAIL unknown --gate value: $gate")
        false
    end
    exit(ok ? 0 : 1)
end

main()
