using GLLVModels, Test, SHA, TOML, LinearAlgebra, ForwardDiff

# The two-part families could report converged = true far from the maximum likelihood
# (#484). The per-site mode search `_twopart_mode` took full Fisher-scoring steps with no
# damping and stopped at `maxiter` without a signal, and `twopart_loglik_site` scored
# whatever latent score it held. The value was finite, so no failure sentinel fired.
# At the simulation truth of the ZIP seed-101 data, 9 of 80 sites ran out of iterations
# and the package objective read -10085.3 against -930.4 with converged mode searches.
# The search now halves any step that lowers the site log-posterior, falls back to a
# damped Newton search on the observed positive-part curvature when Fisher scoring does
# not converge, and returns -Inf for a site whose search still fails, so the fitters'
# failure sentinel fires.
# Data: the class-audit receipts; recipes in the fixture header.

const _TP484_FIXTURE = joinpath(@__DIR__, "fixtures", "twopart_mode_search_484.toml")

function _tp484_data(name)
    d = TOML.parsefile(_TP484_FIXTURE)
    ds = d["datasets"][name]
    Y = reshape(Int.(ds["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Int64.(Y))))) == ds["data_sha256"]
    return Y, ds, d["K"]
end

# ---------------------------------------------------------------------------------------
# Independent ZIP site evaluator. Its own density, its own mode search (damped Newton on
# the full ForwardDiff Hessian of the site log-posterior) and its own expected information
# (a brute-force sum over the support). It shares no code with the package kernel. The
# Laplace value it returns is the package's formula for ZIP, ℓ(ẑ) − ½ẑ'ẑ − ½logdet A,
# where A uses the expected information (ZIP has no observed-curvature override).
# ---------------------------------------------------------------------------------------
_tp484_lgt(x) = 1 / (1 + exp(-x))
_tp484_clamp(x) = clamp(x, -30.0, 30.0)
function _tp484_zip_logf(y, ηz, ηc)
    π = _tp484_lgt(ηz); μ = exp(ηc)
    return y == 0 ? log(π + (1 - π) * exp(-μ)) : log1p(-π) + y * ηc - μ - sum(log, 2:y; init = 0.0)
end
_tp484_q(y, Λ, βz, βc, z) =
    sum(_tp484_zip_logf(y[t], _tp484_clamp(βz[t]), _tp484_clamp(βc[t] + dot(Λ[t, :], z)))
        for t in eachindex(y)) - 0.5 * dot(z, z)

function _tp484_indep_mode(y, Λ, βz, βc)
    f = z -> _tp484_q(y, Λ, βz, βc, z)
    z = zeros(size(Λ, 2))
    for _ in 1:500
        g = ForwardDiff.gradient(f, z)
        maximum(abs, g) < 1e-11 && break
        C = cholesky(Symmetric(-ForwardDiff.hessian(f, z)); check = false)
        d = issuccess(C) ? C \ g : g
        step = 1.0
        while f(z .+ step .* d) < f(z) && step > 1e-12
            step /= 2
        end
        maximum(abs, step .* d) < 1e-14 && break
        z = z .+ step .* d
    end
    return z
end

function _tp484_zip_expected_info(ηz, ηc)
    μ = exp(ηc)
    acc = 0.0
    for y in 0:ceil(Int, μ + 25 * sqrt(μ) + 40)
        s = ForwardDiff.derivative(e -> _tp484_zip_logf(y, ηz, e), ηc)
        acc += exp(_tp484_zip_logf(y, ηz, ηc)) * s^2
    end
    return acc
end

function _tp484_indep_site(y, Λ, βz, βc)
    z = _tp484_indep_mode(y, Λ, βz, βc)
    W = [_tp484_zip_expected_info(_tp484_clamp(βz[t]), _tp484_clamp(βc[t] + dot(Λ[t, :], z)))
         for t in eachindex(y)]
    return _tp484_q(y, Λ, βz, βc, z) - 0.5 * logdet(Symmetric(Λ' * (W .* Λ) + I))
end

# ---------------------------------------------------------------------------------------
# The pre-#484 per-site kernel (undamped Fisher scoring, then the Laplace value), kept
# apart from also reporting whether its loop converged. Used only where it DID converge,
# to check that the fix leaves those values alone.
# ---------------------------------------------------------------------------------------
function _tp484_old_site(fam, y, Λc, βz, βc; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λc)
    Λz = zeros(p, K)
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        ηz = G._clamp_eta.(βz .+ Λz * z)
        ηc = G._clamp_eta.(βc .+ Λc * z)
        P = [G._tp_pieces(fam, y[t], ηz[t], ηc[t]) for t in 1:p]
        A = Λz' * (getindex.(P, 3) .* Λz)
        A .+= Λc' * (getindex.(P, 4) .* Λc)
        for d in 1:K
            A[d, d] += 1.0
        end
        g = Λz' * getindex.(P, 1)
        g .= g .+ Λc' * getindex.(P, 2) .- z
        Δ = G._safe_solve(Symmetric(A), g)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    ηz = G._clamp_eta.(βz .+ Λz * z)
    ηc = G._clamp_eta.(βc .+ Λc * z)
    P = [G._tp_pieces(fam, y[t], ηz[t], ηc[t]) for t in 1:p]
    Wo = [G._tp_observed_Wc(fam, y[t], ηc[t], P[t][4]) for t in 1:p]
    A = Λz' * (getindex.(P, 3) .* Λz)
    A .+= Λc' * (Wo .* Λc)
    for d in 1:K
        A[d, d] += 1.0
    end
    ℓ = 0.0
    for t in 1:p
        ℓ += P[t][5]
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Symmetric(A)), conv
end

@testset "Two-part mode search: damped, and fails loudly (#484)" begin
    G = GLLVModels

    @testset "each two-part family's score is the derivative of its own log-density" begin
        # The damped search accepts a step only if it raises the site log-posterior, so the
        # score that sets the step must be the gradient of that log-posterior. Before #484,
        # HurdleNB's positive-part score lacked the NB2 factor r/(r + mu) (off by up to a
        # factor 2.5 here), so its search converged to a point that is not the mode.
        fams = ((G.ZIPoisson(), (0, 1, 3, 9)), (G.ZINB(2.0), (0, 1, 5, 20)), (G.ZIB(10), (0, 1, 4, 10)),
                (G.HurdlePoisson(), (0, 1, 2, 7)), (G.HurdleNB(3.0), (0, 1, 2, 7, 30)),
                (G.DeltaGamma(2.0), (0.0, 0.3, 1.7)), (G.DeltaLogNormal(0.7), (0.0, 0.4, 2.2)),
                (G.BetaHurdle(5.0), (0.0, 0.1, 0.6)))
        for (fam, ys) in fams
            worst = 0.0
            for y in ys, ηc in (-1.5, 0.3, 1.5), ηz in (-1.0, 0.5)
                d = ForwardDiff.derivative(e -> G._tp_pieces(fam, y, ηz, e)[5], ηc)
                worst = max(worst, abs(G._tp_pieces(fam, y, ηz, ηc)[2] - d) / max(1, abs(d)))
            end
            @test worst < 1e-10
        end
    end

    @testset "ZIP seed-101 truth: every site reaches its mode and its true Laplace value" begin
        # Before: 9 of 80 sites stopped at maxiter = 100; the objective read -10085.3 and the
        # worst site -4871 against -21.25.
        Y, ds, K = _tp484_data("zip_s101")
        p, n = size(Y)
        βz = Float64.(ds["truth_beta_z"]); βc = Float64.(ds["truth_beta_c"])
        Λc = reshape(Float64.(ds["truth_Lambda_c_column_major"]), p, K)
        Λz = zeros(p, K)
        worst_grad = 0.0
        worst_gap = 0.0
        total = 0.0
        total_indep = 0.0
        for s in 1:n
            y = Y[:, s]
            z = G._twopart_mode(G.ZIPoisson(), y, Λz, Λc, βz, βc)
            worst_grad = max(worst_grad,
                             maximum(abs, ForwardDiff.gradient(zz -> _tp484_q(y, Λc, βz, βc, zz), z)))
            v = G.twopart_loglik_site(G.ZIPoisson(), y, Λz, Λc, βz, βc)
            vi = _tp484_indep_site(y, Λc, βz, βc)
            worst_gap = max(worst_gap, abs(v - vi))
            total += v; total_indep += vi
        end
        @test worst_grad < 1e-6        # every site's search stopped at a stationary point
        @test worst_gap < 1e-6         # and scored the independent converged value
        @test abs(G.zip_marginal_loglik_laplace(Y, Λc, βz, βc) - total_indep) < 1e-5
    end

    @testset "a search that cannot converge returns -Inf, never a finite value" begin
        Y, ds, K = _tp484_data("zip_s101")
        p, n = size(Y)
        βz = Float64.(ds["truth_beta_z"]); βc = Float64.(ds["truth_beta_c"])
        Λc = reshape(Float64.(ds["truth_Lambda_c_column_major"]), p, K)
        Λz = zeros(p, K)
        # One iteration per stage cannot reach tol = 1e-9 from z = 0 at these sites.
        for s in (24, 51, 74)
            @test G.twopart_loglik_site(G.ZIPoisson(), Y[:, s], Λz, Λc, βz, βc; maxiter = 1) == -Inf
        end
        @test G.zip_marginal_loglik_laplace(Y, Λc, βz, βc; maxiter = 1) == -Inf
        # ... so the fitter's sentinel fires and the fit does not claim convergence.
        fit = G.fit_zip_gllvm(Y; K = K, newton_maxiter = 1)
        @test !fit.converged
        @test fit.loglik == -Inf
    end

    @testset "stress sites: the step halving and the Newton guard are both needed" begin
        # Step halving in the Fisher stage. At a heavily zero-inflated stress point built
        # from the ZIP seed-101 truth (loadings x3, count intercepts - 1, zero-inflation
        # logits + 2), undamped Fisher scoring leaves sites 23 and 78 where the Newton
        # fallback cannot recover.
        Y, ds, K = _tp484_data("zip_s101")
        p = size(Y, 1)
        βz = Float64.(ds["truth_beta_z"]) .+ 2.0; βc = Float64.(ds["truth_beta_c"]) .- 1.0
        Λc = 3.0 .* reshape(Float64.(ds["truth_Lambda_c_column_major"]), p, K)
        worst = maximum(abs(G.twopart_loglik_site(G.ZIPoisson(), Y[:, s], zeros(p, K), Λc, βz, βc) -
                            _tp484_indep_site(Y[:, s], Λc, βz, βc)) for s in (23, 78))
        @test worst < 1e-6
        # The Newton guard. Two zero counts at a high Poisson mean: the observed
        # curvature is negative (the zero-inflated mixture bends the wrong way), so the
        # Newton matrix is indefinite on the way to the mode (z = -1.13) and its negative
        # weights must be dropped for the step to climb.
        y2 = [0, 0]; Λ2 = fill(2.5, 2, 1); βz2 = fill(-0.5, 2); βc2 = fill(2.0, 2)
        v2 = G.twopart_loglik_site(G.ZIPoisson(), y2, zeros(2, 1), Λ2, βz2, βc2)
        @test abs(v2 - _tp484_indep_site(y2, Λ2, βz2, βc2)) < 1e-6
    end

    @testset "where the old loop converged, the site value is unchanged (to 1e-8)" begin
        # Parameter points: the ZIP seed-101 truth and warm start, and the clean ZINB and
        # ZIB datasets at their warm starts. At every site where the pre-#484 undamped loop
        # converged, the new kernel must return the same value.
        cases = Any[]
        Y, ds, K = _tp484_data("zip_s101")
        p = size(Y, 1)
        push!(cases, (G.ZIPoisson(), Y, Float64.(ds["truth_beta_z"]), Float64.(ds["truth_beta_c"]),
                      reshape(Float64.(ds["truth_Lambda_c_column_major"]), p, K)))
        push!(cases, (G.ZIPoisson(), Y, G._zi_warmstart(Y, K)...))
        Y3, _, _ = _tp484_data("zinb_s103")
        push!(cases, (G.ZINB(10.0), Y3, G._zi_warmstart(Y3, K)...))
        Y4, ds4, _ = _tp484_data("zib_s104")
        push!(cases, (G.ZIB(ds4["N"]), Y4, G._zib_warmstart(Y4, ds4["N"], K)...))
        n_compared = 0
        worst = 0.0
        for (fam, Yc, βz, βc, Λc) in cases
            Λz = zeros(size(Λc))
            for s in axes(Yc, 2)
                old, conv = _tp484_old_site(fam, Yc[:, s], Λc, βz, βc)
                conv || continue
                new = G.twopart_loglik_site(fam, Yc[:, s], Λz, Λc, βz, βc)
                worst = max(worst, abs(new - old))
                n_compared += 1
            end
        end
        @test n_compared >= 240        # measured: 256 of the 320 site evaluations
        @test worst < 1e-8
    end

    @testset "public ZIP fit reaches the better optimum (base-s1 data)" begin
        # Before: -935.296 with converged = true. A restart reaches -920.603, and raising
        # the inner maxiter to 2000 alone moved the fit to -921.41.
        Y, _, K = _tp484_data("zip_base1")
        fit = G.fit_zip_gllvm(Y; K = K)
        @test fit.loglik >= -920.7
    end

    @testset "fits that were already fine do not move (to 1e-6)" begin
        # Pre-#484 values from main @ d9bc77412 (Julia 1.10.12).
        # ZINB: the old fit reported -781.2303888021, but at that point 2 of 80 site
        # searches had stopped at maxiter, together 3.5e-6 above their converged values.
        # The same point re-scored with converged modes reads -781.2303923103 (the
        # class-audit probe's own damped search gives -781.2303923161). The fit must land
        # there.
        Y3, _, K = _tp484_data("zinb_s103")
        f3 = G.fit_zinb_gllvm(Y3; K = K)
        @test abs(f3.loglik - (-781.2303923103)) < 1e-6
        @test f3.converged
        # ZIB: every old site search at the fitted point but one converged (that one to
        # within 2e-8), so the old value itself is the reference.
        Y4, ds4, _ = _tp484_data("zib_s104")
        f4 = G.fit_zib_gllvm(Y4; K = K, N = ds4["N"])
        @test abs(f4.loglik - (-820.1151365707786)) < 1e-6
        @test f4.converged
    end

    @testset "Newton stage: a near-converged site does not walk away when Λz != 0 (#500)" begin
        # PR #500 review, SHOULD-FIX 1: the small-step bypass at the top of the accept
        # test in `_twopart_mode_stage` (`norm(Δ) <= 1e-3 * (1 + norm(z))`) took every
        # step of that size unconditionally, without checking whether it raised the site
        # log-posterior q(z). For zero-inflated families with Λz != 0 (occurrence loadings
        # present) the Newton stage's step matrix omits the η^z/η^c cross-curvature and
        # mixes a Fisher W^z with an observed W^c, so a "small" step by that norm test can
        # still be a descent step. Reproducer found by random search (recorded verbatim,
        # not the reviewer's own untracked script): a ZINB(2) site with occurrence and
        # count loadings, y = [0, 13, 0, 0, 0, 1]. Fisher scoring nearly converges
        # (|grad q| = 6.1e-7); the pre-#500 bypass then raises |grad q| to 1.4e-3 and
        # fails at maxiter = 100, though the true Hessian at the Fisher iterate is
        # negative-definite (eigenvalues -11.67, -2.44), i.e. a healthy mode.
        y500 = [0, 13, 0, 0, 0, 1]
        Λz500 = [-0.08411049838001773 -1.2980988495589005
                 2.2575996449911377 1.9114606922694863
                 -0.4383545906976807 0.34147738096407787
                 -1.343429707551161 0.3948510212415177
                 1.5837541874068504 2.334839366832539
                 1.9465952693840505 1.647696097038088]
        Λc500 = [-0.9594119548138331 1.652194889645204
                 1.6188634145274499 -0.023811974272768602
                 1.7078897517336884 1.498032015079005
                 0.93769961857294 0.4597311511262226
                 2.761485044760618 -2.937346684327989
                 -2.023583688938316 -0.7485773875914757]
        βz500 = [-1.9808674062121012, -0.5805482177256127, -0.1957007600068396,
                 -0.17979167750042335, 0.6197668169969093, -0.04932546244202318]
        βc500 = [2.0166340534625276, 2.0178540308130906, 2.3121356912840962,
                 1.4421664943730117, 0.1963323668296605, 0.07421497633337726]
        fam500 = G.ZINB(2.0)

        z_fisher, ok_fisher = G._twopart_mode_stage(fam500, y500, Λz500, Λc500, βz500, βc500,
                                                     :fisher; maxiter = 100, tol = 1e-9)
        @test !ok_fisher    # Fisher scoring alone does not reach tol at this site
        g_fisher = ForwardDiff.gradient(
            zz -> G._twopart_logpost(fam500, y500, Λz500, Λc500, βz500, βc500, false, false, zz),
            z_fisher)
        @test maximum(abs, g_fisher) < 1e-5    # ... but it is already near the mode

        # The Newton fallback must converge at the default maxiter, not return -Inf.
        val500 = G.twopart_loglik_site(fam500, y500, Λz500, Λc500, βz500, βc500)
        @test isfinite(val500)

        # And it must land at a genuine stationary point of q, not merely stop early.
        z_newton, ok_newton = G._twopart_mode_stage(fam500, y500, Λz500, Λc500, βz500, βc500,
                                                     :newton; z0 = z_fisher, maxiter = 100, tol = 1e-9)
        @test ok_newton
        g_newton = ForwardDiff.gradient(
            zz -> G._twopart_logpost(fam500, y500, Λz500, Λc500, βz500, βc500, false, false, zz),
            z_newton)
        @test maximum(abs, g_newton) < 1e-6
        H_newton = ForwardDiff.hessian(
            zz -> G._twopart_logpost(fam500, y500, Λz500, Λc500, βz500, βc500, false, false, zz),
            z_newton)
        @test all(eigvals(Symmetric(H_newton)) .< 0)    # a healthy (negative-definite) mode
    end
end
