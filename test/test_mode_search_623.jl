using GLLVModels, Test, LinearAlgebra, Optim, SHA, TOML

# #623: `_laplace_mode` (src/families/laplace.jl) is meant to return each site's conditional
# mode z-hat of L(z) = sum_t _glm_logpdf(fam, mu_t, n_t, y_t) - |z|^2 / 2, with
# mu_t = linkinv(link, beta_t + (Lambda z)_t). For TweedieED and the shared-parameter
# StudentTFamily it returns points far from that mode at some sites, and the per-site
# Laplace objective is then discontinuous in the parameters.
#
# The literal reproducers live in test/fixtures/mode_search_623.toml (see its header for
# origin, date and the origin/main commit they were measured on). This file asserts only
# RELATIONS, never recorded numbers, so it is platform independent:
#   1. z-hat agrees (max abs < 1e-5) with a reference mode that does not call
#      `_laplace_mode` (K = 1: dense grid + golden-section refinement; K = 2: multistart
#      LBFGS with finite differences), or, where the joint has two peaks, z-hat is a
#      strict local maximum below the reference one (kept visible as @test_broken);
#   2. the central-difference gradient of L at z-hat is below 1e-4 at every site;
#   3. at the named bad site `laplace_loglik_site` is continuous under a 1e-6 shift of beta.

const _G623 = GLLVModels
const _FIX623 = TOML.parsefile(joinpath(@__DIR__, "fixtures", "mode_search_623.toml"))

function _case623(c)
    p, n, K = c["p"], c["n"], c["K"]
    Y = reshape(Float64.(c["Y"]), p, n)
    @assert bytes2hex(sha256(reinterpret(UInt8, vec(Y)))) == c["data_sha256"]
    Λ = reshape(Float64.(c["Lambda"]), p, K)
    β = Float64.(c["beta"])
    fam = c["family"] == "tweedie" ? _G623.TweedieED(c["phi"], c["power"]) :
          _G623.StudentTFamily(c["nu"], c["sigma"])
    link = c["link"] == "log" ? _G623.LogLink() : _G623.IdentityLink()
    return (; Y, N = ones(Int, p, n), Λ, β, fam, link, K, n, p, bad = c["bad_site"])
end

_lp623(d, y, nn, z) = _G623._laplace_mode_logpost(d.fam, y, nn, d.Λ, d.β, d.link, z)

function _grad623(d, y, nn, z; h = 1e-5)
    g = similar(z)
    for k in eachindex(z)
        e = zeros(length(z)); e[k] = h
        g[k] = (_lp623(d, y, nn, z .+ e) - _lp623(d, y, nn, z .- e)) / (2h)
    end
    return g
end

# Golden-section maximiser of f on [a, b].
function _golden623(f, a, b; tol = 1e-11)
    φ = (sqrt(5) - 1) / 2
    c = b - φ * (b - a); dd = a + φ * (b - a)
    fc = f(c); fd = f(dd)
    while b - a > tol
        if fc > fd
            b = dd; dd = c; fd = fc; c = b - φ * (b - a); fc = f(c)
        else
            a = c; c = dd; fc = fd; dd = a + φ * (b - a); fd = f(dd)
        end
    end
    return (a + b) / 2
end

function _refmode623(d, y, nn)
    if d.K == 1
        grid = -8.0:1e-3:8.0
        vals = [_lp623(d, y, nn, [z]) for z in grid]
        i = argmax(vals)
        lo = grid[max(i - 1, 1)]; hi = grid[min(i + 1, length(grid))]
        return [_golden623(z -> _lp623(d, y, nn, [z]), lo, hi)]
    end
    f(z) = -_lp623(d, y, nn, z)
    starts = [zeros(d.K)]
    for s in Iterators.product(ntuple(_ -> (-2.0, 2.0), d.K)...)
        push!(starts, collect(Float64, s))
    end
    best = zeros(d.K); bv = Inf
    for s in starts
        r = Optim.optimize(f, s, LBFGS(), Optim.Options(g_tol = 1e-10, iterations = 500);
                           autodiff = :finite)
        if Optim.minimum(r) < bv
            bv = Optim.minimum(r); best = Optim.minimizer(r)
        end
    end
    return best
end

@testset "#623 mode search stationarity (Tweedie, shared Student-t)" begin
    for name in ("tweedie_K2_a", "tweedie_K2_b", "studentt_K1_true", "studentt_K1_fitted")
        @testset "$name" begin
            d = _case623(_FIX623[name])
            # The Tweedie series log-density costs ~3 ms per evaluation, so for K = 2 the
            # multistart reference is computed on every 8th site, on the named site, and on the
            # first 3 sites whose gradient check fails; the gradient check runs on every site.
            stride = d.K == 1 ? 1 : 8
            ẑs = [zeros(d.K) for _ in 1:d.n]; gs = zeros(d.n)
            Threads.@threads for i in 1:d.n
                y = d.Y[:, i]; nn = d.N[:, i]
                ẑs[i] = _G623._laplace_mode(d.fam, y, nn, d.Λ, d.β, d.link)
                gs[i] = maximum(abs, _grad623(d, y, nn, ẑs[i]))
            end
            refset = sort(unique(vcat(1:stride:d.n, d.bad, first(findall(>=(1e-4), gs), 3))))
            dzs = zeros(d.n)
            Threads.@threads for i in refset
                dzs[i] = maximum(abs, ẑs[i] .- _refmode623(d, d.Y[:, i], d.N[:, i]))
            end
            maxdz = maximum(dzs); maxg = maximum(gs)
            nbad_z = count(>=(1e-5), dzs); nbad_g = count(>=(1e-4), gs)
            @info "#623 $name: sites=$(d.n) bad(|z-ref|>=1e-5, over $(length(refset)) ref sites)=$nbad_z bad(|grad|>=1e-4)=$nbad_g " *
                  "max|z-ref|=$maxdz max|grad|=$maxg"
            # A site whose z-hat disagrees with the reference must still be a strict local
            # maximum of L (negative-definite finite-difference Hessian). The Student-t joint
            # can have two peaks; a local Newton search, like TMB's, reaches the nearer one.
            # Measured on studentt_K1_true after the #623 fix: 2 of 120 sites (54, 103) sit
            # on a lower peak, 0.24 and 2.85 below the global one. That is a separate
            # limitation (global mode search), kept visible here as @test_broken.
            off = [i for i in refset if dzs[i] >= 1e-5]
            for i in off
                y = d.Y[:, i]; nn = d.N[:, i]; z = ẑs[i]; h = 1e-4
                H = [(_lp623(d, y, nn, z .+ h .* ((1:d.K) .== a) .+ h .* ((1:d.K) .== b)) -
                      _lp623(d, y, nn, z .+ h .* ((1:d.K) .== a) .- h .* ((1:d.K) .== b)) -
                      _lp623(d, y, nn, z .- h .* ((1:d.K) .== a) .+ h .* ((1:d.K) .== b)) +
                      _lp623(d, y, nn, z .- h .* ((1:d.K) .== a) .- h .* ((1:d.K) .== b))) / (4h^2)
                     for a in 1:d.K, b in 1:d.K]
                @test isposdef(Symmetric(-H))                  # a genuine local maximum
                @test _lp623(d, y, nn, _refmode623(d, y, nn)) > _lp623(d, y, nn, z)
            end
            if isempty(off)
                @test maxdz < 1e-5
            else
                @test_broken maxdz < 1e-5                      # lower-peak sites (see above)
            end
            @test maxg < 1e-4

            # Continuity of the per-site Laplace objective at the named site.
            yb = d.Y[:, d.bad]; nb = d.N[:, d.bad]
            v0 = _G623.laplace_loglik_site(d.fam, yb, nb, d.Λ, d.β, d.link)
            for j in 1:d.p
                βp = copy(d.β); βp[j] += 1e-6
                v1 = _G623.laplace_loglik_site(d.fam, yb, nb, d.Λ, βp, d.link)
                @test abs(v1 - v0) < 1e-3
            end
        end
    end
end
