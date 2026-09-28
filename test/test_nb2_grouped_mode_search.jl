using GLLVModels, Test, SHA, TOML, LinearAlgebra, Distributions

# `_nb_grouped_loglik_site` (the per-site Laplace kernel behind `fit_nb_gllvm_grouped`,
# the default NegativeBinomial route of `fit_gllvm`) ran undamped Fisher scoring and
# returned whatever z the loop held when it stopped, converged or not (part of #503,
# the NB2-grouped row). Where a count sits far above its mean, the NB2 Fisher weight
# μr/(r+μ) under-states the observed curvature μr(r+y)/(r+μ)² by the factor
# (r+y)/(r+μ); the Fisher step then overshoots into a 2-cycle and the site returns a
# finite value away from the mode. On the seed-1 fixture below this steered L-BFGS to
# poor optima that reported converged = true (K=3 loglik -19113.13, far below K=2's
# -18874.03, although a K=3 model nests K=2).

const _NB2_503_SITE = joinpath(@__DIR__, "fixtures", "nb2_grouped_mode_search_503.toml")
const _NB2_503_Y    = joinpath(@__DIR__, "fixtures", "nb2_grouped_seed1_Y.toml")
const _NB2_503_SMALL = joinpath(@__DIR__, "fixtures", "nb2_grouped_small_Y.toml")

function _nb2_503_Y(path = _NB2_503_Y)
    d = TOML.parsefile(path)
    v = Int64.(d["Y_row_major"])
    Y = permutedims(reshape(v, d["n"], d["p"]))
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Y)))) == d["Y_sha256"]
    return Y
end

function _nb2_503_site()
    d = TOML.parsefile(_NB2_503_SITE)
    y = Int64.(d["y"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == d["y_sha256"]
    p, K = d["p"], d["K"]
    Λ = reshape(Float64.(d["Lambda_column_major"]), p, K)
    return y, Λ, Float64.(d["beta"]), Float64(d["r"]), d["v_on_main"]
end

# The pre-fix per-site kernel, verbatim apart from also reporting whether its undamped
# Fisher loop converged. Used only where it DID converge, to check that the fix keeps
# those values.
function _nb2_503_old_site(fams, y, Λ, β; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λ)
    n = ones(Int, p)
    link = G.LogLink()
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        η  = G._clamp_eta.(β .+ Λ * z)
        μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
        me = G.mu_eta.(Ref(link), η)
        s  = G._glm_score.(fams, μ, n, me, y)
        W  = G._nb_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link))
        Δ  = G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    η  = G._clamp_eta.(β .+ Λ * z)
    μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    W  = G._nb_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link))
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Independent per-site Laplace value (shares no code with the kernel): closed-form
# NB2/log score and observed curvature, plain Newton from z = 0 to a tight tolerance,
# Distributions.jl log-density. The observed weight is positive for NB2/log, so the
# log-posterior is concave in z and Newton converges to the unique mode.
function _nb2_503_ref(y, Λ, β, r)
    p, K = size(Λ)
    z = zeros(K)
    for _ in 1:200
        μ = exp.(β .+ Λ * z)
        g = Λ' * ((y .- μ) .* r ./ (r .+ μ)) .- z
        W = μ .* r .* (r .+ y) ./ (r .+ μ) .^ 2
        Δ = (Λ' * (W .* Λ) + I) \ g
        z .+= Δ
        maximum(abs, Δ) < 1e-13 && break
    end
    μ = exp.(β .+ Λ * z)
    g = Λ' * ((y .- μ) .* r ./ (r .+ μ)) .- z
    W = μ .* r .* (r .+ y) ./ (r .+ μ) .^ 2
    ℓ = sum(logpdf(NegativeBinomial(r, r / (r + μ[t])), y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Symmetric(Λ' * (W .* Λ) + I)), norm(g)
end

@testset "NB2 grouped kernel: 2-cycle site returns the mode's value (part of #503)" begin
    y, Λ, β, r, v_on_main = _nb2_503_site()
    p = length(y)
    fams = [NegativeBinomial(r, 0.5) for _ in 1:p]
    v_ref, gref = _nb2_503_ref(y, Λ, β, r)
    @test gref < 1e-8
    # The fixture's premise: the old loop did not converge here, and its value was
    # not the mode's.
    v_old, conv_old = _nb2_503_old_site(fams, y, Λ, β)
    @test !conv_old
    # v_on_main was recorded on macOS; Linux CI reproduces it to the last bit or two
    # (-392.0160180480272 vs ...269), so compare to 1e-12, not bitwise.
    @test isapprox(v_old, v_on_main; rtol = 1e-12)
    @test abs(v_on_main - v_ref) > 1e-3
    v = GLLVModels._nb_grouped_loglik_site(fams, y, ones(Int, p), Λ, β,
                                           GLLVModels.LogLink())
    @test isfinite(v)
    @test isapprox(v, v_ref; atol = 1e-7)
end

# Sites the old loop converged at keep their value. Most (76-100% per cell here) are
# bit-identical; where a full Fisher step would have lowered the per-site log-posterior
# on the way, the halving rule takes a different path to the same mode, so the value
# moves by roundoff only (measured up to 1.7e-10 on this fixture). Sites the old loop did
# not converge at now match an independent reference, so none returns a false -Inf.
@testset "NB2 grouped kernel: sites the old loop converged at keep their value" begin
    Y = _nb2_503_Y()
    p, n = size(Y)
    for K in (1, 2, 3)
        Z = log.(Y .+ 0.5)
        β = vec(sum(Z; dims = 2)) ./ n
        F = svd(Z .- β)
        Λ = F.U[:, 1:K] .* (F.S[1:K]' ./ sqrt(n))
        for r in (0.5, 2.0, 10.0)
            fams = [NegativeBinomial(r, 0.5) for _ in 1:p]
            nconv = 0; maxdiff = 0.0; maxerr_bad = 0.0
            for s in 1:n
                y = Y[:, s]
                v_old, conv_old = _nb2_503_old_site(fams, y, Λ, β)
                v = GLLVModels._nb_grouped_loglik_site(fams, y, ones(Int, p), Λ, β,
                                                       GLLVModels.LogLink())
                if conv_old
                    nconv += 1
                    maxdiff = max(maxdiff, abs(v - v_old))
                else
                    v_ref, _ = _nb2_503_ref(y, Λ, β, r)
                    maxerr_bad = max(maxerr_bad, abs(v - v_ref))
                end
            end
            @test nconv > 0
            @test maxdiff < 1e-8
            @test maxerr_bad < 1e-6
        end
    end
end

@testset "NB2 grouped fit escapes a poor optimum (part of #503)" begin
    Y = _nb2_503_Y(_NB2_503_SMALL)
    fit = fit_nb_gllvm_grouped(Y; K = 1, group = collect(1:size(Y, 1)))
    # origin/main @ d55a8e3af: loglik -1635.0089, converged = true.
    @test fit.converged
    @test fit.loglik > -1400
end

# The seed-1 case the auto-d lane found (p = 20, n = 300): about 9 minutes, so opt-in.
if get(ENV, "GLLVM_SLOW_TESTS", "") == "1"
    @testset "NB2 grouped fit escapes the seed-1 poor optimum (part of #503, slow)" begin
        Y = _nb2_503_Y()
        p = size(Y, 1)
        fit3 = fit_nb_gllvm_grouped(Y; K = 3, group = collect(1:p))
        # origin/main @ d55a8e3af: loglik -19113.13, converged = true, a loading row of
        # norm 8.73 (latent SD 8.7 against a true 0.8 * sqrt(3) ≈ 1.4).
        @test fit3.converged
        @test fit3.loglik > -17500
        @test maximum(norm.(eachrow(fit3.Λ))) < 5
        # Nesting: a K=3 model contains every K=2 model, so its maximum cannot be lower.
        fit2 = fit_nb_gllvm_grouped(Y; K = 2, group = collect(1:p))
        @test fit3.loglik >= fit2.loglik - 1e-6
    end
end
