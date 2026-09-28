# #504, zero-inflated migration (after Poisson #516, Gamma and Ordinal): the
# bootstrap `refit` closures in `src/confint_family.jl` (`_family_ci` for ZIPFit,
# ZIPCovFit, ZINBFit, ZINBCovFit and ZIBFit) returned a bare parameter vector,
# so `_bootstrap_refit_ok` (#508) could only check `isfinite`. A refit that ended
# on the fitter's failure sentinel (θ is the finite warm start) was counted as a
# good replicate. They now report `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict. Whether the ZINB size `r` needs such a flag (as
# NB2 `r` does) is left open and is not tested here.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_zi_verdict(kind::Symbol, p::Integer, n::Integer; seed::Integer,
                         Ntr::Integer = 8, r::Real = 3.0)
    rng = MersenneTwister(seed)
    βz = 0.3 .* randn(rng, p) .- 0.6
    βc = kind === :zib ? 0.3 .* randn(rng, p) : 0.3 .* randn(rng, p) .+ 1.2
    Λc = 0.4 .* randn(rng, p, 1)
    Y = zeros(Int, p, n)
    for s in 1:n
        ηc = βc .+ Λc * randn(rng, 1)
        for t in 1:p
            if rand(rng) < inv(1 + exp(-βz[t]))
                Y[t, s] = 0
            elseif kind === :zip
                Y[t, s] = rand(rng, Poisson(exp(ηc[t])))
            elseif kind === :zinb
                μ = exp(ηc[t])
                Y[t, s] = rand(rng, NegativeBinomial(r, r / (r + μ)))
            else
                Y[t, s] = rand(rng, Binomial(Ntr, inv(1 + exp(-ηc[t]))))
            end
        end
    end
    return Y
end

_zi504_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

const _ZI504_NTR = 8

function _zi504_routes(Yp, Yn, Yb)
    p, n = size(Yp)
    X = _zi504_X(p, n)
    f0 = GM.fit_zip_gllvm(Yp; K = 1)
    f1 = GM.fit_zip_gllvm_cov(Yp; X = X, K = 1)
    f2 = GM.fit_zinb_gllvm(Yn; K = 1)
    f3 = GM.fit_zinb_gllvm_cov(Yn; X = X, K = 1)
    f4 = GM.fit_zib_gllvm(Yb; K = 1, N = _ZI504_NTR)
    free1 = findall(!, f1.γ_fixed); free3 = findall(!, f3.γ_fixed)
    return (
        (:zip, f0, Yp, GM._family_ci(f0, Yp),
         Yb_ -> GM.fit_zip_gllvm(Yb_; K = 1),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc))),
        (:zip_cov, f1, Yp, GM._family_ci(f1, Yp; X = X),
         Yb_ -> GM.fit_zip_gllvm_cov(Yb_; X = X, K = 1, γ_fixed = f1.γ_fixed),
         fb -> vcat(fb.βz, fb.γz[free1], fb.βc, fb.γc[free1], GM.pack_lambda(fb.Λc))),
        (:zinb, f2, Yn, GM._family_ci(f2, Yn),
         Yb_ -> GM.fit_zinb_gllvm(Yb_; K = 1),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc), log(fb.r))),
        (:zinb_cov, f3, Yn, GM._family_ci(f3, Yn; X = X),
         Yb_ -> GM.fit_zinb_gllvm_cov(Yb_; X = X, K = 1, γ_fixed = f3.γ_fixed),
         fb -> vcat(fb.βz, fb.γz[free3], fb.βc, fb.γc[free3], GM.pack_lambda(fb.Λc), log(fb.r))),
        (:zib, f4, Yb, GM._family_ci(f4, Yb),
         Yb_ -> GM.fit_zib_gllvm(Yb_; K = 1, N = _ZI504_NTR),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc))),
    )
end

# A draw no fit on the route can evaluate, so each fitter ends on its failure
# sentinel (`converged = false`, `loglik = -Inf`, finite θ, no throw), with or
# without bounds checking. Data-driven, not RNG-driven. For ZIP and ZINB one
# cell holds a half count: the count log-density converts `y` with `Int(y)`, the
# `InexactError` is caught by the objective, and every evaluation returns the
# sentinel. For ZIB one cell exceeds the trial number (log-density -Inf). A
# negative count is NOT a failing draw: `y > 0` is false, so it is scored as a
# zero and the fit converges.
function _zi504_bad(route::Symbol, Y)
    if route === :zib
        B = copy(Y)
        B[1, 1] = _ZI504_NTR + 1
    else
        B = Float64.(Y)
        B[1, 1] = 0.5
    end
    return B
end

@testset "Zero-inflated bootstrap refit reports the fitter's own verdict (#504)" begin
    p, n = 4, 120
    Yp = _sim_zi_verdict(:zip, p, n; seed = 71)
    Yn = _sim_zi_verdict(:zinb, p, n; seed = 72)
    Yb = _sim_zi_verdict(:zib, p, n; seed = 73, Ntr = _ZI504_NTR)
    routes = _zi504_routes(Yp, Yn, Yb)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, Y, ad, direct, pack) in routes
        @test fit.converged   # sanity: a healthy fit
        Ydraw = ad.simulate(MersenneTwister(504))
        raw = ad.refit(Ydraw)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Ydraw)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test !haskey(raw, :upper_boundary)   # deliberately not flagged (see header)
    end

    @testset "a failed refit is rejected ($route)" for (route, fit, Y, ad, direct, pack) in routes
        Y_bad = _zi504_bad(route, Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, direct, pack) in routes
        Y_bad = _zi504_bad(route, Y)
        m = length(ad.θ)
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, direct, pack = routes[1]   # ZIP, no covariates: the fastest route
        m = length(ad.θ)
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                               Yb_ -> (r = ad.refit(Yb_); r === nothing ? nothing : r.θ))
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates.
        n_boot = 10
        new = GM._family_bootstrap(ad, collect(1:m), 0.95, n_boot, 11, false)
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 11, false)
        @test new.n_converged == n_boot && old.n_converged == n_boot
        @test all(isfinite, new.lower) && all(isfinite, new.upper)   # non-vacuous
        @test isequal(new.lower, old.lower) && isequal(new.upper, old.upper)
    end
end
