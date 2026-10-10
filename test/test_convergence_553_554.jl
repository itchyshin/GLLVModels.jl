# #553 / #554: the `converged` flag must reflect the fit.
#
# #553 (converged = true far below the optimum). On mvabund::spider, the NB2 grouped fit
# with K = 0 and per-species r stepped species 10's intercept to 35.3, past the ±30 η
# clamp. There the objective is exactly flat in that intercept, so the gradient test fired
# and the fit reported converged = true at −878.81, 27.46 units below the per-species NB2
# MLE (−851.345577; gllvmTMB `value ~ 0 + trait`, nbinom2, gives the same).
#
# #554 (converged = false at the optimum). Poisson K = 2 on spider ended at gllvmTMB's
# logLik (−845.685747234, |Δ| ≈ 1.4e-7) but L-BFGS stopped at its 500-iteration cap and the
# fit reported converged = false. The NB2 exemplar is the Wave 1 simulation, cell 6 rep 26
# (seed 20266026; n = 60, p = 10, K = 2, species-specific x1/x2 slopes, true φ = 2), where
# commit 824d22a reported converged = false at gllvmTMB's −1000.575867.
using Test, GLLVModels
const _G553 = GLLVModels

# mvabund::spider$abund (mvabund 4.2.8), transposed to species × sites (12 × 28).
const _SPIDER553 = [
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    0 0 2 0 0 0 0 0 0 1 2 0 1 0 0 0 0 0 0 0 0 16 15 20 9 6 11 14;
    0 0 2 1 2 6 12 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 7 5 0 18 4 1;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    0 1 1 1 1 0 1 55 0 0 0 0 1 3 6 6 2 5 12 13 16 0 2 0 1 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    12 15 18 29 135 27 89 2 1 0 0 1 53 15 0 2 0 0 1 0 0 0 0 0 6 0 0 0;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2;
    4 9 1 25 17 34 16 3 0 0 0 0 22 32 3 4 2 0 3 2 2 0 0 0 6 0 0 0]

# Wave 1 cell 6 rep 26 (data inline in #554): counts, species × sites (10 × 60) …
const _Y554 = [
    2 3 1 8 31 0 0 0 1 0 4 0 2 13 0 4 1 4 2 5 6 2 0 0 5 3 0 1 1 15 3 0 1 0 1 0 3 9 1 3 2 2 4 1 2 15 6 0 7 6 16 0 1 0 0 8 0 3 23 0;
    1 4 1 0 1 1 1 0 0 0 0 0 0 0 2 0 2 2 1 0 0 0 2 0 0 1 0 0 0 2 0 0 0 0 1 9 0 3 1 1 0 2 0 0 1 1 1 0 1 0 0 0 0 0 1 2 0 0 0 3;
    0 1 1 2 1 2 1 4 0 4 1 4 0 1 0 1 1 0 1 2 2 1 3 1 5 1 0 0 3 1 1 0 2 0 0 0 1 5 0 6 0 1 10 4 0 1 3 5 6 1 0 0 4 2 0 2 2 1 3 4;
    4 11 2 2 5 1 0 0 2 2 4 2 2 2 1 0 9 3 1 0 1 0 0 0 1 1 0 0 0 2 7 5 0 0 0 3 3 36 0 2 3 0 1 6 0 44 2 0 4 1 2 0 1 0 1 37 0 1 0 1;
    3 3 3 0 0 3 0 0 2 0 0 5 2 2 0 0 2 2 2 2 2 0 5 2 0 3 1 2 1 0 1 7 3 0 0 5 0 15 3 2 9 7 2 2 3 0 3 1 0 2 1 0 1 0 1 6 0 1 0 3;
    0 6 1 0 0 1 0 0 0 0 0 0 0 0 0 1 3 0 1 0 1 0 0 0 1 1 3 0 1 0 1 1 1 2 1 8 1 0 1 1 3 1 0 2 0 0 1 0 0 1 0 1 0 0 0 0 4 2 0 0;
    1 1 3 0 1 1 1 0 1 1 0 1 4 1 0 0 9 2 2 1 0 0 2 3 0 1 4 6 1 0 2 4 2 0 0 5 5 1 4 5 14 3 7 0 3 0 1 0 2 6 0 0 1 0 0 2 2 0 0 6;
    0 3 1 1 1 0 8 1 1 0 0 0 5 1 5 1 1 0 17 0 0 6 1 4 0 0 0 0 2 3 0 1 5 0 0 0 0 0 1 1 2 0 1 2 2 0 0 1 0 0 1 3 3 2 4 0 0 2 0 1;
    5 0 1 1 0 2 0 0 0 4 1 1 2 1 0 3 4 0 2 0 1 1 1 0 2 2 2 2 0 0 1 0 0 0 5 2 3 0 1 0 1 0 0 2 2 1 0 0 0 0 0 1 1 1 0 4 3 1 3 3;
    5 7 1 1 0 3 1 3 0 7 5 0 4 0 0 1 1 7 3 3 3 0 0 0 9 0 6 3 1 2 1 0 1 0 2 3 2 2 0 0 1 2 8 12 3 1 3 1 3 1 1 1 2 4 1 7 2 2 2 3]
# … and the site covariates x1, x2 (60 × 2).
const _X554 = [
    -0.804431774146874 1.74773477757254;
    -1.70967284218483 -0.853596294207403;
    1.71979014999382 -0.646483861016139;
    0.741501490747024 -0.554885548265385;
    -0.357448221165797 -1.80050019832684;
    0.717790866356482 0.3964315530261;
    -0.665617820720576 0.535312471484325;
    1.09194922375688 0.534757643280969;
    0.123653400802249 1.0373499536009;
    1.19727417457737 -0.435571801988553;
    -0.315187742217763 -0.525498665175265;
    -0.462448616044116 -1.0175120279353;
    0.093719117607448 1.213215232415;
    0.291432562246777 -1.64743368128602;
    0.128249986070377 0.576810808525427;
    0.504330184805889 0.629944730110835;
    -1.67186644138099 -0.047149732437746;
    0.260811119560426 0.945039639423762;
    -1.484584656393 -1.4011797902211;
    0.864037997319604 -0.97014350958358;
    0.333474325966349 0.362761460115476;
    0.927571674121574 0.015092518545557;
    0.579757244721017 0.781955479331213;
    0.440399967895209 1.15686035679282;
    -0.40498076902115 0.21355778269651;
    -0.27347257078259 -0.109200458313777;
    0.990928286662105 0.329056471607081;
    1.02632614943176 -1.70463735356272;
    0.515894756902566 0.82325361281754;
    -1.75553537195657 0.329363088841397;
    -1.69691965026402 -0.931445678835284;
    -1.21188321370083 0.14793562133043;
    -0.0276209160978175 0.472114295155682;
    0.653135125116518 2.48835443613204;
    1.71709512567689 0.295030476959914;
    -1.47386035665463 0.0234103112087531;
    -0.49774126658277 0.0754991489146789;
    -1.35723859606274 -1.66176598627622;
    -1.11226833842795 -0.00769217213332508;
    1.24956706241618 -1.04544451925345;
    -2.49550025493594 -0.110185921088303;
    0.429413392954978 -0.731759895740104;
    0.160633623160788 -2.03062679210329;
    -1.60344805651342 -1.08696122507187;
    -1.33567815420842 0.96296214945366;
    -0.847255653889263 -1.2955538369071;
    -0.85465577782431 -0.569860276498864;
    0.892322718585621 0.245784685251421;
    0.0524252793426501 -1.6748682867088;
    0.498852346354496 -0.0972318187909232;
    0.696019684043263 0.0615110230046228;
    1.02328087747278 2.24325416616051;
    -1.46869066731112 -0.629516914785751;
    0.362598289903979 1.2316725012557;
    1.46597452169218 0.957098050553885;
    -2.01762629548581 -0.939758536349814;
    0.787846823995251 2.51638845032261;
    0.0655876153449441 1.22838866257142;
    0.639133785549135 -1.34363666677025;
    -0.544526998940457 1.05127126588321]

@testset "#553 / #554 convergence flag" begin
    @testset "#553 NB2 grouped K = 0 does not stop on the η-clamp plateau" begin
        f = fit_nb_gllvm_grouped(_SPIDER553; K = 0, group = collect(1:12))
        @test f.converged
        @test isapprox(f.loglik, -851.345577; atol = 1e-4)   # per-species NB2 MLE / gllvmTMB
        @test maximum(f.β) < log(100)                         # Pardpull mean 20.79, not e^35
        @test isapprox(f.r_group[10], 0.1438; rtol = 0.01)    # Pardpull MLE size
        # The objective guard: an intercept above the clamp is the failure sentinel.
        @test _G553._nb_eta_above_clamp([1.0, 30.5], nothing)
        @test !_G553._nb_eta_above_clamp([1.0, 29.5], nothing)
        @test _G553._nb_eta_above_clamp([1.0, 29.5], [0.0 0.0; 0.0 1.0])
        @test !_G553._nb_eta_above_clamp([-40.0, 2.0], nothing)   # lower cap is allowed
    end

    @testset "#554 Poisson K = 2 at the optimum reports converged" begin
        f = fit_poisson_gllvm(Float64.(_SPIDER553); K = 2)
        @test f.converged
        @test isapprox(f.loglik, -845.685747234; atol = 1e-5)   # gllvmTMB poisson(), d = 2
    end

    @testset "#554 NB2 grouped-cov (cell 6 rep 26) at the optimum reports converged" begin
        p, n = size(_Y554)
        Xb = zeros(p, n, 2p)                       # species-specific slopes by block expansion
        for t in 1:p, i in 1:n
            Xb[t, i, t] = _X554[i, 1]; Xb[t, i, p + t] = _X554[i, 2]
        end
        f = fit_nb_gllvm_grouped_cov(_Y554; X = Xb, K = 2)
        @test f.converged
        @test isapprox(f.loglik, -1000.575867; atol = 1e-4)    # gllvmTMB nbinom2(), d = 2
    end

    @testset "_bfgs_continuation keeps converged and sentinel runs unchanged" begin
        rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        opts = _G553.Optim.Options(iterations = 3)
        run(alg, θ) = _G553.Optim.optimize(rosen, θ, alg, opts; autodiff = :finite)
        res = run(_G553.Optim.LBFGS(), [-1.2, 1.0])            # stops at the 3-iteration cap
        @test !_G553.Optim.converged(res)
        θ, f, c, it = _G553._bfgs_continuation(run, res)
        @test f <= _G553.Optim.minimum(res) + 1e-6
        @test it > _G553.Optim.iterations(res)
        sentinel(alg, θ) = _G553.Optim.optimize(_ -> 1e12, θ, alg, opts; autodiff = :finite)
        rs = sentinel(_G553.Optim.LBFGS(), [0.0, 0.0])
        @test _G553._bfgs_continuation(sentinel, rs)[2] == _G553.Optim.minimum(rs)
    end
end
