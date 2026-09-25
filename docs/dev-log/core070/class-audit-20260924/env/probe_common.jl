
report(name, sc, nsite, bad, finbad, worst, vmin) = println(rpad(name, 34), " scale=", sc,
    "  nonconverged=", lpad(bad, 3), "/", nsite, "  finite-value-when-nonconverged=", lpad(finbad,3),
    "  worst |grad|=", round(worst, sigdigits=3), "  min site value=", round(vmin, sigdigits=3))

function run(name, sc; nsite=300, p=10, K=2, seed=11, gen, mode, resid, val)
    rng = MersenneTwister(seed)
    Λ = sc .* randn(rng, p, K); β = 0.5 .* randn(rng, p)
    bad = 0; finbad = 0; worst = 0.0; vmin = Inf
    for s in 1:nsite
        z0 = randn(rng, K); η = β .+ Λ*z0
        y = gen(rng, η)
        z = mode(y, Λ, β)
        r = resid(y, Λ, β, z)
        v = try val(y, Λ, β) catch; NaN end
        if !(r <= 1e-4)
            bad += 1; isfinite(v) && (finbad += 1); worst = max(worst, isfinite(r) ? r : Inf)
            isfinite(v) && (vmin = min(vmin, v))
        end
    end
    report(name, sc, nsite, bad, finbad, worst, vmin)
end

