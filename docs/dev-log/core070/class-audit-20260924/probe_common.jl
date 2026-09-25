cl(η) = G._clamp_eta(η)
function refmode(q, K, zk)
    best = nothing
    for z0 in (zeros(K), zk isa AbstractVector && all(isfinite, zk) ? clamp.(zk, -50, 50) : zeros(K))
        r = try Optim.optimize(z -> -q(z), z0, Optim.BFGS(), Optim.Options(g_tol=1e-10, iterations=2000); autodiff=:forward) catch; nothing end
        r === nothing && continue
        (best === nothing || Optim.minimum(r) < Optim.minimum(best)) && (best = r)
    end
    return best === nothing ? nothing : Optim.minimizer(best)
end
gnorm(q, z) = try maximum(abs, ForwardDiff.gradient(q, z)) catch; Inf end
function probe(name, ntr, gen)
    nbad = 0; nfin = 0; nband = 0; worst = 0.0; ex = nothing
    for _ in 1:ntr
        K, zk_fn, q, v_fn = gen()
        zk = zk_fn()
        gk = gnorm(q, zk)
        gk > 1e-4 || continue
        zr = refmode(q, K, zk); zr === nothing && continue
        gr = gnorm(q, zr); gr < 1e-5 || continue
        nbad += 1
        v = try v_fn() catch; NaN end
        if isfinite(v)
            nfin += 1
            dq = q(zr) - q(zk)
            abs(v) < 1e11 && (nband += 1)
            if abs(v) < 1e11 && dq > worst
                worst = dq; ex = (K=K, grad_at_kernel_z=gk, logpost_gap=dq, site_value=v, zk=round.(zk; sigdigits=4), zref=round.(zr; sigdigits=4))
            end
        end
    end
    println(rpad(name, 44), " non-mode returns: $nbad/$ntr; finite site value: $nfin; finite & |v|<1e11 (escapes sentinel): $nband; worst logpost gap in-band: $(round(worst; sigdigits=4))")
    ex === nothing || println("    worst example: ", ex)
end
