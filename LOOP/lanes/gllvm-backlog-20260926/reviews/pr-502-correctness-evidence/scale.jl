using GLLVModels, Optim, Random, Distributions, LinearAlgebra, Printf
include(joinpath(@__DIR__, "probe_common.jl"))
for seed in 1:10
    Y = sim(:gamma, seed; p = 8, n = 100)
    N = length(Y)
    for (lab, f) in (("gamma-FD", Yc -> fit_gamma_gllvm(Yc; K = 2, gradient = :finite)),
                     ("gammagrp-FD", Yc -> GLLVModels.fit_gamma_gllvm_grouped(Yc; K = 2, group = repeat(1:2, 4))))
        empty!(LOG[]); fit = f(Y); r0 = LOG[][end]
        c = exp(-r0.nll / N)         # nll(cY) = nll(Y) + N log c  -> ~0
        run("$lab s$seed scale=1", () -> f(Y))
        run("$lab s$seed scale=$(round(c, sigdigits=3))", () -> f(c .* Y))
    end
end
