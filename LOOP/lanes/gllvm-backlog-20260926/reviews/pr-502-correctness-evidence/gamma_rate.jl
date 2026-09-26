using GLLVModels, Optim, Random, Distributions, LinearAlgebra, Printf
include(joinpath(@__DIR__, "probe_common.jl"))
for seed in 1:20
    Y = sim(:gamma, seed; p = 8, n = 100)
    run("gamma-analytic s$seed", () -> fit_gamma_gllvm(Y; K = 2))
    run("gamma-offset0(FD) s$seed", () -> fit_gamma_gllvm(Y; K = 2, offset = zeros(size(Y))))
end
