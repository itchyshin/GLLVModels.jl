using GLLVModels, Optim, Random, Distributions, LinearAlgebra, Printf
include(joinpath(@__DIR__, "probe_common.jl"))
const G = GLLVModels
for seed in (3, 7)
    Y = sim(:gamma, seed; p = 8, n = 100); p, n = size(Y); K = 2
    fa = fit_gamma_gllvm(Y; K = K)
    rr = length(G.pack_lambda(fa.Λ))
    hs = G._default_hessian(Gamma(1.0, 1.0), LogLink())
    f(θ) = -G.gamma_marginal_loglik_laplace(float.(Y), G.unpack_lambda(θ[p+1:p+rr], p, K), θ[1:p], exp(θ[end]); hessian = hs)
    θ = vcat(fa.β, G.pack_lambda(fa.Λ), log(fa.α))
    ga = -G.gamma_laplace_grad(float.(Y), fa.Λ, fa.β, fa.α)
    @printf("seed %d  f(θ)=%.10f  |analytic g|∞=%.3e (argmax %d of %d)\n", seed, f(θ), maximum(abs, ga), argmax(abs.(ga)), length(θ))
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        g = [ (f(θ .+ h .* (1:length(θ) .== i)) - f(θ .- h .* (1:length(θ) .== i))) / (2h) for i in eachindex(θ)]
        @printf("   central FD h=%.0e  |g|∞=%.3e argmax=%d\n", h, maximum(abs, g), argmax(abs.(g)))
    end
    # objective noise: f on a tiny grid along the worst FD coordinate
    i = argmax(abs.(ga))
    vals = [f(θ .+ t .* (1:length(θ) .== i)) - f(θ) for t in (-1e-5, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 1e-5)]
    println("   f(θ+t e_i)-f(θ), t = ±1e-5,1e-6,1e-7: ", join((@sprintf("%.2e", v) for v in vals), " "))
    # descent test along -ga
    d = -ga ./ norm(ga)
    for t in (1e-4, 1e-3, 1e-2)
        @printf("   step t=%.0e along -analytic g: Δf=%.3e (first-order predicts %.3e)\n", t, f(θ .+ t .* d) - f(θ), -t * norm(ga))
    end
end
