using GLLVModels
using Random, LinearAlgebra
Random.seed!(20)
p3, K3, n3 = 4, 1, 400
Λ3 = reshape([0.8, 0.6, 0.4, -0.3], p3, K3)
y3 = Λ3 * randn(K3, n3) + 0.4 .* randn(p3, n3)
fit3 = fit_gaussian_gllvm(y3; K = K3)
corr_fn = θ -> begin
    spec = GLLVModels._derived_spec(fit3)
    GLLVModels._correlation_packed(θ, spec, 1, 2)
end
c_hat = corr_fn(fit3.pars.θ_packed)
println("fitted correlation(1,2) = ", round(c_hat, digits = 4))
for c_target in (0.95, 0.99, 0.999, 0.9999)
    ll_c, ok, θ_new, g_at_min = GLLVModels._derived_refit_with_fixed(
        fit3, corr_fn, c_target, y3, nothing, nothing)
    achieved = corr_fn(θ_new)
    println("target=", c_target, " success=", ok, " achieved=", round(achieved, digits=5),
            " |gap|=", round(abs(achieved - c_target), digits = 5))
end
