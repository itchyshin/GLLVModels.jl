using GLLVModels, Test, ForwardDiff

# Issue #871: the NB2 AD kernels `_nb_logker` (src/laplace_grad.jl),
# `_nb_logpmf_full` (src/missing_predictor_poisson.jl) and the VA ELBO
# `_nb_logpmf_eta` (src/families/variational_negbin.jl) used
# loggamma(r+y) - loggamma(r) + r log r - (r+y) log(r+μ), which subtracts O(r log r)
# terms whose true difference is O(1). At large r (the Poisson limit) the rounding
# error swamps the answer. Reference: 512-bit exact log-pmf.
function nb2_hp_logpmf(y, mu, r)
    setprecision(512) do
        m = BigFloat(mu); rr = BigFloat(r)
        sum((log1p(BigFloat(k) / rr) for k in 0:(y - 1)); init = BigFloat(0)) +
            y * log(m) - GLLVModels.loggamma(BigFloat(y + 1)) - (rr + y) * log1p(m / rr)
    end
end

@testset "NB2 AD kernels are stable at large r (#871)" begin
    for r in [50.0, 1e8, 1e12, 1e14], mu in [0.5, 4.0, 20.0], y in [0, 1, 5, 20]
        exact = Float64(nb2_hp_logpmf(y, mu, r))
        @test isapprox(GLLVModels._nb_logpmf_full(mu, r, y), exact; atol = 1e-9, rtol = 1e-12)
        @test isapprox(GLLVModels._nb_logpmf_eta(log(mu), y, r), exact; atol = 1e-9, rtol = 1e-12)
        # `_nb_logker` drops the theta-free constant loggamma(y + 1).
        @test isapprox(GLLVModels._nb_logker(mu, r, y),
                       exact + Float64(GLLVModels.loggamma(big(y + 1.0))); atol = 1e-9, rtol = 1e-12)
        # d/d(log r) through ForwardDiff, against a 512-bit central difference.
        g = ForwardDiff.derivative(t -> GLLVModels._nb_logpmf_full(mu, exp(t), y), log(r))
        gk = ForwardDiff.derivative(t -> GLLVModels._nb_logker(mu, exp(t), y), log(r))
        gref = setprecision(512) do
            x = BigFloat(log(r)); h = big"1e-20"
            Float64((nb2_hp_logpmf(y, mu, exp(x + h)) - nb2_hp_logpmf(y, mu, exp(x - h))) / (2h))
        end
        @test isapprox(g, gref; atol = 1e-9, rtol = 1e-9)
        @test isapprox(gk, gref; atol = 1e-9, rtol = 1e-9)
    end
    # r -> infinity is exactly the Poisson pmf.
    for mu in [0.5, 4.0], y in [0, 3, 20]
        pois = y * log(mu) - mu - Float64(GLLVModels.loggamma(big(y + 1.0)))
        @test isapprox(GLLVModels._nb_logpmf_full(mu, 1e14, y), pois; atol = 1e-6)
    end
end
