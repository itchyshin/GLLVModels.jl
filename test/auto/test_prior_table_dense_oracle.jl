# Independent dense Gaussian oracle for the GLLVModels Gaussian K=0 prior
# adapter. `oracle_*` below uses only the written density formulas; it does not
# call package objective, prior, gradient, Hessian, or fitting helpers.
using Test
using LinearAlgebra
using ForwardDiff
using SpecialFunctions: loggamma
using GLLVModels

function _oracle_dense_gaussian_gamma(theta, y; fixed_index, fixed_loc,
        fixed_scale, gamma_mean, gamma_shape)
    p, _ = size(y)
    beta = theta[1:p]
    eta = theta[p + 1]
    sigma = exp(eta)
    residual = y .- reshape(beta, p, 1)
    r = vec(residual)
    V = Matrix(Diagonal(fill(sigma^2, length(r))))
    loglik_nll = 0.5 * length(r) * log(2π) + 0.5 * logdet(Symmetric(V)) +
        0.5 * dot(r, V \ r)
    rate = gamma_shape / gamma_mean
    gamma_nll = -(gamma_shape - 1) * eta + rate * sigma -
        gamma_shape * log(rate) + loggamma(gamma_shape)
    delta = beta[fixed_index] - fixed_loc
    normal_nll = log(fixed_scale) + 0.5 * log(2π) +
        0.5 * (delta / fixed_scale)^2
    value = loglik_nll + gamma_nll + normal_nll

    # Closed-form derivatives in beta and eta=log(sigma), derived from the
    # dense quadratic form above. The Gamma density is with respect to sigma.
    gradient = zeros(length(theta))
    hessian = zeros(length(theta), length(theta))
    sumsquares = sum(abs2, residual)
    beta_data_gradient = zeros(p)
    for j in 1:p
        beta_data_gradient[j] = sum(beta[j] .- view(y, j, :)) / sigma^2
        gradient[j] = beta_data_gradient[j]
        hessian[j, j] = size(y, 2) / sigma^2
    end
    gradient[fixed_index] += delta / fixed_scale^2
    hessian[fixed_index, fixed_index] += 1 / fixed_scale^2
    gradient[p + 1] = length(r) - sumsquares / sigma^2 -
        (gamma_shape - 1) + rate * sigma
    for j in 1:p
        hessian[j, p + 1] = hessian[p + 1, j] = -2 * beta_data_gradient[j]
    end
    hessian[p + 1, p + 1] = 2 * sumsquares / sigma^2 + rate * sigma
    return value, gradient, hessian
end

@testset "Gaussian prior-table objective against independent dense oracle" begin
    y = [0.1 0.8 1.2 0.4 1.5 0.7;
         -1.1 -0.4 0.2 -0.8 -0.1 0.5]
    names = ["oak", "pine"]
    fixed_loc, fixed_scale = 0.25, 0.8
    gamma_mean, gamma_shape = 0.9, 2.4
    priors = (prior = ["normal($fixed_loc, $fixed_scale)",
                       "gamma($gamma_mean, $gamma_shape)"],
        class = ["fixef", "residual_sd"],
        coef = ["oak::(Intercept)", "(Intercept)"],
        dpar = ["mu", "sigma"], scale = ["link", "natural_sd"])
    resolved = GLLVModels._resolve_gaussian_intercept_prior_table(priors, 2;
        response_names = names)
    engine(theta) = GLLVModels._gllvm_gaussian_intercept_prior_objective(
        theta, y, resolved)

    # Five non-boundary points. Residual sums are nonzero so eta-beta Hessian
    # cross terms are exercised at every point.
    points = ([0.1, -0.8, log(0.7)], [1.1, 0.4, log(1.2)],
              [-0.4, -0.1, log(0.5)], [1.5, -1.2, log(0.9)],
              [0.6, 0.8, log(1.5)])
    max_objective_error = 0.0
    max_gradient_error = 0.0
    max_scaled_hessian_error = 0.0
    max_gradient_fd_error = 0.0
    max_hessian_fd_scaled_error = 0.0
    for theta in points
        expected, g_oracle, H_oracle = _oracle_dense_gaussian_gamma(theta, y;
            fixed_index = 1, fixed_loc, fixed_scale, gamma_mean, gamma_shape)
        actual = engine(theta)
        g_engine = ForwardDiff.gradient(engine, theta)
        H_engine = ForwardDiff.hessian(engine, theta)
        max_objective_error = max(max_objective_error,
            abs(actual - expected) / max(1.0, abs(expected)))
        max_gradient_error = max(max_gradient_error, maximum(abs.(g_engine .- g_oracle)))
        max_scaled_hessian_error = max(max_scaled_hessian_error,
            maximum(abs.(H_engine .- H_oracle) ./ max.(1.0, abs.(H_oracle))))

        # Central-difference steps are coordinate-scaled: h_g=1e-5 max(1,|x|)
        # for objective gradients; h_H=1e-4 max(1,|x|) for gradient Hessians.
        fd_gradient = similar(g_oracle)
        for j in eachindex(theta)
            h = 1e-5 * max(1.0, abs(theta[j]))
            plus, minus = copy(theta), copy(theta)
            plus[j] += h; minus[j] -= h
            fd_gradient[j] = (first(_oracle_dense_gaussian_gamma(plus, y;
                fixed_index = 1, fixed_loc, fixed_scale, gamma_mean, gamma_shape)) -
                first(_oracle_dense_gaussian_gamma(minus, y; fixed_index = 1,
                fixed_loc, fixed_scale, gamma_mean, gamma_shape))) / (2h)
        end
        max_gradient_fd_error = max(max_gradient_fd_error,
            maximum(abs.(fd_gradient .- g_oracle)))
        fd_hessian = similar(H_oracle)
        for j in eachindex(theta)
            h = 1e-4 * max(1.0, abs(theta[j]))
            plus, minus = copy(theta), copy(theta)
            plus[j] += h; minus[j] -= h
            gp = _oracle_dense_gaussian_gamma(plus, y; fixed_index = 1,
                fixed_loc, fixed_scale, gamma_mean, gamma_shape)[2]
            gm = _oracle_dense_gaussian_gamma(minus, y; fixed_index = 1,
                fixed_loc, fixed_scale, gamma_mean, gamma_shape)[2]
            fd_hessian[:, j] .= (gp .- gm) ./ (2h)
        end
        fd_hessian = (fd_hessian + fd_hessian') / 2
        max_hessian_fd_scaled_error = max(max_hessian_fd_scaled_error,
            maximum(abs.(fd_hessian .- H_oracle) ./ max.(1.0, abs.(H_oracle))))
    end
    @test max_objective_error <= 1e-8
    @test max_gradient_error <= 1e-6
    @test max_scaled_hessian_error <= 1e-5
    @test max_gradient_fd_error <= 1e-6
    @test max_hessian_fd_scaled_error <= 1e-5
    @test maximum(abs(_oracle_dense_gaussian_gamma(point, y; fixed_index = 1,
        fixed_loc, fixed_scale, gamma_mean, gamma_shape)[3][1, 3]) for point in points) > 0.1
    public_fit = fit_gllvm(y; family=GLLVModels.Normal(), K=0, priors,
        response_names=names, iterations=500)
    @test public_fit.objective ≈ public_fit.logLik + public_fit.log_prior
    @test public_fit.penalized
    @test public_fit.converged == (public_fit.status === :converged)
    @test public_fit.prior_metadata.fixed_effect_penalty +
        public_fit.prior_metadata.residual_sd_penalty ≈ public_fit.prior_metadata.total_penalty
    @test public_fit.prior_metadata.logLik == public_fit.logLik
    @info "Dense Gaussian prior oracle errors" max_objective_error max_gradient_error max_scaled_hessian_error max_gradient_fd_error max_hessian_fd_scaled_error
end
