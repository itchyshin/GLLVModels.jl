using Test, LinearAlgebra
include("core070_postcoef_fit_health.jl")

@testset "non-admitting Julia fit-health collector" begin
    f(x) = sum(abs2, x)
    grad(x) = 2 .* x
    hess(x) = Matrix(2.0I, 2, 2)
    x = [1.0, -2.0]
    collect_health(obj, theta, expected; prior="none", free=[1,2], gradient=grad, hessian=hess) =
        postcoef_julia_fit_health("no_fixed_effects", obj, theta, expected;
            penalty_state=prior, free_indices=free, gradient_eval=gradient, hessian_eval=hessian)
    h = collect_health(f, x, 5.0)
    @test h["parameters"] == x
    @test h["pd_hessian"] === true
    @test h["hessian"] == [[2.0, 0.0], [0.0, 2.0]]
    near(x) = [2.0 1e-15; 0.0 2.0]
    @test collect_health(f, x, 5.0; hessian=near)["error"] === nothing
    asymmetric(x) = [2.0 0.1; 0.0 2.0]
    @test collect_health(f, x, 5.0; hessian=asymmetric)["error"] !== nothing
    @test h["max_abs_gradient"] == 4.0
    @test h["objective_verified"] === true
    @test h["penalty_configuration_recorded"] === true
    @test h["error"] === nothing
    @test collect_health(f, x, 6.0)["objective_verified"] === false
    roundoff = collect_health(f, x, 5.0 + 4eps(5.0))
    @test roundoff["objective_verified"] === true
    @test roundoff["objective_difference"] > 0
    @test collect_health(f, x, 5.0 + 1e-7)["objective_verified"] === false
    @test collect_health(f, x, 5.0; prior=nothing)["penalty_configuration_recorded"] === false
    # penalty_state comes from the fit call keywords, not from an assumption.
    @test postcoef_julia_penalty_state((K = 1,)) == "none"
    @test postcoef_julia_penalty_state(NamedTuple()) == "none"
    @test postcoef_julia_penalty_state((K = 1, prior = 1.0)) == "active"
    @test postcoef_julia_penalty_state((K = 1, ridge_tau = 2.0)) == "active"
    @test postcoef_julia_penalty_state((K = 1, beta_prior_sd = 10.0)) == "active"
    @test postcoef_julia_penalty_state((K = 1, Penalty = :l2)) == "active"
    @test postcoef_julia_penalty_state((K = 1, optimizer = :LBFGS)) === nothing
    @test postcoef_julia_penalty_state((K = 1, family = :Poisson)) === nothing
    @test postcoef_julia_penalty_state(Dict(:K => 1)) === nothing
    @test collect_health(f, x, 5.0; prior=postcoef_julia_penalty_state((K = 1, prior = 1.0)))["penalty_state"] == "active"
    @test collect_health(f, x, 5.0; prior=postcoef_julia_penalty_state((K = 1, optimizer = 1)))["penalty_configuration_recorded"] === false
    indefinite(x) = x[1]^2 - x[2]^2
    bad = collect_health(indefinite, x, -3.0; gradient=x -> [2x[1], -2x[2]],
                         hessian=x -> [2.0 0.0; 0.0 -2.0])
    @test bad["pd_hessian"] === false
    missing = collect_health(f, [NaN, 1.0], 0.0)
    @test missing["max_abs_gradient"] === nothing
    @test missing["error"] !== nothing
    @test collect_health(f, x, 5.0; free=[1,1])["error"] !== nothing
    @test collect_health(f, x, 5.0; gradient=x -> [NaN, 0.0])["error"] !== nothing
end

@testset "certifier fields: coordinate, pinned indices, declared radii" begin
    f(x) = sum(abs2, x)
    args = (penalty_state="none", gradient_eval=x -> 2 .* x, hessian_eval=x -> Matrix(2.0I, 3, 3))
    x = [1.0, -2.0, 0.5]
    # Generic collector: no coordinate or radius unless the caller declares them.
    h = postcoef_julia_fit_health("c", f, x, 5.25; free_indices=[1, 3], args...)
    @test h["pinned_indices"] == [2]
    @test h["hessian_coordinate"] === nothing
    @test h["hessian_error_radius"] === nothing
    h = postcoef_julia_fit_health("c", f, x, 5.25; free_indices=[1, 2, 3], args...,
        hessian_coordinate="packed_free_index", derivative_method="forwarddiff_exact")
    @test h["pinned_indices"] == Int[]
    @test h["hessian_coordinate"] == "packed_free_index"
    @test h["hessian_error_radius"] == 1024.0 * 2.0^-52 * 2.0
    @test occursin("declared", h["hessian_error_radius_basis"])
    # NOT RUN here (no Julia in the review lane): radius status and the packed-gradient radius.
    @test h["radius_status"] == "declared_unproven"
    @test h["packed_gradient_error_radius"] == 1024.0 * 2.0^-52 * 5.25
    # A non-AD method records no radius (certifier then reports INCOMPLETE).
    h = postcoef_julia_fit_health("c", f, x, 5.25; free_indices=[1, 2, 3], args...,
        hessian_coordinate="packed_free_index", derivative_method="finite_difference")
    @test h["hessian_error_radius"] === nothing
    @test h["radius_status"] === nothing && h["packed_gradient_error_radius"] === nothing  # NOT RUN here
    # Adapter: pinned index set is the complement; coordinate is the packed one.
    fit = (pars=(θ_packed=[1.0, -2.0, 0.5],), logLik=-5.25)
    h = postcoef_julia_health_from_fit("c", fit, :data, nothing;
        penalty_state="none", nll_factory=(f, y, X, phy) -> (x -> sum(abs2, x)),
        pinned_indices=f -> [2], gradient_eval=(f, x) -> 2 .* x,
        hessian_eval=(f, x) -> Matrix(2.0I, 3, 3), derivative_method="forwarddiff_exact")
    @test h["free_indices"] == [1, 3] && h["pinned_indices"] == [2]
    @test h["hessian_coordinate"] == "packed_free_index"
    @test h["hessian_error_radius"] > 0
    # NOT RUN here: the adapter records the fit's own lambda_constraint map (nothing if no callback).
    @test h["lambda_constraint_map"] === nothing
    lcm = Dict{String,Any}("layout" => POSTCOEF_LAMBDA_MAP_LAYOUT, "p" => 2, "K" => 1, "q_free" => 0,
                           "entries" => Any[Any[nothing], Any[-2.0]])
    h = postcoef_julia_health_from_fit("c", fit, :data, nothing;
        penalty_state="none", nll_factory=(f, y, X, phy) -> (x -> sum(abs2, x)),
        pinned_indices=f -> [2], gradient_eval=(f, x) -> 2 .* x,
        hessian_eval=(f, x) -> Matrix(2.0I, 3, 3), derivative_method="forwarddiff_exact",
        lambda_map=f -> lcm)
    @test h["lambda_constraint_map"] === lcm
    # Map helper (NOT RUN here): layout, free (NaN or no constraint) as nothing, pins as numbers.
    mfit = (model=(p=3, K=1, K_W=0, has_diag=false, K_phy=0, has_phy_unique=false),
            pars=(lambda_constraint=[NaN; NaN; 0.5;;],))
    m = postcoef_lambda_constraint_map(mfit; q_free=1)
    @test m["layout"] == "beta_free,sigma_eps,lambda_b_packed" && m["p"] == 3 && m["K"] == 1 && m["q_free"] == 1
    @test m["entries"] == Any[Any[nothing], Any[nothing], Any[0.5]]
    @test postcoef_lambda_constraint_map((model=(p=3, K=1, K_W=0, has_diag=true, K_phy=0, has_phy_unique=false),
                                          pars=(lambda_constraint=nothing,)); q_free=0) === nothing
    @test postcoef_lambda_constraint_map((model=mfit.model, pars=(lambda_constraint=nothing,)); q_free=0)["entries"] ==
          Any[Any[nothing], Any[nothing], Any[nothing]]
    # Optimizer record carries the gradient radius only for the declared method.
    result = (point=[0.0, 0.0], value=0.0)
    oargs = (minimizer_eval=r -> r.point, minimum_eval=r -> r.value, gradient_eval=(g, z) -> 2 .* z .+ 1e-7)
    o = postcoef_julia_optimizer_health(result, z -> sum(abs2, z); oargs...)
    @test o["gradient_error_radius"] === nothing
    @test o["radius_status"] === nothing  # NOT RUN here
    o = postcoef_julia_optimizer_health(result, z -> sum(abs2, z); oargs..., derivative_method="forwarddiff_exact")
    @test o["radius_status"] == "declared_unproven"  # NOT RUN here
    @test o["gradient_error_radius"] == 1024.0 * 2.0^-52
    # Serialisation: shortest round-trip repr (the runner's to_json) is lossless.
    for v in (0.1, 1e-7, 1024.0 * 2.0^-52, nextfloat(1.0), 5e-324, floatmax())
        @test parse(Float64, repr(v)) === v
    end
end

@testset "fit adapter forwards design and free indices" begin
    fit = (pars=(θ_packed=[1.0, -2.0],), logLik=-5.0)
    design = reshape([3.0, 4.0], 1, 2, 1)
    seen = Ref{Any}(nothing)
    factory(f, y, X, phy) = (seen[] = (y, X, phy); x -> sum(abs2, x))
    h = postcoef_julia_health_from_fit("trait_intercepts_plus_covariate", fit, :data, design;
        penalty_state="none", nll_factory=factory, pinned_indices=f -> [2],
        gradient_eval=(f, x) -> 2 .* x,
        hessian_eval=(f, x) -> Matrix(2.0I, 2, 2))
    @test seen[] === (:data, design, nothing)
    @test h["free_indices"] == [1]
    @test h["max_abs_gradient"] == 2.0
    @test h["objective_verified"] === true
end

@testset "profile optimizer diagnostics" begin
    result = (point=[0.0, 0.0], value=0.0)
    args = (minimizer_eval=r -> r.point, minimum_eval=r -> r.value,
            gradient_eval=(f,x) -> 2 .* x)
    h = postcoef_julia_optimizer_health(result, x -> sum(abs2,x); args...)
    @test h["optimizer_parameters"] == [0.0, 0.0]
    @test h["optimizer_gradient"] == [0.0, 0.0]
    @test h["optimizer_profile_verified"] === true
    @test h["optimizer_max_abs_gradient"] == 0.0
    bad = postcoef_julia_optimizer_health((point=[0.0,0.0],value=1.0), x -> sum(abs2,x); args...)
    @test bad["optimizer_profile_verified"] === false
    bad = postcoef_julia_optimizer_health(result, x -> sum(abs2,x);
        minimizer_eval=r -> r.point, minimum_eval=r -> r.value,
        gradient_eval=(f,x) -> [NaN, 0.0])
    @test bad["optimizer_error"] !== nothing
end

# Fixed-point control for cancellation between equivalent objective routes.
# This does not exercise a fitted model or establish an error bound.
module ObjectiveCancellationControl
using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "packing.jl"))
include(joinpath(@__DIR__, "..", "src", "profile.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihood.jl"))
include(joinpath(@__DIR__, "core070_postcoef_fit_health.jl"))

@testset "profile/full cancellation remains diagnostic" begin
    p, n = 10, 40
    spec = (q=0, p=p, K_B=1, K_W=0, has_diag=false, K_phy=0, has_phy_unique=false)
    base = [sin(0.37*t*s) + 0.11*t - 0.04*s for t=1:p, s=1:n]
    theta = [sin(t*0.7)*0.3 for t=1:p]
    a0 = gaussian_profile_nll(theta, base; spec=spec, profile_beta=false)
    Y = exp(-a0/(p*n)) .* base
    recovered = profile_recover(theta, Y; spec=spec, profile_beta=false)
    fulltheta = vcat([log(recovered.σ_eps)], pack_lambda(recovered.Λ_B))
    profile_value = gaussian_profile_nll(theta, Y; spec=spec, profile_beta=false)
    full(v) = gaussian_nll_packed(v, Y; spec=spec)
    h = postcoef_julia_fit_health("synthetic", full, fulltheta, profile_value;
        penalty_state="none", free_indices=collect(eachindex(fulltheta)),
        gradient_eval=x -> zeros(length(x)),
        hessian_eval=x -> Matrix{Float64}(I, length(x), length(x)))
    @test h["error"] === nothing
    @test h["objective_difference"] == abs(profile_value - full(fulltheta))
    @test h["objective_verified"] == (h["objective_difference"] <= h["objective_roundoff_limit"])
    @test h["gradient"] !== nothing && h["hessian"] !== nothing
    # Objective-sensitivity probes: changed data or parameters must not reuse the
    # original profile value. These checks are diagnostic, not a parity bound.
    changed_Y = copy(Y)
    changed_Y[1, 1] += 0.1
    changed_theta = copy(theta)
    changed_theta[1] += 0.1
    changed_data_value = gaussian_profile_nll(theta, changed_Y; spec=spec, profile_beta=false)
    changed_parameter_value = gaussian_profile_nll(changed_theta, Y; spec=spec, profile_beta=false)
    @test changed_data_value != profile_value
    @test changed_parameter_value != profile_value
    recover(v) = begin
        rec = profile_recover(v, Y; spec=spec, profile_beta=false)
        (parameters=vcat(rec.β, [log(rec.σ_eps)], pack_lambda(rec.Λ_B)), loglik=rec.logLik)
    end
    original = recover(theta)
    h2 = postcoef_julia_reconstruction_health(theta, original.parameters, original.loglik; recover_eval=recover)
    @test h2["parameters_match"] && h2["loglik_matches"]
    changed = copy(original.parameters); changed[end] += 0.1
    @test !postcoef_julia_reconstruction_health(theta, changed, original.loglik; recover_eval=recover)["parameters_match"]
end

@testset "reconstruction on fixed-effect design shapes" begin
    p, n = 5, 8
    Y = [sin(0.37*t*s)+0.11*t-0.04*s for t=1:p, s=1:n]
    for q in (5, 6)
        X = zeros(p, n, q)
        for t=1:p, site=1:n
            X[t,site,t] = 1.0
            if q == 6; X[t,site,6] = (site-4.5)/4; end
        end
        spec = (q=q, p=p, K_B=1, K_W=0, has_diag=false, K_phy=0, has_phy_unique=false)
        theta = vcat([0.05*i for i=1:q], [0.4,-0.2,0.3,0.1,-0.1])
        recover(v) = begin
            rec = profile_recover(v, Y; spec=spec, X=X, profile_beta=false)
            (parameters=vcat(rec.β, [log(rec.σ_eps)], pack_lambda(rec.Λ_B)), loglik=rec.logLik)
        end
        actual = recover(theta)
        h = postcoef_julia_reconstruction_health(theta, actual.parameters, actual.loglik; recover_eval=recover)
        @test h["parameters_match"] && h["loglik_matches"] && h["error"] === nothing
        altered = copy(actual.parameters); altered[1] += 0.1
        @test !postcoef_julia_reconstruction_health(theta, altered, actual.loglik; recover_eval=recover)["parameters_match"]
    end
end
end

@testset "diagnostic fit reconstruction binding" begin
    recover(x) = (parameters=[x[1], 2*x[2]], loglik=-3.0)
    h = postcoef_julia_reconstruction_health([1.0, 2.0], [1.0, 4.0], -3.0; recover_eval=recover)
    @test h["parameters_match"] && h["loglik_matches"] && h["error"] === nothing
    @test h["optimizer_parameters"] == [1.0, 2.0]  # NOT RUN here: the point the reconstruction was recovered from
    @test !postcoef_julia_reconstruction_health([1.0, 2.0], [1.0, 5.0], -3.0; recover_eval=recover)["parameters_match"]
    @test !postcoef_julia_reconstruction_health([1.0, 2.0], [1.0, 4.0], -4.0; recover_eval=recover)["loglik_matches"]
    @test postcoef_julia_reconstruction_health([1.0, 2.0], [1.0, NaN], -3.0; recover_eval=recover)["error"] !== nothing
end
