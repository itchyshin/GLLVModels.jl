using Test
using GLLVModels

# #758: `_lv_effect_profile` used to return `pd_hessian = true` as a literal,
# even when `crossing` left an endpoint as NaN. A constant objective never
# reaches the χ² cutoff, so both sides stay NaN without a data fit.
@testset "confint_lv_effects profile pd_hessian follows endpoint health (#758)" begin
    p, K, q_lv = 1, 1, 1
    x̂ = [0.0, 1.0, 0.5]
    nll = (_θ) -> 0.0
    wald_se = [1.0]
    ci = GLLVModels._lv_effect_profile(
        nll, x̂, p, K, q_lv, 0.95,
        GLLVModels._lv_effects_from_packed, wald_se;
        ad = false,
        profile_indices = [1],
        profile_iterations = 1,
        profile_max_expand = 1,
        profile_max_bisect = 1,
    )
    @test ci.method === :profile
    @test any(isnan, ci.lower) || any(isnan, ci.upper)
    @test ci.pd_hessian === false
end
