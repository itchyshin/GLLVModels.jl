using GLLVModels, Test, Random, LinearAlgebra, Distributions

# `fit_nb_gllvm_grouped`, `fit_nb1_gllvm_grouped` and `fit_beta_gllvm_grouped` accept
# `β_init` / `Λ_init`, as `fit_nb_gllvm` does (src/families/negbin.jl). Before this they
# threw a MethodError, so `fit_gllvm(Y; family = NegativeBinomial(), K, β_init, Λ_init)`
# (the default per-species dispersion route) could not be warm-started at all.

function _grouped_init_data(kind; p = 6, n = 60, K = 2, seed = 20260927)
    rng = MersenneTwister(seed)
    Λ = 0.6 .* randn(rng, p, K)
    η = (kind === :beta ? 0.0 : log(3.0)) .+ Λ * randn(rng, K, n)
    kind === :beta && return [rand(rng, Beta(5 / (1 + exp(-x)), 5 - 5 / (1 + exp(-x)))) for x in η]
    return [rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + exp(x)))) for x in η]
end

# The fitters' own default start (empirical link-means + scaled SVD loadings),
# reimplemented here so the identity test does not read it from the code under test.
function _grouped_default_start(Z, K)
    n = size(Z, 2)
    β = vec(sum(Z; dims = 2)) ./ n
    F = svd(Z .- β)
    return β, F.U[:, 1:K] .* (F.S[1:K]' ./ sqrt(n))
end

const _GROUPED_INIT_CASES = (
    (:nb,   fit_nb_gllvm_grouped,   Y -> log.(Y .+ 0.5)),
    (:nb1,  fit_nb1_gllvm_grouped,  Y -> log.(Y .+ 0.5)),
    (:beta, fit_beta_gllvm_grouped,
            Y -> GLLVModels.linkfun.(Ref(GLLVModels.LogitLink()), clamp.(float.(Y), 1e-6, 1 - 1e-6))),
)

@testset "grouped fitters accept β_init / Λ_init ($kind)" for (kind, fitter, zemp) in _GROUPED_INIT_CASES
    Y = _grouped_init_data(kind)
    p, K = size(Y, 1), 2
    grp = collect(1:p)
    base = fitter(Y; K = K, group = grp)

    # Passing the default start explicitly is the same fit.
    β0, Λ0 = _grouped_default_start(zemp(Y), K)
    same = fitter(Y; K = K, group = grp, β_init = β0, Λ_init = Λ0)
    @test same.loglik == base.loglik
    @test same.β == base.β
    @test same.Λ == base.Λ

    # The start is used: with no optimizer iterations the fit returns it.
    held = fitter(Y; K = K, group = grp, β_init = base.β, Λ_init = base.Λ, iterations = 0)
    @test held.β == base.β
    @test held.Λ == base.Λ

    # Wrong shapes are rejected with an ArgumentError, not a silent reshape.
    @test_throws ArgumentError fitter(Y; K = K, group = grp, β_init = zeros(p + 1))
    @test_throws ArgumentError fitter(Y; K = K, group = grp, Λ_init = zeros(p, K + 1))
end

@testset "fit_gllvm NegativeBinomial per-species route forwards β_init / Λ_init" begin
    Y = _grouped_init_data(:nb)
    base = fit_gllvm(Y; family = NegativeBinomial(), K = 2, disp_group = :species)
    held = fit_gllvm(Y; family = NegativeBinomial(), K = 2, disp_group = :species,
                     β_init = base.β, Λ_init = base.Λ, iterations = 0)
    @test held.β == base.β
    @test held.Λ == base.Λ
end
