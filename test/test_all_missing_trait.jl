# A trait (row of the p x n response matrix) with no observed cell is refused with an
# ArgumentError that names the trait. Before this, the Laplace and variational (VA)
# fitters returned beta = 0 with converged = true. The Gaussian routes already refused.
# The shared check lives in `_resolve_obs_mask` (laplace.jl).

using GLLVModels, Test, Random, Distributions

const _AM_K = 1
const _AM_IT = (iterations = 3, g_tol = 1e-2)

function _am_msg(f)
    try
        f()
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return nothing
end

_am_ll(f) = hasproperty(f, :loglik) ? f.loglik : f.logLik

function _am_data(rng)
    p, n = 4, 12
    return (count = rand(rng, 0:4, p, n), bin = rand(rng, 0:1, p, n),
            pos = 0.5 .+ rand(rng, p, n), unit = 0.05 .+ 0.9 .* rand(rng, p, n),
            gauss = randn(rng, p, n))
end

# Each entry: (name, fitter(Y; kwargs...), data key). Y is the p x n response.
const _AM_FITTERS = [
    ("poisson",  (Y; kw...) -> fit_poisson_gllvm(Y; K = _AM_K, _AM_IT..., kw...), :count),
    ("nb",       (Y; kw...) -> fit_nb_gllvm(Y; K = _AM_K, _AM_IT..., kw...), :count),
    ("binomial", (Y; kw...) -> fit_binomial_gllvm(Y; K = _AM_K, _AM_IT..., kw...), :bin),
    ("gamma",    (Y; kw...) -> fit_gamma_gllvm(Y; K = _AM_K, _AM_IT..., kw...), :pos),
    ("poisson_va",  (Y; kw...) -> fit_poisson_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :count),
    ("nb_va",       (Y; kw...) -> fit_nb_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :count),
    ("binomial_va", (Y; kw...) -> fit_binomial_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :bin),
    ("gamma_va",    (Y; kw...) -> fit_gamma_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :pos),
    ("beta_va",     (Y; kw...) -> fit_beta_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :unit),
    ("exponential_va", (Y; kw...) -> fit_exponential_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :pos),
    ("delta_gamma_va", (Y; kw...) -> fit_delta_gamma_gllvm_va(Y; K = _AM_K, _AM_IT..., kw...), :pos),
    ("gaussian",  (Y; kw...) -> fit_gaussian_gllvm(Y; K = _AM_K, kw...), :gauss),
    ("gaussian_pervar", (Y; kw...) -> fit_gaussian_pervar_gllvm(Y; K = _AM_K, kw...), :gauss),
]

@testset "All-missing trait is an ArgumentError naming the trait" begin
    rng = MersenneTwister(1112)
    D = _am_data(rng)
    @testset "$name" for (name, fit, key) in _AM_FITTERS
        Y = D[key]
        p, n = size(Y)
        # via `missing`
        Ym = Matrix{Union{Missing,eltype(Y)}}(Y)
        Ym[3, :] .= missing
        msg = _am_msg(() -> fit(Ym))
        @test msg !== nothing
        msg !== nothing && @test occursin("trait 3", msg)
        # via an explicit mask column of false (here: a whole row of the p x n mask)
        mask = trues(p, n); mask[2, :] .= false
        msg = _am_msg(() -> fit(Y; mask = mask))
        @test msg !== nothing
        msg !== nothing && @test occursin("trait 2", msg)
        # one observed cell per trait is enough: still fits
        mask1 = falses(p, n); mask1[:, 1] .= true; mask1[:, 2:6] .= true
        fm = fit(Y; mask = mask1)
        @test isfinite(_am_ll(fm))
        # a trait observed at a single cell only is also accepted
        mask2 = trues(p, n); mask2[2, 2:n] .= false
        @test isfinite(_am_ll(fit(Y; mask = mask2)))
    end

    @testset "complete data and partial missing are unchanged" begin
        Y = D[:count]
        a = fit_poisson_gllvm(Y; K = _AM_K, _AM_IT...)
        b = fit_poisson_gllvm(Y; K = _AM_K, _AM_IT..., mask = trues(size(Y)))
        @test a.loglik == b.loglik
        Ym = Matrix{Union{Missing,Int}}(Y); Ym[1, 1] = missing
        mk = trues(size(Y)); mk[1, 1] = false
        @test fit_poisson_gllvm(Ym; K = _AM_K, _AM_IT...).loglik ==
              fit_poisson_gllvm(Y; K = _AM_K, _AM_IT..., mask = mk).loglik
    end

    @testset "a site with no observed cell is still accepted (not decided; unchanged)" begin
        Y = D[:count]
        mask = trues(size(Y)); mask[:, 5] .= false
        @test isfinite(fit_poisson_gllvm(Y; K = _AM_K, _AM_IT..., mask = mask).loglik)
    end

    # cv_gllvm(split = :species) on a non-Normal family used to fit all p rows with the
    # held-out species fully masked, which the all-missing check now refuses. It fits the
    # training species only (test_cv.jl #768 covers the Poisson shape; this pins the route).
    @testset "species-split CV runs with the check in force; a direct fit still raises" begin
        rng = MersenneTwister(1134)
        p, n = 6, 20
        Yc = rand(rng, 0:5, p, n)
        Yb = rand(rng, 0:1, p, n)
        Nb = fill(3, p, n); Yt = rand(rng, 0:3, p, n)
        for (fam, Y, kw) in ((Poisson(), Yc, (;)), (NegativeBinomial(), Yc, (;)),
                             (Binomial(), Yb, (;)), (Binomial(), Yt, (N = Nb,)))
            cv = cv_gllvm(Y; k_folds = 3, split = :species, family = fam, K = 1,
                          rng = MersenneTwister(1134), kw...)
            @test cv.split === :species
            @test isfinite(cv.loglik) && isfinite(cv.mse)
            @test all(f -> size(f.Λ, 1) == 4, cv.fits)   # 2 of 6 species held out per fold
        end
        mask = trues(p, n); mask[2, :] .= false
        @test_throws ArgumentError fit_gllvm(Yc; family = Poisson(), K = 1, mask = mask)
    end
end
