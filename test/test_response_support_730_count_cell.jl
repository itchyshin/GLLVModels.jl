# #730 follow-up to the response-support checks: every count route names the first offending cell
# for a non-integer or infinite response, and accepts integer-valued Float64 (3.0) exactly as the Int
# fit. Routes covered here are the ones that did not do so on the model-structure branch: the
# two-part count fitters (ZIP, ZINB, hurdle Poisson / NB, ZIP / ZINB with covariates; Inf only, a
# non-integer count there ends on loglik = -Inf and is left to the ZIP-NONINT decision), the per-trait
# truncated NB2 fitter, and the Poisson / NB / Binomial VA fitters (typed on Integer, so a Float64
# matrix used to be a MethodError).
using Test, GLLVModels, Random, Distributions

@testset "#730 count routes: first offending cell named, integer-valued floats accepted" begin
    rng = MersenneTwister(11)
    p, n = 4, 30
    Ycnt = rand(rng, 0:6, p, n)
    Ypos = max.(Ycnt, 1)
    Ybin = rand(rng, 0:5, p, n); Nbin = fill(5, p, n)
    Xcov = randn(rng, p, n, 1)
    thrown(f) = try; f(); nothing; catch e; e; end
    function bad_cell(Y, v)
        Z = float.(Y); Z[2, 3] = v; Z
    end
    names_cell(e, name, v) = e isa ArgumentError && occursin(name, e.msg) &&
                             occursin("Y[2, 3]", e.msg) && occursin(string(v), e.msg)
    it = 30
    function same_fit(a, b)       # every numeric field bit-identical
        fs = [f for f in fieldnames(typeof(a)) if getfield(a, f) isa Union{Number, AbstractArray{<:Number}}]
        return !isempty(fs) && all(f -> isequal(getfield(a, f), getfield(b, f)), fs)
    end

    twopart = Set(["fit_zip_gllvm", "fit_zinb_gllvm", "fit_hurdle_poisson_gllvm", "fit_hurdle_nb_gllvm",
                   "fit_zip_gllvm_cov", "fit_zinb_gllvm_cov"])
    routes = [
        ("fit_zip_gllvm",                Ycnt, Y -> fit_zip_gllvm(Y; K = 1, iterations = it)),
        ("fit_zinb_gllvm",               Ycnt, Y -> fit_zinb_gllvm(Y; K = 1, iterations = it)),
        ("fit_hurdle_poisson_gllvm",     Ycnt, Y -> fit_hurdle_poisson_gllvm(Y; K = 1, iterations = it)),
        ("fit_hurdle_nb_gllvm",          Ycnt, Y -> fit_hurdle_nb_gllvm(Y; K = 1, iterations = it)),
        ("fit_zip_gllvm_cov",            Ycnt, Y -> fit_zip_gllvm_cov(Y; X = Xcov, K = 1, iterations = it)),
        ("fit_zinb_gllvm_cov",           Ycnt, Y -> fit_zinb_gllvm_cov(Y; X = Xcov, K = 1, iterations = it)),
        ("fit_truncated_nbinom2_gllvm_pertrait", Ypos,
            Y -> fit_truncated_nbinom2_gllvm_pertrait(Y; K = 1, iterations = it)),
        ("fit_poisson_gllvm_va",         Ycnt, Y -> fit_poisson_gllvm_va(Y; K = 1, iterations = it)),
        ("fit_nb_gllvm_va",              Ycnt, Y -> fit_nb_gllvm_va(Y; K = 1, iterations = it)),
        ("fit_binomial_gllvm_va",        Ybin, Y -> fit_binomial_gllvm_va(Y; K = 1, N = Nbin, iterations = it)),
    ]
    for (name, Y, call) in routes
        @testset "$name" begin
            name in twopart || @test names_cell(thrown(() -> call(bad_cell(Y, 2.5))), name, 2.5)
            @test names_cell(thrown(() -> call(bad_cell(Y, Inf))), name, Inf)
            @test thrown(() -> call(bad_cell(Y, -1.0))) isa ArgumentError
            fi = call(Y); ff = call(float.(Y))           # (d): 3.0 is accepted and equals the Int fit
            @test ff.loglik == fi.loglik
            @test same_fit(fi, ff)
        end
    end

    @testset "VA: masked NaN and missing in a Float64 count matrix" begin
        mask = trues(p, n); mask[2, 3] = false
        Z = bad_cell(Ycnt, NaN)
        @test isfinite(fit_poisson_gllvm_va(Z; K = 1, iterations = it, mask = mask).loglik)
        Zm = Matrix{Union{Missing, Float64}}(float.(Ycnt)); Zm[2, 3] = missing
        @test isfinite(fit_poisson_gllvm_va(Zm; K = 1, iterations = it).loglik)
        @test names_cell(thrown(() -> fit_poisson_gllvm_va(bad_cell(Ycnt, 2.5); K = 1, mask = trues(p, n))),
                         "fit_poisson_gllvm_va", 2.5)
    end
end
