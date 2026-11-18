# #730 (model-structure and phylogenetic fitters): the fitters that take `family` (covariates,
# species covariates, fourth-corner, row effects, row random effects, constrained, RRR and
# quadratic ordination, `fit_phylo_glm`), the phylogenetic x latent-variable fitters, and the
# last two grouped NB fitters must reject out-of-support or non-finite OBSERVED responses with an
# ArgumentError that names the fitter and the first bad cell, instead of returning a fit with
# loglik = -Inf or failing late with an InexactError. Masked and `missing` cells stay legal.
using Test, GLLVModels, Random, Distributions

@testset "#730 response-domain validation, model-structure fitters" begin
    rng = MersenneTwister(11)
    p, n = 4, 30
    Ycnt = rand(rng, 0:6, p, n)
    Ypos = exp.(randn(rng, p, n))
    Yunit = 0.05 .+ 0.9 .* rand(rng, p, n)
    Ybin = rand(rng, 0:1, p, n)
    Nbin = fill(5, p, n)
    Xcov = randn(rng, p, n, 1)       # (p, n, q)
    Xsite = randn(rng, n, 1)         # (n, q)
    Xenv = randn(rng, n, 1)
    TR = randn(rng, p, 1)

    function with_cell(Y, v)
        Z = Matrix{Any}(float.(Y)); Z[1, 1] = v
        return identity.(Z)
    end
    thrown(f) = try; f(); nothing; catch e; e; end
    msg_ok(e, name) = e isa ArgumentError && occursin(name, e.msg) && occursin("Y[1, 1]", e.msg)

    # (name, call(Y; family), family, clean Y, bad values)
    fams = [
        (Poisson(),           Ycnt,  (-1.0, 2.5, NaN)),
        (NegativeBinomial(),  Ycnt,  (-1.0, 2.5, NaN)),
        (NB1(),               Ycnt,  (-1.0, 2.5, NaN)),
        (Gamma(),             Ypos,  (-1.0, 0.0, NaN)),
        (Exponential(),       Ypos,  (-1.0, 0.0, NaN)),
        (Beta(),              Yunit, (0.0, 1.0, 1.5, NaN)),
    ]
    callers = [
        ("fit_gllvm_cov",          (Y, f) -> fit_gllvm_cov(Y; family = f, X = Xcov, K = 1)),
        ("fit_gllvm_speciescov",   (Y, f) -> fit_gllvm_speciescov(Y; family = f, X = Xcov, K = 1)),
        ("fit_fourthcorner_gllvm", (Y, f) -> fit_fourthcorner_gllvm(Y; family = f, Xenv = Xenv, TR = TR, K = 1)),
        ("fit_roweffect_gllvm",    (Y, f) -> fit_roweffect_gllvm(Y; family = f, K = 1)),
        ("fit_row_random_gllvm",   (Y, f) -> fit_row_random_gllvm(Y; family = f, K = 1)),
        ("fit_constrained_gllvm",  (Y, f) -> fit_constrained_gllvm(Y; family = f, X = Xsite, K = 1)),
        ("fit_rrr_gllvm",          (Y, f) -> fit_rrr_gllvm(Y; family = f, X = Xsite, K = 1)),
        ("fit_quadratic_gllvm",    (Y, f) -> fit_quadratic_gllvm(Y; family = f, K = 1)),
    ]
    for (cname, call) in callers
        @testset "$cname rejects out-of-support responses" begin
            for (fam, Y, bads) in fams, bad in bads
                @test msg_ok(thrown(() -> call(with_cell(Y, bad), fam)), cname)
            end
            @test msg_ok(thrown(() -> call(with_cell(Ybin, 2), Binomial())), cname)
            @test msg_ok(thrown(() -> call(with_cell(Ybin, -1), Binomial())), cname)
        end
    end

    @testset "binomial y > N, with y == N legal (fit_gllvm_cov)" begin
        Y = float.(rand(rng, 0:5, p, n)); Y[1, 1] = 6
        @test msg_ok(thrown(() -> fit_gllvm_cov(Y; family = Binomial(), X = Xcov, K = 1, N = Nbin)),
                     "fit_gllvm_cov")
        Y[1, 1] = 5
        @test thrown(() -> fit_gllvm_cov(Y; family = Binomial(), X = Xcov, K = 1, N = Nbin,
                                         iterations = 30)) === nothing
    end

    @testset "boundary and masked data still fit" begin
        Yz = copy(Ycnt); Yz[1, 1] = 0                       # zero count is legal
        fit = fit_gllvm_cov(float.(Yz); family = Poisson(), X = Xcov, K = 1, iterations = 30)
        @test isfinite(fit.loglik)
        mask = trues(p, n); mask[1, 1] = false
        for (f, fam, Y, bad) in ((fit_gllvm_cov, Poisson(), Ycnt, NaN), (fit_gllvm_cov, Gamma(), Ypos, -1.0),
                                 (fit_roweffect_gllvm, Poisson(), Ycnt, NaN), (fit_roweffect_gllvm, Beta(), Yunit, 1.5))
            Z = float.(Y); Z[1, 1] = bad
            kw = f === fit_gllvm_cov ? (; X = Xcov) : (;)
            fit = f(Z; family = fam, K = 1, mask = mask, iterations = 30, kw...)
            @test isfinite(fit.loglik)
        end
        Ym = Matrix{Union{Missing, Float64}}(float.(Ycnt)); Ym[1, 1] = missing
        fit = fit_gllvm_cov(Ym; family = Poisson(), X = Xcov, K = 1, iterations = 30)
        @test isfinite(fit.loglik)
    end

    @testset "grouped NB and NB1 (a NaN in a masked cell no longer hits Int64(NaN))" begin
        for (name, f, kw) in (("fit_nb_gllvm_grouped", fit_nb_gllvm_grouped, (; group = [1, 1, 2, 2])),
                              ("fit_nb_gllvm_grouped_cov", fit_nb_gllvm_grouped_cov, (; X = Xcov)),
                              ("fit_nb1_gllvm_grouped", fit_nb1_gllvm_grouped, (; group = [1, 1, 2, 2])),
                              ("fit_nb1_gllvm_grouped_cov", fit_nb1_gllvm_grouped_cov, (; X = Xcov)))
            for bad in (-1.0, 2.5, NaN)
                @test msg_ok(thrown(() -> f(with_cell(Ycnt, bad); K = 1, kw...)), name)
            end
            mask = trues(p, n); mask[1, 1] = false
            Z = with_cell(Ycnt, NaN)
            fit = f(Z; K = 1, mask = mask, iterations = 30, kw...)
            @test isfinite(fit.loglik)
        end
    end

    @testset "phylogenetic fitters" begin
        phy = GLLVModels.augmented_phy("((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2);")
        Xlv = reshape(collect(range(-1.0, 1.0; length = n)), n, 1)
        for (name, f, Y, bads) in (
                ("_fit_phylo_poisson_xlv", GLLVModels._fit_phylo_poisson_xlv, Ycnt, (-1.0, 2.5, NaN)),
                ("_fit_phylo_nb_xlv",      GLLVModels._fit_phylo_nb_xlv,      Ycnt, (-1.0, 2.5, NaN)),
                ("_fit_phylo_gamma_xlv",   GLLVModels._fit_phylo_gamma_xlv,   Ypos, (-1.0, 0.0, NaN)),
                ("_fit_phylo_beta_xlv",    GLLVModels._fit_phylo_beta_xlv,    Yunit, (0.0, 1.0, NaN)))
            for bad in bads
                @test msg_ok(thrown(() -> f(with_cell(Y, bad), phy; K = 1, X_lv = Xlv)), name)
            end
        end
        @test msg_ok(thrown(() -> GLLVModels._fit_phylo_binomial_xlv(with_cell(Ybin, 6), Nbin, phy;
                                                                      K = 1, X_lv = Xlv)),
                     "_fit_phylo_binomial_xlv")
        for (fam, Y, bads) in ((Poisson(), Ycnt, (-1.0, 2.5, NaN)), (NegativeBinomial(), Ycnt, (-1.0,)),
                               (Gamma(), Ypos, (-1.0, 0.0)), (Beta(), Yunit, (0.0, 1.0)))
            for bad in bads
                @test msg_ok(thrown(() -> fit_phylo_glm(with_cell(Y, bad), phy; family = fam)), "fit_phylo_glm")
            end
        end
        @test msg_ok(thrown(() -> fit_phylo_glm(with_cell(Ybin, 2), phy; family = Binomial(), N = ones(p, n))),
                     "fit_phylo_glm")
    end
end
