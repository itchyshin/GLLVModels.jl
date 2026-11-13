using GLLVModels, Test, Random, Distributions, LinearAlgebra, SparseArrays, StatsModels

# #1217: fitters that take neither `mask` nor `missing` responses used to end a masked call,
# or a call with `missing` in `Y`, in a raw `MethodError` that named an internal signature.
# They now throw one `ArgumentError` that names the route and says that a mask / missing
# responses are not supported. Mask SUPPORT on these routes is not added: the refusal fires
# before any work, and routes that take a mask are untouched.

@testset "mask / missing responses on routes without mask support (#1217)" begin
    rng = MersenneTwister(1217)
    p, n = 4, 30
    cnt = Float64.(rand(rng, 0:4, p, n)); cnt[1, 1] = 0.0
    zi = Float64.([rand(rng) < 0.3 ? 0 : 1 + rand(rng, Poisson(2)) for _ in 1:p, _ in 1:n])
    pos0 = Float64.([rand(rng) < 0.3 ? 0.0 : 0.5 + 3rand(rng) for _ in 1:p, _ in 1:n])
    unit0 = Float64.([rand(rng) < 0.3 ? 0.0 : 0.1 + 0.8rand(rng) for _ in 1:p, _ in 1:n])
    gau = randn(rng, p, n) .+ 1
    X2 = randn(rng, n, 2); X3 = randn(rng, p, n, 2)
    M = trues(p, n); M[2, 3] = false; M[3, 7] = false

    unionY(Y) = (Z = Matrix{Union{Missing, Float64}}(Y); Z[.!M] .= missing; Z)

    # (name, data, call(Y; kw...))
    routes = [
        ("fit_row_random_gllvm", cnt, (Y; kw...) -> fit_row_random_gllvm(Y; family = Poisson(), K = 1, kw...)),
        ("fit_rrr_gllvm", cnt, (Y; kw...) -> fit_rrr_gllvm(Y; family = Poisson(), X = X2, K = 1, kw...)),
        ("fit_constrained_gllvm", cnt, (Y; kw...) -> fit_constrained_gllvm(Y; family = Poisson(), X = X2, K = 1, kw...)),
        ("fit_constrained_gllvm", cnt, (Y; kw...) -> fit_concurrent_gllvm(Y; family = Poisson(), X = X2, K = 1, kw...)),
        ("fit_quadratic_gllvm", cnt, (Y; kw...) -> fit_quadratic_gllvm(Y; family = Poisson(), K = 1, kw...)),
        ("fit_studentt_gllvm", gau, (Y; kw...) -> fit_studentt_gllvm(Y; K = 1, kw...)),
        ("fit_beta_hurdle_gllvm", unit0, (Y; kw...) -> fit_beta_hurdle_gllvm(Y; K = 1, kw...)),
        ("fit_delta_lognormal_gllvm", pos0, (Y; kw...) -> fit_delta_lognormal_gllvm(Y; K = 1, kw...)),
        ("fit_delta_gamma_gllvm", pos0, (Y; kw...) -> fit_delta_gamma_gllvm(Y; K = 1, kw...)),
        ("fit_hurdle_poisson_gllvm", zi, (Y; kw...) -> fit_hurdle_poisson_gllvm(Y; K = 1, kw...)),
        ("fit_lognormal_gllvm", pos0 .+ 1, (Y; kw...) -> fit_lognormal_gllvm(Y; K = 1, kw...)),
        ("fit_hurdle_nb_gllvm", zi, (Y; kw...) -> fit_hurdle_nb_gllvm(Y; K = 1, kw...)),
        ("fit_zip_gllvm", zi, (Y; kw...) -> fit_zip_gllvm(Y; K = 1, kw...)),
        ("fit_zinb_gllvm", zi, (Y; kw...) -> fit_zinb_gllvm(Y; K = 1, kw...)),
        ("fit_zib_gllvm", cnt, (Y; kw...) -> fit_zib_gllvm(Y; K = 1, N = 4, kw...)),
        ("fit_zip_gllvm_cov", zi, (Y; kw...) -> fit_zip_gllvm_cov(Y; X = X3, K = 1, kw...)),
        ("fit_zinb_gllvm_cov", zi, (Y; kw...) -> fit_zinb_gllvm_cov(Y; X = X3, K = 1, kw...)),
        ("fit_zib_gllvm_cov", cnt, (Y; kw...) -> fit_zib_gllvm_cov(Y; X = X3, K = 1, N = 4, kw...)),
        ("fit_grouped_gaussian", gau, (Y; kw...) -> GLLVModels.fit_grouped_gaussian(Y; terms = [GLLVModels.GroupingTerm(:unit)], unit = collect(1:n), kw...)),
        ("fit_grouped_nongaussian", cnt, (Y; kw...) -> GLLVModels.fit_grouped_nongaussian(Y; family = Poisson(), terms = [GLLVModels.GroupingTerm(:unit)], unit = collect(1:n), kw...)),
        # public front doors
        ("fit_row_random_gllvm", cnt, (Y; kw...) -> fit_gllvm(Y; family = Poisson(), K = 1, row_eff = :random, kw...)),
        ("fit_zip_gllvm", zi, (Y; kw...) -> fit_gllvm(Y; family = ZIPoisson(), K = 1, kw...)),
        ("fit_zinb_gllvm", zi, (Y; kw...) -> fit_gllvm(Y; family = ZINegBin(), K = 1, kw...)),
        ("fit_hurdle_poisson_gllvm", zi, (Y; kw...) -> fit_gllvm(Y; family = HurdlePoisson(), K = 1, kw...)),
        ("fit_delta_gamma_gllvm", pos0, (Y; kw...) -> fit_gllvm(Y; family = DeltaGamma(), K = 1, kw...)),
        ("fit_studentt_gllvm", gau, (Y; kw...) -> fit_gllvm(Y; family = StudentTFamily(), K = 1, kw...)),
        ("fit_row_random_gllvm", cnt, (Y; kw...) -> gllvm(@formula(y ~ 1), Y, (site = 1:n,); family = Poisson(), K = 1, row_eff = :random, kw...)),
        ("fit_zip_gllvm", zi, (Y; kw...) -> gllvm(@formula(y ~ 1), Y, (site = 1:n,); family = ZIPoisson(), K = 1, kw...)),
    ]

    for (name, Y, call) in routes
        @testset "$name" begin
            e1 = try call(Y; mask = M); nothing catch e; e end
            @test e1 isa ArgumentError
            e1 isa ArgumentError && @test occursin(name, e1.msg) && occursin("mask is not supported", e1.msg)
            e2 = try call(unionY(Y)); nothing catch e; e end
            @test e2 isa ArgumentError
            e2 isa ArgumentError && @test occursin(name, e2.msg) && occursin("missing responses are not supported", e2.msg) &&
                occursin("trait 2, unit 3", e2.msg)
        end
    end

    @testset "fit_zi_gllvm (gllvmTMB twin): mask refused here, missing already refused" begin
        e1 = try GLLVModels.fit_zi_gllvm(zi; family = GLLVModels.ZiPoisson(), K = 1, mask = M); nothing catch e; e end
        @test e1 isa ArgumentError && occursin("fit_zi_gllvm", e1.msg) && occursin("mask is not supported", e1.msg)
        e2 = try GLLVModels.fit_zi_gllvm(unionY(zi); family = GLLVModels.ZiPoisson(), K = 1); nothing catch e; e end
        @test e2 isa ArgumentError && occursin("no missing values", e2.msg)
    end

    @testset "route with Gaussian precision phylogeny" begin
        q = sparse(2.5 .* [4.0 -1.0 -1.0 0.0; -1.0 3.0 0.0 -1.0; -1.0 0.0 3.0 0.0; 0.0 -1.0 0.0 3.0])
        ii, jj, xx = findnz(q)
        pp = GLLVModels.PrecisionPhy(ii, jj, xx, 4, 2, ["a1", "a2", "sA", "sB"],
                          logdet(cholesky(Symmetric(Matrix(q)))), 2.5, [3, 4])
        y = [0.4 -0.7 0.2; -0.3 0.1 0.9]
        ym = Matrix{Union{Missing, Float64}}(y); ym[1, 2] = missing
        for (name, f) in (("fit_precision_multivariate", GLLVModels.fit_precision_multivariate),
                          ("fit_joint_phylo_grouped_gaussian", GLLVModels.fit_joint_phylo_grouped_gaussian))
            kw = name == "fit_joint_phylo_grouped_gaussian" ? (; terms = [GLLVModels.GroupingTerm(:unit)]) : (;)
            e1 = try f(y, pp; mask = trues(2, 3), kw...); nothing catch e; e end
            @test e1 isa ArgumentError && occursin(name, e1.msg) && occursin("mask is not supported", e1.msg)
            e2 = try f(ym, pp; kw...); nothing catch e; e end
            @test e2 isa ArgumentError && occursin(name, e2.msg) && occursin("missing responses are not supported", e2.msg)
            # A second argument that is not a `PrecisionPhy` has no method on main: it must stay a
            # `MethodError` (an untyped forwarding method once recursed into a `StackOverflowError`).
            aphy = GLLVModels.augmented_phy("((A:1,B:1):1,(C:1,D:1):1);")
            @test_throws MethodError f(y, aphy; kw...)
            @test_throws MethodError f(y, 3; kw...)
            @test_throws MethodError f(ym, aphy; kw...)
            @test_throws MethodError f(ym, 3; kw...)
        end
    end

    @testset "fit_phylo_glm, fit_spde_latent_gllvm, fit_mixed_gllvm, fit_gaussian_reml (untyped Y)" begin
        phy = GLLVModels.augmented_phy("((A:1,B:1):1,(C:1,D:1):1);")
        Yp = Float64.(rand(rng, 0:3, 4, 10)); Mp = trues(4, 10); Mp[1, 1] = false
        Ypm = Matrix{Union{Missing, Float64}}(Yp); Ypm[1, 1] = missing
        nodes = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]; tris = [1 2 3; 2 4 3]; locs = nodes[1:3, :]
        Ys = Float64.(rand(rng, 0:3, 2, 3)); Ms = trues(2, 3); Ms[1, 1] = false
        Ysm = Matrix{Union{Missing, Float64}}(Ys); Ysm[1, 1] = missing
        cases = [
            ("fit_phylo_glm", () -> fit_phylo_glm(Yp, phy; mask = Mp), () -> fit_phylo_glm(Ypm, phy)),
            ("fit_spde_latent_gllvm", () -> fit_spde_latent_gllvm(Ys, nodes, tris, locs; mask = Ms),
                                      () -> fit_spde_latent_gllvm(Ysm, nodes, tris, locs)),
        ]
        for (name, withmask, withmissing) in cases
            e1 = try withmask(); nothing catch e; e end
            @test e1 isa ArgumentError && occursin(name, e1.msg) && occursin("mask is not supported", e1.msg)
            e2 = try withmissing(); nothing catch e; e end
            @test e2 isa ArgumentError && occursin(name, e2.msg) && occursin("missing responses are not supported", e2.msg)
        end

        # fit_mixed_gllvm refuses a mask only. `missing` in the Normal rows of Y is fitted on main
        # (NA-aware likelihood); this change must not refuse it. (Missing in a non-Normal row is a
        # raw `convert` MethodError on main and is not changed here.)
        fams = Distribution[Poisson(), Poisson(), Normal(), Normal()]
        e1 = try fit_mixed_gllvm(cnt; families = fams, K = 1, mask = M); nothing catch e; e end
        @test e1 isa ArgumentError && occursin("fit_mixed_gllvm", e1.msg) && occursin("mask is not supported", e1.msg)
        uN = Matrix{Union{Missing, Float64}}(cnt); uN[3, 5] = missing; uN[4, 9] = missing
        fmix = try fit_mixed_gllvm(uN; families = fams, K = 1, iterations = 40) catch e; e end
        @test !(fmix isa Exception)
        fmix isa Exception || @test isfinite(fmix.loglik)
        fcom = fit_mixed_gllvm(Matrix{Union{Missing, Float64}}(cnt); families = fams, K = 1, iterations = 40)
        @test isfinite(fcom.loglik)

        # fit_gaussian_reml: mask refused here; `missing` gets the same text as draft PR 1252.
        e1 = try fit_gaussian_reml(gau, X3; K = 1, mask = M); nothing catch e; e end
        @test e1 isa ArgumentError && occursin("fit_gaussian_reml", e1.msg) && occursin("mask is not supported", e1.msg)
        e2 = try fit_gaussian_reml(unionY(gau), X3; K = 1); nothing catch e; e end
        @test e2 isa ArgumentError
        e2 isa ArgumentError && @test e2.msg == "fit_gaussian_reml does not accept `missing` in `Y`: 2 missing " *
            "cell(s), first at (row, column) = (2, 3). This fitter has no `mask`; remove or impute the " *
            "missing cells, or use `fit_gaussian_gllvm`, which derives a mask from `missing`."
    end

    @testset "unchanged: no mask, no missing, and routes that take a mask" begin
        # A `Union{Missing,Float64}` matrix with no missing cell is the plain fit, bit for bit.
        Z = Matrix{Union{Missing, Float64}}(zi)
        @test fit_zip_gllvm(Z; K = 1, iterations = 30).loglik == fit_zip_gllvm(zi; K = 1, iterations = 30).loglik
        Zc = Matrix{Union{Missing, Float64}}(cnt)
        @test fit_row_random_gllvm(Zc; family = Poisson(), K = 1, iterations = 30).loglik ==
              fit_row_random_gllvm(cnt; family = Poisson(), K = 1, iterations = 30).loglik
        # `mask = nothing` is the same as no keyword.
        @test fit_zip_gllvm(zi; K = 1, iterations = 30, mask = nothing).loglik == fit_zip_gllvm(zi; K = 1, iterations = 30).loglik
        # Routes that take a mask still fit it, and `missing` still means the same cells.
        fm = fit_gllvm(cnt; family = Poisson(), K = 1, mask = M, iterations = 60)
        fn = fit_gllvm(unionY(cnt); family = Poisson(), K = 1, iterations = 60)
        @test fm.loglik == fn.loglik
        @test isfinite(fit_gllvm(cnt; family = Poisson(), K = 1, row_eff = :fixed, mask = M, iterations = 60).loglik)
        # NaN is not a mask: this change does not touch it (still the route's own error).
        cn = copy(zi); cn[2, 3] = NaN
        @test_throws ArgumentError fit_zip_gllvm(cn; K = 1)
    end
end
