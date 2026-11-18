using GLLVModels, Test, LinearAlgebra, Random

# Direct tests of the private helpers behind extract_Gamma and
# extract_coevolution_modules (src/cross_lineage_extract.jl). The public
# functions are covered by test_cross_lineage_p1.jl and test_postfit_tables.jl;
# here each helper is checked against the algebra it implements. The fixture is a
# hand-built GaussianSourcesFit (no optimiser), so the trait covariances are exact
# by construction: Sigma = Lambda Lambda' for known loadings.
const _CLI = GLLVModels

function _cli_fit(covs::Vector{Matrix{Float64}}, srcs)
    p = size(covs[1], 1)
    n = 2
    GaussianSourcesFit(zeros(p), 1.0, covs, srcs, Float64[], -1.0, true, 0.0, 1.0, true,
                       1, :test, p * n, false, zeros(p * n, p), (p, n),
                       Union{String,Symbol}[Symbol("t$i") for i in 1:p])
end

@testset "cross-lineage extract internals" begin
    Random.seed!(20261020)
    p = 4
    Lam = [1.0 0.0; 0.6 0.8; 0.3 -0.5; -0.4 0.7]
    Sigma = Lam * Lam'
    node_cov = Matrix{Float64}(I, 2, 2)
    s_cross = SourceCovariance(node_cov; groups = [1, 2], name = :cross, mode = :latent, rank = 2)
    s_uniq = SourceCovariance(node_cov; groups = [1, 2], name = :withuniq, mode = :latent,
                              rank = 1, unique = true)
    s_indep = SourceCovariance(node_cov; groups = [1, 2], name = :indep, mode = :indep)
    s_dep = SourceCovariance(node_cov; groups = [1, 2], name = :dep, mode = :dep)
    Sigma2 = Matrix{Float64}(Diagonal(1.0:p))
    fit = _cli_fit([Sigma, Sigma2, Sigma2, Sigma2 .+ 1.0],
                   [s_cross, s_uniq, s_indep, s_dep])

    @testset "_matrix_inv_sqrt_rel" begin
        tol = sqrt(eps())
        B = randn(5, 5)
        A = B * B' + 0.5I
        R = _CLI._matrix_inv_sqrt_rel(A, tol, "row_traits")
        @test R ≈ R'                         # symmetric square root
        @test R * A * R' ≈ Matrix(I, 5, 5) atol = 1e-10
        @test R * R' ≈ inv(A) atol = 1e-10   # R = A^(-1/2)

        # ill-conditioned (cond 1e6, above the truncation floor): still whitens
        Q = Matrix(qr(randn(5, 5)).Q)
        Ail = Q * Diagonal([1.0, 1e-1, 1e-2, 1e-4, 1e-6]) * Q'
        Ril = _CLI._matrix_inv_sqrt_rel(Ail, tol, "row_traits")
        @test opnorm(Ril * Ail * Ril' - I) < 1e-6

        # an asymmetric input is symmetrised first
        N = A .+ 1e-3 .* randn(5, 5)
        @test _CLI._matrix_inv_sqrt_rel(N, tol, "x") ≈
              _CLI._matrix_inv_sqrt_rel((N .+ N') ./ 2, tol, "x")

        # near-singular: eigenvalues <= tol * max(max|eig|, 1) are truncated to 0,
        # so R*A*R' is the projector onto the kept eigenspace, not I.
        Asing = Matrix(Diagonal([1.0, 1e-12]))
        Rs = _CLI._matrix_inv_sqrt_rel(Asing, 1e-8, "row_traits")
        @test Rs ≈ Diagonal([1.0, 0.0]) atol = 1e-12
        @test Rs * Asing * Rs' ≈ Diagonal([1.0, 0.0]) atol = 1e-12
        # the threshold is relative to the largest eigenvalue: 1e-3 survives next to 1
        # but is dropped next to 1e6 (threshold 1e-8 * 1e6 = 1e-2)
        @test _CLI._matrix_inv_sqrt_rel(Matrix(Diagonal([1.0, 1e-3])), 1e-8, "x")[2, 2] ≈
              1 / sqrt(1e-3)
        @test _CLI._matrix_inv_sqrt_rel(Matrix(Diagonal([1e6, 1e-3])), 1e-8, "x")[2, 2] == 0.0
        # ... and floored at an absolute scale of 1: a uniformly tiny matrix is "zero"
        @test_throws ArgumentError _CLI._matrix_inv_sqrt_rel(1e-12 .* Matrix(I, 3, 3), 1e-8, "x")
        @test_throws ArgumentError _CLI._matrix_inv_sqrt_rel(zeros(3, 3), tol, "x")
        # a clearly indefinite block throws and names the argument
        err = try
            _CLI._matrix_inv_sqrt_rel(Matrix(Diagonal([1.0, -0.5])), tol, "col_traits"); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("col_traits", err.msg) && occursin("semidefinite", err.msg)
        # a tiny negative eigenvalue inside the tolerance is tolerated, not thrown
        @test _CLI._matrix_inv_sqrt_rel(Matrix(Diagonal([1.0, -1e-12])), 1e-8, "x")[1, 1] ≈ 1.0
    end

    @testset "_cross_lineage_scale" begin
        # a scale parser, not a numeric function: it has no trait scaling to be
        # invariant to. :shape (Symbol or String) is accepted and returns nothing.
        @test _CLI._cross_lineage_scale(:shape) === nothing
        @test _CLI._cross_lineage_scale("shape") === nothing
        @test_throws ArgumentError _CLI._cross_lineage_scale(:effect)
        @test_throws ArgumentError _CLI._cross_lineage_scale("effect")
        @test_throws ArgumentError _CLI._cross_lineage_scale(:other)
        @test_throws ArgumentError _CLI._cross_lineage_scale(1)

        # What the :shape scale does promise, seen through the public entries: Gamma
        # scales by exactly c_r * c_c when one lineage's traits are rescaled, while the
        # standardised block R (and so the singular values) does not change.
        ri, ci = 1:2, 3:4
        c_r, c_c = 3.0, 0.25
        D = Diagonal([c_r, c_r, c_c, c_c])
        fit_s = _cli_fit([Matrix(D * Sigma * D)], [s_cross])
        fit_1 = _cli_fit([Sigma], [s_cross])
        G1 = extract_Gamma(fit_1; level = :cross, row_traits = ri, col_traits = ci)
        Gs = extract_Gamma(fit_s; level = :cross, row_traits = ri, col_traits = ci)
        @test Gs ≈ c_r * c_c .* G1
        m1 = extract_coevolution_modules(fit_1; level = :cross, row_traits = ri, col_traits = ci)
        ms = extract_coevolution_modules(fit_s; level = :cross, row_traits = ri, col_traits = ci)
        @test ms.R ≈ m1.R atol = 1e-9
        @test ms.modules.singular_value ≈ m1.modules.singular_value atol = 1e-9
        @test_throws ArgumentError extract_Gamma(fit_1; level = :cross, row_traits = ri,
                                                 col_traits = ci, scale = :effect)
    end

    @testset "_coevolution_axes" begin
        M = [1.0 2.0 9.0; 3.0 4.0 9.0]       # third column must be dropped (nm = 2)
        t = _CLI._coevolution_axes(M, ["a", "b"], ["module_1", "module_2"], "cross", "row")
        @test keys(t) == (:component, :side, :module, :trait, :loading)
        @test t.loading == [1.0, 3.0, 2.0, 4.0]          # column-major, trait fastest
        @test t.trait == ["a", "b", "a", "b"]
        @test t.module == ["module_1", "module_1", "module_2", "module_2"]
        @test all(==("cross"), t.component) && all(==("row"), t.side)

        # Orthonormal axes in the Euclidean metric, from the public decomposition.
        for (r, c) in ((1:2, 3:4), ([1, 3], [2, 4]))
            m = extract_coevolution_modules(fit; level = :cross, row_traits = r, col_traits = c)
            nm = length(m.modules.module)
            U = reshape(m.row_axes.loading, length(r), nm)
            V = reshape(m.col_axes.loading, length(c), nm)
            @test U' * U ≈ Matrix(I, nm, nm) atol = 1e-10
            @test V' * V ≈ Matrix(I, nm, nm) atol = 1e-10
            sv = m.modules.singular_value
            @test issorted(sv; rev = true)                 # descending order
            @test sum(m.modules.squared_share) ≈ 1
            # sign convention: only the product U_k V_k' is identified, and it rebuilds R
            @test U * Diagonal(sv) * V' ≈ m.R atol = 1e-10
            flip = Diagonal([-1.0; ones(nm - 1)])
            @test (U * flip) * Diagonal(sv) * (V * flip)' ≈ m.R atol = 1e-10
        end
        # n_modules keeps the leading axes unchanged
        mfull = extract_coevolution_modules(fit; level = :cross, row_traits = 1:2, col_traits = 3:4)
        m1 = extract_coevolution_modules(fit; level = :cross, row_traits = 1:2, col_traits = 3:4,
                                         n_modules = 1)
        @test m1.modules.singular_value == mfull.modules.singular_value[1:1]
        @test m1.row_axes.loading == mfull.row_axes.loading[1:2]
        @test m1.col_axes.side == ["column", "column"] && m1.row_axes.side == ["row", "row"]
    end

    @testset "_sources_shared_covariance" begin
        S = _CLI._sources_shared_covariance(fit, :cross)
        @test S == Sigma                      # equals the dense Lambda Lambda'
        @test S ≈ Lam * Lam' atol = 1e-14
        @test issymmetric(S)
        @test minimum(eigvals(Symmetric(S))) > -1e-12        # PSD (rank 2 of 4)
        @test rank(S; atol = 1e-10) == 2
        @test _CLI._sources_shared_covariance(fit, "cross") == S   # String level
        # returns a copy: mutating it must not touch the fit
        S[1, 1] = 99.0
        @test fit.trait_covariances[1][1, 1] == Sigma[1, 1]
        # the named source is picked, not the first
        @test _CLI._sources_shared_covariance(fit, :dep) == Sigma2 .+ 1.0
        # :dep has a shared covariance; :indep and latent-with-unique do not
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, :indep)
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, :withuniq)
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, :nosuch)
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, "")
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, 1)
        @test_throws ArgumentError _CLI._sources_shared_covariance(fit, nothing)
    end

    @testset "_cross_lineage_traits" begin
        names = ["h1", "h2", "p1", "p2"]
        hi, hl = _CLI._cross_lineage_traits([1, 2], "row_traits", p, nothing)
        pi_, pl = _CLI._cross_lineage_traits([3, 4], "col_traits", p, nothing)
        @test hi == hl == [1, 2] && pi_ == pl == [3, 4]
        # the two lineages are disjoint and together cover every trait
        @test isempty(intersect(hi, pi_))
        @test sort(vcat(hi, pi_)) == collect(1:p)
        # names resolve to the same positions as integers, in the order given
        ni, nl = _CLI._cross_lineage_traits(["p2", "h1"], "row_traits", p, names)
        @test ni == [4, 1] && nl == ["p2", "h1"]
        si, _ = _CLI._cross_lineage_traits([:p1, :h2], "row_traits", p, Symbol.(names))
        @test si == [3, 2]
        @test _CLI._cross_lineage_traits(["h1", "h2"], "r", p, names)[1] == hi
        # validation
        @test_throws ArgumentError _CLI._cross_lineage_traits(Int[], "r", p, nothing)
        @test _CLI._cross_lineage_traits(1:2, "r", p, nothing)[1] == [1, 2]   # ranges accepted
        @test_throws ArgumentError _CLI._cross_lineage_traits([1, 1], "r", p, nothing)
        @test_throws ArgumentError _CLI._cross_lineage_traits([0, 1], "r", p, nothing)
        @test_throws ArgumentError _CLI._cross_lineage_traits([1, p + 1], "r", p, nothing)
        @test_throws ArgumentError _CLI._cross_lineage_traits([true, false], "r", p, nothing)
        @test_throws ArgumentError _CLI._cross_lineage_traits(["h1"], "r", p, nothing)
        @test_throws ArgumentError _CLI._cross_lineage_traits(["h1"], "r", p, ["a", "b"])
        @test_throws ArgumentError _CLI._cross_lineage_traits(["zz"], "r", p, names)
        @test_throws ArgumentError _CLI._cross_lineage_traits([1, "h1"], "r", p, names)
    end
end
