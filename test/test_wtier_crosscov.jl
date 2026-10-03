using GLLVModels, Test, LinearAlgebra

# Issue #135: the W-tier reduced-rank term (Λ_W) must carry the full
# cross-trait covariance Λ_W Λ_Wᵀ, as the C++ twin does
# (gllvmTMB.cpp: eta(o) += sum_k Lambda_W(t, k) * z_W(k, ss), with z_W shared by
# every trait of one unit). Every expected value below comes from a dense
# multivariate-normal density written out by hand, or from the R twin.

# Dense reference: sum over columns of the MVN(0, Σ) log density of R[:, s].
function _wtier_ref_loglik(R::AbstractMatrix, Σ::AbstractMatrix)
    p, n = size(R)
    Σs = Symmetric(Matrix{Float64}(Σ))
    quad = sum(dot(R[:, s], Σs \ R[:, s]) for s in 1:n)
    return -0.5 * (n * p * log(2π) + n * logdet(Σs) + quad)
end

# Dense reference for the phylogenetic layout: vec(R) ~ MVN(0, I_n ⊗ A + J_n ⊗ B).
function _wtier_ref_loglik_phy(R::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    p, n = size(R)
    Σfull = Symmetric(kron(Matrix(1.0I, n, n), A) .+ kron(ones(n, n), B))
    r = vec(R)
    return -0.5 * (n * p * log(2π) + logdet(Σfull) + dot(r, Σfull \ r))
end

@testset "#135 W-tier rr term carries the C++ cross-trait covariance" begin
    p, n = 4, 6
    y = [sin(0.9 * t + 1.7 * s) + 0.3 * cos(2.1 * t * s) for t in 1:p, s in 1:n]
    Λ_B = reshape([0.8, -0.4, 0.5, 0.3], p, 1)
    Λ_W = [0.6 0.0; 0.3 0.45; -0.35 0.25; 0.2 -0.3]     # lower-triangular, p × 2
    σ_eps = 0.6
    σ²_B = [0.10, 0.20, 0.05, 0.15]
    σ²_W = [0.07, 0.03, 0.12, 0.09]
    X = Array{Float64}(undef, p, n, 2)
    for t in 1:p, s in 1:n
        X[t, s, 1] = 1.0
        X[t, s, 2] = 0.1 * (t - s)
    end
    β = [0.2, -0.5]
    R = y .- [sum(X[t, s, k] * β[k] for k in 1:2) for t in 1:p, s in 1:n]

    # The C++ per-unit covariance, with the off-diagonal Λ_W Λ_Wᵀ terms.
    Σ_cpp = Λ_B * Λ_B' + Λ_W * Λ_W' + Diagonal(σ²_B .+ σ²_W .+ σ_eps^2)
    # The old diagonal-only covariance, used only to show the case discriminates.
    Σ_diag = Λ_B * Λ_B' + Diagonal(vec(sum(Λ_W .^ 2; dims = 2)) .+ σ²_B .+ σ²_W .+ σ_eps^2)
    @test diag(Σ_cpp) ≈ diag(Σ_diag)
    @test abs(_wtier_ref_loglik(R, Σ_cpp) - _wtier_ref_loglik(R, Σ_diag)) > 1e-2

    # Phylogenetic pieces. Σ_phy for ((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2), written
    # by hand: Σ_phy[i, j] = σ²_phy * (shared root-to-MRCA path length).
    σ²_phy = 0.9
    Σ_phy = σ²_phy .* [0.7 0.4 0.0 0.0;
                       0.4 0.6 0.0 0.0;
                       0.0 0.0 0.7 0.2;
                       0.0 0.0 0.2 0.3]
    Λ_phy = reshape([0.4, 0.2, -0.3, 0.5], p, 1)
    σ_phy = [0.3, -0.2, 0.25, 0.1]
    B_phy = (hcat(Λ_phy, σ_phy) * hcat(Λ_phy, σ_phy)') .* Σ_phy

    @testset "marginal log-likelihood equals the dense C++ reference" begin
        ll = GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; X = X, β = β,
                                                 Λ_W = Λ_W, σ²_B = σ²_B, σ²_W = σ²_W)
        @test ll ≈ _wtier_ref_loglik(R, Σ_cpp) atol = 1e-8

        # Λ_W alone (no diagonal random effects, no X).
        Σ_noD = Λ_B * Λ_B' + Λ_W * Λ_W' + σ_eps^2 * I
        @test GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; Λ_W = Λ_W) ≈
              _wtier_ref_loglik(y, Σ_noD) atol = 1e-8

        # Packed driver (log σ scale for σ_eps, σ_B, σ_W).
        spec = (q = 2, p = p, K_B = 1, K_W = 2, has_diag = true)
        params = vcat(β, log(σ_eps), 0.5 .* log.(σ²_B), 0.5 .* log.(σ²_W),
                      GLLVModels.pack_lambda(Λ_B), GLLVModels.pack_lambda(Λ_W))
        @test -GLLVModels.gaussian_nll_packed(params, y; spec = spec, X = X) ≈
              _wtier_ref_loglik(R, Σ_cpp) atol = 1e-8
    end

    @testset "phylogenetic paths use the same within-unit covariance" begin
        ref = _wtier_ref_loglik_phy(R, Σ_cpp, B_phy)
        ll_dense = GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; X = X, β = β,
                       Λ_W = Λ_W, σ²_B = σ²_B, σ²_W = σ²_W,
                       Λ_phy = Λ_phy, σ_phy = σ_phy, Σ_phy = Σ_phy)
        @test ll_dense ≈ ref atol = 1e-8
        phy = augmented_phy("((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2);")
        ll_sparse = gaussian_marginal_loglik_sparse_phy(y, Λ_B, σ_eps; X = X, β = β,
                        Λ_W = Λ_W, σ²_B = σ²_B, σ²_W = σ²_W,
                        Λ_phy = Λ_phy, σ_phy = σ_phy, phy = phy, σ²_phy = σ²_phy)
        @test ll_sparse ≈ ref atol = 1e-8
    end

    @testset "profiled objective (the fitting path) uses the full covariance" begin
        # Profile parameterisation: Λ = σ_eps L, σ² = σ_eps² τ. The profiled σ̂²_eps
        # and the profile log-likelihood are computed here from a dense Ã.
        L_B = Λ_B ./ σ_eps
        L_W = Λ_W ./ σ_eps
        τ_B = σ²_B ./ σ_eps^2
        τ_W = σ²_W ./ σ_eps^2
        Ã = L_B * L_B' + L_W * L_W' + Diagonal(τ_B .+ τ_W .+ 1.0)
        σ̂² = sum(dot(y[:, s], Symmetric(Ã) \ y[:, s]) for s in 1:n) / (n * p)
        ref = _wtier_ref_loglik(y, σ̂² .* Ã)

        spec = (q = 0, p = p, K_B = 1, K_W = 2, has_diag = true)
        params = vcat(0.5 .* log.(τ_B), 0.5 .* log.(τ_W),
                      GLLVModels.pack_lambda(L_B), GLLVModels.pack_lambda(L_W))
        @test -GLLVModels.gaussian_profile_nll(params, y; spec = spec) ≈ ref atol = 1e-8
        rec = GLLVModels.profile_recover(params, y; spec = spec)
        @test rec.logLik ≈ ref atol = 1e-8
        @test rec.σ_eps^2 ≈ σ̂² rtol = 1e-10

        # Phylogenetic profile path (β stays a parameter there; none here).
        ρ_phy = σ_phy ./ σ_eps
        L_phy = Λ_phy ./ σ_eps
        B̃ = (hcat(L_phy, ρ_phy) * hcat(L_phy, ρ_phy)') .* Σ_phy
        Σ̃full = Symmetric(kron(Matrix(1.0I, n, n), Ã) .+ kron(ones(n, n), B̃))
        σ̂²_phy = dot(vec(y), Σ̃full \ vec(y)) / (n * p)
        ref_phy = _wtier_ref_loglik_phy(y, σ̂²_phy .* Ã, σ̂²_phy .* B̃)
        spec_phy = (q = 0, p = p, K_B = 1, K_W = 2, has_diag = true,
                    K_phy = 1, has_phy_unique = true)
        params_phy = vcat(params, ρ_phy, GLLVModels.pack_lambda(L_phy))
        @test -GLLVModels.gaussian_profile_nll(params_phy, y; spec = spec_phy,
                                               Σ_phy = Σ_phy) ≈ ref_phy atol = 1e-8
        @test GLLVModels.profile_recover(params_phy, y; spec = spec_phy,
                                         Σ_phy = Σ_phy).logLik ≈ ref_phy atol = 1e-8
    end

    @testset "REML uses the full covariance" begin
        Σi = inv(Symmetric(Σ_cpp))
        M = sum(X[:, s, :]' * Σi * X[:, s, :] for s in 1:n)
        v = sum(X[:, s, :]' * Σi * y[:, s] for s in 1:n)
        β̂ = M \ v
        R̂ = y .- [sum(X[t, s, k] * β̂[k] for k in 1:2) for t in 1:p, s in 1:n]
        ref = _wtier_ref_loglik(R̂, Σ_cpp) + 1.0 * log(2π) - 0.5 * logdet(M)
        @test gaussian_reml_loglik(y, X, Λ_B, σ_eps; Λ_W = Λ_W, σ²_B = σ²_B,
                                   σ²_W = σ²_W) ≈ ref atol = 1e-8
    end

    @testset "R twin objective at a fixed point (gllvmTMB, one unit per site)" begin
        # test/fixtures/gen_wtier_crosscov_twin.R: TMB objective of
        # value ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE) +
        #         latent(0 + trait | site_species, d = 2, unique = FALSE)
        # with one site_species per site, at the parameter point below.
        R_TMB_NLL = 29.240731652199255
        p3, n3 = 3, 8
        yw = [sin(0.9 * t + 1.7 * s) + 0.3 * cos(2.1 * t * s) for t in 1:p3, s in 1:n3]
        alpha = [0.2, -0.1, 0.05]
        Λ_B3 = reshape([0.8, -0.4, 0.5], p3, 1)
        Λ_W3 = GLLVModels.unpack_lambda([0.6, 0.45, 0.3, -0.35, 0.25], p3, 2)
        @test Λ_W3 == [0.6 0.0; 0.3 0.45; -0.35 0.25]
        ll = GLLVModels.gaussian_marginal_loglik(yw .- alpha, Λ_B3, 0.6; Λ_W = Λ_W3)
        @test -ll ≈ R_TMB_NLL atol = 1e-8
    end

    @testset "fitted model: logLik and site covariance match the dense reference" begin
        nf = 40
        yf = [sin(0.37 * t * s + 0.5 * t) + 0.5 * cos(1.3 * s) * (t - 2.5) / 2
              for t in 1:p, s in 1:nf]
        # A W-tier start with several non-zero rows, so Λ_W Λ_Wᵀ has real
        # off-diagonal entries at the fitted point.
        fit = fit_gaussian_gllvm(yf; K = 1, K_W = 1, has_diag = true,
                                 λ_W_init = reshape([0.3, 0.2, -0.2, 0.25], p, 1))
        ΛWWt = fit.pars.Λ_W * fit.pars.Λ_W'
        @test maximum(abs, ΛWWt - Diagonal(ΛWWt)) > 1e-3
        Σ_hat = fit.pars.Λ * fit.pars.Λ' + fit.pars.Λ_W * fit.pars.Λ_W' +
                Diagonal(fit.pars.σ²_B .+ fit.pars.σ²_W .+ fit.pars.σ_eps^2)
        @test fit.logLik ≈ _wtier_ref_loglik(yf, Σ_hat) atol = 1e-8
        @test sigma_y_site(fit) ≈ Σ_hat atol = 1e-10
        @test GLLVModels._bootstrap_site_cov(fit) ≈ Σ_hat atol = 1e-10
        @test GLLVModels._derived_site_cov(fit) ≈ Σ_hat atol = 1e-10
        @test correlation(fit) ≈ Σ_hat ./ sqrt.(diag(Σ_hat) * diag(Σ_hat)') atol = 1e-10
    end

    @testset "models without a W-tier rr term are unchanged (pinned before the fix)" begin
        # Values computed on the unmodified tree (origin/main 896a0a228) with the
        # same inputs; the fix must not move them.
        @test GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; X = X, β = β,
                  σ²_B = σ²_B, σ²_W = σ²_W) ≈ -31.947911378495462 rtol = 1e-12
        @test GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps; σ²_B = σ²_B, σ²_W = σ²_W,
                  Λ_phy = Λ_phy, σ_phy = σ_phy,
                  Σ_phy = sigma_phy_dense(augmented_phy("((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2);");
                                          σ²_phy = σ²_phy)) ≈ -31.299943364983754 rtol = 1e-12
        spec3 = (q = 0, p = p, K_B = 1, K_W = 0, has_diag = true)
        params3 = vcat(0.5 .* log.(σ²_B ./ σ_eps^2), 0.5 .* log.(σ²_W ./ σ_eps^2),
                       GLLVModels.pack_lambda(Λ_B ./ σ_eps))
        @test GLLVModels.gaussian_profile_nll(params3, y; spec = spec3) ≈
              30.388929959971016 rtol = 1e-12
        spec4 = (q = 0, p = p, K_B = 1, K_W = 0, has_diag = true, K_phy = 1,
                 has_phy_unique = true)
        params4 = vcat(params3, σ_phy ./ σ_eps, GLLVModels.pack_lambda(Λ_phy ./ σ_eps))
        @test GLLVModels.gaussian_profile_nll(params4, y; spec = spec4,
                  Σ_phy = sigma_phy_dense(augmented_phy("((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2);");
                                          σ²_phy = σ²_phy)) ≈ 31.29547476784621 rtol = 1e-12
        @test gaussian_marginal_loglik_sparse_phy(y, Λ_B, σ_eps; σ²_B = σ²_B, σ²_W = σ²_W,
                  Λ_phy = Λ_phy, σ_phy = σ_phy,
                  phy = augmented_phy("((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2);"),
                  σ²_phy = σ²_phy) ≈ -31.299943364983747 rtol = 1e-12
        @test gaussian_reml_loglik(y, X, Λ_B, σ_eps; σ²_B = σ²_B, σ²_W = σ²_W) ≈
              -30.425916489143344 rtol = 1e-12
        nf = 40
        yf = [sin(0.37 * t * s + 0.5 * t) + 0.5 * cos(1.3 * s) * (t - 2.5) / 2
              for t in 1:p, s in 1:nf]
        fit0 = fit_gaussian_gllvm(yf; K = 1, has_diag = true)
        @test fit0.logLik ≈ -179.40977230697754 rtol = 1e-8
    end
end
