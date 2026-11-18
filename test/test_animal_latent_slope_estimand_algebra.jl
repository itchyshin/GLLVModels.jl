# C2 row namespace/export/animal_latent (UNBOUND, signed 2026-10-05 "leave unbound").
# Pure algebra plus a surface guard. NO fit, no R, no tolerance, no parity claim.
# Pins (1) the R random-slope covariance the row would have to match and (2) why the
# existing Julia Hadamard wrapper is a different estimand, so a later fitter is aligned
# term by term. The surface guard fails if a same-named export or a slope argument appears
# without a ruling (the 10-05 signature must then be revisited).
using GLLVModels
using Test
using LinearAlgebra
using Random

# Contract (docs/dev-log/core070/slopes-input-contract.md):
#   Cov(eta_o, eta_v) = C[g_o, g_v] * (U0[t_o, t_v] + x_o x_v U1[t_o, t_v]),  Uj = Lj Lj'
function _al_target_cov(C, L0, L1, g, t, x)
    U0, U1 = L0 * L0', L1 * L1'
    n = length(g)
    [C[g[o], g[v]] * (U0[t[o], t[v]] + x[o] * x[v] * U1[t[o], t[v]]) for o in 1:n, v in 1:n]
end

@testset "animal_latent slope estimand algebra (no fit, row stays unbound)" begin
    Random.seed!(1)
    ns, p, d = 6, 3, 2                       # fixture shape: 6 species, 3 traits, d = 2
    C = Matrix(0.7 * I(ns) + 0.3 * ones(ns, ns))
    L0, L1 = randn(p, d), randn(p, d)
    g = repeat(1:ns, inner = p * 3); t = repeat(1:p, outer = ns * 3); x = randn(length(g))
    V = _al_target_cov(C, L0, L1, g, t, x)
    V0 = _al_target_cov(C, L0, zeros(p, d), g, t, x)
    V1 = _al_target_cov(C, zeros(p, d), L1, g, t, x)

    @test issymmetric(V)
    @test minimum(eigvals(Symmetric(V))) > -1e-10                 # PSD
    @test V ≈ V0 + V1                                             # no intercept-slope cross term
    @test rank(L0 * L0') == d && rank(L1 * L1') == d              # per-block rank d (d <= p)
    # A slope block that is switched off (x == 0) leaves only the intercept covariance.
    @test _al_target_cov(C, L0, L1, g, t, zeros(length(g))) ≈ V0

    # Kronecker target vs the Julia Hadamard wrapper, at n_ind = p_traits = 3.
    A = Matrix(C[1:p, 1:p]); U = L0 * L0'
    K = kron(A, U)                            # index (i-1)*p + t  =>  K[(i,t),(j,s)] = A[i,j] * U[t,s]
    H = U .* A                                # fit_animal_latent_gllvm: (Lambda Lambda') .* A on individuals
    @test size(K) == (p * p, p * p) && size(H) == (p, p)
    @test K[(2 - 1) * p + 1, (3 - 1) * p + 2] ≈ A[2, 3] * U[1, 2]
    # H is only the "trait index tied to individual index" sub-block of K ...
    idx = [(i - 1) * p + i for i in 1:p]
    @test H ≈ K[idx, idx]
    # ... so K carries cross-trait, cross-individual structure that H cannot represent.
    @test norm(K[(1 - 1) * p + 1, (2 - 1) * p + 3] - H[1, 2]) > 1e-3
    @test rank(K) == rank(U) * rank(A)        # Kronecker rank rule: rank(U) = d here

    # Wide `1 + x | g` and long `0 + trait + (0 + trait):x | g` share the same per-row design.
    Xl = hcat([Float64(t[i] == k) for i in eachindex(t), k in 1:p],
              [Float64(t[i] == k) * x[i] for i in eachindex(t), k in 1:p])
    @test all(Xl[i, t[i]] == 1 && Xl[i, p + t[i]] == x[i] for i in eachindex(t))
end

@testset "animal_latent surface guard (revisit the 2026-10-05 signature if this fails)" begin
    # No Julia symbol with R's name: a registration-tier receipt would have no same-name target.
    @test !isdefined(GLLVModels, :animal_latent)
    # The wrapper stays the Hadamard row-relatedness fitter: (Y, A, d), no slope argument.
    ms = methods(fit_animal_latent_gllvm)
    @test !isempty(ms)
    for m in ms
        argnames = String.(Base.method_argnames(m)[2:end])
        @test !any(a -> a in ("x", "X", "covariate", "formula", "slope"), argnames)
    end
    # R's d <= n_traits rule is NOT what the wrapper enforces: it checks d <= p rows (individuals).
    Y = randn(4, 10); A = Matrix(1.0I, 4, 4)
    @test_throws ArgumentError fit_animal_latent_gllvm(Y, A, 5)   # 5 > p = 4 individuals
    @test_throws ArgumentError fit_animal_latent_gllvm(Y, A, 2; unique = true)
end
