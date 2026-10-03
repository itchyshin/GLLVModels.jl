# #573: fit_zip_gllvm / fit_zinb_gllvm stalled below the nested Poisson / NB2 optimum.
#
# ZIP nests Poisson and ZINB nests the shared-r NB2 (π → 0), so a ZI fit must never end
# below the nested fit. From `_zi_warmstart` the ZI surface has a second basin where
# structural zeros absorb the zeros the latent factors explain, and L-BFGS settled there
# (mvabund::spider, 12 × 28, K = 2: ZIP −888.23 vs Poisson −845.69). The fixtures are
# species subsets of mvabund::spider$abund (mvabund 4.2.8; data reproduced in #573).
# On unmodified main: ZIP on species [1,2,3,6,8,10] (K = 2) ends at −499.03 against
# Poisson −452.40; ZINB on species 1–6 (K = 2) ends at −303.10 against NB2 −289.87.
using Test, GLLVModels, Optim
const _G573 = GLLVModels

# mvabund::spider$abund, transposed to species × sites (12 × 28).
const _SPIDER573 = Float64[
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    0 0 2 0 0 0 0 0 0 1 2 0 1 0 0 0 0 0 0 0 0 16 15 20 9 6 11 14;
    0 0 2 1 2 6 12 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 7 5 0 18 4 1;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    0 1 1 1 1 0 1 55 0 0 0 0 1 3 6 6 2 5 12 13 16 0 2 0 1 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    12 15 18 29 135 27 89 2 1 0 0 1 53 15 0 2 0 0 1 0 0 0 0 0 6 0 0 0;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2;
    4 9 1 25 17 34 16 3 0 0 0 0 22 32 3 4 2 0 3 2 2 0 0 0 6 0 0 0]

@testset "#573 ZI fits never end below the nested count fit" begin
    @testset "ZIP >= Poisson where main stalled (species 1,2,3,6,8,10; K = 2)" begin
        Y = _SPIDER573[[1, 2, 3, 6, 8, 10], :]
        pf = _G573.fit_poisson_gllvm(Y; K = 2)
        zf = _G573.fit_zip_gllvm(Y; K = 2)
        @test isfinite(zf.loglik)
        @test zf.loglik >= pf.loglik - 1e-6
    end

    @testset "ZINB >= shared-r NB2 where main stalled (species 1-6; K = 2)" begin
        Y = _SPIDER573[1:6, :]
        nb = _G573.fit_nb_gllvm(Y; K = 2)
        zf = _G573.fit_zinb_gllvm(Y; K = 2)
        @test isfinite(zf.loglik)
        @test zf.loglik >= nb.loglik - 1e-6
    end

    @testset "healthy control: fits already above the nested fit are unchanged" begin
        # Species 2,4,5,9,11,12 (K = 1): the warm-start fits already end above the
        # nested fits on main (ZIP +0.26, ZINB +0.23), so the guard must not fire.
        Y = _SPIDER573[[2, 4, 5, 9, 11, 12], :]
        p, K = size(Y, 1), 1
        rr = _G573.rr_theta_len(p, K)
        negll(θ) = begin
            v = try
                -_G573.zip_marginal_loglik_laplace(Y, _G573.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K),
                                                   θ[1:p], θ[(p + 1):(2p)])
            catch
                return 1e12
            end
            isfinite(v) ? v : 1e12
        end
        βz0, βc0, Λc0 = _G573._zi_warmstart(Y, K)
        ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
        res = Optim.optimize(negll, vcat(βz0, βc0, _G573.pack_lambda(Λc0)), ls,
                             Optim.Options(g_tol = 1e-5, iterations = 500); autodiff = :finite)
        zf = _G573.fit_zip_gllvm(Y; K = K)
        pf = _G573.fit_poisson_gllvm(Y; K = K)
        @test zf.loglik >= pf.loglik - 1e-6
        # Bit-identical to the unguarded warm-start fit (the pre-#573 path).
        @test zf.loglik == -Optim.minimum(res)
        @test vcat(zf.βz, zf.βc) == Optim.minimizer(res)[1:(2p)]
        # The guard returns its input untouched when the fit is at/above the nested fit.
        @test _G573._zi_nested_guard(negll, res, -Optim.minimum(res) - 1.0, zeros(2p + rr),
                                     1e-5, 10) === res
        zb = _G573.fit_zinb_gllvm(Y; K = K)
        nb = _G573.fit_nb_gllvm(Y; K = K)
        @test zb.loglik >= nb.loglik - 1e-6
    end
end
