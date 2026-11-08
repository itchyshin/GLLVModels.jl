"""Controls for the exact K = 1 derivative prototype (core070_postcoef_exact_k1).

The reference is an independent mpmath transcription of the Julia functions
(src/likelihood.jl gaussian_marginal_loglik K = 1 Woodbury path; src/profile.jl
gaussian_profile_nll, profile_beta=false), differentiated numerically at high
precision. It shares no code with the formulas under test. The "recorded"
binary64 values are a numpy float64 stand-in, not ForwardDiff output.
"""
import math
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

import mpmath as mp
import numpy as np

import core070_postcoef_exact_k1 as ex
import core070_postcoef_fit_health as health

mp.mp.dps = 60


# ---------------------------------------------------------------------------
# Synthetic data (Julia layout: Y p x n, X p x n x q)
# ---------------------------------------------------------------------------
def make_case(p=4, n=12, q=0, seed=1, shift=0.0, scale=1.0):
    rng = random.Random(seed)
    lam = [rng.gauss(0, 1) for _ in range(p)]
    Y = [[0.0] * n for _ in range(p)]
    X = None if q == 0 else [[[1.0 if j == 0 else rng.gauss(0, 1) for j in range(q)] for _ in range(n)]
                             for _ in range(p)]
    for s in range(n):
        z = rng.gauss(0, 1)
        for t in range(p):
            Y[t][s] = scale * (lam[t] * z + 0.6 * rng.gauss(0, 1)) + shift
    beta = [rng.gauss(0, 1) for _ in range(q)]
    return (Y, X, q) + bound_point(beta, [scale * v for v in lam], math.exp(math.log(0.6 * scale) + 0.03))


def bound_point(beta, lam_target, sig):
    """(theta, x) exactly as the writer derives them: L = target/sigma, lambda = fl(sigma*L),
    l = fl(log sigma). sigma is stepped up by ulps until check_binding accepts (math.log is not
    always correctly rounded)."""
    for _ in range(64):
        L = [v / sig for v in lam_target]
        theta = list(beta) + [math.log(sig)] + [sig * v for v in L]
        x = list(beta) + L
        try:
            ex.check_binding(theta, x, len(beta))
            return theta, x
        except ex.ExactInputError:
            sig = math.nextafter(sig, math.inf)
    raise AssertionError('no bound point found')


def mp_packed_nll(Y, X, q):
    p, n = len(Y), len(Y[0])

    def f(*th):
        beta, ell, lam = th[:q], th[q], th[q + 1:]
        sig = mp.exp(ell)
        s2 = sig * sig
        dinv = 1 / s2
        AK = 1 + sum(l * l * dinv for l in lam)
        logdet = p * mp.log(s2) + mp.log(AK)
        quad = 0
        for s in range(n):
            r = [mp.mpf(Y[t][s]) - (sum(mp.mpf(X[t][s][j]) * beta[j] for j in range(q)) if q else 0)
                 for t in range(p)]
            dr = [dinv * v for v in r]
            z = sum(lam[t] * dr[t] for t in range(p)) / AK
            quad += sum(r[t] * (dr[t] - dinv * lam[t] * z) for t in range(p))
        return mp.mpf('0.5') * (n * p * mp.log(2 * mp.pi) + n * logdet + quad)
    return f


def mp_profile_nll(Y, X, q):
    p, n = len(Y), len(Y[0])

    def f(*x):
        beta, L = x[:q], x[q:]
        AK = 1 + sum(l * l for l in L)
        quad = 0
        for s in range(n):
            r = [mp.mpf(Y[t][s]) - (sum(mp.mpf(X[t][s][j]) * beta[j] for j in range(q)) if q else 0)
                 for t in range(p)]
            z = sum(L[t] * r[t] for t in range(p)) / AK
            quad += sum(r[t] * (r[t] - L[t] * z) for t in range(p))
        N = n * p
        return mp.mpf('0.5') * (N * mp.log(2 * mp.pi) + N + N * mp.log(quad / N) + n * mp.log(AK))
    return f


def mp_grad(f, pt):
    pt = [mp.mpf(v) for v in pt]
    k = len(pt)
    return [mp.diff(f, pt, tuple(1 if i == j else 0 for j in range(k))) for i in range(k)]


def mp_hess(f, pt):
    pt = [mp.mpf(v) for v in pt]
    k = len(pt)
    H = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            o = [0] * k
            o[i] += 1
            o[j] += 1
            H[i][j] = H[j][i] = mp.diff(f, pt, tuple(o))
    return H


def float_recorded(Y, X, q, theta):
    """numpy float64 stand-in for the recorded Hessian and packed gradient (closed form)."""
    Yn = np.array(Y).T  # n x p
    p = Yn.shape[1]
    Xn = None if q == 0 else np.transpose(np.array(X), (1, 0, 2))  # n x p x q
    b, l, lam = np.array(theta[:q]), theta[q], np.array(theta[q + 1:])
    s = math.exp(l) ** 2
    c = s + lam @ lam
    P = (np.eye(p) - np.outer(lam, lam) / c) / s
    R = Yn - (Xn @ b if q else 0)
    PR = R @ P
    E = np.eye(p)
    sig = [2 * s * E] + [np.outer(E[i], lam) + np.outer(lam, E[i]) for i in range(p)]

    def s2(a, b_):
        if a == 0 and b_ == 0:
            return 4 * s * E
        if a == 0 or b_ == 0:
            return 0 * E
        return np.outer(E[a - 1], E[b_ - 1]) + np.outer(E[b_ - 1], E[a - 1])
    n = Yn.shape[0]
    k = q + 1 + p
    g = np.zeros(k)
    H = np.zeros((k, k))
    if q:
        g[:q] = -np.einsum('sij,si->j', Xn, PR)
        H[:q, :q] = np.einsum('sia,ij,sjb->ab', Xn, P, Xn)
    for a in range(p + 1):
        g[q + a] = 0.5 * (n * np.trace(P @ sig[a]) - np.einsum('si,ij,sj->', PR, sig[a], PR))
        if q:
            v = np.einsum('sia,ij,sj->a', Xn, P @ sig[a], PR)
            H[:q, q + a] = v
            H[q + a, :q] = v
        for b_ in range(p + 1):
            H[q + a, q + b_] = (0.5 * n * (np.trace(P @ s2(a, b_)) - np.trace(P @ sig[b_] @ P @ sig[a]))
                                + np.einsum('si,ij,sj->', PR, sig[b_] @ P @ sig[a], PR)
                                - 0.5 * np.einsum('si,ij,sj->', PR, s2(a, b_), PR))
    return [float(v) for v in g], [[float(v) for v in row] for row in H]


def mid(iv):
    return (iv.lo + iv.hi) / 2


def mpf(fr):
    return mp.mpf(fr.numerator) / fr.denominator


CASES = {
    'no_X_p4': dict(p=4, n=12, q=0, seed=1),
    'intercept_cov_p3': dict(p=3, n=10, q=2, seed=7),
    'shift_1e4_p3': dict(p=3, n=15, q=0, seed=3, shift=1e4),
}


class ExpEnclosure(unittest.TestCase):
    def test_contains_high_precision_exp(self):
        with mp.workdps(120):
            for t in [0.0, 1e-300, -0.2557, 0.5, -3.75, 17.25, -700.5, 1.3e2]:
                e = ex.exp_enclosure(Fraction(t))
                ref = mp.exp(mp.mpf(t))
                self.assertTrue(mpf(e.lo) <= ref <= mpf(e.hi), t)
                self.assertLess((e.hi - e.lo) / e.lo, Fraction(1, 2 ** 200), t)

    def test_negative_control_neighbour_excluded(self):
        # The enclosure of exp(t) must not contain exp(next binary64 after t).
        with mp.workdps(120):
            for t in [-0.2557, 0.75, 9.0]:
                e = ex.exp_enclosure(Fraction(t))
                ref = mp.exp(mp.mpf(math.nextafter(t, math.inf)))
                self.assertFalse(mpf(e.lo) <= ref <= mpf(e.hi), t)

    def test_remainder_term_is_load_bearing(self):
        # With outward rounding disabled the enclosure is [S - R, S + R] exactly; it must
        # still contain the irrational exp(t), so a zero remainder R would fail here.
        ident = lambda v, bits=None: v
        with patch.object(ex, '_floor_dyadic', ident), patch.object(ex, '_ceil_dyadic', ident), \
                mp.workdps(400):
            for t in [0.3, -0.45, 3.0]:
                e = ex.exp_enclosure(Fraction(t))
                ref = mp.exp(mp.mpf(t))
                self.assertTrue(mpf(e.lo) < ref < mpf(e.hi), t)

    def test_out_of_range_refused(self):
        with self.assertRaises(ex.ExactInputError):
            ex.exp_enclosure(Fraction(1500))


class IntervalArithmetic(unittest.TestCase):
    def test_outward_rounding_contains_exact_results(self):
        rng = random.Random(11)
        for _ in range(300):
            a = Fraction(rng.uniform(-5, 5)) / Fraction(3 ** rng.randint(0, 40))
            b = Fraction(rng.uniform(0.1, 5)) * Fraction(7 ** rng.randint(0, 40))
            A, B = ex.Iv(a), ex.Iv(b)
            for op, exact in ((A + B, a + b), (A - B, a - b), (A * B, a * b), (A / B, a / b)):
                self.assertTrue(op.contains(exact))

    def test_wide_intervals_enclose_every_pointwise_result(self):
        rng = random.Random(5)
        for _ in range(200):
            a, b = sorted(Fraction(rng.uniform(-3, 3)) for _ in range(2))
            c, d = sorted(Fraction(rng.uniform(-3, 3)) for _ in range(2))
            A, B = ex.Iv(a, b), ex.Iv(c, d)
            for u in (a, b, (a + b) / 2):
                for v in (c, d, (c + d) / 2):
                    self.assertTrue((A * B).contains(u * v))
                    self.assertTrue((A + B).contains(u + v) and (A - B).contains(u - v))

    def test_radius_about_uses_both_endpoints(self):
        self.assertEqual(ex.Iv(1, 3).radius_about(1), 2)
        self.assertEqual(ex.Iv(1, 3).radius_about(3), 2)
        self.assertEqual(ex.Iv(1, 3).radius_about(Fraction(5, 2)), Fraction(3, 2))

    def test_division_by_interval_containing_zero_refused(self):
        with self.assertRaises(ZeroDivisionError):
            ex.Iv(1) / ex.Iv(-1, 1)


class ExactOptimizerGradient(unittest.TestCase):
    def check(self, name):
        Y, X, q, theta, x = make_case(**CASES[name])
        g = ex.exact_optimizer_gradient(Y, X, x, q)
        ref = mp_grad(mp_profile_nll(Y, X, q), x)
        scale = max(1, max(abs(v) for v in ref))
        err = max(abs(mpf(a) - b) for a, b in zip(g, ref))
        self.assertLess(err, mp.mpf('1e-25') * scale, name)
        return Y, X, q, x, ref, scale

    def test_matches_independent_reference(self):
        for name in CASES:
            self.check(name)

    def test_returns_exact_fractions(self):
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        self.assertTrue(all(type(v) is Fraction for v in ex.exact_optimizer_gradient(Y, X, x, q)))

    def test_negative_control_altered_data_detected(self):
        Y, X, q, x, ref, scale = self.check('intercept_cov_p3')
        Y2 = [row[:] for row in Y]
        Y2[1][4] = math.nextafter(Y2[1][4], math.inf) + 1e-9
        g2 = ex.exact_optimizer_gradient(Y2, X, x, q)
        self.assertGreater(max(abs(mpf(a) - b) for a, b in zip(g2, ref)), mp.mpf('1e-15') * scale)

    def test_negative_control_mutated_formula_detected(self):
        # Drop the n L_i / a term by mutating the transpose helper used for beta only
        # is too narrow; mutate the shared trace helper instead.
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        ref = mp_grad(mp_profile_nll(Y, X, q), x)
        real = ex._tr_prod
        with patch.object(ex, '_tr_prod', lambda A, B: real(A, B) * Fraction(1001, 1000)):
            g = ex.exact_optimizer_gradient(Y, X, x, q)
        self.assertGreater(max(abs(mpf(a) - b) for a, b in zip(g, ref)), mp.mpf('1e-6'))


class PackedEnclosure(unittest.TestCase):
    def enclose(self, name):
        Y, X, q, theta, x = make_case(**CASES[name])
        g, H = ex.packed_enclosure(Y, X, theta, q)
        f = mp_packed_nll(Y, X, q)
        return Y, X, q, theta, g, H, mp_grad(f, theta), mp_hess(f, theta)

    def test_enclosure_matches_reference_and_is_narrow(self):
        for name in CASES:
            Y, X, q, theta, g, H, gref, Href = self.enclose(name)
            k = len(theta)
            scale_g = max(1, max(abs(v) for v in gref))
            scale_H = max(1, max(abs(v) for row in Href for v in row))
            for i in range(k):
                self.assertLess(abs(mpf(mid(g[i])) - gref[i]), mp.mpf('1e-22') * scale_g, (name, i))
                self.assertLess(mpf(g[i].hi - g[i].lo), mp.mpf('1e-50') * scale_g, (name, i))
                for j in range(k):
                    self.assertLess(abs(mpf(mid(H[i][j])) - Href[i][j]), mp.mpf('1e-20') * scale_H, (name, i, j))
                    self.assertLess(mpf(H[i][j].hi - H[i][j].lo), mp.mpf('1e-50') * scale_H, (name, i, j))

    def test_enclosure_is_symmetric(self):
        Y, X, q, theta, x = make_case(**CASES['intercept_cov_p3'])
        g, H = ex.packed_enclosure(Y, X, theta, q)
        k = len(theta)
        for i in range(k):
            for j in range(k):
                self.assertTrue(H[i][j].lo <= H[j][i].hi and H[j][i].lo <= H[i][j].hi)

    def test_negative_control_altered_log_sigma_detected(self):
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        f = mp_packed_nll(Y, X, q)
        Href = mp_hess(f, theta)
        t2 = theta[:]
        t2[q] += 1e-6
        _, H2 = ex.packed_enclosure(Y, X, t2, q)
        self.assertGreater(max(abs(mpf(mid(H2[i][j])) - Href[i][j]) for i in range(len(theta)) for j in range(len(theta))),
                           mp.mpf('1e-9'))

    def test_negative_control_mutated_formula_detected(self):
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        Href = mp_hess(mp_packed_nll(Y, X, q), theta)
        # Mutation: every trace term scaled by 0.999 (a wrong-constant formula).
        with patch.object(ex, '_tr_prod', lambda A, B: ex.Iv.lift(real_tr(A, B)) * Fraction(999, 1000)):
            _, H2 = ex.packed_enclosure(Y, X, theta, q)
        self.assertGreater(max(abs(mpf(mid(H2[i][j])) - Href[i][j]) for i in range(len(theta)) for j in range(len(theta))),
                           mp.mpf('1e-6'))


real_tr = ex._tr_prod


class ComputedRadii(unittest.TestCase):
    def radii(self, name, perturb=None):
        Y, X, q, theta, x = make_case(**CASES[name])
        g_rec, H_rec = float_recorded(Y, X, q, theta)
        if perturb:
            perturb(H_rec, g_rec)
        g_opt_rec = [float(v) for v in ex.exact_optimizer_gradient(Y, X, x, q)]
        return ex.computed_radii(Y, X, q, theta, x, H_rec, g_rec, g_opt_rec), (Y, X, q, theta, x, H_rec, g_opt_rec)

    def test_radius_bounds_true_error(self):
        for name in CASES:
            r, (Y, X, q, theta, x, H_rec, g_opt_rec) = self.radii(name)
            self.assertEqual(r['radius_status'], ex.STATUS_COMPUTED)
            Href = mp_hess(mp_packed_nll(Y, X, q), theta)
            true_err = max(abs(mp.mpf(H_rec[i][j]) - Href[i][j]) for i in range(len(theta)) for j in range(len(theta)))
            # The radius is rigorous; the mp reference carries ~1e-20 relative diff error.
            slack = mp.mpf('1e-18') * max(1, max(abs(v) for row in Href for v in row))
            self.assertGreaterEqual(mpf(r['hessian_radius']) + slack, true_err, name)
            # The recorded optimizer gradient is the exact one rounded to binary64, so its
            # exact error is at most half an ulp of the largest entry, and is computed exactly.
            self.assertLessEqual(r['optimizer_gradient_radius'], max(Fraction(math.ulp(v)) for v in g_opt_rec) / 2)

    def test_negative_control_perturbed_recorded_hessian(self):
        delta = 1e-3
        def bump(H, g):
            H[0][0] += delta
        r, _ = self.radii('no_X_p4', bump)
        self.assertGreaterEqual(r['hessian_radius'], Fraction(delta) * Fraction(999, 1000))

    def test_negative_control_perturbed_recorded_gradient(self):
        def bump(H, g):
            g[1] += 1e-4
        r, _ = self.radii('no_X_p4', bump)
        self.assertGreaterEqual(r['packed_gradient_radius'], Fraction(1, 10 ** 4) * Fraction(999, 1000))

    def test_large_mean_shift_radius_vs_declared(self):
        # Diagnostic: compare the computed radius with the writer's declared radius
        # 2^10 * 2^-52 * max(1, max|H|) on the float64 stand-in. Recorded, not asserted.
        r, (Y, X, q, theta, x, H_rec, _) = self.radii('shift_1e4_p3')
        declared = Fraction(2 ** 10, 2 ** 52) * max(1, max(abs(Fraction(v)) for row in H_rec for v in row))
        self.__class__.shift_ratio = float(r['hessian_radius'] / declared)
        self.assertGreater(r['hessian_radius'], 0)

    def test_refuses_bad_inputs(self):
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        bad = [row[:] for row in Y]
        bad[0][0] = float('nan')
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(bad, X, theta, q)
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient(Y, X, x + [0.0], q)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, [[[1.0]]], theta, 0)
        ints = [[1 for _ in row] for row in Y]
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient(ints, X, x, q)


class LiveCertifierReadOnly(unittest.TestCase):
    """Calls the live certifier functions read-only; nothing is wired or patched."""

    def test_computed_radius_against_representation_floor(self):
        Y, X, q, theta, x = make_case(**CASES['no_X_p4'])
        g_rec, H_rec = float_recorded(Y, X, q, theta)
        H_sym = [[(H_rec[i][j] + H_rec[j][i]) / 2 for j in range(len(theta))] for i in range(len(theta))]
        g_opt = [float(v) for v in ex.exact_optimizer_gradient(Y, X, x, q)]
        r = ex.computed_radii(Y, X, q, theta, x, H_sym, g_rec, g_opt)
        radius = ex.round_up_binary64(r['hessian_radius'])
        k = len(theta)
        floor = k * health.CERT_EPS * max(abs(Fraction(v)) for row in H_sym for v in row)
        out = health.certify_hessian(H_sym, list(range(1, k + 1)), health.HESSIAN_COORDINATE, radius,
                                     0.0, [], parameter_count=k)
        if Fraction(radius) < floor:
            # Design finding: a computed radius can sit below the declared-radius floor,
            # which the live certifier then refuses. Recorded in the design note.
            self.assertEqual(out['verdict'], health.CERT_INCOMPLETE)
            self.assertIn('representation floor', out['reason'])
        else:
            self.assertIn(out['verdict'], (health.CERT_PASS, health.CERT_FAIL, health.CERT_INCOMPLETE))
        self.__class__.floor_info = (float(radius), float(floor), out['verdict'], out['reason'])

    def test_flag_untouched(self):
        self.assertIs(health.NUMERICAL_ADMISSION_IMPLEMENTED, False)
        self.assertEqual(health.RADIUS_STATUS, 'declared_unproven')


if __name__ == '__main__':
    unittest.main()
