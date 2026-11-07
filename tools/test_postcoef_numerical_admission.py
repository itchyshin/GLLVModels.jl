"""Controls for the exact/interval numerical-admission certifier (diagnostic)."""
import copy
import unittest
from fractions import Fraction
from unittest.mock import patch

import core070_postcoef_fit_health as health
from test_postcoef_fit_health import INDEFINITE, good, with_indefinite_julia_hessian

COORD = 'packed_free_index'
PASS, FAIL, INC = health.CERT_PASS, health.CERT_FAIL, health.CERT_INCOMPLETE


def certified_good():
    """good() plus the fields a certifiable Julia record must declare."""
    data = good()
    for row in data['Julia']:
        row.update(hessian_coordinate=COORD, hessian_error_radius=1e-9, gradient_error_radius=1e-9)
    return data


def enabled():
    return patch.object(health, 'NUMERICAL_ADMISSION_IMPLEMENTED', True)


def hess(matrix, indices, radius=1e-9, coordinate=COORD, **kw):
    return health.certify_hessian(matrix, indices, coordinate, radius, 0.0, **kw)


class CertifierTests(unittest.TestCase):
    def test_1_float64_cholesky_counterexample_is_rejected(self):
        self.assertIs(health.hessian_pd(INDEFINITE, [1, 2, 3], 3), True)  # the Float64 screen is fooled
        # Exact lambda_min is about -1e-16: a tiny radius certifies indefiniteness,
        # a radius above it cannot decide; neither passes.
        self.assertEqual(hess(INDEFINITE, [1, 2, 3], 1e-25)['verdict'], FAIL)
        self.assertEqual(hess(INDEFINITE, [1, 2, 3], 1e-12)['verdict'], INC)
        data = with_indefinite_julia_hessian(certified_good())
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            data['Julia'][0]['hessian_error_radius'] = 1e-25
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL')

    def test_2_pd_with_adequate_margin_is_certified(self):
        out = hess([[2.0, 0.5], [0.5, 1.0]], [1, 2])
        self.assertEqual(out['verdict'], PASS)
        self.assertEqual(out['coordinate'], COORD)
        data = certified_good()
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')  # hold flag still False

    def test_3_gradient_straddling_the_limit_is_not_a_pass(self):
        self.assertEqual(health.certify_gradient(1e-7, 1e-9), PASS)
        self.assertEqual(health.certify_gradient(9.9e-6, 1e-6), INC)
        self.assertEqual(health.certify_gradient(1.1e-5, 1e-6), INC)
        self.assertEqual(health.certify_gradient(2e-5, 1e-6), FAIL)
        for bad in (None, float('nan'), -1.0, True):
            self.assertEqual(health.certify_gradient(1e-7, bad), INC)
        data = certified_good()
        row = data['Julia'][0]
        row['optimizer']['optimizer_gradient'] = [9.9e-6, 0.0]
        row['optimizer']['optimizer_max_abs_gradient'] = 9.9e-6
        row['gradient_error_radius'] = 1e-6
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            del row['gradient_error_radius']
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_4_verdict_is_not_transferred_between_coordinates(self):
        # theta = phi^2 at phi = 1 (J = 2I): H_phi = J'HJ + diag(2 g). At a
        # non-stationary point the Hessian changes definiteness.
        h = [[2.0, 0.0], [0.0, 2.0]]
        g = [-10.0, -10.0]
        h_phi = [[4 * h[i][j] + (2 * g[i] if i == j else 0.0) for j in range(2)] for i in range(2)]
        self.assertEqual(hess(h, [1, 2])['verdict'], PASS)
        self.assertEqual(hess(h_phi, [1, 2])['verdict'], FAIL)  # same point, other coordinates
        for label in ('optimizer', 'natural', 'log_sd', None, ''):
            out = hess(h, [1, 2], coordinate=label)
            self.assertEqual(out['verdict'], INC)
            self.assertIn('coordinate', out['reason'])
        data = certified_good()
        data['Julia'][0]['hessian_coordinate'] = 'optimizer'
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            del data['Julia'][0]['hessian_coordinate']
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_5_error_larger_than_margin_is_incomplete(self):
        m = [[1e-3, 0.0], [0.0, 1e-3]]
        self.assertEqual(hess(m, [1, 2], 1e-6)['verdict'], PASS)
        out = hess(m, [1, 2], 1e-3)  # bound 2e-3 exceeds the smallest eigenvalue 1e-3
        self.assertEqual(out['verdict'], INC)
        self.assertIn('error bound', out['reason'])
        for bad in (None, 0.0, -1e-9, float('nan'), float('inf'), True, '1e-9'):
            with self.subTest(radius=bad):
                self.assertEqual(hess(m, [1, 2], bad)['verdict'], INC)
        # Recorded asymmetry widens the radius.
        self.assertEqual(health.certify_hessian(m, [1, 2], COORD, 1e-6, 2e-3)['verdict'], INC)

    def test_6_pinned_indices_are_reported_not_dropped(self):
        m = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -5.0]]  # index 3 pinned
        out = hess(m, [1, 2], pinned_indices=[3])
        self.assertEqual((out['verdict'], out['free_indices'], out['excluded_indices']), (PASS, [1, 2], [3]))
        self.assertEqual(hess(m, [1, 2, 3])['verdict'], FAIL)  # the free index is never dropped
        self.assertEqual(hess(m, [1, 2], pinned_indices=[2])['verdict'], INC)  # pinned record disagrees
        self.assertEqual(hess(m, [2, 1])['verdict'], PASS)  # order irrelevant
        for bad in ([], [1, 1], [0, 1], [1, 4], [True], None, 'x'):
            with self.subTest(indices=bad):
                self.assertEqual(hess(m, bad)['verdict'], INC)
        asym = [[1.0, 0.5], [0.4, 1.0]]
        self.assertEqual(hess(asym, [1, 2])['verdict'], INC)

    def test_7_penalised_models_are_not_certified_by_the_no_prior_diagnostic(self):
        data = certified_good()
        data['Julia'][0]['penalty_state'] = 'ridge'
        self.assertEqual(health.certify_fit(data['Julia'][0])['verdict'], INC)
        data['R'][0]['ridge_penalised'] = True
        data['R'][0]['ridge_tau'] = 1.0
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = certified_good()
        data['R'][0]['fixed_prior_state'] = 'prior'
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_8_size_and_time_caps_refuse(self):
        n = health.CERT_MAX_FREE_DIMENSION
        ident = lambda k: [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        self.assertEqual(hess(ident(n), list(range(1, n + 1)))['verdict'], PASS)
        out = hess(ident(n + 1), list(range(1, n + 2)))
        self.assertEqual(out['verdict'], INC)
        self.assertIn('size cap', out['reason'])
        out = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, max_seconds=-1.0)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('time cap', out['reason'])
        out = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, max_dimension=2)
        self.assertEqual(out['verdict'], INC)

    def test_failure_needs_valid_evidence_and_is_exact_on_singular(self):
        # Exactly singular recorded matrix: a positive error radius cannot rule out PD, so not a failure.
        self.assertEqual(hess([[1.0, 1.0], [1.0, 1.0]], [1, 2])['verdict'], INC)
        self.assertEqual(hess([[-1.0, 0.0], [0.0, 1.0]], [1, 2])['verdict'], FAIL)
        data = certified_good()
        row = data['Julia'][0]
        row['hessian'] = row['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')  # Float64 flag mismatch kept
            row['objective_verified'] = False
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')  # unbound: never a failure
            row['objective_verified'] = True
            data['Julia'][1]['case'] = data['Julia'][2]['case']
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')  # invalid case set

    def test_unrecorded_radius_is_incomplete_and_real_writer_shape_cannot_pass(self):
        data = good()  # no radius, coordinate or gradient error: what the Julia writer emits today
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_certifier_is_exact_on_binary64_values(self):
        # Decisions use exact Fractions: a near-singular matrix whose exact
        # determinant is 2^-70 > 0 is certified with a far smaller radius.
        a = 2.0 ** -10
        m = [[a, a], [a, a + 2.0 ** -60]]
        self.assertGreater(Fraction(m[0][0]) * Fraction(m[1][1]) - Fraction(m[0][1]) ** 2, 0)
        self.assertEqual(hess(m, [1, 2], 2.0 ** -200)['verdict'], PASS)
        self.assertEqual(hess(m, [1, 2], 2.0 ** -40)['verdict'], INC)


if __name__ == '__main__':
    unittest.main()
