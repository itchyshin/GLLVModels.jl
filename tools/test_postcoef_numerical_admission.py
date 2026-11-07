"""Controls for the exact/interval numerical-admission certifier (diagnostic)."""
import copy
import math
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
        row.update(hessian_coordinate=COORD, hessian_error_radius=1e-9, gradient_error_radius=1e-9,
                   pinned_indices=[])
    return data


def enabled():
    return patch.object(health, 'NUMERICAL_ADMISSION_IMPLEMENTED', True)


def hess(matrix, indices, radius=1e-9, coordinate=COORD, **kw):
    """Certify with the complement recorded as pinned unless the test overrides it."""
    n = len(matrix)
    if 'pinned_indices' not in kw:
        kw['pinned_indices'] = ([i for i in range(1, n + 1) if i not in indices]
                                if isinstance(indices, list) and all(type(i) is int for i in indices) else [])
    kw.setdefault('parameter_count', n)
    return health.certify_hessian(matrix, indices, coordinate, radius, 0.0, **kw)


class CertifierTests(unittest.TestCase):
    def test_1_float64_cholesky_counterexample_is_rejected(self):
        self.assertIs(health.hessian_pd(INDEFINITE, [1, 2, 3], 3), True)  # the Float64 screen is fooled
        # Exact lambda_min is about -1e-16, far below the binary64 representation
        # floor of the recorded entries: no admissible radius can decide it.
        self.assertEqual(hess(INDEFINITE, [1, 2, 3], 1e-25)['verdict'], INC)  # sub-floor radius refused
        self.assertEqual(hess(INDEFINITE, [1, 2, 3], 1e-12)['verdict'], INC)
        data = with_indefinite_julia_hessian(certified_good())
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            data['Julia'][0]['hessian_error_radius'] = 1e-25
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')  # floor, not FAIL

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
        row['gradient'], row['max_abs_gradient'] = [9.9e-6, 0.0], 9.9e-6  # one recorded gradient throughout
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
        self.assertEqual(health.certify_hessian(m, [1, 2], COORD, 1e-6, 2e-3, [], parameter_count=2)['verdict'], INC)
        self.assertEqual(health.certify_hessian(m, [1, 2], COORD, 1e-6, 0.0, [], parameter_count=2)['verdict'], PASS)

    def test_6_pinned_indices_are_reported_not_dropped(self):
        m = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -5.0]]  # index 3 pinned
        out = hess(m, [1, 2], pinned_indices=[3])
        self.assertEqual((out['verdict'], out['free_indices'], out['excluded_indices']), (PASS, [1, 2], [3]))
        self.assertEqual(hess(m, [1, 2, 3])['verdict'], FAIL)  # the free index is never dropped
        self.assertEqual(hess(m, [1, 2], pinned_indices=[2])['verdict'], INC)  # pinned record disagrees
        self.assertEqual(hess(m, [1, 2], pinned_indices=None)['verdict'], INC)  # pinned record mandatory
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
        out = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_seconds=-1.0, parameter_count=3)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('time cap', out['reason'])
        out = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_dimension=2, parameter_count=3)
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
        # At the floor radius 2**-51 (k = 2, max|H| = 1) the bound is 2**-50:
        # a pivot 2**-100 above it passes, exactly on it or below it does not.
        r = 2.0 ** -51
        for x, verdict in ((2.0 ** -50 + 2.0 ** -100, PASS), (2.0 ** -50, INC), (2.0 ** -50 - 2.0 ** -100, INC)):
            with self.subTest(x=x):
                self.assertEqual(hess([[1.0, 0.0], [0.0, x]], [1, 2], r)['verdict'], verdict)


def three_parameter_julia(rows, free, pinned=None):
    """Julia rows with 3 recorded parameters whose third coordinate is -5 curvature."""
    m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -5.0]]
    for r in rows:
        r['parameters'] = [0.0] * 3
        r['gradient'] = [1e-7, 0.0, 0.0]
        r['free_indices'] = list(free)
        r['reconstruction']['recovered_parameters'] = [0.0] * 3
        r['hessian'] = copy.deepcopy(m)
        r['hessian_raw'] = copy.deepcopy(m)
        r['hessian_asymmetry_limit'] = 32 * 2.220446049250313e-16 * 3 * 5.0
        if pinned is None:
            r.pop('pinned_indices', None)
        else:
            r['pinned_indices'] = pinned
    return rows


class ReviewDefectTests(unittest.TestCase):
    """Regressions for defects D1-D4 found by the independent adversarial review."""

    # D1
    def test_d1_omitted_index_cannot_hide_a_bad_coordinate(self):
        data = certified_good()
        three_parameter_julia(data['Julia'], [1, 2], pinned=None)  # index 3 omitted, none recorded
        with enabled():
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')
        three_parameter_julia(data['Julia'], [1, 2], pinned=[3])  # complete partition: certifiable
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
        three_parameter_julia(data['Julia'], [1, 2], pinned=[])  # complement not recorded
        with enabled():
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')

    def test_d1_partition_must_be_exact_and_use_the_recorded_parameter_count(self):
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -5.0]]
        ok = health.certify_hessian(m, [1, 2], COORD, 1e-9, 0.0, [3], parameter_count=3)
        self.assertEqual(ok['verdict'], PASS)
        for pinned, count in ((None, 3), ([], 3), ([3, 3], 3), ([1, 3], 3), ([3], 4), ([3], None), ([3], 2),
                              ([3], True), ('3', 3)):
            with self.subTest(pinned=pinned, count=count):
                out = health.certify_hessian(m, [1, 2], COORD, 1e-9, 0.0, pinned, parameter_count=count)
                self.assertEqual(out['verdict'], INC)
        # the matrix must be the full parameter-count matrix
        out = health.certify_hessian([[1.0, 0.0], [0.0, 1.0]], [1, 2], COORD, 1e-9, 0.0, [], parameter_count=3)
        self.assertEqual(out['verdict'], INC)

    # D2
    def test_d2_declared_radius_has_a_representation_floor(self):
        out = health.certify_hessian([[1e-17, 0.0], [0.0, 1.0]], [1, 2], COORD, 1e-300, 0.0, [], parameter_count=2)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('floor', out['reason'])
        eps = 2.0 ** -52
        m = [[2.0, 1.0], [1.0, 2.0]]
        floor = 2 * eps * 2.0  # k * 2**-52 * max|H|
        below = math.nextafter(floor, 0.0)
        self.assertEqual(hess(m, [1, 2], below)['verdict'], INC)
        self.assertEqual(hess(m, [1, 2], floor)['verdict'], PASS)
        self.assertEqual(health.certify_gradient(1e-3, 1e-300), INC)
        gfloor = eps * 1e-3
        self.assertEqual(health.certify_gradient(1e-3, math.nextafter(gfloor, 0.0)), INC)
        self.assertEqual(health.certify_gradient(1e-7, 1e-9), PASS)
        self.assertEqual(health.certify_gradient(1e-3, gfloor), FAIL)  # radius exactly at the floor is admissible
        # floor is over the free block only: a huge pinned entry does not raise it
        big = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e30]]
        self.assertEqual(hess(big, [1, 2], 1e-9)['verdict'], PASS)

    # D3
    def test_d3_certified_hessian_failure_needs_a_record_that_passes_its_own_checks(self):
        def fresh():
            data = certified_good()
            row = data['Julia'][0]
            row['hessian'] = row['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
            return data, row
        data, row = fresh()
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')  # baseline: confirmed
        mutations = {
            'hessian_raw missing': lambda r: r.pop('hessian_raw'),
            'raw vs symmetrised check invalid': lambda r: r.update(hessian_asymmetry_limit='x'),
            'parameters_match false': lambda r: r['reconstruction'].update(parameters_match=False),
            'profile unverified': lambda r: r['optimizer'].update(optimizer_profile_verified=False),
            'profile value differs': lambda r: r['optimizer'].update(optimizer_profile_value=3.5),
            'optimizer error recorded': lambda r: r['optimizer'].update(optimizer_error='boom'),
        }
        for label, mutate in mutations.items():
            with self.subTest(label):
                data, row = fresh()
                mutate(row)
                with enabled():
                    self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    # D4
    def test_d4_bound_carries_the_block_size_factor(self):
        dense = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]  # eigenvalues 4, 1, 1
        self.assertEqual(hess(dense, [1, 2, 3], 0.4)['verdict'], INC)  # k*r = 1.2 > 1; r alone would pass
        self.assertEqual(hess(dense, [1, 2, 3], 0.3)['verdict'], PASS)  # k*r = 0.9 < 1
        shifted = [[0.5, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 0.5]]  # eigenvalues 2.5, -0.5, -0.5
        self.assertEqual(hess(shifted, [1, 2, 3], 0.4)['verdict'], INC)  # k*r = 1.2 > 0.5; r alone would FAIL
        self.assertEqual(hess(shifted, [1, 2, 3], 0.1)['verdict'], FAIL)  # k*r = 0.3 < 0.5

    def test_d4_exactly_singular_shift_is_not_positive_definite(self):
        out = hess([[1.0]], [1], 1.0)  # A - bound*I = [[0]] exactly
        self.assertEqual(out['verdict'], INC)
        out = hess([[-1.0]], [1], 1.0)  # A + bound*I = [[0]] exactly: true lambda_min <= 0, a valid failure
        self.assertEqual(out['verdict'], FAIL)

    def test_d4_negative_asymmetry_is_refused(self):
        out = health.certify_hessian([[-1.0]], [1], COORD, 1e-9, -2.0, [], parameter_count=1)
        self.assertEqual(out['verdict'], INC)  # a negative term would shrink the bound and certify PASS
        self.assertIn('asymmetry', out['reason'])

    def test_d4_gradient_boundary_is_exact(self):
        limit = Fraction(1, 4)
        with patch.object(health, 'CERT_GRADIENT_LIMIT', limit):
            self.assertEqual(health.certify_gradient(0.125, 0.125), INC)  # g + e == limit: not a pass
            self.assertEqual(health.certify_gradient(0.125, 0.125 - 2.0 ** -56), PASS)
            self.assertEqual(health.certify_gradient(0.375, 0.125), FAIL)  # g - e == limit: failure
            self.assertEqual(health.certify_gradient(0.375, 0.125 + 2.0 ** -55), INC)
        # With the real decimal limit 1e-5 no binary64 sum equals it; use neighbours.
        lim = health.CERT_GRADIENT_LIMIT
        g = 5e-6
        e_pass = float(lim - Fraction(g))
        e_pass = e_pass if Fraction(g) + Fraction(e_pass) < lim else math.nextafter(e_pass, 0.0)
        self.assertLess(Fraction(g) + Fraction(e_pass), lim)
        self.assertEqual(health.certify_gradient(g, e_pass), PASS)
        e_inc = math.nextafter(e_pass, 1.0)
        while Fraction(g) + Fraction(e_inc) < lim:
            e_inc = math.nextafter(e_inc, 1.0)
        self.assertEqual(health.certify_gradient(g, e_inc), INC)

    def test_d4_integrity_rejection_precedes_a_certified_failure(self):
        data = certified_good()
        data['Julia'][1]['max_abs_gradient'] = 5e-8  # contradicts the recorded vector
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INTEGRITY_REJECTED')
        data = certified_good()
        bad = data['Julia'][0]
        bad['hessian'] = bad['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')
            data['Julia'][1]['max_abs_gradient'] = 5e-8
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INTEGRITY_REJECTED')

    def test_fit_and_optimizer_gradient_scalars_must_agree(self):
        data = certified_good()
        row = data['Julia'][0]
        # internally consistent optimizer record, but not the fit's own gradient
        row['optimizer'].update(optimizer_gradient=[1e-9, 0.0], optimizer_max_abs_gradient=1e-9)
        cert = health.certify_fit(row)
        self.assertNotEqual(cert['verdict'], PASS)
        self.assertEqual(cert['gradient'], INC)
        # The baseline records these as distinct objects (test_julia_limit_uses_optimizer_coordinates),
        # so the certifier, not the baseline classifier, refuses to certify a disagreement.
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
