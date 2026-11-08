"""Controls for the exact/interval numerical-admission certifier (diagnostic)."""
import copy
import json
import math
import unittest
from fractions import Fraction
from unittest.mock import patch

import core070_postcoef_fit_health as health
from test_postcoef_fit_health import INDEFINITE, good, with_indefinite_julia_hessian

COORD = 'packed_free_index'
PASS, FAIL, INC = health.CERT_PASS, health.CERT_FAIL, health.CERT_INCOMPLETE


def lambda_map(n, pinned=(), values=None):
    """lambda_constraint_map of a K = 1 fit with n packed parameters (q_free = 0, p = n - 1).

    Packed index 1 is log sigma; index j >= 2 is the loading of trait j - 1.
    """
    entries = [[None] for _ in range(n - 1)]
    for j in pinned:
        entries[j - 2][0] = 0.0 if values is None else values[j]
    return {'layout': health.LAMBDA_MAP_LAYOUT, 'p': n - 1, 'K': 1, 'q_free': 0, 'entries': entries}


def bind(row, pinned=()):
    """Add the binding fields a certifiable Julia record carries (parameters assumed 0.0 at pins)."""
    n = len(row['parameters'])
    row['lambda_constraint_map'] = lambda_map(n, pinned)
    row['radius_status'] = 'declared_unproven'
    row['optimizer']['radius_status'] = 'declared_unproven'
    row['reconstruction']['optimizer_parameters'] = list(row['optimizer']['optimizer_parameters'])
    return row


def certified_good():
    """good() plus the fields a certifiable Julia record must declare."""
    data = good()
    for row in data['Julia']:
        row.update(hessian_coordinate=COORD, hessian_error_radius=1e-9, gradient_error_radius=1e-9,
                   packed_gradient_error_radius=1e-9, pinned_indices=[])
        bind(row)
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
        row['optimizer']['optimizer_gradient'] = [9.9e-6]
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
        out = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_work=0, parameter_count=3)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('work cap', out['reason'])
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
        r['optimizer']['optimizer_parameters'] = [0.0] * 2
        r['optimizer']['optimizer_gradient'] = [1e-7, 0.0]
        bind(r, pinned or ())
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

    def test_fit_and_optimizer_gradients_are_two_separate_bounds(self):
        # M3/W4: the packed gradient at the Hessian point and the optimizer-coordinate
        # gradient are different quantities; each must clear the limit on its own
        # radius. Equality of their maxima is neither required nor sufficient.
        data = certified_good()
        row = data['Julia'][0]
        row['optimizer'].update(optimizer_gradient=[1e-9], optimizer_max_abs_gradient=1e-9)
        self.assertEqual(health.certify_fit(row)['verdict'], PASS)  # 1e-9 vs 1e-7: both far below the limit
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
        # packed gradient large, optimizer gradient small: blocked, and never a failure
        row['gradient'], row['max_abs_gradient'] = [5e-3, 0.0], 5e-3
        cert = health.certify_fit(row)
        self.assertEqual((cert['verdict'], cert['gradient']), (INC, INC))
        # optimizer gradient above the signed limit with a small packed gradient: a certified failure
        data = certified_good()
        row = data['Julia'][0]
        row['optimizer'].update(optimizer_gradient=[2e-5], optimizer_max_abs_gradient=2e-5)
        cert = health.certify_fit(row)
        self.assertEqual((cert['verdict'], cert['gradient']), (FAIL, FAIL))


class SecondReviewC1Tests(unittest.TestCase):
    """C1: an unhashable or non-string case label must never crash the enabled classifier."""

    def test_c1_unhashable_case_label_is_incomplete_not_a_crash(self):
        for junk in (['x'], {'a': 1}, [], {}, None, 1, 1.5, True, ('t',), b'x'):
            with self.subTest(case=junk):
                data = certified_good()
                data['Julia'][1]['case'] = junk
                with enabled():
                    self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
                self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
                self.assertEqual(health.certify_evidence(data), ([], False))

    def test_c1_fuzz_classify_never_raises_with_the_flag_enabled(self):
        import random
        rnd = random.Random(20261107)
        junk = [None, True, False, 0, 1, -1, 0.0, 1e-5, -1e-5, 1e-7, 2e-5, float('nan'), float('inf'),
                'x', '', [], {}, ['x'], {'a': 1}, [1], [[1.0]], 1e300, 3.0, 5e-324]

        def paths(o, p=()):
            yield p
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from paths(v, p + (k,))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    yield from paths(v, p + (i,))

        def get(o, p):
            for k in p:
                o = o[k]
            return o
        outcomes = set()
        with enabled():
            for _ in range(5000):
                d = certified_good()
                for _ in range(rnd.choice([1, 1, 2, 3])):
                    p = rnd.choice([q for q in paths(d) if q])
                    parent, k = get(d, p[:-1]), p[-1]
                    r = rnd.random()
                    if r < .25:
                        parent.pop(k, None) if isinstance(parent, dict) else parent.pop(k)
                    elif r < .8:
                        parent[k] = rnd.choice(junk)
                    elif isinstance(parent[k], float) and math.isfinite(parent[k]):
                        parent[k] = parent[k] * rnd.choice([-1, 2, 0.5, 1e3]) + rnd.choice([0, 1e-9])
                outcomes.add(health.classify(d))  # an exception here fails the test
        self.assertTrue(outcomes <= {'FIT_HEALTH_PASS', 'FIT_HEALTH_INCOMPLETE', 'FIT_HEALTH_FAIL',
                                     'FIT_HEALTH_FAIL_WITH_INCOMPLETE', 'FIT_HEALTH_INTEGRITY_REJECTED'})


class SecondReviewC3ExactnessTests(unittest.TestCase):
    """C3: tests a float implementation would fail (exact Fraction decisions)."""

    # Each matrix is decided differently by a pure-float LDL^T with float(shift) at the same radius.
    # Found by search at the exact PASS/INCOMPLETE and FAIL/INCOMPLETE boundaries; verified below by a float oracle.
    PASS_FLIPS = [([[1.0, 1.3751353334342902], [1.3751353334342902, 1.890997186190759]], 1.610729092553186e-10),
                  ([[1.0, 0.6431982437247019], [0.6431982437247019, 0.41370398173054107]], 3.53680841551169e-10)]
    INC_FLIP = ([[1.0, 1.0], [1.0, 1.1663355006277607]], 0.0398576363841728)  # float says PASS, exact INCOMPLETE
    FAIL_FLIP = ([[1.0, 1.1285362149100029], [1.1285362149100029, 1.2735939877113645]], 1.4339230721875957e-10)  # float says INC

    @staticmethod
    def float_oracle(m, radius, asym=0.0):
        k = len(m)
        bound = float(k * (Fraction(radius) + Fraction(asym)))

        def pd(shift):
            a = [[m[i][j] - (shift if i == j else 0.0) for j in range(k)] for i in range(k)]
            low, d = [[0.0] * k for _ in range(k)], []
            for j in range(k):
                p = a[j][j] - sum(low[j][t] ** 2 * d[t] for t in range(j))
                if p <= 0:
                    return False
                d.append(p)
                for i in range(j + 1, k):
                    low[i][j] = (a[i][j] - sum(low[i][t] * low[j][t] * d[t] for t in range(j))) / p
            return True
        if pd(bound):
            return PASS
        return FAIL if not pd(-bound) else INC

    def test_c3_float_arithmetic_mis_decides_where_exact_decides(self):
        for m, r in self.PASS_FLIPS:
            with self.subTest(kind='pass', r=r):
                self.assertEqual(self.float_oracle(m, r), INC)  # the float oracle is wrong here
                self.assertEqual(hess(m, [1, 2], r)['verdict'], PASS)
                neg = [[-x for x in row] for row in m]
                self.assertEqual(hess(neg, [1, 2], r)['verdict'], FAIL)
        m, r = self.INC_FLIP
        self.assertEqual(self.float_oracle(m, r), PASS)
        self.assertEqual(hess(m, [1, 2], r)['verdict'], INC)
        m, r = self.FAIL_FLIP
        self.assertEqual(self.float_oracle(m, r), INC)
        self.assertEqual(hess(m, [1, 2], r)['verdict'], FAIL)

    def test_c3_bound_and_shift_rounding_flip_the_verdict(self):
        # radius + asymmetry = 1 - 2**-54 exactly; float addition rounds it to 1.0, making the pivot 0.
        r, a = 0.5, 0.5 - 2.0 ** -54
        self.assertEqual(float(Fraction(r) + Fraction(a)), 1.0)
        out = health.certify_hessian([[1.0]], [1], COORD, r, a, [], parameter_count=1)
        self.assertEqual(out['verdict'], PASS)
        self.assertEqual(out['error_bound'], 1 - Fraction(1, 2 ** 54))
        # Same mirror image: A + bound*I is exactly indefinite only with the exact bound.
        out = health.certify_hessian([[-1.0]], [1], COORD, r, a, [], parameter_count=1)
        self.assertEqual(out['verdict'], FAIL)
        out = health.certify_hessian([[1.0]], [1], COORD, r, 0.5, [], parameter_count=1)  # bound exactly 1
        self.assertEqual(out['verdict'], INC)

    def test_c3_gradient_limit_is_the_exact_decimal_not_the_float(self):
        self.assertEqual(health.CERT_GRADIENT_LIMIT, Fraction(1, 100000))
        self.assertNotEqual(health.CERT_GRADIENT_LIMIT, Fraction(1e-5))
        limit = health.CERT_GRADIENT_LIMIT
        g, e = 9e-06, 1e-06
        total = Fraction(g) + Fraction(e)
        self.assertTrue(limit <= total < Fraction(1e-5))  # lies between decimal 1e-5 and float(1e-5)
        self.assertEqual(health.certify_gradient(g, e), INC)  # exact sum is not below the limit
        below = math.nextafter(e, 0.0)  # neighbour below: exact sum strictly below the decimal limit
        self.assertLess(Fraction(g) + Fraction(below), limit)
        self.assertEqual(health.certify_gradient(g, below), PASS)
        above = math.nextafter(e, 1.0)
        self.assertGreaterEqual(Fraction(g) + Fraction(above), limit)
        self.assertEqual(health.certify_gradient(g, above), INC)
        # the float 1e-5 itself is above the decimal limit
        self.assertEqual(health.certify_gradient(1e-5, 1e-20), INC)
        self.assertEqual(health.certify_gradient(2e-5, 1e-6), FAIL)

    def test_c3_survivor_controls_from_the_second_mutation_run(self):
        # zero radius with an all-zero block: refused, not a (zero-margin) FAIL
        self.assertEqual(hess([[0.0]], [1], 0.0)['verdict'], INC)
        self.assertEqual(hess([[0.0]], [1], -0.0)['verdict'], INC)
        # duplicated free index: refused
        three = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertEqual(hess(three, [1, 1, 2], pinned_indices=[3])['verdict'], INC)
        # only the Julia engine has a recorded Hessian to certify
        row = certified_good()['Julia'][0]
        self.assertEqual(health.certify_fit(row, 'Julia')['verdict'], PASS)
        out = health.certify_fit(row, 'R')
        self.assertEqual(out['verdict'], INC)
        self.assertIn('engine', out['reason'])
        # Hessian size must equal the recorded gradient length
        row = certified_good()['Julia'][0]
        row['gradient'], row['parameters'] = [1e-7, 0.0, 0.0], [0.0] * 3
        row['optimizer'].update(optimizer_parameters=[0.0] * 2, optimizer_gradient=[1e-7, 0.0])
        bind(row)  # a 3-parameter record whose Hessian is still 2 x 2
        out = health.certify_fit(row)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('parameter count', out['hessian']['reason'])

    def test_c3_floors_are_compared_exactly(self):
        # k = 3: the exact floor 3*2**-52*peak is not a binary64 number; a radius equal to its
        # round-down float is below the exact floor and must be refused.
        peak = 1.000000000078736
        exact_floor = 3 * health.CERT_EPS * Fraction(peak)
        r_down = float(exact_floor)
        self.assertLess(Fraction(r_down), exact_floor)
        m = [[peak, 0.0, 0.0], [0.0, peak, 0.0], [0.0, 0.0, peak]]
        out = hess(m, [1, 2, 3], r_down)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('floor', out['reason'])
        r_up = math.nextafter(r_down, 1.0)
        self.assertGreaterEqual(Fraction(r_up), exact_floor)
        self.assertEqual(hess(m, [1, 2, 3], r_up)['verdict'], PASS)

    def test_c3_gradient_failure_boundary_just_above_the_float_limit(self):
        g = math.nextafter(1e-5, 1.0)
        e = 2.3e-21  # at least the representation floor 2**-52 * g, and g - e is within 1 ulp of 1e-5
        self.assertGreaterEqual(Fraction(e), health.CERT_EPS * Fraction(g))
        self.assertGreaterEqual(Fraction(g) - Fraction(e), health.CERT_GRADIENT_LIMIT)
        self.assertEqual(health.certify_gradient(g, e), FAIL)
        self.assertEqual(health.certify_gradient(g, math.nextafter(e, 1.0)), FAIL)
        self.assertEqual(health.certify_gradient(g, 1e-20), INC)  # g - e now below the limit

    def test_c3_negative_values_in_minus_one_to_zero_are_refused(self):
        m = [[1.0, 0.0], [0.0, 1.0]]
        for bad in (-0.5, -1e-300, math.nextafter(0.0, -1.0), -0.999):
            with self.subTest(bad=bad):
                self.assertEqual(hess(m, [1, 2], bad)['verdict'], INC)  # radius
                out = health.certify_hessian(m, [1, 2], COORD, 1e-9, bad, [], parameter_count=2)  # asymmetry
                self.assertEqual(out['verdict'], INC)
                self.assertIn('asymmetry', out['reason'])
                self.assertEqual(health.certify_gradient(bad, 1e-9), INC)  # max_abs_gradient
                self.assertEqual(health.certify_gradient(1e-7, bad), INC)  # gradient radius
        # a -0.0 asymmetry or gradient is a harmless zero
        self.assertEqual(health.certify_hessian(m, [1, 2], COORD, 1e-9, -0.0, [], parameter_count=2)['verdict'], PASS)
        self.assertEqual(health.certify_gradient(0.0, 1e-9), PASS)

    def test_c3_standalone_parameter_count_must_match_the_matrix(self):
        two = [[1.0, 0.0], [0.0, 1.0]]
        for count in (3, 1, 0, None, True, 2.0, '2'):
            with self.subTest(count=count):
                out = health.certify_hessian(two, [1, 2], COORD, 1e-9, 0.0, [3], parameter_count=count)
                self.assertEqual(out['verdict'], INC)
        out = health.certify_hessian(two, [1, 2], COORD, 1e-9, 0.0, [3], parameter_count=3)  # 2x2, pc=3, pinned=[3]
        self.assertEqual(out['verdict'], INC)
        self.assertIn('parameter count', out['reason'])
        three = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        out = health.certify_hessian(three, [1, 2], COORD, 1e-9, 0.0, [3], parameter_count=2)
        self.assertEqual(out['verdict'], INC)

    def test_c3_recorded_pass_with_certified_failure_is_never_a_pass(self):
        data = certified_good()
        row = data['Julia'][0]
        row['hessian'] = row['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        with enabled(), patch.object(health, '_classify_recorded', return_value='FIT_HEALTH_PASS'):
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL')  # stricter than incomplete
            row['reconstruction']['parameters_match'] = False  # the failure is no longer confirmable
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        # without the patch the recorded layer flags the Float64 mismatch, so a failure is never a PASS either
        data = certified_good()
        data['Julia'][0]['hessian'] = data['Julia'][0]['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        with enabled():
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')


class SecondReviewC4ConfirmationTests(unittest.TestCase):
    """C4: a certified failure also needs objective, log-likelihood and R-block consistency."""

    @staticmethod
    def failing():
        data = certified_good()
        row = data['Julia'][0]
        row['hessian'] = row['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        return data, row

    def test_c4_baseline_failure_is_confirmed(self):
        data, row = self.failing()
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')
        certs, confirmable = health.certify_evidence(data)
        self.assertTrue(confirmable)
        self.assertEqual((certs[0]['verdict'], certs[0]['confirmable']), (FAIL, True))

    def test_c4_each_condition_downgrades_to_incomplete(self):
        mutations = {
            'recomputed objective differs beyond roundoff': lambda d, r: r.update(recomputed_objective=3.5),
            'recorded objective difference missing': lambda d, r: r.pop('objective_difference'),
            'recorded roundoff limit missing': lambda d, r: r.pop('objective_roundoff_limit'),
            'loglik_matches false': lambda d, r: r['reconstruction'].update(loglik_matches=False),
            'loglik_matches missing': lambda d, r: r['reconstruction'].pop('loglik_matches'),
            'recovered loglik differs from fit': lambda d, r: r['reconstruction'].update(recovered_loglik=-4.0),
            'recovered loglik missing': lambda d, r: r['reconstruction'].pop('recovered_loglik'),
            'reconstruction error recorded': lambda d, r: r['reconstruction'].update(error='boom'),
            'R block missing': lambda d, r: d.pop('R'),
            'R block garbage string': lambda d, r: d.update(R='garbage'),
            'R block empty list': lambda d, r: d.update(R=[]),
            'R block short': lambda d, r: d['R'].pop(),
            'R fit not a dict': lambda d, r: d['R'].__setitem__(0, 'x'),
            'R parameters malformed': lambda d, r: d['R'][0].update(parameters='x'),
            'R gradient non-finite': lambda d, r: d['R'][1].update(gradient=[float('nan'), 0.0]),
            'R prior recorded': lambda d, r: d['R'][2].update(fixed_prior_state='prior'),
            'R objective unverified': lambda d, r: d['R'][0].update(objective_verified=False),
            'R case duplicated': lambda d, r: d['R'][2].update(case=d['R'][1]['case']),
        }
        for label, mutate in mutations.items():
            with self.subTest(label):
                data, row = self.failing()
                mutate(data, row)
                with enabled():
                    self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
                certs, confirmable = health.certify_evidence(data)
                self.assertEqual(certs[0]['verdict'], FAIL)  # still certified, but not confirmable
                self.assertFalse(certs[0]['confirmable'])

    def test_c4_record_checks_reject_subthreshold_objective_disagreement_and_bad_r_flag(self):
        data, row = self.failing()
        r_valid = health._r_block_valid(data)
        self.assertTrue(health._record_checks_valid(row, r_valid))
        self.assertFalse(health._record_checks_valid(row, False))
        self.assertFalse(health._record_checks_valid(row, None))
        row['recomputed_objective'] = 3.0 + 4e-16  # inside the roundoff limit, but the record says difference 0.0
        self.assertFalse(health._record_checks_valid(row, r_valid))
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INTEGRITY_REJECTED')
        row['objective_difference'] = abs(3.0 - row['recomputed_objective'])  # recorded check now consistent
        row['objective_roundoff_limit'] = 8 * 2.220446049250313e-16 * row['recomputed_objective']
        self.assertTrue(health._record_checks_valid(row, r_valid))
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_c4_roundoff_limit_is_enforced_even_when_the_record_is_self_consistent(self):
        data, row = self.failing()
        row['recomputed_objective'] = 3.5
        row['objective_difference'] = 0.5
        row['objective_roundoff_limit'] = 8 * 2.220446049250313e-16 * 3.5  # recorded values match the recomputation
        self.assertFalse(health._record_checks_valid(row, health._r_block_valid(data)))
        row['recomputed_objective'] = 3.0
        row['objective_difference'] = 0.0
        row['objective_roundoff_limit'] = 8 * 2.220446049250313e-16 * 3.0
        self.assertTrue(health._record_checks_valid(row, health._r_block_valid(data)))

    def test_c4_r_block_validity_is_checked_field_by_field(self):
        data = certified_good()
        self.assertTrue(health._r_block_valid(data))
        mutations = {
            'free indices partial': lambda f: f.update(free_indices=[1]),
            'free indices duplicated': lambda f: f.update(free_indices=[1, 1, 2]),
            'free indices non-int': lambda f: f.update(free_indices=[1.0, 2.0]),
            'max_abs_gradient disagrees': lambda f: f.update(max_abs_gradient=5e-8),
            'max_abs_gradient missing': lambda f: f.pop('max_abs_gradient'),
            'objective differs': lambda f: f.update(recomputed_objective=3.5),
            'pd flag not bool': lambda f: f.update(pd_hessian=1),
            'pd flag missing': lambda f: f.pop('pd_hessian'),
            'ridge penalised': lambda f: f.update(ridge_penalised=True),
            'ridge flag not bool': lambda f: f.update(ridge_penalised=0),
            'ridge tau finite': lambda f: f.update(ridge_tau=1.0),
            'prior recorded': lambda f: f.update(fixed_prior_state='prior'),
            'penalty unrecorded': lambda f: f.update(penalty_configuration_recorded=False),
            'error recorded': lambda f: f.update(error='x'),
            'error field missing': lambda f: f.pop('error'),
            'gradient length differs': lambda f: f.update(gradient=[1e-7]),
            'parameters non-finite': lambda f: f.update(parameters=[float('nan'), 0.0]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label):
                d = copy.deepcopy(data)
                mutate(d['R'][1])
                self.assertFalse(health._r_block_valid(d))
        for bad in (None, {}, {'R': None}, {'R': {}}):
            self.assertFalse(health._r_block_valid(bad))
        zero = copy.deepcopy(data)
        for row in zero['R']:
            row.update(gradient=[0.0, 0.0], max_abs_gradient=0.0)
        self.assertTrue(health._r_block_valid(zero))  # an exactly zero gradient is well formed
        # a failing R pd flag or a large R gradient is well-formed evidence, not garbage
        d = copy.deepcopy(data)
        d['R'][0]['pd_hessian'] = False
        self.assertTrue(health._r_block_valid(d))

    def test_c4_gradient_certified_failure_has_the_same_requirements(self):
        data = certified_good()
        row = data['Julia'][0]
        row['optimizer']['optimizer_gradient'] = [2e-5]
        row['optimizer']['optimizer_max_abs_gradient'] = 2e-5
        row['gradient'], row['max_abs_gradient'] = [2e-5, 0.0], 2e-5
        certs, confirmable = health.certify_evidence(data)
        self.assertEqual((certs[0]['gradient'], certs[0]['verdict'], certs[0]['confirmable']), (FAIL, FAIL, True))
        for label, mutate in {'loglik false': lambda d: d['Julia'][0]['reconstruction'].update(loglik_matches=False),
                              'objective differs': lambda d: d['Julia'][0].update(recomputed_objective=3.5),
                              'R garbage': lambda d: d.update(R=None)}.items():
            with self.subTest(label):
                d = copy.deepcopy(data)
                mutate(d)
                self.assertFalse(health.certify_evidence(d)[0][0]['confirmable'])



INCOMPLETE_V, INTEGRITY_V = 'FIT_HEALTH_INCOMPLETE', 'FIT_HEALTH_INTEGRITY_REJECTED'
EPS = 2.220446049250313e-16


def rec_of(mutate, engine='Julia', index=2):
    """_classify_recorded verdict of good() after ``mutate(row, data)`` on one row.

    The recorded layer is asserted directly: ``classify`` converts every
    would-be pass to INCOMPLETE while the hold flag is False, so a test that
    only asserts classify() == INCOMPLETE cannot see a deleted rejection branch.
    """
    data = good()
    mutate(data[engine][index], data)
    return health._classify_recorded(data)


class RejectionBranchTests(unittest.TestCase):
    """F2 (third review): every would-be-pass rejection branch of _classify_recorded is pinned.

    The defect is applied to the last row of the last engine so that no later
    check can mask a deleted branch, and the exact verdict is asserted.
    """

    def check(self, table, engine, index=2):
        for label, mutate, expected in table:
            for idx in {index, 0}:
                with self.subTest(engine=engine, row=idx, defect=label):
                    self.assertEqual(rec_of(mutate, engine, idx), expected)

    @staticmethod
    def set_(**kw):
        return lambda row, data: row.update(kw)

    @staticmethod
    def drop(*keys):
        return lambda row, data: [row.pop(k) for k in keys]

    def test_baseline_good_record_is_a_recorded_pass(self):
        self.assertEqual(health._classify_recorded(good()), 'FIT_HEALTH_PASS')

    def test_evidence_shape_defects(self):
        for label, mutate in {
            'extra engine key': lambda d: d.update(Extra=[]),
            'engine block not a list': lambda d: d.update(R='x'),
            'engine block a string of three characters': lambda d: d.update(R='abc'),
            'engine block a tuple of three fits': lambda d: d.update(R=tuple(d['R'])),
            'Julia block not a list': lambda d: d.update(Julia={}),
            'R fit not a dict': lambda d: d['R'].__setitem__(2, 'x'),
            'Julia fit not a dict': lambda d: d['Julia'].__setitem__(2, 'x'),
            'Julia block too short': lambda d: d['Julia'].pop(),
            'Julia block empty': lambda d: d.update(Julia=[]),
            'R block empty': lambda d: d.update(R=[]),
            'extra Julia row with an unhashable case': lambda d: d['Julia'].append(
                dict(copy.deepcopy(d['Julia'][0]), case=['x'])),
            'extra R row with an unhashable case': lambda d: d['R'].append(dict(copy.deepcopy(d['R'][0]), case={})),
            'Julia block too long': lambda d: d['Julia'].append(copy.deepcopy(d['Julia'][0])),
            'duplicate case': lambda d: d['Julia'][2].update(case=d['Julia'][1]['case']),
            'unknown case': lambda d: d['Julia'][2].update(case='other'),
            'unhashable case': lambda d: d['Julia'][2].update(case=['x']),
            'non-string case': lambda d: d['R'][2].update(case=3),
        }.items():
            with self.subTest(label):
                data = good()
                mutate(data)
                self.assertEqual(health._classify_recorded(data), INCOMPLETE_V)
        data = good()
        data['R'][1]['case'] = data['R'][0]['case']
        data['R'][2].update(gradient=[2e-5, 0.0], max_abs_gradient=2e-5)  # failing, but its case set is invalid
        self.assertEqual(health._classify_recorded(data), INCOMPLETE_V)  # an unbound case set confirms no failure
        data = good()
        data['R'][2].update(gradient=[2e-5, 0.0], max_abs_gradient=2e-5)
        self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_FAIL')
        data['Julia'][1]['case'] = 'other'  # another engine's invalid case set does not unconfirm R's failure
        self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_common_row_defects_on_both_engines(self):
        s, d = self.set_, self.drop
        common = [
            ('objective unverified', s(objective_verified=False), INCOMPLETE_V),
            ('objective_verified missing', d('objective_verified'), INCOMPLETE_V),
            ('penalty configuration unrecorded', s(penalty_configuration_recorded=False), INCOMPLETE_V),
            ('error recorded', s(error='x'), INCOMPLETE_V),
            ('error key missing', d('error'), INCOMPLETE_V),
            ('pd flag false', s(pd_hessian=False), INCOMPLETE_V),
            ('pd flag non-bool', s(pd_hessian=1), INCOMPLETE_V),
            ('pd flag None', s(pd_hessian=None), INCOMPLETE_V),
            ('pd flag missing', d('pd_hessian'), INCOMPLETE_V),
            ('gradient scalar NaN', s(max_abs_gradient=float('nan')), INCOMPLETE_V),
            ('gradient scalar negative', s(max_abs_gradient=-1e-7), INCOMPLETE_V),
            ('gradient scalar bool', s(max_abs_gradient=True), INCOMPLETE_V),
            ('gradient scalar missing', d('max_abs_gradient'), INCOMPLETE_V),
            ('gradient scalar contradicts vector', s(max_abs_gradient=5e-8), INTEGRITY_V),
            ('gradient vector missing', d('gradient'), INCOMPLETE_V),
            ('gradient vector empty', s(gradient=[]), INCOMPLETE_V),
            ('gradient vector non-finite', s(gradient=[float('nan'), 0.0]), INCOMPLETE_V),
            ('parameters missing', d('parameters'), INCOMPLETE_V),
            ('parameters shorter than gradient', s(parameters=[0.0]), INCOMPLETE_V),
            ('parameters non-finite', s(parameters=[float('inf'), 0.0]), INCOMPLETE_V),
            ('free indices missing', d('free_indices'), INCOMPLETE_V),
            ('free indices empty', s(free_indices=[]), INCOMPLETE_V),
            ('free index out of range', s(free_indices=[1, 3]), INCOMPLETE_V),
            ('free index zero', s(free_indices=[0, 1]), INCOMPLETE_V),
            ('free index duplicated', s(free_indices=[1, 1]), INCOMPLETE_V),
            ('free index float', s(free_indices=[1.0, 2.0]), INCOMPLETE_V),
            ('recomputed objective NaN', s(recomputed_objective=float('nan')), INCOMPLETE_V),
            ('recomputed objective missing', d('recomputed_objective'), INCOMPLETE_V),
            ('optimizer objective missing', d('optimizer_objective'), INCOMPLETE_V),
        ]
        for engine in health.ENGINES:
            self.check(common, engine)

    def test_r_row_defects(self):
        s, d = self.set_, self.drop
        table = [
            ('free indices partial (R needs all)', s(free_indices=[1]), INCOMPLETE_V),
            ('recomputed objective differs', s(recomputed_objective=3.5), INCOMPLETE_V),
            ('ridge penalised', s(ridge_penalised=True, ridge_tau=1.0), INCOMPLETE_V),
            ('ridge flag int', s(ridge_penalised=0), INCOMPLETE_V),
            ('ridge on with tau the string Inf', s(ridge_penalised=True, ridge_tau='Inf'), INCOMPLETE_V),
            ('ridge on with tau NaN', s(ridge_penalised=True, ridge_tau=float('nan')), INCOMPLETE_V),
            ('ridge on with tau zero', s(ridge_penalised=True, ridge_tau=0.0), INCOMPLETE_V),
            ('ridge on with tau negative', s(ridge_penalised=True, ridge_tau=-1.0), INCOMPLETE_V),
            ('ridge on with tau missing', lambda r, dta: (r.update(ridge_penalised=True), r.pop('ridge_tau')),
             INCOMPLETE_V),
            ('ridge flag missing', d('ridge_penalised'), INCOMPLETE_V),
            ('ridge off but tau finite', s(ridge_tau=1.0), INCOMPLETE_V),
            ('ridge tau missing', d('ridge_tau'), INCOMPLETE_V),
            ('prior recorded', s(fixed_prior_state='prior'), INCOMPLETE_V),
            ('prior missing', d('fixed_prior_state'), INCOMPLETE_V),
            ('failing gradient is a confirmed failure', s(gradient=[2e-5, 0.0], max_abs_gradient=2e-5), 'FIT_HEALTH_FAIL'),
            ('failing gradient, objective unverified', s(gradient=[2e-5, 0.0], max_abs_gradient=2e-5,
                                                        objective_verified=False), INCOMPLETE_V),
            ('failing gradient, error recorded', s(gradient=[2e-5, 0.0], max_abs_gradient=2e-5, error='x'),
             INCOMPLETE_V),
            ('failing gradient, parameters mismatched', s(gradient=[2e-5, 0.0], max_abs_gradient=2e-5,
                                                         parameters=[0.0]), INCOMPLETE_V),
        ]
        self.check(table, 'R')

    def test_julia_row_defects(self):
        s, d = self.set_, self.drop
        rec = lambda **kw: (lambda row, data: row['reconstruction'].update(kw))
        opt = lambda **kw: (lambda row, data: row['optimizer'].update(kw))
        table = [
            ('penalty state ridge', s(penalty_state='ridge'), INCOMPLETE_V),
            ('penalty state missing', d('penalty_state'), INCOMPLETE_V),
            ('reconstruction missing', d('reconstruction'), INCOMPLETE_V),
            ('parameters_match false', rec(parameters_match=False), INCOMPLETE_V),
            ('loglik_matches false', rec(loglik_matches=False), INCOMPLETE_V),
            ('reconstruction error recorded', rec(error='x'), INCOMPLETE_V),
            ('reconstruction error key missing', lambda r, dta: r['reconstruction'].pop('error'), INCOMPLETE_V),
            ('recovered parameters differ', rec(recovered_parameters=[0.0, 1.0]), INCOMPLETE_V),
            ('recovered parameters missing', lambda r, dta: r['reconstruction'].pop('recovered_parameters'),
             INCOMPLETE_V),
            ('recovered parameters non-finite', rec(recovered_parameters=[0.0, float('nan')]), INCOMPLETE_V),
            ('recovered loglik wrong', rec(recovered_loglik=-4.0), INCOMPLETE_V),
            ('recovered loglik NaN', rec(recovered_loglik=float('nan')), INCOMPLETE_V),
            ('optimizer block missing', d('optimizer'), INCOMPLETE_V),
            ('profile unverified', opt(optimizer_profile_verified=False), INCOMPLETE_V),
            ('optimizer error recorded', opt(optimizer_error='x'), INCOMPLETE_V),
            ('optimizer error key missing', lambda r, dta: r['optimizer'].pop('optimizer_error'), INCOMPLETE_V),
            ('profile value differs from minimum', opt(optimizer_profile_value=3.5), INCOMPLETE_V),
            ('profile value NaN', opt(optimizer_profile_value=float('nan')), INCOMPLETE_V),
            ('optimizer minimum missing', lambda r, dta: r['optimizer'].pop('optimizer_minimum'), INCOMPLETE_V),
            ('optimizer point empty', opt(optimizer_parameters=[]), INCOMPLETE_V),
            ('optimizer point non-finite', opt(optimizer_parameters=[float('nan')]), INCOMPLETE_V),
            ('optimizer gradient length differs', opt(optimizer_gradient=[1e-7, 0.0]), INCOMPLETE_V),
            ('optimizer gradient non-finite', opt(optimizer_gradient=[float('inf')]), INCOMPLETE_V),
            ('optimizer scalar NaN', opt(optimizer_max_abs_gradient=float('nan')), INCOMPLETE_V),
            ('optimizer scalar negative', opt(optimizer_max_abs_gradient=-1e-7), INCOMPLETE_V),
            ('optimizer scalar contradicts vector', opt(optimizer_max_abs_gradient=5e-8), INTEGRITY_V),
            ('optimizer gradient fails with a bound record',
             opt(optimizer_gradient=[2e-5], optimizer_max_abs_gradient=2e-5), 'FIT_HEALTH_FAIL'),
            ('optimizer gradient fails, objective unverified',
             lambda r, dta: (r['optimizer'].update(optimizer_gradient=[2e-5], optimizer_max_abs_gradient=2e-5),
                             r.update(objective_verified=False)), INCOMPLETE_V),
            ('optimizer gradient fails, profile unverified',
             lambda r, dta: r['optimizer'].update(optimizer_gradient=[2e-5], optimizer_max_abs_gradient=2e-5,
                                                  optimizer_profile_verified=False), INCOMPLETE_V),
            ('Hessian asymmetry missing', d('hessian_asymmetry'), INCOMPLETE_V),
            ('Hessian asymmetry limit missing', d('hessian_asymmetry_limit'), INCOMPLETE_V),
            ('Hessian raw missing', d('hessian_raw'), INCOMPLETE_V),
            ('Hessian missing', d('hessian'), INCOMPLETE_V),
            ('stored Hessian is not the symmetrised raw', lambda r, dta: r['hessian'][0].__setitem__(0, 2.0),
             INTEGRITY_V),
            ('recorded asymmetry differs', s(hessian_asymmetry=1e-9), INTEGRITY_V),
            ('recorded asymmetry limit differs', s(hessian_asymmetry_limit=1.0), INTEGRITY_V),
            ('Hessian indefinite under the Float64 screen',
             lambda r, dta: (r.update(pd_hessian=False), r['hessian'][0].__setitem__(0, -1.0),
                             r['hessian_raw'][0].__setitem__(0, -1.0)), INCOMPLETE_V),
            ('Hessian screen disagrees with the flag',
             lambda r, dta: (r['hessian'][0].__setitem__(0, -1.0), r['hessian_raw'][0].__setitem__(0, -1.0)),
             INCOMPLETE_V),
            ('recomputed objective beyond roundoff (record self-consistent)',
             s(recomputed_objective=3.5, objective_difference=0.5, objective_roundoff_limit=8 * EPS * 3.5),
             INCOMPLETE_V),
            ('recorded objective difference differs', s(objective_difference=1e-9), INTEGRITY_V),
            ('recorded objective limit differs', s(objective_roundoff_limit=1.0), INTEGRITY_V),
            ('recorded objective difference missing', d('objective_difference'), INCOMPLETE_V),
            ('recorded objective limit missing', d('objective_roundoff_limit'), INCOMPLETE_V),
        ]
        self.check(table, 'Julia')

    def test_boundaries_that_are_accepted(self):
        """Exact boundary values accepted by the recorded rules (kills < versus <= swaps)."""
        data = good()
        for engine in health.ENGINES:
            for row in data[engine]:
                row.update(gradient=[0.0, 0.0], max_abs_gradient=0.0)
        for row in data['Julia']:
            row['optimizer'].update(optimizer_gradient=[0.0], optimizer_max_abs_gradient=0.0)
        self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_PASS')  # zero gradient is a valid scalar
        # Objective roundoff: difference exactly equal to the limit (|objective| < 1 so the limit is 8 eps).
        data = good()
        row = data['Julia'][0]
        r = 0.5 + 8 * EPS
        row.update(optimizer_objective=0.5, recomputed_objective=r, objective_difference=8 * EPS,
                   objective_roundoff_limit=8 * EPS)
        row['optimizer'].update(optimizer_minimum=0.5, optimizer_profile_value=0.5)
        row['reconstruction'].update(recovered_loglik=-0.5)
        self.assertEqual(abs(0.5 - r), 8 * EPS)
        self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_PASS')
        # The floor of 1.0 in the limit: 12 eps at |objective| < 1 is beyond the limit 8 eps.
        row.update(recomputed_objective=0.5 + 12 * EPS, objective_difference=12 * EPS)
        self.assertEqual(health._classify_recorded(data), INCOMPLETE_V)
        # Hessian asymmetry exactly equal to its bound (n = 2, max|H| <= 1 so the bound is 64 eps).
        raw = [[1.0, 0.5], [0.5 + 64 * EPS, 1.0]]
        sym = [[1.0, (0.5 + 0.5 + 64 * EPS) / 2], [(0.5 + 0.5 + 64 * EPS) / 2, 1.0]]
        fit = {'hessian_raw': raw, 'hessian': sym, 'hessian_asymmetry': 64 * EPS,
               'hessian_asymmetry_limit': 64 * EPS}
        self.assertEqual(health.hessian_record_status(fit, 2), 'valid')
        raw[1][0] = 0.5 + 96 * EPS  # asymmetry 96 eps > 64 eps with the 1.0 floor: not valid
        sym[0][1] = sym[1][0] = (0.5 + 0.5 + 96 * EPS) / 2
        fit.update(hessian_asymmetry=96 * EPS)
        self.assertEqual(health.hessian_record_status(fit, 2), 'incomplete')

    def test_gradient_on_a_non_free_index_is_not_part_of_the_recorded_maximum(self):
        data = good()
        for row in data['Julia']:
            row.update(free_indices=[1], gradient=[1e-7, 5e-3])  # the second coordinate is pinned
            row['optimizer'].update(optimizer_gradient=[1e-7])
        self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_PASS')
        data['Julia'][1]['gradient'] = [1e-7, 0.0]
        data['Julia'][1]['max_abs_gradient'] = 5e-3  # a scalar that includes a non-free entry contradicts the free maximum
        self.assertEqual(health._classify_recorded(data), INTEGRITY_V)

    def test_every_rejected_record_also_fails_closed_with_the_flag_enabled(self):
        with enabled():
            for label, engines, mutate in (
                    ('reconstruction mismatch', ('Julia',), lambda r: r['reconstruction'].update(
                        loglik_matches=False, recovered_loglik=-999.0)),
                    ('objective unverified', health.ENGINES, lambda r: r.update(objective_verified=False)),
                    ('pd flag False', health.ENGINES, lambda r: r.update(pd_hessian=False))):
                for engine in engines:
                    with self.subTest(label, engine=engine):
                        data = certified_good()
                        mutate(data[engine][0])
                        self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')



def k2_map(pinned_values, q_free=1):
    """lambda_constraint_map for p = 3, K = 2 and the given {(i, k): value} pins.

    Packed layout with q_free = 1: [beta, log sigma, L11, L22, L21, L31, L32] (indices 1..7).
    """
    entries = [[None, None] for _ in range(3)]
    for (i, k), v in pinned_values.items():
        entries[i - 1][k - 1] = v
    return {'layout': health.LAMBDA_MAP_LAYOUT, 'p': 3, 'K': 2, 'q_free': q_free, 'entries': entries}


class PinBindingTests(unittest.TestCase):
    """W1: the pinned indices are recomputed from the fit's own lambda_constraint map."""

    @staticmethod
    def row(pinned_values, pinned, theta=None):
        theta = theta if theta is not None else [0.1, -0.2, 1.0, 0.9, 0.0, 0.0, 0.0]
        theta = list(theta)
        return {'lambda_constraint_map': k2_map(pinned_values), 'pinned_indices': list(pinned), 'parameters': theta}

    def test_map_derived_pins_for_every_loading_position(self):
        cases = [({(2, 1): 0.25}, [5]), ({(3, 1): -0.5}, [6]), ({(3, 2): 0.75}, [7]),
                 ({(1, 1): 1.0}, [3]), ({(2, 2): 0.9}, [4]),
                 ({(2, 1): 0.25, (3, 1): -0.5, (3, 2): 0.75}, [5, 6, 7]), ({}, [])]
        for pins, indices in cases:
            with self.subTest(pins=pins):
                theta = [0.1, -0.2, 1.0, 0.9, 0.0, 0.0, 0.0]
                for (i, k), v in pins.items():
                    theta[{(1, 1): 2, (2, 2): 3, (2, 1): 4, (3, 1): 5, (3, 2): 6}[(i, k)]] = v
                row = self.row(pins, indices, theta)
                self.assertIsNone(health._lambda_pin_binding(row))
                self.assertIsNotNone(health._lambda_pin_binding(dict(row, pinned_indices=[1] if not indices else [])))
        # a structural zero (above the diagonal) is not in theta: a number there pins nothing
        row = self.row({(1, 2): 0.5}, [])
        self.assertIsNone(health._lambda_pin_binding(row))
        row = self.row({(1, 2): 0.5, (2, 1): 0.25}, [5], [0.1, -0.2, 1.0, 0.9, 0.25, 0.0, 0.0])
        self.assertIsNone(health._lambda_pin_binding(row))

    def test_a_rank_three_layout(self):
        # p = 4, K = 3, q_free = 0: theta = [log sigma, L11, L22, L33, L21, L31, L41, L32, L42, L43] (10 entries)
        entries = [[None] * 3 for _ in range(4)]
        theta = [0.0] * 10
        for (i, k), index, value in (((2, 1), 5, 0.5), ((3, 1), 6, 0.6), ((4, 1), 7, 0.7), ((3, 2), 8, 0.8),
                                     ((4, 2), 9, 0.9), ((4, 3), 10, 1.1), ((2, 2), 3, 1.2)):
            entries[i - 1][k - 1] = value
            theta[index - 1] = value
            row = {'lambda_constraint_map': {'layout': health.LAMBDA_MAP_LAYOUT, 'p': 4, 'K': 3, 'q_free': 0,
                                             'entries': [list(r) for r in entries]},
                   'parameters': list(theta), 'pinned_indices': sorted(
                       j for j in (5, 6, 7, 8, 9, 10, 3) if theta[j - 1] != 0.0)}
            with self.subTest(pinned=(i, k)):
                self.assertIsNone(health._lambda_pin_binding(row))
                wrong = dict(row, pinned_indices=[j + 1 if j < 10 else 9 for j in row['pinned_indices']])
                self.assertIsInstance(health._lambda_pin_binding(wrong), str)
        short = dict(row, parameters=theta[:-1])
        self.assertIsInstance(health._lambda_pin_binding(short), str)  # 9 entries is not a rank-3 layout
        self.assertIsInstance(health._lambda_pin_binding(dict(row, parameters=theta + [0.0])), str)

    def test_binding_refuses_a_mismatch(self):
        good_row = self.row({(2, 1): 0.25}, [5], [0.1, -0.2, 1.0, 0.9, 0.25, 0.0, 0.0])
        self.assertIsNone(health._lambda_pin_binding(good_row))
        wrong = {
            'no map': lambda r: r.pop('lambda_constraint_map'),
            'map not a dict': lambda r: r.update(lambda_constraint_map=[1]),
            'extra key': lambda r: r['lambda_constraint_map'].update(extra=1),
            'missing key': lambda r: r['lambda_constraint_map'].pop('entries'),
            'wrong layout': lambda r: r['lambda_constraint_map'].update(layout='other'),
            'p is bool': lambda r: r['lambda_constraint_map'].update(p=True),
            'p is float': lambda r: r['lambda_constraint_map'].update(p=3.0),
            'K is float': lambda r: r['lambda_constraint_map'].update(K=2.0),
            'q_free is float': lambda r: r['lambda_constraint_map'].update(q_free=1.0),
            'p zero': lambda r: r['lambda_constraint_map'].update(p=0),
            'K zero': lambda r: r['lambda_constraint_map'].update(K=0),
            'K above p': lambda r: r['lambda_constraint_map'].update(K=4),
            'q_free negative': lambda r: r['lambda_constraint_map'].update(q_free=-1),
            'entries not a list': lambda r: r['lambda_constraint_map'].update(entries='x'),
            'entries too few rows': lambda r: r['lambda_constraint_map']['entries'].pop(),
            'entries row not a list': lambda r: r['lambda_constraint_map']['entries'].__setitem__(0, 'ab'),
            'entries row too short': lambda r: r['lambda_constraint_map']['entries'][0].pop(),
            'entry NaN': lambda r: r['lambda_constraint_map']['entries'][2].__setitem__(0, float('nan')),
            'entry bool': lambda r: r['lambda_constraint_map']['entries'][2].__setitem__(0, True),
            'entry string': lambda r: r['lambda_constraint_map']['entries'][2].__setitem__(0, 'x'),
            'theta too short': lambda r: r['parameters'].pop(),
            'theta too long': lambda r: r['parameters'].append(0.0),
            'parameters not a list': lambda r: r.update(parameters='x'),
            'declared pins missing one': lambda r: r.update(pinned_indices=[]),
            'declared pins have an extra': lambda r: r.update(pinned_indices=[5, 6]),
            'declared pins are other indices': lambda r: r.update(pinned_indices=[6]),
            'declared pins not a list': lambda r: r.update(pinned_indices=None),
            'theta not at the pinned value': lambda r: r['parameters'].__setitem__(4, 0.2499),
            'theta at the pin differs in the last index': lambda r: (
                r['lambda_constraint_map']['entries'][2].__setitem__(1, 0.5),
                r.update(pinned_indices=[5, 7])),
        }
        for label, mutate in wrong.items():
            with self.subTest(label):
                r = copy.deepcopy(good_row)
                mutate(r)
                self.assertIsInstance(health._lambda_pin_binding(r), str)

    def test_layouts_with_other_dimensions(self):
        # K = 1, no betas: theta = [log sigma, L11, L21, L31]
        m = {'layout': health.LAMBDA_MAP_LAYOUT, 'p': 3, 'K': 1, 'q_free': 0,
             'entries': [[None], [None], [0.5]]}
        row = {'lambda_constraint_map': m, 'pinned_indices': [4], 'parameters': [0.0, 1.0, 0.0, 0.5]}
        self.assertIsNone(health._lambda_pin_binding(row))
        row['pinned_indices'] = [3]
        self.assertIsInstance(health._lambda_pin_binding(row), str)
        # K = p = 2, q_free = 2: theta = [b1, b2, log sigma, L11, L22, L21]
        m = {'layout': health.LAMBDA_MAP_LAYOUT, 'p': 2, 'K': 2, 'q_free': 2,
             'entries': [[None, None], [0.3, None]]}
        row = {'lambda_constraint_map': m, 'pinned_indices': [6], 'parameters': [0.0, 0.0, 0.0, 1.0, 1.0, 0.3]}
        self.assertIsNone(health._lambda_pin_binding(row))
        row['parameters'] = [0.0, 0.0, 0.0, 1.0, 1.0]  # a K = p fit has K(K+1)/2 loadings: length 5 is wrong
        self.assertIsInstance(health._lambda_pin_binding(row), str)

    def test_a_hidden_bad_coordinate_is_not_certified(self):
        """review3 W1 repro: coordinate 3 relabelled as pinned, no map or an empty map."""
        m = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, -9.0]]
        row = writer_like(m, free=[1, 2], gradient=[1e-8, 2e-8, 5.0])
        self.assertEqual(health.certify_fit(row)['verdict'], PASS)  # honest pin: bound, theta at the pinned value
        liar = copy.deepcopy(row)
        liar.pop('lambda_constraint_map')
        out = health.certify_fit(liar)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('lambda_constraint_map', out['reason'])
        liar = copy.deepcopy(row)
        liar['lambda_constraint_map'] = lambda_map(3)  # a map that pins nothing
        out = health.certify_fit(liar)
        self.assertEqual(out['verdict'], INC)
        self.assertIn('pinned indices differ', out['reason'])
        tampered = copy.deepcopy(row)
        tampered['parameters'][2] = 0.001
        tampered['reconstruction']['recovered_parameters'][2] = 0.001
        self.assertEqual(health.certify_fit(tampered)['verdict'], INC)


def writer_like(matrix, free, gradient, radius=1e-9):
    """A certifiable Julia row for ``matrix`` (n x n, K = 1, q_free = 0) with the complement of free pinned at 0."""
    n = len(matrix)
    row = copy.deepcopy(good()['Julia'][0])
    pinned = [i for i in range(1, n + 1) if i not in free]
    row.update(parameters=[0.0] * n, gradient=list(gradient), free_indices=list(free), pinned_indices=pinned,
               max_abs_gradient=max(abs(gradient[i - 1]) for i in free),
               hessian=copy.deepcopy(matrix), hessian_raw=copy.deepcopy(matrix), hessian_asymmetry=0.0,
               hessian_asymmetry_limit=32 * EPS * n * max(1.0, max(abs(x) for r in matrix for x in r)),
               hessian_coordinate=COORD, hessian_error_radius=radius, gradient_error_radius=radius,
               packed_gradient_error_radius=radius)
    row['reconstruction']['recovered_parameters'] = [0.0] * n
    top = row['max_abs_gradient']
    row['optimizer'].update(optimizer_parameters=[0.0] * (n - 1), optimizer_gradient=[top] + [0.0] * (n - 2),
                            optimizer_max_abs_gradient=top)
    return bind(row, pinned)



class OptimizerBindingTests(unittest.TestCase):
    """F1: the optimizer-coordinate block is bound to the fit (objective, point, status)."""

    def row(self):
        return certified_good()['Julia'][0]

    def test_bound_block_certifies(self):
        self.assertIsNone(health._optimizer_binding(self.row()))
        self.assertEqual(health.certify_fit(self.row())['verdict'], PASS)

    def test_each_unbound_leaf_is_refused(self):
        opt = lambda **kw: (lambda r: r['optimizer'].update(**kw))
        cases = {
            'optimizer missing': lambda r: r.pop('optimizer'),
            'reconstruction missing': lambda r: r.pop('reconstruction'),
            'optimizer not a dict': lambda r: r.update(optimizer=[]),
            'reconstruction not a dict': lambda r: r.update(reconstruction=[]),
            'point empty': opt(optimizer_parameters=[]),
            'point not finite': opt(optimizer_parameters=[float('nan')]),
            'point not a list': opt(optimizer_parameters='ab'),
            'gradient not finite': opt(optimizer_gradient=[float('inf')]),
            'gradient empty': opt(optimizer_gradient=[]),
            'gradient length differs': opt(optimizer_gradient=[1e-7, 0.0]),
            'scalar is not the maximum': opt(optimizer_max_abs_gradient=5e-8),
            'scalar missing': lambda r: r['optimizer'].pop('optimizer_max_abs_gradient'),
            'profile unverified': opt(optimizer_profile_verified=False),
            'profile verified not a bool': opt(optimizer_profile_verified=1),
            'optimizer error recorded': opt(optimizer_error='x'),
            'optimizer error key missing': lambda r: r['optimizer'].pop('optimizer_error'),
            'profile value differs': opt(optimizer_profile_value=3.0 + 1e-9),
            'profile value NaN': opt(optimizer_profile_value=float('nan'), optimizer_minimum=float('nan')),
            'minimum NaN': opt(optimizer_minimum=float('nan')),
            'minimum disagrees with the fitted objective': opt(optimizer_profile_value=999.0, optimizer_minimum=999.0),
            'fitted objective missing': lambda r: r.pop('optimizer_objective'),
            'fitted objective NaN': lambda r: r.update(optimizer_objective=float('nan')),
            'reconstruction from another point': lambda r: r['reconstruction'].update(optimizer_parameters=[42.0, 0.0]),
            'reconstruction point missing': lambda r: r['reconstruction'].pop('optimizer_parameters'),
            'optimizer radius status missing': lambda r: r['optimizer'].pop('radius_status'),
            'optimizer radius status proven': opt(radius_status='proven'),
            'record radius status missing': lambda r: r.pop('radius_status'),
            'record radius status proven': lambda r: r.update(radius_status='proven'),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                r = self.row()
                mutate(r)
                self.assertIsInstance(health._optimizer_binding(r), str)
                self.assertEqual(health.certify_fit(r)['verdict'], INC)

    def test_objective_agreement_follows_the_roundoff_rule_exactly(self):
        """|minimum - objective| <= 8 eps max(1, |minimum|, |objective|), boundary included."""
        def with_values(objective, minimum):
            r = self.row()
            r.update(optimizer_objective=objective)
            r['optimizer'].update(optimizer_minimum=minimum, optimizer_profile_value=minimum)
            return r
        self.assertIsNone(health._optimizer_binding(with_values(0.5, 0.5 + 8 * EPS)))  # equal to the limit
        self.assertIsInstance(health._optimizer_binding(with_values(0.5, 0.5 + 8.5 * EPS)), str)
        self.assertIsNone(health._optimizer_binding(with_values(0.5 + 8 * EPS, 0.5)))
        big = 4096.0  # limit scales with the larger magnitude: 8 eps * 4096
        self.assertIsNone(health._optimizer_binding(with_values(big, big + 8 * EPS * big)))
        self.assertIsInstance(health._optimizer_binding(with_values(big, big + 9 * EPS * big)), str)

    def test_review3_f1_repros_are_refused(self):
        r = self.row()
        r['optimizer'].update(optimizer_minimum=999.0, optimizer_profile_value=999.0)
        self.assertEqual(health.certify_fit(r)['verdict'], INC)  # (A) contradictory objective
        r = self.row()
        r['optimizer'].update(optimizer_parameters=[42.0], optimizer_gradient=[1e-7])
        self.assertEqual(health.certify_fit(r)['verdict'], INC)  # (B) point not the reconstruction's
        data = certified_good()
        data['Julia'][0]['optimizer'].update(optimizer_minimum=999.0, optimizer_profile_value=999.0)
        with enabled():
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')


class CertifyFitTests(unittest.TestCase):
    """M2, M3/W4, F3, F5 and W2/W3 at the per-fit level."""

    def test_m2_certify_fit_alone_needs_the_packed_gradient_vector(self):
        vec = 'packed gradient vector missing'
        free = 'free indices missing or invalid'
        for label, mutate, reason in (
            ('vector missing', lambda r: r.pop('gradient'), vec),
            ('vector empty', lambda r: r.update(gradient=[]), vec),
            ('vector not a list', lambda r: r.update(gradient='ab'), vec),
            ('vector non-finite', lambda r: r.update(gradient=[float('nan'), 0.0]), vec),
            ('vector shorter than parameters', lambda r: r.update(gradient=[1e-7]), vec),
            ('parameters missing', lambda r: r.pop('parameters'), vec),
            ('parameters non-finite', lambda r: r.update(parameters=[float('inf'), 0.0]), vec),
            ('free indices missing', lambda r: r.pop('free_indices'), free),
            ('free indices empty', lambda r: r.update(free_indices=[]), free),
            ('free indices not a list', lambda r: r.update(free_indices='12'), free),
            ('free index out of range', lambda r: r.update(free_indices=[1, 3]), free),
            ('free index zero', lambda r: r.update(free_indices=[0, 1]), free),
            ('free index duplicated', lambda r: r.update(free_indices=[1, 1]), free),
            ('free index bool', lambda r: r.update(free_indices=[True, 2]), free),
            ('free index float', lambda r: r.update(free_indices=[1.0, 2]), free),
            ('scalar is not the free maximum', lambda r: r.update(max_abs_gradient=5e-8), 'not the maximum over the free'),
            ('scalar missing', lambda r: r.pop('max_abs_gradient'), 'not the maximum over the free'),
        ):
            with self.subTest(label):
                row = certified_good()['Julia'][0]
                mutate(row)
                out = health.certify_fit(row)
                self.assertEqual(out['verdict'], INC)
                self.assertIn(reason, out['reason'])  # refused for this reason, not by a caught exception

    def test_m3_the_two_gradient_bounds_are_independent_and_both_required(self):
        base = certified_good()['Julia'][0]
        self.assertEqual(health.certify_fit(base)['verdict'], PASS)
        for key in ('gradient_error_radius', 'packed_gradient_error_radius'):
            for bad in (None, 0.0, -1e-9, float('nan'), True, '1e-9', 1e-30):
                with self.subTest(key=key, bad=bad):
                    row = copy.deepcopy(base)
                    row[key] = bad
                    self.assertEqual(health.certify_fit(row)['verdict'], INC)
        # a packed gradient large enough to straddle the limit blocks a pass but is never a failure
        row = copy.deepcopy(base)
        row.update(gradient=[2e-5, 0.0], max_abs_gradient=2e-5)
        out = health.certify_fit(row)
        self.assertEqual((out['verdict'], out['gradient']), (INC, INC))
        # ... and its radius matters: the radius may not close the gap
        row = copy.deepcopy(base)
        row.update(gradient=[9e-6, 0.0], max_abs_gradient=9e-6, packed_gradient_error_radius=2e-6)
        self.assertEqual(health.certify_fit(row)['verdict'], INC)
        row['packed_gradient_error_radius'] = 5e-7
        self.assertEqual(health.certify_fit(row)['verdict'], PASS)
        # optimizer gradient is the signed rule: its failure is a failure, its straddle is not
        for opt_value, verdict in ((2e-5, FAIL), (9e-6, INC), (9e-7, PASS)):
            row = copy.deepcopy(base)
            row['optimizer'].update(optimizer_gradient=[opt_value], optimizer_max_abs_gradient=opt_value)
            row['gradient_error_radius'] = 1e-6
            with self.subTest(opt_value=opt_value):
                self.assertEqual(health.certify_fit(row)['verdict'], verdict)

    def test_pinned_coordinate_gradient_is_not_part_of_the_free_maximum(self):
        m = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, -9.0]]
        row = writer_like(m, [1, 2], [1e-8, 2e-8, 5.0])  # pinned third coordinate with a large gradient
        self.assertEqual(row['max_abs_gradient'], 2e-8)
        self.assertEqual(health.certify_fit(row)['verdict'], PASS)
        row['gradient'] = [5.0, 2e-8, 1e-8]  # large gradient on a free coordinate
        row['max_abs_gradient'] = 5.0
        self.assertEqual(health.certify_fit(row)['verdict'], INC)
        row = writer_like(m, [1, 2], [2e-8, 1e-8, 5.0])
        row['max_abs_gradient'] = 5.0  # a scalar that includes the pinned entry is not the free maximum
        self.assertEqual(health.certify_fit(row)['verdict'], INC)

    def test_reasons_name_the_blocking_part(self):
        row = certified_good()['Julia'][0]
        row['packed_gradient_error_radius'] = 1e-30  # below the representation floor of the gradient
        out = health.certify_fit(row)
        self.assertEqual(out['reason'], 'gradient bound not certified')
        row['packed_gradient_error_radius'] = None
        self.assertEqual(health.certify_fit(row)['reason'], 'gradient error radius not recorded or not positive')
        row = certified_good()['Julia'][0]
        row['hessian_error_radius'] = None
        self.assertIn('radius', health.certify_fit(row)['reason'])

    def test_f3_a_gradient_radius_of_exactly_zero_is_refused(self):
        self.assertEqual(health.certify_gradient(0.0, 0.0), INC)
        self.assertEqual(health.certify_gradient(1e-7, 0.0), INC)
        self.assertEqual(health.certify_gradient(1e-7, -0.0), INC)
        data = certified_good()
        for row in data['Julia']:
            row.update(gradient=[0.0, 0.0], max_abs_gradient=0.0, gradient_error_radius=0.0,
                       packed_gradient_error_radius=0.0)
            row['optimizer'].update(optimizer_gradient=[0.0], optimizer_max_abs_gradient=0.0)
        with enabled():
            self.assertEqual(health._classify_recorded(data), 'FIT_HEALTH_PASS')
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            for row in data['Julia']:
                row['gradient_error_radius'] = row['packed_gradient_error_radius'] = 1e-9
            self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')  # zero gradient, positive radius

    def test_f4_the_radius_floor_uses_the_peak_of_the_free_block(self):
        eps = 2.0 ** -52
        # off-diagonal peak 3.0 in the free block {2, 3} of a 3 x 3 matrix
        m = [[1.0, 0.0, 0.0], [0.0, 0.5, 3.0], [0.0, 3.0, 0.5]]
        floor = 2 * eps * 3.0
        out = hess(m, [2, 3], math.nextafter(floor, 0.0), pinned_indices=[1])
        self.assertEqual(out['verdict'], INC)
        self.assertIn('floor', out['reason'])
        self.assertEqual(hess(m, [2, 3], floor, pinned_indices=[1])['verdict'], FAIL)  # indefinite: certified
        # the pinned coordinate does not raise the floor
        m = [[1e30, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]
        self.assertEqual(hess(m, [2, 3], 1e-9, pinned_indices=[1])['verdict'], PASS)
        # peak in the last free cell of a free block that does not start at index 1
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
        floor = 2 * eps * 4.0
        self.assertEqual(hess(m, [2, 3], math.nextafter(floor, 0.0), pinned_indices=[1])['verdict'], INC)
        self.assertEqual(hess(m, [2, 3], floor, pinned_indices=[1])['verdict'], PASS)
        m = [[1.0, 0.0], [0.0, 4.0]]
        floor = 2 * eps * 4.0
        self.assertEqual(hess(m, [1, 2], math.nextafter(floor, 0.0))['verdict'], INC)
        self.assertEqual(hess(m, [1, 2], floor)['verdict'], PASS)
        # peak in a leading cell only
        m = [[4.0, 0.0], [0.0, 1.0]]
        self.assertEqual(hess(m, [1, 2], math.nextafter(floor, 0.0))['verdict'], INC)
        self.assertEqual(hess(m, [2, 1], floor)['verdict'], PASS)

    def test_f5_a_certified_hessian_failure_needs_a_well_formed_optimizer_block_and_radii(self):
        def failing():
            data = certified_good()
            row = data['Julia'][0]
            row['hessian'] = row['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
            return data, row
        data, row = failing()
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')
        for label, mutate in {
            'optimizer parameters a string': lambda r: r['optimizer'].update(optimizer_parameters='x'),
            'gradient radius a bool': lambda r: r.update(gradient_error_radius=True),
            'gradient radius missing': lambda r: r.pop('gradient_error_radius'),
            'gradient radius zero': lambda r: r.update(gradient_error_radius=0.0),
            'packed radius zero': lambda r: r.update(packed_gradient_error_radius=0.0),
            'packed radius negative': lambda r: r.update(packed_gradient_error_radius=-1e-9),
            'packed radius missing': lambda r: r.pop('packed_gradient_error_radius'),
            'optimizer gradient missing': lambda r: r['optimizer'].pop('optimizer_gradient'),
            'optimizer scalar wrong': lambda r: r['optimizer'].update(optimizer_max_abs_gradient=5e-8),
            'unbound optimizer point': lambda r: r['reconstruction'].update(optimizer_parameters=[7.0, 0.0]),
        }.items():
            with self.subTest(label):
                data, row = failing()
                mutate(row)
                with enabled():
                    self.assertNotIn(health.classify(data), ('FIT_HEALTH_FAIL', 'FIT_HEALTH_FAIL_WITH_INCOMPLETE'))
                self.assertEqual(health.certify_fit(row)['verdict'], INC)

    def test_w2_pass_carries_the_declared_unproven_radius_status(self):
        row = certified_good()['Julia'][0]
        self.assertEqual(health.certify_fit(row)['radius_status'], 'declared_unproven')
        self.assertEqual(health.RADIUS_STATUS, 'declared_unproven')
        row['hessian_error_radius'] = None
        self.assertIsNone(health.certify_fit(row)['radius_status'])  # nothing certified, nothing qualified
        data = certified_good()
        with enabled():
            verdict = health.classify(data)
            self.assertEqual(verdict, 'FIT_HEALTH_PASS')
            self.assertEqual(verdict.radius_status, 'declared_unproven')
            self.assertEqual(json.loads(json.dumps(verdict)), 'FIT_HEALTH_PASS')  # still a plain string
            data['Julia'][1]['radius_status'] = 'proven'  # no proof procedure exists: refused
            other = health.classify(data)
            self.assertEqual(other, 'FIT_HEALTH_INCOMPLETE')
            self.assertIsNone(getattr(other, 'radius_status', None))
        self.assertIsNone(getattr(health.classify(certified_good()), 'radius_status', None))  # held: no pass at all


class EvidenceBindingTests(unittest.TestCase):
    """F7: case and dimension binding across the six records (necessary consistency only)."""

    @staticmethod
    def with_q(data, qs):
        for row, q in zip(data['Julia'], qs):
            row['lambda_constraint_map']['q_free'] = q
        return data

    def test_consistent_evidence_is_bound(self):
        data = certified_good()
        self.assertIsNone(health._evidence_binding(data))
        # Different dimensions per case: p = 1, K = 1 gives q_free + 2 packed parameters; R and Julia
        # optimizer points follow with one constant offset (R carries one more parameter here).
        data = self.with_q(certified_good(), [0, 1, 2])
        for i, q in enumerate([0, 1, 2]):
            data['Julia'][i]['parameters'] = [0.0] * (q + 2)
            data['Julia'][i]['optimizer']['optimizer_parameters'] = [0.0] * (q + 1)
            data['R'][i]['parameters'] = [0.0] * (q + 2)
        self.assertIsNone(health._evidence_binding(data))
        data['R'][2]['parameters'] = [0.0] * 5  # offset no longer constant
        self.assertIsInstance(health._evidence_binding(data), str)

    def test_each_inconsistency_is_refused(self):
        def reason(mutate):
            data = certified_good()
            mutate(data)
            return health._evidence_binding(data)
        cases = {
            'p differs across cases': lambda d: d['Julia'][1]['lambda_constraint_map'].update(p=2),
            'K differs across cases': lambda d: d['Julia'][2]['lambda_constraint_map'].update(K=2),
            'no-fixed-effects case has coefficients': lambda d: self.with_q(d, [1, 1, 1]),
            'coefficient counts decrease': lambda d: self.with_q(d, [0, 2, 1]),
            'q_free not an int': lambda d: d['Julia'][1]['lambda_constraint_map'].update(q_free=0.0),
            'map missing': lambda d: d['Julia'][1].pop('lambda_constraint_map'),
            'optimizer point missing': lambda d: d['Julia'][1]['optimizer'].pop('optimizer_parameters'),
            'R parameters missing': lambda d: d['R'][1].pop('parameters'),
            'R/Julia counts bound to different cases': lambda d: d['R'][2].update(parameters=[0.0, 0.0, 0.0],
                                                                                  gradient=[1e-7, 0.0, 0.0],
                                                                                  free_indices=[1, 2, 3]),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                self.assertIsInstance(reason(mutate), str)

    def test_unbound_evidence_cannot_pass_or_confirm_a_failure(self):
        data = certified_good()
        data['Julia'][2]['lambda_constraint_map'].update(p=2)  # layout length also off: certify_fit refuses it too
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        # internally consistent per row, inconsistent across rows: still refused
        data = certified_good()
        row = data['Julia'][2]
        three_parameter_julia([row], [1, 2], pinned=None)
        row.pop('pinned_indices', None)
        row['pinned_indices'] = []
        row['free_indices'] = [1, 2, 3]
        row['hessian'] = row['hessian_raw'] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        row['hessian_asymmetry_limit'] = 32 * EPS * 3
        bind(row, ())
        self.assertEqual(health.certify_fit(row)['verdict'], PASS)  # the row alone is certifiable
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            certs, confirmable = health.certify_evidence(data)
        self.assertEqual([c['verdict'] for c in certs], [INC, INC, INC])  # an unbound case set passes nothing
        self.assertFalse(confirmable)
        self.assertTrue(all(c['confirmable'] is False for c in certs))
        # a failure on an unbound case set is not confirmed either
        data = certified_good()
        data['Julia'][0]['hessian'] = data['Julia'][0]['hessian_raw'] = [[-1.0, 0.0], [0.0, 1.0]]
        self.assertTrue(health.certify_evidence(data)[1])
        data['Julia'][1]['lambda_constraint_map']['q_free'] = 5  # decreasing counts along the case order
        data['Julia'][1]['parameters'] = [0.0] * 7
        data['Julia'][2]['lambda_constraint_map']['q_free'] = 0
        certs, confirmable = health.certify_evidence(data)
        self.assertFalse(confirmable)
        self.assertEqual((certs[0]['verdict'], certs[0]['confirmable']), (FAIL, False))
        with enabled():
            self.assertNotEqual(health.classify(data), 'FIT_HEALTH_FAIL')

    def test_certify_evidence_shape_results(self):
        self.assertEqual(health.certify_evidence({}), ([], False))
        self.assertEqual(health.certify_evidence({'Julia': 'x', 'R': []}), ([], False))
        short = certified_good()
        short['Julia'].pop()
        self.assertEqual(health.certify_evidence(short), ([], False))
        dup = certified_good()
        dup['Julia'][2]['case'] = dup['Julia'][1]['case']
        certs, confirmable = health.certify_evidence(dup)
        self.assertEqual((len(certs), confirmable), (3, False))
        other = certified_good()
        other['Julia'][2]['case'] = 'other'  # a unique string that is not a case: the set differs, the count does not
        certs, confirmable = health.certify_evidence(other)
        self.assertEqual((len(certs), confirmable), (3, False))
        unbound = certified_good()
        unbound['Julia'][2]['objective_verified'] = False
        certs, confirmable = health.certify_evidence(unbound)
        self.assertFalse(confirmable)
        self.assertTrue(certs[0]['confirmable'])  # the other rows are unaffected
        ok_certs, ok_conf = health.certify_evidence(certified_good())
        self.assertTrue(ok_conf)
        self.assertEqual([c['verdict'] for c in ok_certs], [PASS] * 3)
        self.assertTrue(all(c['confirmable'] for c in ok_certs))
        bad_r = certified_good()
        bad_r['R'] = 'garbage'
        certs, confirmable = health.certify_evidence(bad_r)
        self.assertEqual([c['verdict'] for c in certs], [PASS] * 3)  # still certified, never confirmable
        self.assertFalse(any(c['confirmable'] for c in certs))



class WorkCapTests(unittest.TestCase):
    """M5: the verdict is a function of the record alone; a deterministic work cap replaces wall-clock."""

    def test_no_clock_is_consulted(self):
        self.assertFalse(hasattr(health, 'time'))
        self.assertFalse(hasattr(health, 'CERT_MAX_SECONDS'))
        with patch('time.monotonic', side_effect=AssertionError('the certifier must not read the clock')):
            self.assertEqual(hess([[2.0, 0.5], [0.5, 1.0]], [1, 2])['verdict'], PASS)

    def test_work_meter_arithmetic(self):
        w = health._Work(10)
        self.assertEqual((w.cap, w.used), (10, 0))
        w.charge(Fraction(1))  # numerator 1 bit + denominator 1 bit = 2 bits: 1 + 2 // 64
        self.assertEqual(w.used, 1)
        w.charge(Fraction(2 ** 31 - 1, 2 ** 31), Fraction(2 ** 31 - 1, 2 ** 31))  # 31 + 32 bits each
        self.assertEqual(w.used, 1 + 1 + 126 // 64)
        w.charge(Fraction(2 ** 63 - 1, 1), Fraction(1))  # 63 + 1 + 1 + 1 = 66 bits
        self.assertEqual(w.used, 1 + 2 + 1 + 66 // 64)
        w = health._Work(10)
        w.charge(Fraction(2 ** 62))  # exactly 64 bits (63 + 1): the divisor is 64, not 65
        self.assertEqual(w.used, 2)
        w = health._Work(3)
        for _ in range(3):
            w.charge(Fraction(1))  # used == cap is allowed
        with self.assertRaises(health._CertRefusal):
            w.charge(Fraction(1))  # cap + 1 is refused
        w = health._Work(0)
        with self.assertRaises(health._CertRefusal):
            w.charge(Fraction(1))

    def test_work_units_are_deterministic_and_recorded(self):
        ident = lambda k: [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        first = hess(ident(3), [1, 2, 3])
        self.assertEqual(first['verdict'], PASS)
        self.assertEqual(first['work_units'], hess(ident(3), [1, 2, 3])['work_units'])
        self.assertEqual(hess([[2.0]], [1])['work_units'], 0)  # one pivot, no multiply-adds
        self.assertEqual(hess([[2.0, 0.5], [0.5, 1.0]], [1, 2])['work_units'], 9)  # the pinned count is deterministic
        self.assertEqual(first['work_units'], 21)
        self.assertEqual(hess([[-1.0]], [1])['work_units'], 0)
        refused = hess(ident(3), [1, 2, 3], coordinate='nope')
        self.assertEqual(refused['work_units'], 0)  # refused before any work
        # FAIL runs the second (indefinite) pass as well
        self.assertEqual(hess([[1.0, 2.0], [2.0, 1.0]], [1, 2])['verdict'], FAIL)

    def test_cap_refusal_is_reproducible_and_clock_free(self):
        ident = lambda k: [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        need = hess(ident(3), [1, 2, 3])['work_units']
        ok = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_work=need, parameter_count=3)
        self.assertEqual(ok['verdict'], PASS)
        low = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_work=need - 1, parameter_count=3)
        self.assertEqual(low['verdict'], INC)
        self.assertIn('work cap', low['reason'])
        self.assertEqual(low['work_units'], need)  # the meter stops at the first charge above the cap
        for _ in range(2):
            again = health.certify_hessian(ident(3), [1, 2, 3], COORD, 1e-9, 0.0, [], max_work=need - 1,
                                           parameter_count=3)
            self.assertEqual((again['verdict'], again['reason']), (low['verdict'], low['reason']))

    def test_the_default_cap_admits_a_full_block_and_refuses_an_enormous_exact_computation(self):
        self.assertEqual(health.CERT_MAX_WORK, 2000000)
        self.assertEqual(health.CERT_MAX_FREE_DIMENSION, 32)
        import random
        rnd = random.Random(3)

        def block(span):
            n = 32
            m = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i):
                    m[i][j] = m[j][i] = rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-span, span)
            for i in range(n):
                m[i][i] = 1e3 * 2.0 ** rnd.randint(-span, span) + sum(abs(x) for x in m[i])
            return m
        for span, verdict, reason in ((40, PASS, None), (300, INC, 'work cap')):
            m = block(span)
            peak = max(abs(x) for r in m for x in r)
            out = health.certify_hessian(m, list(range(1, 33)), COORD, 1024 * 2.0 ** -52 * max(1.0, peak), 0.0, [],
                                         parameter_count=32)
            with self.subTest(span=span):
                self.assertEqual(out['verdict'], verdict)
                if reason:
                    self.assertIn(reason, out['reason'])

    def test_size_cap_literals(self):
        ident = lambda k: [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        self.assertEqual(hess(ident(32), list(range(1, 33)))['verdict'], PASS)
        out = hess(ident(33), list(range(1, 34)))
        self.assertEqual(out['verdict'], INC)
        self.assertIn('size cap of 32', out['reason'])


class HostileObjectTests(unittest.TestCase):
    """F6: classify and certify_fit never raise on objects whose comparison or hashing raises."""

    class Boom:
        __hash__ = None

        def __eq__(self, other):
            raise RuntimeError('boom')
        __ne__ = __eq__
        __lt__ = __gt__ = __le__ = __ge__ = __eq__

        def __bool__(self):
            raise RuntimeError('boom')

    def test_classify_returns_incomplete_never_raises(self):
        for flag in (False, True):
            for engine in health.ENGINES:
                keys = ['case', 'pd_hessian', 'max_abs_gradient', 'gradient', 'parameters', 'free_indices', 'error']
                keys += (['fixed_prior_state', 'ridge_tau'] if engine == 'R' else
                         ['penalty_state', 'hessian', 'optimizer', 'reconstruction', 'lambda_constraint_map',
                          'hessian_error_radius', 'radius_status'])
                for key in keys:
                    with self.subTest(flag=flag, engine=engine, key=key):
                        data = certified_good()
                        data[engine][0][key] = self.Boom()
                        with patch.object(health, 'NUMERICAL_ADMISSION_IMPLEMENTED', flag):
                            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = certified_good()
        data['Julia'][0]['penalty_state'] = self.Boom()
        with self.assertRaises(RuntimeError):
            health._classify_recorded(data)  # the unguarded layer still raises: only classify is total

    def test_certify_fit_returns_incomplete_never_raises(self):
        row = certified_good()['Julia'][0]
        row['hessian_error_radius'] = self.Boom()
        out = health.certify_fit(row)
        self.assertEqual(out['verdict'], INC)
        self.assertEqual(set(out), {'verdict', 'reason', 'hessian', 'gradient', 'radius_status'})
        self.assertEqual(health.certify_fit(self.Boom())['verdict'], INC)

    def test_deep_nesting_and_cycles_are_incomplete(self):
        data = certified_good()
        cycle = []
        cycle.append(cycle)
        data['Julia'][0]['hessian'] = cycle
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        deep = []
        for _ in range(100000):
            deep = [deep]
        data = certified_good()
        data['R'][0]['gradient'] = deep
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')


class HelperUnitTests(unittest.TestCase):
    """Direct controls for the helper predicates (F2/F8 survivors)."""

    def test_finite_vector(self):
        self.assertTrue(health._finite_vector([1.0, 2]))
        for bad in ([], (1.0,), 'ab', None, [float('nan')], [1.0, True], [1.0, '2'], [1, float('inf')]):
            self.assertFalse(health._finite_vector(bad))

    def test_hessian_pd_shape_and_numeric_checks(self):
        ident = [[1.0, 0.0], [0.0, 1.0]]
        self.assertIs(health.hessian_pd(ident, [1, 2], 2), True)
        for label, matrix in {'not a list': 'ab', 'rows tuples': ((1.0, 0.0), (0.0, 1.0)),
                              'ragged': [[1.0, 0.0], [0.0]], 'non-finite': [[1.0, 0.0], [0.0, float('nan')]],
                              'asymmetric': [[1.0, 0.1], [0.0, 1.0]], 'wrong size': [[1.0]],
                              'bool entry': [[True, 0.0], [0.0, 1.0]]}.items():
            with self.subTest(label):
                self.assertIsNone(health.hessian_pd(matrix, [1, 2], 2))
        self.assertIsNone(health.hessian_pd([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [1, 2], 2))
        self.assertIsNone(health.hessian_pd([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1, 2], 3))
        # pivots: exactly zero is not positive definite; the 3 x 3 recursion uses both products
        self.assertIs(health.hessian_pd([[1.0, 1.0], [1.0, 1.0]], [1, 2], 2), False)
        self.assertIs(health.hessian_pd([[1.0, 1.1], [1.1, 1.0]], [1, 2], 2), False)
        self.assertIs(health.hessian_pd([[0.0]], [1], 1), False)
        dense = [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]
        self.assertIs(health.hessian_pd(dense, [1, 2, 3], 3), True)
        notpd = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
        self.assertIs(health.hessian_pd(notpd, [1, 2, 3], 3), False)
        self.assertIs(health.hessian_pd([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]], [1, 2, 3], 3), True)
        # only the free block matters
        pinned = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -5.0]]
        self.assertIs(health.hessian_pd(pinned, [1, 2], 3), True)
        self.assertIs(health.hessian_pd(pinned, [1, 3], 3), False)

    def test_hessian_record_status_shape_and_precedence(self):
        def fit(raw, sym=None, asym=0.0, limit=None):
            n = len(raw)
            limit = 32 * EPS * n * max(1.0, max(abs(x) for r in raw for x in r)) if limit is None else limit
            return {'hessian_raw': raw, 'hessian': sym if sym is not None else raw,
                    'hessian_asymmetry': asym, 'hessian_asymmetry_limit': limit}
        ident = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(health.hessian_record_status(fit(ident), 2), 'valid')
        self.assertTrue(health.hessian_roundoff_valid(fit(ident), 2))
        self.assertFalse(health.hessian_roundoff_valid(fit(ident, asym=1e-9), 2))
        self.assertFalse(health.hessian_roundoff_valid(fit(ident, sym=[[1.0, 0.1], [0.1, 1.0]]), 2))
        self.assertFalse(health.hessian_roundoff_valid({}, 2))
        for label, bad in {'raw missing': lambda f: f.pop('hessian_raw'), 'sym missing': lambda f: f.pop('hessian'),
                           'raw ragged': lambda f: f.update(hessian_raw=[[1.0, 0.0], [0.0]]),
                           'sym ragged': lambda f: f.update(hessian=[[1.0], [0.0, 1.0]]),
                           'raw non-finite': lambda f: f.update(hessian_raw=[[1.0, 0.0], [0.0, float('inf')]]),
                           'raw not a list of lists': lambda f: f.update(hessian_raw=['ab', 'cd']),
                           'raw wrong size': lambda f: f.update(hessian_raw=[[1.0]]),
                           'asym missing': lambda f: f.pop('hessian_asymmetry'),
                           'limit missing': lambda f: f.pop('hessian_asymmetry_limit'),
                           'asym NaN': lambda f: f.update(hessian_asymmetry=float('nan')),
                           'limit NaN': lambda f: f.update(hessian_asymmetry_limit=float('nan'))}.items():
            with self.subTest(label):
                f = fit(copy.deepcopy(ident))
                bad(f)
                self.assertEqual(health.hessian_record_status(f, 2), 'incomplete')
        # a non-finite asymmetry (finite entries whose difference overflows) cannot be 'valid'
        huge = [[1.0, 1e308], [-1e308, 1.0]]
        self.assertEqual(health.hessian_record_status(fit(huge, sym=[[1.0, 0.0], [0.0, 1.0]], asym=float('inf')), 2),
                         'incomplete')
        # ... and a finite recorded value does not turn that into an integrity finding
        self.assertEqual(health.hessian_record_status(fit(huge, sym=[[1.0, 0.0], [0.0, 1.0]], asym=1.0), 2),
                         'incomplete')
        # sym that is not the symmetrised raw: integrity, which outranks a bad recorded scalar
        f = fit([[1.0, 0.5], [0.5, 1.0]], sym=[[1.0, 0.4], [0.4, 1.0]])
        self.assertEqual(health.hessian_record_status(f, 2), 'integrity')
        f['hessian_asymmetry'] = float('nan')
        self.assertEqual(health.hessian_record_status(f, 2), 'integrity')

    def test_certify_hessian_shape_refusals(self):
        ok = [[2.0, 0.5], [0.5, 1.0]]
        self.assertEqual(hess(ok, [1, 2])['verdict'], PASS)
        for matrix in (None, [], 'ab', 5):
            out = health.certify_hessian(matrix, [1], COORD, 1e-9, 0.0, [], parameter_count=1)
            self.assertEqual((out['verdict'], out['reason']), (INC, 'matrix missing or malformed'))
        for label, matrix in {'None': None, 'empty': [], 'string': 'ab', 'ragged': [[1.0, 0.0], [0.0]],
                              'tuple rows': ((1.0, 0.0), (0.0, 1.0)), 'non-square': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                              'NaN': [[1.0, 0.0], [0.0, float('nan')]], 'bool': [[True, 0.0], [0.0, 1.0]],
                              'int huge': [[10 ** 400, 0], [0, 1]]}.items():
            with self.subTest(label):
                out = health.certify_hessian(matrix, [1, 2], COORD, 1e-9, 0.0, [], parameter_count=2)
                self.assertEqual(out['verdict'], INC)
        for label, free in {'None': None, 'empty': [], 'string': '12', 'zero': [0, 1], 'above': [1, 3],
                            'float': [1.0, 2.0], 'bool': [True, 2], 'duplicate': [1, 1]}.items():
            with self.subTest(free=label):
                self.assertEqual(health.certify_hessian(ok, free, COORD, 1e-9, 0.0, [], parameter_count=2)['verdict'], INC)
        out = health.certify_hessian(ok, [1, 2], COORD, 1e-9, float('nan'), [], parameter_count=2)
        self.assertEqual(out['verdict'], INC)
        out = health.certify_hessian(ok, [1, 2], COORD, 1e-9, 0.0, [], max_dimension=1, parameter_count=2)
        self.assertEqual(out['verdict'], INC)
        out = health.certify_hessian(ok, [1, 2], COORD, 1e-9, 0.0, [], max_dimension=2, parameter_count=2)
        self.assertEqual(out['verdict'], PASS)  # exactly at the cap is allowed

    def test_record_checks_valid_direct(self):
        data = certified_good()
        row = data['Julia'][0]
        r_valid = health._r_block_valid(data)
        self.assertTrue(health._record_checks_valid(row, r_valid))
        for label, bad in {'gradient missing': lambda r: r.pop('gradient'), 'gradient empty': lambda r: r.update(gradient=[]),
                           'gradient a string': lambda r: r.update(gradient='ab'),
                           'optimizer missing': lambda r: r.pop('optimizer'),
                           'optimizer not a dict': lambda r: r.update(optimizer=[]),
                           'profile unverified': lambda r: r['optimizer'].update(optimizer_profile_verified=False),
                           'optimizer error key missing': lambda r: r['optimizer'].pop('optimizer_error'),
                           'optimizer error recorded': lambda r: r['optimizer'].update(optimizer_error='x'),
                           'value NaN': lambda r: r['optimizer'].update(optimizer_profile_value=float('nan')),
                           'minimum NaN': lambda r: r['optimizer'].update(optimizer_minimum=float('nan')),
                           'value differs': lambda r: r['optimizer'].update(optimizer_profile_value=3.5),
                           'reconstruction missing': lambda r: r.pop('reconstruction'),
                           'parameters_match false': lambda r: r['reconstruction'].update(parameters_match=False)}.items():
            with self.subTest(label):
                r = copy.deepcopy(row)
                bad(r)
                self.assertFalse(health._record_checks_valid(r, r_valid))
        # objective roundoff boundary inside the record checks (limit 8 eps at |objective| < 1)
        r = copy.deepcopy(row)
        r.update(optimizer_objective=0.5, recomputed_objective=0.5 + 8 * EPS, objective_difference=8 * EPS,
                 objective_roundoff_limit=8 * EPS)
        r['optimizer'].update(optimizer_minimum=0.5, optimizer_profile_value=0.5)
        r['reconstruction'].update(recovered_loglik=-0.5)
        self.assertTrue(health._record_checks_valid(r, r_valid))
        r.update(recomputed_objective=0.5 + 12 * EPS, objective_difference=12 * EPS)
        self.assertFalse(health._record_checks_valid(r, r_valid))


def row_n(n, q=0, optimizer_length=None):
    """A certifiable Julia row with n packed parameters (K = 1, p = n - 1 - q, no pins)."""
    p = n - 1 - q
    row = copy.deepcopy(certified_good()['Julia'][0])
    m = n - 1 if optimizer_length is None else optimizer_length
    row.update(parameters=[0.0] * n, gradient=[1e-7] + [0.0] * (n - 1), free_indices=list(range(1, n + 1)),
               hessian=[[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
               hessian_raw=[[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
               hessian_asymmetry_limit=32 * 2.0 ** -52 * n)
    row['reconstruction']['recovered_parameters'] = [0.0] * n
    row['optimizer'].update(optimizer_parameters=[0.0] * m, optimizer_gradient=[1e-7] + [0.0] * (m - 1))
    bind(row)
    row['lambda_constraint_map'] = {'layout': health.LAMBDA_MAP_LAYOUT, 'p': p, 'K': 1, 'q_free': q,
                                    'entries': [[None] for _ in range(p)]}
    return row


def scaled_evidence(qs=(0, 5, 6), p=5, r_lengths=None):
    """Six records with per-case dimensions: Julia n = q + 1 + p, R n = optimizer length + 1 by default."""
    data = certified_good()
    for i, q in enumerate(qs):
        row = row_n(q + 1 + p, q)
        row['case'] = health.CASES[i]
        data['Julia'][i] = row
        m = (q + p) + 1 if r_lengths is None else r_lengths[i]
        r = data['R'][i]
        r.update(parameters=[0.0] * m, gradient=[1e-7] + [0.0] * (m - 1), free_indices=list(range(1, m + 1)))
    return data


class FourthReviewTests(unittest.TestCase):
    """D1 malformed map in a sibling fit, D2 optimizer length, C3/C-a at the verifier level."""
    FAILS = ('FIT_HEALTH_FAIL', 'FIT_HEALTH_FAIL_WITH_INCOMPLETE', 'FIT_HEALTH_INTEGRITY_REJECTED')

    @staticmethod
    def recorded_fail(data):
        r = data['Julia'][0]
        r['optimizer']['optimizer_gradient'] = [2e-5]
        r['optimizer']['optimizer_max_abs_gradient'] = 2e-5
        return data

    def test_d1_garbage_map_never_downgrades_a_recorded_verdict(self):
        garbage = [[1], {}, [[1]], True, 1.5, 'x', None, float('nan'), {'a': 1}]
        makers = {'FIT_HEALTH_FAIL': lambda d: self.recorded_fail(d),
                  'FIT_HEALTH_FAIL_WITH_INCOMPLETE': lambda d: (self.recorded_fail(d),
                                                                d['R'][0].update(pd_hessian=False))[0],
                  'FIT_HEALTH_INTEGRITY_REJECTED': lambda d: d['Julia'][0]['optimizer'].update(
                      optimizer_max_abs_gradient=5e-8) or d}
        for expected, make in makers.items():
            clean = make(certified_good())
            self.assertEqual(health.classify(copy.deepcopy(clean)), expected)  # flag False
            with enabled():
                self.assertEqual(health.classify(copy.deepcopy(clean)), expected)
                for key in ('p', 'K', 'q_free'):
                    for value in garbage:
                        with self.subTest(expected=expected, key=key, value=repr(value)):
                            d = copy.deepcopy(clean)
                            d['Julia'][1]['lambda_constraint_map'][key] = value
                            self.assertEqual(health.classify(d), expected)

    def test_d1_garbage_map_refuses_the_pass_path_only(self):
        with enabled():
            self.assertEqual(health.classify(certified_good()), 'FIT_HEALTH_PASS')
            for key in ('p', 'K', 'q_free'):
                for value in ([1], {}, True, 'x', None, 1.0):
                    d = certified_good()
                    d['Julia'][1]['lambda_constraint_map'][key] = value
                    self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE', (key, value))
                    self.assertIsInstance(health._evidence_binding(d), str)

    def test_d1_a_raising_certifier_cannot_downgrade_a_recorded_failure(self):
        clean = self.recorded_fail(certified_good())
        with enabled(), patch.object(health, 'certify_evidence', side_effect=TypeError('boom')):
            self.assertEqual(health.classify(clean), 'FIT_HEALTH_FAIL')
            self.assertEqual(health.classify(certified_good()), 'FIT_HEALTH_INCOMPLETE')

    def test_d2_optimizer_length_is_the_layout_length_minus_log_sigma(self):
        self.assertEqual(health._optimizer_length(row_n(3)), 2)  # q = 0, p = 2, K = 1
        self.assertEqual(health.certify_fit(row_n(3))['verdict'], PASS)
        for length in (1, 3, 4, 7):  # length-1 point on a 3-parameter fit, packed length, too long
            with self.subTest(length=length):
                row = row_n(3, optimizer_length=length)
                self.assertIsInstance(health._optimizer_binding(row), str)
                self.assertEqual(health.certify_fit(row)['verdict'], INC)
        # the review's fabricated [42, -7] point on the 2-parameter fixture (expected length 1)
        data = certified_good()
        for r in data['Julia']:
            r['optimizer'].update(optimizer_parameters=[42.0, -7.0], optimizer_gradient=[1e-7, 0.0])
            r['reconstruction']['optimizer_parameters'] = [42.0, -7.0]
        self.assertIsInstance(health._optimizer_binding(data['Julia'][0]), str)
        with enabled():
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        # the layout length matches the retained real writer record: 7 packed, 6 optimizer coordinates
        self.assertEqual(health._optimizer_length({'lambda_constraint_map': {'p': 6, 'K': 1, 'q_free': 0}}), 6)

    def test_d2_r_counts_follow_the_julia_cases_by_one_constant(self):
        with enabled():
            self.assertEqual(health.classify(scaled_evidence()), 'FIT_HEALTH_PASS')
            self.assertEqual(health.classify(scaled_evidence(r_lengths=[7, 7, 7])), 'FIT_HEALTH_INCOMPLETE')
            self.assertEqual(health.classify(scaled_evidence(r_lengths=[6, 11, 13])), 'FIT_HEALTH_INCOMPLETE')
        self.assertIsInstance(health._evidence_binding(scaled_evidence(r_lengths=[7, 7, 7])), str)

    def test_c3_verifier_compares_each_map_with_the_authenticated_coefficient_length(self):
        import core070_verify_postfit_policy_batch as verifier
        evidence = {'fit_health': scaled_evidence()}
        self.assertIsNone(verifier.check_postcoef_authenticated_inputs(evidence))
        for qs in ([0, 5, 5], [0, 4, 6], [1, 5, 6], [0, 5, 7]):
            with self.subTest(qs=qs), self.assertRaisesRegex(ValueError, 'authenticated coefficient length'):
                verifier.check_postcoef_authenticated_inputs({'fit_health': scaled_evidence(qs)})
        for mutate in (lambda e: e['Julia'][0].pop('lambda_constraint_map'),
                       lambda e: e['Julia'][0]['lambda_constraint_map'].update(q_free=0.0),
                       lambda e: e['Julia'][0]['lambda_constraint_map'].update(q_free=False),
                       lambda e: e['Julia'][1].update(case='no_fixed_effects'),
                       lambda e: e['Julia'].pop()):
            bad = scaled_evidence()
            mutate(bad)
            with self.assertRaises(ValueError):
                verifier.check_postcoef_authenticated_inputs({'fit_health': bad})
        with self.assertRaises(ValueError):
            verifier.check_postcoef_authenticated_inputs({})

    def test_c3_live_ref_makes_the_same_comparison(self):
        import core070_postcoef_live_ref as live
        self.assertIsNone(live.check_lambda_maps(scaled_evidence(), [0, 5, 6]))
        with self.assertRaisesRegex(ValueError, 'authenticated coefficient length'):
            live.check_lambda_maps(scaled_evidence([0, 5, 5]), [0, 5, 6])
        with self.assertRaises(ValueError):
            live.check_lambda_maps(scaled_evidence(), [0, 5, 5])
        with self.assertRaises(ValueError):
            live.check_lambda_maps({}, [0, 5, 6])

    def test_c_a_pass_traces_reach_both_verifier_outputs(self):
        import inspect
        import core070_postcoef_live_ref as live
        import core070_verify_postfit_policy_batch as verifier
        with enabled():
            self.assertEqual(verifier.postcoef_pass_trace({'fit_health': scaled_evidence()}), 'declared_unproven')
            self.assertEqual(live.pass_trace(scaled_evidence(), [0, 5, 6]), 'declared_unproven')
            with self.assertRaises(ValueError):
                verifier.postcoef_pass_trace({'fit_health': scaled_evidence([0, 5, 5])})
            with self.assertRaises(ValueError):
                live.pass_trace(scaled_evidence([0, 5, 5]), [0, 5, 6])
        for fn in (verifier.postcoef_pass_trace, live.pass_trace):  # held: no pass, so nothing to trace
            with self.assertRaises(ValueError):
                fn(*([{'fit_health': scaled_evidence()}] if fn is verifier.postcoef_pass_trace
                     else [scaled_evidence(), [0, 5, 6]]))
        # the emitted outputs name the status (source-level: the full verify paths need a retained capture)
        self.assertIn('"postcoef_radius_status": radius_status', inspect.getsource(verifier.verify_state))
        self.assertIn("'radius_status': radius_status", inspect.getsource(live.verify))
        self.assertIn('radius_status = postcoef_pass_trace(evidence)', inspect.getsource(verifier.verify_state))
        self.assertIn('radius_status = pass_trace(', inspect.getsource(live.verify))

    def test_c_a_a_pass_leaves_a_visible_radius_status_trace(self):
        with enabled():
            self.assertEqual(health.pass_radius_status(scaled_evidence()), 'declared_unproven')
            for bad in (certified_good() and self.recorded_fail(certified_good()), {}, None):
                with self.assertRaises(ValueError):
                    health.pass_radius_status(bad)
            verdict = health.classify(scaled_evidence())
            self.assertEqual(verdict.radius_status, 'declared_unproven')
            # a verdict equal to PASS but without the trace is refused
            with patch.object(health, 'classify', return_value='FIT_HEALTH_PASS'), self.assertRaises(ValueError):
                health.pass_radius_status(scaled_evidence())
        with self.assertRaises(ValueError):  # held: no pass, so no status
            health.pass_radius_status(scaled_evidence())


if __name__ == '__main__':
    unittest.main()
