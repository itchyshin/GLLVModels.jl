"""Writer-shaped Julia records through the held certifier (synthetic; no fit was run).

``writer_record`` mirrors the fields tools/core070_postcoef_fit_health.jl now
emits: hessian_coordinate, pinned_indices (complement of free_indices in
1..len(parameters)), and, only for derivative_method "forwarddiff_exact", the
declared radii  2^10 * 2^-52 * max(1, scale).  The radius formula is repeated
here on purpose so a change in the Julia constants must be mirrored knowingly.
"""
import copy
import json
import unittest
from unittest.mock import patch

import core070_postcoef_fit_health as health
from test_postcoef_fit_health import good
from test_postcoef_numerical_admission import bind

BUDGET = 1024.0 * 2.0 ** -52


def hessian_radius(matrix):
    return BUDGET * max(1.0, max(abs(x) for row in matrix for x in row))


def gradient_radius(objective, max_gradient):
    return BUDGET * max(1.0, abs(objective), max_gradient)


def writer_record(matrix, free, max_gradient=1e-7, exact_ad=True):
    n = len(matrix)
    row = copy.deepcopy(good()['Julia'][0])
    grad = [0.0] * n
    grad[free[0] - 1] = max_gradient
    row.update(parameters=[0.0] * n, gradient=grad, free_indices=list(free),
               max_abs_gradient=max_gradient,
               pinned_indices=[i for i in range(1, n + 1) if i not in free],
               hessian=copy.deepcopy(matrix), hessian_raw=copy.deepcopy(matrix),
               hessian_asymmetry=0.0,
               hessian_asymmetry_limit=32 * 2.0 ** -52 * n * max(1.0, max(abs(x) for r in matrix for x in r)),
               hessian_coordinate='packed_free_index',
               hessian_error_radius=None, hessian_error_radius_basis=None)
    row['reconstruction']['recovered_parameters'] = [0.0] * n
    opt = row['optimizer']
    # the writer's optimizer point omits log sigma (profile_beta=false): one coordinate shorter
    opt.update(optimizer_parameters=[0.0] * (n - 1), optimizer_gradient=grad[:n - 1],
               optimizer_max_abs_gradient=max_gradient,
               gradient_error_radius=None, gradient_error_radius_basis=None)
    row['gradient_error_radius'] = row['packed_gradient_error_radius'] = None
    pinned = [i for i in range(1, n + 1) if i not in free]
    if pinned and pinned[0] == 1:
        raise ValueError('packed index 1 (log sigma) cannot be pinned by a lambda constraint')
    bind(row, pinned)
    if not exact_ad:  # the writer records no radius and no status without the exact-AD declaration
        row.pop('radius_status')
        opt.pop('radius_status')
    if exact_ad:
        row['hessian_error_radius'] = hessian_radius(matrix)
        row['gradient_error_radius'] = opt['gradient_error_radius'] = gradient_radius(
            opt['optimizer_profile_value'], max_gradient)
        row['packed_gradient_error_radius'] = gradient_radius(row['recomputed_objective'], max_gradient)
    return json.loads(json.dumps(row))  # exactly what a JSON round trip delivers


class WriterFieldTests(unittest.TestCase):
    def test_pd_record_passes_and_is_decided_on_the_free_block_only(self):
        # Entry (3,3) is pinned and indefinite; the free block is positive definite.
        m = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, -9.0]]
        row = writer_record(m, [1, 2])
        self.assertEqual(row['pinned_indices'], [3])
        r = health.certify_fit(row)
        self.assertEqual(r['verdict'], health.CERT_PASS, r)
        self.assertEqual(r['hessian']['excluded_indices'], [3])

    def test_indefinite_free_block_fails(self):
        row = writer_record([[1.0, 0.0], [0.0, -1.0]], [1, 2])
        self.assertEqual(health.certify_fit(row)['verdict'], health.CERT_FAIL)

    def test_near_singular_within_radius_is_incomplete(self):
        row = writer_record([[1.0, 0.0], [0.0, 1e-14]], [1, 2])
        self.assertEqual(health.certify_fit(row)['verdict'], health.CERT_INCOMPLETE)

    def test_gradient_above_limit_fails_and_below_passes(self):
        m = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(health.certify_fit(writer_record(m, [1, 2], 2e-5))['verdict'], health.CERT_FAIL)
        self.assertEqual(health.certify_fit(writer_record(m, [1, 2], 9e-6))['verdict'], health.CERT_PASS)

    def test_no_declared_radius_is_incomplete(self):
        row = writer_record([[1.0, 0.0], [0.0, 1.0]], [1, 2], exact_ad=False)
        r = health.certify_fit(row)
        self.assertEqual(r['verdict'], health.CERT_INCOMPLETE)

    def test_mislabelled_coordinate_is_incomplete(self):
        row = writer_record([[1.0, 0.0], [0.0, 1.0]], [1, 2])
        row['hessian_coordinate'] = 'optimizer'
        self.assertEqual(health.certify_fit(row)['verdict'], health.CERT_INCOMPLETE)
        row['hessian_coordinate'] = None
        self.assertEqual(health.certify_fit(row)['verdict'], health.CERT_INCOMPLETE)

    def test_pinned_must_be_the_exact_complement(self):
        row = writer_record([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1, 2])
        row['pinned_indices'] = []
        self.assertEqual(health.certify_fit(row)['verdict'], health.CERT_INCOMPLETE)

    def test_declared_radii_clear_the_certifier_floors(self):
        m = [[1e3, 1.0], [1.0, 1e3]]
        k = 2
        self.assertGreater(hessian_radius(m), k * 2.0 ** -52 * 1e3)
        self.assertGreater(health.CERT_MAX_FREE_DIMENSION, 0)
        self.assertGreater(1024, health.CERT_MAX_FREE_DIMENSION)  # floor k*eps*max|H| < radius for k <= cap

    def test_full_three_case_set_classifies_when_enabled_and_flag_patched(self):
        data = good()
        for i, case in enumerate(health.CASES):
            row = writer_record([[2.0, 0.0], [0.0, 3.0]], [1, 2])
            row['case'] = case
            data['Julia'][i] = row
        self.assertFalse(health.NUMERICAL_ADMISSION_IMPLEMENTED)
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')  # held by default
        with patch.object(health, 'NUMERICAL_ADMISSION_IMPLEMENTED', True):
            self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
            data['Julia'][1] = writer_record([[2.0, 0.0], [0.0, -3.0]], [1, 2])
            data['Julia'][1]['case'] = health.CASES[1]
            # pd_hessian flag stays True (patched) while the matrix is indefinite: a
            # confirmed failure, reported alongside the flag/matrix mismatch.
            self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')
            data['Julia'][1]['hessian_error_radius'] = None
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_writer_records_carry_status_binding_map_and_both_radii(self):
        m = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, -9.0]]
        row = writer_record(m, [1, 2])
        self.assertEqual(row['radius_status'], 'declared_unproven')
        self.assertEqual(row['optimizer']['radius_status'], 'declared_unproven')
        self.assertEqual(row['reconstruction']['optimizer_parameters'], row['optimizer']['optimizer_parameters'])
        self.assertEqual(row['lambda_constraint_map']['entries'], [[None], [0.0]])
        self.assertEqual(row['packed_gradient_error_radius'], gradient_radius(3.0, 1e-7))
        self.assertEqual(health.certify_fit(row)['radius_status'], 'declared_unproven')
        for key in ('radius_status', 'packed_gradient_error_radius', 'lambda_constraint_map'):
            lost = copy.deepcopy(row)
            lost.pop(key)
            with self.subTest(key=key):
                self.assertEqual(health.certify_fit(lost)['verdict'], health.CERT_INCOMPLETE)

    def test_r_records_have_no_hessian_to_certify(self):
        self.assertEqual(health.certify_fit(good()['R'][0], engine='R')['verdict'], health.CERT_INCOMPLETE)

    def test_float_round_trip_is_exact(self):
        # Julia repr(Float64) and R digits=17 are both lossless; Python's own
        # json round trip (shortest repr) is the contract the exact-equality checks rely on.
        for x in (0.1, 1e-7, 2.0 ** -52 * 1024, 1.0000000000000002, 4.9e-324, 1.7976931348623157e308):
            self.assertEqual(json.loads(json.dumps(x)), x)
        # A 15-digit serialisation is NOT lossless: this is what must never be used.
        self.assertNotEqual(float('%.15g' % 1.0000000000000002), 1.0000000000000002)


if __name__ == '__main__':
    unittest.main()
