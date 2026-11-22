"""Tests for behaviour that mutation testing showed was unpinned (wave 16).

Each class names the code it pins.  The mutants these tests kill were produced by
flipping a comparison, dropping a guard or changing a constant in
``core070_postcoef_fit_health.py``; before these tests every one of them passed the
whole ``test_postcoef*`` / ``test_core070_postcoef*`` suite.  Nothing here changes a
tolerance, a threshold or a verdict: every expected value is the behaviour of the
reviewed code, and where an independent oracle exists (packed-layout arithmetic,
round-to-nearest of log) the test computes it independently.
"""
import copy
import math
import pathlib
import unittest
from fractions import Fraction
from unittest.mock import patch

import core070_postcoef_fit_health as health
from test_postcoef_fit_health import good
from test_postcoef_numerical_admission import certified_good, enabled, row_n

INC, PASS, FAIL = health.CERT_INCOMPLETE, health.CERT_PASS, health.CERT_FAIL
TREE = 'df63c52066625049087a321bb9b010f8d834ddcf'
MAX_FLOAT = 1.7976931348623157e308


def r_provenance():
    return dict(pin='P1', reference_commit='9539352f66f2db2cc26b1c393e67212a359b60c9',
                source_tree_sha256='86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db',
                version='0.7.1', oracle_build_sha256='a' * 64, call_verified=True,
                call_text='gllvmTMB(formula = f, data = df_long, trait = "trait", unit = "site", family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))')


def j_provenance(case='trait_intercepts'):
    return dict(source_tree=TREE, case=case, route='gaussian_profile_default', rank=1,
                integration='exact', prior='none')


class ProvenanceValidatorTests(unittest.TestCase):
    """historical_r_prior_bound / julia_prior_bound return False on hostile input, never raise."""

    def test_baseline_binds(self):
        self.assertIs(health.historical_r_prior_bound(r_provenance(), 'a' * 64), True)
        self.assertIs(health.julia_prior_bound(j_provenance(), TREE, 'trait_intercepts'), True)

    def test_non_dict_provenance_is_false(self):
        for bad in (None, [], (), 'x', 5, [('pin', 'P1')], {'pin'}):
            with self.subTest(bad=bad):
                self.assertIs(health.historical_r_prior_bound(bad, 'a' * 64), False)
                self.assertIs(health.julia_prior_bound(bad, TREE, 'trait_intercepts'), False)

    def test_missing_or_extra_key_is_false_not_keyerror(self):
        for make, call in ((r_provenance, lambda p: health.historical_r_prior_bound(p, 'a' * 64)),
                           (j_provenance, lambda p: health.julia_prior_bound(p, TREE, 'trait_intercepts'))):
            base = make()
            for key in base:
                short = dict(base); del short[key]
                with self.subTest(missing=key):
                    self.assertIs(call(short), False)
            with self.subTest(extra=True):
                self.assertIs(call(dict(base, extra=1)), False)

    def test_build_hash_must_be_a_lowercase_64_hex_string(self):
        for build in (5, None, b'a' * 64, ['a'] * 64):
            with self.subTest(build=build):
                p = r_provenance(); p['oracle_build_sha256'] = build
                self.assertIs(health.historical_r_prior_bound(p, build), False)
        for build in ('g' * 64, 'A' * 64, 'a' * 63, 'a' * 65, 'a' * 64 + '\n', ''):
            with self.subTest(build=build):
                p = r_provenance(); p['oracle_build_sha256'] = build  # claim equals the authenticated value
                self.assertIs(health.historical_r_prior_bound(p, build), False)

    def test_julia_authenticated_tree_case_and_rank_are_each_required(self):
        other = 'a' * 40
        p = j_provenance(); p['source_tree'] = other  # claim equals the authenticated value but is not the audited tree
        self.assertIs(health.julia_prior_bound(p, other, 'trait_intercepts'), False)
        for case in ('bogus', [], None):
            p = j_provenance(); p['case'] = case  # claim equals the supplied case but that is not a case
            self.assertIs(health.julia_prior_bound(p, TREE, case), False)
        for rank in (0, 2, -1, 1.0, True):
            p = j_provenance(); p['rank'] = rank
            with self.subTest(rank=rank):
                self.assertIs(health.julia_prior_bound(p, TREE, 'trait_intercepts'), False)


class LambdaMapAndLengthTests(unittest.TestCase):
    """_optimizer_length and _lambda_pin_binding on K >= 2 and on malformed maps."""

    @staticmethod
    def with_map(p, K, q):
        return {'lambda_constraint_map': {'p': p, 'K': K, 'q_free': q}}

    def test_optimizer_length_matches_independent_loading_count(self):
        for p, K, q in [(5, 1, 0), (5, 2, 1), (5, 3, 0), (4, 4, 2), (3, 3, 0), (6, 2, 5), (7, 3, 1)]:
            loadings = K + sum(p - k for k in range(1, K + 1))  # diagonal, then strict lower triangle
            got = health._optimizer_length(self.with_map(p, K, q))
            with self.subTest(p=p, K=K, q=q):
                self.assertIs(type(got), int)
                self.assertEqual(got, q + loadings)

    def test_optimizer_length_is_none_for_malformed_maps(self):
        for m in (None, 5, [], {'p': 3, 'K': 1}, {'K': 1, 'q_free': 0}, {'p': 3, 'q_free': 0}):
            with self.subTest(map=m):
                self.assertIsNone(health._optimizer_length({'lambda_constraint_map': m}))
        self.assertIsNone(health._optimizer_length({}))
        for p, K, q in [(0, 1, 0), (-1, 1, 0), (3, 0, 0), (3, 4, 0), (3, 1, -1), (3.0, 1, 0), (3, 1.0, 0),
                        (3, 1, 0.0), (True, 1, 0), (3, True, True), ('3', 1, 0), (3, 1, None)]:
            with self.subTest(p=p, K=K, q=q):
                self.assertIsNone(health._optimizer_length(self.with_map(p, K, q)))
        self.assertEqual(health._optimizer_length(self.with_map(3, 3, 0)), 6)  # K = p is allowed
        self.assertEqual(health._optimizer_length(self.with_map(3, 1, 0)), 3)  # q_free = 0 is allowed

    @staticmethod
    def pinned_row(p, K, q, cells):
        """Row with the given {(i, k): value} pinned; the layout index is computed independently."""
        order = [(k, k) for k in range(1, K + 1)]
        for k in range(1, K + 1):
            order += [(i, k) for i in range(k + 1, p + 1)]
        index = {cell: q + 1 + n + 1 for n, cell in enumerate(order)}  # betas, log sigma, then loadings
        theta = [0.0] * (q + 1 + len(order))
        entries = [[None] * K for _ in range(p)]
        for cell, value in cells.items():
            entries[cell[0] - 1][cell[1] - 1] = value
            theta[index[cell] - 1] = value
        return {'lambda_constraint_map': {'layout': health.LAMBDA_MAP_LAYOUT, 'p': p, 'K': K, 'q_free': q,
                                          'entries': entries},
                'pinned_indices': sorted(index[c] for c in cells), 'parameters': theta}, index

    def test_pin_binding_matches_independent_packed_layout(self):
        for p, K, q in [(3, 2, 1), (4, 3, 0), (5, 2, 2), (4, 4, 1)]:
            lower = [(i, k) for k in range(1, K + 1) for i in range(k, p + 1)]
            for cell in lower:
                row, index = self.pinned_row(p, K, q, {cell: 0.5})
                with self.subTest(p=p, K=K, q=q, cell=cell):
                    self.assertIsNone(health._lambda_pin_binding(row))
                    for wrong in (index[cell] - 1, index[cell] + 1):
                        moved = copy.deepcopy(row); moved['pinned_indices'] = [wrong]
                        self.assertIsInstance(health._lambda_pin_binding(moved), str)
            row, _ = self.pinned_row(p, K, q, {(p, K): 0.5, (K, K): 0.25})  # two pins at once
            self.assertIsNone(health._lambda_pin_binding(row))

    def test_entries_above_the_diagonal_are_structural_zeros_and_ignored(self):
        row, _ = self.pinned_row(4, 3, 1, {})
        row['lambda_constraint_map']['entries'][0][2] = 7.0  # (1, 3): above the diagonal, not a parameter
        row['lambda_constraint_map']['entries'][1][2] = 7.0  # (2, 3)
        self.assertIsNone(health._lambda_pin_binding(row))

    def test_malformed_maps_give_a_reason_not_a_raise(self):
        base, _ = self.pinned_row(3, 2, 1, {(2, 1): 0.5})

        def with_(edit):
            row = copy.deepcopy(base); edit(row['lambda_constraint_map']); return row

        cases = {
            'p zero': lambda m: m.update(p=0, entries=[]),
            'K zero': lambda m: m.update(K=0, entries=[[], [], []]),
            'K above p': lambda m: m.update(K=4, entries=[[None] * 4] * 3),
            'q negative': lambda m: m.update(q_free=-1),
            'p float': lambda m: m.update(p=3.0),
            'K bool': lambda m: m.update(K=True),
            'entries not a list': lambda m: m.update(entries=5),
            'entries tuple': lambda m: m.update(entries=tuple(m['entries'])),
            'too few rows': lambda m: m.update(entries=m['entries'][:2]),
            'row not a list': lambda m: m['entries'].__setitem__(0, 5),
            'row too short': lambda m: m['entries'].__setitem__(0, [None]),
            'nan entry': lambda m: m['entries'][1].__setitem__(0, float('nan')),
            'inf entry': lambda m: m['entries'][1].__setitem__(0, float('inf')),
            'string entry': lambda m: m['entries'][1].__setitem__(0, 'x'),
            'bool entry': lambda m: m['entries'][1].__setitem__(0, True),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                self.assertEqual(health._lambda_pin_binding(with_(edit)), 'lambda_constraint_map is malformed')
        for theta in (None, (0.0,) * 6, 5, 'abcdef'):
            row = copy.deepcopy(base); row['parameters'] = theta
            with self.subTest(theta=theta):
                self.assertEqual(health._lambda_pin_binding(row),
                                 'parameters length does not match the lambda_constraint_map layout')


class OptimizerBindingGuardTests(unittest.TestCase):
    """_optimizer_binding refuses (with a reason) instead of raising or passing on malformed blocks."""

    def row(self):
        return certified_good()['Julia'][0]

    def test_baseline_binds(self):
        self.assertIsNone(health._optimizer_binding(self.row()))

    def test_point_and_gradient_must_be_finite_and_equal_length(self):
        for key, value in (('optimizer_parameters', [float('nan')]), ('optimizer_parameters', None),
                           ('optimizer_parameters', 5.0), ('optimizer_parameters', []),
                           ('optimizer_gradient', [float('inf')]), ('optimizer_gradient', None),
                           ('optimizer_gradient', [True]), ('optimizer_gradient', [1e-7, 0.0]),
                           ('optimizer_parameters', [0.0, 0.0])):
            row = self.row(); row['optimizer'][key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(health._optimizer_binding(row),
                                 'optimizer point or gradient missing, non-finite or of unequal length')

    def test_point_length_must_be_the_layout_length(self):
        row = row_n(7, optimizer_length=7)  # point and gradient are 7 long; the layout implies 6
        self.assertEqual(health._optimizer_binding(row),
                         'optimizer point length is not the length the lambda_constraint_map layout implies')
        row = self.row(); del row['lambda_constraint_map']
        self.assertEqual(health._optimizer_binding(row),
                         'optimizer point length is not the length the lambda_constraint_map layout implies')

    def test_packed_parameters_must_be_one_longer_than_the_point(self):
        for theta in (None, (0.0, 0.0), [0.0, 0.0, 0.0], [0.0], [float('nan'), 0.0], 5):
            row = self.row(); row['parameters'] = theta
            with self.subTest(theta=theta):
                self.assertEqual(health._optimizer_binding(row),
                                 'optimizer point is not the packed parameters in profiled (sigma_eps-scaled) coordinates')


class CertifyFitGuardTests(unittest.TestCase):
    def test_non_dict_fit_is_refused_with_the_engine_reason(self):
        for bad in (None, 5, [], 'x'):
            with self.subTest(bad=bad):
                out = health.certify_fit(bad)
                self.assertEqual((out['verdict'], out['reason']), (INC, 'no recorded Hessian for this engine'))

    def test_free_indices_must_be_a_list_of_unique_ints(self):
        for bad in ((1, 2), 'ab', 5, None, {1, 2}, [], [1, 1], [1, 2, 2], [1, True], [0, 1], [1, 3], [1.0, 2]):
            row = certified_good()['Julia'][0]; row['free_indices'] = bad
            with self.subTest(free=bad):
                out = health.certify_fit(row)
                self.assertEqual((out['verdict'], out['reason']), (INC, 'free indices missing or invalid'))


class CertifyHessianAndGradientGuardTests(unittest.TestCase):
    def test_matrix_rows_must_be_lists(self):
        for matrix in ([5, [1.0, 2.0]], [[1.0, 2.0], 5], [(1.0, 2.0), (2.0, 1.0)]):
            out = health.certify_hessian(matrix, [1, 2], 'packed_free_index', 1e-9, 0.0, [], parameter_count=2)
            with self.subTest(matrix=matrix):
                self.assertEqual((out['verdict'], out['reason']),
                                 (INC, 'matrix missing, non-square or non-finite'))
                self.assertIsNone(health.hessian_pd(matrix, [1, 2], 2))

    def test_free_and_pinned_indices_must_be_valid_lists(self):
        pd = [[2.0, 0.0], [0.0, 2.0]]
        for free in ((1, 2), 'ab', None, [1, 1], [1, 2, 2]):
            out = health.certify_hessian(pd, free, 'packed_free_index', 1e-9, 0.0, [], parameter_count=2)
            with self.subTest(free=free):
                self.assertEqual((out['verdict'], out['reason']), (INC, 'free indices missing or invalid'))
        # True and 1.0 hash and compare equal to 1: only the type check keeps them out of the partition
        for pinned in ([1.0], ['1'], [True]):
            out = health.certify_hessian(pd, [2], 'packed_free_index', 1e-9, 0.0, pinned, parameter_count=2)
            with self.subTest(pinned=pinned):
                self.assertEqual(out['verdict'], INC)
                self.assertIn('exact partition', out['reason'])
        self.assertEqual(health.certify_hessian(pd, [2], 'packed_free_index', 1e-9, 0.0, [1], parameter_count=2)['verdict'], PASS)

    def test_gradient_value_must_be_finite_and_non_negative(self):
        for bad in (None, float('nan'), float('inf'), -1e-9, 'x', True, [1e-7]):
            with self.subTest(gradient=bad):
                self.assertEqual(health.certify_gradient(bad, 1e-9), INC)

    def test_hessian_pd_is_none_when_the_factor_overflows(self):
        self.assertIsNone(health.hessian_pd([[1.0, 1e200], [1e200, 1e200]], [1, 2], 2))

    def test_roundoff_status_requires_square_finite_numeric_raw_and_symmetric(self):
        base = good()['Julia'][0]
        self.assertEqual(health.hessian_record_status(base, 2), 'valid')
        for key in ('hessian_raw', 'hessian'):
            for bad in ([[1.0, 0.0]], [5, [0.0, 1.0]], [[1.0, 0.0], 5], [[1.0, 0.0], [0.0, float('nan')]],
                        [[1.0, 0.0], [0.0, True]], [[1.0, 0.0], [0.0, 'x']], None, [[1.0], [0.0, 1.0]]):
                row = copy.deepcopy(base); row[key] = bad
                with self.subTest(key=key, bad=bad):
                    self.assertEqual(health.hessian_record_status(row, 2), 'incomplete')
                    self.assertIs(health.hessian_roundoff_valid(row, 2), False)


class EvidenceStructureTests(unittest.TestCase):
    """certify_evidence, _r_block_valid, _record_checks_valid and _evidence_binding on malformed evidence."""

    def test_baseline(self):
        d = certified_good()
        certs, ok = health.certify_evidence(d)
        self.assertTrue(ok)
        self.assertEqual([c['confirmable'] for c in certs], [True, True, True])
        self.assertIs(health._r_block_valid(d), True)

    def test_julia_block_shape(self):
        for edit in (lambda d: d['Julia'].append(copy.deepcopy(d['Julia'][0])), lambda d: d['Julia'].pop(),
                     lambda d: d['Julia'].__setitem__(1, 5), lambda d: d['Julia'].__setitem__(2, None),
                     lambda d: d['Julia'][0].__setitem__('case', None), lambda d: d['Julia'][0].__setitem__('case', 5),
                     lambda d: d['Julia'][0].__setitem__('case', ['x'])):
            d = certified_good(); edit(d)
            self.assertEqual(health.certify_evidence(d), ([], False))
        d = certified_good(); d['Julia'] = 'abc'
        self.assertEqual(health.certify_evidence(d), ([], False))
        for edit in (lambda d: d['Julia'][0].__setitem__('case', 'x'),
                     lambda d: d['Julia'][0].__setitem__('case', d['Julia'][1]['case'])):
            d = certified_good(); edit(d)
            certs, ok = health.certify_evidence(d)
            self.assertFalse(ok)

    def test_r_block_shape(self):
        extras = (lambda d: d['R'].append(copy.deepcopy(d['R'][0])), lambda d: d['R'].pop(),
                  lambda d: d['R'].__setitem__(1, 5),
                  lambda d: d['R'][0].__setitem__('case', None), lambda d: d['R'][0].__setitem__('case', 5),
                  lambda d: d['R'][0].__setitem__('case', ['x']), lambda d: d['R'][0].__setitem__('case', 'x'),
                  lambda d: d['R'][0].__setitem__('case', d['R'][1]['case']))
        for n, edit in enumerate(extras):
            d = certified_good(); edit(d)
            with self.subTest(edit=n):
                self.assertIs(health._r_block_valid(d), False)
                certs, ok = health.certify_evidence(d)  # must not raise; no cert may be confirmable
                self.assertEqual([c['confirmable'] for c in certs], [False, False, False])
        self.assertIs(health._r_block_valid({'R': 'abc'}), False)
        self.assertIs(health._r_block_valid(None), False)

    def test_r_block_field_table(self):
        edits = {
            'gradient not list': ('gradient', None), 'gradient empty': ('gradient', []),
            'gradient nan': ('gradient', [float('nan'), 0.0]), 'gradient bool': ('gradient', [True, 0.0]),
            'gradient tuple': ('gradient', (1e-7, 0.0)),
            'parameters not list': ('parameters', None), 'parameters short': ('parameters', [0.0]),
            'parameters nan': ('parameters', [float('nan'), 0.0]), 'parameters tuple': ('parameters', (0.0, 0.0)),
            'indices not list': ('free_indices', 5), 'indices tuple': ('free_indices', (1, 2)),
            'indices float': ('free_indices', [1.0, 2]), 'indices partial': ('free_indices', [1]),
            'indices duplicate': ('free_indices', [1, 1]), 'indices bool': ('free_indices', [True, 2]),
            'scalar missing': ('max_abs_gradient', None), 'scalar mismatch': ('max_abs_gradient', 2e-7),
            'scalar nan': ('max_abs_gradient', float('nan')),
            'objective nan': ('optimizer_objective', float('nan')), 'recomputed none': ('recomputed_objective', None),
            'objectives differ': ('recomputed_objective', 3.0000000001),
            'pd not bool': ('pd_hessian', 1), 'verified false': ('objective_verified', False),
            'penalty unrecorded': ('penalty_configuration_recorded', 1), 'error set': ('error', 'x'),
            'prior': ('fixed_prior_state', 'active'), 'ridge': ('ridge_penalised', True), 'tau': ('ridge_tau', 2.0),
        }
        for name, (key, value) in edits.items():
            d = certified_good(); d['R'][1][key] = value
            with self.subTest(name):
                self.assertIs(health._r_block_valid(d), False)
        d = certified_good(); del d['R'][1]['error']
        self.assertIs(health._r_block_valid(d), False)

    def test_confirmable_needs_each_recorded_check(self):
        edits = {
            'objective_verified False': lambda r: r.__setitem__('objective_verified', False),
            'penalty flag truthy list': lambda r: r.__setitem__('penalty_configuration_recorded', [0.0]),
            'penalty flag 1': lambda r: r.__setitem__('penalty_configuration_recorded', 1),
            'error key missing': lambda r: r.pop('error'),
            'error is a list': lambda r: r.__setitem__('error', [0.0]),
            'error is text': lambda r: r.__setitem__('error', 'x'),
            'recovered != parameters': lambda r: r['reconstruction'].__setitem__('recovered_parameters', [0.5, 0.0]),
            'reconstruction error key missing': lambda r: r['reconstruction'].pop('error'),
            'reconstruction error set': lambda r: r['reconstruction'].__setitem__('error', 'x'),
            'recovered loglik text': lambda r: r['reconstruction'].__setitem__('recovered_loglik', 'x'),
            'recovered loglik nan': lambda r: r['reconstruction'].__setitem__('recovered_loglik', float('nan')),
            'objective text': lambda r: r.__setitem__('optimizer_objective', 'x'),
            'recomputed text': lambda r: r.__setitem__('recomputed_objective', 'x'),
            'recomputed deleted': lambda r: r.pop('recomputed_objective'),
            'difference not recorded': lambda r: r.__setitem__('objective_difference', None),
            'limit not recorded': lambda r: r.__setitem__('objective_roundoff_limit', None),
            'difference nan': lambda r: r.__setitem__('objective_difference', float('nan')),
            'limit nan': lambda r: r.__setitem__('objective_roundoff_limit', float('nan')),
            'recorded difference differs': lambda r: r.__setitem__('objective_difference', 1e-6),
            'recorded limit differs by one ulp': lambda r: r.__setitem__(
                'objective_roundoff_limit', math.nextafter(r['objective_roundoff_limit'], math.inf)),
            'difference above limit': lambda r: r.update(recomputed_objective=3.0 + 1e-6, objective_difference=1e-6),
            'raw hessian dropped': lambda r: r.pop('hessian_raw'),
            'raw hessian has a bool': lambda r: r.__setitem__('hessian_raw', [[1.0, 0.0], [0.0, True]]),
        }
        for name, edit in edits.items():
            d = certified_good(); edit(d['Julia'][1])
            with self.subTest(name):
                certs, ok = health.certify_evidence(d)
                self.assertFalse(certs[1]['confirmable'])
        d = certified_good(); d['Julia'][1]['objective_verified'] = False
        self.assertFalse(health.certify_evidence(d)[1])  # one unbound fit makes the whole set unconfirmable

    def test_difference_above_limit_fails_the_record_check_even_when_the_record_agrees_with_itself(self):
        d = certified_good(); r = d['Julia'][1]
        r['recomputed_objective'] = 3.0 + 1e-6
        r['objective_difference'] = abs(r['optimizer_objective'] - r['recomputed_objective'])
        self.assertFalse(health.certify_evidence(d)[0][1]['confirmable'])
        self.assertIs(health._record_checks_valid(r, True), False)

    def test_record_checks_valid_needs_r_valid_exactly_true(self):
        r = certified_good()['Julia'][0]
        self.assertIs(health._record_checks_valid(r, True), True)
        for flag in (False, None, 1, 'yes'):
            self.assertIs(health._record_checks_valid(r, flag), False)

    def test_evidence_binding_tolerates_missing_optimizer(self):
        for value in (None, {}, 0, False, ''):
            d = certified_good(); d['Julia'][1]['optimizer'] = value
            self.assertEqual(health._evidence_binding(d), 'case binding evidence missing')

    def test_evidence_binding_rules(self):
        self.assertIsNone(health._evidence_binding(certified_good()))
        d = certified_good(); d['Julia'][0]['lambda_constraint_map']['q_free'] = 1
        self.assertEqual(health._evidence_binding(d), 'free-coefficient counts are inconsistent with the case order')
        d = certified_good(); d['Julia'][1]['lambda_constraint_map']['q_free'] = 'x'
        self.assertEqual(health._evidence_binding(d), 'case binding evidence malformed')
        d = certified_good(); d['Julia'][1]['lambda_constraint_map']['p'] = 4
        self.assertEqual(health._evidence_binding(d), 'the three cases do not share the response dimension and rank')

    def test_malformed_map_in_a_sibling_fit_does_not_raise(self):
        d = certified_good(); d['Julia'][1]['lambda_constraint_map']['K'] = 'x'
        certs, ok = health.certify_evidence(d)
        self.assertFalse(ok)
        d = certified_good(); d['Julia'][1]['lambda_constraint_map']['p'] = 1.5
        self.assertIsNotNone(health._evidence_binding(d))


class ClassifyRecordedRobustnessTests(unittest.TestCase):
    """A malformed record in one fit must not mask a failure or an integrity rejection in another."""

    @staticmethod
    def julia_gradient_failure():
        d = certified_good(); r = d['Julia'][0]
        r['optimizer']['optimizer_gradient'] = [3e-5]; r['optimizer']['optimizer_max_abs_gradient'] = 3e-5
        r['gradient'] = [3e-5, 0.0]; r['max_abs_gradient'] = 3e-5
        return d

    @staticmethod
    def r_gradient_failure():
        d = certified_good()
        for r in d['R']:
            r['gradient'] = [3e-5, 0.0]; r['max_abs_gradient'] = 3e-5
        return d

    def test_baselines(self):
        self.assertEqual(health.classify(self.julia_gradient_failure()), 'FIT_HEALTH_FAIL')
        self.assertEqual(health.classify(self.r_gradient_failure()), 'FIT_HEALTH_FAIL')

    def test_non_list_free_indices_in_a_sibling_keeps_the_failure(self):
        for engine in ('R', 'Julia'):
            d = self.julia_gradient_failure(); d[engine][1]['free_indices'] = 5
            with self.subTest(engine=engine):
                self.assertEqual(health.classify(d), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_duplicate_free_indices_are_incomplete_not_an_integrity_rejection(self):
        d = self.julia_gradient_failure(); d['Julia'][1]['free_indices'] = [2, 2]
        self.assertEqual(health.classify(d), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_non_list_parameters_in_a_sibling_keeps_the_failure(self):
        d = self.r_gradient_failure(); d['R'][0]['parameters'] = 0
        self.assertEqual(health.classify(d), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_non_list_optimizer_parameters_in_a_sibling_keeps_the_integrity_rejection(self):
        d = certified_good()
        d['Julia'][0]['optimizer']['optimizer_max_abs_gradient'] = 2e-7  # contradicts the vector: integrity
        d['Julia'][2]['optimizer']['optimizer_parameters'] = float('nan')
        self.assertEqual(health.classify(d), 'FIT_HEALTH_INTEGRITY_REJECTED')
        d = certified_good()
        d['Julia'][0]['optimizer']['optimizer_max_abs_gradient'] = 2e-7
        d['Julia'][2]['optimizer']['optimizer_gradient'] = 5.0
        self.assertEqual(health.classify(d), 'FIT_HEALTH_INTEGRITY_REJECTED')

    def test_misspelled_r_case_label_blocks_an_r_failure(self):
        d = self.r_gradient_failure(); d['R'][0]['case'] = 'x'
        self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE')
        with enabled():
            self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE')

    def test_fourth_fit_in_a_block_blocks_a_failure(self):
        d = self.r_gradient_failure(); d['R'].append(copy.deepcopy(d['R'][0]))
        self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE')
        d = self.julia_gradient_failure(); d['Julia'].append(copy.deepcopy(d['Julia'][1]))
        self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE')

    def test_bool_in_recovered_parameters_is_not_a_number(self):
        for edit in (lambda r: r['reconstruction'].__setitem__('recovered_parameters', [False, 0.0]),
                     lambda r: r['reconstruction'].__setitem__('recovered_parameters', [0.0, False])):
            d = good(); edit(d['Julia'][0])
            self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')
            self.assertEqual(health.classify(d), 'FIT_HEALTH_INCOMPLETE')
        d = good()
        d['Julia'][0]['parameters'] = [1.0, 0.0]
        d['Julia'][0]['reconstruction']['recovered_parameters'] = [True, 0.0]
        self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')

    def test_raw_hessian_of_wrong_shape_is_incomplete_without_an_exception(self):
        d = good(); d['Julia'][0]['hessian_raw'] = [[1.0, 0.0]]
        self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')

    def test_private_classifier_does_not_raise_on_malformed_shapes(self):
        edits = (lambda d: d.__setitem__('R', 'abc'), lambda d: d['R'].__setitem__(0, 5),
                 lambda d: d['Julia'].__setitem__(1, 5), lambda d: d['Julia'][0].__setitem__('gradient', 7),
                 lambda d: d['Julia'][0].__setitem__('hessian', [5, [0.0, 1.0]]),
                 lambda d: d['Julia'][0].__setitem__('hessian_raw', [5, [0.0, 1.0]]),
                 lambda d: d['Julia'][0]['optimizer'].__setitem__('optimizer_gradient', 7),
                 lambda d: d['Julia'][0]['optimizer'].__setitem__('optimizer_parameters', 7),
                 lambda d: d['R'][0].__setitem__('parameters', 7), lambda d: d['R'][0].__setitem__('free_indices', 7))
        for n, edit in enumerate(edits):
            d = good(); edit(d)
            with self.subTest(edit=n):
                self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')


class PolicyConstantTests(unittest.TestCase):
    """Constants that bound work and precision: a silent edit must be a visible test failure."""

    def test_pinned_values(self):
        self.assertEqual(health.PROFILE_SCALE_MAX_CANDIDATES, 64)
        self.assertEqual(health.PROFILE_LOG_MAX_TERMS, 640)
        self.assertEqual(health.CERT_MAX_FREE_DIMENSION, 32)
        self.assertEqual(health.CERT_MAX_WORK, 2000000)
        self.assertEqual(health.CERT_GRADIENT_LIMIT, Fraction(1, 100000))
        self.assertEqual(health.CERT_EPS, Fraction(1, 2 ** 52))
        self.assertEqual(health.GRADIENT_LIMIT, 1e-5)
        self.assertFalse(health.NUMERICAL_ADMISSION_IMPLEMENTED)


def decimal_rounds_to_log(s, ell):
    """Independent oracle: is binary64 ``ell`` the round-to-nearest value of log(s)?  (60-digit decimal)"""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    exact = Decimal(s).ln()
    lo = (Decimal(math.nextafter(ell, -math.inf)) + Decimal(ell)) / 2
    hi = (Decimal(ell) + Decimal(math.nextafter(ell, math.inf))) / 2
    return lo < exact < hi


class LogRoundingTests(unittest.TestCase):
    """_rounds_to_log decides 'ell is the correctly rounded log(s)' exactly."""

    def test_agrees_with_a_decimal_oracle(self):
        for s in (2.0, 0.5, 3.0, 1e-3, 1e3, 1.0000000000000002, 0.9999999999999999, 12345.678, 1e-300, 1e300, 5e-324):
            ell = math.log(s)
            for candidate in (ell, math.nextafter(ell, math.inf), math.nextafter(ell, -math.inf)):
                with self.subTest(s=s, ell=candidate):
                    self.assertIs(health._rounds_to_log(s, candidate), decimal_rounds_to_log(s, candidate))

    def test_one_needs_positive_zero(self):
        self.assertIs(health._rounds_to_log(1.0, 0.0), True)
        self.assertIs(health._rounds_to_log(1.0, -0.0), False)
        self.assertIs(health._rounds_to_log(1.0, 1e-300), False)

    def test_extreme_ell_has_no_neighbour_and_is_false_not_an_overflow(self):
        self.assertIs(health._rounds_to_log(1e300, MAX_FLOAT), False)
        self.assertIs(health._rounds_to_log(1e-300, -MAX_FLOAT), False)
        self.assertIs(health._rounds_to_log(2.0, math.inf), False)

    def test_refinement_budget_is_honoured(self):
        s = 5e-324
        ell = math.log(s)
        self.assertIs(health._rounds_to_log(s, ell), True)
        with patch.object(health, 'PROFILE_LOG_MAX_TERMS', 39):
            self.assertIs(health._rounds_to_log(s, ell), False)
        with patch.object(health, 'PROFILE_LOG_MAX_TERMS', 40):
            self.assertIs(health._rounds_to_log(s, ell), True)

    def test_enclosures_contain_the_true_value(self):
        from decimal import Decimal, getcontext
        getcontext().prec = 60

        def dec(q):
            return Decimal(q.numerator) / Decimal(q.denominator)
        for z in (Fraction(1, 3), Fraction(-1, 3), Fraction(1, 7), Fraction(0)):
            lo, hi = health._atanh_enclosure(z, 6)
            exact = (Decimal(1 + z.numerator / z.denominator) / Decimal(1 - z.numerator / z.denominator)).ln() / 2 \
                if z else Decimal(0)
            # the Decimal built from a float quotient is accurate to ~1e-16, far below the tolerance used here
            self.assertLessEqual(dec(lo), exact + Decimal('1e-15'))
            self.assertGreaterEqual(dec(hi), exact - Decimal('1e-15'))
            self.assertLess(hi - lo, Fraction(1, 10 ** 4) if z else Fraction(1, 10 ** 30))  # not vacuous
        for s in (2.0, 0.3, 1e300, 5e-324, 1e-3):
            lo, hi = health._log_enclosure(s, 12)
            exact = Decimal(s).ln()
            self.assertLessEqual(dec(lo), exact)
            self.assertGreaterEqual(dec(hi), exact)
            self.assertLess(hi - lo, Fraction(1, 10 ** 6))


class ProfileScaleTests(unittest.TestCase):
    """_profile_scale_binds: exact binding of profiled loadings to the packed loadings and log sigma."""

    def test_pinned_vectors(self):
        cases = [
            # (profiled L, packed lambda, packed log sigma, expected)
            ([-1.0, 1e-310], [-1.0, 1e-310], 0.0, True),                    # sigma = 1, subnormal loading
            ([0.0003722414900689092], [math.inf], -0.6931471805599453, False),  # infinite loading never binds
            ([1e-310], [1.1278784784762e-310], -0.0, False),                # log sigma = -0.0 is not log of any sigma != 1
            ([-0.4124040451658484], [-0.0], -744.4400719213812, False),     # a nonzero loading cannot become zero
            ([1.0, 0.0007969577517018753, -0.0], [5e-324, 0.0, -0.0], -744.4400719213812, False),
            ([-1.0], [-2.22940800335902], 0.8017360808830725, False),
            ([-1.0, 1.0], [-1.29629783886404, 1.29629783886404], 0.2595123854652075, True),
            ([0.6794762586065861, 1e-310], [0.33973812930329306, 0.3397381293032931], -0.6931471805599453, False),
            ([-1.0, 1e-310, -1.5306291134515644, -0.6343185440837305],
             [-0.5, 5e-311, -0.7653145567257822, -0.3171592720418652], -0.6931471805599453, True),
            ([0.0, -0.0], [0.0, -0.0], 0.0, True),                          # all loadings zero: any sigma binds
            ([0.0, -0.0], [0.0, 0.0], 0.0, False),                          # but zero signs must match
            ([1.0], [MAX_FLOAT], 0.0, False),                               # no neighbour above the largest binary64
        ]
        for scaled, packed, log_sigma, expected in cases:
            with self.subTest(scaled=scaled, packed=packed, log_sigma=log_sigma):
                self.assertIs(health._profile_scale_binds(scaled, packed, log_sigma), expected)

    def test_sign_rules(self):
        self.assertIs(health._profile_scale_binds([1.0], [2.0], math.log(2.0)), True)
        self.assertIs(health._profile_scale_binds([1.0], [-2.0], math.log(2.0)), False)
        self.assertIs(health._profile_scale_binds([-1.0], [2.0], math.log(2.0)), False)
        self.assertIs(health._profile_scale_binds([0.0], [1e-300], 0.0), False)

    def test_candidate_budget_is_honoured(self):
        theta = ([0.75, -0.3, 1e-3], [1.5, -0.6, 2e-3], math.log(2.0))
        self.assertIs(health._profile_scale_binds(*theta), True)
        with patch.object(health, 'PROFILE_SCALE_MAX_CANDIDATES', 0):
            self.assertIs(health._profile_scale_binds(*theta), False)

    def test_log_sigma_must_round_correctly_for_the_candidate(self):
        theta = ([0.75], [1.5], math.log(2.0))
        self.assertIs(health._profile_scale_binds(*theta), True)
        for wrong in (math.nextafter(theta[2], math.inf), math.nextafter(theta[2], -math.inf), 0.0):
            self.assertIs(health._profile_scale_binds(theta[0], theta[1], wrong), False)

    def test_overflowing_neighbours_do_not_raise(self):
        for packed in (math.inf, -math.inf, MAX_FLOAT, -MAX_FLOAT):
            self.assertIs(health._profile_scale_binds([1.0], [packed], 0.0), False)


class SecondPassTests(unittest.TestCase):
    """Behaviours the first pass of tests left unpinned (surviving mutants after the first test file)."""

    @staticmethod
    def gradient_failure_row():
        d = certified_good(); r = d['Julia'][0]
        r['optimizer']['optimizer_gradient'] = [3e-5]; r['optimizer']['optimizer_max_abs_gradient'] = 3e-5
        r['gradient'] = [3e-5, 0.0]; r['max_abs_gradient'] = 3e-5
        return r

    def test_duplicate_free_indices_never_certify_a_failure(self):
        r = self.gradient_failure_row()
        self.assertEqual(health.certify_fit(r)['verdict'], FAIL)
        r['free_indices'] = [1, 1]
        out = health.certify_fit(r)
        self.assertEqual((out['verdict'], out['reason']), (INC, 'free indices missing or invalid'))

    def test_evidence_that_is_not_a_dict_is_incomplete_not_an_error(self):
        for ev in (None, 5, [], 'abc', (), 7.5):
            with self.subTest(evidence=ev):
                self.assertEqual(health._classify_recorded(ev), 'FIT_HEALTH_INCOMPLETE')
                self.assertEqual(health.certify_evidence(ev), ([], False))
                self.assertEqual(health.classify(ev), 'FIT_HEALTH_INCOMPLETE')

    def test_engine_block_that_is_not_a_list_is_incomplete_not_an_error(self):
        for block in (5, None, 7.5, True, 'abc', {}):
            for engine in ('R', 'Julia'):
                d = good(); d[engine] = block
                with self.subTest(engine=engine, block=block):
                    self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')

    def test_empty_optimizer_vectors_are_incomplete_not_an_error(self):
        d = good()
        d['Julia'][0]['optimizer']['optimizer_parameters'] = []
        d['Julia'][0]['optimizer']['optimizer_gradient'] = []
        self.assertEqual(health._classify_recorded(d), 'FIT_HEALTH_INCOMPLETE')

    def test_empty_gradient_never_passes_the_record_check(self):
        r = certified_good()['Julia'][0]; r['gradient'] = []
        self.assertIs(health._record_checks_valid(r, True), False)

    def test_r_block_with_empty_or_text_vectors_is_invalid_not_an_error(self):
        d = certified_good(); R = d['R'][1]
        R['gradient'], R['parameters'], R['free_indices'] = [], [], []
        self.assertIs(health._r_block_valid(d), False)
        for bad in (['x', 0.0], [None, 0.0], [0.0, [1.0]]):
            d = certified_good(); d['R'][1]['gradient'] = bad
            with self.subTest(gradient=bad):
                self.assertIs(health._r_block_valid(d), False)

    def test_asymmetry_limit_has_a_floor_of_one_in_the_matrix_scale(self):
        eps = 2.0 ** -52
        for entry, scale in ((0.25, 1.0), (1.0, 1.0), (4.0, 4.0)):
            row = good()['Julia'][0]
            row['hessian'] = [[entry, 0.0], [0.0, entry]]; row['hessian_raw'] = [[entry, 0.0], [0.0, entry]]
            row['hessian_asymmetry_limit'] = 32 * eps * 2 * scale
            with self.subTest(entry=entry):
                self.assertEqual(health.hessian_record_status(row, 2), 'valid')
            row['hessian_asymmetry_limit'] = 32 * eps * 2 * scale * 1.0000000000000002
            self.assertEqual(health.hessian_record_status(row, 2), 'integrity')


class EnclosureAndSearchCostTests(unittest.TestCase):
    """The exact-arithmetic helpers: stated tail bound, refinement schedule and candidate loop."""

    def test_atanh_tail_bound_is_the_stated_one(self):
        for z, terms in ((Fraction(1, 3), 6), (Fraction(-1, 7), 5), (Fraction(1, 3), 40), (Fraction(2, 9), 1)):
            lo, hi = health._atanh_enclosure(z, terms)
            tail = abs(z) ** (2 * terms + 1) / ((2 * terms + 1) * (1 - z * z))
            self.assertEqual(hi - lo, 2 * tail)
            self.assertEqual((hi + lo) / 2, sum(z ** (2 * k + 1) / (2 * k + 1) for k in range(terms)))

    def test_log_enclosure_orientation_for_negative_and_positive_exponents(self):
        from decimal import Decimal, getcontext
        getcontext().prec = 60
        for s in (4.0, 0.25, 1e10, 1e-10, 0.7, 1.3):
            lo, hi = health._log_enclosure(s, 30)
            self.assertLess(lo, hi)
            exact = Decimal(s).ln()
            self.assertLess(abs(Decimal(lo.numerator) / Decimal(lo.denominator) - exact), Decimal('1e-20'))
            self.assertLess(abs(Decimal(hi.numerator) / Decimal(hi.denominator) - exact), Decimal('1e-20'))

    def test_refinement_decides_at_the_first_clear_enclosure(self):
        calls = []
        real = health._log_enclosure

        def counting(s, terms):
            calls.append(terms)
            return real(s, terms)
        s = 2.0
        exact = math.log(s)
        with patch.object(health, '_log_enclosure', counting):
            self.assertIs(health._rounds_to_log(s, exact), True)
            self.assertEqual(calls, [40])
            for ell in (exact + 1.0, exact - 1.0, 5.0, -5.0):  # clearly outside: also one enclosure
                del calls[:]
                self.assertIs(health._rounds_to_log(s, ell), False)
                self.assertEqual(calls, [40], ell)

    def test_refinement_doubles_until_decided_and_then_gives_up(self):
        s, ell = 2.0, math.log(2.0)
        seen = []

        def undecided_until(limit):
            def fake(_s, terms):
                seen.append(terms)
                return (ell, ell) if terms >= limit else (ell - 1.0, ell + 1.0)
            return fake
        with patch.object(health, '_log_enclosure', undecided_until(160)):
            self.assertIs(health._rounds_to_log(s, ell), True)
        self.assertEqual(seen, [40, 80, 160])
        del seen[:]
        with patch.object(health, '_log_enclosure', undecided_until(10 ** 9)):
            self.assertIs(health._rounds_to_log(s, ell), False)
        self.assertEqual(seen, [40, 80, 160, 320, 640])

    def test_candidate_search_edge_cases(self):
        cases = [
            ([1e-300], [2e-300], math.log(2.0), True),          # first candidate, budget below
            ([5e-324], [1e300], 0.0, False),                     # needed sigma exceeds the largest binary64
            ([0.5], [MAX_FLOAT / 2], 0.0, False),                # candidates run off the end of binary64
            ([-1.0], [-MAX_FLOAT], 0.0, False),                  # no neighbour below the most negative binary64
            ([1.0, 1.0], [1.0, 2.0], 0.0, False),                # incompatible loadings: empty sigma interval
        ]
        for scaled, packed, log_sigma, expected in cases:
            with self.subTest(scaled=scaled, packed=packed):
                self.assertIs(health._profile_scale_binds(scaled, packed, log_sigma), expected)
        with patch.object(health, 'PROFILE_SCALE_MAX_CANDIDATES', 1):
            self.assertIs(health._profile_scale_binds([1e-300], [2e-300], math.log(2.0)), True)


class JuliaPenaltyStateSourcePinTests(unittest.TestCase):
    """Change detectors for the Julia writer's penalty_state derivation.

    The behaviour is exercised by ``test_postcoef_fit_health_jl.jl`` (run on Totoro/DRAC, not on the
    Mac).  Mutating the helper's source left every Python test green, so the reviewed text is pinned
    here: an edit must update this test and re-run the Julia test.  This is a change detector, not
    a behavioural test.
    """

    TOOLS = pathlib.Path(__file__).resolve().parent

    def source(self, name):
        return (self.TOOLS / name).read_text()

    def test_derivation_function_is_the_reviewed_text(self):
        text = self.source('core070_postcoef_fit_health.jl')
        start = text.index('function postcoef_julia_penalty_state(call_kwargs)')
        body = text[start:text.index('\nend\n', start) + len('\nend\n')]
        self.assertEqual(body, (
            'function postcoef_julia_penalty_state(call_kwargs)\n'
            '    call_kwargs isa NamedTuple || return nothing\n'
            '    names = keys(call_kwargs)\n'
            '    any(k -> occursin(r"prior|ridge|penal"i, String(k)), names) && return "active"\n'
            '    all(==(:K), names) || return nothing\n'
            '    return "none"\n'
            'end\n'))

    def test_recorded_flag_requires_a_known_string_state(self):
        text = self.source('core070_postcoef_fit_health.jl')
        self.assertIn('out["penalty_configuration_recorded"] = penalty_state isa AbstractString &&\n'
                      '                                               penalty_state in ("none", "active")', text)

    def test_batch_attaches_provenance_only_for_a_recorded_none_state(self):
        text = self.source('core070_postfit_policy_batch.jl')
        self.assertIn('if h["penalty_state"] == "none" && h["penalty_configuration_recorded"] === true &&', text)


if __name__ == '__main__':
    unittest.main()
