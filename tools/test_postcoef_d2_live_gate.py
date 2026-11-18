#!/usr/bin/env python3
"""D2 (2026-11-09, "no certifier"): the live gate narrowed to the POST-COEF-EMPTY row and to nothing else.

Pure Python; no Julia, R, capture or network. Fixtures come from test_core070_postcoef_live_ref.
"""
import copy
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import core070_postcoef_fit_health as health
import core070_postcoef_live_ref as live
import test_core070_postcoef_live_ref as base

OTHER = 'CORE070-OTHER-ROW'
PASS = 'FIT_HEALTH_PASS'


def three_fit(report):
    return report['cases'][live.CID]['three_fit']


class D2LiveGateTests(unittest.TestCase):
    def test_d2_set_is_exactly_the_postcoef_row(self):
        self.assertEqual(live.D2_NO_CERTIFIER, frozenset({'CORE070-POSTFIT-COEF-EMPTY-NATIVE'}))
        self.assertFalse(live.health_pass_required(live.CID))
        self.assertTrue(live.health_pass_required(OTHER))
        self.assertTrue(live.health_pass_required(live.RECEIPT_CID))

    def test_postcoef_row_is_accepted_without_a_certified_health_pass(self):
        case, report, oracle = base.fixture()
        self.assertNotEqual(health.classify(three_fit(report)['fit_health']), PASS)  # the hold is still in force
        self.assertIsNone(live.check_comparison(case, report, oracle))

    def test_postcoef_row_is_accepted_with_failed_health_records(self):
        from test_postcoef_fit_health import with_indefinite_julia_hessian
        case, report, oracle = base.fixture()
        with_indefinite_julia_hessian(three_fit(report)['fit_health'])
        self.assertIsNone(live.check_comparison(case, report, oracle))
        case, report, oracle = base.fixture()
        three_fit(report)['fit_health']['Julia'][0]['max_abs_gradient'] = 1e-5
        self.assertNotEqual(health.classify(three_fit(report)['fit_health']), PASS)
        self.assertIsNone(live.check_comparison(case, report, oracle))

    def test_another_row_id_still_requires_a_certified_health_pass(self):
        with patch.object(live, 'CID', OTHER):
            case, report, oracle = base.fixture()
            self.assertEqual(case['harness_case_id'], OTHER)
            with self.assertRaisesRegex(ValueError, 'fit health failed or is incomplete'):
                live.check_comparison(case, report, oracle)
            with base.preliminary_health():  # only the health requirement separates the two rows
                self.assertIsNone(live.check_comparison(case, report, oracle))

    def test_emptying_the_set_restores_the_requirement_for_the_postcoef_row(self):
        case, report, oracle = base.fixture()
        with patch.object(live, 'D2_NO_CERTIFIER', frozenset()):
            with self.assertRaisesRegex(ValueError, 'fit health failed or is incomplete'):
                live.check_comparison(case, report, oracle)

    def test_postcoef_row_still_rejects_wrong_lengths_and_controls(self):
        for bad_lengths in ([0, 0, 0], [1, 6, 7], [0, 5, 5]):
            case, report, oracle = base.fixture()
            three_fit(report)['julia_lengths'] = list(bad_lengths)
            case['harness_fields'] = copy.deepcopy(report['cases'][live.CID])
            case['comparison']['cases'][0]['julia_value'] = list(bad_lengths)
            with self.subTest(julia_lengths=bad_lengths), self.assertRaisesRegex(ValueError, 'comparison differs'):
                live.check_comparison(case, report, oracle)
        case, report, oracle = base.fixture()
        oracle['postcoef_three_fit']['fits'][0]['length'] = 1
        with self.assertRaises(ValueError):
            live.check_comparison(case, report, oracle)
        for mutate in (lambda c: c.__setitem__('verdict', 'FAIL'),
                       lambda c: c.__setitem__('harness_pass', False),
                       lambda c: c['comparison']['cases'][0].__setitem__('discriminating', False)):
            case, report, oracle = base.fixture()
            mutate(case)
            with self.assertRaises(ValueError):
                live.check_comparison(case, report, oracle)

    def test_lambda_map_comparison_is_kept_for_the_postcoef_row(self):
        with self.assertRaises(ValueError):
            live.check_lambda_maps({}, [0, 5, 6])
        source = inspect.getsource(live.verify)
        self.assertIn('health_pass_required(CID)', source)
        self.assertIn('check_lambda_maps(three_fit', source)

    def test_source_binding_needs_no_provenance_for_the_postcoef_row_but_keeps_build_checks(self):
        case, record, files = self.binding_fixture()
        self.assertIsNone(live.check_case_source_binding(case, record, files.__getitem__))
        with patch.object(live, 'D2_NO_CERTIFIER', frozenset()):
            with self.assertRaisesRegex(ValueError, 'Julia penalty provenance'):
                live.check_case_source_binding(case, record, files.__getitem__)
        files[case['oracle_build_receipt']] = b'wrong build'
        with self.assertRaisesRegex(ValueError, 'oracle build'):
            live.check_case_source_binding(case, record, files.__getitem__)
        case, record, files = self.binding_fixture()
        bad = copy.deepcopy(case)
        bad['glvmodels_src_tree'] = 'wrong'
        with self.assertRaises(ValueError):
            live.check_case_source_binding(bad, record, files.__getitem__)

    @staticmethod
    def binding_fixture():
        case, _, _ = base.fixture()
        pin = {'reference_commit': base.PIN, 'archive_sha256': 'a' * 64, 'namespace_sha256': 'b' * 64,
               'source_tree_sha256': '86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db',
               'installed_tree_sha256': 'd' * 64, 'marker_sha256': 'e' * 64, 'version': '0.7.1'}
        build = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'source_tree_sha256',
                                      'installed_tree_sha256', 'marker_sha256')}
        build['gllvmTMB_version'] = pin['version']
        build_bytes = json.dumps(build).encode()
        source = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'namespace_sha256', 'source_tree_sha256')}
        tree = '627fa323cd2a8c0f3f2875df3aaa6d3d68133bbc'  # a tree other than the one the health audit pinned
        record = {'postcoef_capture_identity': {'source_tree': tree}, 'source_pin': pin,
                  'oracle_build_sha256': hashlib.sha256(build_bytes).hexdigest()}
        case.update(glvmodels_src_tree=tree, gllvmtmb_version='0.7.1',
                    oracle_build_receipt=f'{base.ROOT}/build-totoro-20261104.json',
                    oracle_source_receipt=f'{base.ROOT}/source.json')
        files = {case['oracle_build_receipt']: build_bytes, case['oracle_source_receipt']: json.dumps(source).encode()}
        return case, record, files


if __name__ == '__main__':
    unittest.main()
