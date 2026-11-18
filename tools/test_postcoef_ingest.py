#!/usr/bin/env python3
"""Offline controls for the --ingest-postcoef mode (write-once case receipt and capture prefix).

Pure Python on a throwaway git repo; no Julia, R, Totoro or network. The private source self-test is stubbed (it
needs the capture host's python), so what is checked here is the mode's own gates and its writes.
"""
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core070_postcoef_live_ref as live
import core070_postfit_p1_receipts as receipt

CID = receipt.COEF_EMPTY_CID
PREFIX = f'{receipt.REC_REL}/postcoef-empty-p1'
CASE = f'{receipt.REC_REL}/cases/{receipt.COEF_THREE_FIT_CID}.json'
MARKERS = ('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\nCORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
           'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\nCORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')
PIN = {'reference_commit': 'r' * 40, 'archive_sha256': 'a' * 64, 'source_tree_sha256': 'b' * 64,
       'installed_tree_sha256': 'c' * 64, 'marker_sha256': 'd' * 64, 'namespace_sha256': 'e' * 64, 'version': '0.7.1'}
H = lambda data: hashlib.sha256(data).hexdigest()


class Fixture:
    """A clean one-commit repo (S = HEAD) holding the oracle receipts, plus a capture directory outside it."""
    def __init__(self, tmp):
        self.root = Path(tmp) / 'repo'; self.state = Path(tmp) / 'state'
        self.root.mkdir(); self.state.mkdir()
        oracle_dir = self.root / 'docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle'
        oracle_dir.mkdir(parents=True)
        self.build = json.dumps(dict(PIN, gllvmTMB_version='0.7.1')).encode()
        (oracle_dir / 'build.json').write_bytes(self.build)
        (oracle_dir / 'source.json').write_text(json.dumps(PIN))
        (self.root / 'src').mkdir(); (self.root / 'src/a.jl').write_text('x = 1\n')
        for args in (['init', '-q'], ['config', 'user.name', 'Fixture'], ['config', 'user.email', 'f@example.invalid'],
                     ['add', '.'], ['commit', '-qm', 'S']):
            self.git(*args)
        self.head = self.git('rev-parse', 'HEAD')
        files = {'julia-results.json': json.dumps({'cases': {CID: {
                     'pass': True, 'r_length': 0, 'julia_length': 0,
                     'three_fit': {'julia_lengths': [0, 5, 6], 'fit_health': {'R': [{'h': n} for n in (0, 5, 6)], 'Julia': []}}}}}),
                 'r-oracle.json': json.dumps({'postcoef_three_fit': {'fits': [{'length': n, 'fit_health': {'h': n}} for n in (0, 5, 6)]}}),
                 'results.tsv': 'case\tpass\n', 'diagnostics.log': '', 'julia-stdout.log': 'out', 'julia-stderr.log': '',
                 'julia-Manifest.toml': 'manifest', 'oracle-verify-preload.log': 'p', 'oracle-verify-before.log': 'b',
                 'oracle-verify-after.log': 'a', 'runner-snapshots/julia-launcher': 'L',
                 'runner-snapshots/core070_postfit_policy_batch.R': 'R', 'runner-snapshots/core070_postfit_policy_batch.jl': 'J'}
        for name, text in files.items():
            (self.state / name).parent.mkdir(parents=True, exist_ok=True); (self.state / name).write_text(text)
        h = lambda name: H((self.state / name).read_bytes())
        record = {'status': 'PASS', 'oracle_build_sha256': H(self.build), 'source_pin': PIN,
                  'postcoef_capture_identity': {'source_commit': self.head, 'source_tree': self.git('rev-parse', 'HEAD:src'),
                                                'r_oracle_sha256': h('r-oracle.json'),
                                                'julia_runner_sha256': h('runner-snapshots/core070_postfit_policy_batch.jl')},
                  'postcoef_runtime': {'manifest_sha256': h('julia-Manifest.toml')}}
        record.update({key: h(name) for key, name in receipt.POSTCOEF_CAPTURE_HASHES})
        self.write_record(record)
        self.transcript = MARKERS

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True, text=True).stdout.strip()

    def write_record(self, record):
        (self.state / 'receipt.json').write_text(json.dumps(record)); self.record = record

    def prepare(self, state, transcript_sink=None):
        """Stand-in for prepare_separate_postcoef_case: same draft shape, after the (stubbed) private self-test."""
        if transcript_sink is not None:
            transcript_sink.append(self.transcript)
        report = json.loads((state / 'julia-results.json').read_text())['cases'][CID]
        cmp = {'case_id': CID, 'kind': 'integer_equality', 'r_value': [0, 5, 6], 'julia_value': [0, 5, 6],
               'max_abs_diff': 0, 'tolerance': 0.5, 'discriminating': True}
        return {'case_id': receipt.COEF_THREE_FIT_CID, 'harness_case_id': CID, 'source_ids': ['postfit-policy/POST-COEF-EMPTY'],
                'verdict': 'UNQUALIFIED', 'historical_case': {'path': 'x', 'sha256': 'y'}, 'harness_fields': report,
                'comparison': {'pin': 'P1', 'cases': [cmp]}, 'glvmodels_commit': self.git('rev-parse', 'HEAD'),
                'glvmodels_src_tree': self.record['postcoef_capture_identity']['source_tree'], 'glvmodels_worktree_dirty': None,
                'oracle_build_receipt': receipt.ORACLE_BUILD, 'postcoef_capture': {'receipt': f'{PREFIX}/receipt.json', 'sha256': 'z'},
                'raw': [f'{PREFIX}/julia-results.json', f'{PREFIX}/r-oracle.json'],
                'batch_verifier': {'status': 'UNVERIFIED', 'log': f'{PREFIX}/verify.txt',
                                   'log_sha256': H(self.transcript.encode())}}

    def run(self, source=None, prepare=None):
        with patch.object(receipt, 'ROOT', self.root), patch.object(receipt, 'REC', self.root / receipt.REC_REL), \
                patch.object(receipt, 'prepare_separate_postcoef_case', prepare or self.prepare), \
                contextlib.redirect_stdout(open(os.devnull, 'w')):
            receipt.ingest_postcoef(self.state, source or self.head)

    def written(self):
        return sorted(str(p.relative_to(self.root)) for p in (self.root / receipt.REC_REL).rglob('*') if p.is_file())


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)

    def refused(self, pattern, **kwargs):
        with self.assertRaisesRegex(SystemExit, pattern):
            self.fx.run(**kwargs)
        self.assertEqual(self.fx.written(), [])

    def test_happy_path_writes_case_prefix_and_verify_txt(self):
        self.fx.run()
        written = self.fx.written()
        self.assertIn(CASE, written)
        self.assertIn(f'{PREFIX}/verify.txt', written)
        self.assertEqual(len(written), len([f for f in self.fx.state.rglob('*') if f.is_file()]) + 2)  # capture + verify.txt + case
        self.assertEqual((self.fx.root / PREFIX / 'verify.txt').read_text(), MARKERS)
        for name in ('receipt.json', 'julia-results.json', 'r-oracle.json', 'runner-snapshots/julia-launcher'):
            self.assertEqual((self.fx.root / PREFIX / name).read_bytes(), (self.fx.state / name).read_bytes())
        case = json.loads((self.fx.root / CASE).read_text())
        self.assertEqual((case['verdict'], case['harness_pass'], case['glvmodels_worktree_dirty']), ('PASS', True, []))
        self.assertEqual(case['batch_verifier']['log_sha256'], H(MARKERS.encode()))
        self.assertNotIn('signed_by', json.dumps(case))
        self.assertFalse((self.fx.root / 'docs/dev-log/core070/true-parity-latest/case-map-postfit.json').exists())
        report = json.loads((self.fx.state / 'julia-results.json').read_text())
        live.check_comparison(case, report, json.loads((self.fx.state / 'r-oracle.json').read_text()))

    def test_refuses_existing_case_or_prefix_without_writing(self):
        target = self.fx.root / CASE
        target.parent.mkdir(parents=True); target.write_text('old')
        subprocess.run(['git', '-C', str(self.fx.root), 'add', '-f', '.'], check=True)
        subprocess.run(['git', '-C', str(self.fx.root), 'commit', '-qm', 'case'], check=True, capture_output=True)
        with self.assertRaisesRegex(SystemExit, 'overwrite'):
            self.fx.run()
        self.assertEqual(target.read_text(), 'old'); self.assertFalse((self.fx.root / PREFIX).exists())
        target.unlink()
        subprocess.run(['git', '-C', str(self.fx.root), 'commit', '-qam', 'rm'], check=True, capture_output=True)
        (self.fx.root / PREFIX).mkdir(parents=True); (self.fx.root / PREFIX / 'keep').write_text('k')
        subprocess.run(['git', '-C', str(self.fx.root), 'add', '-f', '.'], check=True)
        subprocess.run(['git', '-C', str(self.fx.root), 'commit', '-qm', 'prefix'], check=True, capture_output=True)
        with self.assertRaisesRegex(SystemExit, 'overwrite'):
            self.fx.run()
        self.assertEqual(os.listdir(self.fx.root / PREFIX), ['keep'])

    def test_refuses_dirty_tree_before_any_check_or_write(self):
        (self.fx.root / 'stray.txt').write_text('x')
        boom = lambda *a, **k: self.fail('prepare must not run')
        self.refused('dirty', prepare=boom)
        (self.fx.root / 'stray.txt').unlink()
        (self.fx.root / 'src/a.jl').write_text('x = 2\n')
        self.refused('dirty', prepare=boom)

    def test_refuses_capture_from_a_different_source_commit(self):
        boom = lambda *a, **k: self.fail('prepare must not run')
        self.refused('not the expected one', source='1' * 40, prepare=boom)
        self.refused('full commit id', source='abc', prepare=boom)
        record = self.fx.record
        record['postcoef_capture_identity']['source_tree'] = '2' * 40
        self.fx.write_record(record)
        self.refused('src tree', prepare=boom)

    def test_refuses_any_hash_that_differs_from_the_capture_receipt(self):
        for name in ('r-oracle.json', 'julia-results.json', 'julia-Manifest.toml', 'runner-snapshots/julia-launcher'):
            with self.subTest(name=name):
                path = self.fx.state / name; old = path.read_bytes(); path.write_bytes(old + b' ')
                self.refused('hash differs', prepare=lambda *a, **k: self.fail('prepare must not run'))
                path.write_bytes(old)

    def test_refuses_diagnostics_extra_files_and_non_three_fit(self):
        extra = self.fx.state / 'notes.txt'; extra.write_text('x')
        self.refused('allow-list', prepare=lambda *a, **k: self.fail('prepare must not run'))
        extra.unlink()
        results = self.fx.state / 'julia-results.json'
        report = json.loads(results.read_text()); del report['cases'][CID]['three_fit']
        results.write_text(json.dumps(report))
        self.fx.record['julia_results_sha256'] = H(results.read_bytes()); self.fx.write_record(self.fx.record)
        self.refused('three-fit')

    def test_refuses_when_the_live_checks_reject_the_finalized_case(self):
        self.fx.transcript = 'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n'  # three markers missing
        with self.assertRaisesRegex(ValueError, 'markers'):
            self.fx.run()
        self.assertEqual(self.fx.written(), [])
        self.fx.transcript = MARKERS
        with patch.object(live, 'check_comparison', side_effect=ValueError('boom')):
            with self.assertRaisesRegex(ValueError, 'boom'):
                self.fx.run()
        self.assertEqual(self.fx.written(), [])

    def test_partial_write_is_rolled_back(self):
        real = Path.open; calls = []
        def flaky(path, mode='r', *a, **k):
            if mode == 'xb':
                calls.append(path)
                if len(calls) == 5:
                    raise OSError('disk full')
            return real(path, mode, *a, **k)
        with patch.object(Path, 'open', flaky), self.assertRaisesRegex(OSError, 'disk full'):
            self.fx.run()
        self.assertEqual(len(calls), 5); self.assertEqual(self.fx.written(), [])

    def test_prepare_returns_transcript_through_the_sink(self):
        state = Path(self.tmp.name) / 'prep'; state.mkdir()
        harness = {'pass': True, 'three_fit': {'julia_lengths': [0, 5, 6]}}
        oracle = {'postcoef_three_fit': {'fits': [{'length': n} for n in (0, 5, 6)]}}
        record = {'status': 'PASS', 'postcoef_capture_identity': {'source_commit': 'a' * 40, 'source_tree': 'b' * 40},
                  'postcoef_runtime': {'python_executable_path': sys.executable}}
        for name, value in [('receipt.json', record), ('julia-results.json', {'cases': {CID: harness}}), ('r-oracle.json', oracle)]:
            (state / name).write_text(json.dumps(value))
        sink = []
        with patch.object(receipt, 'postcoef_source_compatible', return_value=True), \
                patch.object(receipt, 'run_private_capture_selftest', return_value='fixture transcript'), \
                patch.object(receipt, 'select_postcoef_build', return_value='selected-build'):
            receipt.prepare_separate_postcoef_case(state, transcript_sink=sink)
        self.assertEqual(sink, ['fixture transcript'])

    def test_cli_routes_to_ingest_and_forwards_expected_source(self):
        with patch('sys.argv', ['tool', '--ingest-postcoef', '/s', '--expected-source', 'f' * 40]), \
                patch.object(receipt, 'ingest_postcoef') as ingest:
            receipt.main()
        ingest.assert_called_once_with(Path('/s'), 'f' * 40)


if __name__ == '__main__':
    unittest.main()
