#!/usr/bin/env python3
"""The live gate under the content rule: later history of any shape passes, bound changes and tampering fail.

Runs core070_postcoef_live_ref.verify on synthetic repositories. Stubbed (they need the capture host): the
private source self-test, the case-to-oracle-build binding, the capture-trust tools-tree check and the
lambda-map check. Real: the content rule, the identity tuple, case and transcript hashes, the capture
comparison, and the runtime-root checks. No Julia, R, capture or network.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import core070_postcoef_live_ref as live  # noqa: E402
import core070_postfit_p1_receipts as receipts  # noqa: E402
from test_core070_postcoef_live_ref import TRANSCRIPT, fixture  # noqa: E402
from test_postcoef_bind_commit import BIND_FILES, COMMIT_PINS, LAUNCHER, ASSEMBLE, SHA_PINS, set_value  # noqa: E402

JSON_TOOL = 'tools/core070_json_semantics.py'
BUILD = 'docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json'
BOARD = 'docs/dev-log/core070/true-parity-latest/scoreboard.md'


class LiveContentRule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='postcoef-live-content-')
        self.root = Path(self.tmp.name) / 'repo'
        self.root.mkdir()
        self.git('init', '-q', '-b', 'main')
        self.write('README.md', 'base\n')
        self.git('add', '--', 'README.md'); self.commit('base')
        self.base = self.head()
        sources = [*live.SOURCE_FILES, *BIND_FILES, JSON_TOOL]
        for rel in sources:
            real = T.parent / rel
            self.write(rel, real.read_text() if rel in (*BIND_FILES, JSON_TOOL) else f'source {rel}\n')
        self.write('src/a.jl', 'x = 1\n')
        self.write(BOARD, 'board one\n')
        self.write(live.HISTORICAL_CASE, (T.parent / live.HISTORICAL_CASE).read_text())
        self.build = b'{"build": "one"}'
        (self.root / BUILD).parent.mkdir(parents=True, exist_ok=True); (self.root / BUILD).write_bytes(self.build)
        self.git('add', '--', *sources, 'src/a.jl', BOARD, live.HISTORICAL_CASE, BUILD)
        self.commit('capture source S')
        self.source = self.head()
        self.texts = {rel: (self.root / rel).read_text() for rel in BIND_FILES}
        self.commit_files(self.bound(), 'bind B')
        self.make_evidence()
        self.runtime = Path(self.tmp.name) / 'runtime'
        subprocess.run(['git', 'clone', '-q', '--no-local', str(self.root), str(self.runtime)], check=True)
        subprocess.run(['git', '-C', str(self.runtime), 'checkout', '-q', '--detach', self.source], check=True)
        self.patches = [patch.object(live, name, new) for name, new in (
            ('check_case_source_binding', lambda *a: None), ('check_capture_trust', lambda *a: None),
            ('check_lambda_maps', lambda *a: None), ('run_source_selftest', lambda *a: TRANSCRIPT.decode()))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True,
                              text=True, env=receipts.postcoef_git_env()).stdout.strip()

    def head(self):
        return self.git('rev-parse', 'HEAD')

    def write(self, rel, text):
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    def commit(self, message):
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message)

    def commit_files(self, files, message):
        for rel, text in files.items():
            self.write(rel, text)
        self.git('add', '--', *files)
        self.commit(message)
        return self.head()

    def bound(self):
        new = dict(self.texts)
        for rel, prefix in COMMIT_PINS:
            new[rel] = set_value(new[rel], prefix, self.source)
        digest = hashlib.sha256(new[LAUNCHER].encode()).hexdigest()
        for rel, prefix in SHA_PINS:
            new[rel] = set_value(new[rel], prefix, digest)
        return new

    def make_evidence(self):
        case, report, oracle = fixture()
        record = {'postcoef_capture_identity': {'source_commit': self.source, 'source_tree': self.git('rev-parse', f'{self.source}:src')},
                  'oracle_build_sha256': hashlib.sha256(self.build).hexdigest(),
                  'postcoef_runtime': {'python_executable_path': sys.executable, 'preferences_sha256': 'absent'}}
        files = {name: f'capture file {name}\n' for name in live.FILES}
        files.update({'receipt.json': json.dumps(record), 'julia-results.json': json.dumps(report),
                      'r-oracle.json': json.dumps(oracle)})
        self.prefix = live.PREFIX
        evidence = {f'{self.prefix}/{name}': text for name, text in files.items()}
        evidence[f'{self.prefix}/verify.txt'] = TRANSCRIPT.decode()
        case.update({'glvmodels_commit': self.head(), 'oracle_build_receipt': BUILD,
                     'historical_case': {'path': live.HISTORICAL_CASE, 'sha256': live.HISTORICAL_SHA256},
                     'postcoef_capture': {'receipt': f'{self.prefix}/receipt.json',
                                          'sha256': hashlib.sha256(files['receipt.json'].encode()).hexdigest()}})
        case['batch_verifier']['log_sha256'] = hashlib.sha256(TRANSCRIPT).hexdigest()
        evidence[live.CASE] = json.dumps(case)
        self.commit_files(evidence, 'evidence E')
        self.evidence_head = self.head()

    def verify(self, head=None):
        return live.verify(self.root, head or self.head(), self.runtime)

    def reset_to_evidence(self):
        self.git('checkout', '-q', '-B', 'main', self.evidence_head)

    def test_evidence_head_verifies_and_reports_engine(self):
        result = self.verify()
        self.assertEqual(result['status'], 'POSTCOEF_LIVE_REF_VERIFIED')
        self.assertIs(result['engine_tree_matches'], True)

    def test_later_commits_of_every_shape_verify(self):
        self.commit_files({'docs/notes.md': 'later docs\n'}, 'docs')
        self.commit_files({BOARD: 'board two\n'}, 'board regenerated for another row')
        self.commit_files({'tools/other_row_tool.py': '# another lane\n'}, 'other tool')
        self.assertEqual(self.verify()['status'], 'POSTCOEF_LIVE_REF_VERIFIED')
        self.git('checkout', '-q', '-b', 'side', self.evidence_head)
        self.commit_files({'docs/side.md': 'side\n'}, 'side docs')
        self.git('checkout', '-q', 'main')
        self.git('-c', 'user.name=f', '-c', 'user.email=f@example.invalid', 'merge', '--no-ff', '-q', '-m', 'merge', 'side')
        self.assertEqual(self.verify()['status'], 'POSTCOEF_LIVE_REF_VERIFIED')

    def test_squash_shape_verifies_while_source_object_is_kept(self):
        pr = self.evidence_head
        self.git('checkout', '-q', '-B', 'squashed', self.base)
        self.git('checkout', pr, '--', '.')
        self.commit('squash of the PR')
        self.assertEqual(self.verify()['status'], 'POSTCOEF_LIVE_REF_VERIFIED')

    def test_engine_change_is_reported_by_default_and_refused_when_strict(self):
        self.commit_files({'src/a.jl': 'x = 2\n'}, 'engine change')
        self.assertIs(self.verify()['engine_tree_matches'], False)
        with patch.object(live, 'ENGINE_BOUND_STRICT', True), self.assertRaisesRegex(ValueError, 'engine'):
            self.verify()

    def test_judge_swap_after_the_capture_is_refused(self):
        self.commit_files({ASSEMBLE: (self.root / ASSEMBLE).read_text() + '\n# skip the live gate\n'}, 'swap judge')
        with self.assertRaisesRegex(ValueError, 'bound input'):
            self.verify()

    def test_receipt_hash_edit_is_refused(self):
        receipt = f'{self.prefix}/receipt.json'
        self.commit_files({receipt: (self.root / receipt).read_text() + ' '}, 'edit receipt')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.verify()

    def test_transcript_edit_is_refused(self):
        self.commit_files({f'{self.prefix}/verify.txt': 'forged\n'}, 'edit transcript')
        with self.assertRaisesRegex(ValueError, 'transcript'):
            self.verify()

    def test_evidence_byte_change_is_refused_by_the_comparison(self):
        path = f'{self.prefix}/julia-results.json'
        report = json.loads((self.root / path).read_text())
        report['cases'][live.CID]['three_fit']['julia_lengths'] = [0, 5, 7]
        self.commit_files({path: json.dumps(report)}, 'edit results')
        with self.assertRaisesRegex(ValueError, 'case identity or harness differs'):
            self.verify()

    def test_extra_capture_file_is_refused(self):
        self.commit_files({f'{self.prefix}/notes.txt': 'extra\n'}, 'extra file')
        with self.assertRaisesRegex(ValueError, 'bound input'):
            self.verify()

    def test_runtime_checkout_not_at_the_source_is_refused(self):
        subprocess.run(['git', '-C', str(self.runtime), 'checkout', '-q', '--detach', self.evidence_head], check=True)
        with self.assertRaisesRegex(ValueError, 'runtime checkout is not the capture commit'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
