#!/usr/bin/env python3
"""Offline controls for the POST-COEF selected-ref live-verifier boundary."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import core070_postcoef_live_ref as live

ROOT = 'docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle'
PIN = '9539352f66f2db2cc26b1c393e67212a359b60c9'
CAPTURE = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1'
LEGACY = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postfit-policy-p1'
TRANSCRIPT = (b'CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\n'
              b'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
              b'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\n'
              b'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')


def fixture():
    actual = {'pass': True, 'three_fit': {'julia_lengths': [0, 5, 6]}}
    case = {
        'case_id': live.RECEIPT_CID, 'harness_case_id': live.CID, 'source_ids': ['postfit-policy/POST-COEF-EMPTY'],
        'verdict': 'PASS', 'harness_pass': True, 'evidence_kind': 'numeric_r_vs_julia',
        'glvmodels_worktree_dirty': [], 'pin': 'P1', 'reference_commit': PIN,
        'harness_fields': actual,
        'batch_verifier': {'status': 'PASS', 'exit_code': 0,
            'tool': 'tools/core070_verify_postfit_policy_batch.py',
            'accept_marker': 'CORE070_POSTFIT_POLICY_BATCH_VERIFIED', 'log': f'{CAPTURE}/verify.txt',
            'log_sha256': 'a' * 64},
        'raw': [f'{CAPTURE}/julia-results.json', f'{CAPTURE}/r-oracle.json'],
        'comparison': {'pin': 'P1', 'cases': [{'case_id': live.CID, 'kind': 'integer_equality',
            'r_value': [0, 5, 6], 'julia_value': [0, 5, 6], 'max_abs_diff': 0,
            'tolerance': 0.5, 'discriminating': True}]},
    }
    report = {'cases': {live.CID: actual}}
    oracle = {'postcoef_three_fit': {'fits': [{'length': n} for n in (0, 5, 6)]}}
    return case, report, oracle


class LiveRefTests(unittest.TestCase):
    def test_assembler_rejects_untrusted_launcher_bytes_and_symlinks(self):
        from true_parity_assemble import postcoef_launcher_problem
        launcher = Path(__file__).with_name('core070_postcoef_gate_launcher.py')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'launcher.py'; path.write_bytes(launcher.read_bytes())
            self.assertIsNone(postcoef_launcher_problem(path))
            path.write_bytes(path.read_bytes() + b'\n# tampered\n')
            self.assertIsNotNone(postcoef_launcher_problem(path))
            path.unlink(); path.symlink_to(launcher)
            self.assertIsNotNone(postcoef_launcher_problem(path))

    def test_assembler_protected_launcher_rejects_unapproved_pin(self):
        from true_parity_assemble import run_postcoef_gate
        launcher = Path(__file__).with_name('core070_postcoef_gate_launcher.py')
        with self.assertRaisesRegex(ValueError, 'protected gate rejected'):
            run_postcoef_gate(launcher, 'identity', ['--repo', str(launcher.parent.parent),
                              '--head', 'a' * 40, '--runtime-root', str(launcher.parent.parent)],
                              dict(os.environ))

    def test_assembler_executes_the_bytes_it_checked(self):
        import true_parity_assemble as assemble
        launcher = Path(__file__).with_name('core070_postcoef_gate_launcher.py')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'launcher.py'; path.write_bytes(launcher.read_bytes())
            real_run = subprocess.run
            def swap_then_run(*args, **kwargs):
                path.write_text("print('forged launcher')\n")
                return real_run(*args, **kwargs)
            with patch.object(assemble.subprocess, 'run', side_effect=swap_then_run):
                with self.assertRaisesRegex(ValueError, 'protected gate rejected'):
                    assemble.run_postcoef_gate(path, 'identity', ['--repo', tmp, '--head', 'a' * 40,
                                                '--runtime-root', tmp], dict(os.environ))

    def test_rejects_mixed_capture_pointers(self):
        case, _, _ = fixture()
        self.assertIsNone(live.check_capture_pointers(case))
        for field in ('julia-results.json', 'r-oracle.json', 'verify.txt'):
            bad = copy.deepcopy(case)
            if field == 'verify.txt':
                bad['batch_verifier']['log'] = f'{LEGACY}/{field}'
            else:
                i = 0 if field == 'julia-results.json' else 1
                bad['raw'][i] = f'{LEGACY}/{field}'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'capture pointers'):
                live.check_capture_pointers(bad)
        for value in (None, 'bad', 'A' * 64):
            bad = copy.deepcopy(case); bad['batch_verifier']['log_sha256'] = value
            with self.subTest(log_hash=value), self.assertRaisesRegex(ValueError, 'capture pointers'):
                live.check_capture_pointers(bad)

    def test_assembler_rejects_mixed_capture_pointers(self):
        from true_parity_assemble import postcoef_capture_pointers_problem
        case, _, _ = fixture()
        case['postcoef_capture'] = {'receipt': f'{CAPTURE}/receipt.json', 'sha256': 'a' * 64}
        self.assertIsNone(postcoef_capture_pointers_problem(case))
        for field in ('receipt.json', 'julia-results.json', 'r-oracle.json', 'verify.txt'):
            bad = copy.deepcopy(case)
            if field == 'receipt.json': bad['postcoef_capture']['receipt'] = f'{LEGACY}/{field}'
            elif field == 'verify.txt': bad['batch_verifier']['log'] = f'{LEGACY}/{field}'
            else: bad['raw'][0 if field == 'julia-results.json' else 1] = f'{LEGACY}/{field}'
            with self.subTest(field=field): self.assertIsNotNone(postcoef_capture_pointers_problem(bad))

    def test_verifier_transcript_is_hash_bound_and_contains_controls(self):
        case, _, _ = fixture()
        log = (b'CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\n'
               b'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
               b'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\n'
               b'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')
        case['batch_verifier']['log_sha256'] = hashlib.sha256(log).hexdigest()
        self.assertIsNone(live.check_verifier_transcript(case, log))
        with self.assertRaisesRegex(ValueError, 'transcript'):
            live.check_verifier_transcript(case, log + b'changed')
        forged = b'forged\n'; case['batch_verifier']['log_sha256'] = hashlib.sha256(forged).hexdigest()
        with self.assertRaisesRegex(ValueError, 'transcript'):
            live.check_verifier_transcript(case, forged)

    def test_assembler_rejects_symlinked_capture_directory(self):
        import true_parity_assemble as assemble
        case, _, _ = fixture()
        case['postcoef_capture'] = {'receipt': f'{CAPTURE}/receipt.json', 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'actual'; actual.mkdir()
            link = root / CAPTURE; link.parent.mkdir(parents=True); link.symlink_to(actual, target_is_directory=True)
            with patch.object(assemble.subprocess, 'run', side_effect=AssertionError('verifier must not run')):
                self.assertIn('symlink', assemble.postcoef_capture_problem(case, root, {}))

    def test_assembler_rejects_symlinked_capture_artifact(self):
        import true_parity_assemble as assemble
        case, _, _ = fixture()
        case['postcoef_capture'] = {'receipt': f'{CAPTURE}/receipt.json', 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state = root / CAPTURE; state.mkdir(parents=True)
            (state / 'r-oracle.json').symlink_to(root / 'missing')
            with patch.object(assemble.subprocess, 'run', side_effect=AssertionError('verifier must not run')):
                self.assertIn('symlink', assemble.postcoef_capture_problem(case, root, {}))

    def test_assembler_rejects_missing_or_forged_transcript(self):
        import true_parity_assemble as assemble
        case, _, _ = fixture()
        case['postcoef_capture'] = {'receipt': f'{CAPTURE}/receipt.json', 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state = root / CAPTURE; state.mkdir(parents=True)
            with patch.object(assemble.subprocess, 'run', side_effect=AssertionError('verifier must not run')):
                self.assertIn('transcript missing', assemble.postcoef_capture_problem(case, root, {}))
                forged = b'forged\n'; (state / 'verify.txt').write_bytes(forged)
                case['batch_verifier']['log_sha256'] = hashlib.sha256(forged).hexdigest()
                self.assertIn('transcript differs', assemble.postcoef_capture_problem(case, root, {}))
    def test_assembler_rejects_each_mismatched_identity_field(self):
        from true_parity_assemble import postcoef_live_identity_matches
        identity = {'selected_commit': 'a' * 40, 'selected_tree': 'b' * 40,
                    'source_commit': 'c' * 40, 'case_commit': 'd' * 40,
                    'case_sha256': 'e' * 64, 'capture_sha256': 'f' * 64,
                    'oracle_build_sha256': '1' * 64, 'runtime_record': {'executable_sha256': '2' * 64},
                    'runtime_root': '/runtime', 'python_executable': '/python'}
        verified = dict(identity, status='POSTCOEF_LIVE_REF_VERIFIED', capture_commit=identity['source_commit'])
        self.assertTrue(postcoef_live_identity_matches(verified, identity))
        self.assertFalse(postcoef_live_identity_matches(verified, {}))
        for key in identity:
            with self.subTest(field=key):
                bad = dict(verified); bad[key] = 'wrong'
                self.assertFalse(postcoef_live_identity_matches(bad, identity))
                bad = dict(verified); del bad[key]
                self.assertFalse(postcoef_live_identity_matches(bad, identity))
        bad = dict(verified, capture_commit='wrong')
        self.assertFalse(postcoef_live_identity_matches(bad, identity))

    def test_separate_receipt_identity_requires_exact_harness_mapping(self):
        case, report, oracle = fixture()
        case['case_id'] = 'CORE070-POSTFIT-COEF-THREE-FIT-NATIVE'
        case['harness_case_id'] = 'CORE070-POSTFIT-COEF-EMPTY-NATIVE'
        case['comparison']['cases'][0]['case_id'] = live.CID
        self.assertIsNone(live.check_comparison(case, report, oracle))
        for field, value in [('case_id', live.CID), ('harness_case_id', 'spoofed')]:
            bad = copy.deepcopy(case); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                live.check_comparison(bad, report, oracle)

    def test_historical_binding_requires_same_non_discriminating_bytes(self):
        data = (Path(__file__).resolve().parent.parent / live.HISTORICAL_CASE).read_bytes()
        case = {'historical_case': {'path': f'docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/{live.CID}.json', 'sha256': hashlib.sha256(data).hexdigest()}}
        self.assertIsNone(live.check_historical_case_binding(case, data, data))
        for original, selected in [(data, data+b' '), (b'{}',data)]:
            with self.assertRaises(ValueError): live.check_historical_case_binding(case, original, selected)
        for historical in ({'case_id':live.CID, 'comparison':{'cases':[{'discriminating': False, 'r_value':999}]}},
                           {'case_id':live.CID, 'comparison':{'cases':[1]}}):
            bad = json.dumps(historical).encode()
            case['historical_case']['sha256'] = hashlib.sha256(bad).hexdigest()
            with self.assertRaises(ValueError): live.check_historical_case_binding(case, bad, bad)
        with patch.object(live, 'HISTORICAL_SHA256', hashlib.sha256(bad).hexdigest()):
            with self.assertRaises(ValueError): live.check_historical_case_binding(case, bad, bad)

    def test_exact_comparison_and_mutations(self):
        case, report, oracle = fixture()
        self.assertIsNone(live.check_comparison(case, report, oracle))
        mutations = [
            lambda c: c['comparison']['cases'][0].__setitem__('case_id', live.RECEIPT_CID),
            lambda c: c['comparison']['cases'][0].__setitem__('tolerance', 1),
            lambda c: c['comparison']['cases'][0]['r_value'].__setitem__(0, False),
            lambda c: c['comparison']['cases'][0].__setitem__('max_abs_diff', False),
            lambda c: c['comparison']['cases'][0].__setitem__('discriminating', False),
            lambda c: c['batch_verifier'].__setitem__('exit_code', False),
            lambda c: c['harness_fields']['three_fit']['julia_lengths'].__setitem__(0, False),
            lambda c: c.__setitem__('reference_commit', '0' * 40),
            lambda c: c.__setitem__('verdict', 'FAIL'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(case); mutate(bad)
                with self.assertRaises(ValueError):
                    live.check_comparison(bad, report, oracle)

    def test_case_must_reference_exact_capture_build(self):
        case, _, _ = fixture()
        pin = {'reference_commit': PIN, 'archive_sha256': 'a' * 64,
               'namespace_sha256': 'b' * 64, 'source_tree_sha256': 'c' * 64,
               'installed_tree_sha256': 'd' * 64, 'marker_sha256': 'e' * 64,
               'version': '0.7.1'}
        build = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'source_tree_sha256',
                                      'installed_tree_sha256', 'marker_sha256')}
        build['gllvmTMB_version'] = pin['version']
        build_bytes = json.dumps(build).encode()
        source = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'namespace_sha256', 'source_tree_sha256')}
        record = {'postcoef_capture_identity': {'source_tree': '1' * 40}, 'source_pin': pin,
                  'oracle_build_sha256': hashlib.sha256(build_bytes).hexdigest()}
        case.update(glvmodels_src_tree='1' * 40, gllvmtmb_version='0.7.1',
                    oracle_build_receipt=f'{ROOT}/build-totoro-20261104.json', oracle_source_receipt=f'{ROOT}/source.json')
        files = {case['oracle_build_receipt']: build_bytes, case['oracle_source_receipt']: json.dumps(source).encode()}
        self.assertIsNone(live.check_case_source_binding(case, record, files.__getitem__))
        for changed in ('glvmodels_src_tree', 'oracle_build_receipt', 'gllvmtmb_version'):
            bad = copy.deepcopy(case); bad[changed] = 'wrong'
            with self.subTest(field=changed), self.assertRaises((ValueError, KeyError)):
                live.check_case_source_binding(bad, record, files.__getitem__)
        files[case['oracle_build_receipt']] = b'wrong build'
        with self.assertRaisesRegex(ValueError, 'oracle build'):
            live.check_case_source_binding(case, record, files.__getitem__)

    def test_verification_identity_tracks_exact_tree_case_build_and_runtime(self):
        with tempfile.TemporaryDirectory(prefix='postcoef-identity-') as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout
            git('init', '-q'); git('config', 'user.name', 'fixture'); git('config', 'user.email', 'fixture@example.invalid')
            build_path = f'{ROOT}/build.json'; build_bytes = b'{"build":"one"}'
            build = root / build_path; build.parent.mkdir(parents=True); build.write_bytes(build_bytes)
            historical = root / live.HISTORICAL_CASE; historical.parent.mkdir(parents=True)
            historical_bytes = (Path(__file__).resolve().parent.parent / live.HISTORICAL_CASE).read_bytes()
            historical.write_bytes(historical_bytes)
            git('add', live.HISTORICAL_CASE, build_path); git('commit', '-qm', 'captured source')
            source = git('rev-parse', 'HEAD').strip()
            record = {'postcoef_capture_identity': {'source_commit': source},
                      'oracle_build_sha256': hashlib.sha256(build_bytes).hexdigest(),
                      'postcoef_runtime': {'executable_sha256': 'b' * 64}}
            recbytes = json.dumps(record).encode(); recpath = root / live.PREFIX / 'receipt.json'
            recpath.parent.mkdir(parents=True); recpath.write_bytes(recbytes)
            case = {'glvmodels_commit': source, 'oracle_build_receipt': build_path,
                    'historical_case': {'path': live.HISTORICAL_CASE, 'sha256': hashlib.sha256(historical_bytes).hexdigest()},
                    'postcoef_capture': {'receipt': live.PREFIX + '/receipt.json',
                                         'sha256': hashlib.sha256(recbytes).hexdigest()},
                    'raw': [live.PREFIX + '/julia-results.json', live.PREFIX + '/r-oracle.json'],
                    'batch_verifier': {'log': live.PREFIX + '/verify.txt', 'log_sha256': hashlib.sha256(TRANSCRIPT).hexdigest()}}
            casepath = root / live.CASE; casepath.parent.mkdir(parents=True, exist_ok=True); casepath.write_text(json.dumps(case))
            (root / live.PREFIX / 'verify.txt').write_bytes(TRANSCRIPT)
            git('add', live.CASE, live.PREFIX + '/receipt.json', live.PREFIX + '/verify.txt', build_path); git('commit', '-qm', 'selected one')
            first = git('rev-parse', 'HEAD').strip(); identity = live.verification_identity(root, first, root)
            self.assertEqual(identity['selected_tree'], git('rev-parse', 'HEAD^{tree}').strip())
            with self.assertRaisesRegex(ValueError, 'not a commit'):
                live.verification_identity(root, identity['selected_tree'], root)
            self.assertEqual(identity['capture_sha256'], hashlib.sha256(recbytes).hexdigest())
            self.assertEqual(identity['oracle_build_sha256'], hashlib.sha256(build_bytes).hexdigest())
            self.assertNotEqual(identity, live.verification_identity(root, first, root / 'other-runtime'))
            case['review_note'] = 'changed'; casepath.write_text(json.dumps(case)); git('add', live.CASE); git('commit', '-qm', 'selected two')
            second = git('rev-parse', 'HEAD').strip(); changed = live.verification_identity(root, second, root)
            for field in ('selected_commit', 'selected_tree', 'case_sha256'):
                self.assertNotEqual(identity[field], changed[field])
            self.assertEqual(identity['capture_sha256'], changed['capture_sha256'])

    def test_selected_blob_ignores_git_replacement_refs(self):
        with tempfile.TemporaryDirectory(prefix='postcoef-replace-') as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout
            git('init', '-q'); git('config', 'user.name', 'fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (root / 'value').write_text('original'); git('add', 'value'); git('commit', '-qm', 'original')
            original = git('rev-parse', 'HEAD').strip()
            (root / 'value').write_text('forged'); git('add', 'value'); git('commit', '-qm', 'forged')
            forged = git('rev-parse', 'HEAD').strip(); git('replace', original, forged)
            self.assertEqual(live.blob(root, original, 'value'), b'original')

    def test_nested_git_calls_ignore_ambient_overrides_and_restore_environment(self):
        from core070_postfit_p1_receipts import postcoef_isolated_git_environment
        with tempfile.TemporaryDirectory(prefix='postcoef-env-') as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout
            git('init', '-q'); git('config', 'user.name', 'fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (root / 'value').write_text('original'); git('add', 'value'); git('commit', '-qm', 'original')
            original = git('rev-parse', 'HEAD').strip()
            (root / 'value').write_text('forged'); git('add', 'value'); git('commit', '-qm', 'forged')
            git('replace', original, 'HEAD')
            with patch.dict(os.environ, {'GIT_DIR': '/invalid/foreign/repo', 'GIT_NO_REPLACE_OBJECTS': '0'}):
                with postcoef_isolated_git_environment():
                    self.assertNotIn('GIT_DIR', os.environ)
                    self.assertEqual(git('show', original + ':value'), 'original')
                self.assertEqual(os.environ['GIT_DIR'], '/invalid/foreign/repo')
                self.assertEqual(os.environ['GIT_NO_REPLACE_OBJECTS'], '0')

    def test_selected_blob_requires_regular_file_and_safe_path(self):
        with tempfile.TemporaryDirectory(prefix='postcoef-live-ref-') as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout
            git('init', '-q'); (root / 'data').mkdir(); (root / 'data/x').write_bytes(b'\x00\xff')
            (root / 'data/link').symlink_to('x')
            git('add', 'data/x', 'data/link')
            git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
            head = git('rev-parse', 'HEAD').strip()
            (root / 'data/x').write_bytes(b'changed')
            self.assertEqual(live.blob(root, head, 'data/x'), b'\x00\xff')
            for commit, path in ((head, 'data/link'), (head, '../escape'), ('HEAD', 'data/x')):
                with self.subTest(commit=commit, path=path), self.assertRaises(ValueError):
                    live.blob(root, commit, path)


class SourceSelftestTests(unittest.TestCase):
    """Synthetic verifier stubs exercise the source-pinning wrapper only."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo'; self.root.mkdir(); self.state=Path(self.temp.name)/'state'; self.state.mkdir()
        self.git('init','-q'); self.git('config','user.name','fixture'); self.git('config','user.email','fixture@example.invalid')
        for rel in live.SOURCE_FILES:
            path=self.root/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fixture\n')
        dependency=self.root/'tools/parity_oracle.py'
        dependency.write_text(("VALUE = 'safe'\n").ljust(512, ' '))
        verifier=self.root/live.SOURCE_FILES[0]
        lines=['import os, sys, subprocess', 'from pathlib import Path', 'sys.path.insert(0, str(Path(__file__).resolve().parent))',
               "assert not (Path(__file__).resolve().parents[1] / '.git/objects/info/alternates').exists()",
               'import parity_oracle', "assert parity_oracle.VALUE == 'safe'", "assert sys.flags.isolated and sys.dont_write_bytecode and os.environ['GIT_NO_REPLACE_OBJECTS'] == '1'", "assert 'GIT_DIR' not in os.environ",
               "subprocess.run([sys.executable, '-c', \"import os, sys; assert sys.dont_write_bytecode; assert 'PYTHONPYCACHEPREFIX' not in os.environ\"], check=True)"]
        lines += [f'print({marker!r})' for marker in live.SELFTEST_MARKERS]
        verifier.write_text('\n'.join(lines)+'\n')
        self.git('add',*live.SOURCE_FILES); self.git('commit','-qm','source')
        self.source=self.git('rev-parse','HEAD'); self.runtime={'python_executable_path':__import__('sys').executable}
    def git(self,*args):
        return subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True,text=True).stdout.strip()
    def test_source_pinned_stub_runs_with_isolated_environment(self):
        escaped=self.root/'escaped-cache'
        with patch.dict(__import__('os').environ, {'GIT_DIR':'/invalid/foreign/repo', 'PYTHONPYCACHEPREFIX':str(escaped)}):
            output=live.run_source_selftest(self.root,self.source,self.state,self.runtime)
        self.assertTrue(all(marker in output.splitlines() for marker in live.SELFTEST_MARKERS))
        self.assertFalse((self.root/'tools/__pycache__').exists())
        self.assertFalse(escaped.exists())
    def test_shared_source_does_not_leak_object_alternates(self):
        shared = self.root.parent / 'shared-source'
        subprocess.run(['git', 'clone', '--quiet', '--shared', str(self.root), str(shared)], check=True)
        self.assertTrue((shared / '.git/objects/info/alternates').is_file())
        output = live.run_source_selftest(shared, self.source, self.state, self.runtime)
        self.assertTrue(all(marker in output.splitlines() for marker in live.SELFTEST_MARKERS))

    def test_untracked_forged_bytecode_is_not_executed(self):
        import importlib.util, os, py_compile
        source = self.root/'tools/parity_oracle.py'; info=source.stat()
        sentinel = self.root.parent/'forged-executed'
        payload = f"open({str(sentinel)!r}, 'w').write('forged')\nVALUE = 'safe'\n"
        self.assertLess(len(payload), 512)
        fake=self.root.parent/'fake.py'; fake.write_text(payload.ljust(512, ' '))
        os.utime(fake, ns=(info.st_atime_ns, info.st_mtime_ns))
        cache=Path(importlib.util.cache_from_source(str(source))); cache.parent.mkdir()
        py_compile.compile(str(fake), cfile=str(cache), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        original=cache.read_bytes()
        output=live.run_source_selftest(self.root,self.source,self.state,self.runtime)
        self.assertTrue(all(marker in output.splitlines() for marker in live.SELFTEST_MARKERS))
        self.assertFalse(sentinel.exists())
        self.assertEqual(cache.read_bytes(),original)

    def test_changed_verifier_and_wrong_python_rejected(self):
        verifier=self.root/live.SOURCE_FILES[0]; verifier.write_text('print("forged")\n')
        with self.assertRaisesRegex(ValueError,'differs'): live.run_source_selftest(self.root,self.source,self.state,self.runtime)
        with self.assertRaisesRegex(ValueError,'Python'): live.run_source_selftest(self.root,self.source,self.state,{'python_executable_path':'/invalid/python'})
    def test_missing_selftest_marker_is_rejected(self):
        verifier=self.root/live.SOURCE_FILES[0]; verifier.write_text('print("incomplete")\n')
        self.git('add',live.SOURCE_FILES[0]); self.git('commit','-qm','incomplete verifier')
        with self.assertRaisesRegex(ValueError,'self-test rejected'):
            live.run_source_selftest(self.root,self.git('rev-parse','HEAD'),self.state,self.runtime)



if __name__ == '__main__':
    unittest.main()
