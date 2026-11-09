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
import core070_postcoef_fit_health as health

ROOT = 'docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle'
PIN = '9539352f66f2db2cc26b1c393e67212a359b60c9'
CAPTURE = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1'
LEGACY = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postfit-policy-p1'
TRANSCRIPT = (b'CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\n'
              b'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
              b'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\n'
              b'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')


def fixture():
    health_rows = {engine: [dict(case=name, pd_hessian=True, max_abs_gradient=1e-7,
                                 parameters=[0.0, 0.0], gradient=[1e-7, 0.0], free_indices=[1, 2],
                                 optimizer_objective=3.0, recomputed_objective=3.0,
                                 objective_verified=True, penalty_configuration_recorded=True, error=None,
                                 **({'fixed_prior_state': 'none', 'ridge_tau': 'Inf', 'ridge_penalised': False}
                                    if engine == 'R' else {'reconstruction': {'recovered_parameters': [0.0, 0.0], 'recovered_loglik': -3.0, 'parameters_match': True, 'loglik_matches': True, 'error': None}, 'penalty_state': 'none', 'objective_difference': 0.0, 'objective_roundoff_limit': 5.329070518200751e-15, 'hessian': [[1.0, 0.0], [0.0, 1.0]], 'hessian_raw': [[1.0, 0.0], [0.0, 1.0]], 'hessian_asymmetry': 0.0, 'hessian_asymmetry_limit': 1.4210854715202004e-14, 'optimizer': {'optimizer_parameters': [0.0, 0.0], 'optimizer_gradient': [1e-7, 0.0], 'optimizer_max_abs_gradient': 1e-7, 'optimizer_minimum': 3.0, 'optimizer_profile_value': 3.0, 'optimizer_profile_verified': True, 'optimizer_error': None}}))
                            for name in health.CASES] for engine in health.ENGINES}
    actual = {'pass': True, 'three_fit': {'julia_lengths': [0, 5, 6], 'fit_health': health_rows}}
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
    oracle = {'postcoef_three_fit': {'fits': [{'length': n, 'fit_health': copy.deepcopy(row)}
                                             for n, row in zip((0, 5, 6), health_rows['R'])]}}
    return case, report, oracle


def preliminary_health():
    """Bypass only the numerical-admission hold so non-health checks stay live.

    The real classifier never returns a pass while the hold is in force; tests
    of identity and comparison fields use the pre-hold verdict instead.
    """
    return patch.object(live, 'classify_fit_health', health._classify_recorded)


class LiveRefTests(unittest.TestCase):
    def test_isolated_offline_interruption_terminates_checker_group(self):
        import signal
        import time
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-offline-interrupt-') as tmp:
            root = Path(tmp); caller = root / 'caller'; tools = caller / 'tools'; tools.mkdir(parents=True)
            started, escaped = root / 'started', root / 'escaped'
            child = ('from pathlib import Path; import os,time; Path(' + repr(str(started)) +
                     ').write_text(str(os.getpgrp())); time.sleep(1.5); Path(' + repr(str(escaped)) + ').write_text("escaped")')
            checker = 'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",' + repr(child) + '])\ntime.sleep(8)\n'
            def git(*args):
                return subprocess.run(['git', '-C', str(caller), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (tools / 'core070_postcoef_offline_check.py').write_text(checker)
            (caller / 'Project.toml').write_text('name = "fixture"\n')
            contract = caller / 'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json'
            contract.parent.mkdir(parents=True); contract.write_text('{}')
            git('add', 'tools/core070_postcoef_offline_check.py', 'Project.toml', str(contract.relative_to(caller)))
            git('commit', '-qm', 'trusted gate'); gate = git('rev-parse', 'HEAD')
            launcher = tools / 'core070_postcoef_gate_launcher.py'; launcher.write_text("TRUSTED_GATE_COMMIT = '" + gate + "'\n")
            git('add', str(launcher.relative_to(caller))); git('commit', '-qm', 'bound caller')
            original = subprocess.Popen.communicate; interrupted = []; owned = []
            def communicate(proc, *args, **kwargs):
                if not interrupted:
                    interrupted.append(True); owned.append(proc)
                    deadline = time.monotonic() + 5
                    while not started.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(started.exists(), 'fixture child must have started')
                    raise KeyboardInterrupt
                return original(proc, *args, **kwargs)
            try:
                with patch.object(assemble, 'ROOT', caller), patch.object(assemble, 'POSTCOEF_GATE_COMMIT', gate), \
                     patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', hashlib.sha256(launcher.read_bytes()).hexdigest()), \
                     patch.object(subprocess.Popen, 'communicate', communicate):
                    with self.assertRaises(KeyboardInterrupt):
                        assemble.run_postcoef_offline_isolated(root, root, 'a' * 64)
                time.sleep(1.8)
                self.assertFalse(escaped.exists(), 'checker descendant survived interruption')
            finally:
                if owned and owned[0].returncode is None:
                    try: os.killpg(owned[0].pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    original(owned[0])

    def test_isolated_offline_outer_timeout_removes_private_checkout(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-outer-timeout-') as tmp:
            root = Path(tmp); caller = root / 'caller'; tools = caller / 'tools'; tools.mkdir(parents=True)
            record = root / 'checkout-record'
            checker = ('from pathlib import Path\nPath(' + repr(str(record)) +
                       ').write_text(str(Path(__file__).resolve().parents[1]))\nimport time\ntime.sleep(6)\n')
            def git(*args):
                return subprocess.run(['git', '-C', str(caller), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (tools / 'core070_postcoef_offline_check.py').write_text(checker)
            (caller / 'Project.toml').write_text('name = "fixture"\n')
            contract = caller / 'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json'
            contract.parent.mkdir(parents=True); contract.write_text('{}')
            git('add', 'tools/core070_postcoef_offline_check.py', 'Project.toml', str(contract.relative_to(caller)))
            git('commit', '-qm', 'trusted gate'); gate = git('rev-parse', 'HEAD')
            launcher = tools / 'core070_postcoef_gate_launcher.py'
            launcher.write_text("TRUSTED_GATE_COMMIT = '" + gate + "'\n")
            git('add', str(launcher.relative_to(caller))); git('commit', '-qm', 'bound caller')
            with patch.object(assemble, 'ROOT', caller), patch.object(assemble, 'POSTCOEF_GATE_COMMIT', gate), \
                 patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', hashlib.sha256(launcher.read_bytes()).hexdigest()), \
                 patch.object(assemble, 'POSTCOEF_OFFLINE_OUTER_TIMEOUT_SECONDS', 2, create=True):
                with self.assertRaises(subprocess.TimeoutExpired):
                    assemble.run_postcoef_offline_isolated(root, root, 'a' * 64)
            self.assertTrue(record.exists(), 'checker must record its private checkout')
            self.assertFalse(Path(record.read_text()).exists(), 'outer timeout leaked the private checkout')

    def test_isolated_offline_timeout_terminates_checker_descendant(self):
        import shutil
        import time
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-offline-timeout-') as tmp:
            root = Path(tmp); caller = root / 'caller'; tools = caller / 'tools'; tools.mkdir(parents=True)
            started, escaped, checkout_record = root / 'started', root / 'escaped', root / 'checkout-record'
            child = ('from pathlib import Path; import time; Path(' + repr(str(started)) +
                     ').write_text("started"); time.sleep(4); Path(' + repr(str(escaped)) + ').write_text("escaped")')
            checker = ('from pathlib import Path\nPath(' + repr(str(checkout_record)) +
                       ').write_text(str(Path(__file__).resolve().parents[1]))\n'
                       'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",' + repr(child) +
                       '])\ntime.sleep(8)\n')
            def git(*args):
                return subprocess.run(['git', '-C', str(caller), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (tools / 'core070_postcoef_offline_check.py').write_text(checker)
            (caller / 'Project.toml').write_text('name = "fixture"\n')
            contract = caller / 'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json'
            contract.parent.mkdir(parents=True); contract.write_text('{}')
            git('add', 'tools/core070_postcoef_offline_check.py', 'Project.toml', str(contract.relative_to(caller)))
            git('commit', '-qm', 'trusted gate'); gate = git('rev-parse', 'HEAD')
            launcher = tools / 'core070_postcoef_gate_launcher.py'
            launcher.write_text("TRUSTED_GATE_COMMIT = '" + gate + "'\n")
            git('add', str(launcher.relative_to(caller))); git('commit', '-qm', 'bound caller')
            with patch.object(assemble, 'ROOT', caller), patch.object(assemble, 'POSTCOEF_GATE_COMMIT', gate), \
                 patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', hashlib.sha256(launcher.read_bytes()).hexdigest()), \
                 patch.object(assemble, 'POSTCOEF_OFFLINE_TIMEOUT_SECONDS', 2, create=True):
                result = assemble.run_postcoef_offline_isolated(root, root, 'a' * 64)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(started.exists(), 'fixture child must have started')
            self.assertTrue(checkout_record.exists(), 'checker must record its private checkout')
            checkout = Path(checkout_record.read_text())
            self.assertTrue(checkout.parent.name.startswith('postcoef-offline-gate-'))
            self.addCleanup(shutil.rmtree, checkout.parent, ignore_errors=True)
            self.assertFalse(checkout.exists(), 'private checkout survived checker timeout')
            time.sleep(2.5)
            self.assertFalse(escaped.exists(), 'checker descendant survived the timeout')

    def test_snapshot_rejects_changed_missing_and_linked_selected_artifacts(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-state-snapshot-') as tmp:
            root = Path(tmp); state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            names = ('receipt.json', 'r-oracle.json', 'results.tsv', 'oracle-verify-preload.log',
                     'julia-control-always_empty.json', 'runner-snapshots/julia-launcher')
            for name in names:
                path = state / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'selected evidence')
            destination = root / 'snapshot'; destination.mkdir()
            self.assertEqual(assemble.snapshot_postcoef_state(root, state, destination), git('rev-parse', 'HEAD'))
            with patch.object(assemble.os, 'supports_dir_fd', set()), tempfile.TemporaryDirectory() as out:
                with self.assertRaisesRegex(NotImplementedError, 'directory traversal'):
                    assemble.snapshot_postcoef_state(root, state, Path(out))
            for name in names:
                with self.subTest(name=name), tempfile.TemporaryDirectory() as out:
                    path = state / name; original = path.read_bytes(); path.write_bytes(original + b'changed')
                    try:
                        with self.assertRaisesRegex(ValueError, 'differs'):
                            assemble.snapshot_postcoef_state(root, state, Path(out))
                    finally:
                        path.write_bytes(original)
            path = state / 'r-oracle.json'; original = path.read_bytes(); path.unlink()
            with tempfile.TemporaryDirectory() as out, self.assertRaisesRegex(ValueError, 'missing'):
                assemble.snapshot_postcoef_state(root, state, Path(out))
            outside = root / 'outside'; outside.write_bytes(original); path.symlink_to(outside)
            with tempfile.TemporaryDirectory() as out, self.assertRaises(ValueError):
                assemble.snapshot_postcoef_state(root, state, Path(out))

    def test_snapshot_closes_registered_directories_on_interruption(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-fd-cleanup-') as tmp:
            root = Path(tmp); state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            (state / 'receipt.json').write_text('{}')
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'selected evidence')
            opened = []; real_open = os.open
            def tracked_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                if path == root or kwargs.get('dir_fd') is not None:
                    opened.append(fd)
                return fd
            unrelated = real_open(os.devnull, os.O_RDONLY)
            real_close = os.close; close_errors = []
            def tracked_close(fd):
                real_close(fd)
                if fd in opened and not close_errors:
                    close_errors.append(fd)
                    raise OSError('simulated close error after close')
            try:
                with tempfile.TemporaryDirectory() as out, \
                     patch.object(os, 'supports_dir_fd', os.supports_dir_fd | {tracked_open}), \
                     patch.object(os, 'open', new=tracked_open), \
                     patch.object(os, 'close', new=tracked_close), \
                     patch.object(os, 'fwalk', side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        assemble.snapshot_postcoef_state(root, state, Path(out))
                self.assertGreater(len(opened), 1)
                self.assertEqual(len(close_errors), 1)
                for fd in opened:
                    with self.assertRaises(OSError): os.fstat(fd)
                os.fstat(unrelated)
            finally:
                os.close(unrelated)

    def test_snapshot_reports_close_error_inside_callers_exception_handler(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-fd-outer-exception-') as tmp:
            root = Path(tmp); state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            (state / 'receipt.json').write_text('{}')
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'selected evidence')
            opened = []; real_open = os.open; real_close = os.close
            def tracked_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                if path == root: opened.append(fd)
                return fd
            def tracked_close(fd):
                real_close(fd)
                if opened and fd == opened[0]:
                    raise OSError('simulated root close error')
            with tempfile.TemporaryDirectory() as out, \
                 patch.object(os, 'supports_dir_fd', os.supports_dir_fd | {tracked_open}), \
                 patch.object(os, 'open', new=tracked_open), \
                 patch.object(os, 'close', new=tracked_close):
                try:
                    raise RuntimeError('caller is handling this')
                except RuntimeError:
                    with self.assertRaisesRegex(OSError, 'simulated root close error'):
                        assemble.snapshot_postcoef_state(root, state, Path(out))

    def test_snapshot_refuses_a_symlinked_capture_ancestor(self):
        import shutil
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-ancestor-link-') as tmp:
            root = Path(tmp) / 'repo'; root.mkdir()
            state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            (state / 'receipt.json').write_text('{}')
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'selected evidence')
            ancestor = root / 'docs/dev-log/core070/true-parity-latest/receipts/postfit'
            external = Path(tmp) / 'relocated'; shutil.move(str(ancestor), str(external))
            ancestor.symlink_to(external, target_is_directory=True)
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaises((ValueError, OSError)):
                    assemble.snapshot_postcoef_state(root, state, Path(out))

    def test_unsupported_snapshot_primitives_fail_closed(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-unsupported-snapshot-') as tmp:
            root = Path(tmp); (root / assemble.POSTCOEF_CAPTURE_PREFIX).mkdir(parents=True)
            case, _, _ = fixture()
            case['postcoef_capture'] = {'receipt': assemble.POSTCOEF_CAPTURE_PREFIX + '/receipt.json',
                                        'sha256': 'a' * 64}
            with patch.object(assemble, 'snapshot_postcoef_state', side_effect=NotImplementedError('unsupported')):
                self.assertEqual(assemble.postcoef_capture_problem(case, root, {}),
                                 'POST-COEF safe capture snapshot unavailable')

    def test_assembler_uses_same_snapshot_for_check_and_comparison(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-state-race-') as tmp:
            root = Path(tmp); state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            case, report, oracle = fixture()
            receipt = b'{"fixture":true}'
            (state / 'receipt.json').write_bytes(receipt)
            (state / 'verify.txt').write_bytes(TRANSCRIPT)
            (state / 'julia-results.json').write_text(json.dumps(report))
            (state / 'r-oracle.json').write_text(json.dumps(oracle))
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'synthetic state')
            case['postcoef_capture'] = {'receipt': assemble.POSTCOEF_CAPTURE_PREFIX + '/receipt.json',
                                        'sha256': hashlib.sha256(receipt).hexdigest()}
            case['batch_verifier']['log_sha256'] = hashlib.sha256(TRANSCRIPT).hexdigest()
            original = (state / 'r-oracle.json').read_bytes()
            seen = []
            def checker(repo, checked_state, digest):
                checked_state = Path(checked_state); seen.append(checked_state)
                (state / 'r-oracle.json').write_text('{"changed":"after offline check"}')
                self.assertNotEqual(checked_state.resolve(), state.resolve())
                self.assertEqual((checked_state / 'r-oracle.json').read_bytes(), original)
                return subprocess.CompletedProcess([], 0, 'ARTIFACT_INTEGRITY_PASS_RUNTIME_NOT_VERIFIED\n', '')
            with patch.object(assemble, 'run_postcoef_offline_isolated', side_effect=checker), \
                 patch.dict(os.environ, {'GLLVM_POSTCOEF_RUNTIME_ROOT': ''}):
                problem = assemble.postcoef_capture_problem(case, root, {})
            self.assertEqual(len(seen), 1)
            self.assertEqual(problem, 'POST-COEF runtime and oracle build not independently verified')

    def test_assembler_rejects_selected_commit_change_after_snapshot(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-head-race-') as tmp:
            root = Path(tmp); state = root / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            case, report, oracle = fixture(); receipt = b'{"fixture":true}'
            (state / 'receipt.json').write_bytes(receipt); (state / 'verify.txt').write_bytes(TRANSCRIPT)
            (state / 'julia-results.json').write_text(json.dumps(report)); (state / 'r-oracle.json').write_text(json.dumps(oracle))
            git('add', assemble.POSTCOEF_CAPTURE_PREFIX); git('commit', '-qm', 'synthetic state')
            case['postcoef_capture'] = {'receipt': assemble.POSTCOEF_CAPTURE_PREFIX + '/receipt.json',
                                        'sha256': hashlib.sha256(receipt).hexdigest()}
            case['batch_verifier']['log_sha256'] = hashlib.sha256(TRANSCRIPT).hexdigest()
            def checker(repo, checked_state, digest):
                (root / 'new-head').write_text('changed')
                git('add', 'new-head'); git('commit', '-qm', 'change selected commit')
                return subprocess.CompletedProcess([], 0, 'ARTIFACT_INTEGRITY_PASS_RUNTIME_NOT_VERIFIED\n', '')
            with patch.object(assemble, 'run_postcoef_offline_isolated', side_effect=checker), \
                 patch.dict(os.environ, {'GLLVM_POSTCOEF_RUNTIME_ROOT': str(root)}):
                self.assertEqual(assemble.postcoef_capture_problem(case, root, {}),
                                 'POST-COEF selected commit changed during verification')

    def test_isolated_checker_disables_site_initialization(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-no-site-') as tmp:
            root = Path(tmp); caller = root / 'caller'; tools = caller / 'tools'; tools.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(caller), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            checker = tools / 'core070_postcoef_offline_check.py'
            checker.write_text('import sys\nassert sys.flags.no_site, "site initialization enabled"\nprint("NO_SITE_CHECKER_PASS")\n')
            (caller / 'Project.toml').write_text('name = "fixture"\n')
            contract = caller / 'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json'
            contract.parent.mkdir(parents=True); contract.write_text('{}')
            git('add', 'tools/core070_postcoef_offline_check.py', 'Project.toml', str(contract.relative_to(caller)))
            git('commit', '-qm', 'gate'); gate = git('rev-parse', 'HEAD')
            launcher = tools / 'core070_postcoef_gate_launcher.py'
            launcher.write_text("TRUSTED_GATE_COMMIT = '" + gate + "'\n")
            git('add', str(launcher.relative_to(caller))); git('commit', '-qm', 'caller')
            original_popen = subprocess.Popen
            with patch.object(assemble, 'ROOT', caller), patch.object(assemble, 'POSTCOEF_GATE_COMMIT', gate), \
                 patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', hashlib.sha256(launcher.read_bytes()).hexdigest()), \
                 patch.object(subprocess, 'Popen', wraps=original_popen) as popen:
                result = assemble.run_postcoef_offline_isolated(root, root, 'a' * 64)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('NO_SITE_CHECKER_PASS', result.stdout)
            bootstrap = popen.call_args_list[0].args[0]
            self.assertIn('-S', bootstrap[:bootstrap.index('-c')])

    def test_isolated_offline_gate_pin_must_match_launcher(self):
        import true_parity_assemble as assemble
        with patch.object(assemble, 'POSTCOEF_GATE_COMMIT', '0' * 40):
            with self.assertRaisesRegex(ValueError, 'gate pin'):
                assemble.run_postcoef_offline_isolated(assemble.ROOT, assemble.ROOT, 'a' * 64)

    def test_assembler_does_not_execute_substituted_offline_checker(self):
        import true_parity_assemble as assemble
        with tempfile.TemporaryDirectory(prefix='postcoef-forged-offline-') as tmp:
            root = Path(tmp); caller = root / 'caller'; evidence = root / 'evidence'
            tools = caller / 'tools'; tools.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(caller), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            checker = tools / 'core070_postcoef_offline_check.py'
            checker.write_text('raise SystemExit("trusted offline rejected")\n')
            (caller / 'Project.toml').write_text('name = "fixture"\n')
            contract = caller / 'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json'
            contract.parent.mkdir(parents=True); contract.write_text('{}')
            git('add', 'tools/core070_postcoef_offline_check.py', 'Project.toml', str(contract.relative_to(caller)))
            git('commit', '-qm', 'trusted gate'); gate = git('rev-parse', 'HEAD')
            launcher = tools / 'core070_postcoef_gate_launcher.py'
            launcher.write_text("TRUSTED_GATE_COMMIT = '" + gate + "'\n")
            git('add', str(launcher.relative_to(caller))); git('commit', '-qm', 'bound caller')
            state = evidence / assemble.POSTCOEF_CAPTURE_PREFIX; state.mkdir(parents=True)
            (state / 'verify.txt').write_bytes(TRANSCRIPT)
            (state / 'receipt.json').write_bytes(b'synthetic receipt')
            def evidence_git(*args):
                return subprocess.run(['git', '-C', str(evidence), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            evidence_git('init', '-q'); evidence_git('config', 'user.name', 'Fixture')
            evidence_git('config', 'user.email', 'fixture@example.invalid')
            evidence_git('add', assemble.POSTCOEF_CAPTURE_PREFIX); evidence_git('commit', '-qm', 'synthetic evidence')
            marker = root / 'forged-executed'
            checker.write_text(
                'from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("executed")\n'
                'print("ARTIFACT_INTEGRITY_PASS_RUNTIME_NOT_VERIFIED")\n')
            case, _, _ = fixture()
            case['postcoef_capture'] = {'receipt': assemble.POSTCOEF_CAPTURE_PREFIX + '/receipt.json',
                                        'sha256': 'a' * 64}
            case['batch_verifier']['log_sha256'] = hashlib.sha256(TRANSCRIPT).hexdigest()
            launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()
            with patch.object(assemble, 'ROOT', caller), patch.object(assemble, 'POSTCOEF_GATE_COMMIT', gate), \
                 patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', launcher_hash):
                problem = assemble.postcoef_capture_problem(case, evidence, {})
            self.assertIn('trusted offline rejected', problem)
            self.assertFalse(marker.exists(), 'substituted checker must not run before rejection')

    def test_staged_capture_includes_preload_for_source_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            calls = []
            def selected(name):
                calls.append(name)
                return b'CORE070_ORACLE_VERIFY_PASS\n' if name == 'oracle-verify-preload.log' else b'fixture\n'
            live.stage_capture_files(state, live.FILES, selected)
            # The same staged directory is passed to the source verifier.
            source_verifier = lambda path: (path / 'oracle-verify-preload.log').read_bytes()
            self.assertEqual(source_verifier(state), b'CORE070_ORACLE_VERIFY_PASS\n')
            self.assertIn('oracle-verify-preload.log', calls)
            self.assertEqual(len(calls), len(set(calls)))

    def test_live_comparison_rejects_failed_or_unbound_fit_health(self):
        import core070_postcoef_fit_health as health
        case, report, oracle = fixture()
        rows = {engine: [dict(case=name, pd_hessian=True, max_abs_gradient=1e-7,
                              parameters=[0.0, 0.0], gradient=[1e-7, 0.0], free_indices=[1, 2],
                                 optimizer_objective=3.0, recomputed_objective=3.0,
                                 objective_verified=True, penalty_configuration_recorded=True, error=None,
                                 **({'fixed_prior_state': 'none', 'ridge_tau': 'Inf', 'ridge_penalised': False}
                                    if engine == 'R' else {'reconstruction': {'recovered_parameters': [0.0, 0.0], 'recovered_loglik': -3.0, 'parameters_match': True, 'loglik_matches': True, 'error': None}, 'penalty_state': 'none', 'objective_difference': 0.0, 'objective_roundoff_limit': 5.329070518200751e-15, 'hessian': [[1.0, 0.0], [0.0, 1.0]], 'hessian_raw': [[1.0, 0.0], [0.0, 1.0]], 'hessian_asymmetry': 0.0, 'hessian_asymmetry_limit': 1.4210854715202004e-14, 'optimizer': {'optimizer_parameters': [0.0, 0.0], 'optimizer_gradient': [1e-7, 0.0], 'optimizer_max_abs_gradient': 1e-7, 'optimizer_minimum': 3.0, 'optimizer_profile_value': 3.0, 'optimizer_profile_verified': True, 'optimizer_error': None}}))
                         for name in health.CASES] for engine in health.ENGINES}
        report['cases'][live.CID]['three_fit']['fit_health'] = rows
        for fit, row in zip(oracle['postcoef_three_fit']['fits'], rows['R']):
            fit['fit_health'] = copy.deepcopy(row)
        # The real classifier holds every record, so the nominal comparison rejects.
        with self.assertRaisesRegex(ValueError, 'fit health failed or is incomplete'):
            live.check_comparison(case, report, oracle)
        with preliminary_health():
            live.check_comparison(case, report, oracle)
            bad_case, bad_report, bad_oracle = copy.deepcopy((case, report, oracle))
            bad_report['cases'][live.CID]['three_fit']['fit_health']['Julia'][0]['max_abs_gradient'] = 1e-5
            with self.assertRaises(ValueError):
                live.check_comparison(bad_case, bad_report, bad_oracle)
            bad_case, bad_report, bad_oracle = copy.deepcopy((case, report, oracle))
            bad_oracle['postcoef_three_fit']['fits'][0]['fit_health']['pd_hessian'] = False
            with self.assertRaises(ValueError):
                live.check_comparison(bad_case, bad_report, bad_oracle)

    def test_live_comparison_rejects_float64_cholesky_counterexample(self):
        from test_postcoef_fit_health import with_indefinite_julia_hessian
        case, report, oracle = fixture()
        with_indefinite_julia_hessian(report['cases'][live.CID]['three_fit']['fit_health'])
        # Before the hold this indefinite matrix reached a passing comparison.
        with preliminary_health():
            self.assertIsNone(live.check_comparison(case, report, oracle))
        with self.assertRaisesRegex(ValueError, 'fit health failed or is incomplete'):
            live.check_comparison(case, report, oracle)

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
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (state / 'receipt.json').write_text('{}')
            git('add', str((state / 'receipt.json').relative_to(root))); git('commit', '-qm', 'missing transcript')
            with patch.object(assemble, 'run_postcoef_offline_isolated', side_effect=AssertionError('verifier must not run')):
                self.assertIn('transcript missing', assemble.postcoef_capture_problem(case, root, {}))
                forged = b'forged\n'; (state / 'verify.txt').write_bytes(forged)
                git('add', str((state / 'verify.txt').relative_to(root))); git('commit', '-qm', 'forged transcript')
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
        with self.assertRaisesRegex(ValueError, 'fit health'):
            live.check_comparison(case, report, oracle)
        with preliminary_health():
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
        with self.assertRaisesRegex(ValueError, 'fit health'):
            live.check_comparison(case, report, oracle)
        held = preliminary_health(); held.start(); self.addCleanup(held.stop)
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

    def test_capture_sources_must_match_trusted_gate_bytes(self):
        trusted = {path: ('trusted ' + path).encode() for path in live.TRUSTED_CAPTURE_FILES}
        captured = dict(trusted)
        self.assertIsNone(live.check_trusted_capture_sources(captured.__getitem__, trusted.__getitem__))
        for path in live.TRUSTED_CAPTURE_FILES:
            changed = dict(captured); changed[path] += b' tampered'
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'trusted gate'):
                live.check_trusted_capture_sources(changed.__getitem__, trusted.__getitem__)

    def test_committed_transitive_dependency_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='postcoef-committed-dep-') as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', tmp, *args], check=True, capture_output=True).stdout
            git('init', '-q'); git('config', 'user.email', 'test@example.invalid'); git('config', 'user.name', 'test')
            for rel in live.TRUSTED_CAPTURE_FILES:
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('trusted ' + rel)
            git('add', *live.TRUSTED_CAPTURE_FILES); git('commit', '-qm', 'trusted fixture')
            trusted = git('rev-parse', 'HEAD').decode().strip()
            path = root / 'tools/core070_json_semantics.py'; path.write_text('modified dependency')
            git('add', 'tools/core070_json_semantics.py'); git('commit', '-qm', 'modified dependency')
            modified = git('rev-parse', 'HEAD').decode().strip()
            read_trusted = lambda rel: live.blob(root, trusted, rel)
            self.assertIsNone(live.check_trusted_capture_sources(read_trusted, read_trusted))
            with self.assertRaisesRegex(ValueError, 'core070_json_semantics.py'):
                live.check_trusted_capture_sources(lambda rel: live.blob(root, modified, rel), read_trusted)

    def test_added_shadow_module_rejected_by_capture_trust(self):
        with tempfile.TemporaryDirectory(prefix='postcoef-shadow-') as tmp:
            root = Path(tmp) / 'capture'; root.mkdir()
            trusted = Path(tmp) / 'trusted'
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout
            git('init', '-q'); git('config', 'user.email', 'test@example.invalid'); git('config', 'user.name', 'test')
            for rel in live.TRUSTED_CAPTURE_FILES:
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('trusted ' + rel)
            (root / 'tools/core070_postcoef_gate_launcher.py').write_text('trusted launcher')
            git('add', *live.TRUSTED_CAPTURE_FILES, 'tools/core070_postcoef_gate_launcher.py'); git('commit', '-qm', 'trusted fixture')
            subprocess.run(['git', 'clone', '-q', '--no-local', str(root), str(trusted)], check=True)
            source = git('rev-parse', 'HEAD').decode().strip()
            self.assertIsNone(live.check_capture_trust(root, source, trusted))
            (root / 'tools/core070_postcoef_gate_launcher.py').write_text('separately rebound launcher')
            git('add', 'tools/core070_postcoef_gate_launcher.py'); git('commit', '-qm', 'rebind launcher')
            rebound = git('rev-parse', 'HEAD').decode().strip()
            self.assertIsNone(live.check_capture_trust(root, rebound, trusted))
            (root / 'tools/tomllib.py').write_text('raise SystemExit(0)')
            git('add', 'tools/tomllib.py'); git('commit', '-qm', 'shadow module')
            changed = git('rev-parse', 'HEAD').decode().strip()
            with self.assertRaisesRegex(ValueError, 'tools tree'):
                live.check_capture_trust(root, changed, trusted)

    def test_case_must_reference_exact_capture_build(self):
        case, _, _ = fixture()
        pin = {'reference_commit': PIN, 'archive_sha256': 'a' * 64,
               'namespace_sha256': 'b' * 64, 'source_tree_sha256': '86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db',
               'installed_tree_sha256': 'd' * 64, 'marker_sha256': 'e' * 64,
               'version': '0.7.1'}
        build = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'source_tree_sha256',
                                      'installed_tree_sha256', 'marker_sha256')}
        build['gllvmTMB_version'] = pin['version']
        build_bytes = json.dumps(build).encode()
        source = {k: pin[k] for k in ('reference_commit', 'archive_sha256', 'namespace_sha256', 'source_tree_sha256')}
        record = {'postcoef_capture_identity': {'source_tree': 'df63c52066625049087a321bb9b010f8d834ddcf'}, 'source_pin': pin,
                  'oracle_build_sha256': hashlib.sha256(build_bytes).hexdigest()}
        case.update(glvmodels_src_tree='df63c52066625049087a321bb9b010f8d834ddcf', gllvmtmb_version='0.7.1',
                    oracle_build_receipt=f'{ROOT}/build-totoro-20261104.json', oracle_source_receipt=f'{ROOT}/source.json')
        files = {case['oracle_build_receipt']: build_bytes, case['oracle_source_receipt']: json.dumps(source).encode()}
        for row in case['harness_fields']['three_fit']['fit_health']['Julia']:
            row['penalty_provenance'] = dict(source_tree=record['postcoef_capture_identity']['source_tree'],
                case=row['case'], route='gaussian_profile_default', rank=1, integration='exact', prior='none')
        for row in case['harness_fields']['three_fit']['fit_health']['R']:
            row['prior_provenance'] = dict(pin='P1', reference_commit=PIN,
                source_tree_sha256=pin['source_tree_sha256'], version=pin['version'],
                oracle_build_sha256=record['oracle_build_sha256'], call_verified=True, call_text='gllvmTMB(formula = f, data = df_long, trait = "trait", unit = "site", family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))')
        self.assertIsNone(live.check_case_source_binding(case, record, files.__getitem__))
        bad = copy.deepcopy(case)
        bad['harness_fields']['three_fit']['fit_health']['Julia'][0]['penalty_provenance']['source_tree'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'Julia penalty provenance'):
            live.check_case_source_binding(bad, record, files.__getitem__)
        bad = copy.deepcopy(case)
        bad['harness_fields']['three_fit']['fit_health']['R'][0]['prior_provenance']['oracle_build_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'prior provenance'):
            live.check_case_source_binding(bad, record, files.__getitem__)
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
    def test_staged_preload_reaches_source_verifier_process(self):
        verifier = self.root / live.SOURCE_FILES[0]
        with verifier.open('a') as out:
            out.write("\nstate = Path(sys.argv[sys.argv.index('--state') + 1])\n")
            out.write("assert (state / 'oracle-verify-preload.log').read_text() == 'CORE070_ORACLE_VERIFY_PASS\\n'\n")
        self.git('add', live.SOURCE_FILES[0]); self.git('commit', '-qm', 'verify preload')
        source = self.git('rev-parse', 'HEAD')
        live.stage_capture_files(self.state, live.FILES,
            lambda name: b'CORE070_ORACLE_VERIFY_PASS\n' if name == 'oracle-verify-preload.log' else b'fixture\n')
        output = live.run_source_selftest(self.root, source, self.state, self.runtime)
        self.assertTrue(all(marker in output.splitlines() for marker in live.SELFTEST_MARKERS))
        (self.state / 'oracle-verify-preload.log').unlink()
        with self.assertRaisesRegex(ValueError, 'self-test rejected'):
            live.run_source_selftest(self.root, source, self.state, self.runtime)

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

    def test_changed_health_collector_is_rejected(self):
        names = ['tools/core070_postcoef_fit_health.py', 'tools/core070_postcoef_fit_health.R',
                 'tools/core070_postcoef_fit_health.jl']
        for name in names:
            (self.root / name).write_text('recorded collector\n')
        self.git('add', *names); self.git('commit', '-qm', 'collector fixture')
        source = self.git('rev-parse', 'HEAD')
        live.run_source_selftest(self.root, source, self.state, self.runtime)
        for name in names:
            path = self.root / name; path.write_text('tampered collector\n')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'differs'):
                live.run_source_selftest(self.root, source, self.state, self.runtime)
            path.write_text('recorded collector\n')
    def test_missing_selftest_marker_is_rejected(self):
        verifier=self.root/live.SOURCE_FILES[0]; verifier.write_text('print("incomplete")\n')
        self.git('add',live.SOURCE_FILES[0]); self.git('commit','-qm','incomplete verifier')
        with self.assertRaisesRegex(ValueError,'self-test rejected'):
            live.run_source_selftest(self.root,self.git('rev-parse','HEAD'),self.state,self.runtime)



if __name__ == '__main__':
    unittest.main()
