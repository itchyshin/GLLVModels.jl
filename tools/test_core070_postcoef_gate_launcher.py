#!/usr/bin/env python3
"""Controls for a source-only POST-COEF gate launcher."""
import importlib.util
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).with_name('core070_postcoef_gate_launcher.py')
EXPECTED_GATE = 'UNBOUND'

class GateHistoryTests(unittest.TestCase):
    def test_capture_only_gate_fails_closed(self):
        if EXPECTED_GATE != 'UNBOUND':
            self.skipTest('bound caller; capture-only control is not applicable')
        repo = LAUNCHER.parent.parent
        source = LAUNCHER.read_bytes()
        fd = os.open(repo, os.O_RDONLY)
        try:
            bootstrap = ("__file__ = " + repr(str(LAUNCHER)) + "; __gate_root_fd__ = " + str(fd) +
                         "; exec(compile(bytes.fromhex(" + repr(source.hex()) + "), __file__, 'exec'))")
            for mode in ('ancestry', 'identity', 'live'):
                args = ([str(repo), 'source', 'head'] if mode == 'ancestry' else
                        ['--repo', str(repo), '--head', 'head', '--runtime-root', str(repo)])
                result = subprocess.run([sys.executable, '-B', '-I', '-c', bootstrap, mode, *args],
                                        pass_fds=(fd,), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, mode)
                self.assertIn('trusted gate commit is not configured', result.stderr)
        finally:
            os.close(fd)

    def test_bound_gate_reachable_from_candidate_history(self):
        if EXPECTED_GATE == 'UNBOUND':
            self.skipTest('capture-only source has no bound gate')
        repo = LAUNCHER.parent.parent
        result = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor',
                                 EXPECTED_GATE, 'HEAD'], capture_output=True)
        self.assertEqual(result.returncode, 0, 'trusted gate commit must be an ancestor of candidate HEAD')

def launcher_with_pin(pin):
    source = LAUNCHER.read_text()
    result, count = re.subn(r"(?m)^TRUSTED_GATE_COMMIT = '[^']*'$", f"TRUSTED_GATE_COMMIT = '{pin}'", source)
    if count != 1:
        raise ValueError('expected exactly one gate pin')
    return result

class GateLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'gate'; self.root.mkdir()
        self.tools = self.root / 'tools'; self.tools.mkdir()
        self.git('init', '-q'); self.git('config', 'user.name', 'fixture'); self.git('config', 'user.email', 'fixture@example.invalid')
        self.helper = self.tools / 'gate_helper.py'
        self.helper.write_text("VALUE = 'rejected'\n".ljust(512, ' '))
        for name in ('core070_postcoef_live_ref.py', 'core070_postcoef_ref_compatible.py'):
            (self.tools / name).write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nassert not (Path(__file__).resolve().parents[1] / '.git/objects/info/alternates').exists()\nimport gate_helper\nprint(gate_helper.VALUE)\n")
        self.git('add', *['tools/' + p.name for p in self.tools.iterdir()]); self.git('commit', '-qm', 'gate fixture')
        self.head = self.git('rev-parse', 'HEAD')
        launcher = launcher_with_pin(self.head)
        (self.tools / LAUNCHER.name).write_text(launcher)
        self.git('add', 'tools/' + LAUNCHER.name); self.git('commit', '-qm', 'launcher fixture')
    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True, text=True).stdout.strip()
    def launch(self, entry, env=None):
        args = [str(self.root), 'source', 'head'] if entry == 'ancestry' else ['--repo', str(self.root), '--head', 'head', '--runtime-root', str(self.root)]
        fd = os.open(self.root, os.O_RDONLY)
        try:
            source = (self.tools / LAUNCHER.name).read_bytes()
            bootstrap = ("__file__ = " + repr(str(self.tools / LAUNCHER.name)) +
                         "; __gate_root_fd__ = " + str(fd) +
                         "; exec(compile(bytes.fromhex(" + repr(source.hex()) + "), __file__, 'exec'))")
            return subprocess.run([sys.executable, '-B', '-I', '-c', bootstrap, entry, *args],
                                  pass_fds=(fd,), capture_output=True, text=True, env=env)
        finally:
            os.close(fd)
    def test_gate_pin_matches_reviewed_commit(self):
        self.assertIn(f"TRUSTED_GATE_COMMIT = '{EXPECTED_GATE}'", LAUNCHER.read_text())

    def test_both_entries_ignore_forged_source_checkout_bytecode(self):
        info = self.helper.stat(); fake = self.root.parent / 'fake.py'
        fake.write_text("VALUE = 'accepted'\n".ljust(512, ' ')); os.utime(fake, ns=(info.st_atime_ns, info.st_mtime_ns))
        cache = Path(importlib.util.cache_from_source(str(self.helper))); cache.parent.mkdir()
        py_compile.compile(str(fake), cfile=str(cache), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        before = cache.read_bytes(); env = dict(os.environ, GIT_DIR='/invalid', PYTHONPYCACHEPREFIX=str(self.root.parent/'redirected'))
        for entry in ('live', 'identity', 'ancestry'):
            p = self.launch(entry, env)
            self.assertEqual(p.returncode, 0, p.stderr); self.assertEqual(p.stdout.strip(), 'rejected')
        self.assertEqual(cache.read_bytes(), before); self.assertFalse((self.root.parent/'redirected').exists())
    def test_bytecode_in_untrusted_descendant_is_ignored(self):
        bytecode = self.tools / 'bad.pyc'; bytecode.write_bytes(b'forged')
        self.git('add', 'tools/bad.pyc'); self.git('commit', '-qm', 'untrusted descendant')
        p = self.launch('live')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'rejected')

    def test_committed_helper_descendant_is_not_used(self):
        self.helper.write_text("VALUE = 'accepted'\n")
        self.git('add', 'tools/gate_helper.py'); self.git('commit', '-qm', 'untrusted descendant')
        p = self.launch('live')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'rejected')

    def test_modified_helper_is_not_used(self):
        self.helper.write_text("VALUE = 'accepted'\n")
        p = self.launch('live')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'rejected')

    def test_trusted_gate_missing_transition_helper_is_rejected(self):
        source_tools = Path(__file__).parent
        names = ('core070_postcoef_ref_compatible.py', 'core070_postfit_p1_receipts.py',
                 'core070_json_semantics.py', 'core070_oracle_pins.toml')
        for name in names:
            (self.tools / name).write_bytes((source_tools / name).read_bytes())
        self.git('add', *['tools/' + name for name in names]); self.git('commit', '-qm', 'missing helper fixture')
        trusted = self.git('rev-parse', 'HEAD')
        (self.tools / LAUNCHER.name).write_text(launcher_with_pin(trusted))
        p = self.launch('ancestry')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('core070_postcoef_transition', p.stderr)

    def test_real_ancestry_entrypoint_accepts_same_commit_and_rejects_code_change(self):
        source_tools = Path(__file__).parent
        names = ('core070_postcoef_ref_compatible.py', 'core070_postfit_p1_receipts.py',
                 'core070_json_semantics.py', 'core070_postcoef_transition.py', 'core070_oracle_pins.toml')
        for name in names:
            (self.tools / name).write_bytes((source_tools / name).read_bytes())
        self.git('add', *['tools/' + name for name in names]); self.git('commit', '-qm', 'real gate')
        trusted = self.git('rev-parse', 'HEAD')
        launcher = launcher_with_pin(trusted)
        (self.tools / LAUNCHER.name).write_text(launcher)
        fd = os.open(self.root, os.O_RDONLY); self.addCleanup(os.close, fd)
        source = (self.tools / LAUNCHER.name).read_bytes()
        bootstrap = ("__file__ = " + repr(str(self.tools / LAUNCHER.name)) + "; __gate_root_fd__ = " + str(fd) + "; exec(compile(bytes.fromhex(" + repr(source.hex()) + "), __file__, 'exec'))")
        command = [sys.executable, '-B', '-I', '-c', bootstrap, 'ancestry', str(self.root), trusted]
        good = subprocess.run([*command, trusted], pass_fds=(fd,), capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(good.stdout.strip(), 'POSTCOEF_REF_COMPATIBLE')
        transition = self.tools / 'core070_postcoef_transition.py'
        transition.write_text('raise RuntimeError("forged helper")\n')
        unchanged = subprocess.run([*command, trusted], pass_fds=(fd,), capture_output=True, text=True)
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        transition.unlink()
        missing = subprocess.run([*command, trusted], pass_fds=(fd,), capture_output=True, text=True)
        self.assertEqual(missing.returncode, 0, missing.stderr)
        transition.write_bytes((source_tools / 'core070_postcoef_transition.py').read_bytes())
        helper = self.tools / 'core070_postfit_p1_receipts.py'; info = helper.stat()
        fake = self.root.parent / 'fake-real-helper.py'
        fake.write_text('def postcoef_ref_compatible(*args): return True\n'.ljust(info.st_size, ' '))
        os.utime(fake, ns=(info.st_atime_ns, info.st_mtime_ns))
        cache = Path(importlib.util.cache_from_source(str(helper))); cache.parent.mkdir(exist_ok=True)
        py_compile.compile(str(fake), cfile=str(cache), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        before = cache.read_bytes()
        (self.tools / 'gate_helper.py').write_text("VALUE = 'accepted'\n")
        self.git('add', 'tools/gate_helper.py'); self.git('commit', '-qm', 'disallowed code change')
        bad = subprocess.run([*command, self.git('rev-parse', 'HEAD')], pass_fds=(fd,), capture_output=True, text=True)
        self.assertEqual(cache.read_bytes(), before)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(bad.stdout.strip(), 'POSTCOEF_REF_INCOMPATIBLE')

    def test_unapproved_gate_pin_fails_closed(self):
        (self.tools / LAUNCHER.name).write_text(launcher_with_pin('UNAPPROVED'))
        p = self.launch('live')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('not configured', p.stderr)

    def test_real_ancestry_entry_accepts_only_approved_preload_path(self):
        source_tools = Path(__file__).parent
        names = ('core070_postcoef_ref_compatible.py', 'core070_postfit_p1_receipts.py',
                 'core070_json_semantics.py', 'core070_postcoef_transition.py', 'core070_oracle_pins.toml')
        for name in names:
            (self.tools / name).write_bytes((source_tools / name).read_bytes())
        self.git('add', *['tools/' + name for name in names]); self.git('commit', '-qm', 'real gate')
        trusted = self.git('rev-parse', 'HEAD')
        (self.tools / LAUNCHER.name).write_text(launcher_with_pin(trusted))
        fd = os.open(self.root, os.O_RDONLY); self.addCleanup(os.close, fd)
        source = (self.tools / LAUNCHER.name).read_bytes()
        bootstrap = ("__file__ = " + repr(str(self.tools / LAUNCHER.name)) +
                     "; __gate_root_fd__ = " + str(fd) +
                     "; exec(compile(bytes.fromhex(" + repr(source.hex()) + "), __file__, 'exec'))")
        command = [sys.executable, '-B', '-I', '-c', bootstrap, 'ancestry', str(self.root), trusted]
        rel = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1/oracle-verify-preload.log'
        path = self.root / rel; path.parent.mkdir(parents=True)
        path.write_text('CORE070_ORACLE_VERIFY_PASS\n')
        self.git('add', rel); self.git('commit', '-qm', 'approved evidence')
        good = subprocess.run([*command, self.git('rev-parse', 'HEAD')], pass_fds=(fd,), capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr + good.stdout)
        self.assertEqual(good.stdout.strip(), 'POSTCOEF_REF_COMPATIBLE')
        extra = rel + '.extra'; (self.root / extra).write_text('unapproved\n')
        self.git('add', extra); self.git('commit', '-qm', 'unapproved evidence')
        bad = subprocess.run([*command, self.git('rev-parse', 'HEAD')], pass_fds=(fd,), capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(bad.stdout.strip(), 'POSTCOEF_REF_INCOMPATIBLE')

    def test_relative_repository_path_is_rejected(self):
        p = subprocess.run([sys.executable, '-B', '-I', str(self.tools / LAUNCHER.name),
                            'ancestry', '.', 'source', 'head'], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('absolute', p.stderr)

    def test_root_descriptor_survives_path_swap(self):
        root_fd = os.open(self.root, os.O_RDONLY)
        self.addCleanup(os.close, root_fd)
        original = self.root.parent / 'original-root'
        alternate = self.root.parent / 'alternate-root'; alternate.mkdir()
        source = (self.tools / LAUNCHER.name).read_bytes()
        bootstrap = ("__file__ = " + repr(str(self.tools / LAUNCHER.name)) +
                     "; __gate_root_fd__ = " + str(root_fd) +
                     "; exec(compile(bytes.fromhex(" + repr(source.hex()) + "), __file__, 'exec'))")
        self.root.rename(original); self.root.symlink_to(alternate, target_is_directory=True)
        p = subprocess.run([sys.executable, '-B', '-I', '-c', bootstrap, 'ancestry',
                            str(original), 'source', 'head'], pass_fds=(root_fd,),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'rejected')

    def test_node_passes_root_handle_through_path_swap(self):
        original = self.root.parent / 'original-node-root'
        alternate = self.root.parent / 'alternate-node-root'; alternate.mkdir()
        code = r"""
const fs = require('fs'), cp = require('child_process');
const [root, original, alternate, python] = process.argv.slice(1);
const fd = fs.openSync(root, 'r');
const source = fs.readFileSync(root + '/tools/core070_postcoef_gate_launcher.py');
const bootstrap = '__file__ = ' + JSON.stringify(root + '/tools/core070_postcoef_gate_launcher.py') +
  '; __gate_root_fd__ = 3; exec(compile(bytes.fromhex(' + JSON.stringify(source.toString('hex')) + "), __file__, 'exec'))";
fs.renameSync(root, original); fs.symlinkSync(alternate, root, 'dir');
try {
  const out = cp.execFileSync(python, ['-B', '-I', '-c', bootstrap, 'ancestry', original, 'source', 'head'],
                             {stdio:['ignore','pipe','pipe',fd], encoding:'utf8'});
  process.stdout.write(out);
} finally { fs.closeSync(fd); }
"""
        p = subprocess.run(['node', '-e', code, str(self.root), str(original), str(alternate), sys.executable],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'rejected')

    def test_assembler_caller_uses_original_root_after_swap(self):
        import hashlib
        from unittest.mock import patch
        import true_parity_assemble as assemble
        entry = self.tools / 'core070_postcoef_ref_compatible.py'
        entry.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nimport gate_helper\nimport json\nprint(json.dumps(gate_helper.VALUE))\n")
        self.git('add', 'tools/core070_postcoef_ref_compatible.py'); self.git('commit', '-qm', 'json fixture')
        trusted = self.git('rev-parse', 'HEAD')
        launcher = launcher_with_pin(trusted)
        path = self.tools / LAUNCHER.name; path.write_text(launcher)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        original = self.root.parent / 'original-assembler-root'
        alternate = self.root.parent / 'alternate-assembler-root'; alternate.mkdir()
        real_run = subprocess.run
        def swap_then_run(*args, **kwargs):
            self.root.rename(original); self.root.symlink_to(alternate, target_is_directory=True)
            return real_run(*args, **kwargs)
        with patch.object(assemble, 'POSTCOEF_LAUNCHER_SHA256', digest), patch.object(assemble.subprocess, 'run', side_effect=swap_then_run):
            result = assemble.run_postcoef_gate(path, 'ancestry', [str(self.root), 'source', 'head'], dict(os.environ))
        self.assertEqual(result, 'rejected')

    def test_shared_source_does_not_leak_object_alternates(self):
        shared = self.root.parent / 'shared-gate'
        subprocess.run(['git', 'clone', '--quiet', '--shared', str(self.root), str(shared)], check=True)
        self.assertTrue((shared / '.git/objects/info/alternates').is_file())
        self.root = shared; self.tools = shared / 'tools'
        for entry in ('live', 'identity', 'ancestry'):
            result = self.launch(entry)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'rejected')

    def test_unreachable_gate_commit_fails_closed(self):
        tree = self.git('rev-parse', self.head + '^{tree}')
        commit = subprocess.run(['git', '-C', str(self.root), 'commit-tree', tree], input='unreferenced gate\n',
                                check=True, capture_output=True, text=True).stdout.strip()
        launcher = launcher_with_pin(commit)
        (self.tools / LAUNCHER.name).write_text(launcher)
        result = self.launch('ancestry')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('POSTCOEF_GATE_REJECTED', result.stderr)
        self.assertIn('checkout', result.stderr)

    def test_modified_entrypoint_is_rejected(self):
        (self.tools/'core070_postcoef_live_ref.py').write_text("print('forged')\n")
        p = self.launch('live')
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn('forged', p.stdout)

if __name__ == '__main__': unittest.main()
