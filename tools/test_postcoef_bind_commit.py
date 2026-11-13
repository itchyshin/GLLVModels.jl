#!/usr/bin/env python3
"""The POST-COEF evidence rule admits one narrow bind commit and nothing wider.

A bind commit sits directly on the capture source S and moves only the five gate
pins (launcher TRUSTED_GATE_COMMIT, launcher-test EXPECTED_GATE, assemble
POSTCOEF_GATE_COMMIT and POSTCOEF_LAUNCHER_SHA256, mjs launcher sha256) to S and
to sha256 of the rebound launcher. Each negative control is one attack the rule
must still reject. Pure Python and git; no Julia, R, capture or network.
"""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import core070_postfit_p1_receipts as receipts  # noqa: E402

LAUNCHER = 'tools/core070_postcoef_gate_launcher.py'
LAUNCHER_TEST = 'tools/test_core070_postcoef_gate_launcher.py'
ASSEMBLE = 'tools/true_parity_assemble.py'
MJS = 'tools/true_parity_check.mjs'
BIND_FILES = (LAUNCHER, LAUNCHER_TEST, ASSEMBLE, MJS)
OTHER_TOOL = 'tools/core070_postcoef_fit_health.py'
EVIDENCE = f'{receipts.REC_REL}/postcoef-empty-p1/results.tsv'
# Written independently of the rule under test: (file, literal prefix before the value).
COMMIT_PINS = ((LAUNCHER, "TRUSTED_GATE_COMMIT = '"), (LAUNCHER_TEST, "EXPECTED_GATE = '"),
               (ASSEMBLE, "POSTCOEF_GATE_COMMIT = '"))
SHA_PINS = ((ASSEMBLE, "POSTCOEF_LAUNCHER_SHA256 = '"), (MJS, "digest('hex') !==\n          '"))


def set_value(text, prefix, value):
    assert text.count(prefix) == 1, prefix
    start = text.index(prefix) + len(prefix)
    end = text.index("'", start)
    return text[:start] + value + text[end:]


class BindCommitRule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='postcoef-bind-')
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        for rel in (*BIND_FILES, OTHER_TOOL):
            self.write(rel, (T.parent / rel).read_text())
        self.git('add', '--', *BIND_FILES, OTHER_TOOL)
        self.commit('capture source S')
        self.source = self.head()
        self.texts = {rel: (self.root / rel).read_text() for rel in BIND_FILES}

    def tearDown(self):
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
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-qm', message)

    def bound(self, commit=None, sha=None, launcher_edit=None):
        """Texts with all pins moved; sha defaults to sha256 of the rebound launcher."""
        commit = commit or self.source
        new = dict(self.texts)
        for rel, prefix in COMMIT_PINS:
            new[rel] = set_value(new[rel], prefix, commit)
        if launcher_edit:
            new[LAUNCHER] = launcher_edit(new[LAUNCHER])
        digest = sha or hashlib.sha256(new[LAUNCHER].encode()).hexdigest()
        for rel, prefix in SHA_PINS:
            new[rel] = set_value(new[rel], prefix, digest)
        return new

    def commit_files(self, files, message='bind'):
        for rel, text in files.items():
            self.write(rel, text)
        self.git('add', '--', *files)
        self.commit(message)
        return self.head()

    def evidence(self):
        return self.commit_files({EVIDENCE: 'case\tvalue\n'}, 'evidence')

    def accepts(self, source=None, head=None):
        return receipts.postcoef_ref_compatible(self.root, source or self.source, head or self.head())

    # Positive controls: fail before the change, pass after.
    def test_bind_commit_on_source_is_compatible(self):
        bind = self.commit_files(self.bound())
        self.assertTrue(self.accepts())
        self.evidence()
        self.assertTrue(self.accepts())
        self.assertTrue(self.accepts(bind))

    def test_gate_entry_cli_accepts_bind_then_evidence(self):
        self.commit_files(self.bound())
        self.evidence()
        proc = subprocess.run([sys.executable, '-B', '-I', str(T / 'core070_postcoef_ref_compatible.py'),
                               str(self.root), self.source, self.head()], capture_output=True, text=True)
        self.assertEqual((proc.returncode, proc.stdout.strip()), (0, 'POSTCOEF_REF_COMPATIBLE'))

    # Unchanged behaviour.
    def test_evidence_only_history_still_compatible(self):
        self.evidence()
        self.assertTrue(self.accepts())

    # Attack 1: smuggled code in tools/.
    def test_rejects_bind_that_edits_another_tools_file(self):
        files = self.bound()
        files[OTHER_TOOL] = (self.root / OTHER_TOOL).read_text() + '\nTRUST_EVERYTHING = True\n'
        self.commit_files(files)
        self.assertFalse(self.accepts())

    def test_rejects_bind_that_adds_a_tools_file(self):
        files = self.bound()
        files['tools/core070_json_semantics.py'] = 'def json_value_equal(a, b): return True\n'
        self.commit_files(files)
        self.assertFalse(self.accepts())

    def test_rejects_code_change_inside_a_bind_file(self):
        cases = {
            'launcher': (LAUNCHER, lambda t: t.replace("raise ValueError('gate entry differs from committed source')",
                                                     'pass', 1)),
            'assemble': (ASSEMBLE, lambda t: t + '\nPOSTCOEF_OFFLINE_TIMEOUT_SECONDS = 10**9\n'),
            'mjs': (MJS, lambda t: t.replace("throw new Error('untrusted launcher');", ';', 1)),
            'launcher-test': (LAUNCHER_TEST, lambda t: t + '\n# harmless?\n'),
        }
        for label, (rel, edit) in cases.items():
            with self.subTest(file=label):
                self.git('reset', '-q', '--hard', self.source)
                if rel == LAUNCHER:
                    files = self.bound(launcher_edit=edit)  # sha recomputed: only the code edit differs
                else:
                    files = self.bound()
                    files[rel] = edit(files[rel])
                self.assertNotEqual(files[rel], self.bound()[rel])
                self.commit_files(files)
                self.assertFalse(self.accepts())

    def test_rejects_whitespace_or_duplicate_pin_lines(self):
        for label, edit in {
            'mjs whitespace': lambda f: {**f, MJS: f[MJS].replace("!==\n          '", "!== '", 1)},
            'duplicate launcher pin': lambda f: {**f, LAUNCHER: f[LAUNCHER] + "TRUSTED_GATE_COMMIT = '" + self.source + "'\n"},
        }.items():
            with self.subTest(case=label):
                self.git('reset', '-q', '--hard', self.source)
                self.commit_files(edit(self.bound()))
                self.assertFalse(self.accepts())

    def test_rejects_launcher_mode_change(self):
        self.commit_files(self.bound(), 'bind')
        self.git('update-index', '--chmod=+x', LAUNCHER)
        self.git('-c', 'user.name=f', '-c', 'user.email=f@example.invalid', 'commit', '-q', '--amend', '--no-edit')
        self.assertIn('100755', self.git('ls-tree', 'HEAD', '--', LAUNCHER))
        self.assertFalse(self.accepts())

    # Attack 2: pins naming a commit other than the capture source.
    def test_rejects_pin_to_wrong_commit(self):
        later = self.commit_files({EVIDENCE: 'x\n'}, 'later commit')
        self.git('reset', '-q', '--hard', self.source)
        sibling = self.commit_files({OTHER_TOOL: '# sibling gate\n'}, 'sibling of S')
        for label, commit in {'later commit': later, 'sibling commit': sibling, 'zero id': '0' * 40}.items():
            with self.subTest(pin=label):
                self.git('reset', '-q', '--hard', self.source)
                self.commit_files(self.bound(commit=commit))
                self.assertFalse(self.accepts())

    def test_rejects_inconsistent_commit_pins(self):
        for rel, prefix in COMMIT_PINS:
            with self.subTest(pin=rel):
                self.git('reset', '-q', '--hard', self.source)
                files = self.bound()
                files[rel] = set_value(files[rel], prefix, '1' * 40)
                if rel == LAUNCHER:  # keep the hash pins honest so only the commit pin is wrong
                    digest = hashlib.sha256(files[LAUNCHER].encode()).hexdigest()
                    for srel, sprefix in SHA_PINS:
                        files[srel] = set_value(files[srel], sprefix, digest)
                self.commit_files(files)
                self.assertFalse(self.accepts())

    # Attack 3: launcher hash swap.
    def test_rejects_launcher_hash_swap(self):
        old_sha = hashlib.sha256(self.texts[LAUNCHER].encode()).hexdigest()
        evil = self.bound(launcher_edit=lambda t: t + '\n# evil\n')[LAUNCHER]
        for label, files in {
            'stale sha of S launcher': self.bound(sha=old_sha),
            'sha of a different launcher': self.bound(sha=hashlib.sha256(evil.encode()).hexdigest()),
            'assemble and mjs disagree': {**self.bound(),
                                          MJS: set_value(self.bound()[MJS], SHA_PINS[1][1], 'f' * 64)},
        }.items():
            with self.subTest(case=label):
                self.git('reset', '-q', '--hard', self.source)
                self.commit_files(files)
                self.assertFalse(self.accepts())

    # Attack 4: tools-tree drift around the bind.
    def test_rejects_tools_drift_after_or_before_bind(self):
        self.commit_files(self.bound())
        self.commit_files({OTHER_TOOL: '# drift\n'}, 'drift after bind')
        self.assertFalse(self.accepts())
        self.git('reset', '-q', '--hard', self.source)
        self.evidence()
        self.commit_files(self.bound())  # bind not directly on S
        self.assertFalse(self.accepts())

    def test_rejects_second_bind_and_partial_bind(self):
        self.commit_files(self.bound())
        bind = self.head()
        files = {rel: (self.root / rel).read_text() for rel in BIND_FILES}
        files[LAUNCHER] = set_value(files[LAUNCHER], "TRUSTED_GATE_COMMIT = '", bind)
        digest = hashlib.sha256(files[LAUNCHER].encode()).hexdigest()
        files[LAUNCHER_TEST] = set_value(files[LAUNCHER_TEST], "EXPECTED_GATE = '", bind)
        files[ASSEMBLE] = set_value(files[ASSEMBLE], "POSTCOEF_GATE_COMMIT = '", bind)
        for rel, prefix in SHA_PINS:
            files[rel] = set_value(files[rel], prefix, digest)
        self.commit_files(files, 'second bind')
        self.assertFalse(self.accepts())
        self.git('reset', '-q', '--hard', self.source)
        partial = self.bound()
        del partial[LAUNCHER_TEST]
        self.commit_files(partial, 'partial bind')
        self.assertFalse(self.accepts())

    def test_rejects_bind_mixed_with_evidence(self):
        files = self.bound()
        files[EVIDENCE] = 'case\tvalue\n'
        self.commit_files(files)
        self.assertFalse(self.accepts())

    # Attack 5: S already holds the mask token, so masked equality could hide moved text.
    def rebase_source(self, rel, edit, message):
        """New capture source S2 = S plus one edit; bind tests then run against S2."""
        self.write(rel, edit((self.root / rel).read_text()))
        self.git('add', '--', rel)
        self.commit(message)
        self.source = self.head()
        self.texts = {r: (self.root / r).read_text() for r in BIND_FILES}

    def test_rejects_token_decoy_line_in_launcher(self):
        line = "TRUSTED_GATE_COMMIT = '"
        self.rebase_source(LAUNCHER, lambda t: t.replace("\nTRUSTED_GATE_COMMIT = '", "\nTRUSTED_GATE_COMMIT = '<launcher_commit>'\nTRUSTED_GATE_COMMIT = '", 1), 'S with decoy')
        files = dict(self.texts)
        old = files[LAUNCHER]
        real = old[old.index(line, old.index("<launcher_commit>")) + len(line):][:40]
        # S: decoy line, then the real pin. Bind: the real line takes the bound commit's place
        # and the decoy keeps the token, so the two masked texts are equal.
        moved = old.replace(line + "<launcher_commit>'\n" + line + real + "'",
                            line + self.source + "'\n" + line + "<launcher_commit>'", 1)
        self.assertNotEqual(moved, old)
        files[LAUNCHER] = moved
        digest = hashlib.sha256(moved.encode()).hexdigest()
        for rel, prefix in COMMIT_PINS[1:]:
            files[rel] = set_value(files[rel], prefix, self.source)
        for rel, prefix in SHA_PINS:
            files[rel] = set_value(files[rel], prefix, digest)
        self.commit_files(files)
        self.assertFalse(self.accepts())

    def test_rejects_token_decoy_comment_in_mjs(self):
        check = "      if (createHash('sha256')"
        self.rebase_source(MJS, lambda t: t.replace(
            check, "      // update(launcherBytes).digest('hex') !== '<mjs_sha>'\n" + check, 1), 'S with decoy')
        files = self.bound()
        # Real hash moves into the comment; the real check now compares against the token.
        old = self.texts[MJS]
        real = SHA_PINS[1][1]
        start = old.index(real) + len(real)
        decoy = "// update(launcherBytes).digest('hex') !== '"
        digest = hashlib.sha256(files[LAUNCHER].encode()).hexdigest()
        moved = old[:start] + '<mjs_sha>' + old[start + 64:]
        moved = moved.replace(decoy + "<mjs_sha>'", decoy + digest + "'", 1)
        self.assertNotEqual(moved, old)
        files[MJS] = moved
        self.commit_files(files)
        self.assertFalse(self.accepts())


if __name__ == '__main__':
    unittest.main()
