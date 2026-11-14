#!/usr/bin/env python3
"""The POST-COEF content rule: bound inputs must equal the capture source's; history shape does not matter.

Each scenario is a synthetic git repository (no Julia, R, capture or network). S is the capture source, B the
bind commit on S (five pins moved to S), E the evidence commit. The rule under test is
postcoef_content_bound_problem(repo, S, head): None means accepted, a string is the refusal reason.
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
from test_postcoef_bind_commit import (  # noqa: E402
    ASSEMBLE, BIND_FILES, COMMIT_PINS, LAUNCHER, LAUNCHER_TEST, MJS, SHA_PINS, set_value)

JSON_TOOL = 'tools/core070_json_semantics.py'
OTHER_TOOL = 'tools/core070_postcoef_fit_health.py'
PREFIX = f'{receipts.REC_REL}/postcoef-empty-p1'
EVIDENCE = (f'{PREFIX}/receipt.json', f'{PREFIX}/verify.txt', f'{PREFIX}/runner-snapshots/julia-launcher',
            f'{receipts.REC_REL}/cases/{receipts.COEF_THREE_FIT_CID}.json')
BOARD = 'docs/dev-log/core070/true-parity-latest/scoreboard.md'


class ContentRule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='postcoef-content-')
        self.root = Path(self.tmp.name)
        self.git('init', '-q', '-b', 'main')
        self.write('README.md', 'base\n')
        self.git('add', '--', 'README.md'); self.commit('base P')
        self.base = self.head()
        for rel in (*BIND_FILES, JSON_TOOL, OTHER_TOOL):
            self.write(rel, (T.parent / rel).read_text())
        self.write('src/a.jl', 'x = 1\n')
        self.write(BOARD, 'board one\n')
        self.git('add', '--', *BIND_FILES, JSON_TOOL, OTHER_TOOL, 'src/a.jl', BOARD)
        self.commit('capture source S')
        self.source = self.head()
        self.texts = {rel: (self.root / rel).read_text() for rel in BIND_FILES}
        self.commit_files(self.bound(), 'bind B')
        self.commit_files({rel: f'{rel}\n' for rel in EVIDENCE}, 'evidence E')
        self.evidence_head = self.head()

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
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message)

    def commit_files(self, files, message):
        for rel, text in files.items():
            self.write(rel, text)
        self.git('add', '--', *files)
        self.commit(message)
        return self.head()

    def bound(self, launcher_edit=None):
        new = dict(self.texts)
        for rel, prefix in COMMIT_PINS:
            new[rel] = set_value(new[rel], prefix, self.source)
        if launcher_edit:
            new[LAUNCHER] = launcher_edit(new[LAUNCHER])
        digest = hashlib.sha256(new[LAUNCHER].encode()).hexdigest()
        for rel, prefix in SHA_PINS:
            new[rel] = set_value(new[rel], prefix, digest)
        return new

    def edited(self, rel, suffix):
        return (self.root / rel).read_text() + suffix

    def problem(self, head=None):
        return receipts.postcoef_content_bound_problem(self.root, self.source, head or self.head())

    def reset_to_evidence(self):
        self.git('checkout', '-q', '-B', 'main', self.evidence_head)

    # Positive controls: later history of every shape passes while the bound set is unchanged.
    def test_bind_and_evidence_heads_pass(self):
        self.assertIsNone(self.problem())
        bind = self.git('rev-parse', 'HEAD~1')
        self.assertIsNone(self.problem(bind))

    def test_later_docs_commit_passes(self):
        self.commit_files({'docs/notes.md': 'later\n'}, 'docs')
        self.assertIsNone(self.problem())

    def test_later_unrelated_tools_and_engine_commits_pass(self):
        self.commit_files({OTHER_TOOL: self.edited(OTHER_TOOL, '\n# other lane\n')}, 'other tool')
        self.commit_files({'src/a.jl': 'x = 2\n'}, 'engine')
        self.commit_files({BOARD: 'board two, another row closed\n'}, 'board regenerated')
        self.assertIsNone(self.problem())

    def test_merge_commit_passes(self):
        self.git('checkout', '-q', '-b', 'side', self.evidence_head)
        self.commit_files({'docs/side.md': 'side\n'}, 'side docs')
        self.git('checkout', '-q', 'main')
        self.commit_files({'docs/main.md': 'main\n'}, 'main docs')
        self.git('-c', 'user.name=f', '-c', 'user.email=f@example.invalid', 'merge', '--no-ff', '-q', '-m', 'merge', 'side')
        self.assertEqual(len(self.git('rev-list', '--parents', '-n', '1', 'HEAD').split()), 3)
        self.assertIsNone(self.problem())

    def test_squash_shaped_history_passes_when_source_object_is_kept(self):
        pr = self.evidence_head
        self.git('checkout', '-q', '-B', 'main', self.base)
        self.git('checkout', pr, '--', '.')
        self.commit('squash of the PR')
        squash = self.head()
        self.assertNotEqual(self.git('merge-base', self.source, squash), self.source)  # S is not an ancestor
        self.assertIsNone(self.problem(squash))
        self.assertFalse(receipts.postcoef_ref_compatible(self.root, self.source, squash))  # the old rule fails here

    def test_rebased_shape_passes(self):
        self.git('checkout', '-q', '-B', 'rebased', self.base)
        self.commit_files({'docs/new-base.md': 'main moved\n'}, 'main moved')
        self.git('-c', 'user.name=f', '-c', 'user.email=f@example.invalid', 'cherry-pick', f'{self.base}..{self.evidence_head}')
        self.assertNotEqual(self.head(), self.evidence_head)
        self.assertIsNone(self.problem())

    # Negative controls: each attack the rule must still stop.
    def test_bound_tool_change_after_bind_is_rejected(self):
        for rel in (ASSEMBLE, MJS, LAUNCHER_TEST, JSON_TOOL):
            with self.subTest(file=rel):
                self.reset_to_evidence()
                self.commit_files({rel: self.edited(rel, '\n// judge swapped\n')}, 'swap judge')
                self.assertIsNotNone(self.problem())

    def test_launcher_swap_with_matching_hash_pins_is_rejected(self):
        self.reset_to_evidence()
        files = self.bound(launcher_edit=lambda t: t + '\n# evil\n')
        self.commit_files({rel: files[rel] for rel in (LAUNCHER, ASSEMBLE, MJS)}, 'swap launcher and rehash')
        self.assertIsNotNone(self.problem())

    def test_wrong_or_missing_pins_are_rejected(self):
        files = self.bound()
        files[ASSEMBLE] = set_value(files[ASSEMBLE], COMMIT_PINS[2][1], '1' * 40)
        self.reset_to_evidence()
        self.commit_files({ASSEMBLE: files[ASSEMBLE]}, 'pin to another commit')
        self.assertIsNotNone(self.problem())
        self.assertIsNotNone(self.problem(self.source))  # S itself carries the old pins

    def test_bound_change_inside_a_merge_or_squash_is_rejected(self):
        self.git('checkout', '-q', '-b', 'side', self.evidence_head)
        self.commit_files({ASSEMBLE: self.edited(ASSEMBLE, '\n# skip live gate\n')}, 'side edit')
        self.git('checkout', '-q', 'main')
        self.commit_files({'docs/main.md': 'main\n'}, 'main docs')
        self.git('-c', 'user.name=f', '-c', 'user.email=f@example.invalid', 'merge', '--no-ff', '-q', '-m', 'merge', 'side')
        self.assertIsNotNone(self.problem())
        pr = self.git('rev-parse', 'side')
        self.git('checkout', '-q', '-B', 'squashed', self.base)
        self.git('checkout', pr, '--', '.')
        self.commit('squash with a swapped judge')
        self.assertIsNotNone(self.problem())

    def test_extra_or_linked_capture_file_is_rejected(self):
        self.commit_files({f'{PREFIX}/extra-note.txt': 'smuggled\n'}, 'extra file')
        self.assertIn('allow-list', self.problem())
        self.reset_to_evidence()
        (self.root / f'{PREFIX}/linked').symlink_to('receipt.json')
        self.git('add', '--', f'{PREFIX}/linked'); self.commit('linked entry')
        self.assertIn('allow-list', self.problem())

    def test_unknown_or_malformed_commits_are_rejected(self):
        self.assertIn('not in the repository', receipts.postcoef_content_bound_problem(self.root, '0' * 40, self.head()))
        self.assertIn('not in the repository', receipts.postcoef_content_bound_problem(self.root, self.source, '1' * 40))
        self.assertIn('full commit id', receipts.postcoef_content_bound_problem(self.root, 'HEAD', self.head()))

    def test_bound_list_names_the_bind_files_and_the_assembler_import(self):
        self.assertEqual(set(receipts.POSTCOEF_BOUND_FILES), {*receipts.POSTCOEF_BIND_FILES, JSON_TOOL})
        imports = (T / 'true_parity_assemble.py').read_text()
        self.assertIn('from core070_json_semantics import', imports)
        self.assertNotIn('from core070_postfit', imports)


if __name__ == '__main__':
    unittest.main()
