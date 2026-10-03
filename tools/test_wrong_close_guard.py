#!/usr/bin/env python3
"""Tests for tools/wrong_close_guard.py (no network)."""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import wrong_close_guard as g  # noqa: E402


def bad(text, intended=""):
    return sorted((r, n) for r, n, *_ in g.check([("t", text)], g.parse_intended(intended)))


class Incidents(unittest.TestCase):
    def test_142_negated_keyword_in_pr_body(self):
        self.assertEqual(bad("Does not fix #142."), [(None, 142)])

    def test_897_negated_keyword_cross_repo_body(self):
        self.assertEqual(bad("Does not fix itchyshin/gllvmTMB#897."),
                         [("itchyshin/gllvmtmb", 897)])
        self.assertEqual(bad('Does not fix #897'), [(None, 897)])

    def test_149_conventional_commit_subject(self):
        subj = "fix(n<p): Gaussian n_sites >= p guard is an ArgumentError; Lognormal inherits it (refs #149)"
        self.assertEqual(bad(subj), [(None, 149)])

    def test_142_commit_subjects(self):
        self.assertEqual(bad("fix(derived-ci): natural-bounds clamp for profile_ci_derived (#142)"),
                         [(None, 142)])

    def test_intended_set_silences(self):
        self.assertEqual(bad("Fixes #149", "149"), [])
        self.assertEqual(bad("fix(n<p): thing (refs #149)", "#149"), [])


class Syntax(unittest.TestCase):
    def test_all_keywords_case_and_colon(self):
        for kw in ["close", "closes", "closed", "fix", "fixes", "fixed", "resolve",
                   "resolves", "resolved", "CLOSES", "Fixes", "Resolved"]:
            self.assertEqual(bad(f"{kw} #5"), [(None, 5)], kw)
            self.assertEqual(bad(f"{kw}: #5"), [(None, 5)], kw)

    def test_owner_repo_and_url_forms(self):
        self.assertEqual(bad("closes owner/repo#7"), [("owner/repo", 7)])
        self.assertEqual(bad("Resolves https://github.com/Owner/Repo/issues/9"),
                         [("owner/repo", 9)])

    def test_refs_without_keyword_is_clean(self):
        self.assertEqual(bad("refs #701\nSee also #702 and owner/repo#3"), [])
        self.assertEqual(bad("test(n<p): pin the guard (#149)"), [])

    def test_keyword_inside_word_is_ignored(self):
        self.assertEqual(bad("prefix #1, fixture #2, unresolved #3, disclosed #4"), [])

    def test_keyword_on_other_line_is_ignored(self):
        self.assertEqual(bad("This fixes the guard.\nrefs #149"), [])

    def test_reference_before_keyword_is_ignored(self):
        self.assertEqual(bad("#149 is open; we fix it later"), [])

    def test_list_flags_every_ref_on_the_line(self):
        self.assertEqual(bad("closes #1, #2 and #3", "1"), [(None, 2), (None, 3)])

    def test_code_fence_and_html_comment_skipped(self):
        txt = "```\nfixes #1\n```\n<!-- closes #2 -->\nrefs #3"
        self.assertEqual(bad(txt), [])

    def test_html_entity_and_path_not_a_ref(self):
        self.assertEqual(bad("fixes a&#35; and src/a#b"), [])


class Cli(unittest.TestCase):
    def run_cli(self, body, intended=""):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(body)
        try:
            r = subprocess.run([sys.executable, g.__file__, "--intended", intended,
                                "--body-file", f.name], capture_output=True, text=True)
        finally:
            os.unlink(f.name)
        return r.returncode, r.stdout

    def test_exit_codes(self):
        self.assertEqual(self.run_cli("refs #149")[0], 0)
        rc, out = self.run_cli("Does not fix #142")
        self.assertEqual(rc, 1)
        self.assertIn("#142", out)
        self.assertEqual(self.run_cli("Fixes #5", "5")[0], 0)

    def test_intended_from_body(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("Fixes #5\n<!-- intended-closes: 5 -->\n")
        try:
            run = lambda *x: subprocess.run([sys.executable, g.__file__, "--body-file", f.name, *x],
                                            capture_output=True, text=True).returncode
            self.assertEqual(run("--intended-from-body"), 0)
            self.assertEqual(run(), 1)
        finally:
            os.unlink(f.name)

    def test_git_range(self):
        with tempfile.TemporaryDirectory() as d:
            def git(*a):
                subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                               check=True, capture_output=True)
            git("init", "-q")
            git("commit", "-q", "--allow-empty", "-m", "base")
            git("commit", "-q", "--allow-empty", "-m", "fix(x): y (refs #149)")
            r = subprocess.run([sys.executable, g.__file__, "--git-range", "HEAD~1..HEAD"],
                               capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 1)
            self.assertIn("#149", r.stdout)


if __name__ == "__main__":
    unittest.main()
