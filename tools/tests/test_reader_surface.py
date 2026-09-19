"""Tests for the public-documentation reader-surface scanner."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).parents[1] / "check_reader_surface.py"
SPEC = importlib.util.spec_from_file_location("check_reader_surface", CHECKER)
assert SPEC is not None and SPEC.loader is not None
reader_surface = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader_surface)


class ReaderSurfaceTests(unittest.TestCase):
    def write_doc(self, root: Path, name: str, text: str) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_clean_public_text_has_no_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "guide.md", "# Fit a model\n\nInspect the fitted model.\n")

            self.assertEqual(reader_surface.scan(root), [])

    def test_reports_rule_and_line_for_process_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "guide.md", "# Guide\n\nSee issue #42 before fitting.\n")

            findings = reader_surface.scan(root)

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].path, Path("guide.md"))
            self.assertEqual(findings[0].line, 3)
            self.assertEqual(findings[0].rule, "issue-reference")

    def test_excludes_explicit_developer_only_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "getting-started.md", "A reader-facing guide.\n")
            self.write_doc(root, "developer-notes/process.md", "PR #42 is tracked here.\n")

            self.assertEqual(reader_surface.scan(root), [])

    def test_reports_internal_validation_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "guide.md", "The capability ledger is green.\n")

            findings = reader_surface.scan(root)

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].rule, "internal-validation-language")

    def test_rejects_missing_or_file_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = self.write_doc(root, "guide.md", "Fit a model.\n")
            for invalid in (root / "missing", doc):
                with self.subTest(root=invalid):
                    with self.assertRaises(ValueError):
                        reader_surface.scan(invalid)

    def test_cli_rejects_invalid_root_without_success_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = self.write_doc(root, "guide.md", "Fit a model.\n")
            for invalid in (root / "missing", doc):
                with self.subTest(root=invalid):
                    result = subprocess.run(
                        [sys.executable, "-B", str(CHECKER), str(invalid)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("READER_SURFACE_PASS", result.stdout)
                    self.assertIn("directory", result.stderr)

    def test_allows_ordinary_technical_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "guide.md", (
                "A circular arc shows the confidence region.\n"
                "Use Fisher information and an optimization arc.\n"
                "The model includes agent interactions and personal effects.\n"
            ))
            self.assertEqual(reader_surface.scan(root), [])

    def test_rejects_process_phrases(self) -> None:
        examples = (
            "Check the scoreboard before fitting.",
            "The fixture evidence confirms this route.",
            "The fixture\nevidence confirms this route.",
            "The agent approved this example.",
            "The agents have approved this example.",
            "An agent-approved example.",
            "The persona audited this example.",
            "The personas\naudited this example.",
            "A worktree holds this lane and dev-log.",
            "See PR #428 or issue #12.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for example in examples:
                with self.subTest(text=example):
                    self.write_doc(root, "guide.md", example)
                    self.assertTrue(reader_surface.scan(root))

    def test_wrapped_capability_ledger_reports_first_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_doc(root, "guide.md", "# Guide\nThe capability\nledger is green.\n")
            findings = reader_surface.scan(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].line, 2)
            self.assertEqual(findings[0].rule, "internal-validation-language")


if __name__ == "__main__":
    unittest.main()
